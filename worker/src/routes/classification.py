import time
import numpy as np
import numpy.typing as npt
from fastapi import APIRouter, Depends, HTTPException, Body

from config import Settings
from dependencies import get_logger, get_storage_client, get_ckks, get_settings
from mictlanx import AsyncClient
from mictlanx.utils.segmentation import Chunks
from models import ExperimentLogEntry
from models.requests.classification import (
    KnnPredictWorkerRequest,
    SknnPredictWorkerRequest,
    PqcSknnPredictWorkerRequest,
)
from models.responses.classification import (
    KnnPredictResponse,
    SknnPredictStep1Response,
    SknnPredictStep2Response,
)
from models.responses.clustering import HealthCheckResponse
from option import Some
from rory.core.classification.knn import KNearestNeighbors as KNN
from rory.core.classification.secure.conventional.sknn import SecureKNearestNeighbors as SKNN
from rory.core.classification.secure.pqc.sknn import SecureKNearestNeighbors as SKNNPQC
from rory.core.utils.constants import Constants
from rorycommon import Common as RoryCommon

router = APIRouter(prefix="/classification", tags=["Classification"])


@router.get("/test")
@router.post("/test", response_model=HealthCheckResponse)
async def test():
    return {"component_type": "worker"}


async def sknn_pedict_1(
    body: SknnPredictWorkerRequest,
    logger,
    settings: Settings,
    storage_client: AsyncClient,
):
    local_start_time = time.time()
    worker_id = settings.node_id
    model_id = body.model_id
    encrypted_model_id = "encrypted{}".format(model_id)
    model_labels_id = "{}labels".format(model_id)
    records_test_id = body.records_test_id
    encrypted_records_id = "encrypted{}".format(records_test_id)
    algorithm = Constants.ClassificationAlgorithms.SKNN_PREDICT
    _encrypted_model_shape = body.encrypted_model_shape
    _encrypted_model_dtype = body.encrypted_model_dtype
    _encrypted_records_shape = body.encrypted_records_shape
    _encrypted_records_dtype = body.encrypted_records_dtype
    distance = settings.distance
    experiment_id = body.experiment_id or ""
    mictlanx_timeout = settings.mictlanx_timeout
    mictlanx_delay = settings.mictlanx_delay
    mictlanx_backoff_factor = settings.mictlanx_backoff_factor
    mictlanx_max_retries = settings.mictlanx_max_retries

    if _encrypted_model_dtype is None:
        raise HTTPException(status_code=500, detail="Encrypted-Model-Dtype")
    if _encrypted_model_shape is None:
        raise HTTPException(status_code=500, detail="Encrypted-Model-Shape header is required")

    if _encrypted_records_dtype is None:
        raise HTTPException(status_code=500, detail="Encrypted-Records-Dtype")
    if _encrypted_records_shape is None:
        raise HTTPException(status_code=500, detail="Encrypted-Records-Shape header is required")

    encrypted_model_shape = eval(_encrypted_model_shape)
    encrypted_records_shape = eval(_encrypted_records_shape)
    num_chunks_str = body.num_chunks
    if num_chunks_str is None:
        raise HTTPException(status_code=503, detail="Num-Chunks header is required")
    num_chunks = int(num_chunks_str)

    try:
        get_merge_encrypted_model_start_time = time.time()
        encrypted_model = await RoryCommon.get_and_merge(
            client=storage_client,
            bucket_id=settings.mictlanx_bucket_id,
            key=encrypted_model_id,
            delay=mictlanx_delay,
            max_retries=mictlanx_max_retries,
            timeout=mictlanx_timeout,
            backoff_factor=mictlanx_backoff_factor,
        )

        get_encrypted_ptm_entry = ExperimentLogEntry(
            event="GET",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=get_merge_encrypted_model_start_time,
            end_time=time.time(),
            id=encrypted_model_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
        )
        logger.info(get_encrypted_ptm_entry.model_dump())

        encrypted_records = await RoryCommon.get_and_merge(
            client=storage_client,
            key=encrypted_records_id,
            bucket_id=settings.mictlanx_bucket_id,
            max_retries=mictlanx_max_retries,
            delay=mictlanx_delay,
            backoff_factor=mictlanx_backoff_factor,
            timeout=mictlanx_timeout,
        )

        get_encrypted_ptm_entry = ExperimentLogEntry(
            event="GET",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=get_merge_encrypted_model_start_time,
            end_time=time.time(),
            id=encrypted_records_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
        )
        logger.info(get_encrypted_ptm_entry.model_dump())

        all_distances = SKNN.calculate_distances(
            dataset=encrypted_records,
            model=encrypted_model,
            distance=distance,
        )

        distances_id = "distances{}".format(records_test_id)
        distances_shape = all_distances.shape
        distances_dtype = all_distances.dtype

        calculate_distances_entry = ExperimentLogEntry(
            event="CALCULATE.DISTANCES",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=get_merge_encrypted_model_start_time,
            end_time=time.time(),
            id=distances_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
        )
        logger.info(calculate_distances_entry.model_dump())

        maybe_all_distances_chunks = Chunks.from_ndarray(
            ndarray=all_distances,
            group_id=distances_id,
            chunk_prefix=Some(distances_id),
            num_chunks=num_chunks,
        )
        if maybe_all_distances_chunks.is_none:
            raise "something went wrong creating the chunks"
        put_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
            client=storage_client,
            bucket_id=settings.mictlanx_bucket_id,
            key=distances_id,
            chunks=maybe_all_distances_chunks.unwrap(),
            max_tries=mictlanx_max_retries,
            timeout=mictlanx_timeout,
            tags={
                "full_shape": str(distances_shape),
                "full_dtype": "float32",
            },
        )

        end_time = time.time()
        service_time = end_time - local_start_time

        classification_completed_entry = ExperimentLogEntry(
            event="COMPLETED",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=local_start_time,
            end_time=time.time(),
            id=model_id,
            num_chunks=num_chunks,
            time=service_time,
        )
        logger.info(classification_completed_entry.model_dump())

        return {
            "distances_id": distances_id,
            "distances_shape": str(distances_shape),
            "distances_dtype": str(distances_dtype),
            "service_time": service_time,
        }
    except Exception as e:
        logger.error({"msg": str(e)})
        raise HTTPException(status_code=503, detail=str(e))


async def sknn_predict_2(
    body: SknnPredictWorkerRequest,
    logger,
    settings: Settings,
    storage_client: AsyncClient,
):
    local_start_time = time.time()
    worker_id = settings.node_id
    model_id = body.model_id
    model_labels_id = "{}labels".format(model_id)
    records_test_id = body.records_test_id
    _model_labels_shape = body.model_labels_shape
    if _model_labels_shape is None:
        raise HTTPException(status_code=500, detail="Model-Labels-Shape header is required")
    model_labels_shape = eval(_model_labels_shape)
    min_distances_index_id = "distancesindex{}".format(records_test_id)
    algorithm = Constants.ClassificationAlgorithms.SKNN_PREDICT
    experiment_id = body.experiment_id or ""
    mictlanx_timeout = settings.mictlanx_timeout
    mictlanx_delay = settings.mictlanx_delay
    mictlanx_backoff_factor = settings.mictlanx_backoff_factor
    mictlanx_max_retries = settings.mictlanx_max_retries

    try:
        model_labels_get_start_time = time.time()
        model_labels = await RoryCommon.get_and_merge(
            client=storage_client,
            key=model_labels_id,
            bucket_id=settings.mictlanx_bucket_id,
            backoff_factor=mictlanx_backoff_factor,
            delay=mictlanx_delay,
            max_retries=mictlanx_max_retries,
            timeout=mictlanx_timeout,
        )

        get_encrypted_ptm_entry = ExperimentLogEntry(
            event="GET",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=model_labels_get_start_time,
            end_time=time.time(),
            id=model_labels_id,
            worker_id=worker_id,
        )
        logger.info(get_encrypted_ptm_entry.model_dump())

        min_distances_start_time = time.time()
        min_distances_index = await RoryCommon.get_and_merge(
            client=storage_client,
            key=min_distances_index_id,
            bucket_id=settings.mictlanx_bucket_id,
            backoff_factor=mictlanx_backoff_factor,
            delay=mictlanx_delay,
            max_retries=mictlanx_max_retries,
            timeout=mictlanx_timeout,
        )

        get_encrypted_ptm_entry = ExperimentLogEntry(
            event="GET",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=min_distances_start_time,
            end_time=time.time(),
            id=min_distances_index_id,
            worker_id=worker_id,
        )
        logger.info(get_encrypted_ptm_entry.model_dump())

        label_vector = SKNN.get_label_vector(
            model_labels=model_labels.reshape((model_labels_shape[1],)),
            min_indexes=min_distances_index,
        )
        label_vector = label_vector.reshape((label_vector.shape[0],))
        end_time = time.time()
        service_time = end_time - local_start_time

        classification_completed_entry = ExperimentLogEntry(
            event="COMPLETED",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=local_start_time,
            end_time=time.time(),
            id=model_id,
            time=service_time,
        )
        logger.info(classification_completed_entry.model_dump())

        return {
            "label_vector": list(map(int, label_vector.flatten().tolist())),
            "service_time": service_time,
        }
    except Exception as e:
        logger.error({"msg": str(e)})
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/sknn/predict")
async def sknn_predict(
    body: SknnPredictWorkerRequest,
    logger=Depends(get_logger),
    settings: Settings = Depends(get_settings),
    storage_client: AsyncClient = Depends(get_storage_client),
):
    step_index = body.step_index
    if step_index == 1:
        return await sknn_pedict_1(body, logger, settings, storage_client)
    elif step_index == 2:
        return await sknn_predict_2(body, logger, settings, storage_client)
    else:
        raise HTTPException(status_code=400, detail="Invalid step_index")


@router.post("/knn/predict", response_model=KnnPredictResponse)
async def knn_predict(
    body: KnnPredictWorkerRequest,
    logger=Depends(get_logger),
    settings: Settings = Depends(get_settings),
    storage_client: AsyncClient = Depends(get_storage_client),
):
    local_start_time = time.time()
    worker_id = settings.node_id
    model_id = body.model_id
    model_labels_id = "{}labels".format(model_id)
    records_test_id = body.records_test_id
    algorithm = Constants.ClassificationAlgorithms.KNN_PREDICT
    distance = settings.distance
    experiment_id = body.experiment_id or ""
    mictlanx_timeout = settings.mictlanx_timeout
    mictlanx_delay = settings.mictlanx_delay
    mictlanx_backoff_factor = settings.mictlanx_backoff_factor
    mictlanx_max_retries = settings.mictlanx_max_retries

    _model_labels_shape = body.model_labels_shape
    if _model_labels_shape is None:
        raise HTTPException(status_code=500, detail="Model-Labels-Shape header is required")
    model_labels_shape = eval(_model_labels_shape)

    try:
        get_model_start_time = time.time()
        model = await RoryCommon.get_and_merge(
            client=storage_client,
            bucket_id=settings.mictlanx_bucket_id,
            key=model_id,
            max_retries=mictlanx_max_retries,
            delay=mictlanx_delay,
            backoff_factor=mictlanx_backoff_factor,
            timeout=mictlanx_timeout,
        )

        get_model_entry = ExperimentLogEntry(
            event="GET",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=get_model_start_time,
            end_time=time.time(),
            id=model_id,
            worker_id=worker_id,
        )
        logger.info(get_model_entry.model_dump())

        get_model_labels_start_time = time.time()
        model_labels = await RoryCommon.get_and_merge(
            client=storage_client,
            bucket_id=settings.mictlanx_bucket_id,
            key=model_labels_id,
            max_retries=mictlanx_max_retries,
            delay=mictlanx_delay,
            backoff_factor=mictlanx_backoff_factor,
            timeout=mictlanx_timeout,
        )

        get_model_labels_entry = ExperimentLogEntry(
            event="GET",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=get_model_labels_start_time,
            end_time=time.time(),
            id=model_labels_id,
            worker_id=worker_id,
        )
        logger.info(get_model_labels_entry.model_dump())

        get_records_start_time = time.time()
        records = await RoryCommon.get_and_merge(
            client=storage_client,
            key=records_test_id,
            bucket_id=settings.mictlanx_bucket_id,
            max_retries=mictlanx_max_retries,
            delay=mictlanx_delay,
            backoff_factor=mictlanx_backoff_factor,
            timeout=mictlanx_timeout,
        )

        get_records_entry = ExperimentLogEntry(
            event="GET",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=get_records_start_time,
            end_time=time.time(),
            id=records_test_id,
            worker_id=worker_id,
        )
        logger.info(get_records_entry.model_dump())

        knn_predict_start_time = time.time()
        label_vector: npt.NDArray = KNN.predict(
            dataset=records,
            model=model,
            model_labels=model_labels.reshape((model_labels_shape[1],)),
            distance=distance,
        )

        end_time = time.time()
        service_time = end_time - local_start_time

        classification_completed_entry = ExperimentLogEntry(
            event="COMPLETED",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=local_start_time,
            end_time=time.time(),
            id=model_id,
            time=service_time,
        )
        logger.info(classification_completed_entry.model_dump())

        return {
            "label_vector": list(map(int, label_vector.flatten().tolist())),
            "service_time": service_time,
        }
    except Exception as e:
        logger.error({"msg": str(e)})
        raise HTTPException(status_code=503, detail=str(e))


async def sknn_pqc_pedict_1(
    body: PqcSknnPredictWorkerRequest,
    logger,
    settings: Settings,
    storage_client: AsyncClient,
    ckks,
):
    local_start_time = time.time()
    worker_id = settings.node_id
    model_id = body.model_id
    encrypted_model_id = "encrypted{}".format(model_id)
    model_labels_id = "{}labels".format(model_id)
    records_test_id = body.records_test_id
    distances_id = "distances{}".format(records_test_id)
    encrypted_records_id = "encrypted{}".format(records_test_id)
    algorithm = Constants.ClassificationAlgorithms.SKNN_PQC_PREDICT
    _encrypted_model_shape = body.encrypted_model_shape
    _encrypted_model_dtype = body.encrypted_model_dtype
    _encrypted_records_shape = body.encrypted_records_shape
    _encrypted_records_dtype = body.encrypted_records_dtype
    experiment_id = body.experiment_id or ""
    mictlanx_timeout = settings.mictlanx_timeout
    mictlanx_delay = settings.mictlanx_delay
    mictlanx_backoff_factor = settings.mictlanx_backoff_factor
    mictlanx_max_retries = settings.mictlanx_max_retries
    mictlanx_chunk_size = settings.mictlanx_chunk_size
    mictlanx_max_parallel_gets = settings.mictlanx_max_parallel_gets

    if _encrypted_model_dtype is None:
        raise HTTPException(status_code=500, detail="Encrypted-Model-Dtype")
    if _encrypted_model_shape is None:
        raise HTTPException(status_code=500, detail="Encrypted-Model-Shape header is required")

    if _encrypted_records_dtype is None:
        raise HTTPException(status_code=500, detail="Encrypted-Records-Dtype")
    if _encrypted_records_shape is None:
        raise HTTPException(status_code=500, detail="Encrypted-Records-Shape header is required")

    encrypted_model_shape = eval(_encrypted_model_shape)
    encrypted_records_shape = eval(_encrypted_records_shape)
    num_chunks_str = body.num_chunks
    if num_chunks_str is None:
        raise HTTPException(status_code=503, detail="Num-Chunks header is required")
    num_chunks = int(num_chunks_str)

    try:
        get_merge_encrypted_model_start_time = time.time()
        encrypted_model = await RoryCommon.get_pyctxt_matrix(
            client=storage_client,
            bucket_id=settings.mictlanx_bucket_id,
            key=encrypted_model_id,
            ckks=ckks,
            backoff_factor=mictlanx_backoff_factor,
            chunk_size=mictlanx_chunk_size,
            delay=mictlanx_delay,
            max_paralell_gets=mictlanx_max_parallel_gets,
            max_retries=mictlanx_max_retries,
            timeout=mictlanx_timeout,
        )

        get_encrypted_ptm_entry = ExperimentLogEntry(
            event="GET",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=get_merge_encrypted_model_start_time,
            end_time=time.time(),
            id=encrypted_model_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
        )
        logger.info(get_encrypted_ptm_entry.model_dump())

        t1 = time.time()
        encrypted_records = await RoryCommon.get_pyctxt_matrix(
            client=storage_client,
            bucket_id=settings.mictlanx_bucket_id,
            key=encrypted_records_id,
            ckks=ckks,
            backoff_factor=mictlanx_backoff_factor,
            chunk_size=mictlanx_chunk_size,
            delay=mictlanx_delay,
            headers={},
            max_paralell_gets=mictlanx_max_parallel_gets,
            max_retries=mictlanx_max_retries,
            timeout=mictlanx_timeout,
        )

        get_encrypted_ptm_entry = ExperimentLogEntry(
            event="GET",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=t1,
            end_time=time.time(),
            id=encrypted_records_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
        )
        logger.info(get_encrypted_ptm_entry.model_dump())

        t1 = time.time()
        all_distances = SKNNPQC.calculate_distances(
            dataset=encrypted_records,
            model=encrypted_model,
            model_shape=encrypted_model_shape,
            dataset_shape=encrypted_records_shape,
        )

        calculate_distances_entry = ExperimentLogEntry(
            event="CALCULATE.DISTANCES",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=t1,
            end_time=time.time(),
            id=distances_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
        )
        logger.info(calculate_distances_entry.model_dump())

        distances_shape = all_distances.shape
        distances_dtype = all_distances.dtype
        maybe_distances_chunks = RoryCommon.from_pyctxt_matrix_to_chunks(
            key=distances_id,
            num_chunks=num_chunks,
            xs=all_distances,
        )
        if maybe_distances_chunks.is_none:
            raise HTTPException(status_code=500, detail="Failed to create distances chunks")

        t1 = time.time()
        z = await RoryCommon.put_chunks(
            client=storage_client,
            bucket_id=settings.mictlanx_bucket_id,
            key=distances_id,
            chunks=maybe_distances_chunks.unwrap(),
        )

        put_encrypted_ptm_entry = ExperimentLogEntry(
            event="PUT",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=t1,
            end_time=time.time(),
            id=distances_id,
            worker_id="",
            num_chunks=num_chunks,
        )
        logger.info(put_encrypted_ptm_entry.model_dump())

        end_time = time.time()
        service_time = end_time - local_start_time

        classification_completed_entry = ExperimentLogEntry(
            event="COMPLETED",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=local_start_time,
            end_time=time.time(),
            id=model_id,
            num_chunks=num_chunks,
            time=service_time,
        )
        logger.info(classification_completed_entry.model_dump())

        return {
            "distances_id": distances_id,
            "distances_shape": str(distances_shape),
            "distances_dtype": str(distances_dtype),
            "service_time": service_time,
        }
    except Exception as e:
        logger.error({"msg": str(e)})
        raise HTTPException(status_code=503, detail=str(e))


async def sknn_pqc_predict_2(
    body: PqcSknnPredictWorkerRequest,
    logger,
    settings: Settings,
    storage_client: AsyncClient,
):
    local_start_time = time.time()
    worker_id = settings.node_id
    model_id = body.model_id
    model_labels_id = "{}labels".format(model_id)
    records_test_id = body.records_test_id
    min_distances_index_id = "distancesindex{}".format(records_test_id)
    algorithm = Constants.ClassificationAlgorithms.SKNN_PQC_PREDICT
    experiment_id = body.experiment_id or ""
    mictlanx_timeout = settings.mictlanx_timeout
    mictlanx_delay = settings.mictlanx_delay
    mictlanx_backoff_factor = settings.mictlanx_backoff_factor
    mictlanx_max_retries = settings.mictlanx_max_retries
    mictlanx_chunk_size = settings.mictlanx_chunk_size
    mictlanx_max_parallel_gets = settings.mictlanx_max_parallel_gets

    try:
        model_labels_get_start_time = time.time()
        model_labels = await RoryCommon.get_and_merge(
            client=storage_client,
            key=model_labels_id,
            bucket_id=settings.mictlanx_bucket_id,
            backoff_factor=mictlanx_backoff_factor,
            chunk_size=mictlanx_chunk_size,
            delay=mictlanx_delay,
            max_paralell_gets=mictlanx_max_parallel_gets,
            max_retries=mictlanx_max_retries,
            timeout=mictlanx_timeout,
        )

        get_encrypted_ptm_entry = ExperimentLogEntry(
            event="GET",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=model_labels_get_start_time,
            end_time=time.time(),
            id=model_labels_id,
            worker_id=worker_id,
        )
        logger.info(get_encrypted_ptm_entry.model_dump())

        min_distances_start_time = time.time()
        min_distances_index = await RoryCommon.get_and_merge(
            client=storage_client,
            key=min_distances_index_id,
            bucket_id=settings.mictlanx_bucket_id,
            backoff_factor=mictlanx_backoff_factor,
            chunk_size=mictlanx_chunk_size,
            delay=mictlanx_delay,
            max_paralell_gets=mictlanx_max_parallel_gets,
            max_retries=mictlanx_max_retries,
            timeout=mictlanx_timeout,
        )

        get_encrypted_ptm_entry = ExperimentLogEntry(
            event="GET",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=min_distances_start_time,
            end_time=time.time(),
            id=min_distances_index_id,
            worker_id=worker_id,
        )
        logger.info(get_encrypted_ptm_entry.model_dump())

        label_vector = SKNNPQC.get_label_vector(
            model_labels=model_labels.flatten(),
            min_indexes=min_distances_index.flatten(),
        )
        end_time = time.time()
        service_time = end_time - local_start_time

        classification_completed_entry = ExperimentLogEntry(
            event="COMPLETED",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=local_start_time,
            end_time=time.time(),
            id=model_id,
            time=service_time,
        )
        logger.info(classification_completed_entry.model_dump())

        return {
            "label_vector": list(map(int, label_vector.flatten().tolist())),
            "service_time": service_time,
        }
    except Exception as e:
        logger.error({"msg": str(e)})
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/pqc/sknn/predict")
async def sknn_pqc_predict(
    body: PqcSknnPredictWorkerRequest,
    logger=Depends(get_logger),
    settings: Settings = Depends(get_settings),
    storage_client: AsyncClient = Depends(get_storage_client),
    ckks=Depends(get_ckks),
):
    step_index = body.step_index
    if step_index == 1:
        return await sknn_pqc_pedict_1(body, logger, settings, storage_client, ckks)
    elif step_index == 2:
        return await sknn_pqc_predict_2(body, logger, settings, storage_client)
    else:
        raise HTTPException(status_code=400, detail="Invalid step_index")
