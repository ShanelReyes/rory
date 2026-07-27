import os
import time
import json
import numpy as np
from uuid import uuid4
from requests import Session
from fastapi import APIRouter, Depends, HTTPException
from rory.core.interfaces.rorymanager import RoryManager
from rory.core.interfaces.roryworker import RoryWorker
from rory.core.security.dataowner import DataOwner
from rory.core.security.pqc.dataowner import DataOwner as DataOwnerPQC
from rory.core.security.cryptosystem.liu import Liu
from rory.core.utils.constants import Constants
from rorycommon import Common as RoryCommon
from mictlanx import AsyncClient
from mictlanx.utils.segmentation import Chunks
from concurrent.futures import ProcessPoolExecutor
from option import Some
from models import ExperimentLogEntry
from rory.core.security.cryptosystem.pqc.ckks import Ckks

from dependencies import (
    get_ckks,
    get_dataowner,
    get_executor,
    get_liu,
    get_logger,
    get_manager,
    get_settings,
    get_storage_client,
)
from config import Settings
from models.requests.classification import (
    KnnTrainRequest,
    KnnPredictRequest,
    SknnTrainRequest,
    SknnPredictRequest,
    PqcSknnTrainRequest,
    PqcSknnPredictRequest,
)
from models.responses.classification import (
    KnnTrainResponse,
    PredictResponse,
    SknnTrainResponse,
    PqcSknnTrainResponse,
)
from models.responses.clustering import HealthCheckResponse

router = APIRouter(prefix="/classification", tags=["Classification"])


@router.api_route(
    "/test",
    methods=["GET", "POST"],
    response_model=HealthCheckResponse,
    summary="Classification health check",
    description="Diagnostic and health check endpoint for the classification component.",
)
def test():
    """Diagnostic and health check endpoint for the classification component.

    This method provides a simple mechanism to verify that the
    classification routes are active and reachable. It is primarily used
    by the Rory platform's orchestration layer to identify the node type
    and ensure proper network synchronization before initiating machine
    learning workflows.

    Returns:
        Response: A JSON payload containing:
            component_type (str): "dataowner".

        Headers:
            Component-Type: "dataowner"

        Status Code:
            200: If the classification service is operational.
    """
    return {"component_type": "dataowner"}


@router.post(
    "/sknn/train",
    response_model=SknnTrainResponse,
    summary="Secure KNN training using Liu homomorphic encryption",
    description="Encrypts the model matrix with Liu scheme and uploads to Cloud Storage.",
)
async def sknn_train(
    body: SknnTrainRequest,
    logger=Depends(get_logger),
    settings: Settings = Depends(get_settings),
    liu: Liu = Depends(get_liu),
    dataowner: DataOwner = Depends(get_dataowner),
    STORAGE_CLIENT: AsyncClient = Depends(get_storage_client),
    executor: ProcessPoolExecutor = Depends(get_executor),
):
    """
    This method manages the "training" phase for the privacy-preserving KNN algorithm.
    Unlike the standard KNN preparation, this workflow incorporates Liu's
    homomorphic encryption scheme to protect the model's feature matrix. The Client
    performs local encryption using a ProcessPoolExecutor before segmenting and
    secure uploading it to the Cloud Storage System (CSS).

    Note:
    **Model Generation**: Configuration for model training, including target storage IDs,
    must be provided through **HTTP Headers**.

    Attributes:
        Model-Id (str): Unique identifier for the model and its labels.
            Defaults to "matrix-0_model".
        Model-Filename (str): Local name of the feature matrix file.
        Model-Labels-Filename (str): Local name of the labels vector file.
        Experiment-Id (str): Unique ID for performance tracking.
        Extension (str): Source data file extension. Defaults to "npy".

    Returns:
        response_time (float): Total end-to-end preparation time.
        encrypted_model_shape (str): The 3D shape of the Liu-encrypted matrix
            (Rows, Attributes, security parameter M).
        encrypted_model_dtype (str): Data type of the encrypted matrix.
        algorithm (str): The specific constant for sknn_train.
        model_labels_shape (list): Dimensions of the uploaded labels matrix.

    Raises:
        HTTPException (500): If the ProcessPoolExecutor is missing or if the model/label
            files are not found in the local source path.
        HTTPException (500): If any error occurs during the encryption chain,
            chunking process, or CSS communication.
    """
    try:
        local_start_time = time.time()
        BUCKET_ID: str = settings.mictlanx_bucket_id
        SOURCE_PATH = settings.source_path
        max_workers = settings.max_workers
        num_chunks = settings.num_chunks
        np_random = settings.np_random
        security_level = settings.liu_security_level
        algorithm = Constants.ClassificationAlgorithms.SKNN_TRAIN
        s = Session()
        experiment_id = body.experiment_id
        model_id = body.model_id
        model_filename = body.model_filename
        model_labels_id = "{}labels".format(model_id)
        model_labels_filename = body.model_labels_filename
        encrypted_model_id = "encrypted{}".format(model_id)
        extension = body.extension
        m = dataowner.m
        model_path = "{}/{}.{}".format(SOURCE_PATH, model_filename, extension)
        model_labels_path = "{}/{}.{}".format(SOURCE_PATH, model_labels_filename, extension)
        MICTLANX_TIMEOUT = settings.mictlanx_timeout
        model_path_exists = os.path.exists(model_path)
        model_path_labels_exists = os.path.exists(model_labels_path)
        if not model_path_exists or not model_path_labels_exists:
            raise HTTPException(status_code=500, detail="Either model or label vector not found")
        else:
            model_result = await RoryCommon.read_numpy_from(
                path=model_path,
                extension="npy"
            )
            if model_result.is_err:
                raise HTTPException(status_code=500, detail="Something went wrong reading the model")
            model = model_result.unwrap()
            read_local_model_labels_start_time = time.time()

            model_labels_result = await RoryCommon.read_numpy_from(
                extension="npy",
                path=model_labels_path
            )
            if model_labels_result.is_err:
                raise HTTPException(status_code=500, detail="Something went wrong reading the model labels")
            model_labels = model_labels_result.unwrap()
            model_labels = model_labels.reshape((1, model_labels.shape[0]))

            local_read_entry = ExperimentLogEntry(
                event="LOCAL.READ",
                experiment_id=experiment_id,
                algorithm=algorithm,
                start_time=read_local_model_labels_start_time,
                end_time=time.time(),
                id=model_id,
                worker_id="",
                num_chunks=num_chunks,
                security_level=security_level,
                m=m
            )
            logger.info(local_read_entry.model_dump())

            put_model_labels_start_time = time.time()
            maybe_models_labels_chunks = Chunks.from_ndarray(
                ndarray=model_labels,
                group_id=model_labels_id,
                num_chunks=1,
                chunk_prefix=Some(model_labels_id)
            )
            if maybe_models_labels_chunks.is_none:
                raise HTTPException(status_code=500, detail="Something went wrong generating the chunks of model labels")

            put_model_labels_result = await RoryCommon.delete_and_put_chunks(
                client=STORAGE_CLIENT,
                bucket_id=BUCKET_ID,
                key=model_labels_id,
                chunks=maybe_models_labels_chunks.unwrap(),
                timeout=MICTLANX_TIMEOUT,
                tags={
                    "full_dtype": str(model_labels.dtype),
                    "full_shape": str(model_labels.shape)
                }
            )

            put_encrypted_ptm_entry = ExperimentLogEntry(
                event="PUT",
                experiment_id=experiment_id,
                algorithm=algorithm,
                start_time=put_model_labels_start_time,
                end_time=time.time(),
                id=model_id,
                worker_id="",
                num_chunks=num_chunks,
                security_level=security_level,
                m=m
            )
            logger.info(put_encrypted_ptm_entry.model_dump())

            r: int = model.shape[0]
            a: int = model.shape[1]
            encrypted_model_shape = "({},{},{})".format(r, a, m)
            n = a * r * int(m)

            segment_encrypt_model_start_time = time.time()
            encrypted_model_chunks = RoryCommon.segment_and_encrypt_liu_with_executor(
                executor=executor,
                key=encrypted_model_id,
                plaintext_matrix=model,
                dataowner=dataowner,
                n=n,
                num_chunks=num_chunks,
                np_random=np_random
            )

            segment_encrypt_entry = ExperimentLogEntry(
                event="SEGMENT.ENCRYPT",
                experiment_id=experiment_id,
                algorithm=algorithm,
                start_time=segment_encrypt_model_start_time,
                end_time=time.time(),
                id=model_id,
                worker_id="",
                num_chunks=num_chunks,
                security_level=security_level,
                m=m
            )
            logger.info(segment_encrypt_entry.model_dump())

            put_chunked_start_time = time.time()
            encrypted_model_put_chunks = await RoryCommon.delete_and_put_chunks(
                client=STORAGE_CLIENT,
                bucket_id=BUCKET_ID,
                key=encrypted_model_id,
                chunks=encrypted_model_chunks,
                timeout=MICTLANX_TIMEOUT,
                tags={
                    "full_shape": str(encrypted_model_shape),
                    "full_dtype": "float32"
                }
            )

            put_encrypted_ptm_entry = ExperimentLogEntry(
                event="PUT",
                experiment_id=experiment_id,
                algorithm=algorithm,
                start_time=put_chunked_start_time,
                end_time=time.time(),
                id=model_id,
                worker_id="",
                num_chunks=num_chunks,
                security_level=security_level,
                m=m
            )
            logger.info(put_encrypted_ptm_entry.model_dump())

            endTime = time.time()
            response_time = endTime - local_start_time

            classification_completed_entry = ExperimentLogEntry(
                event="COMPLETED",
                experiment_id=experiment_id,
                algorithm=algorithm,
                start_time=local_start_time,
                end_time=time.time(),
                id=model_id,
                num_chunks=num_chunks,
                security_level=security_level,
                workers=max_workers,
                time=response_time
            )
            logger.info(classification_completed_entry.model_dump())

            return {
                "response_time": response_time,
                "encrypted_model_shape": str(encrypted_model_shape),
                "encrypted_model_dtype": "float32",
                "algorithm": algorithm,
                "model_labels_shape": list(model_labels.shape)
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error({
            "msg": str(e)
        })
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/sknn/predict",
    response_model=PredictResponse,
    summary="Secure KNN prediction using Liu homomorphic encryption",
    description="Interactive 2-round protocol with decryption oracle for secure nearest neighbor search.",
)
async def sknn_predict(
    body: SknnPredictRequest,
    logger=Depends(get_logger),
    settings: Settings = Depends(get_settings),
    liu: Liu = Depends(get_liu),
    dataowner: DataOwner = Depends(get_dataowner),
    STORAGE_CLIENT: AsyncClient = Depends(get_storage_client),
    executor: ProcessPoolExecutor = Depends(get_executor),
    managerResponse: RoryManager = Depends(get_manager),
):
    """
    This method orchestrates a privacy-preserving K-Nearest Neighbors prediction
    using Liu's homomorphic encryption scheme. It follows a protocol where the Worker
    performs heavy computations on encrypted data, and
    the Client acts as a decryption oracle to identify the nearest neighbors
    without revealing plaintext information to the remote nodes.

    Note:
    **Secure Inference**: This method manages the transition between encryption domains.
    All required keys and IDs must be in the **HTTP Headers**.

    Attributes:
        Model-Id (str): ID of the encrypted model in CSS. Defaults to "model0".
        Records-Test-Id (str): ID for the encrypted test records.
        Encrypted-Model-Shape (str): The 3D shape of the model (r, a, m).
        Encrypted-Model-Dtype (str): Data type of the encrypted model.
        Model-Labels-Shape (str): The shape of the labels vector.
        Experiment-Id (str): Tracking ID for performance auditing.

    Returns:
        label_vector (list): The final predicted classes.
        worker_id (str): ID of the node that handled the encrypted computation.
        service_time_metrics: Timing data for Client, Manager, and Worker.
        algorithm (str): The specific constant for sknn_predict.

    Raises:
        HTTPException (500): If the process executor is missing or if mandatory
            headers (Shape/Dtype) are not provided.
        HTTPException (500): If errors occur during the interactive encryption/decryption
            chain or during Worker orchestration.
    """
    try:
        local_start_time = time.time()
        BUCKET_ID: str = settings.mictlanx_bucket_id
        TESTING = settings.testing
        SOURCE_PATH = settings.source_path
        max_workers = settings.max_workers
        num_chunks = settings.num_chunks
        np_random = settings.np_random
        security_level = settings.liu_security_level
        WORKER_TIMEOUT = settings.worker_timeout
        algorithm = Constants.ClassificationAlgorithms.SKNN_PREDICT
        s = Session()
        model_id = body.model_id
        model_filename = body.model_filename
        records_test_id = body.records_test_id
        records_test_filename = body.records_test_filename
        encrypted_records_test_id = "encrypted{}".format(records_test_id)
        extension = body.extension
        m = dataowner.m
        model_labels_id = "{}labels".format(model_id)
        _encrypted_model_shape = body.encrypted_model_shape
        _encrypted_model_dtype = body.encrypted_model_dtype
        _model_labels_shape = body.model_labels_shape
        experiment_id = body.experiment_id
        records_test_path = "{}/{}.{}".format(SOURCE_PATH, records_test_filename, extension)
        MICTLANX_TIMEOUT = settings.mictlanx_timeout
        MICTLANX_DELAY = settings.mictlanx_delay
        MICTLANX_BACKOFF_FACTOR = settings.mictlanx_backoff_factor
        MICTLANX_MAX_RETRIES = settings.mictlanx_max_retries

        read_local_start_time = time.time()
        records_test_ext = "npy"
        records_test_result = await RoryCommon.read_numpy_from(
            path=records_test_path,
            extension=records_test_ext
        )
        read_local_st = time.time() - read_local_start_time
        if records_test_result.is_err:
            raise HTTPException(status_code=500, detail=f"Failed to read {records_test_path}")
        records_test = records_test_result.unwrap()

        local_read_entry = ExperimentLogEntry(
            event="LOCAL.READ",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=read_local_st,
            end_time=time.time(),
            id=model_id,
            worker_id="",
            num_chunks=num_chunks,
            security_level=security_level,
            m=m
        )
        logger.info(local_read_entry.model_dump())

        r: int = records_test.shape[0]
        a: int = records_test.shape[1]
        n = a * r * int(m)

        segment_encrypt_start_time = time.time()
        encrypted_records_chunks = RoryCommon.segment_and_encrypt_liu_with_executor(
            executor=executor,
            key=encrypted_records_test_id,
            dataowner=dataowner,
            plaintext_matrix=records_test,
            n=n,
            num_chunks=num_chunks,
            np_random=np_random
        )

        segment_encrypt_entry = ExperimentLogEntry(
            event="SEGMENT.ENCRYPT",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=segment_encrypt_start_time,
            end_time=time.time(),
            id=model_id,
            worker_id="",
            num_chunks=num_chunks,
            security_level=security_level,
            m=m
        )
        logger.info(segment_encrypt_entry.model_dump())

        put_chunks_start_time = time.time()
        encrypted_records_shape = (r, a, int(m))

        put_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
            client=STORAGE_CLIENT,
            bucket_id=BUCKET_ID,
            key=encrypted_records_test_id,
            chunks=encrypted_records_chunks,
            timeout=MICTLANX_TIMEOUT,
            tags={
                "full_shape": str(encrypted_records_shape),
                "full_dtype": "float32"
            }
        )
        if put_chunks_generator_results.is_err:
            raise HTTPException(status_code=500, detail="Failed to put encrypted records test in the storage")
        service_time_dataowner = time.time() - local_start_time

        put_encrypted_ptm_entry = ExperimentLogEntry(
            event="PUT",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=put_chunks_start_time,
            end_time=time.time(),
            id=model_id,
            worker_id="",
            num_chunks=num_chunks,
            security_level=security_level,
            m=m
        )
        logger.info(put_encrypted_ptm_entry.model_dump())

        get_worker_start_time = time.time()
        get_worker_result = managerResponse.getWorker(
            headers={
                "Algorithm": algorithm,
                "Start-Request-Time": str(local_start_time),
                "Start-Get-Worker-Time": str(get_worker_start_time),
                "Matrix-Id": model_id
            }
        )

        if get_worker_result.is_err:
            error = get_worker_result.unwrap_err()
            logger.error(str(error))
            raise HTTPException(status_code=500, detail=str(error))
        (_worker_id, port) = get_worker_result.unwrap()

        get_worker_end_time = time.time()
        get_worker_service_time = get_worker_end_time - get_worker_start_time
        worker_id = "localhost" if TESTING else _worker_id

        get_worker_entry = ExperimentLogEntry(
            event="GET.WORKER",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=get_worker_start_time,
            end_time=time.time(),
            id=model_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            workers=max_workers,
            security_level=security_level,
            m=m
        )
        logger.info(get_worker_entry.model_dump())

        worker_start_time = time.time()
        worker = RoryWorker(
            workerId=worker_id,
            port=port,
            session=s,
            algorithm=algorithm,
        )

        encrypted_records_dtype = "float32"
        run1_time = time.time()
        run1_headers = {
            "Step-Index": "1",
            "Records-Test-Id": records_test_id,
            "Model-Id": model_id,
            "Encrypted-Model-Shape": _encrypted_model_shape,
            "Encrypted-Model-Dtype": _encrypted_model_dtype,
            "Encrypted-Records-Shape": str(encrypted_records_shape),
            "Encrypted-Records-Dtype": str(encrypted_records_dtype),
            "Num-Chunks": str(num_chunks),
            "Model-Labels-Shape": _model_labels_shape
        }

        worker_run1_response = worker.run(
            headers=run1_headers,
            timeout=WORKER_TIMEOUT
        )
        worker_run1_response.raise_for_status()

        jsonWorkerResponse = worker_run1_response.json()
        endTime = time.time()
        distances_id = jsonWorkerResponse["distances_id"]
        distances_shape = jsonWorkerResponse["distances_shape"]
        distances_dtype = jsonWorkerResponse["distances_dtype"]
        worker_service_time = jsonWorkerResponse["service_time"]

        run1_worker_entry = ExperimentLogEntry(
            event="RUN1",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=run1_time,
            end_time=time.time(),
            id=model_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            m=m,
            workers=max_workers,
            security_level=security_level
        )
        logger.info(run1_worker_entry.model_dump())

        get_all_distances_start_time = time.time()
        all_distances = await RoryCommon.get_and_merge(
            client=STORAGE_CLIENT,
            key=distances_id,
            bucket_id=BUCKET_ID,
            max_retries=MICTLANX_MAX_RETRIES,
            delay=MICTLANX_DELAY,
            backoff_factor=MICTLANX_BACKOFF_FACTOR,
            timeout=MICTLANX_TIMEOUT
        )

        get_encrypted_entry = ExperimentLogEntry(
            event="GET",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=get_all_distances_start_time,
            end_time=time.time(),
            id=model_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            m=m,
            workers=max_workers,
            security_level=security_level
        )
        logger.info(get_encrypted_entry.model_dump())

        decrypt_matrix_start_time = time.time()
        matrix_distances_plain = liu.decryptMatrix(
            ciphertext_matrix=all_distances,
            secret_key=dataowner.sk,
        )

        min_distances_index = np.argmin(matrix_distances_plain.matrix, axis=1)
        min_distances_index_id = "distancesindex{}".format(records_test_id)
        decrypt_matrix_end_time = time.time()
        decrypt_matrix_service_time = decrypt_matrix_start_time - decrypt_matrix_end_time

        decrypt_entry = ExperimentLogEntry(
            event="DECRYPT",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=decrypt_matrix_start_time,
            end_time=time.time(),
            id=model_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            m=m,
            workers=max_workers,
            security_level=security_level
        )
        logger.info(decrypt_entry.model_dump())

        maybe_min_distances_chunks = Chunks.from_ndarray(
            ndarray=min_distances_index.reshape(-1, 1),
            group_id=min_distances_index_id,
            chunk_prefix=Some(min_distances_index_id),
            num_chunks=num_chunks,
        )

        if maybe_min_distances_chunks.is_none:
            raise HTTPException(status_code=500, detail="something went wrong creating the chunks")

        t_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
            client=STORAGE_CLIENT,
            bucket_id=BUCKET_ID,
            key=min_distances_index_id,
            chunks=maybe_min_distances_chunks.unwrap(),
            tags={
                "full_shape": str(min_distances_index.shape),
                "full_dtype": str(min_distances_index.dtype)
            }
        )

        run2_time = time.time()
        run2_headers = {
            "Step-Index": "2",
            "Records-Test-Id": records_test_id,
            "Model-Id": model_id,
            "Encrypted-Model-Shape": _encrypted_model_shape,
            "Encrypted-Model-Dtype": _encrypted_model_dtype,
            "Encrypted-Records-Shape": str(encrypted_records_shape),
            "Encrypted-Records-Dtype": str(encrypted_records_dtype),
            "Num-Chunks": str(num_chunks),
            "Min_Distances_Index_Id": min_distances_index_id,
            "Model-Labels-Shape": _model_labels_shape
        }

        worker_run2_response = worker.run(
            headers=run2_headers,
            timeout=WORKER_TIMEOUT
        )
        worker_run2_response.raise_for_status()
        jsonWorkerResponse2 = worker_run2_response.json()
        service_time_worker = worker_run2_response.headers.get("Service-Time", 0)
        worker_end_time = time.time()
        worker_response_time = worker_end_time - worker_start_time

        run2_worker_entry = ExperimentLogEntry(
            event="RUN2",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=run2_time,
            end_time=time.time(),
            id=model_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            m=m,
            workers=max_workers,
            security_level=security_level,
        )
        logger.info(run2_worker_entry.model_dump())

        response_time = endTime - local_start_time

        classification_completed_entry = ExperimentLogEntry(
            event="COMPLETED",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=local_start_time,
            end_time=time.time(),
            id=model_id,
            num_chunks=num_chunks,
            security_level=security_level,
            workers=max_workers,
            time=response_time,
            m=m
        )
        logger.info(classification_completed_entry.model_dump())

        label_vector = jsonWorkerResponse2["label_vector"]
        return {
            "label_vector": label_vector,
            "worker_id": worker_id,
            "service_time_manager": get_worker_service_time,
            "service_time_worker": worker_response_time,
            "service_time_dataowner": service_time_dataowner,
            "service_time_predict": response_time,
            "algorithm": algorithm,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error({
            "msg": str(e)
        })
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/knn/train",
    response_model=KnnTrainResponse,
    summary="KNN training (plaintext reference dataset upload)",
    description="Uploads the plaintext feature matrix and labels to Cloud Storage for distributed KNN.",
)
async def knn_train(
    body: KnnTrainRequest,
    logger=Depends(get_logger),
    settings: Settings = Depends(get_settings),
    STORAGE_CLIENT: AsyncClient = Depends(get_storage_client),
    executor: ProcessPoolExecutor = Depends(get_executor),
):
    """
    This method handles the "training" phase of the KNN algorithm within the Rory
    platform's distributed architecture. Since KNN is a lazy learner, this phase
    focuses on reading the reference dataset (features and labels) from local
    storage, segmenting them into chunks, and uploading them to the Cloud Storage
    System (CSS). This ensures that execution nodes (Workers) can access the
    model data for future prediction tasks.

    Note:
    **Model Generation**: Configuration for model training, including target storage IDs,
    must be provided through **HTTP Headers**.

    Attributes:
        Model-Id (str): Unique identifier for the model. Defaults to "matrix0model".
        Model-Filename (str): Local name of the feature matrix file (without extension).
        Model-Labels-Filename (str): Local name of the labels file (without extension).
        Extension (str): File extension of the source data. Defaults to "npy".
        Experiment-Id (str): Unique tracking ID for auditing and benchmarking.

    Returns:
        response_time (float): Total time taken for the preparation and upload.
        algorithm (str): The specific algorithm constant (knn_train).
        model_labels_shape (list): The final dimensions of the uploaded labels matrix.

    Raises:
        HTTPException (500): If the process pool executor is not available in the
            app configuration.
        HTTPException (500): If local file reading fails or if errors occur during
            the chunking and upload process to the CSS.
    """
    local_start_time = time.time()
    BUCKET_ID: str = settings.mictlanx_bucket_id
    SOURCE_PATH = settings.source_path
    max_workers = settings.max_workers
    num_chunks = body.num_chunks
    algorithm = Constants.ClassificationAlgorithms.KNN_TRAIN
    s = Session()
    model_id = body.model_id
    model_filename = body.model_filename
    model_labels_id = "{}labels".format(model_id)
    model_labels_filename = body.model_labels_filename
    extension = body.extension
    experiment_id = body.experiment_id
    model_path = "{}/{}.{}".format(SOURCE_PATH, model_filename, extension)
    model_labels_path = "{}/{}.{}".format(SOURCE_PATH, model_labels_filename, extension)

    get_model_start_time = time.time()
    model_ext = "npy"
    model_result = await RoryCommon.read_numpy_from(
        path=model_path,
        extension=model_ext
    )

    if model_result.is_err:
        raise HTTPException(status_code=500, detail="Failed to read model")

    model = model_result.unwrap()

    local_read_entry = ExperimentLogEntry(
        event="LOCAL.READ",
        experiment_id=experiment_id,
        algorithm=algorithm,
        start_time=get_model_start_time,
        end_time=time.time(),
        id=model_id,
        worker_id="",
        num_chunks=num_chunks,
    )
    logger.info(local_read_entry.model_dump())

    model_labels_ext = "npy"
    model_labels_result = await RoryCommon.read_numpy_from(
        path=model_labels_path,
        extension=model_labels_ext
    )

    if model_labels_result.is_err:
        raise HTTPException(status_code=500, detail="Failed to read model labels")

    model_labels = model_labels_result.unwrap()
    model_labels = model_labels.reshape(1, -1)

    put_model_start_time = time.time()
    maybe_model_chunks = Chunks.from_ndarray(
        ndarray=model,
        group_id=model_id,
        chunk_prefix=Some(model_id),
        num_chunks=num_chunks,
    )

    if maybe_model_chunks.is_none:
        raise HTTPException(status_code=500, detail="something went wrong creating the chunks")

    put_model_result = await RoryCommon.delete_and_put_chunks(
        client=STORAGE_CLIENT,
        bucket_id=BUCKET_ID,
        key=model_id,
        chunks=maybe_model_chunks.unwrap(),
        tags={
            "full_shape": str(model.shape),
            "full_dtype": str(model.dtype)
        }
    )

    put_encrypted_model_entry = ExperimentLogEntry(
        event="PUT",
        experiment_id=experiment_id,
        algorithm=algorithm,
        start_time=put_model_start_time,
        end_time=time.time(),
        id=model_id,
        worker_id="",
        num_chunks=num_chunks
    )
    logger.info(put_encrypted_model_entry.model_dump())

    put_model_labels_start_time = time.time()
    maybe_model_labels_chunks = Chunks.from_ndarray(
        ndarray=model_labels,
        group_id=model_labels_id,
        chunk_prefix=Some(model_labels_id),
        num_chunks=num_chunks,
    )

    if maybe_model_labels_chunks.is_none:
        raise HTTPException(status_code=500, detail="something went wrong creating the chunks")

    model_labels_results = await RoryCommon.delete_and_put_chunks(
        client=STORAGE_CLIENT,
        bucket_id=BUCKET_ID,
        key=model_labels_id,
        chunks=maybe_model_labels_chunks.unwrap(),
        tags={
            "full_shape": str(model_labels.shape),
            "full_dtype": str(model_labels.dtype)
        }
    )

    put_encrypted_model_labels_entry = ExperimentLogEntry(
        event="PUT",
        experiment_id=experiment_id,
        algorithm=algorithm,
        start_time=put_model_labels_start_time,
        end_time=time.time(),
        id=model_id,
        worker_id="",
        num_chunks=num_chunks
    )
    logger.info(put_encrypted_model_labels_entry.model_dump())

    end_time = time.time()
    response_time = end_time - local_start_time

    classification_completed_entry = ExperimentLogEntry(
        event="COMPLETED",
        experiment_id=experiment_id,
        algorithm=algorithm,
        start_time=local_start_time,
        end_time=time.time(),
        id=model_id,
        num_chunks=num_chunks,
        time=response_time
    )
    logger.info(classification_completed_entry.model_dump())

    return {
        "response_time": response_time,
        "algorithm": algorithm,
        "model_labels_shape": list(model_labels.shape)
    }


@router.post(
    "/knn/predict",
    response_model=PredictResponse,
    summary="KNN prediction (distributed nearest neighbor search)",
    description="Orchestrates distributed KNN classification with plaintext data across Manager/Worker.",
)
async def knn_predict(
    body: KnnPredictRequest,
    logger=Depends(get_logger),
    settings: Settings = Depends(get_settings),
    STORAGE_CLIENT: AsyncClient = Depends(get_storage_client),
    executor: ProcessPoolExecutor = Depends(get_executor),
    managerResponse: RoryManager = Depends(get_manager),
):
    """
    This method orchestrates the classification of new data points using a
    previously "trained" (externalized) KNN model. The Client reads the test
    records locally, uploads them to the Cloud Storage System (CSS), and then
    communicates with the Manager and a designated Worker to perform the
    distributed nearest neighbor search.

    Note:
    **Plaintext Inference**: Input record identifiers and model mapping parameters
    are passed exclusively via **HTTP Headers**.

    Attributes:
        Model-Id (str): ID of the pre-trained model stored in CSS. Defaults to "model-0".
        Records-Test-Id (str): Unique ID for the test records to be stored in CSS.
        Records-Test-Filename (str): Local filename for the test dataset.
        Model-Labels-Shape (str): The shape of the model's labels (Required for
            distributed distance calculations).
        Extension (str): File extension of the source data. Defaults to "npy".
        Experiment-Id (str): Tracking ID for performance benchmarking.

    Returns:
        label_vector (list): The predicted class for each test record.
        worker_id (str): ID of the worker node that performed the prediction.
        service_time_manager (float): Latency introduced by the Manager interaction.
        service_time_worker (float): Time spent in remote computation.
        service_time_dataowner (float): Time spent in local I/O and data preparation.
        service_time_predict (float): Total end-to-end prediction time.
        algorithm (str): The specific algorithm constant (knn_predict).

    Raises:
        HTTPException (500): If the "Model-Labels-Shape" header is missing, if the
            process executor is not configured, or if any error occurs during
            CSS interaction or Worker communication.
    """
    try:
        local_start_time = time.time()
        BUCKET_ID: str = settings.mictlanx_bucket_id
        TESTING = settings.testing
        SOURCE_PATH = settings.source_path
        max_workers = settings.max_workers
        num_chunks = settings.num_chunks
        WORKER_TIMEOUT = settings.worker_timeout
        MICTLANX_TIMEOUT = settings.mictlanx_timeout
        algorithm = Constants.ClassificationAlgorithms.KNN_PREDICT
        s = Session()
        model_id = body.model_id
        records_test_id = body.records_test_id
        records_test_filename = body.records_test_filename
        extension = body.extension
        experiment_id = body.experiment_id
        records_test_path = "{}/{}.{}".format(SOURCE_PATH, records_test_filename, extension)
        _model_labels_shape = body.model_labels_shape

        local_read_start_time = time.time()
        records_test_ext = "npy"
        records_test_result = await RoryCommon.read_numpy_from(
            path=records_test_path,
            extension=records_test_ext)
        if records_test_result.is_err:
            raise HTTPException(status_code=500, detail="Failed to local read the records")
        records_test = records_test_result.unwrap()

        local_read_entry = ExperimentLogEntry(
            event="LOCAL.READ",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=local_read_start_time,
            end_time=time.time(),
            id=model_id,
            worker_id="",
            num_chunks=num_chunks,
        )
        logger.info(local_read_entry.model_dump())
        try:
            put_records_start_time = time.time()
            maybe_records_test_chunks = Chunks.from_ndarray(
                ndarray=records_test,
                group_id=records_test_id,
                chunk_prefix=Some(records_test_id),
                num_chunks=num_chunks,
            )

            if maybe_records_test_chunks.is_none:
                logger.error({
                    "error": "Failed to create chunks"
                })
                raise HTTPException(status_code=500, detail="something went wrong creating the chunks")

            put_records_test_result = await RoryCommon.delete_and_put_chunks(
                client=STORAGE_CLIENT,
                bucket_id=BUCKET_ID,
                key=records_test_id,
                chunks=maybe_records_test_chunks.unwrap(),
                timeout=MICTLANX_TIMEOUT,
                tags={
                    "full_shape": str(records_test.shape),
                    "full_dtype": str(records_test.dtype)
                }
            )

            if put_records_test_result.is_err:
                logger.error(str(put_records_test_result.unwrap_err()))
                raise HTTPException(status_code=500, detail="Failed to put the records test")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(str(e))

        service_time_dataowner_end = time.time()
        service_time_dataowner = service_time_dataowner_end - local_start_time

        put_records_entry = ExperimentLogEntry(
            event="PUT",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=put_records_start_time,
            end_time=time.time(),
            id=model_id,
            worker_id="",
            num_chunks=num_chunks,
        )
        logger.info(put_records_entry.model_dump())

        get_worker_start_time = time.time()
        get_worker_result = managerResponse.getWorker(
            headers={
                "Algorithm": algorithm,
                "Start-Request-Time": str(local_start_time),
                "Start-Get-Worker-Time": str(get_worker_start_time),
                "Matrix-Id": model_id
            }
        )

        if get_worker_result.is_err:
            error = get_worker_result.unwrap_err()
            logger.error(str(error))
            raise HTTPException(status_code=500, detail=str(error))

        (_worker_id, port) = get_worker_result.unwrap()

        get_worker_end_time = time.time()
        get_worker_service_time = get_worker_end_time - get_worker_start_time
        worker_id = "localhost" if TESTING else _worker_id

        get_worker_entry = ExperimentLogEntry(
            event="GET.WORKER",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=get_worker_start_time,
            end_time=time.time(),
            id=model_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
        )
        logger.info(get_worker_entry.model_dump())

        worker_start_time = time.time()
        worker = RoryWorker(
            workerId=worker_id,
            port=port,
            session=s,
            algorithm=algorithm,
        )

        workerResponse = worker.run(
            headers={
                "Records-Test-Id": records_test_id,
                "Model-Id": model_id,
                "Model-Labels-Shape": _model_labels_shape
            },
            timeout=WORKER_TIMEOUT
        )
        workerResponse.raise_for_status()

        worker_end_time = time.time()
        worker_response_time = worker_end_time - worker_start_time
        jsonWorkerResponse = workerResponse.json()
        endTime = time.time()
        worker_service_time = jsonWorkerResponse["service_time"]
        label_vector = jsonWorkerResponse["label_vector"]
        response_time = endTime - local_start_time

        classification_completed_entry = ExperimentLogEntry(
            event="COMPLETED",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=local_start_time,
            end_time=time.time(),
            id=model_id,
            num_chunks=num_chunks,
            workers=max_workers,
            time=response_time,
        )
        logger.info(classification_completed_entry.model_dump())

        return {
            "label_vector": label_vector,
            "worker_id": worker_id,
            "service_time_manager": get_worker_service_time,
            "service_time_worker": worker_response_time,
            "service_time_dataowner": service_time_dataowner,
            "service_time_predict": response_time,
            "algorithm": algorithm,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("DATAOWNER_ERROR " + str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/pqc/sknn/train",
    response_model=PqcSknnTrainResponse,
    summary="PQC Secure KNN training using CKKS encryption",
    description="Encrypts the model matrix with CKKS post-quantum scheme and uploads to Cloud Storage.",
)
async def sknn_pqc_train(
    body: PqcSknnTrainRequest,
    logger=Depends(get_logger),
    settings: Settings = Depends(get_settings),
    STORAGE_CLIENT: AsyncClient = Depends(get_storage_client),
    executor: ProcessPoolExecutor = Depends(get_executor),
):
    """
    This method manages the "training" phase for the PQC-enabled Secure K-Nearest Neighbors
    algorithm. It utilizes the CKKS homomorphic encryption scheme to protect
    the model's feature matrix, allowing for complex arithmetic operations on encrypted
    floating-point data. The Client handles the local encryption and segmentation before
    externalizing the secure artifacts to the Cloud Storage System (CSS).

    Note:
    **Post-Quantum Classification**: CKKS parameters and model identifiers are
    strictly handled via **HTTP Headers** to ensure protocol integrity.

    Attributes:
        Model-Id (str): Unique identifier for the model. Defaults to "matrix-0_model".
        Model-Filename (str): Local filename for the feature matrix.
        Model-Labels-Filename (str): Local filename for the label vector.
        Experiment-Id (str): Tracking ID for performance auditing.
        Extension (str): Source file extension. Defaults to "npy".

    Returns:
        response_time (str): Total execution time for the preparation phase.
        encrypted_model_shape (str): The dimensions of the CKKS-encrypted matrix.
        encrypted_model_dtype (str): Data type (float32).
        algorithm (str): The SKNN_PQC_TRAIN constant.
        model_labels_shape (list): Dimensions of the uploaded labels matrix.

    Raises:
        HTTPException (500): If the ProcessPoolExecutor is unavailable or if model/label
            files are missing from the source path.
        HTTPException (500): If errors occur during CKKS encryption, chunking, or
            asynchronous upload to the CSS.
    """
    try:
        local_start_time = time.time()
        BUCKET_ID: str = settings.mictlanx_bucket_id
        SOURCE_PATH = settings.source_path
        max_workers = settings.max_workers
        num_chunks = settings.num_chunks
        np_random = settings.np_random
        security_level = settings.liu_security_level
        algorithm = Constants.ClassificationAlgorithms.SKNN_PQC_TRAIN
        s = Session()
        model_id = body.model_id
        model_filename = body.model_filename
        model_labels_id = "{}labels".format(model_id)
        model_labels_filename = body.model_labels_filename
        encrypted_model_id = "encrypted{}".format(model_id)
        extension = body.extension
        model_path = "{}/{}.{}".format(SOURCE_PATH, model_filename, extension)
        model_labels_path = "{}/{}.{}".format(SOURCE_PATH, model_labels_filename, extension)
        experiment_id = body.experiment_id
        _round = settings.ckks_round
        decimals = settings.ckks_decimals
        path = settings.keys_path
        ctx_filename = settings.ctx_filename
        pubkey_filename = settings.pubkey_filename
        secretkey_filename = settings.secret_key_filename
        relinkey_filename = settings.relinkey_filename
        MICTLANX_TIMEOUT = settings.mictlanx_timeout

        # _______________________________________________________________________________
        ckks = Ckks.from_pyfhel(
            _round=_round,
            decimals=decimals,
            path=path,
            ctx_filename=ctx_filename,
            pubkey_filename=pubkey_filename,
            secretkey_filename=secretkey_filename,
            relinkey_filename=relinkey_filename
        )
        # _______________________________________________________________________________
        dataowner = DataOwnerPQC(scheme=ckks)

        model_path_exists = os.path.exists(model_path)
        model_path_labels_exists = os.path.exists(model_labels_path)
        if not model_path_exists or not model_path_labels_exists:
            raise HTTPException(status_code=500, detail="Either model or label vector not found")
        else:

            read_local_model_start_time = time.time()
            model_result = await RoryCommon.read_numpy_from(
                path=model_path,
                extension="npy"
            )
            if model_result.is_err:
                raise HTTPException(status_code=500, detail="Failed to read the model")
            model = model_result.unwrap()

            local_read_entry = ExperimentLogEntry(
                event="LOCAL.READ",
                experiment_id=experiment_id,
                algorithm=algorithm,
                start_time=read_local_model_start_time,
                end_time=time.time(),
                id=model_id,
                worker_id="",
                num_chunks=num_chunks,
                security_level=security_level,
                workers=max_workers,
                description=f"Read model from: {model_path}"
            )
            logger.info(local_read_entry.model_dump())

            read_local_model_labels_start_time = time.time()

            model_labels_result = await RoryCommon.read_numpy_from(
                path=model_labels_path,
                extension="npy"
            )

            if model_labels_result.is_err:
                raise HTTPException(status_code=500, detail="Failed to read model labels")
            model_labels = model_labels_result.unwrap()
            model_labels = model_labels.reshape((1, model_labels.shape[0]))

            local_read_entry = ExperimentLogEntry(
                event="LOCAL.READ",
                experiment_id=experiment_id,
                algorithm=algorithm,
                start_time=read_local_model_labels_start_time,
                end_time=time.time(),
                id=model_id,
                worker_id="",
                num_chunks=num_chunks,
                security_level=security_level,
                workers=max_workers,
                description=f"Read model labels from: {model_labels_path}"
            )
            logger.info(local_read_entry.model_dump())

            put_model_labels_start_time = time.time()
            maybe_model_labels_chunks = Chunks.from_ndarray(
                ndarray=model_labels,
                group_id=model_labels_id,
                chunk_prefix=Some(model_labels_id),
                num_chunks=num_chunks
            )
            if maybe_model_labels_chunks.is_none:
                raise HTTPException(status_code=500, detail="Failed to convert into chunks the model labels")

            ptm_result = await RoryCommon.delete_and_put_chunks(
                client=STORAGE_CLIENT,
                bucket_id=BUCKET_ID,
                key=model_labels_id,
                chunks=maybe_model_labels_chunks.unwrap(),
                timeout=MICTLANX_TIMEOUT,
                tags={
                    "full_shape": str(model_labels.shape),
                    "full_dtype": str(model_labels.dtype)
                }
            )

            put_encrypted_ptm_entry = ExperimentLogEntry(
                event="PUT",
                experiment_id=experiment_id,
                algorithm=algorithm,
                start_time=put_model_labels_start_time,
                end_time=time.time(),
                id=model_id,
                worker_id="",
                num_chunks=num_chunks,
                security_level=security_level,
                workers=max_workers,
                description=f"Put model labels using id: {model_labels_id}"
            )
            logger.info(put_encrypted_ptm_entry.model_dump())

            r: int = model.shape[0]
            a: int = model.shape[1]
            encrypted_model_shape = "({},{})".format(r, a)
            n = a * r

            segment_encrypt_model_start_time = time.time()
            encrypted_model_chunks = RoryCommon.segment_and_encrypt_ckks_with_executor_v2(
                executor=executor,
                key=encrypted_model_id,
                plaintext_matrix=model,
                n=n,
                num_chunks=num_chunks,
                _round=_round,
                decimals=decimals,
                path=path,
                ctx_filename=ctx_filename,
                pubkey_filename=pubkey_filename,
                secretkey_filename=secretkey_filename,
                relinkey_filename=relinkey_filename,
            )

            segment_encrypt_entry = ExperimentLogEntry(
                event="SEGMENT.ENCRYPT",
                experiment_id=experiment_id,
                algorithm=algorithm,
                start_time=segment_encrypt_model_start_time,
                end_time=time.time(),
                id=model_id,
                worker_id="",
                num_chunks=num_chunks,
                workers=max_workers,
                security_level=security_level,
                description=f"Segment and encryption model: {encrypted_model_id}"
            )
            logger.info(segment_encrypt_entry.model_dump())

            put_chunked_start_time = time.time()
            put_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
                client=STORAGE_CLIENT,
                bucket_id=BUCKET_ID,
                key=encrypted_model_id,
                chunks=encrypted_model_chunks,
                timeout=MICTLANX_TIMEOUT,
                tags={
                    "full_shape": str(encrypted_model_shape),
                    "full_dtype": "float32"
                }
            )
            if put_chunks_generator_results.is_err:
                logger.error({
                    "error": "Failed to put encrypted model",
                    "experiment_id": experiment_id
                })
                raise HTTPException(status_code=500, detail="Failed to put encrypted model")

            put_encrypted_ptm_entry = ExperimentLogEntry(
                event="PUT",
                experiment_id=experiment_id,
                algorithm=algorithm,
                start_time=put_chunked_start_time,
                end_time=time.time(),
                id=model_id,
                worker_id="",
                num_chunks=num_chunks,
                workers=max_workers,
                security_level=security_level,
                description=f"Put encrypted model: {encrypted_model_id}"
            )
            logger.info(put_encrypted_ptm_entry.model_dump())

            endTime = time.time()
            response_time = endTime - local_start_time

            classification_completed_entry = ExperimentLogEntry(
                event="COMPLETED",
                experiment_id=experiment_id,
                algorithm=algorithm,
                start_time=local_start_time,
                end_time=time.time(),
                id=model_id,
                num_chunks=num_chunks,
                security_level=security_level,
                workers=max_workers,
                time=response_time,
                description="SKNN PQC TRAIN Completed Successfully"
            )
            logger.info(classification_completed_entry.model_dump())

            return {
                "response_time": str(response_time),
                "encrypted_model_shape": str(encrypted_model_shape),
                "encrypted_model_dtype": "float32",
                "algorithm": algorithm,
                "model_labels_shape": list(model_labels.shape)
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error({
            "msg": str(e)
        })
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/pqc/sknn/predict",
    response_model=PredictResponse,
    summary="PQC Secure KNN prediction using CKKS encryption",
    description="Interactive 2-round PQC protocol with CKKS decryption oracle for secure classification.",
)
async def sknn_pqc_predict(
    body: PqcSknnPredictRequest,
    logger=Depends(get_logger),
    settings: Settings = Depends(get_settings),
    STORAGE_CLIENT: AsyncClient = Depends(get_storage_client),
    executor: ProcessPoolExecutor = Depends(get_executor),
    managerResponse: RoryManager = Depends(get_manager),
):
    """
    This method orchestrates a privacy-preserving K-Nearest Neighbors prediction
    leveraging the CKKS homomorphic encryption scheme. It follows a
    Double-Blind interactive protocol where the Client acts as a decryption oracle,
    allowing the distributed Rory architecture to identify nearest neighbors and
    assign labels without exposing plaintext data to the cloud infrastructure.

    Note:
    **Post-Quantum Classification**: CKKS parameters and model identifiers are
    strictly handled via **HTTP Headers** to ensure protocol integrity.

    Attributes:
        Model-Id (str): ID of the pre-trained PQC model in CSS. Defaults to "model0".
        Records-Test-Id (str): Unique ID for the test records.
        Encrypted-Model-Shape (str): Dimensions of the CKKS model matrix.
        Encrypted-Model-Dtype (str): Data type of the encrypted model.
        Experiment-Id (str): Tracking ID for performance auditing.
        Records-Test-Extension (str): File extension (e.g., "npy").

    Returns:
        label_vector (list): The predicted class assignments.
        worker_id (str): ID of the worker node that processed the task.
        service_time_metrics: Timing data for Client, Manager, and Worker.
        algorithm (str): The SKNN_PQC_PREDICT constant.

    Raises:
        HTTPException (500): If mandatory headers (Shape/Dtype) are missing or if the
            process executor is not available.
        HTTPException (500): If failures occur during CKKS decryption, chunking,
            or multi-node orchestration.
    """
    try:
        local_start_time = time.time()
        BUCKET_ID: str = settings.mictlanx_bucket_id
        TESTING = settings.testing
        SOURCE_PATH = settings.source_path
        max_workers = settings.max_workers
        num_chunks = settings.num_chunks
        np_random = settings.np_random
        security_level = settings.liu_security_level
        WORKER_TIMEOUT = settings.worker_timeout
        algorithm = Constants.ClassificationAlgorithms.SKNN_PQC_PREDICT
        s = Session()
        model_id = body.model_id
        model_filename = body.model_filename
        records_test_id = body.records_test_id
        records_test_filename = body.records_test_filename
        records_test_extension = body.records_test_extension
        encrypted_records_test_id = "encrypted{}".format(records_test_id)
        extension = body.extension
        model_labels_id = "{}labels".format(model_id)
        _encrypted_model_shape = body.encrypted_model_shape
        _encrypted_model_dtype = body.encrypted_model_dtype
        experiment_id = body.experiment_id
        records_test_path = "{}/{}.{}".format(SOURCE_PATH, records_test_filename, extension)

        _round = settings.ckks_round
        decimals = settings.ckks_decimals
        path = settings.keys_path
        ctx_filename = settings.ctx_filename
        pubkey_filename = settings.pubkey_filename
        secretkey_filename = settings.secret_key_filename
        relinkey_filename = settings.relinkey_filename

        MICTLANX_TIMEOUT = settings.mictlanx_timeout
        MICTLANX_DELAY = settings.mictlanx_delay
        MICTLANX_BACKOFF_FACTOR = settings.mictlanx_backoff_factor
        MICTLANX_MAX_RETRIES = settings.mictlanx_max_retries

        # _______________________________________________________________________________
        ckks = Ckks.from_pyfhel(
            _round=_round,
            decimals=decimals,
            path=path,
            ctx_filename=ctx_filename,
            pubkey_filename=pubkey_filename,
            secretkey_filename=secretkey_filename,
            relinkey_filename=relinkey_filename,
        )
        # _______________________________________________________________________________
        dataowner = DataOwnerPQC(scheme=ckks)

        read_local_model_start_time = time.time()
        records_test_result = await RoryCommon.read_numpy_from(
            path=records_test_path,
            extension=records_test_extension
        )
        if records_test_result.is_err:
            raise HTTPException(status_code=500, detail="Failed to read local records")
        records_test = records_test_result.unwrap()

        local_read_entry = ExperimentLogEntry(
            event="LOCAL.READ",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=read_local_model_start_time,
            end_time=time.time(),
            id=model_id,
            worker_id="",
            num_chunks=num_chunks,
            security_level=security_level,
            workers=max_workers,
        )
        logger.info(local_read_entry.model_dump())

        r: int = records_test.shape[0]
        a: int = records_test.shape[1]
        n = a * r

        segment_encrypt_start_time = time.time()
        encrypted_records_chunks = RoryCommon.segment_and_encrypt_ckks_with_executor_v2(
            executor=executor,
            key=encrypted_records_test_id,
            plaintext_matrix=records_test,
            n=n,
            num_chunks=num_chunks,
            _round=_round,
            decimals=decimals,
            path=path,
            ctx_filename=ctx_filename,
            pubkey_filename=pubkey_filename,
            secretkey_filename=secretkey_filename,
            relinkey_filename=relinkey_filename,
        )

        segment_encrypt_entry = ExperimentLogEntry(
            event="SEGMENT.ENCRYPT",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=segment_encrypt_start_time,
            end_time=time.time(),
            id=model_id,
            worker_id="",
            num_chunks=num_chunks,
            workers=max_workers,
            security_level=security_level,
        )
        logger.info(segment_encrypt_entry.model_dump())

        put_chunks_start_time = time.time()
        encrypted_records_shape = records_test.shape
        put_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
            client=STORAGE_CLIENT,
            bucket_id=BUCKET_ID,
            key=encrypted_records_test_id,
            chunks=encrypted_records_chunks,
            timeout=MICTLANX_TIMEOUT,
            tags={
                "full_shape": str(encrypted_records_shape),
                "full_dtype": "float32"
            }
        )
        if put_chunks_generator_results.is_err:
            raise HTTPException(status_code=500, detail="Failed to put encrypted records")

        service_time_dataowner = time.time() - local_start_time

        put_encrypted_ptm_entry = ExperimentLogEntry(
            event="PUT",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=put_chunks_start_time,
            end_time=time.time(),
            id=model_id,
            worker_id="",
            num_chunks=num_chunks,
            workers=max_workers,
            security_level=security_level,
        )
        logger.info(put_encrypted_ptm_entry.model_dump())

        get_worker_start_time = time.time()
        get_worker_result = managerResponse.getWorker(
            headers={
                "Algorithm": algorithm,
                "Start-Request-Time": str(local_start_time),
                "Start-Get-Worker-Time": str(get_worker_start_time),
                "Matrix-Id": model_id
            }
        )

        if get_worker_result.is_err:
            error = get_worker_result.unwrap_err()
            logger.error(str(error))
            raise HTTPException(status_code=500, detail=str(error))
        (_worker_id, port) = get_worker_result.unwrap()

        get_worker_end_time = time.time()
        get_worker_service_time = get_worker_end_time - get_worker_start_time
        worker_id = "localhost" if TESTING else _worker_id

        get_worker_entry = ExperimentLogEntry(
            event="GET.WORKER",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=get_worker_start_time,
            end_time=time.time(),
            id=model_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            workers=max_workers,
            security_level=security_level,
        )
        logger.info(get_worker_entry.model_dump())

        worker_start_time = time.time()
        worker = RoryWorker(
            workerId=worker_id,
            port=port,
            session=s,
            algorithm=algorithm,
        )

        inner_interaction_arrival_time = time.time()
        encrypted_records_dtype = "float32"
        run1_headers = {
            "Step-Index": "1",
            "Records-Test-Id": records_test_id,
            "Model-Id": model_id,
            "Encrypted-Model-Shape": _encrypted_model_shape,
            "Encrypted-Model-Dtype": _encrypted_model_dtype,
            "Encrypted-Records-Shape": str(encrypted_records_shape),
            "Encrypted-Records-Dtype": str(encrypted_records_dtype),
            "Num-Chunks": str(num_chunks),
        }

        worker_run1_response = worker.run(
            headers=run1_headers,
            timeout=WORKER_TIMEOUT
        )
        worker_run1_response.raise_for_status()

        jsonWorkerResponse = worker_run1_response.json()
        endTime = time.time()
        distances_id = jsonWorkerResponse["distances_id"]
        distances_shape = jsonWorkerResponse["distances_shape"]
        distances_dtype = jsonWorkerResponse["distances_dtype"]
        worker_service_time = jsonWorkerResponse["service_time"]

        run1_worker_entry = ExperimentLogEntry(
            event="RUN1",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=inner_interaction_arrival_time,
            end_time=time.time(),
            id=model_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            workers=max_workers,
            security_level=security_level,
        )
        logger.info(run1_worker_entry.model_dump())

        get_all_distances_start_time = time.time()
        all_distances = await RoryCommon.get_pyctxt_matrix(
            client=STORAGE_CLIENT,
            bucket_id=BUCKET_ID,
            key=distances_id,
            ckks=ckks,
            backoff_factor=MICTLANX_BACKOFF_FACTOR,
            delay=MICTLANX_DELAY,
            force=True,
            max_retries=MICTLANX_MAX_RETRIES,
            timeout=MICTLANX_TIMEOUT
        )

        get_encrypted_sm_entry = ExperimentLogEntry(
            event="GET",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=get_all_distances_start_time,
            end_time=time.time(),
            id=model_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            workers=max_workers,
            security_level=security_level,
        )
        logger.info(get_encrypted_sm_entry.model_dump())

        decrypt_matrix_start_time = time.time()
        matrix_distances_plain = ckks.decrypt_matrix_list(
            xs=all_distances,
            take=1
        )
        _x = np.array(matrix_distances_plain).reshape(all_distances.shape)

        min_distances_index = np.argmin(matrix_distances_plain, axis=1).reshape(1, -1)
        min_distances_index_id = "distancesindex{}".format(records_test_id)
        decrypt_entry = ExperimentLogEntry(
            event="DECRYPT",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=decrypt_matrix_start_time,
            end_time=time.time(),
            id=model_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            workers=max_workers,
            security_level=security_level,
        )
        logger.info(decrypt_entry.model_dump())

        t1 = time.time()
        maybe_min_distances_chunks = Chunks.from_ndarray(
            ndarray=min_distances_index,
            group_id=min_distances_index_id,
            chunk_prefix=Some(min_distances_index_id),
            num_chunks=num_chunks,
        )

        if maybe_min_distances_chunks.is_none:
            raise HTTPException(status_code=500, detail="something went wrong creating the chunks")

        min_distances_put_result = await RoryCommon.delete_and_put_chunks(
            client=STORAGE_CLIENT,
            bucket_id=BUCKET_ID,
            key=min_distances_index_id,
            chunks=maybe_min_distances_chunks.unwrap(),
            timeout=MICTLANX_TIMEOUT,
            tags={
                "full_shape": str(min_distances_index.shape),
                "full_dtype": str(min_distances_index.dtype)
            }
        )
        if min_distances_put_result.is_err:
            raise HTTPException(status_code=500, detail="Failed to put min distances")

        put_sm_entry = ExperimentLogEntry(
            event="PUT",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=t1,
            end_time=time.time(),
            id=model_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            workers=max_workers,
            security_level=security_level,
        )
        logger.info(put_sm_entry.model_dump())

        run2_headers = {
            "Step-Index": "2",
            "Records-Test-Id": records_test_id,
            "Model-Id": model_id,
            "Encrypted-Model-Shape": _encrypted_model_shape,
            "Encrypted-Model-Dtype": _encrypted_model_dtype,
            "Encrypted-Records-Shape": str(encrypted_records_shape),
            "Encrypted-Records-Dtype": str(encrypted_records_dtype),
            "Num-Chunks": str(num_chunks),
            "Min_Distances_Index_Id": min_distances_index_id
        }

        worker_run2_response = worker.run(
            headers=run2_headers,
            timeout=WORKER_TIMEOUT
        )
        worker_run2_response.raise_for_status()
        jsonWorkerResponse2 = worker_run2_response.json()
        service_time_worker = worker_run2_response.headers.get("Service-Time", 0)
        worker_end_time = time.time()
        worker_response_time = worker_end_time - worker_start_time

        run2_worker_entry = ExperimentLogEntry(
            event="RUN2",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=inner_interaction_arrival_time,
            end_time=time.time(),
            id=model_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            workers=max_workers,
            security_level=security_level,
        )
        logger.info(run2_worker_entry.model_dump())

        response_time = endTime - local_start_time

        classification_completed_entry = ExperimentLogEntry(
            event="COMPLETED",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=local_start_time,
            end_time=time.time(),
            id=model_id,
            num_chunks=num_chunks,
            security_level=security_level,
            workers=max_workers,
            time=response_time
        )
        logger.info(classification_completed_entry.model_dump())

        label_vector = jsonWorkerResponse2["label_vector"]
        return {
            "label_vector": label_vector,
            "worker_id": worker_id,
            "service_time_manager": get_worker_service_time,
            "service_time_worker": worker_response_time,
            "service_time_dataowner": service_time_dataowner,
            "service_time_predict": response_time,
            "algorithm": algorithm,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error({
            "msg": str(e)
        })
        raise HTTPException(status_code=500, detail=str(e))
