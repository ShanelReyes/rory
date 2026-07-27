import time, json
import numpy as np
import numpy.typing as npt
import copy
from typing import List, Tuple
from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import JSONResponse
from rory.core.clustering.kmeans import kmeans as kMeans
from rory.core.clustering.secure.conventional.dbsnnc import Dbsnnc
from rory.core.clustering.nnc import Nnc
from rory.core.utils.utils import Utils
from rory.core.utils.constants import Constants
from rory.core.clustering.secure.conventional.skmeans import SKMeans
from rory.core.clustering.secure.conventional.dbskmeans import DBSKMeans
from rory.core.clustering.secure.pqc.skmeans import Skmeans as SkmeansPQC
from rory.core.clustering.secure.pqc.dbskmeans import DBSKMeans as DbskmeansPQC
from rory.core.security.cryptosystem.pqc.ckks import Ckks
from rorycommon import StorageBuilder, StorageParams, Scheme, CkksParams, LiuParams
from mictlanx import AsyncClient
from option import Result, Some
from mictlanx.utils.segmentation import Chunks
from option import Option, Some, NONE
from rorycommon import Common as RoryCommon
from Pyfhel import PyCtxt, Pyfhel
from models.experiment import ExperimentLogEntry
from dependencies import get_logger, get_storage_client, get_ckks, get_settings
from models.requests.clustering import (
    KmeansWorkerRequest,
    SkmeansWorkerRequest,
    DbskmeansWorkerRequest,
    DbsnncWorkerRequest,
    NncWorkerRequest,
    PqcSkmeansWorkerRequest,
    PqcDbskmeansWorkerRequest,
)
from models.responses.clustering import (
    HealthCheckResponse,
    WorkerRun1Response,
    WorkerDbsnncResponse,
    WorkerNncResponse,
)

router = APIRouter(prefix="/clustering", tags=["Clustering"])


@router.get("/test")
@router.post("/test")
def test():
    return JSONResponse(
        content={"component_type": "worker"},
        status_code=200,
        headers={"Component-Type": "worker"},
    )

async def skmeans_1(body: SkmeansWorkerRequest, logger, storage_client, settings) -> Response:
    arrival_time = time.time()
    worker_id = settings.node_id
    BUCKET_ID: str = settings.mictlanx_bucket_id
    status = int(body.clustering_status)
    is_start_status = status == Constants.ClusteringStatus.START
    k = int(body.k)
    m = int(body.m)
    algorithm = Constants.ClusteringAlgorithms.SKMEANS
    plaintext_matrix_id = body.plaintext_matrix_id
    encrypted_matrix_id = body.encrypted_matrix_id
    udm_id = "{}udm".format(plaintext_matrix_id)
    _encrypted_matrix_shape = body.encrypted_matrix_shape
    _encrypted_matrix_dtype = body.encrypted_matrix_dtype
    experiment_id = body.experiment_id
    MICTLANX_TIMEOUT = settings.mictlanx_timeout

    if _encrypted_matrix_dtype is None:
        raise HTTPException(status_code=500, detail="Encrypted-Matrix-Dtype")
    if _encrypted_matrix_shape is None:
        raise HTTPException(status_code=500, detail="Encrypted-Matrix-Shape header is required")

    encrypted_matrix_shape: tuple = eval(_encrypted_matrix_shape)

    encrypted_shift_matrix_id = "{}encryptedshiftmatrix".format(plaintext_matrix_id)
    centroids_id = "{}centroids".format(plaintext_matrix_id)
    num_chunks_str = body.num_chunks
    skmeans = SKMeans()
    responseHeaders = {}

    if num_chunks_str is None:
        logger.error({"msg": "Num-Chunks header is required"})
        raise HTTPException(status_code=503, detail="Num-Chunks header is required")

    num_chunks = int(num_chunks_str)

    storage_backend = (
        StorageBuilder(storage_client=storage_client, scheme=None)
        .with_storage_params(StorageParams(num_chunks=num_chunks, timeout=MICTLANX_TIMEOUT))
        .build()
    )
    try:
        responseHeaders["Start-Time"] = str(arrival_time)
        encryptedMatrix_result = await storage_backend.get(
            bucket_id=BUCKET_ID,
            ball_id=encrypted_matrix_id,
            segment=True,
            encrypt=False,
            scheme=None
        )
        if encryptedMatrix_result.is_err:
            logger.error(f"Failed to get encrypted matrix: {encryptedMatrix_result.unwrap_err()}")
            raise HTTPException(status_code=500, detail="Failed to get encrypted matrix")
        encrypted_matrix_get_result = encryptedMatrix_result.unwrap()
        encrypted_matrix: npt.NDArray = encrypted_matrix_get_result.raw_value

        logger.debug({
            "event": "GET",
            "experiment_id": experiment_id,
            "bucket_id": BUCKET_ID,
            "ball_id": encrypted_matrix_id,
            "matrix_id": encrypted_matrix_id,
            "shape": str(encrypted_matrix.shape if hasattr(encrypted_matrix, 'shape') else (1,)),
            "dtype": "float32",
            "read_time": encrypted_matrix_get_result.read_time,
        })

        udm_result = await storage_backend.get(
            bucket_id=BUCKET_ID,
            ball_id=udm_id,
            segment=True,
            encrypt=False,
            scheme=None
        )

        if udm_result.is_err:
            logger.error(f"Failed to get udm matrix: {udm_result.unwrap_err()}")
            raise HTTPException(status_code=500, detail="Failed to get encrypted matrix")
        udm_get_result = udm_result.unwrap()
        udm: npt.NDArray = udm_get_result.raw_value
        udm_shape = udm.shape if hasattr(udm, 'shape') else udm.shape

        logger.debug({
            "event": "GET",
            "experiment_id": experiment_id,
            "bucket_id": BUCKET_ID,
            "ball_id": udm_id,
            "matrix_id": udm_id,
            "shape": udm_shape,
            "dtype": "float32",
            "read_time": udm_get_result.read_time,
        })

        responseHeaders["Udm-Matrix-Dtype"] = str("float32")
        responseHeaders["Udm-Matrix-Shape"] = str(udm_shape)

        if is_start_status:
            __Cent_i = None
        else:
            centroids_result = await storage_backend.get(
                bucket_id=BUCKET_ID,
                ball_id=centroids_id,
                segment=True,
                encrypt=False,
                scheme=None
            )

            if centroids_result.is_err:
                logger.error(f"Failed to get cent j matrix: {centroids_result.unwrap_err()}")
                raise HTTPException(status_code=500, detail="Failed to get cent j matrix")
            centroids_get_result = centroids_result.unwrap()
            centroids = centroids_get_result.raw_value

            centroids_shape = centroids.shape if hasattr(centroids, 'shape') else (centroids.shape)
            __Cent_i = copy.deepcopy(centroids)

            logger.debug({
                "event": "GET",
                "experiment_id": experiment_id,
                "bucket_id": BUCKET_ID,
                "ball_id": centroids_id,
                "matrix_id": centroids_id,
                "shape": centroids_shape,
                "dtype": "float32",
                "read_time": centroids_get_result.read_time,
            })

            status = Constants.ClusteringStatus.WORK_IN_PROGRESS
        run1_start_time = time.time()

        run1_result: Result[
            Tuple[npt.NDArray, List[List[float]], List[List[float]], List[int]],
            Exception
        ] = skmeans.execute_encrypted_phase(
            status=status,
            k=k,
            encrypted_matrix=encrypted_matrix,
            udm=udm,
            centroids=__Cent_i,
            num_attributes=encrypted_matrix.shape[1],
            m=m
        )

        if run1_result.is_err:
            error = run1_result.unwrap_err()
            logger.error({
                "event": "SKMEANS.RUN1.FAILED",
                "raw_error": str(error)
            })
            raise HTTPException(status_code=500, detail=str(error))
        S1, _, new_centroids, label_vector = run1_result.unwrap()

        logger.debug({
            "event": "RUN1",
            "experiment_id": experiment_id,
            "algorithm": algorithm,
            "start_time": run1_start_time,
            "end_time": time.time(),
            "id": plaintext_matrix_id,
            "worker_id": worker_id,
            "num_chunks": num_chunks,
            "k": k,
            "m": m
        })

        centroids_result = await storage_backend.put(
            bucket_id=BUCKET_ID,
            ball_id=centroids_id,
            data=new_centroids,
            scheme=None,
            segment=True,
            encrypt=False,
            delete=True
        )

        if centroids_result.is_err:
            logger.error("Failed to process centroids: {}".format(centroids_result.unwrap_err()))
            raise HTTPException(status_code=500, detail="Failed to process centroids")
        centroids = centroids_result.unwrap()

        logger.debug({
            "event": "PUT",
            "experiment_id": experiment_id,
            "bucket_id": BUCKET_ID,
            "ball_id": centroids_id,
            "matrix_id": centroids_id,
            "shape": str(centroids.shape),
            "dtype": str(centroids.dtype),
            "read_time": centroids.read_time,
            "segment_time": centroids.segment_time,
            "encrypt_time": getattr(centroids, "encrypt_time", 0.0),
            "upload_time": centroids.upload_time,
        })

        shift1_result = await storage_backend.put(
            bucket_id=BUCKET_ID,
            ball_id=encrypted_shift_matrix_id,
            data=S1,
            scheme=None,
            segment=True,
            encrypt=False,
            delete=True
        )

        if shift1_result.is_err:
            logger.error("Failed to process shift1: {}".format(shift1_result.unwrap_err()))
            raise HTTPException(status_code=500, detail="Failed to process shift1")
        shift1 = shift1_result.unwrap()

        logger.debug({
            "event": "PUT",
            "experiment_id": experiment_id,
            "bucket_id": BUCKET_ID,
            "ball_id": encrypted_shift_matrix_id,
            "matrix_id": encrypted_shift_matrix_id,
            "shape": str(shift1.shape),
            "dtype": str(shift1.dtype),
            "read_time": shift1.read_time,
            "segment_time": shift1.segment_time,
            "encrypt_time": getattr(shift1, "encrypt_time", 0.0),
            "upload_time": shift1.upload_time,
        })

        end_time = time.time()
        service_time = end_time - arrival_time
        n_iterations = int(body.iterations) + 1

        clustering_entry = ExperimentLogEntry(
            event="COMPLETED",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=arrival_time,
            end_time=time.time(),
            id=encrypted_matrix_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            k=k,
            workers=0,
            security_level=0,
            m=m,
            iterations=n_iterations,
            time=service_time
        )
        logger.info(clustering_entry.model_dump())

        return JSONResponse(
            content={
                "label_vector": label_vector,
                "service_time": service_time,
                "n_iterations": n_iterations,
                "encrypted_shift_matrix_id": encrypted_shift_matrix_id
            },
            status_code=200,
            headers=responseHeaders
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error({
            "msg": str(e),
            "at": "worker_skmeans_1"
        })
        raise HTTPException(status_code=500, detail=str(e))

async def skmeans_2(body: SkmeansWorkerRequest, logger, storage_client, settings):
    local_start_time = time.time()
    worker_id = settings.node_id
    BUCKET_ID: str = settings.mictlanx_bucket_id
    algorithm = Constants.ClusteringAlgorithms.SKMEANS
    plaintext_matrix_id = body.plaintext_matrix_id
    encrypted_matrix_id = body.encrypted_matrix_id
    shift_matrix_id = body.shift_matrix_id if body.shift_matrix_id else "{}shiftmatrix".format(plaintext_matrix_id)
    k = int(body.k)
    m = int(body.m)
    iterations = int(body.iterations)
    experiment_id = body.experiment_id

    if not encrypted_matrix_id or not plaintext_matrix_id:
        raise HTTPException(status_code=500, detail="Either Encrypted-Matrix-Id or Plain-Matrix-Id is missing")
    num_chunks_str = body.num_chunks
    udm_id = "{}udm".format(plaintext_matrix_id)
    response_headers = {}
    is_zero = bool(int(body.is_zero)) if body.is_zero else False
    MICTLANX_TIMEOUT = settings.mictlanx_timeout

    num_chunks = int(num_chunks_str) if num_chunks_str else -1

    storage_backend = (
        StorageBuilder(storage_client=storage_client, scheme=None)
        .with_storage_params(StorageParams(num_chunks=num_chunks, timeout=MICTLANX_TIMEOUT))
        .build()
    )

    try:
        if is_zero:
            response_headers["Clustering-Status"] = Constants.ClusteringStatus.COMPLETED
            end_time = time.time()
            service_time = end_time - local_start_time
            response_headers["Total-Service-Time"] = str(service_time)

            clustering_completed_entry = ExperimentLogEntry(
                event="COMPLETED",
                experiment_id=experiment_id,
                algorithm=algorithm,
                start_time=local_start_time,
                end_time=time.time(),
                id=plaintext_matrix_id,
                worker_id=worker_id,
                num_chunks=num_chunks,
                k=k,
                m=m,
                iterations=iterations
            )
            logger.info(clustering_completed_entry.model_dump())

            return Response(
                status_code=204,
                headers=response_headers
            )

        udm_result = await storage_backend.get(
            bucket_id=BUCKET_ID,
            ball_id=udm_id,
            segment=True,
            encrypt=False,
            scheme=None
        )
        if udm_result.is_err:
            logger.error(f"Failed to get udm: {udm_result.unwrap_err()}")
            raise HTTPException(status_code=500, detail="Failed to get udm")
        udm_get_result = udm_result.unwrap()
        udm: npt.NDArray = udm_get_result.raw_value

        logger.debug({
            "event": "GET",
            "experiment_id": experiment_id,
            "bucket_id": BUCKET_ID,
            "ball_id": udm_id,
            "matrix_id": udm_id,
            "shape": str(udm.shape if hasattr(udm, 'shape') else (1,)),
            "dtype": "float32",
            "read_time": udm_get_result.read_time,
        })

        shift_matrix_result = await storage_backend.get(
            bucket_id=BUCKET_ID,
            ball_id=shift_matrix_id,
            segment=True,
            encrypt=False,
            scheme=None
        )
        if shift_matrix_result.is_err:
            logger.error(f"Failed to get shift matrix: {shift_matrix_result.unwrap_err()}")
            raise HTTPException(status_code=500, detail="Failed to get shift matrix")
        shift_matrix_get_result = shift_matrix_result.unwrap()
        shift_matrix: npt.NDArray = shift_matrix_get_result.raw_value

        logger.debug({
            "event": "GET",
            "experiment_id": experiment_id,
            "bucket_id": BUCKET_ID,
            "ball_id": shift_matrix_id,
            "matrix_id": shift_matrix_id,
            "shape": str(shift_matrix.shape if hasattr(shift_matrix, 'shape') else (1,)),
            "dtype": "float32",
            "read_time": shift_matrix_get_result.read_time,
        })

        run2_start_time = time.time()
        skmeans = SKMeans()
        status = Constants.ClusteringStatus.WORK_IN_PROGRESS
        response_headers["Clustering-Status"] = status
        encrypted_matrix_shape = eval(body.encrypted_matrix_shape)
        _udm = skmeans.execute_plaintext_phase(
            k=k,
            udm=udm,
            num_attributes=int(encrypted_matrix_shape[1]),
            shift_matrix=shift_matrix,
        )

        logger.debug({
            "event": "RUN2",
            "experiment_id": experiment_id,
            "algorithm": algorithm,
            "start_time": run2_start_time,
            "end_time": time.time(),
            "id": plaintext_matrix_id,
            "worker_id": worker_id,
            "num_chunks": num_chunks,
            "k": k,
            "m": m,
            "iterations": iterations
        })

        udm_put_result = await storage_backend.put(
            bucket_id=BUCKET_ID,
            ball_id=udm_id,
            data=_udm,
            segment=True,
            encrypt=False,
            scheme=None,
            delete=True
        )
        if udm_put_result.is_err:
            logger.error("Failed to process udm: {}".format(udm_put_result.unwrap_err()))
            raise HTTPException(status_code=500, detail="Failed to process udm")
        udm_response = udm_put_result.unwrap()

        logger.debug({
            "event": "PUT",
            "experiment_id": experiment_id,
            "bucket_id": BUCKET_ID,
            "ball_id": udm_id,
            "matrix_id": udm_id,
            "shape": str(udm_response.shape),
            "dtype": str(udm_response.dtype),
            "read_time": udm_response.read_time,
            "segment_time": udm_response.segment_time,
            "encrypt_time": getattr(udm_response, "encrypt_time", 0.0),
            "upload_time": udm_response.upload_time,
        })

        end_time = time.time()
        service_time = end_time - local_start_time

        logger.debug({
            "event": "UNCOMPLETED",
            "experiment_id": experiment_id,
            "algorithm": algorithm,
            "start_time": local_start_time,
            "end_time": end_time,
            "id": plaintext_matrix_id,
            "worker_id": worker_id,
            "num_chunks": num_chunks,
            "k": k,
            "m": m,
            "iterations": iterations
        })

        response_headers["End-Time"] = str(end_time)
        response_headers["Service-Time"] = str(service_time)

        return Response(
            status_code=204,
            headers=response_headers
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("SKMEANS_2_ERROR: " + encrypted_matrix_id + " " + str(e))
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/skmeans")
async def skmeans(
    body: SkmeansWorkerRequest,
    logger=Depends(get_logger),
    storage_client=Depends(get_storage_client),
    settings=Depends(get_settings),
):
    if body.step_index == 1:
        return await skmeans_1(body, logger, storage_client, settings)
    elif body.step_index == 2:
        return await skmeans_2(body, logger, storage_client, settings)
    else:
        raise HTTPException(status_code=400, detail="Failed invalid step_index")


@router.post("/kmeans")
async def kmeans(
    body: KmeansWorkerRequest,
    logger=Depends(get_logger),
    storage_client=Depends(get_storage_client),
    settings=Depends(get_settings),
):
    local_start_time = time.time()
    experiment_id = body.experiment_id
    algorithm = Constants.ClusteringAlgorithms.KMEANS
    worker_id = settings.node_id
    BUCKET_ID: str = settings.mictlanx_bucket_id
    plaintext_matrix_id = body.plaintext_matrix_id
    k = body.k
    response_headers = {}
    MICTLANX_TIMEOUT = settings.mictlanx_timeout
    MICTLANX_DELAY = settings.mictlanx_delay
    MICTLANX_BACKOFF_FACTOR = settings.mictlanx_backoff_factor
    MICTLANX_MAX_RETRIES = settings.mictlanx_max_retries

    try:
        t1 = time.time()
        plaintext_matrix = await RoryCommon.get_matrix_or_error(
            client=storage_client,
            key=plaintext_matrix_id,
            bucket_id=BUCKET_ID,
            delay=MICTLANX_DELAY,
            max_retries=MICTLANX_MAX_RETRIES,
            timeout=MICTLANX_TIMEOUT,
            backoff_factor=MICTLANX_BACKOFF_FACTOR
        )

        get_ptm_entry = ExperimentLogEntry(
            event="GET",
            experiment_id=experiment_id,
            start_time=t1,
            end_time=time.time(),
            algorithm=algorithm,
            id=plaintext_matrix_id,
            k=k,
            iterations=0,
            num_chunks=0,
            worker_id=worker_id,
            workers=0
        )
        logger.info(get_ptm_entry.model_dump())

        t1 = time.time()
        result = kMeans(
            k=k,
            plaintext_matrix=plaintext_matrix
        )

        clustering_entry = ExperimentLogEntry(
            event="COMPLETED",
            experiment_id=experiment_id,
            start_time=local_start_time,
            end_time=time.time(),
            algorithm=algorithm,
            id=plaintext_matrix_id,
            k=k,
            iterations=result.n_iterations,
            num_chunks=0,
            worker_id=worker_id,
            workers=0
        )
        logger.info(clustering_entry.model_dump())

        response_headers["Service-Time"] = str(clustering_entry.time)
        response_headers["Iterations"] = int(result.n_iterations)

        return JSONResponse(
            content={
                "label_vector": result.label_vector.tolist(),
                "iterations": result.n_iterations,
                "service_time": clustering_entry.time
            },
            status_code=200,
            headers={**response_headers}
        )
    except Exception as e:
        logger.error({
            "msg": str(e)
        })
        raise HTTPException(status_code=500, detail=str(e))

async def dbskmeans_1(body: DbskmeansWorkerRequest, logger, storage_client, settings) -> Response:
    arrival_time = time.time()
    worker_id = settings.node_id
    BUCKET_ID: str = settings.mictlanx_bucket_id
    status = int(body.clustering_status)
    is_start_status = status == Constants.ClusteringStatus.START
    k = int(body.k)
    m = int(body.m)
    algorithm = Constants.ClusteringAlgorithms.DBSKMEANS
    plaintext_matrix_id = body.plaintext_matrix_id
    encrypted_matrix_id = body.encrypted_matrix_id
    _encrypted_matrix_shape = body.encrypted_matrix_shape
    _encrypted_matrix_dtype = body.encrypted_matrix_dtype
    _encrypted_udm_shape = body.encrypted_udm_shape
    _encrypted_udm_dtype = body.encrypted_udm_dtype
    experiment_id = body.experiment_id
    MICTLANX_TIMEOUT = settings.mictlanx_timeout

    if _encrypted_matrix_dtype is None:
        raise HTTPException(status_code=400, detail="Encrypted-Matrix-Dtype")
    if _encrypted_matrix_shape is None:
        raise HTTPException(status_code=400, detail="Encrypted-Matrix-Shape header is required")
    if _encrypted_udm_dtype is None:
        raise HTTPException(status_code=400, detail="Encrypted-UDM-Dtype")
    if _encrypted_udm_shape is None:
        raise HTTPException(status_code=400, detail="Encrypted-UDM-Shape header is required")

    num_chunks_str = body.num_chunks
    if num_chunks_str is None:
        raise HTTPException(status_code=503, detail="Num-Chunks header is required")
    num_chunks = int(num_chunks_str)
    encrypted_matrix_shape: tuple = eval(_encrypted_matrix_shape)
    encrypted_udm_shape: tuple = eval(_encrypted_udm_shape)

    encrypted_udm_id = "{}encryptedudm".format(plaintext_matrix_id)
    centroids_id = "{}centroids".format(plaintext_matrix_id)
    encrypted_shift_matrix_id = "{}encryptedshiftmatrix".format(plaintext_matrix_id)
    dbskmeans = DBSKMeans()
    response_headers = {}

    storage_backend = (
        StorageBuilder(storage_client=storage_client, scheme=None)
        .with_storage_params(StorageParams(num_chunks=num_chunks, timeout=MICTLANX_TIMEOUT))
        .build()
    )

    try:
        response_headers["Start-Time"] = str(arrival_time)
        encryptedMatrix_result = await storage_backend.get(
            bucket_id=BUCKET_ID,
            ball_id=encrypted_matrix_id,
            segment=True,
            encrypt=False,
            scheme=None
        )
        if encryptedMatrix_result.is_err:
            logger.error(f"Failed to get encrypted matrix: {encryptedMatrix_result.unwrap_err()}")
            raise HTTPException(status_code=500, detail="Failed to get encrypted matrix")
        encrypted_matrix_get_result = encryptedMatrix_result.unwrap()
        encryptedMatrix: npt.NDArray = encrypted_matrix_get_result.raw_value

        logger.debug({
            "event": "GET",
            "experiment_id": experiment_id,
            "bucket_id": BUCKET_ID,
            "ball_id": encrypted_matrix_id,
            "matrix_id": encrypted_matrix_id,
            "shape": str(encryptedMatrix.shape if hasattr(encryptedMatrix, 'shape') else (1,)),
            "dtype": "float32",
            "read_time": encrypted_matrix_get_result.read_time,
        })

        encrypted_udm_result = await storage_backend.get(
            bucket_id=BUCKET_ID,
            ball_id=encrypted_udm_id,
            segment=True,
            encrypt=False,
            scheme=None
        )
        if encrypted_udm_result.is_err:
            logger.error(f"Failed to get encrypted udm: {encrypted_udm_result.unwrap_err()}")
            raise HTTPException(status_code=500, detail="Failed to get encrypted udm")
        encrypted_udm_get_result = encrypted_udm_result.unwrap()
        encrypted_udm: npt.NDArray = encrypted_udm_get_result.raw_value

        udm_shape = encrypted_udm.shape if hasattr(encrypted_udm, 'shape') else encrypted_udm.shape
        response_headers["Udm-Matrix-Dtype"] = "float32"
        response_headers["Udm-Matrix-Shape"] = str(udm_shape)

        logger.debug({
            "event": "GET",
            "experiment_id": experiment_id,
            "bucket_id": BUCKET_ID,
            "ball_id": encrypted_udm_id,
            "matrix_id": encrypted_udm_id,
            "shape": str(udm_shape),
            "dtype": "float32",
            "read_time": encrypted_udm_get_result.read_time,
        })

        if is_start_status:
            __Cent_i = None
        else:
            centroids_result = await storage_backend.get(
                bucket_id=BUCKET_ID,
                ball_id=centroids_id,
                segment=True,
                encrypt=False,
                scheme=None
            )
            if centroids_result.is_err:
                logger.error(f"Failed to get centroids: {centroids_result.unwrap_err()}")
                raise HTTPException(status_code=500, detail="Failed to get centroids")
            centroids_get_result = centroids_result.unwrap()
            centroids = centroids_get_result.raw_value
            __Cent_i = copy.deepcopy(centroids)

            centroids_shape = centroids.shape if hasattr(centroids, 'shape') else (centroids.shape)
            logger.debug({
                "event": "GET",
                "experiment_id": experiment_id,
                "bucket_id": BUCKET_ID,
                "ball_id": centroids_id,
                "matrix_id": centroids_id,
                "shape": str(centroids_shape),
                "dtype": "float32",
                "read_time": centroids_get_result.read_time,
            })

            status = Constants.ClusteringStatus.WORK_IN_PROGRESS

        run1_start_time = time.time()
        run1_result: Result[
            Tuple[npt.NDArray, List[List[float]], List[List[float]], List[int]],
            Exception
        ] = dbskmeans.execute_encrypted_phase(
            status=status,
            k=k,
            encrypted_matrix=encryptedMatrix,
            udm=encrypted_udm,
            num_attributes=encryptedMatrix.shape[1],
            centroids=__Cent_i,
            m=m
        )

        if run1_result.is_err:
            error = run1_result.unwrap_err()
            logger.error({
                "event": "DBSKMEANS.RUN1.FAILED",
                "raw_error": str(error)
            })
            raise HTTPException(status_code=500, detail=str(error))
        S1, _, new_centroids, label_vector = run1_result.unwrap()

        logger.debug({
            "event": "RUN1",
            "experiment_id": experiment_id,
            "algorithm": algorithm,
            "start_time": run1_start_time,
            "end_time": time.time(),
            "id": plaintext_matrix_id,
            "worker_id": worker_id,
            "num_chunks": num_chunks,
            "k": k,
            "m": m
        })

        centroids_put_result = await storage_backend.put(
            bucket_id=BUCKET_ID,
            ball_id=centroids_id,
            data=new_centroids,
            scheme=None,
            segment=True,
            encrypt=False,
            delete=True
        )
        if centroids_put_result.is_err:
            logger.error("Failed to process centroids: {}".format(centroids_put_result.unwrap_err()))
            raise HTTPException(status_code=500, detail="Failed to process centroids")
        centroids_response = centroids_put_result.unwrap()

        logger.debug({
            "event": "PUT",
            "experiment_id": experiment_id,
            "bucket_id": BUCKET_ID,
            "ball_id": centroids_id,
            "matrix_id": centroids_id,
            "shape": str(centroids_response.shape),
            "dtype": str(centroids_response.dtype),
            "read_time": centroids_response.read_time,
            "segment_time": centroids_response.segment_time,
            "encrypt_time": getattr(centroids_response, "encrypt_time", 0.0),
            "upload_time": centroids_response.upload_time,
        })

        shift1_put_result = await storage_backend.put(
            bucket_id=BUCKET_ID,
            ball_id=encrypted_shift_matrix_id,
            data=S1,
            scheme=None,
            segment=True,
            encrypt=False,
            delete=True
        )
        if shift1_put_result.is_err:
            logger.error("Failed to process shift1: {}".format(shift1_put_result.unwrap_err()))
            raise HTTPException(status_code=500, detail="Failed to process shift1")
        shift1_response = shift1_put_result.unwrap()

        logger.debug({
            "event": "PUT",
            "experiment_id": experiment_id,
            "bucket_id": BUCKET_ID,
            "ball_id": encrypted_shift_matrix_id,
            "matrix_id": encrypted_shift_matrix_id,
            "shape": str(shift1_response.shape),
            "dtype": str(shift1_response.dtype),
            "read_time": shift1_response.read_time,
            "segment_time": shift1_response.segment_time,
            "encrypt_time": getattr(shift1_response, "encrypt_time", 0.0),
            "upload_time": shift1_response.upload_time,
        })

        end_time = time.time()
        service_time = end_time - arrival_time
        n_iterations = int(body.iterations) + 1

        clustering_entry = ExperimentLogEntry(
            event="COMPLETED",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=arrival_time,
            end_time=time.time(),
            id=encrypted_matrix_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            k=k,
            workers=0,
            security_level=0,
            m=m,
            iterations=n_iterations,
            time=service_time
        )
        logger.info(clustering_entry.model_dump())

        return JSONResponse(
            content={
                "label_vector": label_vector,
                "service_time": service_time,
                "n_iterations": n_iterations,
                "encrypted_shift_matrix_id": encrypted_shift_matrix_id
            },
            status_code=200,
            headers=response_headers
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error({
            "msg": str(e),
            "at": "worker_dbskmeans_1"
        })
        raise HTTPException(status_code=500, detail=str(e))


async def dbskmeans_2(body: DbskmeansWorkerRequest, logger, storage_client, settings):
    local_start_time = time.time()
    worker_id = settings.node_id
    BUCKET_ID: str = settings.mictlanx_bucket_id
    algorithm = Constants.ClusteringAlgorithms.DBSKMEANS
    plaintext_matrix_id = body.plaintext_matrix_id
    encrypted_matrix_id = body.encrypted_matrix_id
    shift_matrix_id = body.shift_matrix_id if body.shift_matrix_id else "{}shiftmatrix".format(plaintext_matrix_id)
    k = int(body.k)
    m = int(body.m)
    iterations = int(body.iterations)
    experiment_id = body.experiment_id
    _encrypted_matrix_shape = body.encrypted_matrix_shape

    if not encrypted_matrix_id or not plaintext_matrix_id:
        raise HTTPException(status_code=500, detail="Either Encrypted-Matrix-Id or Plain-Matrix-Id is missing")
    if _encrypted_matrix_shape is None:
        raise HTTPException(status_code=500, detail="Encrypted-Matrix-Shape header is required")

    num_chunks_str = body.num_chunks
    encrypted_udm_id = "{}encryptedudm".format(plaintext_matrix_id)
    response_headers = {}
    is_zero = bool(int(body.is_zero)) if body.is_zero else False
    MICTLANX_TIMEOUT = settings.mictlanx_timeout
    num_chunks = int(num_chunks_str) if num_chunks_str else -1

    storage_backend = (
        StorageBuilder(storage_client=storage_client, scheme=None)
        .with_storage_params(StorageParams(num_chunks=num_chunks, timeout=MICTLANX_TIMEOUT))
        .build()
    )

    try:
        if is_zero:
            response_headers["Clustering-Status"] = Constants.ClusteringStatus.COMPLETED
            end_time = time.time()
            service_time = end_time - local_start_time
            response_headers["Total-Service-Time"] = str(service_time)

            clustering_completed_entry = ExperimentLogEntry(
                event="COMPLETED",
                experiment_id=experiment_id,
                algorithm=algorithm,
                start_time=local_start_time,
                end_time=time.time(),
                id=plaintext_matrix_id,
                worker_id=worker_id,
                num_chunks=num_chunks,
                k=k,
                m=m,
                iterations=iterations
            )
            logger.info(clustering_completed_entry.model_dump())

            return Response(
                status_code=204,
                headers=response_headers
            )

        encrypted_udm_result = await storage_backend.get(
            bucket_id=BUCKET_ID,
            ball_id=encrypted_udm_id,
            segment=True,
            encrypt=False,
            scheme=None
        )
        if encrypted_udm_result.is_err:
            logger.error(f"Failed to get encrypted udm: {encrypted_udm_result.unwrap_err()}")
            raise HTTPException(status_code=500, detail="Failed to get encrypted udm")
        encrypted_udm_get_result = encrypted_udm_result.unwrap()
        udm: npt.NDArray = encrypted_udm_get_result.raw_value

        logger.debug({
            "event": "GET",
            "experiment_id": experiment_id,
            "bucket_id": BUCKET_ID,
            "ball_id": encrypted_udm_id,
            "matrix_id": encrypted_udm_id,
            "shape": str(udm.shape if hasattr(udm, 'shape') else (1,)),
            "dtype": "float32",
            "read_time": encrypted_udm_get_result.read_time,
        })

        shift_matrix_result = await storage_backend.get(
            bucket_id=BUCKET_ID,
            ball_id=shift_matrix_id,
            segment=True,
            encrypt=False,
            scheme=None
        )
        if shift_matrix_result.is_err:
            logger.error(f"Failed to get shift matrix: {shift_matrix_result.unwrap_err()}")
            raise HTTPException(status_code=500, detail="Failed to get shift matrix")
        shift_matrix_get_result = shift_matrix_result.unwrap()
        shift_matrix: npt.NDArray = shift_matrix_get_result.raw_value

        logger.debug({
            "event": "GET",
            "experiment_id": experiment_id,
            "bucket_id": BUCKET_ID,
            "ball_id": shift_matrix_id,
            "matrix_id": shift_matrix_id,
            "shape": str(shift_matrix.shape if hasattr(shift_matrix, 'shape') else (1,)),
            "dtype": "float32",
            "read_time": shift_matrix_get_result.read_time,
        })

        run2_start_time = time.time()
        dbskmeans = DBSKMeans()
        status = Constants.ClusteringStatus.WORK_IN_PROGRESS
        response_headers["Clustering-Status"] = status
        encrypted_matrix_shape = eval(_encrypted_matrix_shape)
        _udm = dbskmeans.execute_plaintext_phase(
            k=k,
            udm=udm,
            num_attributes=int(encrypted_matrix_shape[1]),
            shift_matrix=shift_matrix,
        )

        logger.debug({
            "event": "RUN2",
            "experiment_id": experiment_id,
            "algorithm": algorithm,
            "start_time": run2_start_time,
            "end_time": time.time(),
            "id": plaintext_matrix_id,
            "worker_id": worker_id,
            "num_chunks": num_chunks,
            "k": k,
            "m": m,
            "iterations": iterations
        })

        udm_put_result = await storage_backend.put(
            bucket_id=BUCKET_ID,
            ball_id=encrypted_udm_id,
            data=_udm,
            segment=True,
            encrypt=False,
            scheme=None,
            delete=True
        )
        if udm_put_result.is_err:
            logger.error("Failed to process encrypted udm: {}".format(udm_put_result.unwrap_err()))
            raise HTTPException(status_code=500, detail="Failed to process encrypted udm")
        udm_response = udm_put_result.unwrap()

        logger.debug({
            "event": "PUT",
            "experiment_id": experiment_id,
            "bucket_id": BUCKET_ID,
            "ball_id": encrypted_udm_id,
            "matrix_id": encrypted_udm_id,
            "shape": str(udm_response.shape),
            "dtype": str(udm_response.dtype),
            "read_time": udm_response.read_time,
            "segment_time": udm_response.segment_time,
            "encrypt_time": getattr(udm_response, "encrypt_time", 0.0),
            "upload_time": udm_response.upload_time,
        })

        end_time = time.time()
        service_time = end_time - local_start_time

        logger.debug({
            "event": "UNCOMPLETED",
            "experiment_id": experiment_id,
            "algorithm": algorithm,
            "start_time": local_start_time,
            "end_time": end_time,
            "id": plaintext_matrix_id,
            "worker_id": worker_id,
            "num_chunks": num_chunks,
            "k": k,
            "m": m,
            "iterations": iterations
        })

        response_headers["End-Time"] = str(end_time)
        response_headers["Service-Time"] = str(service_time)

        return Response(
            status_code=204,
            headers=response_headers
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("DBSKMEANS_2_ERROR: " + encrypted_matrix_id + " " + str(e))
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/dbskmeans")
async def dbskmeans(
    body: DbskmeansWorkerRequest,
    logger=Depends(get_logger),
    storage_client=Depends(get_storage_client),
    settings=Depends(get_settings),
):
    if body.step_index == 1:
        return await dbskmeans_1(body, logger, storage_client, settings)
    elif body.step_index == 2:
        return await dbskmeans_2(body, logger, storage_client, settings)
    else:
        raise HTTPException(status_code=400, detail="Failed invalid step_index")


@router.post("/dbsnnc")
async def dbsnnc(
    body: DbsnncWorkerRequest,
    logger=Depends(get_logger),
    storage_client=Depends(get_storage_client),
    settings=Depends(get_settings),
):
    local_start_time = time.time()
    algorithm = Constants.ClusteringAlgorithms.DBSNNC
    worker_id = settings.node_id
    BUCKET_ID: str = settings.mictlanx_bucket_id
    plaintext_matrix_id = body.plaintext_matrix_id
    encrypted_matrix_id = body.encrypted_matrix_id
    encrypted_dm_id = body.encrypted_dm_id
    encrypted_threshold = float(body.encrypted_threshold) if body.encrypted_threshold else None
    _encrypted_matrix_shape = body.encrypted_matrix_shape
    _encrypted_matrix_dtype = body.encrypted_matrix_dtype
    _encrypted_dm_shape = body.encrypted_dm_shape
    _encrypted_dm_dtype = body.encrypted_dm_dtype
    m = int(body.m)
    experiment_id = body.experiment_id
    MICTLANX_TIMEOUT = settings.mictlanx_timeout
    MICTLANX_DELAY = settings.mictlanx_delay
    MICTLANX_BACKOFF_FACTOR = settings.mictlanx_backoff_factor
    MICTLANX_MAX_RETRIES = settings.mictlanx_max_retries

    if _encrypted_matrix_dtype is None:
        raise HTTPException(status_code=500, detail="Encrypted-Matrix-Dtype")
    if _encrypted_matrix_shape is None:
        raise HTTPException(status_code=500, detail="Encrypted-Matrix-Shape header is required")

    if _encrypted_dm_dtype is None:
        raise HTTPException(status_code=500, detail="Encrypted-DM-Dtype")
    if _encrypted_dm_shape is None:
        raise HTTPException(status_code=500, detail="Encrypted-DM-Shape header is required")

    encrypted_matrix_shape: tuple = eval(_encrypted_matrix_shape)
    encrypted_dm_shape: tuple = eval(_encrypted_dm_shape)

    num_chunks_str = body.num_chunks
    responseHeaders = {}

    if num_chunks_str is None:
        raise HTTPException(status_code=503, detail="Num-Chunks header is required")

    num_chunks = int(num_chunks_str)

    try:
        responseHeaders["Start-Time"] = str(local_start_time)

        get_merge_encrypted_matrix_start_time = time.time()
        encryptedMatrix = await RoryCommon.get_and_merge(
            client=storage_client,
            key=encrypted_matrix_id,
            bucket_id=BUCKET_ID,
            max_retries=MICTLANX_MAX_RETRIES,
            delay=MICTLANX_DELAY,
            backoff_factor=MICTLANX_BACKOFF_FACTOR,
            timeout=MICTLANX_TIMEOUT
        )

        get_encrypted_ptm_entry = ExperimentLogEntry(
            event="GET",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=get_merge_encrypted_matrix_start_time,
            end_time=time.time(),
            id=encrypted_matrix_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            m=m,
        )
        logger.info(get_encrypted_ptm_entry.model_dump())

        responseHeaders["Encrypted-Matrix-Dtype"] = encryptedMatrix.dtype
        responseHeaders["Encrypted-Matrix-Shape"] = encryptedMatrix.shape

        get_merge_encrypted_dm_start_time = time.time()
        distance_matrix: npt.NDArray = await RoryCommon.get_and_merge(
            client=storage_client,
            bucket_id=BUCKET_ID,
            key=encrypted_dm_id,
            max_retries=MICTLANX_MAX_RETRIES,
            delay=MICTLANX_DELAY,
            backoff_factor=MICTLANX_BACKOFF_FACTOR,
            timeout=MICTLANX_TIMEOUT
        )

        get_encrypted_ptm_entry = ExperimentLogEntry(
            event="GET",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=get_merge_encrypted_dm_start_time,
            end_time=time.time(),
            id=encrypted_dm_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            m=m,
        )
        logger.info(get_encrypted_ptm_entry.model_dump())

        responseHeaders["Encrypted-Dm-Dtype"] = distance_matrix.dtype
        responseHeaders["Encrypted-Dm-Shape"] = distance_matrix.shape

        dbsnnc_run_start_time = time.time()
        result = Dbsnnc.run(
            distance_matrix=distance_matrix,
            encrypted_threshold=encrypted_threshold
        )
        end_time = time.time()
        dbsnnc_service_time = end_time - dbsnnc_run_start_time
        service_time = end_time - local_start_time

        clustering_entry = ExperimentLogEntry(
            event="COMPLETED",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=local_start_time,
            end_time=time.time(),
            id=encrypted_matrix_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            workers=0,
            security_level=0,
            m=m,
        )
        logger.info(clustering_entry.model_dump())

        return JSONResponse(
            content={
                "label_vector": result.label_vector,
                "service_time": service_time
            },
            status_code=200,
            headers=responseHeaders
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(e)
        return Response(
            status_code=503,
            headers={"Error-Message": str(e)})

@router.post("/nnc")
async def nnc(
    body: NncWorkerRequest,
    logger=Depends(get_logger),
    storage_client=Depends(get_storage_client),
    settings=Depends(get_settings),
):
    local_start_time = time.time()
    algorithm = Constants.ClusteringAlgorithms.NNC
    worker_id = settings.node_id
    BUCKET_ID: str = settings.mictlanx_bucket_id
    plaintext_matrix_id = body.plaintext_matrix_id
    threshold = float(body.threshold) if body.threshold else None
    _plaintext_matrix_shape = body.plaintext_matrix_shape
    _plaintext_matrix_dtype = body.plaintext_matrix_dtype
    _dm_shape = body.dm_shape
    _dm_dtype = body.dm_dtype
    dm_id = "{}dm".format(plaintext_matrix_id)
    response_headers = {}
    experiment_id = body.experiment_id
    MICTLANX_TIMEOUT = settings.mictlanx_timeout
    MICTLANX_DELAY = settings.mictlanx_delay
    MICTLANX_BACKOFF_FACTOR = settings.mictlanx_backoff_factor
    MICTLANX_MAX_RETRIES = settings.mictlanx_max_retries

    if _plaintext_matrix_dtype is None:
        raise HTTPException(status_code=500, detail="Encrypted-Matrix-Dtype")
    if _plaintext_matrix_shape is None:
        raise HTTPException(status_code=500, detail="Encrypted-Matrix-Shape header is required")

    if _dm_dtype is None:
        raise HTTPException(status_code=500, detail="Encrypted-DM-Dtype")
    if _dm_shape is None:
        raise HTTPException(status_code=500, detail="Encrypted-DM-Shape header is required")

    plaintext_matrix_shape: tuple = eval(_plaintext_matrix_shape)
    dm_shape: tuple = eval(_dm_shape)

    num_chunks_str = body.num_chunks
    responseHeaders = {}

    if num_chunks_str is None:
        raise HTTPException(status_code=503, detail="Num-Chunks header is required")

    num_chunks = int(num_chunks_str)

    try:
        response_headers["Start-Time"] = str(local_start_time)

        get_merge_plaintext_matrix_start_time = time.time()

        plaintextMatrix = await RoryCommon.get_and_merge(
            client=storage_client,
            key=plaintext_matrix_id,
            bucket_id=BUCKET_ID,
            max_retries=MICTLANX_MAX_RETRIES,
            delay=MICTLANX_DELAY,
            backoff_factor=MICTLANX_BACKOFF_FACTOR,
            timeout=MICTLANX_TIMEOUT
        )

        get_merge_plaintext_matrix_st = time.time() - get_merge_plaintext_matrix_start_time

        get_encrypted_ptm_entry = ExperimentLogEntry(
            event="GET",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=get_merge_plaintext_matrix_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
        )
        logger.info(get_encrypted_ptm_entry.model_dump())

        responseHeaders["Plaintext-Matrix-Dtype"] = plaintextMatrix.dtype
        responseHeaders["Plaintext-Matrix-Shape"] = plaintextMatrix.shape

        get_merge_dm_start_time = time.time()

        distance_matrix = await RoryCommon.get_and_merge(
            client=storage_client,
            bucket_id=BUCKET_ID,
            key=dm_id,
            max_retries=MICTLANX_MAX_RETRIES,
            delay=MICTLANX_DELAY,
            backoff_factor=MICTLANX_BACKOFF_FACTOR,
            timeout=MICTLANX_TIMEOUT
        )

        get_merge_dm_st = time.time() - get_merge_dm_start_time

        get_encrypted_ptm_entry = ExperimentLogEntry(
            event="GET",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=get_merge_dm_start_time,
            end_time=time.time(),
            id=dm_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
        )
        logger.info(get_encrypted_ptm_entry.model_dump())

        responseHeaders["Dm-Dtype"] = distance_matrix.dtype
        responseHeaders["Dm-Shape"] = distance_matrix.shape

        nnc_run_start_time = time.time()

        result = Nnc.run(
            distance_matrix=distance_matrix,
            threshold=threshold
        )
        end_time = time.time()
        nnc_run_end_time = end_time - nnc_run_start_time
        service_time = end_time - local_start_time

        clustering_entry = ExperimentLogEntry(
            event="COMPLETED",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=local_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
        )
        logger.info(clustering_entry.model_dump())

        return JSONResponse(
            content={
                "label_vector": result.label_vector,
                "service_time": service_time
            },
            status_code=200,
            headers=response_headers
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(e)
        return Response(
            status_code=503,
            headers={"Error-Message": str(e)})

async def pqc_skmeans_1(body: PqcSkmeansWorkerRequest, logger, storage_client, settings, ckks) -> Response:
    try:
        arrival_time = time.time()
        worker_id = settings.node_id
        BUCKET_ID: str = settings.mictlanx_bucket_id
        status = int(body.clustering_status)
        is_start_status = status == Constants.ClusteringStatus.START
        k = int(body.k)
        algorithm = Constants.ClusteringAlgorithms.SKMEANS_PQC
        plaintext_matrix_id = body.plaintext_matrix_id
        encrypted_matrix_id = body.encrypted_matrix_id
        udm_id = "{}udm".format(plaintext_matrix_id)
        _encrypted_matrix_shape = body.encrypted_matrix_shape
        _encrypted_matrix_dtype = body.encrypted_matrix_dtype
        experiment_id = body.experiment_id
        MICTLANX_TIMEOUT = settings.mictlanx_timeout
        MICTLANX_DELAY = settings.mictlanx_delay
        MICTLANX_BACKOFF_FACTOR = settings.mictlanx_backoff_factor
        MICTLANX_MAX_RETRIES = settings.mictlanx_max_retries

        if _encrypted_matrix_dtype is None:
            raise HTTPException(status_code=500, detail="Encrypted-Matrix-Dtype")
        if _encrypted_matrix_shape is None:
            raise HTTPException(status_code=500, detail="Encrypted-Matrix-Shape header is required")

        encrypted_shift_matrix_id = "{}encryptedshiftmatrix".format(plaintext_matrix_id)
        init_sm_id = "{}initsm".format(plaintext_matrix_id)
        cent_i_id = "{}centi".format(plaintext_matrix_id)
        cent_j_id = "{}centj".format(plaintext_matrix_id)
        num_chunks_str = body.num_chunks
        responseHeaders = {}

        if num_chunks_str is None:
            logger.error({
                "msg": "Num-Chunks header is required"
            })
            raise HTTPException(status_code=503, detail="Num-Chunks header is required")

        num_chunks = int(num_chunks_str)

        responseHeaders["Start-Time"] = str(arrival_time)

        get_merge_encrypted_matrix_start_time = time.time()
        init_shiftmatrix = await RoryCommon.get_pyctxt(
            client=storage_client,
            bucket_id=BUCKET_ID,
            key=init_sm_id,
            ckks=ckks,
            delay=MICTLANX_DELAY,
            max_retries=MICTLANX_MAX_RETRIES,
            timeout=MICTLANX_TIMEOUT,
            backoff_factor=MICTLANX_BACKOFF_FACTOR
        )

        get_init_sm_entry = ExperimentLogEntry(
            event="GET",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=get_merge_encrypted_matrix_start_time,
            end_time=time.time(),
            id=init_sm_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            k=k,
        )
        logger.info(get_init_sm_entry.model_dump())

        skmeans = SkmeansPQC(he_object=ckks.he_object, init_shiftmatrix=init_shiftmatrix)
        get_merge_encrypted_matrix_start_time = time.time()
        encryptedMatrix = await RoryCommon.get_pyctxt(
            client=storage_client,
            bucket_id=BUCKET_ID,
            key=encrypted_matrix_id,
            ckks=ckks,
            delay=MICTLANX_DELAY,
            backoff_factor=MICTLANX_BACKOFF_FACTOR,
            max_retries=MICTLANX_MAX_RETRIES,
            timeout=MICTLANX_TIMEOUT,
        )

        get_encrypted_matrix_entry = ExperimentLogEntry(
            event="GET",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=get_merge_encrypted_matrix_start_time,
            end_time=time.time(),
            id=encrypted_matrix_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            k=k,
        )
        logger.info(get_encrypted_matrix_entry.model_dump())

        udm_get_start_time = time.time()
        udm = await RoryCommon.get_and_merge(
            client=storage_client,
            bucket_id=BUCKET_ID,
            key=udm_id,
            force=True,
            delay=MICTLANX_DELAY,
            backoff_factor=MICTLANX_BACKOFF_FACTOR,
            max_retries=MICTLANX_MAX_RETRIES,
            timeout=MICTLANX_TIMEOUT,
        )

        get_udm_entry = ExperimentLogEntry(
            event="GET",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=udm_get_start_time,
            end_time=time.time(),
            id=udm_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            k=k,
        )
        logger.info(get_udm_entry.model_dump())

        responseHeaders["Udm-Matrix-Dtype"] = str(udm.dtype)
        responseHeaders["Udm-Matrix-Shape"] = str(udm.shape)

        if is_start_status:
            __Cent_j = init_shiftmatrix
        else:
            cent_j_start_time = time.time()

            __Cent_j = await RoryCommon.get_pyctxt(
                client=storage_client,
                bucket_id=BUCKET_ID,
                key=cent_i_id,
                ckks=ckks,
                force=True,
                delay=MICTLANX_DELAY,
                backoff_factor=MICTLANX_BACKOFF_FACTOR,
                max_retries=MICTLANX_MAX_RETRIES,
                timeout=MICTLANX_TIMEOUT,
            )
            status = Constants.ClusteringStatus.WORK_IN_PROGRESS
            cent_j_st = time.time() - cent_j_start_time

            get_udm_entry = ExperimentLogEntry(
                event="GET",
                experiment_id=experiment_id,
                algorithm=algorithm,
                start_time=udm_get_start_time,
                end_time=time.time(),
                id=cent_i_id,
                worker_id=worker_id,
                num_chunks=num_chunks,
                k=k,
            )
            logger.info(get_udm_entry.model_dump())

        _encrypted_matrix_shape = eval(_encrypted_matrix_shape)
        run1_start_time = time.time()
        run1_result = skmeans.run1(
            status=status,
            k=k,
            encryptedMatrix=encryptedMatrix,
            UDM=udm,
            Cent_j=__Cent_j,
            num_attributes=_encrypted_matrix_shape[1]
        )

        if run1_result.is_err:
            error = run1_result.unwrap_err()
            logger.error(str(error))
            raise HTTPException(status_code=500, detail=str(error))
        S1, Cent_i, Cent_j, label_vector = run1_result.unwrap()

        run1_entry = ExperimentLogEntry(
            event="RUN1",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=run1_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            k=k
        )
        logger.info(run1_entry.model_dump())

        t1 = time.time()
        maybe_cent_i_chunks = RoryCommon.from_pyctxts_to_chunks(
            key=cent_i_id,
            xs=Cent_i,
            num_chunks=num_chunks)
        print("MAUBE_CENT_I", maybe_cent_i_chunks)
        if maybe_cent_i_chunks.is_none:
            raise HTTPException(status_code=500, detail="Failed to create the Cent_i chunks")

        x = await RoryCommon.delete_and_put_chunks(
            client=storage_client,
            bucket_id=BUCKET_ID,
            key=cent_i_id,
            chunks=maybe_cent_i_chunks.unwrap(),
            timeout=MICTLANX_TIMEOUT,
            max_tries=MICTLANX_MAX_RETRIES
        )
        print("X", x)
        logger.debug(str(x))

        put_cent_i_entry = ExperimentLogEntry(
            event="PUT",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=t1,
            end_time=time.time(),
            id=cent_i_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            k=k,
        )
        logger.info(put_cent_i_entry.model_dump())

        t1 = time.time()
        maybe_cent_j_chunks = RoryCommon.from_pyctxts_to_chunks(
            key=cent_j_id,
            xs=Cent_j,
            num_chunks=num_chunks
        )
        print(maybe_cent_j_chunks)
        if maybe_cent_j_chunks.is_none:
            raise HTTPException(status_code=500, detail="Failed to create the Cent_j chunks")
        y = await RoryCommon.delete_and_put_chunks(
            client=storage_client,
            bucket_id=BUCKET_ID,
            key=cent_j_id,
            chunks=maybe_cent_j_chunks.unwrap(),
            timeout=MICTLANX_TIMEOUT,
            max_tries=MICTLANX_MAX_RETRIES
        )

        put_cent_j_entry = ExperimentLogEntry(
            event="PUT",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=t1,
            end_time=time.time(),
            id=cent_j_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            k=k,
        )
        logger.info(put_cent_j_entry.model_dump())

        t1 = time.time()
        maybe_encrypted_shift_matrix_chunks = RoryCommon.from_pyctxts_to_chunks(
            xs=S1,
            key=encrypted_shift_matrix_id,
            num_chunks=num_chunks
        )
        print(maybe_encrypted_shift_matrix_chunks)
        if maybe_encrypted_shift_matrix_chunks.is_none:
            raise HTTPException(status_code=500, detail="Failed to create the encrypted shift matrix chunks")
        S1_chunks = maybe_encrypted_shift_matrix_chunks.unwrap()
        z = await RoryCommon.delete_and_put_chunks(
            client=storage_client,
            bucket_id=BUCKET_ID,
            key=encrypted_shift_matrix_id,
            chunks=S1_chunks,
            timeout=MICTLANX_TIMEOUT,
            max_tries=MICTLANX_MAX_RETRIES
        )

        put_encrypted_sm_entry = ExperimentLogEntry(
            event="PUT",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=t1,
            end_time=time.time(),
            id=encrypted_shift_matrix_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            k=k,
        )
        logger.info(put_encrypted_sm_entry.model_dump())

        end_time = time.time()
        service_time = end_time - arrival_time
        n_iterations = int(body.iterations) + 1

        clustering_entry = ExperimentLogEntry(
            event="COMPLETED",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=arrival_time,
            end_time=time.time(),
            id=encrypted_matrix_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            k=k,
            iterations=n_iterations
        )
        logger.info(clustering_entry.model_dump())

        return JSONResponse(
            content={
                "label_vector": label_vector,
                "service_time": service_time,
                "n_iterations": n_iterations,
                "encrypted_shift_matrix_id": encrypted_shift_matrix_id
            },
            status_code=200,
            headers=responseHeaders
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error({
            "msg": str(e),
            "at": "worker_skmeans_1"
        })
        raise HTTPException(status_code=500, detail=str(e))


async def pqc_skmeans_2(body: PqcSkmeansWorkerRequest, logger, storage_client, settings, ckks):
    local_start_time = time.time()
    worker_id = settings.node_id
    BUCKET_ID: str = settings.mictlanx_bucket_id
    algorithm = Constants.ClusteringAlgorithms.SKMEANS_PQC
    status = int(body.clustering_status)
    plaintext_matrix_id = body.plaintext_matrix_id
    encrypted_matrix_id = body.encrypted_matrix_id
    shift_matrix_id = body.shift_matrix_id if body.shift_matrix_id else "{}shiftmatrix".format(plaintext_matrix_id)
    k = int(body.k)
    isZero = bool(int(body.is_zero)) if body.is_zero else False
    iterations = int(body.iterations)
    experiment_id = body.experiment_id
    init_sm_id = "{}initsm".format(plaintext_matrix_id)

    MICTLANX_TIMEOUT = settings.mictlanx_timeout
    MICTLANX_DELAY = settings.mictlanx_delay
    MICTLANX_BACKOFF_FACTOR = settings.mictlanx_backoff_factor
    MICTLANX_MAX_RETRIES = settings.mictlanx_max_retries

    if not encrypted_matrix_id or not plaintext_matrix_id:
        raise HTTPException(status_code=500, detail="Either Encrypted-Matrix-Id or Plain-Matrix-Id is missing")
    num_chunks_str = body.num_chunks
    num_chunks = int(num_chunks_str) if num_chunks_str else -1
    udm_id = "{}udm".format(plaintext_matrix_id)
    cent_i_id = "{}centi".format(plaintext_matrix_id)
    cent_j_id = "{}centj".format(plaintext_matrix_id)
    response_headers = {}

    try:
        get_UDM_start_time = time.time()
        UDM = await RoryCommon.get_and_merge(
            client=storage_client,
            key=udm_id,
            bucket_id=BUCKET_ID,
            force=True,
            delay=MICTLANX_DELAY,
            timeout=MICTLANX_TIMEOUT,
            backoff_factor=MICTLANX_BACKOFF_FACTOR,
            max_retries=MICTLANX_MAX_RETRIES
        )

        get_udm_entry = ExperimentLogEntry(
            event="GET",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=get_UDM_start_time,
            end_time=time.time(),
            id=init_sm_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            k=k,
        )
        logger.info(get_udm_entry.model_dump())

        get_UDM_st = time.time() - get_UDM_start_time
        get_shift_matrix_start_time = time.time()

        shiftMatrix = await RoryCommon.get_and_merge(
            client=storage_client,
            key=shift_matrix_id,
            bucket_id=BUCKET_ID,
            force=True,
            delay=MICTLANX_DELAY,
            backoff_factor=MICTLANX_BACKOFF_FACTOR,
            max_retries=MICTLANX_MAX_RETRIES,
            timeout=MICTLANX_TIMEOUT,
        )

        get_sm_entry = ExperimentLogEntry(
            event="GET",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=get_shift_matrix_start_time,
            end_time=time.time(),
            id=encrypted_matrix_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            k=k,
        )
        logger.info(get_sm_entry.model_dump())

        if (isZero):
            response_headers["Clustering-Status"] = Constants.ClusteringStatus.COMPLETED
            end_time = time.time()
            service_time = end_time - local_start_time
            response_headers["Total-Service-Time"] = str(service_time)

            clustering_entry = ExperimentLogEntry(
                event="COMPLETED",
                experiment_id=experiment_id,
                algorithm=algorithm,
                start_time=local_start_time,
                end_time=time.time(),
                id=encrypted_matrix_id,
                worker_id=worker_id,
                num_chunks=num_chunks,
                k=k,
                iterations=iterations
            )
            logger.info(clustering_entry.model_dump())

            return Response(
                status_code=204,
                headers=response_headers
            )

        else:
            t1 = time.time()
            init_shiftmatrix = await RoryCommon.get_pyctxt(
                client=storage_client,
                bucket_id=BUCKET_ID,
                key=init_sm_id,
                ckks=ckks,
                delay=MICTLANX_DELAY,
                backoff_factor=MICTLANX_BACKOFF_FACTOR,
                max_retries=MICTLANX_MAX_RETRIES,
                timeout=MICTLANX_TIMEOUT,
            )

            get_init_sm_entry = ExperimentLogEntry(
                event="GET",
                experiment_id=experiment_id,
                algorithm=algorithm,
                start_time=t1,
                end_time=time.time(),
                id=init_sm_id,
                worker_id=worker_id,
                num_chunks=num_chunks,
                k=k,
            )
            logger.info(get_init_sm_entry.model_dump())

            skmeans = SkmeansPQC(he_object=ckks.he_object, init_shiftmatrix=init_shiftmatrix)
            status = Constants.ClusteringStatus.WORK_IN_PROGRESS

            response_headers["Clustering-Status"] = status
            encrypted_matrix_shape = eval(body.encrypted_matrix_shape)

            run2_start_time = time.time()
            _UDM = skmeans.run_2(
                k=k,
                UDM=UDM,
                num_attributes=int(encrypted_matrix_shape[1]),
                shiftMatrix=shiftMatrix,
            )
            UDM_array = np.array(_UDM)

            run2_entry = ExperimentLogEntry(
                event="RUN2",
                experiment_id=experiment_id,
                algorithm=algorithm,
                start_time=run2_start_time,
                end_time=time.time(),
                id=plaintext_matrix_id,
                worker_id=worker_id,
                num_chunks=num_chunks,
                k=k
            )
            logger.info(run2_entry.model_dump())

            put_udm_start_time = time.time()
            maybe_udm_chunks = Chunks.from_ndarray(
                ndarray=UDM_array,
                group_id=udm_id,
                chunk_prefix=Some(udm_id),
                num_chunks=num_chunks,
            )
            if maybe_udm_chunks.is_none:
                raise HTTPException(status_code=500, detail="something went wrong creating the chunks")

            put_udm_result = await RoryCommon.delete_and_put_chunks(
                client=storage_client,
                bucket_id=BUCKET_ID,
                key=udm_id,
                chunks=maybe_udm_chunks.unwrap(),
                timeout=MICTLANX_TIMEOUT,
                max_tries=MICTLANX_MAX_RETRIES,
                tags={
                    "full_shape": str(UDM_array.shape),
                    "full_dtype": str(UDM_array.dtype)
                }
            )
            if put_udm_result.is_err:
                error = str(put_udm_result.unwrap_err())
                logger.error({
                    "msg": error
                })
                raise HTTPException(status_code=500, detail=str(error))
            endTime2 = time.time()

            put_udm_entry = ExperimentLogEntry(
                event="PUT",
                experiment_id=experiment_id,
                algorithm=algorithm,
                start_time=put_udm_start_time,
                end_time=time.time(),
                id=udm_id,
                worker_id=worker_id,
                num_chunks=num_chunks,
                k=k,
            )
            logger.info(put_udm_entry.model_dump())

            serviceTime2 = endTime2 - local_start_time
            response_headers["End-Time"] = str(endTime2)
            response_headers["Service-Time"] = str(serviceTime2)

            clutering_uncompleted_entry = ExperimentLogEntry(
                event="UNCOMPLETED",
                experiment_id=experiment_id,
                algorithm=algorithm,
                start_time=local_start_time,
                end_time=time.time(),
                id=plaintext_matrix_id,
                worker_id=worker_id,
                num_chunks=num_chunks,
                k=k,
                iterations=iterations
            )
            logger.info(clutering_uncompleted_entry.model_dump())

            return Response(
                status_code=204,
                headers=response_headers
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("SKMEANS_2_ERROR: " + encrypted_matrix_id + " " + str(e))
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/pqc/skmeans")
async def pqc_skmeans(
    body: PqcSkmeansWorkerRequest,
    logger=Depends(get_logger),
    storage_client=Depends(get_storage_client),
    settings=Depends(get_settings),
    ckks=Depends(get_ckks),
):
    logger.info({
        "X": 1,
        "step_index": body.step_index
    })
    if body.step_index == 1:
        return await pqc_skmeans_1(body, logger, storage_client, settings, ckks)
    elif body.step_index == 2:
        return await pqc_skmeans_2(body, logger, storage_client, settings, ckks)
    else:
        raise HTTPException(status_code=500, detail="Invalid step index")

async def pqc_dbskmeans_1(body: PqcDbskmeansWorkerRequest, logger, storage_client, settings, ckks):
    try:
        arrival_time = time.time()
        worker_id = settings.node_id
        BUCKET_ID: str = settings.mictlanx_bucket_id
        status = int(body.clustering_status)
        is_start_status = status == Constants.ClusteringStatus.START
        k = int(body.k)
        algorithm = Constants.ClusteringAlgorithms.DBSKMEANS_PQC
        plaintext_matrix_id = body.plaintext_matrix_id
        encrypted_matrix_id = body.encrypted_matrix_id
        udm_id = "{}udm".format(plaintext_matrix_id)
        _encrypted_matrix_shape = body.encrypted_matrix_shape
        _encrypted_matrix_dtype = body.encrypted_matrix_dtype
        _encrypted_udm_shape = body.encrypted_udm_shape
        _encrypted_udm_dtype = body.encrypted_udm_dtype
        iterations = int(body.iterations)
        experiment_id = body.experiment_id

        if _encrypted_matrix_dtype is None:
            raise HTTPException(status_code=400, detail="Encrypted-Matrix-Dtype")
        if _encrypted_matrix_shape is None:
            raise HTTPException(status_code=400, detail="Encrypted-Matrix-Shape header is required")
        if _encrypted_udm_dtype is None:
            raise HTTPException(status_code=400, detail="Encrypted-UDM-Dtype")
        if _encrypted_udm_shape is None:
            raise HTTPException(status_code=400, detail="Encrypted-UDM-Shape header is required")

        num_chunks_str = body.num_chunks
        encrypted_matrix_shape: tuple = eval(_encrypted_matrix_shape)
        encrypted_udm_shape: tuple = eval(_encrypted_udm_shape)

        encrypted_shift_matrix_id = "{}encryptedshiftmatrix".format(plaintext_matrix_id)
        encrypted_udm_id = "{}encryptedudm".format(plaintext_matrix_id)
        init_sm_id = "{}initsm".format(plaintext_matrix_id)
        cent_i_id = "{}centi".format(plaintext_matrix_id)
        cent_j_id = "{}centj".format(plaintext_matrix_id)
        responseHeaders = {}

        MICTLANX_TIMEOUT = settings.mictlanx_timeout
        MICTLANX_DELAY = settings.mictlanx_delay
        MICTLANX_BACKOFF_FACTOR = settings.mictlanx_backoff_factor
        MICTLANX_MAX_RETRIES = settings.mictlanx_max_retries

        if num_chunks_str is None:
            logger.error({
                "msg": "Num-Chunks header is required"
            })
            raise HTTPException(status_code=503, detail="Num-Chunks header is required")

        num_chunks = int(num_chunks_str)

        responseHeaders["Start-Time"] = str(arrival_time)
        get_merge_encrypted_matrix_start_time = time.time()
        init_shiftmatrix = await RoryCommon.get_pyctxt(
            client=storage_client,
            bucket_id=BUCKET_ID,
            key=init_sm_id,
            ckks=ckks,
            backoff_factor=MICTLANX_BACKOFF_FACTOR,
            delay=MICTLANX_DELAY,
            force=False,
            max_retries=MICTLANX_MAX_RETRIES,
            timeout=MICTLANX_TIMEOUT,
        )

        get_init_sm_entry = ExperimentLogEntry(
            event="GET",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=get_merge_encrypted_matrix_start_time,
            end_time=time.time(),
            id=init_sm_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            k=k,
        )
        logger.info(get_init_sm_entry.model_dump())

        dbskmeans = DbskmeansPQC(he_object=ckks.he_object, init_shiftmatrix=init_shiftmatrix)
        get_merge_encrypted_matrix_start_time = time.time()

        encryptedMatrix = await RoryCommon.get_pyctxt(
            client=storage_client,
            bucket_id=BUCKET_ID,
            key=encrypted_matrix_id,
            ckks=ckks,
            backoff_factor=MICTLANX_BACKOFF_FACTOR,
            delay=MICTLANX_DELAY,
            force=False,
            max_retries=MICTLANX_MAX_RETRIES,
            timeout=MICTLANX_TIMEOUT,
        )

        get_encrypted_matrix_entry = ExperimentLogEntry(
            event="GET",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=get_merge_encrypted_matrix_start_time,
            end_time=time.time(),
            id=encrypted_matrix_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            k=k,
        )
        logger.info(get_encrypted_matrix_entry.model_dump())

        get_merge_start_time = time.time()
        encrypted_udm = await RoryCommon.get_and_merge(
            client=storage_client,
            bucket_id=BUCKET_ID,
            key=encrypted_udm_id,
            max_retries=MICTLANX_MAX_RETRIES,
            delay=MICTLANX_DELAY,
            backoff_factor=MICTLANX_BACKOFF_FACTOR,
            timeout=MICTLANX_TIMEOUT,
        )

        get_udm_entry = ExperimentLogEntry(
            event="GET",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=get_merge_start_time,
            end_time=time.time(),
            id=encrypted_udm_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            k=k,
        )
        logger.info(get_udm_entry.model_dump())

        responseHeaders["Encrypted-Udm-Dtype"] = str(encrypted_udm.dtype)
        responseHeaders["Encrypted-Udm-Shape"] = str(encrypted_udm.shape)
        if is_start_status:
            __Cent_j = init_shiftmatrix
        else:
            cent_j_start_time = time.time()
            __Cent_j = await RoryCommon.get_pyctxt(
                client=storage_client,
                bucket_id=BUCKET_ID,
                key=cent_i_id,
                ckks=ckks
            )
            status = Constants.ClusteringStatus.WORK_IN_PROGRESS

            get_udm_entry = ExperimentLogEntry(
                event="GET",
                experiment_id=experiment_id,
                algorithm=algorithm,
                start_time=cent_j_start_time,
                end_time=time.time(),
                id=cent_i_id,
                worker_id=worker_id,
                num_chunks=num_chunks,
                k=k,
            )
            logger.info(get_udm_entry.model_dump())

        run1_start_time = time.time()
        run1_result = dbskmeans.run1(
            status=status,
            k=k,
            encryptedMatrix=encryptedMatrix,
            UDM=encrypted_udm,
            Cent_j=__Cent_j,
            num_attributes=encrypted_matrix_shape[1]
        )
        if run1_result.is_err:
            error = run1_result.unwrap_err()
            logger.error(str(error))
            raise HTTPException(status_code=500, detail=str(error))
        S1, Cent_i, Cent_j, label_vector = run1_result.unwrap()

        run1_entry = ExperimentLogEntry(
            event="RUN1",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=run1_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            k=k
        )
        logger.info(run1_entry.model_dump())

        t1 = time.time()
        maybe_cent_i_chunks = RoryCommon.from_pyctxts_to_chunks(
            key=cent_i_id,
            num_chunks=num_chunks,
            xs=Cent_i
        )
        if maybe_cent_i_chunks.is_none:
            raise HTTPException(status_code=500, detail="Failed to create chunks from cent_i")

        x = await RoryCommon.delete_and_put_chunks(
            client=storage_client,
            bucket_id=BUCKET_ID,
            key=cent_i_id,
            chunks=maybe_cent_i_chunks.unwrap(),
            timeout=MICTLANX_TIMEOUT,
            max_tries=MICTLANX_MAX_RETRIES,
        )
        if x.is_err:
            raise HTTPException(status_code=500, detail="Failed to put cent i")

        put_cent_i_entry = ExperimentLogEntry(
            event="PUT",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=t1,
            end_time=time.time(),
            id=cent_i_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            k=k,
        )
        logger.info(put_cent_i_entry.model_dump())

        t1 = time.time()
        maybe_cent_j_chunks = RoryCommon.from_pyctxts_to_chunks(
            key=cent_j_id,
            num_chunks=num_chunks,
            xs=Cent_j
        )

        if maybe_cent_j_chunks.is_none:
            raise HTTPException(status_code=500, detail="Failed to create chunks from cent_j")
        y = await RoryCommon.delete_and_put_chunks(
            client=storage_client,
            bucket_id=BUCKET_ID,
            key=cent_j_id,
            chunks=maybe_cent_j_chunks.unwrap(),
            timeout=MICTLANX_TIMEOUT,
            max_tries=MICTLANX_MAX_RETRIES,
        )
        if y.is_err:
            raise HTTPException(status_code=500, detail="Failed to put cent j")

        put_cent_j_entry = ExperimentLogEntry(
            event="PUT",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=t1,
            end_time=time.time(),
            id=cent_j_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            k=k,
        )
        logger.info(put_cent_j_entry.model_dump())

        t1 = time.time()
        maybe_s1_chunks = RoryCommon.from_pyctxts_to_chunks(
            key=encrypted_shift_matrix_id,
            num_chunks=num_chunks,
            xs=S1
        )
        if maybe_s1_chunks.is_none:
            raise HTTPException(status_code=500, detail="Failed to create chunks from encrypted shiftmatrix")
        z = await RoryCommon.delete_and_put_chunks(
            client=storage_client,
            bucket_id=BUCKET_ID,
            key=encrypted_shift_matrix_id,
            chunks=maybe_s1_chunks.unwrap(),
            timeout=MICTLANX_TIMEOUT,
            max_tries=MICTLANX_MAX_RETRIES,
        )
        if z.is_err:
            raise HTTPException(status_code=500, detail="Failed to put encrypted shift matrix")

        put_encrypted_sm_entry = ExperimentLogEntry(
            event="PUT",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=t1,
            end_time=time.time(),
            id=encrypted_shift_matrix_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            k=k,
        )
        logger.info(put_encrypted_sm_entry.model_dump())

        end_time = time.time()
        service_time = end_time - arrival_time
        n_iterations = int(body.iterations) + 1

        clustering_entry = ExperimentLogEntry(
            event="COMPLETED",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=arrival_time,
            end_time=time.time(),
            id=encrypted_matrix_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            k=k,
            iterations=n_iterations
        )
        logger.info(clustering_entry.model_dump())

        return JSONResponse(
            content={
                "label_vector": label_vector,
                "service_time": service_time,
                "n_iterations": n_iterations,
                "encrypted_shift_matrix_id": encrypted_shift_matrix_id
            },
            status_code=200,
            headers=responseHeaders
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error({
            "msg": str(e),
            "at": "worker_dbskmeans_1"
        })
        raise HTTPException(status_code=500, detail=str(e))

async def pqc_dbskmeans_2(body: PqcDbskmeansWorkerRequest, logger, storage_client, settings, ckks):
    local_start_time = time.time()
    worker_id = settings.node_id
    BUCKET_ID: str = settings.mictlanx_bucket_id
    algorithm = Constants.ClusteringAlgorithms.SKMEANS_PQC
    status = int(body.clustering_status)
    plaintext_matrix_id = body.plaintext_matrix_id
    encrypted_matrix_id = body.encrypted_matrix_id
    shift_matrix_id = body.shift_matrix_id if body.shift_matrix_id else "{}shiftmatrix".format(plaintext_matrix_id)
    k = int(body.k)
    isZero = bool(int(body.is_zero)) if body.is_zero else False
    iterations = int(body.iterations)
    experiment_id = body.experiment_id

    shift_matrix_ope_id = body.shift_matrix_ope_id if body.shift_matrix_ope_id else "{}-shift-matrix-ope".format(plaintext_matrix_id)
    _encrypted_matrix_shape = body.encrypted_matrix_shape
    _encrypted_matrix_dtype = body.encrypted_matrix_dtype
    _encrypted_udm_shape = body.encrypted_udm_shape
    _encrypted_udm_dtype = body.encrypted_udm_dtype

    MICTLANX_TIMEOUT = settings.mictlanx_timeout
    MICTLANX_DELAY = settings.mictlanx_delay
    MICTLANX_BACKOFF_FACTOR = settings.mictlanx_backoff_factor
    MICTLANX_MAX_RETRIES = settings.mictlanx_max_retries

    if not encrypted_matrix_id or not plaintext_matrix_id:
        raise HTTPException(status_code=500, detail="Either Encrypted-Matrix-Id or Plain-Matrix-Id is missing")
    num_chunks_str = body.num_chunks
    if _encrypted_matrix_dtype is None:
        raise HTTPException(status_code=500, detail="Encrypted-Matrix-Dtype")
    if _encrypted_matrix_shape is None:
        raise HTTPException(status_code=500, detail="Encrypted-Matrix-Shape header is required")

    if _encrypted_udm_dtype is None:
        raise HTTPException(status_code=500, detail="Encrypted-UDM-Dtype")
    if _encrypted_udm_shape is None:
        raise HTTPException(status_code=500, detail="Encrypted-UDM-Shape header is required")

    num_chunks = int(num_chunks_str) if num_chunks_str else -1
    encrypted_matrix_shape: tuple = eval(_encrypted_matrix_shape)
    encrypted_udm_shape: tuple = eval(_encrypted_udm_shape)
    encrypted_udm_id = "{}encryptedudm".format(plaintext_matrix_id)
    init_sm_id = "{}initsm".format(plaintext_matrix_id)
    cent_i_id = "{}centi".format(plaintext_matrix_id)
    cent_j_id = "{}centj".format(plaintext_matrix_id)
    response_headers = {}

    try:
        get_merge_start_time = time.time()
        prev_encrypted_udm = await RoryCommon.get_and_merge(
            client=storage_client,
            bucket_id=BUCKET_ID,
            key=encrypted_udm_id,
            timeout=MICTLANX_TIMEOUT,
            max_retries=MICTLANX_MAX_RETRIES,
            delay=MICTLANX_DELAY,
            backoff_factor=MICTLANX_BACKOFF_FACTOR,
            force=True
        )

        get_udm_entry = ExperimentLogEntry(
            event="GET",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=get_merge_start_time,
            end_time=time.time(),
            id=encrypted_udm_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            k=k,
        )
        logger.info(get_udm_entry.model_dump())

        if (isZero):
            response_headers["Clustering-Status"] = Constants.ClusteringStatus.COMPLETED
            end_time = time.time()
            service_time = end_time - local_start_time
            response_headers["Total-Service-Time"] = str(service_time)

            clustering_entry = ExperimentLogEntry(
                event="COMPLETED",
                experiment_id=experiment_id,
                algorithm=algorithm,
                start_time=local_start_time,
                end_time=time.time(),
                id=encrypted_matrix_id,
                worker_id=worker_id,
                num_chunks=num_chunks,
                k=k,
                iterations=iterations
            )
            logger.info(clustering_entry.model_dump())

            return Response(
                status_code=204,
                headers=response_headers
            )

        else:
            t1 = time.time()
            init_shiftmatrix = await RoryCommon.get_pyctxt(
                client=storage_client,
                bucket_id=BUCKET_ID,
                key=init_sm_id,
                ckks=ckks,
                backoff_factor=MICTLANX_BACKOFF_FACTOR,
                delay=MICTLANX_DELAY,
                max_retries=MICTLANX_MAX_RETRIES,
                timeout=MICTLANX_TIMEOUT,
            )

            get_init_sm_entry = ExperimentLogEntry(
                event="GET",
                experiment_id=experiment_id,
                algorithm=algorithm,
                start_time=t1,
                end_time=time.time(),
                id=init_sm_id,
                worker_id=worker_id,
                num_chunks=num_chunks,
                k=k,
            )
            logger.info(get_init_sm_entry.model_dump())

            dbskmeans = DbskmeansPQC(he_object=ckks.he_object, init_shiftmatrix=init_shiftmatrix)
            status = Constants.ClusteringStatus.WORK_IN_PROGRESS

            response_headers["Clustering-Status"] = status
            get_matrix_start_time = time.time()
            shift_matrix_ope_response = await RoryCommon.get_and_merge(
                client=storage_client,
                bucket_id=BUCKET_ID,
                key=shift_matrix_ope_id,
                max_retries=MICTLANX_MAX_RETRIES,
                backoff_factor=MICTLANX_BACKOFF_FACTOR,
                delay=MICTLANX_DELAY,
                force=True,
                timeout=MICTLANX_TIMEOUT,
            )
            shift_matrix_ope: npt.NDArray = shift_matrix_ope_response.value
            response_headers["Clustering-Status"] = status

            get_init_sm_entry = ExperimentLogEntry(
                event="GET",
                experiment_id=experiment_id,
                algorithm=algorithm,
                start_time=get_matrix_start_time,
                end_time=time.time(),
                id=shift_matrix_ope_id,
                worker_id=worker_id,
                num_chunks=num_chunks,
                k=k,
            )
            logger.info(get_init_sm_entry.model_dump())

            run2_start_time = time.time()
            current_udm = dbskmeans.run_2(
                k=k,
                UDM=prev_encrypted_udm,
                attributes=int(encrypted_matrix_shape[1]),
                shiftMatrix=shift_matrix_ope,
            )

            run2_entry = ExperimentLogEntry(
                event="RUN2",
                experiment_id=experiment_id,
                algorithm=algorithm,
                start_time=run2_start_time,
                end_time=time.time(),
                id=plaintext_matrix_id,
                worker_id=worker_id,
                num_chunks=num_chunks,
                k=k
            )
            logger.info(run2_entry.model_dump())
            put_udm_start_time = time.time()
            maybe_udm_chunks: Option[Chunks] = Chunks.from_ndarray(
                ndarray=current_udm,
                group_id=encrypted_udm_id,
                num_chunks=num_chunks,
                chunk_prefix=Some(encrypted_udm_id)
            )
            if maybe_udm_chunks.is_none:
                logger.error({"msg": "Something went wrong segment encrypted udm."})
                raise HTTPException(status_code=500, detail="Something went wrong segment udm.")
            udm_chunks = maybe_udm_chunks.unwrap()
            cm_shape = str(current_udm.shape)
            cm_dtype = str(current_udm.dtype)
            del current_udm

            put_chunks_udm_generator_results = await RoryCommon.delete_and_put_chunks(
                client=storage_client,
                bucket_id=BUCKET_ID,
                key=encrypted_udm_id,
                chunks=udm_chunks,
                timeout=MICTLANX_TIMEOUT,
                max_tries=MICTLANX_MAX_RETRIES,
                tags={
                    "full_shape": cm_shape,
                    "full_dtype": cm_dtype,
                }
            )
            del udm_chunks

            put_udm_entry = ExperimentLogEntry(
                event="PUT",
                experiment_id=experiment_id,
                algorithm=algorithm,
                start_time=put_udm_start_time,
                end_time=time.time(),
                id=encrypted_udm_id,
                worker_id=worker_id,
                num_chunks=num_chunks,
                k=k,
            )
            logger.info(put_udm_entry.model_dump())

            end_time = time.time()
            service_time = end_time - local_start_time
            response_headers["End-Time"] = str(end_time)
            response_headers["Service-Time"] = str(service_time)

            clutering_uncompleted_entry = ExperimentLogEntry(
                event="UNCOMPLETED",
                experiment_id=experiment_id,
                algorithm=algorithm,
                start_time=local_start_time,
                end_time=time.time(),
                id=plaintext_matrix_id,
                worker_id=worker_id,
                num_chunks=num_chunks,
                k=k,
                iterations=iterations
            )
            logger.info(clutering_uncompleted_entry.model_dump())

            del prev_encrypted_udm
            return Response(
                status_code=204,
                headers=response_headers
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("DBSKMEANS_2_ERROR: " + encrypted_matrix_id + " " + str(e))
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/pqc/dbskmeans")
async def pqc_dbskmeans(
    body: PqcDbskmeansWorkerRequest,
    logger=Depends(get_logger),
    storage_client=Depends(get_storage_client),
    settings=Depends(get_settings),
    ckks=Depends(get_ckks),
):
    logger.info({
        "X": 1,
        "step_index": body.step_index
    })
    if body.step_index == 1:
        return await pqc_dbskmeans_1(body, logger, storage_client, settings, ckks)
    elif body.step_index == 2:
        return await pqc_dbskmeans_2(body, logger, storage_client, settings, ckks)
    else:
        return Response()
