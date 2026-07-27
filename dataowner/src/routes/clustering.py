import os
import time
import numpy as np
from option import Some
from requests import Session
from fastapi import APIRouter, Depends, HTTPException, status
from rory.core.interfaces.rorymanager import RoryManager
from rory.core.interfaces.roryworker import RoryWorker
from rory.core.security.dataowner import DataOwner
from rory.core.security.pqc.dataowner import DataOwner as DataOwnerPQC
from rory.core.security.cryptosystem.fdhope import Fdhope
from rory.core.security.cryptosystem.pqc.ckks import Ckks
from rory.core.utils.constants import Constants
from rory.core.utils.utils import Utils as RoryUtils
from mictlanx.utils.segmentation import Chunks
from utils.utils import Utils
from rorycommon import Common as RoryCommon
from rorycommon import StorageBuilder, StorageParams, Scheme, LiuParams, FdhopeParams
from models import ExperimentLogEntry
from models.requests.clustering import (
    KmeansRequest,
    SkmeansRequest,
    DbskmeansRequest,
    DbsnncRequest,
    NncRequest,
    PqcSkmeansRequest,
    PqcDbskmeansRequest,
)
from models.responses.clustering import (
    HealthCheckResponse,
    KmeansResponse,
    SkmeansResponse,
    DbskmeansResponse,
    DbsnncResponse,
    NncResponse,
    PqcSkmeansResponse,
    PqcDbskmeansResponse,
)
from dependencies import get_logger, get_storage_client, get_manager, get_liu, get_dataowner, get_executor, get_ckks, get_settings

router = APIRouter(prefix="/clustering", tags=["Clustering"])


@router.api_route(
    "/test",
    methods=["GET", "POST"],
    response_model=HealthCheckResponse,
    summary="Health check and component identification",
    description="Verify the availability of the Dataowner component and confirm its role within the Rory platform architecture.",
)
def test():
    """Health check and component identification endpoint.

    This method serves as a simple diagnostic tool to verify the availability
    of the Dataowner component and confirm its role within the Rory platform
    architecture. It is used during deployment and orchestration to ensure
    proper network connectivity between nodes.

    Returns:
        component_type (str): Identifies this node as "dataowner".

        Status Code:
            200: If the service is running and reachable.
    """
    return {"component_type": "dataowner"}


# KMEANS
@router.post(
    "/kmeans",
    response_model=KmeansResponse,
    summary="Plaintext K-Means clustering",
    description="Reads a local plaintext dataset, externalizes it to CSS, requests a Worker, and triggers clustering.",
)
async def kmeans(
    body: KmeansRequest,
    logger=Depends(get_logger),
    settings=Depends(get_settings),
    storage=Depends(get_storage_client),
    manager=Depends(get_manager),
):
    """
    This method handles the lifecycle of a clustering task by reading a local plaintext dataset,
    externalizing it to the Cloud Storage System (CSS), requesting an available execution
    node (Worker) from the Manager, and finally triggering the privacy-preserving
    mining process.
    The method also tracks and logs performance metrics (service times) for the Client,
    Manager, and Worker interactions to facilitate experimental auditing.

    Note:
    **Protocol Initiation**: All execution parameters for this algorithm are passed exclusively
    via **HTTP Headers**. The request body must remain empty.

    Attributes:
        Plaintext-Matrix-Id (str): Unique identifier for the matrix in CSS. Defaults to "matrix-0".
        Plaintext-Matrix-Filename (str): Name of the local file (without extension). Defaults to "matrix-0".
        Extension (str): File extension of the local dataset (e.g., "csv", "npy"). Defaults to "csv".
        K (int): The number of clusters to form. Defaults to "3".
        Experiment-Id (str): A unique identifier for the execution trace. Defaults to a hex UUID.

    Returns:
        label_vector (list): The cluster assignment for each dataset point.
        iterations (int): Total iterations performed by the algorithm.
        algorithm (str): The name of the algorithm executed (kmeans).
        worker_id (str): Identifier of the worker node that processed the task.
        service_time_manager (float): Time spent coordinating with the Manager.
        service_time_worker (float): Time spent during Worker execution.
        service_time_dataowner (float): Time spent in local data preparation/reading.
        response_time_clustering (float): Total end-to-end execution time.

    Raises:
        Exception: Captures and logs any failure during local I/O, CSS communication,
            or Manager/Worker interaction, returning a 500 status code with the
            error details in the headers.
    """
    try:
        arrivalTime = time.time()
        TESTING = settings.testing
        SOURCE_PATH = settings.source_path
        BUCKET_ID = settings.mictlanx_bucket_id
        WORKER_TIMEOUT = settings.worker_timeout
        num_chunks = settings.num_chunks
        algorithm = Constants.ClusteringAlgorithms.KMEANS
        s = Session()
        plaintext_matrix_id = body.plaintext_matrix_id
        plaintext_matrix_filename = body.plaintext_matrix_filename
        extension = body.extension
        k = body.k
        experiment_id = body.experiment_id
        plaintext_matrix_path = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_filename, extension)
        read_dataset_start_time = time.time()
        plaintext_matrix_result = await RoryCommon.read_numpy_from(
            path=plaintext_matrix_path,
            extension=extension,
        )

        if plaintext_matrix_result.is_ok:
            plaintextMatrix = plaintext_matrix_result.unwrap()
        else:
            raise plaintext_matrix_result.unwrap_err()

        local_read_entry = ExperimentLogEntry(
            event="LOCAL.READ",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=read_dataset_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            num_chunks=num_chunks,
            k=k,
        )
        logger.info(local_read_entry.model_dump())

        put_pm_start_time = time.time()
        put_ptm_result = await RoryCommon.put_ndarray(
            client=storage,
            key=plaintext_matrix_id,
            matrix=plaintextMatrix,
            tags={},
            bucket_id=BUCKET_ID,
        )
        if put_ptm_result.is_err:
            error = put_ptm_result.unwrap_err()
            logger.error({
                "msg": str(error)
            })
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(error))

        put_ptm_entry = ExperimentLogEntry(
            event="PUT",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=put_pm_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id="",
            num_chunks=num_chunks,
            k=k,
            workers=0,
        )
        logger.info(put_ptm_entry.model_dump())

        service_time_dataowner = time.time() - arrivalTime
        get_worker_start_time = time.time()
        get_worker_result = manager.getWorker(
            headers={
                "Algorithm": algorithm,
                "Start-Request-Time": str(arrivalTime),
            }
        )
        if get_worker_result.is_err:
            error = get_worker_result.unwrap_err()
            logger.error({
                "error": "GET.WORKER.FAILED",
                "message": str(error),
            })
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(error))
        (worker_id, worker_port) = get_worker_result.unwrap()

        get_worker_end_time = time.time()
        worker_id = "localhost" if TESTING else worker_id

        get_worker_entry = ExperimentLogEntry(
            event="GET.WORKER",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=get_worker_start_time,
            end_time=get_worker_end_time,
            id=plaintext_matrix_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            k=k,
            workers=0,
        )
        logger.info(get_worker_entry.model_dump())

        worker_run_1_start_time = time.time()
        manager_service_time = worker_run_1_start_time - get_worker_start_time
        worker = RoryWorker(
            workerId=worker_id,
            port=worker_port,
            session=s,
            algorithm=algorithm,
        )

        interaction_arrival_time = time.time()

        workerResponse = worker.run(
            headers={
                "Plaintext-Matrix-Id": plaintext_matrix_id,
                "K": str(k),
                "Experiment-Id": experiment_id,
            },
            timeout=WORKER_TIMEOUT,
        )
        worker_run_1_entry = ExperimentLogEntry(
            event="WORKER.RUN.1",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=worker_run_1_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            k=k,
            workers=0,
        )
        logger.info(worker_run_1_entry.model_dump())

        jsonWorkerResponse = workerResponse.json()
        iterations = int(jsonWorkerResponse["iterations"])
        endTime = time.time()
        worker_response_time = endTime - worker_run_1_start_time
        response_time = endTime - arrivalTime

        kmeans_completed_entry = ExperimentLogEntry(
            event="COMPLETED",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=arrivalTime,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            k=k,
            iterations=iterations,
            worker_time=worker_response_time,
            dataowner_time=service_time_dataowner,
            manager_time=manager_service_time,
        )
        logger.info(kmeans_completed_entry.model_dump())

        return {
            "label_vector": jsonWorkerResponse.get("label_vector", []),
            "iterations": iterations,
            "algorithm": algorithm,
            "worker_id": worker_id,
            "service_time_manager": manager_service_time,
            "service_time_worker": worker_response_time,
            "service_time_dataowner": service_time_dataowner,
            "response_time_clustering": response_time,
        }
    except Exception as e:
        logger.error({
            "msg": str(e)
        })
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# SKMEANS
@router.post(
    "/skmeans",
    response_model=SkmeansResponse,
    summary="Secure K-Means clustering with Liu homomorphic encryption",
    description="Interactive privacy-preserving K-Means clustering protocol powered by Liu's homomorphic encryption scheme.",
)
async def skmeans(
    body: SkmeansRequest,
    logger=Depends(get_logger),
    settings=Depends(get_settings),
    storage=Depends(get_storage_client),
    manager=Depends(get_manager),
    liu=Depends(get_liu),
    dataowner=Depends(get_dataowner),
    executor=Depends(get_executor),
):
    """
    This method implements an interactive, privacy-preserving K-Means clustering protocol
    powered by Liu's homomorphic encryption scheme. The workflow is designed for
    Privacy-Preserving Data Mining as a Service (PPDMaaS), where the Client (Data Owner)
    remains the only entity capable of decrypting intermediate computations.

    Note:
    **Interactive Protocol**: This endpoint initiates the secure clustering flow. All parameters,
    including cryptographic metadata, must be passed via **HTTP Headers**.

    Attributes:
        Plaintext-Matrix-Id (str): Unique ID for the matrix. Defaults to "matrix0".
        Plaintext-Matrix-Filename (str): Local filename for data reading. Defaults to "matrix0".
        Extension (str): File extension of the dataset. Defaults to "csv".
        K (int): Number of clusters to identify. **Required**.
        Max-Iterations (int): Maximum number of protocol rounds. Defaults to 10.
        Experiment-Id (str): Tracking ID for performance auditing.
        Experiment-Iteration (str): Current loop index of the experiment.

    Returns:
        label_vector (list): Final cluster assignments for the dataset.
        iterations (int): Actual number of iterations performed.
        algorithm (str): "skmeans".
        worker_id (str): ID of the node that performed the secure computations.
        service_time_manager (float): Time spent in Worker allocation.
        service_time_worker (float): Cumulative time of remote computation.
        service_time_dataowner (float): Total local time (Encryption/Decryption/IO).
        response_time_clustering (float): End-to-end execution time.

    Raises:
        Exception: Returns a 500 status code if the process executor is missing,
            or if failures occur during encryption, CSS I/O, or Worker interaction.
    """
    try:
        arrivalTime = time.time()
        BUCKET_ID = settings.mictlanx_bucket_id
        TESTING = settings.testing
        SOURCE_PATH = settings.source_path
        max_workers = settings.max_workers
        num_chunks = settings.num_chunks
        security_level = settings.liu_security_level
        if executor is None:
            raise HTTPException(status_code=500, detail="No process pool executor available")
        algorithm = Constants.ClusteringAlgorithms.SKMEANS
        s = Session()
        plaintext_matrix_id = body.plaintext_matrix_id
        encrypted_matrix_id = "encrypted{}".format(plaintext_matrix_id)
        udm_id = "{}udm".format(plaintext_matrix_id)
        plaintext_matrix_filename = body.plaintext_matrix_filename
        extension = body.extension
        experiment_id = body.experiment_id
        k = body.k
        experiment_iteration = body.experiment_iteration
        requestId = "request-{}".format(plaintext_matrix_id)
        m = dataowner.m
        convergence_threshold = body.convergence_threshold
        plaintext_matrix_path = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_filename, extension)
        MAX_ITERATIONS = body.max_iterations
        WORKER_TIMEOUT = settings.worker_timeout
        MICTLANX_TIMEOUT = settings.mictlanx_timeout

        liu_params = LiuParams(
            _round=liu.round,
            decimals=liu.decimals,
            secure_random=liu.secure_random,
            seed=liu.seed,
            use_np_random=liu.use_np_random,
            security_level=security_level,
        )

        storage_backend = (
            StorageBuilder(storage_client=storage, scheme=Scheme.LIU)
            .with_liu_params(liu_params=liu_params)
            .with_storage_params(StorageParams(num_chunks=num_chunks, timeout=MICTLANX_TIMEOUT))
            .build()
        )

        plaintext_matrix_result = await RoryCommon.read_numpy_from(
            path=plaintext_matrix_path,
            extension=extension,
        )

        if plaintext_matrix_result.is_err:
            logger.error("Failed to process dataset: {}".format(plaintext_matrix_result.unwrap_err()))
            raise HTTPException(status_code=500, detail="Failed to process dataset")
        plaintext_matrix = plaintext_matrix_result.unwrap()

        logger.debug({
            "event": "LOCAL.READ",
            "experiment_id": experiment_id,
            "algorithm": algorithm,
            "id": plaintext_matrix_id,
            "worker_id": "",
            "num_chunks": num_chunks,
            "k": k,
            "workers": max_workers,
            "security_level": security_level,
            "m": m,
        })

        plaintext_matrix_result_2 = await storage_backend.put(
            bucket_id=BUCKET_ID,
            ball_id=encrypted_matrix_id,
            data=plaintext_matrix,
            scheme=Scheme.LIU,
            segment=True,
            encrypt=True,
            delete=True,
        )

        if plaintext_matrix_result_2.is_err:
            logger.error("Failed to process dataset: {}".format(plaintext_matrix_result_2.unwrap_err()))
            raise HTTPException(status_code=500, detail="Failed to process dataset")
        plaintext_matrix_response = plaintext_matrix_result_2.unwrap()

        logger.debug({
            "event": "PUT",
            "experiment_id": experiment_id,
            "bucket_id": BUCKET_ID,
            "ball_id": encrypted_matrix_id,
            "matrix_id": encrypted_matrix_id,
            "shape": str(plaintext_matrix_response.shape),
            "dtype": str(plaintext_matrix_response.dtype),
            "read_time": plaintext_matrix_response.read_time,
            "segment_time": plaintext_matrix_response.segment_time,
            "encrypt_time": getattr(plaintext_matrix_response, "encrypt_time", 0.0),
            "upload_time": plaintext_matrix_response.upload_time,
        })

        udm_start_time = time.time()
        udm = dataowner.get_U(
            plaintext_matrix=plaintext_matrix,
            algorithm=algorithm,
        )

        logger.debug({
            "event": "GET.UDM",
            "experiment_id": experiment_id,
            "shape": str(udm.shape),
            "type": str(udm.dtype),
            "udm_id": udm_id,
            "udm_time": time.time() - udm_start_time,
        })

        udm_put_result = await storage_backend.put(
            bucket_id=BUCKET_ID,
            data=udm,
            ball_id=udm_id,
            segment=True,
            encrypt=False,
            scheme=Scheme.LIU,
            delete=True,
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

        service_time_dataowner = time.time() - arrivalTime
        get_worker_start_time = time.time()
        get_worker_result = manager.getWorker(
            headers={
                "Algorithm": algorithm,
                "Start-Request-Time": str(arrivalTime),
                "Start-Get-Worker-Time": str(get_worker_start_time),
            }
        )
        if get_worker_result.is_err:
            error = get_worker_result.unwrap_err()
            logger.error(str(error))
            raise HTTPException(status_code=500, detail=str(error))
        (_worker_id, port) = get_worker_result.unwrap()

        logger.debug({
            "event": "GET.WORKER",
            "worker_id": _worker_id,
            "port": port,
            "is_local": TESTING,
        })

        get_worker_end_time = time.time()
        get_worker_service_time = get_worker_end_time - get_worker_start_time
        worker_id = "localhost" if TESTING else worker_id

        worker_start_time = time.time()
        worker = RoryWorker(
            workerId=worker_id,
            port=port,
            session=s,
            algorithm=algorithm,
        )
        status_val = Constants.ClusteringStatus.START
        worker_run1_response = None
        iterations = 0
        label_vector = None
        endTime = 0

        while (status_val != Constants.ClusteringStatus.COMPLETED):

            inner_interaction_arrival_time = time.time()
            run1_headers = {
                "Step-Index": "1",
                "Clustering-Status": str(status_val),
                "Plaintext-Matrix-Id": plaintext_matrix_id,
                "Request-Id": requestId,
                "Encrypted-Matrix-Id": encrypted_matrix_id,
                "Encrypted-Matrix-Shape": str(plaintext_matrix.shape),
                "Encrypted-Matrix-Dtype": "float32",
                "Encrypted-Udm-Dtype": "float32",
                "Num-Chunks": str(num_chunks),
                "Iterations": str(iterations),
                "K": str(k),
                "M": str(m),
                "Experiment-Iteration": str(experiment_iteration),
                "Max-Iterations": str(MAX_ITERATIONS),
                "Experiment-Id": experiment_id,
            }
            logger.debug({
                "event": "WORKER.RUN",
                "worker_id": _worker_id,
                "status": str(status_val),
                "experiment_id": experiment_id,
                "plaintext_matrix_id": plaintext_matrix_id,
                "num_chunks": str(num_chunks),
                "iterations": str(iterations),
                "k": str(k),
                "m": str(m),
                "experiment_iteration": str(experiment_iteration),
                "max_iterations": str(MAX_ITERATIONS),
                "current_iteration": iterations,
            })

            worker_run1_response = worker.run(
                timeout=WORKER_TIMEOUT,
                headers=run1_headers,
            )
            worker_run1_status = worker_run1_response.status_code

            if worker_run1_status != 200:
                raise HTTPException(status_code=500, detail="Worker error: {}".format(worker_run1_response.content))

            worker_run1_response.raise_for_status()
            jsonWorkerResponse = worker_run1_response.json()
            encrypted_shift_matrix_id = jsonWorkerResponse["encrypted_shift_matrix_id"]
            run1_service_time = jsonWorkerResponse["service_time"]
            run1_n_iterations = jsonWorkerResponse["n_iterations"]
            label_vector = jsonWorkerResponse["label_vector"]

            logger.debug({
                "event": "WORKER.RUN.COMPLETED",
                "run_1_service_time": run1_service_time,
                "n_iterations": run1_n_iterations,
                "label_vector": str(label_vector),
            })

            encrypted_shift_matrix_result = await storage_backend.get(
                bucket_id=BUCKET_ID,
                ball_id=encrypted_shift_matrix_id,
                segment=True,
                encrypt=True,
                scheme=Scheme.LIU,
            )
            if encrypted_shift_matrix_result.is_err:
                logger.error(f"Failed to get shift matrix: {encrypted_shift_matrix_result.unwrap_err()}")
                raise HTTPException(status_code=500, detail="Failed to get shift matrix")
            encrypted_shift_matrix_get_result = encrypted_shift_matrix_result.unwrap()
            encrypted_shift_matrix = encrypted_shift_matrix_get_result.raw_value

            logger.debug({
                "event": "GET",
                "experiment_id": experiment_id,
                "bucket_id": BUCKET_ID,
                "ball_id": encrypted_shift_matrix_id,
                "matrix_id": encrypted_shift_matrix_id,
                "shape": str(encrypted_shift_matrix.shape if hasattr(encrypted_shift_matrix, 'shape') else (1,)),
                "dtype": "float32",
                "read_time": encrypted_shift_matrix_get_result.read_time,
            })

            decrypt_start_time = time.time()
            shiftMatrix_chipher_schema_res = liu.decryptMatrix(
                ciphertext_matrix=encrypted_shift_matrix,
                secret_key=dataowner.sk,
            )
            end_time_decryption = time.time() - decrypt_start_time

            logger.debug({
                "event": "DECRYPT.SHIFTMATRIX",
                "experiment_id": experiment_id,
                "encrypted_shift_matrix_id": encrypted_shift_matrix_id,
                "decrypt_time": end_time_decryption,
            })

            shift_matrix = shiftMatrix_chipher_schema_res.matrix
            mean_shift_matrix = np.mean(np.abs(shift_matrix))
            logger.debug({
                "Shift_matrix": str(shift_matrix),
                "Mean Shift": str(mean_shift_matrix),
                "fMS": float(mean_shift_matrix),
            })

            shift_matrix_id = "{}shiftmatrix".format(plaintext_matrix_id)

            is_converged = float(mean_shift_matrix) <= convergence_threshold

            logger.debug({
                "event": "CONVERGENCE.CHECK",
                "experiment_id": experiment_id,
                "mean_shift": mean_shift_matrix,
                "threshold": convergence_threshold,
                "is_converged": is_converged,
            })

            if not is_converged:
                put_shift_matrix_start_time = time.time()

                encrypted_shift_matrix_result = await storage_backend.put(
                    bucket_id=BUCKET_ID,
                    data=shift_matrix,
                    ball_id=shift_matrix_id,
                    segment=True,
                    encrypt=False,
                    scheme=Scheme.LIU,
                    delete=True,
                )

                if encrypted_shift_matrix_result.is_err:
                    logger.error("Failed to put encrypted shift matrix in cloud storage: {}".format(encrypted_shift_matrix_result.unwrap_err()))
                    raise HTTPException(status_code=500, detail="Failed to put encrypted shift matrix in cloud storage")
                encrypted_shift_matrix_response = encrypted_shift_matrix_result.unwrap()

                logger.debug({
                    "event": "PUT",
                    "experiment_id": experiment_id,
                    "bucket_id": BUCKET_ID,
                    "ball_id": shift_matrix_id,
                    "matrix_id": shift_matrix_id,
                    "shape": str(shift_matrix.shape),
                    "dtype": str(shift_matrix.dtype),
                    "read_time": encrypted_shift_matrix_response.read_time,
                    "segment_time": encrypted_shift_matrix_response.segment_time,
                    "encrypt_time": getattr(encrypted_shift_matrix_response, "encrypt_time", 0.0),
                    "upload_time": encrypted_shift_matrix_response.upload_time,
                })

            if is_converged:
                status_val = Constants.ClusteringStatus.COMPLETED
            else:
                status_val = Constants.ClusteringStatus.WORK_IN_PROGRESS
                run2_headers = {
                    "Step-Index": "2",
                    "Clustering-Status": str(status_val),
                    "Is-Zero": "0",
                    "Shift-Matrix-Id": shift_matrix_id,
                    "Plaintext-Matrix-Id": plaintext_matrix_id,
                    "Encrypted-Matrix-Id": encrypted_matrix_id,
                    "Encrypted-Matrix-Shape": str(plaintext_matrix.shape),
                    "Encrypted-Matrix-Dtype": "float32",
                    "Num-Chunks": str(num_chunks),
                    "Iterations": str(iterations),
                    "K": str(k),
                    "M": str(m),
                    "Experiment-Iteration": str(experiment_iteration),
                    "Max-Iterations": str(MAX_ITERATIONS),
                    "Experiment-Id": experiment_id,
                }

                logger.debug({
                    "event": "WORKER.RUN.2",
                    "worker_id": _worker_id,
                    "status": str(status_val),
                    "experiment_id": experiment_id,
                    "plaintext_matrix_id": plaintext_matrix_id,
                    "num_chunks": str(num_chunks),
                    "iterations": str(iterations),
                    "k": str(k),
                    "m": str(m),
                    "experiment_iteration": str(experiment_iteration),
                    "max_iterations": str(MAX_ITERATIONS),
                })

                worker_run2_response = worker.run(
                    timeout=WORKER_TIMEOUT,
                    headers=run2_headers,
                )

                worker_run2_response.raise_for_status()
                service_time_worker = worker_run2_response.headers.get("Service-Time", 0)

            iterations += 1
            if iterations >= MAX_ITERATIONS:
                status_val = Constants.ClusteringStatus.COMPLETED
                startTime = float(s.headers.get("Start-Time", 0))
                service_time_worker = time.time() - startTime
                logger.debug({
                    "event": "NO.CONVERGED.MAX_ITERATION_REACHED",
                    "experiment_id": experiment_id,
                    "algorithm": algorithm,
                    "iterations": iterations,
                    "max_iterations": MAX_ITERATIONS,
                })
            elif not is_converged:
                status_val = int(worker_run2_response.headers.get("Clustering-Status", Constants.ClusteringStatus.WORK_IN_PROGRESS))
            endTime = time.time()

            logger.debug({
                "event": "ITERATION.COMPLETED",
                "experiment_id": experiment_id,
                "algorithm": algorithm,
                "start_time": arrivalTime,
                "end_time": time.time(),
                "id": plaintext_matrix_id,
                "worker_id": worker_id,
                "num_chunks": num_chunks,
                "k": k,
                "workers": max_workers,
                "security_level": security_level,
                "m": m,
                "iterations": run1_n_iterations,
            })

        worker_response_time = endTime - worker_start_time
        response_time = endTime - arrivalTime

        logger.info(ExperimentLogEntry(
            event="COMPLETED",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=arrivalTime,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            k=k,
            workers=max_workers,
            security_level=security_level,
            m=m,
            iterations=iterations,
            dataowner_time=service_time_dataowner,
            manager_time=get_worker_service_time,
            worker_time=worker_response_time,
        ).model_dump())

        return {
            "label_vector": label_vector,
            "iterations": iterations,
            "algorithm": algorithm,
            "worker_id": worker_id,
            "service_time_manager": get_worker_service_time,
            "service_time_worker": worker_response_time,
            "service_time_dataowner": service_time_dataowner,
            "response_time_clustering": response_time,
        }
    except Exception as e:
        logger.error({
            "msg": str(e)
        })
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# DBSKMEANS
@router.post(
    "/dbskmeans",
    response_model=DbskmeansResponse,
    summary="Double-Blind Secure K-Means clustering",
    description="Privacy-preserving Double-Blind Secure K-Means protocol using hybrid Liu + FDHOPE encryption.",
)
async def dbskmeans(
    body: DbskmeansRequest,
    logger=Depends(get_logger),
    settings=Depends(get_settings),
    storage=Depends(get_storage_client),
    manager=Depends(get_manager),
    liu=Depends(get_liu),
    dataowner=Depends(get_dataowner),
    executor=Depends(get_executor),
):
    """
    This method implements a privacy-preserving Double-Blind Secure K-Means protocol that ensures
    both the Worker and the Manager remain "blind" to the underlying data. It
    leverages a hybrid encryption approach, using Liu's homomorphic scheme for
    initial data protection and the FDHOPE scheme for secure operations on distance
    metrics.

    The convergence decision is made on the Client side by evaluating the decrypted
    shift matrix. The decision is communicated to the Worker via the Is-Zero header.

    Note:
    **Multi-Party Security**: Parameters for the double-blind execution are handled via **HTTP Headers**.
    Ensure the correct 'Experiment-Id' is provided for session tracking.

    Attributes:
        Plaintext-Matrix-Id (str): Unique ID for the matrix. Defaults to "matrix0".
        Plaintext-Matrix-Filename (str): Local file to be processed. Defaults to "matrix0".
        K (int): Number of clusters. Defaults to "3".
        Sens (float): Sensitivity parameter for the FDHOPE scheme. Defaults to 0.00000001.
        Max-Iterations (int): Maximum protocol rounds. Defaults to 10.
        Convergence-Threshold (float): Tolerance for centroid shift. Defaults to "0.000001".
        Experiment-Id (str): Tracking ID for performance auditing.

    Returns:
        label_vector (list): Final cluster assignments.
        iterations (int): Total rounds performed.
        algorithm (str): "dbskmeans".
        worker_id (str): ID of the node that performed the secure computations.
        service_time_manager (float): Time spent in Worker allocation.
        service_time_worker (float): Cumulative time of remote computation.
        service_time_dataowner (float): Total local time (Encryption/Decryption/IO).
        response_time_clustering (float): End-to-end execution time.

    Raises:
        Exception: Returns a 500 status code if the process executor is unavailable,
            or if failures occur during the hybrid encryption/decryption chain
            or CSS communication.
    """
    try:
        arrivalTime = time.time()
        BUCKET_ID = settings.mictlanx_bucket_id
        TESTING = settings.testing
        SOURCE_PATH = settings.source_path
        max_workers = settings.max_workers
        num_chunks = settings.num_chunks
        security_level = settings.liu_security_level
        if executor is None:
            raise HTTPException(status_code=500, detail="No process pool executor available")
        algorithm = Constants.ClusteringAlgorithms.DBSKMEANS
        s = Session()
        plaintext_matrix_id = body.plaintext_matrix_id
        encrypted_matrix_id = "encrypted{}".format(plaintext_matrix_id)
        encrypted_udm_id = "{}encryptedudm".format(plaintext_matrix_id)
        plaintext_matrix_filename = body.plaintext_matrix_filename
        extension = body.extension
        experiment_id = body.experiment_id
        k = body.k
        m = dataowner.m
        sens = body.sens
        experiment_iteration = body.experiment_iteration
        convergence_threshold = body.convergence_threshold
        plaintext_matrix_path = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_filename, extension)
        MAX_ITERATIONS = body.max_iterations
        WORKER_TIMEOUT = settings.worker_timeout
        MICTLANX_TIMEOUT = settings.mictlanx_timeout

        liu_params = LiuParams(
            _round=liu.round,
            decimals=liu.decimals,
            secure_random=liu.secure_random,
            seed=liu.seed,
            use_np_random=liu.use_np_random,
            security_level=security_level,
        )

        fdhope_params = FdhopeParams(
            scheme=algorithm,
            sens=sens,
            _round=liu.round,
            decimals=liu.decimals,
            secure_random=liu.secure_random,
            seed=liu.seed,
            use_np_random=liu.use_np_random,
            security_level=security_level,
        )

        liu_storage = (
            StorageBuilder(storage_client=storage, scheme=Scheme.LIU)
            .with_liu_params(liu_params=liu_params)
            .with_storage_params(StorageParams(num_chunks=num_chunks, timeout=MICTLANX_TIMEOUT))
            .build()
        )

        fdhope_storage = (
            StorageBuilder(storage_client=storage, scheme=Scheme.FDHOPE)
            .with_fdhope_params(fdhope_params=fdhope_params)
            .with_storage_params(StorageParams(num_chunks=num_chunks, timeout=MICTLANX_TIMEOUT))
            .build()
        )

        plaintext_matrix_result = await RoryCommon.read_numpy_from(
            path=plaintext_matrix_path,
            extension=extension,
        )
        if plaintext_matrix_result.is_err:
            logger.error("Failed to process dataset: {}".format(plaintext_matrix_result.unwrap_err()))
            raise HTTPException(status_code=500, detail="Failed to process dataset")
        plaintext_matrix = plaintext_matrix_result.unwrap()

        r = plaintext_matrix.shape[0]
        a = plaintext_matrix.shape[1]

        logger.debug({
            "event": "LOCAL.READ",
            "experiment_id": experiment_id,
            "algorithm": algorithm,
            "id": plaintext_matrix_id,
            "worker_id": "",
            "num_chunks": num_chunks,
            "k": k,
            "workers": max_workers,
            "security_level": security_level,
            "m": m,
        })

        encrypted_matrix_result = await liu_storage.put(
            bucket_id=BUCKET_ID,
            ball_id=encrypted_matrix_id,
            data=plaintext_matrix,
            scheme=Scheme.LIU,
            segment=True,
            encrypt=True,
            delete=True,
        )
        if encrypted_matrix_result.is_err:
            logger.error("Failed to process dataset: {}".format(encrypted_matrix_result.unwrap_err()))
            raise HTTPException(status_code=500, detail="Failed to process dataset")
        encrypted_matrix_response = encrypted_matrix_result.unwrap()

        logger.debug({
            "event": "PUT",
            "experiment_id": experiment_id,
            "bucket_id": BUCKET_ID,
            "ball_id": encrypted_matrix_id,
            "matrix_id": encrypted_matrix_id,
            "shape": str(encrypted_matrix_response.shape),
            "dtype": str(encrypted_matrix_response.dtype),
            "read_time": encrypted_matrix_response.read_time,
            "segment_time": encrypted_matrix_response.segment_time,
            "encrypt_time": getattr(encrypted_matrix_response, "encrypt_time", 0.0),
            "upload_time": encrypted_matrix_response.upload_time,
        })

        udm_start_time = time.time()
        udm = dataowner.get_U(
            plaintext_matrix=plaintext_matrix,
            algorithm=algorithm,
        )

        logger.debug({
            "event": "GET.UDM",
            "experiment_id": experiment_id,
            "shape": str(udm.shape),
            "type": str(udm.dtype),
            "encrypted_udm_id": encrypted_udm_id,
            "udm_time": time.time() - udm_start_time,
        })

        del plaintext_matrix

        udm_put_result = await fdhope_storage.put(
            bucket_id=BUCKET_ID,
            data=udm,
            ball_id=encrypted_udm_id,
            segment=True,
            encrypt=True,
            scheme=Scheme.FDHOPE,
            delete=True,
        )
        if udm_put_result.is_err:
            logger.error("Failed to process udm: {}".format(udm_put_result.unwrap_err()))
            raise HTTPException(status_code=500, detail="Failed to process udm")
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

        service_time_dataowner = time.time() - arrivalTime
        del udm

        get_worker_start_time = time.time()
        get_worker_result = manager.getWorker(
            headers={
                "Algorithm": algorithm,
                "Start-Request-Time": str(arrivalTime),
                "Start-Get-Worker-Time": str(get_worker_start_time),
            }
        )
        if get_worker_result.is_err:
            error = get_worker_result.unwrap_err()
            logger.error(str(error))
            raise HTTPException(status_code=500, detail=str(error))
        (_worker_id, port) = get_worker_result.unwrap()

        logger.debug({
            "event": "GET.WORKER",
            "worker_id": _worker_id,
            "port": port,
            "is_local": TESTING,
        })

        get_worker_end_time = time.time()
        get_worker_service_time = get_worker_end_time - get_worker_start_time
        worker_id = "localhost" if TESTING else _worker_id

        worker_start_time = time.time()
        worker = RoryWorker(
            workerId=worker_id,
            port=port,
            session=s,
            algorithm=algorithm,
        )
        status_val = Constants.ClusteringStatus.START
        label_vector = None
        iterations = 0
        endTime = 0

        while (status_val != Constants.ClusteringStatus.COMPLETED):

            inner_interaction_arrival_time = time.time()
            run1_headers = {
                "Step-Index": "1",
                "Clustering-Status": str(status_val),
                "Plaintext-Matrix-Id": plaintext_matrix_id,
                "Encrypted-Matrix-Id": encrypted_matrix_id,
                "Encrypted-Matrix-Shape": "({},{},{})".format(r, a, m),
                "Encrypted-Matrix-Dtype": "float32",
                "Encrypted-Udm-Shape": "({},{},{})".format(r, r, a),
                "Encrypted-Udm-Dtype": "float32",
                "Num-Chunks": str(num_chunks),
                "Iterations": str(iterations),
                "K": str(k),
                "M": str(m),
                "Experiment-Iteration": str(experiment_iteration),
                "Max-Iterations": str(MAX_ITERATIONS),
                "Experiment-Id": experiment_id,
            }
            logger.debug({
                "event": "WORKER.RUN",
                "worker_id": _worker_id,
                "status": str(status_val),
                "experiment_id": experiment_id,
                "plaintext_matrix_id": plaintext_matrix_id,
                "num_chunks": str(num_chunks),
                "iterations": str(iterations),
                "k": str(k),
                "m": str(m),
                "experiment_iteration": str(experiment_iteration),
                "max_iterations": str(MAX_ITERATIONS),
                "current_iteration": iterations,
            })

            worker_run1_response = worker.run(
                timeout=WORKER_TIMEOUT,
                headers=run1_headers,
            )
            worker_run1_status = worker_run1_response.status_code

            if worker_run1_status != 200:
                raise HTTPException(status_code=500, detail="Worker error: {}".format(worker_run1_response.content))

            worker_run1_response.raise_for_status()
            jsonWorkerResponse = worker_run1_response.json()
            encrypted_shift_matrix_id = jsonWorkerResponse["encrypted_shift_matrix_id"]
            run1_service_time = jsonWorkerResponse["service_time"]
            run1_n_iterations = jsonWorkerResponse["n_iterations"]
            label_vector = jsonWorkerResponse["label_vector"]

            logger.debug({
                "event": "WORKER.RUN.COMPLETED",
                "run_1_service_time": run1_service_time,
                "n_iterations": run1_n_iterations,
                "label_vector": str(label_vector),
            })

            encrypted_shift_matrix_result = await liu_storage.get(
                bucket_id=BUCKET_ID,
                ball_id=encrypted_shift_matrix_id,
                segment=True,
                encrypt=True,
                scheme=Scheme.LIU,
            )
            if encrypted_shift_matrix_result.is_err:
                logger.error(f"Failed to get shift matrix: {encrypted_shift_matrix_result.unwrap_err()}")
                raise HTTPException(status_code=500, detail="Failed to get shift matrix")
            encrypted_shift_matrix_get_result = encrypted_shift_matrix_result.unwrap()
            encrypted_shift_matrix = encrypted_shift_matrix_get_result.raw_value

            logger.debug({
                "event": "GET",
                "experiment_id": experiment_id,
                "bucket_id": BUCKET_ID,
                "ball_id": encrypted_shift_matrix_id,
                "matrix_id": encrypted_shift_matrix_id,
                "shape": str(encrypted_shift_matrix.shape if hasattr(encrypted_shift_matrix, 'shape') else (1,)),
                "dtype": "float32",
                "read_time": encrypted_shift_matrix_get_result.read_time,
            })

            decrypt_start_time = time.time()
            shiftMatrix_cipher_schema_res = liu.decryptMatrix(
                ciphertext_matrix=encrypted_shift_matrix,
                secret_key=dataowner.sk,
            )
            end_time_decryption = time.time() - decrypt_start_time

            logger.debug({
                "event": "DECRYPT.SHIFTMATRIX",
                "experiment_id": experiment_id,
                "encrypted_shift_matrix_id": encrypted_shift_matrix_id,
                "decrypt_time": end_time_decryption,
            })

            shift_matrix = shiftMatrix_cipher_schema_res.matrix
            mean_shift_matrix = np.mean(np.abs(shift_matrix))
            logger.debug({
                "Shift_matrix": str(shift_matrix),
                "Mean Shift": str(mean_shift_matrix),
                "fMS": float(mean_shift_matrix),
            })

            is_converged = float(mean_shift_matrix) <= convergence_threshold

            logger.debug({
                "event": "CONVERGENCE.CHECK",
                "experiment_id": experiment_id,
                "mean_shift": mean_shift_matrix,
                "threshold": convergence_threshold,
                "is_converged": is_converged,
            })

            shift_matrix_id = "{}shiftmatrix".format(plaintext_matrix_id)

            if not is_converged:
                fdhope_encrypted_shift_matrix = Fdhope.encryptMatrix(
                    plaintext_matrix=shift_matrix,
                    messagespace=dataowner.messageIntervals,
                    cipherspace=dataowner.cypherIntervals,
                )

                encrypted_shift_put_result = await liu_storage.put(
                    bucket_id=BUCKET_ID,
                    data=fdhope_encrypted_shift_matrix.matrix,
                    ball_id=shift_matrix_id,
                    segment=True,
                    encrypt=False,
                    scheme=None,
                    delete=True,
                )
                if encrypted_shift_put_result.is_err:
                    logger.error("Failed to put encrypted shift matrix: {}".format(encrypted_shift_put_result.unwrap_err()))
                    raise HTTPException(status_code=500, detail="Failed to put encrypted shift matrix")
                encrypted_shift_response = encrypted_shift_put_result.unwrap()

                logger.debug({
                    "event": "PUT",
                    "experiment_id": experiment_id,
                    "bucket_id": BUCKET_ID,
                    "ball_id": shift_matrix_id,
                    "matrix_id": shift_matrix_id,
                    "shape": str(encrypted_shift_response.shape),
                    "dtype": str(encrypted_shift_response.dtype),
                    "read_time": encrypted_shift_response.read_time,
                    "segment_time": encrypted_shift_response.segment_time,
                    "encrypt_time": getattr(encrypted_shift_response, "encrypt_time", 0.0),
                    "upload_time": encrypted_shift_response.upload_time,
                })

            if is_converged:
                status_val = Constants.ClusteringStatus.COMPLETED
            else:
                status_val = Constants.ClusteringStatus.WORK_IN_PROGRESS
                run2_headers = {
                    "Step-Index": "2",
                    "Clustering-Status": str(status_val),
                    "Is-Zero": "0",
                    "Shift-Matrix-Id": shift_matrix_id,
                    "Plaintext-Matrix-Id": plaintext_matrix_id,
                    "Encrypted-Matrix-Id": encrypted_matrix_id,
                    "Encrypted-Matrix-Shape": "({},{},{})".format(r, a, m),
                    "Num-Chunks": str(num_chunks),
                    "Iterations": str(iterations),
                    "K": str(k),
                    "M": str(m),
                    "Experiment-Iteration": str(experiment_iteration),
                    "Max-Iterations": str(MAX_ITERATIONS),
                    "Experiment-Id": experiment_id,
                }

                logger.debug({
                    "event": "WORKER.RUN.2",
                    "worker_id": _worker_id,
                    "status": str(status_val),
                    "experiment_id": experiment_id,
                    "plaintext_matrix_id": plaintext_matrix_id,
                    "num_chunks": str(num_chunks),
                    "iterations": str(iterations),
                    "k": str(k),
                    "m": str(m),
                    "experiment_iteration": str(experiment_iteration),
                    "max_iterations": str(MAX_ITERATIONS),
                })

                worker_run2_response = worker.run(
                    timeout=WORKER_TIMEOUT,
                    headers=run2_headers,
                )
                worker_run2_response.raise_for_status()
                service_time_worker = worker_run2_response.headers.get("Service-Time", 0)

            iterations += 1
            if iterations >= MAX_ITERATIONS:
                status_val = Constants.ClusteringStatus.COMPLETED
                logger.debug({
                    "event": "NO.CONVERGED.MAX_ITERATION_REACHED",
                    "experiment_id": experiment_id,
                    "algorithm": algorithm,
                    "iterations": iterations,
                    "max_iterations": MAX_ITERATIONS,
                })
            elif not is_converged:
                status_val = int(worker_run2_response.headers.get("Clustering-Status", Constants.ClusteringStatus.WORK_IN_PROGRESS))
            endTime = time.time()

            logger.debug({
                "event": "ITERATION.COMPLETED",
                "experiment_id": experiment_id,
                "algorithm": algorithm,
                "start_time": arrivalTime,
                "end_time": time.time(),
                "id": plaintext_matrix_id,
                "worker_id": worker_id,
                "num_chunks": num_chunks,
                "k": k,
                "workers": max_workers,
                "security_level": security_level,
                "m": m,
                "iterations": run1_n_iterations,
            })

        worker_response_time = endTime - worker_start_time
        response_time = endTime - arrivalTime

        logger.info(ExperimentLogEntry(
            event="COMPLETED",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=arrivalTime,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            k=k,
            workers=max_workers,
            security_level=security_level,
            m=m,
            iterations=iterations,
            dataowner_time=service_time_dataowner,
            manager_time=get_worker_service_time,
            worker_time=worker_response_time,
        ).model_dump())

        return {
            "label_vector": label_vector,
            "iterations": iterations,
            "algorithm": algorithm,
            "worker_id": worker_id,
            "service_time_manager": get_worker_service_time,
            "service_time_worker": worker_response_time,
            "service_time_dataowner": service_time_dataowner,
            "response_time_clustering": response_time,
        }
    except Exception as e:
        logger.error({
            "msg": str(e)
        })
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# DBSNNC
@router.post(
    "/dbsnnc",
    response_model=DbsnncResponse,
    summary="Double-Blind Secure Nearest Neighbor Clustering",
    description="Non-iterative privacy-preserving clustering protocol based on nearest neighbors with hybrid Liu + FDHOPE encryption.",
)
async def dbsnnc(
    body: DbsnncRequest,
    logger=Depends(get_logger),
    settings=Depends(get_settings),
    storage=Depends(get_storage_client),
    manager=Depends(get_manager),
    dataowner=Depends(get_dataowner),
    executor=Depends(get_executor),
):
    """
    This method implements a non-iterative, privacy-preserving clustering protocol
    based on nearest neighbors. It utilizes a Double-Blind Secure (DBS) approach
    where sensitive data and distance metrics are protected using a combination of
    Liu's homomorphic encryption and the FDHOPE scheme.

    Note:
    All identifiers for the input matrices and distance metrics are extracted from **HTTP Headers**.

    Attributes:
        Plaintext-Matrix-Id (str): Unique ID for the matrix. Defaults to "matrix0".
        Plaintext-Matrix-Filename (str): Local file to be processed. Defaults to "matrix-0".
        Sens (float): Sensitivity parameter for FDHOPE encryption. Defaults to 0.00000001.
        Threshold (float): Distance threshold for clustering. If -1, it is calculated
            automatically from the dataset.
        Experiment-Id (str): Tracking ID for performance auditing.

    Returns:
        label_vector (list): The resulting cluster assignments.
        algorithm (str): "dbsnnc".
        worker_id (str): ID of the node that performed the secure computations.
        service_time_manager (float): Time spent in Worker allocation.
        service_time_worker (float): Cumulative time of remote computation.
        service_time_dataowner (float): Total local time (Encryption/Decryption/IO).
        response_time_clustering (float): End-to-end execution time.
    Raises:
        Exception: Returns a 500 status code if the process executor is missing,
            or if errors occur during encryption, CSS communication, or Worker execution.
    """
    try:
        local_start_time = time.time()
        BUCKET_ID = settings.mictlanx_bucket_id
        TESTING = settings.testing
        SOURCE_PATH = settings.source_path
        max_workers = settings.max_workers
        num_chunks = settings.num_chunks
        np_random = settings.np_random
        securitylevel = settings.liu_security_level
        if executor is None:
            raise HTTPException(status_code=500, detail="No process pool executor available")
        algorithm = Constants.ClusteringAlgorithms.DBSNNC
        s = Session()
        plaintext_matrix_id = body.plaintext_matrix_id
        encrypted_matrix_id = "encrypted{}".format(plaintext_matrix_id)
        dm_id = "{}dm".format(plaintext_matrix_id)
        encrypted_dm_id = "{}encrypteddm".format(plaintext_matrix_id)
        plaintext_matrix_filename = body.plaintext_matrix_filename
        extension = body.extension
        experiment_id = body.experiment_id
        m = dataowner.m
        sens = body.sens
        threshold = body.threshold
        request_id = "request{}".format(plaintext_matrix_id)
        plaintext_matrix_path = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_filename, extension)
        experiment_iteration = "0"
        cores = os.cpu_count()

        WORKER_TIMEOUT = settings.worker_timeout
        MICTLANX_TIMEOUT = settings.mictlanx_timeout
        MICTLANX_MAX_RETRIES = settings.mictlanx_max_retries

        local_read_dataset_start_time = time.time()
        plaintext_matrix_result = await RoryCommon.read_numpy_from(
            path=plaintext_matrix_path,
            extension=extension,
        )
        if plaintext_matrix_result.is_ok:
            plaintext_matrix = plaintext_matrix_result.unwrap()
        else:
            raise plaintext_matrix_result.unwrap_err()

        r = plaintext_matrix.shape[0]
        a = plaintext_matrix.shape[1]

        encryption_start_time = time.time()

        local_read_entry = ExperimentLogEntry(
            event="LOCAL.READ",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=local_read_dataset_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id="",
            num_chunks=num_chunks,
            workers=max_workers,
            m=m,
            security_level=securitylevel,
        )
        logger.info(local_read_entry.model_dump())

        n = r * a * m

        segment_encrypt_start_time = time.time()
        encrypted_matrix_chunks = RoryCommon.segment_and_encrypt_liu_with_executor(
            executor=executor,
            key=encrypted_matrix_id,
            dataowner=dataowner,
            plaintext_matrix=plaintext_matrix,
            n=n,
            np_random=np_random,
            num_chunks=num_chunks,
        )

        segment_encrypt_entry = ExperimentLogEntry(
            event="SEGMENT.ENCRYPT",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=segment_encrypt_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id="",
            num_chunks=num_chunks,
            workers=max_workers,
            m=m,
            security_level=securitylevel,
        )
        logger.info(segment_encrypt_entry.model_dump())

        put_chunks_start_time = time.time()
        put_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
            client=storage,
            bucket_id=BUCKET_ID,
            key=encrypted_matrix_id,
            chunks=encrypted_matrix_chunks,
            timeout=MICTLANX_TIMEOUT,
            max_tries=MICTLANX_MAX_RETRIES,
            tags={
                "full_shape": str((r, a, m)),
                "full_dtype": "float32",
            },
        )

        put_encrypted_ptm_entry = ExperimentLogEntry(
            event="PUT",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=put_chunks_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id="",
            num_chunks=num_chunks,
            workers=max_workers,
            m=m,
            security_level=securitylevel,
        )
        logger.info(put_encrypted_ptm_entry.model_dump())

        dm_start_time = time.time()
        dm = dataowner.get_U(
            plaintext_matrix=plaintext_matrix,
            algorithm=algorithm,
        )

        udm_gen_entry = ExperimentLogEntry(
            event="UDM.GENERATION",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=dm_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id="",
            num_chunks=num_chunks,
            workers=max_workers,
            m=m,
            security_level=securitylevel,
        )
        logger.info(udm_gen_entry.model_dump())

        if threshold == -1:
            threshold = RoryUtils.get_threshold(
                distance_matrix=dm,
            )
        n = r * r

        segment_encrypt_fdhope_start_time = time.time()
        encrypted_matrix_DM_chunks = RoryCommon.segment_and_encrypt_fdhope_with_executor(
            executor=executor,
            algorithm=algorithm,
            key=encrypted_dm_id,
            dataowner=dataowner,
            matrix=dm,
            n=n,
            num_chunks=num_chunks,
            sens=sens,
        )

        segment_encrypt_entry = ExperimentLogEntry(
            event="SEGMENT.ENCRYPT",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=segment_encrypt_fdhope_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id="",
            num_chunks=num_chunks,
            workers=max_workers,
            m=m,
            security_level=securitylevel,
        )
        logger.info(segment_encrypt_entry.model_dump())
        put_chunks_start_time = time.time()

        put_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
            client=storage,
            bucket_id=BUCKET_ID,
            key=encrypted_dm_id,
            chunks=encrypted_matrix_DM_chunks,
            timeout=MICTLANX_TIMEOUT,
            max_tries=MICTLANX_MAX_RETRIES,
            tags={
                "full_shape": str((r, r)),
                "full_dtype": "float32",
            },
        )

        udm_put_entry = ExperimentLogEntry(
            event="PUT",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=put_chunks_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id="",
            num_chunks=num_chunks,
            workers=max_workers,
            m=m,
            security_level=securitylevel,
        )
        logger.info(udm_put_entry.model_dump())

        encrypted_threshold = Fdhope.encrypt(
            plaintext=threshold,
            messagespace=dataowner.messageIntervals,
            cipherspace=dataowner.cypherIntervals,
            sens=sens,
        )

        service_time_dataowner = time.time() - local_start_time
        get_worker_start_time = time.time()

        get_worker_result = manager.getWorker(
            headers={
                "Algorithm": algorithm,
                "Start-Request-Time": str(local_start_time),
                "Start-Get-Worker-Time": str(get_worker_start_time),
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
            id=plaintext_matrix_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            workers=max_workers,
            m=m,
            security_level=securitylevel,
        )
        logger.info(get_worker_entry.model_dump())

        worker_start_time = time.time()
        worker = RoryWorker(
            workerId=worker_id,
            port=port,
            session=s,
            algorithm=algorithm,
        )
        dm_shape = (r, r)

        encrypted_matrix_shape = (r, a, m)
        encrypted_matrix_dtype = "float32"
        run_headers = {
            "Plaintext-Matrix-Id": plaintext_matrix_id,
            "Request-Id": request_id,
            "Encrypted-Matrix-Id": encrypted_matrix_id,
            "Encrypted-Matrix-Shape": str(encrypted_matrix_shape),
            "Encrypted-Matrix-Dtype": encrypted_matrix_dtype,
            "Encrypted-Dm-Id": encrypted_dm_id,
            "Encrypted-Dm-Shape": str(dm_shape),
            "Encrypted-Dm-Dtype": "float32",
            "Num-Chunks": str(num_chunks),
            "M": str(m),
            "Encrypted-Threshold": str(encrypted_threshold),
            "Dm-Shape": str(dm_shape),
            "Dm-Dtype": "float32",
        }

        run1_response = worker.run(
            timeout=WORKER_TIMEOUT,
            headers=run_headers,
        )
        run1_response.raise_for_status()

        jsonWorkerResponse = run1_response.json()
        endTime = time.time()
        worker_service_time = jsonWorkerResponse["service_time"]
        label_vector = jsonWorkerResponse["label_vector"]
        response_time = endTime - local_start_time
        worker_end_time = time.time()
        worker_response_time = worker_end_time - worker_start_time

        clustering_completed_entry = ExperimentLogEntry(
            event="COMPLETED",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=local_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            workers=max_workers,
            m=m,
            security_level=securitylevel,
            dataowner_time=service_time_dataowner,
            manager_time=get_worker_service_time,
            worker_time=worker_response_time,
        )
        logger.info(clustering_completed_entry.model_dump())

        return {
            "label_vector": label_vector,
            "algorithm": algorithm,
            "worker_id": worker_id,
            "service_time_manager": get_worker_service_time,
            "service_time_worker": worker_response_time,
            "service_time_dataowner": service_time_dataowner,
            "response_time_clustering": response_time,
        }
    except Exception as e:
        logger.error({
            "msg": str(e)
        })
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# NNC
@router.post(
    "/nnc",
    response_model=NncResponse,
    summary="Plaintext Nearest Neighbor Clustering",
    description="Distributed Nearest Neighbor Clustering on plaintext data externalized to CSS.",
)
async def nnc(
    body: NncRequest,
    logger=Depends(get_logger),
    settings=Depends(get_settings),
    storage=Depends(get_storage_client),
    manager=Depends(get_manager),
    dataowner=Depends(get_dataowner),
    executor=Depends(get_executor),
):
    """
    This method implements a distributed version of the Nearest Neighbor Clustering
    algorithm. Unlike its secure counterpart (DBSNNC), this version operates on
    plaintext data externalized to the Cloud Storage System (CSS), focusing on
    performance and orchestration within the Rory platform architecture.

    Note:
    All identifiers for the input matrices and distance metrics are extracted from **HTTP Headers**.

    Attributes:
        Plaintext-Matrix-Id (str): Unique identifier for the matrix in CSS.
            Defaults to "matrix0".
        Plaintext-Matrix-Filename (str): Local filename (without extension).
            Defaults to "matrix-0".
        Extension (str): Dataset file extension (e.g., "csv"). Defaults to "csv".
        Threshold (float): Distance limit for clustering. If -1, it is calculated
            dynamically using platform utilities.
        Experiment-Id (str): Unique ID for performance tracking and logging.

    Returns:
        label_vector (list): Final cluster assignments for each data point.
        algorithm (str): "nnc".
        worker_id (str): ID of the worker node that processed the task.
        service_time_manager (float): Time spent coordinating with the Manager.
        service_time_worker (float): Time spent during Worker execution.
        service_time_dataowner (float): Time spent in local data preparation and IO.
        response_time_clustering (float): Total end-to-end execution time.

    Raises:
        Exception: Returns a 500 status code if the process executor is missing,
            or if failures occur during local I/O, CSS communication, or
            Worker interaction.
    """
    try:
        local_start_time = time.time()
        BUCKET_ID = settings.mictlanx_bucket_id
        TESTING = settings.testing
        SOURCE_PATH = settings.source_path
        max_workers = settings.max_workers
        num_chunks = settings.num_chunks
        if executor is None:
            raise HTTPException(status_code=500, detail="No process pool executor available")
        algorithm = Constants.ClusteringAlgorithms.NNC
        s = Session()
        plaintext_matrix_id = body.plaintext_matrix_id
        dm_id = "{}dm".format(plaintext_matrix_id)
        plaintext_matrix_filename = body.plaintext_matrix_filename
        extension = body.extension
        request_id = "request{}".format(plaintext_matrix_id)
        threshold = body.threshold
        plaintext_matrix_path = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_filename, extension)
        experiment_id = body.experiment_id
        WORKER_TIMEOUT = settings.worker_timeout
        MICTLANX_TIMEOUT = settings.mictlanx_timeout
        MICTLANX_MAX_RETRIES = settings.mictlanx_max_retries

        local_read_dataset_start_time = time.time()
        plaintext_matrix_result = await RoryCommon.read_numpy_from(
            path=plaintext_matrix_path,
            extension=extension,
        )

        if plaintext_matrix_result.is_ok:
            plaintext_matrix = plaintext_matrix_result.unwrap()
        else:
            raise plaintext_matrix_result.unwrap_err()

        r = plaintext_matrix.shape[0]
        a = plaintext_matrix.shape[1]

        local_read_entry = ExperimentLogEntry(
            event="LOCAL.READ",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=local_read_dataset_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id="",
            num_chunks=num_chunks,
        )
        logger.info(local_read_entry.model_dump())

        put_ptm_start_time = time.time()

        plaintext_matrix_chunks = Chunks.from_ndarray(
            ndarray=plaintext_matrix,
            group_id=plaintext_matrix_id,
            chunk_prefix=Some(plaintext_matrix_id),
            num_chunks=num_chunks,
        )

        if plaintext_matrix_chunks.is_none:
            raise HTTPException(status_code=500, detail="something went wrong creating the chunks")

        t_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
            client=storage,
            bucket_id=BUCKET_ID,
            key=plaintext_matrix_id,
            chunks=plaintext_matrix_chunks.unwrap(),
            timeout=MICTLANX_TIMEOUT,
            max_tries=MICTLANX_MAX_RETRIES,
            tags={
                "full_shape": str(plaintext_matrix.shape),
                "full_dtype": str(plaintext_matrix.dtype),
            },
        )

        put_encrypted_ptm_entry = ExperimentLogEntry(
            event="PUT",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=put_ptm_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id="",
            num_chunks=num_chunks,
        )
        logger.info(put_encrypted_ptm_entry.model_dump())

        dm_start_time = time.time()
        dm = dataowner.get_U(
            plaintext_matrix=plaintext_matrix,
            algorithm=algorithm,
        )

        dm_gen_entry = ExperimentLogEntry(
            event="DM.GENERATION",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=dm_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id="",
            num_chunks=num_chunks,
        )
        logger.info(dm_gen_entry.model_dump())

        if threshold == -1:
            threshold = RoryUtils.get_threshold(
                distance_matrix=dm,
            )

        put_ptm_start_time = time.time()
        maybe_dm_chunks = Chunks.from_ndarray(
            ndarray=dm,
            group_id=dm_id,
            chunk_prefix=Some(dm_id),
            num_chunks=num_chunks,
        )

        if maybe_dm_chunks.is_none:
            raise HTTPException(status_code=500, detail="something went wrong creating the chunks")

        put_dm_start_time = time.time()
        put_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
            client=storage,
            bucket_id=BUCKET_ID,
            key=dm_id,
            chunks=maybe_dm_chunks.unwrap(),
            timeout=MICTLANX_TIMEOUT,
            max_tries=MICTLANX_MAX_RETRIES,
            tags={
                "full_shape": str(dm.shape),
                "full_dtype": str(dm.dtype),
            },
        )

        service_time_dataowner = time.time() - local_start_time

        dm_put_entry = ExperimentLogEntry(
            event="PUT",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=put_dm_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id="",
            num_chunks=num_chunks,
        )
        logger.info(dm_put_entry.model_dump())

        get_worker_start_time = time.time()
        get_worker_result = manager.getWorker(
            headers={
                "Algorithm": algorithm,
                "Start-Request-Time": str(local_start_time),
                "Start-Get-Worker-Time": str(get_worker_start_time),
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
            id=plaintext_matrix_id,
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
        pm_shape = (r, a)
        dm_shape = (r, r)
        run_headers = {
            "Plaintext-Matrix-Id": plaintext_matrix_id,
            "Request-Id": request_id,
            "Num-Chunks": str(num_chunks),
            "Threshold": str(threshold),
            "Plaintext-Matrix-Shape": str(pm_shape),
            "Plaintext-Matrix-Dtype": str(plaintext_matrix.dtype),
            "Dm-Shape": str(dm_shape),
            "Dm-Dtype": str(dm.dtype),
        }

        worker_response = worker.run(
            timeout=WORKER_TIMEOUT,
            headers=run_headers,
        )

        worker_response.raise_for_status()
        jsonWorkerResponse = worker_response.json()
        end_time = time.time()
        worker_service_time = jsonWorkerResponse["service_time"]
        label_vector = jsonWorkerResponse["label_vector"]
        response_time = end_time - local_start_time
        worker_end_time = time.time()
        worker_response_time = worker_end_time - worker_start_time

        clustering_completed_entry = ExperimentLogEntry(
            event="COMPLETED",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=local_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            workers=max_workers,
            dataowner_time=service_time_dataowner,
            manager_time=get_worker_service_time,
            worker_time=worker_response_time,
        )
        logger.info(clustering_completed_entry.model_dump())

        return {
            "label_vector": label_vector,
            "algorithm": algorithm,
            "worker_id": worker_id,
            "service_time_manager": get_worker_service_time,
            "service_time_worker": worker_response_time,
            "service_time_dataowner": service_time_dataowner,
            "response_time_clustering": response_time,
        }
    except Exception as e:
        logger.error({
            "msg": str(e)
        })
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# PQC-SKMEANS
@router.post(
    "/pqc/skmeans",
    response_model=PqcSkmeansResponse,
    summary="Post-Quantum Secure K-Means clustering with CKKS",
    description="Clustering protocol using CKKS homomorphic encryption for post-quantum privacy-preserving data mining.",
)
async def pqc_skmeans(
    body: PqcSkmeansRequest,
    logger=Depends(get_logger),
    settings=Depends(get_settings),
    storage=Depends(get_storage_client),
    manager=Depends(get_manager),
    executor=Depends(get_executor),
):
    """
    This method implements a clustering protocol using the
    CKKS homomorphic encryption scheme. It is specifically designed
    for Post-Quantum Privacy-Preserving Data Mining as a Service (PPDMaaS), allowing
    complex floating-point computations on encrypted data while the Client
    retains the secret key.

    Note:
    **Post-Quantum Parameters**: Security levels and CKKS-specific metadata are passed via **HTTP Headers**.
    Body content will be ignored.

    Attributes:
        Plaintext-Matrix-Id (str): Unique ID for the matrix. Defaults to "matrix0".
        Plaintext-Matrix-Filename (str): Local file to be processed. Defaults to "matrix0".
        K (int): Number of clusters. **Required**.
        Max-Iterations (int): Maximum protocol rounds. Defaults to 10.
        Experiment-Id (str): Tracking ID for performance auditing.

    Returns:
        label_vector (list): Final cluster assignments for the dataset.
        iterations (int): Actual number of iterations performed.
        algorithm (str): "skmeans pqc".
        worker_id (str): ID of the node that performed the secure computations.
        service_time_manager (float): Time spent in Worker allocation.
        service_time_worker (float): Cumulative time of remote computation.
        service_time_dataowner (float): Total local time (Encryption/Decryption/IO).
        response_time_clustering (float): End-to-end execution time.


    Raises:
        Exception: Returns a 500 status code if the process executor is missing,
            CKKS context fails, or communication errors occur.
    """
    try:
        arrivalTime = time.time()
        BUCKET_ID = settings.mictlanx_bucket_id
        TESTING = settings.testing
        SOURCE_PATH = settings.source_path
        max_workers = settings.max_workers
        num_chunks = settings.num_chunks
        np_random = settings.np_random
        security_level = settings.liu_security_level

        if executor is None:
            raise HTTPException(status_code=500, detail="No process pool executor available")
        algorithm = Constants.ClusteringAlgorithms.SKMEANS_PQC
        s = Session()
        plaintext_matrix_id = body.plaintext_matrix_id
        encrypted_matrix_id = "encrypted{}".format(plaintext_matrix_id)
        udm_id = "{}udm".format(plaintext_matrix_id)
        plaintext_matrix_filename = body.plaintext_matrix_filename
        extension = body.extension
        k = body.k
        experiment_iteration = body.experiment_iteration
        experiment_id = body.experiment_id
        requestId = "request-{}".format(plaintext_matrix_id)
        plaintext_matrix_path = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_filename, extension)

        cent_i_id = "{}centi".format(plaintext_matrix_id)
        cent_j_id = "{}centj".format(plaintext_matrix_id)

        _round = settings.ckks_round
        decimals = settings.ckks_decimals
        path = settings.keys_path
        ctx_filename = settings.ctx_filename
        pubkey_filename = settings.pubkey_filename
        secretkey_filename = settings.secret_key_filename
        relinkey_filename = settings.relinkey_filename

        MAX_ITERATIONS = body.max_iterations
        WORKER_TIMEOUT = settings.worker_timeout
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

        local_read_dataset_start_time = time.time()
        plaintext_matrix_result = await RoryCommon.read_numpy_from(
            path=plaintext_matrix_path,
            extension=extension,
        )
        if plaintext_matrix_result.is_err:
            raise HTTPException(status_code=500, detail="Failed to local read plain text matrix.")
        plaintext_matrix = plaintext_matrix_result.unwrap()

        plaintext_matrix = plaintext_matrix.astype(np.float32)

        r = plaintext_matrix.shape[0]
        a = plaintext_matrix.shape[1]

        local_read_entry = ExperimentLogEntry(
            event="LOCAL.READ",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=local_read_dataset_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id="",
            num_chunks=num_chunks,
            k=k,
            workers=max_workers,
            security_level=security_level,
        )
        logger.info(local_read_entry.model_dump())

        max_workers = Utils.get_workers(num_chunks=num_chunks)

        encryption_start_time = time.time()
        n = a * r

        encrypted_matrix_chunks = RoryCommon.segment_and_encrypt_ckks_with_executor(
            executor=executor,
            key=encrypted_matrix_id,
            plaintext_matrix=plaintext_matrix,
            n=n,
            _round=_round,
            decimals=decimals,
            path=path,
            ctx_filename=ctx_filename,
            pubkey_filename=pubkey_filename,
            secretkey_filename=secretkey_filename,
            num_chunks=num_chunks,
            relinkey_filename=relinkey_filename,
        )

        segment_encrypt_entry = ExperimentLogEntry(
            event="SEGMENT.ENCRYPT",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=encryption_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id="",
            num_chunks=num_chunks,
            k=k,
            workers=max_workers,
            security_level=security_level,
        )
        logger.info(segment_encrypt_entry.model_dump())

        put_chunks_start_time = time.time()
        put_encrypted_matrix_result = await RoryCommon.delete_and_put_chunks(
            client=storage,
            bucket_id=BUCKET_ID,
            key=encrypted_matrix_id,
            chunks=encrypted_matrix_chunks,
            timeout=MICTLANX_TIMEOUT,
            max_tries=MICTLANX_MAX_RETRIES,
            tags={
                "full_shape": str((r, a)),
                "full_dtype": "float32",
            },
        )

        put_encrypted_ptm_entry = ExperimentLogEntry(
            event="PUT",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=put_chunks_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id="",
            num_chunks=num_chunks,
            k=k,
            workers=max_workers,
            security_level=security_level,
        )
        logger.info(put_encrypted_ptm_entry.model_dump())

        udm_start_time = time.time()
        udm = dataowner.get_U(
            plaintext_matrix=plaintext_matrix,
            algorithm=algorithm,
        )

        udm_gen_entry = ExperimentLogEntry(
            event="UDM.GENERATION",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=udm_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id="",
            num_chunks=num_chunks,
            k=k,
            workers=max_workers,
            security_level=security_level,
        )
        logger.info(udm_gen_entry.model_dump())

        udm_put_start_time = time.time()
        maybe_udm_matrix_chunks = Chunks.from_ndarray(
            ndarray=udm,
            group_id=udm_id,
            chunk_prefix=Some(udm_id),
            num_chunks=num_chunks,
        )

        if maybe_udm_matrix_chunks.is_none:
            error = "Something went wrong creating the UDM chunks"
            logger.error(error)
            raise HTTPException(status_code=500, detail=error)

        udm_put_result = await RoryCommon.delete_and_put_chunks(
            client=storage,
            bucket_id=BUCKET_ID,
            key=udm_id,
            chunks=maybe_udm_matrix_chunks.unwrap(),
            timeout=MICTLANX_TIMEOUT,
            max_tries=MICTLANX_MAX_RETRIES,
            tags={
                "full_shape": str(udm.shape),
                "full_dtype": str(udm.dtype),
            },
        )
        if udm_put_result.is_err:
            error = udm_put_result.unwrap_err()
            e = f"Failed to put the udm: {error}"
            raise HTTPException(status_code=500, detail=e)

        service_time_dataowner = time.time() - arrivalTime

        udm_put_entry = ExperimentLogEntry(
            event="PUT",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=udm_put_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id="",
            num_chunks=num_chunks,
            k=k,
            workers=max_workers,
            security_level=security_level,
        )
        logger.info(udm_put_entry.model_dump())

        init_sm_id = "{}initsm".format(plaintext_matrix_id)
        zero_shiftmatrix = np.zeros((k, a))
        n2 = a * k
        encrypt_ckks_start_time = time.time()
        encrypted_zero_shiftmatrix_chunks = RoryCommon.segment_and_encrypt_ckks_with_executor(
            executor=executor,
            key=init_sm_id,
            plaintext_matrix=zero_shiftmatrix,
            n=n2,
            num_chunks=num_chunks,
            _round=_round,
            decimals=decimals,
            path=path,
            ctx_filename=ctx_filename,
            pubkey_filename=pubkey_filename,
            secretkey_filename=secretkey_filename,
            relinkey_filename=relinkey_filename,
        )

        encrypt_entry = ExperimentLogEntry(
            event="ENCRYPT",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=encrypt_ckks_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id="",
            num_chunks=num_chunks,
            k=k,
            workers=max_workers,
            security_level=security_level,
        )
        logger.info(encrypt_entry.model_dump())

        put_chunks_start_time = time.time()
        put_encrypted_matrix_result = await RoryCommon.delete_and_put_chunks(
            client=storage,
            bucket_id=BUCKET_ID,
            key=init_sm_id,
            chunks=encrypted_zero_shiftmatrix_chunks,
            timeout=MICTLANX_TIMEOUT,
            max_tries=MICTLANX_MAX_RETRIES,
            tags={
                "full_shape": str((k, a)),
                "full_dtype": "float32",
            },
        )
        if put_encrypted_matrix_result.is_err:
            e = f"Failed put chunks: {put_encrypted_matrix_result.unwrap_err()}"
            logger.error(e)
            raise HTTPException(status_code=500, detail=e)

        udm_put_entry = ExperimentLogEntry(
            event="PUT",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=put_chunks_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id="",
            num_chunks=num_chunks,
            k=k,
            workers=max_workers,
            security_level=security_level,
        )
        logger.info(udm_put_entry.model_dump())

        get_worker_start_time = time.time()
        get_worker_result = manager.getWorker(
            headers={
                "Algorithm": algorithm,
                "Start-Request-Time": str(arrivalTime),
                "Start-Get-Worker-Time": str(get_worker_start_time),
            }
        )
        if get_worker_result.is_err:
            error = get_worker_result.unwrap_err()
            logger.error(str(error))
            raise HTTPException(status_code=500, detail=str(error))
        (worker_id, port) = get_worker_result.unwrap()

        get_worker_end_time = time.time()
        get_worker_service_time = get_worker_end_time - get_worker_start_time
        worker_id = "localhost" if TESTING else worker_id

        get_worker_entry = ExperimentLogEntry(
            event="GET.WORKER",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=get_worker_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            k=k,
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
        status_val = Constants.ClusteringStatus.START
        worker_run1_response = None

        interaction_arrival_time = time.time()
        iterations = 0
        label_vector = None

        while (status_val != Constants.ClusteringStatus.COMPLETED):

            inner_interaction_arrival_time = time.time()
            run1_headers = {
                "Step-Index": "1",
                "Clustering-Status": str(status_val),
                "Plaintext-Matrix-Id": plaintext_matrix_id,
                "Request-Id": requestId,
                "Encrypted-Matrix-Id": encrypted_matrix_id,
                "Encrypted-Matrix-Shape": "({},{})".format(r, a),
                "Encrypted-Matrix-Dtype": "float32",
                "Encrypted-Udm-Dtype": "float32",
                "Num-Chunks": str(num_chunks),
                "Iterations": str(iterations),
                "K": str(k),
                "Experiment-Iteration": str(experiment_iteration),
                "Max-Iterations": str(MAX_ITERATIONS),
            }

            worker_run1_response = worker.run(
                timeout=WORKER_TIMEOUT,
                headers=run1_headers,
            )
            worker_run1_status = worker_run1_response.status_code

            if worker_run1_status != 200:
                raise HTTPException(status_code=500, detail="Worker error: {}".format(worker_run1_response.content))

            worker_run1_response.raise_for_status()
            jsonWorkerResponse = worker_run1_response.json()
            encrypted_shift_matrix_id = jsonWorkerResponse["encrypted_shift_matrix_id"]
            run1_service_time = jsonWorkerResponse["service_time"]
            run1_n_iterations = jsonWorkerResponse["n_iterations"]
            label_vector = jsonWorkerResponse["label_vector"]

            run1_worker_entry = ExperimentLogEntry(
                event="RUN1",
                experiment_id=experiment_id,
                algorithm=algorithm,
                start_time=inner_interaction_arrival_time,
                end_time=time.time(),
                id=plaintext_matrix_id,
                worker_id=worker_id,
                num_chunks=num_chunks,
                k=k,
                workers=max_workers,
                security_level=security_level,
                iterations=run1_n_iterations,
            )
            logger.info(run1_worker_entry.model_dump())

            encrypted_shift_matrix_start_time = time.time()
            encrypted_shift_matrix = await RoryCommon.get_pyctxt(
                client=storage,
                bucket_id=BUCKET_ID,
                key=encrypted_shift_matrix_id,
                ckks=ckks,
                force=True,
                backoff_factor=MICTLANX_BACKOFF_FACTOR,
                delay=MICTLANX_DELAY,
                max_retries=MICTLANX_MAX_RETRIES,
            )

            get_encrypted_sm_entry = ExperimentLogEntry(
                event="GET",
                experiment_id=experiment_id,
                algorithm=algorithm,
                start_time=encrypted_shift_matrix_start_time,
                end_time=time.time(),
                id=plaintext_matrix_id,
                worker_id=worker_id,
                num_chunks=num_chunks,
                k=k,
                workers=max_workers,
                security_level=security_level,
                iterations=run1_n_iterations,
            )
            logger.info(get_encrypted_sm_entry.model_dump())

            decrypt_start_time = time.time()
            shift_matrix = ckks.decryptMatrix(
                ciphertext_matrix=encrypted_shift_matrix,
                shape=[k, a],
            )

            decrypt_entry = ExperimentLogEntry(
                event="DECRYPT",
                experiment_id=experiment_id,
                algorithm=algorithm,
                start_time=decrypt_start_time,
                end_time=time.time(),
                id=plaintext_matrix_id,
                worker_id=worker_id,
                num_chunks=num_chunks,
                k=k,
                workers=max_workers,
                security_level=security_level,
                iterations=run1_n_iterations,
            )
            logger.info(decrypt_entry.model_dump())

            shift_matrix_id = "{}shiftmatrix".format(plaintext_matrix_id)
            put_shift_matrix_start_time = time.time()

            shift_matrix_chunks = Chunks.from_ndarray(
                ndarray=shift_matrix,
                group_id=shift_matrix_id,
                chunk_prefix=Some(shift_matrix_id),
                num_chunks=num_chunks,
            )
            if shift_matrix_chunks.is_none:
                raise HTTPException(status_code=500, detail="something went wrong creating the chunks")

            put_shift_matrix_result = await RoryCommon.delete_and_put_chunks(
                client=storage,
                bucket_id=BUCKET_ID,
                key=shift_matrix_id,
                chunks=shift_matrix_chunks.unwrap(),
                timeout=MICTLANX_TIMEOUT,
                max_tries=MICTLANX_MAX_RETRIES,
                tags={
                    "full_shape": str(shift_matrix.shape),
                    "full_dtype": str(shift_matrix.dtype),
                },
            )
            if put_shift_matrix_result.is_err:
                raise HTTPException(status_code=500, detail="Failed to put shiftmatrix")

            put_sm_entry = ExperimentLogEntry(
                event="PUT",
                experiment_id=experiment_id,
                algorithm=algorithm,
                start_time=put_shift_matrix_start_time,
                end_time=time.time(),
                id=plaintext_matrix_id,
                worker_id=worker_id,
                num_chunks=num_chunks,
                k=k,
                workers=max_workers,
                security_level=security_level,
                iterations=run1_n_iterations,
            )
            logger.info(put_sm_entry.model_dump())

            Cent_i = await RoryCommon.get_pyctxt(
                client=storage,
                bucket_id=BUCKET_ID,
                key=cent_i_id,
                ckks=ckks,
            )
            Cent_j = await RoryCommon.get_pyctxt(
                client=storage,
                bucket_id=BUCKET_ID,
                key=cent_j_id,
                ckks=ckks,
            )
            decrypted_cent_i = ckks.decryptMatrix(
                ciphertext_matrix=Cent_i,
                shape=[1, k],
            )

            decrypted_cent_j = ckks.decryptMatrix(
                ciphertext_matrix=Cent_j,
                shape=[1, k],
            )
            min_error = 0.15
            isZero = Utils.verify_mean_error(
                old_matrix=decrypted_cent_i,
                new_matrix=decrypted_cent_j,
                min_error=min_error,
            )

            status_val = Constants.ClusteringStatus.WORK_IN_PROGRESS
            run2_headers = {
                "Step-Index": "2",
                "Clustering-Status": str(status_val),
                "Shift-Matrix-Id": shift_matrix_id,
                "Plaintext-Matrix-Id": plaintext_matrix_id,
                "Encrypted-Matrix-Id": encrypted_matrix_id,
                "Encrypted-Matrix-Shape": "({},{})".format(r, a),
                "Encrypted-Matrix-Dtype": "float32",
                "Num-Chunks": str(num_chunks),
                "Iterations": str(iterations),
                "K": str(k),
                "Experiment-Iteration": str(experiment_iteration),
                "Max-Iterations": str(MAX_ITERATIONS),
                "Is-Zero": str(int(isZero)),
            }

            worker_run2_response = worker.run(
                timeout=WORKER_TIMEOUT,
                headers=run2_headers,
            )

            worker_run2_response.raise_for_status()
            service_time_worker = worker_run2_response.headers.get("Service-Time", 0)
            iterations += 1
            if (iterations >= MAX_ITERATIONS):
                status_val = Constants.ClusteringStatus.COMPLETED
                startTime = float(s.headers.get("Start-Time", 0))
                service_time_worker = time.time() - startTime
            else:
                status_val = int(worker_run2_response.headers.get("Clustering-Status", Constants.ClusteringStatus.WORK_IN_PROGRESS))
            endTime = time.time()

            skmeans_iteration_completed_entry = ExperimentLogEntry(
                event="ITERATION.COMPLETED",
                experiment_id=experiment_id,
                algorithm=algorithm,
                start_time=inner_interaction_arrival_time,
                end_time=time.time(),
                id=plaintext_matrix_id,
                worker_id=worker_id,
                num_chunks=num_chunks,
                k=k,
                workers=max_workers,
                security_level=security_level,
                iterations=run1_n_iterations,
            )
            logger.info(skmeans_iteration_completed_entry.model_dump())

        interaction_end_time = time.time()
        interaction_service_time = interaction_end_time - interaction_arrival_time
        worker_response_time = endTime - worker_start_time
        response_time = endTime - arrivalTime

        clustering_completed_entry = ExperimentLogEntry(
            event="COMPLETED",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=arrivalTime,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            k=k,
            workers=max_workers,
            security_level=security_level,
            iterations=iterations,
            dataowner_time=service_time_dataowner,
            manager_time=get_worker_service_time,
            worker_time=worker_response_time,
        )
        logger.info(clustering_completed_entry.model_dump())

        return {
            "label_vector": label_vector,
            "iterations": iterations,
            "algorithm": algorithm,
            "worker_id": worker_id,
            "service_time_manager": get_worker_service_time,
            "service_time_worker": worker_response_time,
            "service_time_dataowner": service_time_dataowner,
            "response_time_clustering": response_time,
        }
    except Exception as e:
        logger.error({
            "msg": str(e)
        })
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# PQC-DBSKMEANS
@router.post(
    "/pqc/dbskmeans",
    response_model=PqcDbskmeansResponse,
    summary="Post-Quantum Double-Blind Secure K-Means clustering",
    description="Hybrid secure protocol combining CKKS for data protection and FDHOPE for secure distance matrix updates.",
)
async def pqc_dbskmeans(
    body: PqcDbskmeansRequest,
    logger=Depends(get_logger),
    settings=Depends(get_settings),
    storage=Depends(get_storage_client),
    manager=Depends(get_manager),
    executor=Depends(get_executor),
    dataowner=Depends(get_dataowner),
):
    """
    This method achieves a "Double-Blind" state by combining CKKS for data protection and FDHOPE for secure
    distance matrix updates.

    Note:
    **Hybrid Secure Protocol**: Combines post-quantum security with double-blind logic.
     Mandatory parameters are required in the **HTTP Headers**.

    Attributes:
        Plaintext-Matrix-Id (str): Unique ID for CSS storage. Defaults to "matrix0".
        Plaintext-Matrix-Filename (str): Local dataset name. Defaults to "matrix0".
        K (int): Number of clusters. **Required**.
        Sens (float): Sensitivity for FDHOPE. Defaults to 0.00000001.
        Max-Iterations (int): Maximum protocol rounds. Defaults to 10.
        Experiment-Id (str): Tracking ID for performance auditing.

    Returns:
        label_vector (list): Final cluster assignments for the dataset.
        iterations (int): Actual number of iterations performed.
        algorithm (str): "dbskmeans pqc".
        worker_id (str): ID of the node that performed the secure computations.
        service_time_manager (float): Time spent in Worker allocation.
        service_time_worker (float): Cumulative time of remote computation.
        service_time_dataowner (float): Total local time (Encryption/Decryption/IO).
        response_time_clustering (float): End-to-end execution time.

    Raises:
        Exception: Returns a 500 status code if failures occur in the hybrid
            encryption chain (CKKS/FDHOPE), CSS I/O, or Worker orchestration.
    """
    try:
        arrivalTime = time.time()
        BUCKET_ID = settings.mictlanx_bucket_id
        TESTING = settings.testing
        SOURCE_PATH = settings.source_path
        max_workers = settings.max_workers
        num_chunks = settings.num_chunks
        np_random = settings.np_random
        do_fdhope = dataowner
        if executor is None:
            raise HTTPException(status_code=500, detail="No process pool executor available")
        algorithm = Constants.ClusteringAlgorithms.DBSKMEANS_PQC
        algorithm_fdhope = Constants.ClusteringAlgorithms.DBSKMEANS
        s = Session()
        security_level = settings.liu_security_level
        plaintext_matrix_id = body.plaintext_matrix_id
        encrypted_matrix_id = "encrypted{}".format(plaintext_matrix_id)
        encrypted_udm_id = "{}encryptedudm".format(plaintext_matrix_id)
        plaintext_matrix_filename = body.plaintext_matrix_filename
        extension = body.extension
        experiment_id = body.experiment_id
        k = body.k
        sens = body.sens
        experiment_iteration = body.experiment_iteration
        request_id = "request{}".format(plaintext_matrix_id)
        plaintext_matrix_path = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_filename, extension)

        init_sm_id = "{}initsm".format(plaintext_matrix_id)
        cent_i_id = "{}centi".format(plaintext_matrix_id)
        cent_j_id = "{}centj".format(plaintext_matrix_id)

        _round = settings.ckks_round
        decimals = settings.ckks_decimals
        path = settings.keys_path
        ctx_filename = settings.ctx_filename
        pubkey_filename = settings.pubkey_filename
        secretkey_filename = settings.secret_key_filename
        relinkey_filename = settings.relinkey_filename

        MAX_ITERATIONS = body.max_iterations
        WORKER_TIMEOUT = settings.worker_timeout
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
        dataowner_pqc = DataOwnerPQC(scheme=ckks, sens=sens)

        local_read_dataset_start_time = time.time()
        plaintext_matrix_result = await RoryCommon.read_numpy_from(
            path=plaintext_matrix_path,
            extension=extension,
        )
        if plaintext_matrix_result.is_ok:
            plaintext_matrix = plaintext_matrix_result.unwrap()
        else:
            raise plaintext_matrix_result.unwrap_err()

        plaintext_matrix = plaintext_matrix.astype(np.float32)

        r = plaintext_matrix.shape[0]
        a = plaintext_matrix.shape[1]
        max_workers = Utils.get_workers(num_chunks=num_chunks)

        local_read_entry = ExperimentLogEntry(
            event="LOCAL.READ",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=local_read_dataset_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id="",
            num_chunks=num_chunks,
            k=k,
            workers=max_workers,
            security_level=security_level,
        )
        logger.info(local_read_entry.model_dump())

        encryption_start_time = time.time()
        n = a * r
        encrypted_matrix_chunks = RoryCommon.segment_and_encrypt_ckks_with_executor(
            executor=executor,
            key=encrypted_matrix_id,
            plaintext_matrix=plaintext_matrix,
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
            start_time=encryption_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id="",
            num_chunks=num_chunks,
            k=k,
            workers=max_workers,
            security_level=security_level,
        )
        logger.info(segment_encrypt_entry.model_dump())

        put_chunks_start_time = time.time()

        put_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
            client=storage,
            bucket_id=BUCKET_ID,
            key=encrypted_matrix_id,
            chunks=encrypted_matrix_chunks,
            timeout=MICTLANX_TIMEOUT,
            max_tries=MICTLANX_MAX_RETRIES,
            tags={
                "full_shape": str((r, a)),
                "full_dtype": "float32",
            },
        )
        if put_chunks_generator_results.is_err:
            raise HTTPException(status_code=500, detail="Failed to put encrypted matrix")

        put_encrypted_ptm_entry = ExperimentLogEntry(
            event="PUT",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=put_chunks_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id="",
            num_chunks=num_chunks,
            k=k,
            workers=max_workers,
            security_level=security_level,
        )
        logger.info(put_encrypted_ptm_entry.model_dump())

        udm_start_time = time.time()
        udm = do_fdhope.get_U(
            plaintext_matrix=plaintext_matrix,
            algorithm=algorithm_fdhope,
        )

        udm_shape = udm.shape
        udm_dtype = udm.dtype
        del plaintext_matrix

        udm_gen_entry = ExperimentLogEntry(
            event="UDM.GENERATION",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=udm_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id="",
            num_chunks=num_chunks,
            k=k,
            workers=max_workers,
            security_level=security_level,
        )
        logger.info(udm_gen_entry.model_dump())

        n = r * r * a
        segment_encrypt_fdhope_start_time = time.time()
        encrypted_matrix_UDM_chunks = RoryCommon.segment_and_encrypt_fdhope_with_executor(
            executor=executor,
            algorithm=algorithm_fdhope,
            key=encrypted_udm_id,
            dataowner=do_fdhope,
            matrix=udm,
            n=n,
            num_chunks=num_chunks,
            sens=sens,
        )

        put_encrypted_ptm_entry = ExperimentLogEntry(
            event="SEGMENT.ENCRYPT",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=segment_encrypt_fdhope_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id="",
            num_chunks=num_chunks,
            k=k,
            workers=max_workers,
            security_level=security_level,
        )
        logger.info(put_encrypted_ptm_entry.model_dump())

        put_chunks_start_time = time.time()
        put_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
            client=storage,
            bucket_id=BUCKET_ID,
            key=encrypted_udm_id,
            chunks=encrypted_matrix_UDM_chunks,
            timeout=MICTLANX_TIMEOUT,
            max_tries=MICTLANX_MAX_RETRIES,
            tags={
                "full_shape": str((r, r, a)),
                "full_dtype": "float32",
            },
        )

        if put_chunks_generator_results.is_err:
            raise HTTPException(status_code=500, detail="Failed to put encrypted udm matrix")

        put_encrypted_ptm_entry = ExperimentLogEntry(
            event="PUT",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=put_chunks_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id="",
            num_chunks=num_chunks,
            k=k,
            workers=max_workers,
            security_level=security_level,
        )
        logger.info(put_encrypted_ptm_entry.model_dump())

        service_time_dataowner = time.time() - arrivalTime

        del udm
        del encrypted_matrix_UDM_chunks
        zero_shiftmatrix = np.zeros((k, a))
        n2 = a * k

        init_shiftmatrix_start_time = time.time()

        encrypted_shiftmatrix_chunks = RoryCommon.segment_and_encrypt_ckks_with_executor(
            executor=executor,
            key=init_sm_id,
            plaintext_matrix=zero_shiftmatrix,
            n=n2,
            num_chunks=num_chunks,
            _round=_round,
            decimals=decimals,
            path=path,
            ctx_filename=ctx_filename,
            pubkey_filename=pubkey_filename,
            secretkey_filename=secretkey_filename,
            relinkey_filename=relinkey_filename,
        )

        encrypt_sm_entry = ExperimentLogEntry(
            event="ENCRYPT",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=init_shiftmatrix_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id="",
            num_chunks=num_chunks,
            k=k,
            workers=max_workers,
            security_level=security_level,
        )
        logger.info(encrypt_sm_entry.model_dump())

        put_chunks_start_time = time.time()

        put_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
            client=storage,
            bucket_id=BUCKET_ID,
            key=init_sm_id,
            chunks=encrypted_shiftmatrix_chunks,
            timeout=MICTLANX_TIMEOUT,
            max_tries=MICTLANX_MAX_RETRIES,
            tags={
                "shape": str((k, a)),
                "dtype": "float32",
            },
        )
        if put_chunks_generator_results.is_err:
            raise HTTPException(status_code=500, detail="Failed to put encrypted init shift matrix")

        put_encrypted_ptm_entry = ExperimentLogEntry(
            event="PUT",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=put_chunks_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id="",
            num_chunks=num_chunks,
            k=k,
            workers=max_workers,
            security_level=security_level,
        )
        logger.info(put_encrypted_ptm_entry.model_dump())

        get_worker_start_time = time.time()
        get_worker_result = manager.getWorker(
            headers={
                "Algorithm": algorithm,
                "Start-Request-Time": str(arrivalTime),
                "Start-Get-Worker-Time": str(get_worker_start_time),
            }
        )
        if get_worker_result.is_err:
            error = get_worker_result.unwrap_err()
            logger.error(str(error))
            raise HTTPException(status_code=500, detail=str(error))
        (worker_id, port) = get_worker_result.unwrap()

        get_worker_end_time = time.time()
        get_worker_service_time = get_worker_end_time - get_worker_start_time
        worker_id = "localhost" if TESTING else worker_id

        get_worker_entry = ExperimentLogEntry(
            event="GET.WORKER",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=get_worker_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            k=k,
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
        status_val = Constants.ClusteringStatus.START
        worker_run1_response = None
        initial_encrypted_udm_shape = (r, r, a)
        interaction_arrival_time = time.time()
        iterations = 0
        label_vector = None

        while (status_val != Constants.ClusteringStatus.COMPLETED):

            inner_interaction_arrival_time = time.time()
            run1_headers = {
                "Step-Index": "1",
                "Clustering-Status": str(status_val),
                "Plaintext-Matrix-Id": plaintext_matrix_id,
                "Request-Id": request_id,
                "Encrypted-Matrix-Id": encrypted_matrix_id,
                "Encrypted-Matrix-Shape": "({},{})".format(r, a),
                "Encrypted-Matrix-Dtype": "float32",
                "Encrypted-Udm-Dtype": "float32",
                "Encrypted-Udm-Shape": str(initial_encrypted_udm_shape),
                "Num-Chunks": str(num_chunks),
                "Iterations": str(iterations),
                "K": str(k),
                "Experiment-Iteration": str(experiment_iteration),
                "Max-Iterations": str(MAX_ITERATIONS),
            }

            worker_run1_response = worker.run(
                timeout=WORKER_TIMEOUT,
                headers=run1_headers,
            )

            logger.info("worker response {}".format(worker_run1_response))
            worker_run1_status = worker_run1_response.status_code

            if worker_run1_status != 200:
                raise HTTPException(status_code=500, detail="Worker error: {}".format(worker_run1_response.content))

            worker_run1_response.raise_for_status()
            jsonWorkerResponse = worker_run1_response.json()
            encrypted_shift_matrix_id = jsonWorkerResponse["encrypted_shift_matrix_id"]
            run1_service_time = jsonWorkerResponse["service_time"]
            run1_n_iterations = jsonWorkerResponse["n_iterations"]
            label_vector = jsonWorkerResponse["label_vector"]

            run1_worker_entry = ExperimentLogEntry(
                event="RUN1",
                experiment_id=experiment_id,
                algorithm=algorithm,
                start_time=inner_interaction_arrival_time,
                end_time=time.time(),
                id=plaintext_matrix_id,
                worker_id=worker_id,
                num_chunks=num_chunks,
                k=k,
                workers=max_workers,
                security_level=security_level,
                iterations=run1_n_iterations,
            )
            logger.info(run1_worker_entry.model_dump())

            encrypted_shift_matrix = await RoryCommon.get_pyctxt(
                client=storage,
                bucket_id=BUCKET_ID,
                key=encrypted_shift_matrix_id,
                ckks=ckks,
                delay=MICTLANX_DELAY,
                max_retries=MICTLANX_MAX_RETRIES,
                backoff_factor=MICTLANX_BACKOFF_FACTOR,
                force=True,
            )

            decrypt_start_time = time.time()
            shift_matrix = ckks.decryptMatrix(
                ciphertext_matrix=encrypted_shift_matrix,
                shape=[k, a],
            )

            encrypted_start_time = time.time()
            shift_matrix_ope_res = Fdhope.encryptMatrix(
                plaintext_matrix=shift_matrix,
                messagespace=do_fdhope.messageIntervals,
                cipherspace=do_fdhope.cypherIntervals,
                sens=sens,
            )

            shift_matrix_ope = shift_matrix_ope_res.matrix
            shift_matrix_ope_shape = shift_matrix_ope.shape
            shift_matrix_ope_dtype = shift_matrix_ope.dtype

            get_encrypted_sm_entry = ExperimentLogEntry(
                event="ENCRYPT",
                experiment_id=experiment_id,
                algorithm=algorithm,
                start_time=encrypted_start_time,
                end_time=time.time(),
                id=plaintext_matrix_id,
                worker_id=worker_id,
                num_chunks=num_chunks,
                k=k,
                workers=max_workers,
                security_level=security_level,
                iterations=run1_n_iterations,
            )
            logger.info(get_encrypted_sm_entry.model_dump())

            shift_matrix_id = "{}shiftmatrix".format(plaintext_matrix_id)
            shift_matrix_ope_id = "{}shiftmatrixope".format(plaintext_matrix_id)

            put_matrix_start_time = time.time()

            maybe_shift_matrix_chunks = Chunks.from_ndarray(
                ndarray=shift_matrix_ope,
                group_id=shift_matrix_ope_id,
                chunk_prefix=Some(shift_matrix_ope_id),
                num_chunks=num_chunks,
            )

            if maybe_shift_matrix_chunks.is_none:
                raise HTTPException(status_code=500, detail="something went wrong creating the chunks")

            encrypted_sm_ope_result = await RoryCommon.delete_and_put_chunks(
                client=storage,
                bucket_id=BUCKET_ID,
                key=shift_matrix_ope_id,
                chunks=maybe_shift_matrix_chunks.unwrap(),
                timeout=MICTLANX_TIMEOUT,
                max_tries=MICTLANX_MAX_RETRIES,
                tags={
                    "full_shape": str(shift_matrix_ope_shape),
                    "full_dtype": str(shift_matrix_ope_dtype),
                },
            )

            del maybe_shift_matrix_chunks
            del shift_matrix_ope
            if encrypted_sm_ope_result.is_err:
                raise HTTPException(status_code=500, detail="Failed to put encrypted shiftmatrix ope")

            put_sm_entry = ExperimentLogEntry(
                event="PUT",
                experiment_id=experiment_id,
                algorithm=algorithm,
                start_time=put_matrix_start_time,
                end_time=time.time(),
                id=plaintext_matrix_id,
                worker_id=worker_id,
                num_chunks=num_chunks,
                k=k,
                workers=max_workers,
                security_level=security_level,
                iterations=run1_n_iterations,
            )
            logger.info(put_sm_entry.model_dump())

            Cent_i = await RoryCommon.get_pyctxt(
                client=storage,
                bucket_id=BUCKET_ID,
                key=cent_i_id,
                delay=MICTLANX_DELAY,
                backoff_factor=MICTLANX_BACKOFF_FACTOR,
                force=True,
                max_retries=MICTLANX_MAX_RETRIES,
                ckks=ckks,
            )

            Cent_j = await RoryCommon.get_pyctxt(
                client=storage,
                ckks=ckks,
                bucket_id=BUCKET_ID,
                key=cent_j_id,
                delay=MICTLANX_DELAY,
                backoff_factor=MICTLANX_BACKOFF_FACTOR,
                force=True,
                max_retries=MICTLANX_MAX_RETRIES,
            )

            decrypted_cent_i = ckks.decryptMatrix(
                ciphertext_matrix=Cent_i,
                shape=[1, k],
            )

            decrypted_cent_j = ckks.decryptMatrix(
                ciphertext_matrix=Cent_j,
                shape=[1, k],
            )

            min_error = 0.15

            isZero = Utils.verify_mean_error(
                old_matrix=decrypted_cent_i,
                new_matrix=decrypted_cent_j,
                min_error=min_error,
            )

            status_val = Constants.ClusteringStatus.WORK_IN_PROGRESS
            run2_headers = {
                "Step-Index": "2",
                "Clustering-Status": str(status_val),
                "Shift-Matrix-Id": shift_matrix_id,
                "Plaintext-Matrix-Id": plaintext_matrix_id,
                "Encrypted-Matrix-Id": encrypted_matrix_id,
                "Encrypted-Matrix-Shape": "({},{})".format(r, a),
                "Encrypted-Matrix-Dtype": "float32",
                "Encrypted-Udm-Dtype": "float32",
                "Encrypted-Udm-Shape": str(initial_encrypted_udm_shape),
                "Num-Chunks": str(num_chunks),
                "Iterations": str(iterations),
                "K": str(k),
                "Experiment-Iteration": str(experiment_iteration),
                "Max-Iterations": str(MAX_ITERATIONS),
                "Is-Zero": str(int(isZero)),
            }

            worker_run2_response = worker.run(
                timeout=WORKER_TIMEOUT,
                headers=run2_headers,
            )

            worker_run2_response.raise_for_status()
            service_time_worker = worker_run2_response.headers.get("Service-Time", 0)
            iterations += 1
            if (iterations >= MAX_ITERATIONS):
                status_val = Constants.ClusteringStatus.COMPLETED
                startTime = float(s.headers.get("Start-Time", 0))
                service_time_worker = time.time() - startTime
            else:
                status_val = int(worker_run2_response.headers.get("Clustering-Status", Constants.ClusteringStatus.WORK_IN_PROGRESS))
            endTime = time.time()

            dbskmeans_iteration_completed_entry = ExperimentLogEntry(
                event="ITERATION.COMPLETED",
                experiment_id=experiment_id,
                algorithm=algorithm,
                start_time=inner_interaction_arrival_time,
                end_time=time.time(),
                id=plaintext_matrix_id,
                worker_id=worker_id,
                num_chunks=num_chunks,
                k=k,
                workers=max_workers,
                security_level=security_level,
                iterations=run1_n_iterations,
            )
            logger.info(dbskmeans_iteration_completed_entry.model_dump())

        worker_response_time = endTime - worker_start_time
        response_time = endTime - arrivalTime

        clustering_completed_entry = ExperimentLogEntry(
            event="COMPLETED",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=arrivalTime,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            k=k,
            workers=max_workers,
            security_level=security_level,
            iterations=iterations,
            dataowner_time=service_time_dataowner,
            manager_time=get_worker_service_time,
            worker_time=worker_response_time,
        )
        logger.info(clustering_completed_entry.model_dump())

        return {
            "label_vector": label_vector,
            "iterations": iterations,
            "algorithm": algorithm,
            "worker_id": worker_id,
            "service_time_manager": get_worker_service_time,
            "service_time_worker": worker_response_time,
            "service_time_dataowner": service_time_dataowner,
            "response_time_clustering": response_time,
        }

    except Exception as e:
        logger.error({
            "msg": str(e)
        })
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
