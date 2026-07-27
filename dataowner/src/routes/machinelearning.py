import os
import time
import gc
import numpy as np
from uuid import uuid4
from requests import Session
from fastapi import APIRouter, Depends, HTTPException
from rory.core.interfaces.rorymanager import RoryManager
from rory.core.interfaces.roryworker import RoryWorker
from rory.core.utils.constants import Constants
from rorycommon import Common as RoryCommon
from rorycommon import StorageBuilder, StorageParams, Scheme, CkksParams
from mictlanx import AsyncClient
from concurrent.futures import ProcessPoolExecutor
from models.experiment import ExperimentLogEntry
from rory.core.security.cryptosystem.pqc.ckks import Ckks, CkksModes
from dependencies import (
    get_settings,
    get_logger,
    get_storage_client,
    get_manager,
    get_executor,
    get_ckks,
)
from config import Settings
from models.requests.machinelearning import (
    LRTrainRequest,
    LRPredictRequest,
    PPLRTrainRequest,
    PPLRPredictRequest,
)
from models.responses.machinelearning import (
    LRTrainResponse,
    LRPredictResponse,
    PPLRTrainResponse,
    PPLRPredictResponse,
)
from models.responses.clustering import HealthCheckResponse

router = APIRouter(prefix="/machine-learning", tags=["Machine Learning"])


@router.api_route(
    "/test",
    methods=["GET", "POST"],
    response_model=HealthCheckResponse,
    summary="Health check",
    description="Diagnostic and health check endpoint for the machine-learning component.",
)
def test():
    """Diagnostic and health check endpoint for the logisticregression component.

    This method provides a simple mechanism to verify that the
    logisticregression routes are active and reachable. It is primarily used
    by the Rory platform's orchestration layer to identify the node type
    and ensure proper network synchronization before initiating machine
    learning workflows.

    Returns:
        component_type (str): "dataowner".

    Status Code:
        200: If the logisticregression service is operational.
    """
    return {"component_type": "dataowner"}


@router.post(
    "/logistic-regression/train",
    response_model=LRTrainResponse,
    summary="Plaintext logistic regression training",
    description="Reads a plaintext training matrix and label vector from local storage, uploads them to MictlanX, then delegates training to a worker.",
)
async def logistic_regression_train(
    body: LRTrainRequest,
    logger=Depends(get_logger),
    settings: Settings = Depends(get_settings),
    storage_client: AsyncClient = Depends(get_storage_client),
    manager: RoryManager = Depends(get_manager),
    executor: ProcessPoolExecutor = Depends(get_executor),
):
    try:
        local_start_time = time.time()
        BUCKET_ID: str = settings.mictlanx_bucket_id
        TESTING = settings.testing
        SOURCE_PATH = settings.source_path
        num_chunks = settings.num_chunks
        if executor is None:
            raise HTTPException(status_code=500, detail="No process pool executor available")
        algorithm = Constants.MachineLearningAlgorithms.LOGISTIC_REGRESSION_TRAIN
        s = Session()
        experiment_id = body.experiment_id
        plaintext_matrix_train_id = body.plaintext_matrix_train_id
        plaintext_label_vector_train_id = body.plaintext_label_vector_train_id
        plaintext_matrix_train_filename = body.plaintext_matrix_train_filename
        plaintext_label_vector_train_filename = body.plaintext_label_vector_train_filename
        extension = body.extension
        plaintext_matrix_train_path = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_train_filename, extension)
        plaintext_label_vector_train_path = "{}/{}.{}".format(SOURCE_PATH, plaintext_label_vector_train_filename, extension)

        epochs = body.epochs
        learning_rate = body.learning_rate
        weights_id = "{}weights".format(plaintext_matrix_train_id)
        bias_id = "{}bias".format(plaintext_matrix_train_id)
        WORKER_TIMEOUT = settings.worker_timeout
        MICTLANX_TIMEOUT = settings.mictlanx_timeout

        storage_backend = (
            StorageBuilder(storage_client=storage_client)
            .with_storage_params(StorageParams(num_chunks=num_chunks, timeout=MICTLANX_TIMEOUT))
            .build()
        )

        plaintext_matrix_train_result = await storage_backend.put_from_file(
            bucket_id=BUCKET_ID,
            ball_id=plaintext_matrix_train_id,
            path=plaintext_matrix_train_path,
            extension=extension,
            segment=True,
            encrypt=False,
            delete=True,
        )

        if plaintext_matrix_train_result.is_err:
            logger.error("Failed to process training dataset: {}".format(plaintext_matrix_train_result.unwrap_err()))
            raise HTTPException(status_code=500, detail="Failed to process training dataset")
        plaintext_matrix_train_response = plaintext_matrix_train_result.unwrap()
        logger.debug({
            "event": "PUT",
            "experiment_id": experiment_id,
            "bucket_id": BUCKET_ID,
            "ball_id": plaintext_matrix_train_id,
            "matrix_id": plaintext_matrix_train_id,
            "shape": str(plaintext_matrix_train_response.shape),
            "dtype": str(plaintext_matrix_train_response.dtype),
            "read_time": plaintext_matrix_train_response.read_time,
            "segment_time": plaintext_matrix_train_response.segment_time,
            "upload_time": plaintext_matrix_train_response.upload_time,
        })

        plaintext_label_vector_train = await storage_backend.put_from_file(
            bucket_id=BUCKET_ID,
            ball_id=plaintext_label_vector_train_id,
            path=plaintext_label_vector_train_path,
            extension=extension,
            segment=True,
            encrypt=False,
            delete=True,
        )

        if plaintext_label_vector_train.is_err:
            logger.error("Failed to process label vector: {}".format(plaintext_label_vector_train.unwrap_err()))
            raise HTTPException(status_code=500, detail="Failed to process label vector")

        plaintext_label_vector_train_response = plaintext_label_vector_train.unwrap()
        logger.debug({
            "event": "PUT",
            "experiment_id": experiment_id,
            "bucket_id": BUCKET_ID,
            "ball_id": plaintext_label_vector_train_id,
            "matrix_id": plaintext_label_vector_train_id,
            "shape": str(plaintext_label_vector_train_response.shape),
            "dtype": str(plaintext_label_vector_train_response.dtype),
            "read_time": plaintext_label_vector_train_response.read_time,
            "segment_time": plaintext_label_vector_train_response.segment_time,
            "upload_time": plaintext_label_vector_train_response.upload_time,
        })

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
        worker_start_time = time.time()

        worker = RoryWorker(
            workerId=worker_id,
            port=port,
            session=s,
            algorithm=algorithm,
        )

        status = Constants.ClusteringStatus.START

        worker_headers = {
            "Clustering-Status": str(status),
            "Experiment-Id": experiment_id,
            "Plaintext-Matrix-Train-Id": plaintext_matrix_train_id,
            "Plaintext-Label-Vector-Train-Id": plaintext_label_vector_train_id,
            "Epochs": str(epochs),
            "Learning-Rate": str(learning_rate),
            "Weights-Id": weights_id,
            "Bias-Id": bias_id,
        }

        worker_response = worker.run(
            timeout=WORKER_TIMEOUT,
            headers=worker_headers,
        )
        worker_status = worker_response.status_code

        if worker_status != 200:
            raise HTTPException(status_code=500, detail="Worker error: {}".format(worker_response.content))

        worker_response.raise_for_status()
        jsonWorkerResponse = worker_response.json()
        worker_service_time = jsonWorkerResponse["service_time"]
        worker_end_time = time.time()

        worker_response_time = worker_end_time - worker_start_time
        response_time = time.time() - local_start_time

        logger.info(ExperimentLogEntry(
            event="COMPLETED",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=local_start_time,
            end_time=time.time(),
            id=plaintext_matrix_train_id,
            epochs=epochs,
            learning_rate=learning_rate,
            worker_id=worker_id,
            dataowner_time=service_time_dataowner,
            manager_time=get_worker_service_time,
            worker_time=worker_response_time,
        ).model_dump())

        return {
            "worker_id": worker_id,
            "service_time_manager": get_worker_service_time,
            "service_time_worker": worker_response_time,
            "service_time_dataowner": service_time_dataowner,
            "service_time_train": response_time,
            "algorithm": algorithm,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("DATAOWNER_ERROR " + str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/logistic-regression/predict",
    response_model=LRPredictResponse,
    summary="Plaintext logistic regression prediction",
    description="Reads a plaintext test matrix from local storage, uploads it to MictlanX, delegates prediction to a worker, and retrieves the predictions.",
)
async def logistic_regression_predict(
    body: LRPredictRequest,
    logger=Depends(get_logger),
    settings: Settings = Depends(get_settings),
    storage_client: AsyncClient = Depends(get_storage_client),
    manager: RoryManager = Depends(get_manager),
    executor: ProcessPoolExecutor = Depends(get_executor),
):
    try:
        arrivalTime = time.time()
        BUCKET_ID: str = settings.mictlanx_bucket_id
        TESTING = settings.testing
        SOURCE_PATH = settings.source_path
        WORKER_TIMEOUT = settings.worker_timeout
        MICTLANX_TIMEOUT = settings.mictlanx_timeout
        num_chunks = settings.num_chunks
        if executor is None:
            raise HTTPException(status_code=500, detail="No process pool executor available")
        algorithm = Constants.MachineLearningAlgorithms.LOGISTIC_REGRESSION_PREDICT
        s = Session()
        experiment_id = body.experiment_id
        plaintext_matrix_train_id = body.plaintext_matrix_train_id
        plaintext_matrix_test_id = body.plaintext_matrix_test_id
        plaintext_matrix_test_filename = body.plaintext_matrix_test_filename
        extension = body.extension
        plaintext_matrix_test_path = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_test_filename, extension)
        weights_id = "{}weights".format(plaintext_matrix_train_id)
        bias_id = "{}bias".format(plaintext_matrix_train_id)

        storage_backend = (
            StorageBuilder(storage_client=storage_client)
            .with_storage_params(StorageParams(num_chunks=num_chunks, timeout=MICTLANX_TIMEOUT))
            .build()
        )

        plaintext_matrix_test_result = await storage_backend.put_from_file(
            bucket_id=BUCKET_ID,
            ball_id=plaintext_matrix_test_id,
            path=plaintext_matrix_test_path,
            extension=extension,
            segment=True,
            encrypt=False,
            delete=True,
        )

        if plaintext_matrix_test_result.is_err:
            logger.error("Failed to process test dataset: {}".format(plaintext_matrix_test_result.unwrap_err()))
            raise HTTPException(status_code=500, detail="Failed to process test dataset")

        plaintext_matrix_test_response = plaintext_matrix_test_result.unwrap()
        logger.debug({
            "event": "PUT",
            "experiment_id": experiment_id,
            "bucket_id": BUCKET_ID,
            "ball_id": plaintext_matrix_test_id,
            "matrix_id": plaintext_matrix_test_id,
            "shape": str(plaintext_matrix_test_response.shape),
            "dtype": str(plaintext_matrix_test_response.dtype),
            "read_time": plaintext_matrix_test_response.read_time,
            "segment_time": plaintext_matrix_test_response.segment_time,
            "upload_time": plaintext_matrix_test_response.upload_time,
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

        status = Constants.ClusteringStatus.START
        worker_headers = {
            "Clustering-Status": str(status),
            "Experiment-Id": experiment_id,
            "Plaintext-Matrix-Test-Id": plaintext_matrix_test_id,
            "Plaintext-Matrix-Train-Id": plaintext_matrix_train_id,
            "Weights-Id": weights_id,
            "Bias-Id": bias_id,
        }

        worker_response = worker.run(
            timeout=WORKER_TIMEOUT,
            headers=worker_headers,
        )
        worker_status = worker_response.status_code

        if worker_status != 200:
            raise HTTPException(status_code=500, detail="Worker error: {}".format(worker_response.content))

        worker_response.raise_for_status()
        jsonWorkerResponse = worker_response.json()
        predictions_id = jsonWorkerResponse["predictions_id"]
        worker_service_time = jsonWorkerResponse["service_time"]
        worker_end_time = time.time()
        worker_response_time = worker_end_time - worker_start_time

        predictions_result = await storage_backend.get(
            bucket_id=BUCKET_ID,
            ball_id=predictions_id,
            segment=True,
            encrypt=False,
        )
        if predictions_result.is_err:
            logger.error(f"Failed to get predictions: {predictions_result.unwrap_err()}")
            raise HTTPException(status_code=500, detail="Failed to get predictions")
        predictions_response = predictions_result.unwrap()
        predictions = predictions_response.raw_value
        label_vector = predictions.astype(int).tolist()

        logger.debug({
            "event": "GET",
            "experiment_id": experiment_id,
            "bucket_id": BUCKET_ID,
            "ball_id": predictions_id,
            "matrix_id": predictions_id,
            "shape": str(predictions.shape),
            "dtype": str(predictions.dtype),
            "read_time": predictions_response.read_time,
        })

        response_time = time.time() - arrivalTime

        logger.info(ExperimentLogEntry(
            event="COMPLETED",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=arrivalTime,
            end_time=time.time(),
            id=plaintext_matrix_train_id,
            worker_id=worker_id,
            dataowner_time=service_time_dataowner,
            manager_time=get_worker_service_time,
            worker_time=worker_response_time,
        ).model_dump())

        return {
            "label_vector": label_vector,
            "algorithm": algorithm,
            "worker_id": worker_id,
            "service_time_manager": get_worker_service_time,
            "service_time_worker": worker_response_time,
            "service_time_dataowner": service_time_dataowner,
            "service_time_predict": response_time,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("DATAOWNER_ERROR " + str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/pplr/train",
    response_model=PPLRTrainResponse,
    summary="Privacy-preserving logistic regression training",
    description="Encrypts training data using CKKS, stores it in MictlanX, and delegates privacy-preserving training to a worker across multiple epochs.",
)
async def pplr_train(
    body: PPLRTrainRequest,
    logger=Depends(get_logger),
    settings: Settings = Depends(get_settings),
    storage_client: AsyncClient = Depends(get_storage_client),
    manager: RoryManager = Depends(get_manager),
    executor: ProcessPoolExecutor = Depends(get_executor),
    ckks=Depends(get_ckks),
):
    encrypted_weights = None
    encrypted_bias = None
    weights_plain_list = None
    weights_plain = None
    bias_plain_list = None
    bias_plain = None
    plaintext_weight = None
    plaintext_bias = None
    encrypted_weight_result = None
    encrypted_weight_response = None
    encrypted_weights_result = None
    encrypted_weights_response = None
    encrypted_bias_result = None
    encrypted_bias_response = None
    encrypted_weight_put_response = None
    encrypted_bias_put_response = None
    ckks_params = None
    storage_backend = None

    try:
        arrivalTime = time.time()
        BUCKET_ID: str = settings.mictlanx_bucket_id
        TESTING = settings.testing
        SOURCE_PATH = settings.source_path
        num_chunks = settings.num_chunks
        security_level = settings.liu_security_level
        if executor is None:
            raise HTTPException(status_code=500, detail="No process pool executor available")
        algorithm = Constants.MachineLearningAlgorithms.PPLR_TRAIN
        MODE = CkksModes.ML
        s = Session()
        experiment_id = body.experiment_id
        plaintext_matrix_train_id = body.plaintext_matrix_train_id
        encrypted_matrix_train_id = "encrypted{}".format(plaintext_matrix_train_id)
        plaintext_label_vector_train_id = body.plaintext_label_vector_train_id
        encrypted_label_vector_train_id = "encrypted{}".format(plaintext_label_vector_train_id)
        plaintext_matrix_train_filename = body.plaintext_matrix_train_filename
        plaintext_label_vector_train_filename = body.plaintext_label_vector_train_filename
        extension = body.extension
        plaintext_matrix_train_path = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_train_filename, extension)
        plaintext_label_vector_train_path = "{}/{}.{}".format(SOURCE_PATH, plaintext_label_vector_train_filename, extension)
        total_epochs = body.epochs
        learning_rate = body.learning_rate
        encrypted_weights_id = "{}encryptedweights".format(plaintext_matrix_train_id)
        encrypted_bias_id = "{}encryptedbias".format(plaintext_matrix_train_id)

        WORKER_TIMEOUT = settings.worker_timeout
        MICTLANX_TIMEOUT = settings.mictlanx_timeout
        _round = settings.ckks_round
        decimals = settings.ckks_decimals
        keys_path = settings.keys_path
        ctx_filename = settings.ctx_filename
        pubkey_filename = settings.pubkey_filename
        secretkey_filename = settings.secret_key_filename
        relinkey_filename = settings.relinkey_filename
        rotatekey_filename = settings.rotatekey_filename

        ckks_params = CkksParams(
            keys_path=keys_path,
            ctx_filename=ctx_filename,
            pubkey_filename=pubkey_filename,
            secretkey_filename=secretkey_filename,
            relinkey_filename=relinkey_filename,
            rotatekey_filename=rotatekey_filename,
            decimals=decimals,
            _round=_round,
        )

        storage_backend = (
            StorageBuilder(storage_client=storage_client, scheme=Scheme.CKKS)
            .with_ckks(ckks)
            .with_ckks_params(ckks_params=ckks_params)
            .with_storage_params(StorageParams(num_chunks=num_chunks, timeout=MICTLANX_TIMEOUT))
            .build()
        )

        plaintext_matrix_train_result = await storage_backend.put_from_file(
            bucket_id=BUCKET_ID,
            ball_id=encrypted_matrix_train_id,
            path=plaintext_matrix_train_path,
            extension=extension,
            segment=True,
            encrypt=True,
            delete=True,
        )

        if plaintext_matrix_train_result.is_err:
            logger.error("Failed to process training dataset: {}".format(plaintext_matrix_train_result.unwrap_err()))
            raise HTTPException(status_code=500, detail="Failed to process training dataset")
        plaintext_matrix_train_respose = plaintext_matrix_train_result.unwrap()

        logger.debug({
            "event": "PUT",
            "experiment_id": experiment_id,
            "bucket_id": BUCKET_ID,
            "ball_id": encrypted_matrix_train_id,
            "matrix_id": encrypted_matrix_train_id,
            "shape": str(plaintext_matrix_train_respose.shape),
            "dtype": str(plaintext_matrix_train_respose.dtype),
            "read_time": plaintext_matrix_train_respose.read_time,
            "segment_time": plaintext_matrix_train_respose.segment_time,
            "encrypt_time": getattr(plaintext_matrix_train_respose, "encrypt_time", 0.0),
            "upload_time": plaintext_matrix_train_respose.upload_time,
        })

        plaintext_label_vector_train = await storage_backend.put_from_file(
            bucket_id=BUCKET_ID,
            ball_id=encrypted_label_vector_train_id,
            path=plaintext_label_vector_train_path,
            extension=extension,
            segment=True,
            encrypt=True,
            delete=True,
        )

        if plaintext_label_vector_train.is_err:
            logger.error("Failed to process label vector: {}".format(plaintext_label_vector_train.unwrap_err()))
            raise HTTPException(status_code=500, detail="Failed to process label vector")
        plaintext_label_vector_train_response = plaintext_label_vector_train.unwrap()
        logger.debug({
            "event": "PUT",
            "experiment_id": experiment_id,
            "bucket_id": BUCKET_ID,
            "ball_id": encrypted_label_vector_train_id,
            "matrix_id": encrypted_label_vector_train_id,
            "shape": str(plaintext_label_vector_train_response.shape),
            "dtype": str(plaintext_label_vector_train_response.dtype),
            "read_time": plaintext_label_vector_train_response.read_time,
            "segment_time": plaintext_label_vector_train_response.segment_time,
            "encrypt_time": getattr(plaintext_label_vector_train_response, "encrypt_time", 0.0),
            "upload_time": plaintext_label_vector_train_response.upload_time,
        })

        scale = ckks.SECURITY_LEVELS[MODE.value][security_level]["scale"]
        n_samples = plaintext_matrix_train_respose.shape[0]
        n_features = plaintext_matrix_train_respose.shape[1]
        plaintext_weight = np.zeros((1, n_features), dtype=np.float32)

        encrypted_weight_result = await storage_backend.put(
            bucket_id=BUCKET_ID,
            data=plaintext_weight,
            ball_id=encrypted_weights_id,
            segment=True,
            encrypt=True,
            scheme=Scheme.CKKS,
            delete=True,
        )

        if encrypted_weight_result.is_err:
            logger.error("Failed to put encrypted weights in cloud storage: {}".format(encrypted_weight_result.unwrap_err()))
            raise HTTPException(status_code=500, detail="Failed to put encrypted weights in cloud storage")
        encrypted_weight_response = encrypted_weight_result.unwrap()
        logger.debug({
            "event": "PUT",
            "experiment_id": experiment_id,
            "bucket_id": BUCKET_ID,
            "ball_id": encrypted_weights_id,
            "matrix_id": encrypted_weights_id,
            "shape": str(encrypted_weight_response.shape),
            "dtype": str(encrypted_weight_response.dtype),
            "read_time": getattr(encrypted_weight_response, "read_time", 0.0),
            "segment_time": getattr(encrypted_weight_response, "segment_time", 0.0),
            "encrypt_time": getattr(encrypted_weight_response, "encrypt_time", 0.0),
            "upload_time": getattr(encrypted_weight_response, "upload_time", 0.0),
        })

        plaintext_bias = np.array([0.0], dtype=np.float32)

        encrypted_bias_result = await storage_backend.put(
            bucket_id=BUCKET_ID,
            data=plaintext_bias,
            ball_id=encrypted_bias_id,
            segment=True,
            encrypt=True,
            scheme=Scheme.CKKS,
            delete=True,
        )

        if encrypted_bias_result.is_err:
            logger.error("Failed to put encrypted bias in cloud storage: {}".format(encrypted_bias_result.unwrap_err()))
            raise HTTPException(status_code=500, detail="Failed to put encrypted bias in cloud storage")
        encrypted_bias_response = encrypted_bias_result.unwrap()
        logger.debug({
            "event": "PUT",
            "experiment_id": experiment_id,
            "bucket_id": BUCKET_ID,
            "ball_id": encrypted_bias_id,
            "matrix_id": encrypted_bias_id,
            "shape": str(encrypted_bias_response.shape),
            "dtype": str(encrypted_bias_response.dtype),
            "read_time": getattr(encrypted_bias_response, "read_time", 0.0),
            "segment_time": getattr(encrypted_bias_response, "segment_time", 0.0),
            "encrypt_time": getattr(encrypted_bias_response, "encrypt_time", 0.0),
            "upload_time": getattr(encrypted_bias_response, "upload_time", 0.0),
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
        worker_id = "localhost" if TESTING else _worker_id

        worker_start_time = time.time()
        worker = RoryWorker(
            workerId=worker_id,
            port=port,
            session=s,
            algorithm=algorithm,
        )

        current_epoch = 0
        status = Constants.ClusteringStatus.START

        while current_epoch < total_epochs:

            if current_epoch > 0:
                status = Constants.ClusteringStatus.WORK_IN_PROGRESS

            worker_headers = {
                "Clustering-Status": str(status),
                "Experiment-Id": experiment_id,
                "Learning-Rate": str(learning_rate),
                "Encrypted-Matrix-Train-Id": encrypted_matrix_train_id,
                "Encrypted-Label-Vector-Train-Id": encrypted_label_vector_train_id,
                "Encrypted-Weights-Id": encrypted_weights_id,
                "Encrypted-Bias-Id": encrypted_bias_id,
                "Scale": str(scale),
                "N-Features": str(n_features),
                "N-Samples": str(n_samples),
                "Num-Chunks": str(num_chunks),
            }
            logger.debug({
                "event": "WORKER.RUN",
                "worker_id": _worker_id,
                "status": str(status),
                "experiment_id": experiment_id,
                "learning_rate": learning_rate,
                "encrypted_matrix_train_id": encrypted_matrix_train_id,
                "encrypted_label_vector_train_id": encrypted_label_vector_train_id,
                "encrypted_weights_id": encrypted_weights_id,
                "encrypted_bais_id": encrypted_bias_id,
                "scale": scale,
                "n_features": n_features,
                "n_samples": n_samples,
                "num_chunks": num_chunks,
                "total_epochs": total_epochs,
                "current_epoch": current_epoch,
            })
            worker_run_start_time = time.time()
            worker_response = worker.run(
                timeout=WORKER_TIMEOUT,
                headers=worker_headers,
            )
            worker_status = worker_response.status_code

            if worker_status != 200:
                logger.error(f"Worker execution failed at epoch {current_epoch + 1}: {worker_response.content}")
                raise HTTPException(status_code=500, detail="Worker error: {}".format(worker_response.content))

            worker_response.raise_for_status()
            jsonWorkerResponse = worker_response.json()
            worker_service_time = jsonWorkerResponse["service_time"]
            worker_end_time = time.time()

            logger.debug({
                "event": "WORKER.RUN.COMPLETED",
                "total_epochs": total_epochs,
                "current_epoch": current_epoch,
                "response_time": worker_end_time - worker_run_start_time,
            })
            current_epoch += 1

            encrypted_weights_result = await storage_backend.get(
                bucket_id=BUCKET_ID,
                ball_id=encrypted_weights_id,
                segment=True,
                encrypt=True,
                scheme=Scheme.CKKS,
            )

            if encrypted_weights_result.is_err:
                logger.error(f"Failed to get encrypted weights: {encrypted_weights_result.unwrap_err()}")
                raise HTTPException(status_code=500, detail="Failed to get encrypted weights")
            encrypted_weights_response = encrypted_weights_result.unwrap()
            encrypted_weights = encrypted_weights_response.raw_value
            logger.debug({
                "event": "GET",
                "experiment_id": experiment_id,
                "bucket_id": BUCKET_ID,
                "ball_id": encrypted_weights_id,
                "matrix_id": encrypted_weights_id,
                "shape": str(encrypted_weight_response.shape),
                "dtype": str(encrypted_weight_response.dtype),
                "read_time": encrypted_weights_response.read_time,
            })

            start_time_decryption = time.time()
            weights_plain_list = ckks.decrypt_list(encrypted_weights, take=n_features)
            weights_plain = weights_plain_list[0].reshape(1, -1).astype(np.float32)
            end_time_decryption = time.time() - start_time_decryption
            logger.debug({
                "event": "DECRYPT",
                "experiment_id": experiment_id,
                "decrypt_time": end_time_decryption,
            })

            encrypted_weight_result = await storage_backend.put(
                bucket_id=BUCKET_ID,
                data=weights_plain,
                ball_id=encrypted_weights_id,
                segment=True,
                encrypt=True,
                scheme=Scheme.CKKS,
                delete=True,
            )

            if encrypted_weight_result.is_err:
                logger.error("Failed to put encrypted weights in cloud storage: {}".format(encrypted_weight_result.unwrap_err()))
                raise HTTPException(status_code=500, detail="Failed to put encrypted weights in cloud storage")
            encrypted_weight_put_response = encrypted_weight_result.unwrap()
            logger.debug({
                "event": "PUT",
                "experiment_id": experiment_id,
                "bucket_id": BUCKET_ID,
                "ball_id": encrypted_weights_id,
                "matrix_id": encrypted_weights_id,
                "shape": str(encrypted_weight_put_response.shape),
                "dtype": str(encrypted_weight_put_response.dtype),
                "read_time": getattr(encrypted_weight_put_response, "read_time", 0.0),
                "segment_time": getattr(encrypted_weight_put_response, "segment_time", 0.0),
                "encrypt_time": getattr(encrypted_weight_put_response, "encrypt_time", 0.0),
                "upload_time": getattr(encrypted_weight_put_response, "upload_time", 0.0),
            })

            encrypted_weight_result = None
            encrypted_weight_put_response = None
            encrypted_weights_result = None
            encrypted_weights_response = None
            weights_plain = None
            encrypted_weights = None
            weights_plain_list = None

            encrypted_bias_result = await storage_backend.get(
                bucket_id=BUCKET_ID,
                ball_id=encrypted_bias_id,
                segment=True,
                encrypt=True,
                scheme=Scheme.CKKS,
            )

            if encrypted_bias_result.is_err:
                logger.error(f"Failed to get encrypted bias: {encrypted_bias_result.unwrap_err()}")
                raise HTTPException(status_code=500, detail="Failed to get encrypted bias")
            encrypted_bias_response = encrypted_bias_result.unwrap()
            encrypted_bias = encrypted_bias_response.raw_value
            logger.debug({
                "event": "GET",
                "experiment_id": experiment_id,
                "bucket_id": BUCKET_ID,
                "ball_id": encrypted_bias_id,
                "matrix_id": encrypted_bias_id,
                "shape": str(encrypted_bias_response.shape if hasattr(encrypted_bias_response, "shape") else (1,)),
                "dtype": "float32",
                "read_time": encrypted_bias_response.read_time,
            })

            start_time_decryption = time.time()
            bias_plain_list = ckks.decrypt_list(encrypted_bias, take=1)
            bias_plain = bias_plain_list[0].reshape(1, -1).astype(np.float32)
            end_time_decryption = time.time() - start_time_decryption
            logger.debug({
                "event": "DECRYPT.BIAS",
                "experiment_id": experiment_id,
                "encrypted_bias_id": encrypted_bias_id,
                "decrypt_time": end_time_decryption,
            })

            encrypted_bias_result = await storage_backend.put(
                bucket_id=BUCKET_ID,
                data=bias_plain,
                ball_id=encrypted_bias_id,
                segment=True,
                encrypt=True,
                scheme=Scheme.CKKS,
                delete=True,
            )

            if encrypted_bias_result.is_err:
                logger.error("Failed to put encrypted bias in cloud storage: {}".format(encrypted_bias_result.unwrap_err()))
                raise HTTPException(status_code=500, detail="Failed to put encrypted bias in cloud storage")
            encrypted_bias_put_response = encrypted_bias_result.unwrap()
            logger.debug({
                "event": "PUT",
                "experiment_id": experiment_id,
                "bucket_id": BUCKET_ID,
                "ball_id": encrypted_bias_id,
                "matrix_id": encrypted_bias_id,
                "shape": str(encrypted_bias_put_response.shape),
                "dtype": str(encrypted_bias_put_response.dtype),
                "read_time": getattr(encrypted_bias_put_response, "read_time", 0.0),
                "segment_time": getattr(encrypted_bias_put_response, "segment_time", 0.0),
                "encrypt_time": getattr(encrypted_bias_put_response, "encrypt_time", 0.0),
                "upload_time": getattr(encrypted_bias_put_response, "upload_time", 0.0),
            })
            encrypted_bias_result = None
            encrypted_bias_response = None
            encrypted_bias_put_response = None
            encrypted_bias = None
            bias_plain_list = None
            bias_plain = None

        worker_response_time = worker_end_time - worker_start_time
        response_time = time.time() - arrivalTime

        logger.info(ExperimentLogEntry(
            event="COMPLETED",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=arrivalTime,
            end_time=time.time(),
            id=plaintext_matrix_train_id,
            epochs=total_epochs,
            learning_rate=learning_rate,
            worker_id=worker_id,
            security_level=security_level,
            dataowner_time=service_time_dataowner,
            manager_time=get_worker_service_time,
            worker_time=worker_response_time,
        ).model_dump())

        return {
            "algorithm": algorithm,
            "worker_id": worker_id,
            "epochs": total_epochs,
            "service_time_manager": get_worker_service_time,
            "service_time_worker": worker_response_time,
            "service_time_dataowner": service_time_dataowner,
            "service_time_train": response_time,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error({
            "msg": str(e),
        })
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        del encrypted_weights
        del encrypted_bias
        del weights_plain_list
        del weights_plain
        del bias_plain_list
        del bias_plain
        del plaintext_weight
        del plaintext_bias
        del encrypted_weight_result
        del encrypted_weight_response
        del encrypted_weights_result
        del encrypted_weights_response
        del encrypted_bias_result
        del encrypted_bias_response
        del encrypted_weight_put_response
        del encrypted_bias_put_response
        del ckks_params
        del storage_backend
        gc.collect()


@router.post(
    "/pplr/predict",
    response_model=PPLRPredictResponse,
    summary="Privacy-preserving logistic regression prediction",
    description="Encrypts a test matrix using CKKS, delegates privacy-preserving prediction to a worker, and decrypts the returned predictions.",
)
async def pplr_predict(
    body: PPLRPredictRequest,
    logger=Depends(get_logger),
    settings: Settings = Depends(get_settings),
    storage_client: AsyncClient = Depends(get_storage_client),
    manager: RoryManager = Depends(get_manager),
    executor: ProcessPoolExecutor = Depends(get_executor),
    ckks=Depends(get_ckks),
):
    encrypted_predictions = None
    encrypted_predictions_result = None
    encrypted_predictions_response = None
    predictions_plain_list = None
    predictions_plain = None
    ckks_params = None
    storage_backend = None

    try:
        arrivalTime = time.time()
        BUCKET_ID: str = settings.mictlanx_bucket_id
        TESTING = settings.testing
        SOURCE_PATH = settings.source_path
        max_workers = settings.max_workers
        num_chunks = settings.num_chunks
        security_level = settings.liu_security_level

        if executor is None:
            raise HTTPException(status_code=500, detail="No process pool executor available")
        algorithm = Constants.MachineLearningAlgorithms.PPLR_PREDICT
        MODE = CkksModes.ML
        s = Session()
        experiment_id = body.experiment_id
        experiment_iteration = body.experiment_iteration

        plaintext_matrix_test_id = body.plaintext_matrix_test_id
        encrypted_matrix_test_id = "encrypted{}".format(plaintext_matrix_test_id)
        plaintext_matrix_test_filename = body.plaintext_matrix_test_filename
        extension = body.extension
        plaintext_matrix_train_id = body.plaintext_matrix_train_id
        plaintext_matrix_test_path = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_test_filename, extension)
        encrypted_weights_id = "{}encryptedweights".format(plaintext_matrix_train_id)
        encrypted_bias_id = "{}encryptedbias".format(plaintext_matrix_train_id)

        _round = settings.ckks_round
        decimals = settings.ckks_decimals
        keys_path = settings.keys_path
        ctx_filename = settings.ctx_filename
        pubkey_filename = settings.pubkey_filename
        secretkey_filename = settings.secret_key_filename
        relinkey_filename = settings.relinkey_filename
        rotatekey_filename = settings.rotatekey_filename
        WORKER_TIMEOUT = settings.worker_timeout
        MICTLANX_TIMEOUT = settings.mictlanx_timeout

        logger.debug({
            "event": "PPLR.PREDICT.STARTED",
            "experiment_id": experiment_id,
            "num_chunks": num_chunks,
        })

        ckks_params = CkksParams(
            keys_path=keys_path,
            ctx_filename=ctx_filename,
            pubkey_filename=pubkey_filename,
            secretkey_filename=secretkey_filename,
            relinkey_filename=relinkey_filename,
            rotatekey_filename=rotatekey_filename,
            decimals=decimals,
            _round=_round,
        )

        storage_backend = (
            StorageBuilder(storage_client=storage_client, scheme=Scheme.CKKS)
            .with_ckks(ckks)
            .with_ckks_params(ckks_params=ckks_params)
            .with_storage_params(StorageParams(num_chunks=num_chunks, timeout=MICTLANX_TIMEOUT))
            .build()
        )

        plaintext_matrix_test_result = await storage_backend.put_from_file(
            bucket_id=BUCKET_ID,
            ball_id=encrypted_matrix_test_id,
            path=plaintext_matrix_test_path,
            extension=extension,
            segment=True,
            encrypt=True,
            delete=True,
        )

        if plaintext_matrix_test_result.is_err:
            logger.error("Failed to process test dataset: {}".format(plaintext_matrix_test_result.unwrap_err()))
            raise HTTPException(status_code=500, detail="Failed to process test dataset")
        plaintext_matrix_test_response = plaintext_matrix_test_result.unwrap()
        logger.debug({
            "event": "PUT",
            "experiment_id": experiment_id,
            "bucket_id": BUCKET_ID,
            "ball_id": encrypted_matrix_test_id,
            "matrix_id": encrypted_matrix_test_id,
            "shape": str(plaintext_matrix_test_response.shape),
            "dtype": str(plaintext_matrix_test_response.dtype),
            "read_time": plaintext_matrix_test_response.read_time,
            "segment_time": plaintext_matrix_test_response.segment_time,
            "encrypt_time": getattr(plaintext_matrix_test_response, "encrypt_time", 0.0),
            "upload_time": plaintext_matrix_test_response.upload_time,
        })

        scale = ckks.SECURITY_LEVELS[MODE.value][security_level]["scale"]
        n_features = plaintext_matrix_test_response.shape[1]

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

        worker_headers = {
            "Experiment-Id": experiment_id,
            "Encrypted-Matrix-Test-Id": encrypted_matrix_test_id,
            "Encrypted-Weights-Id": encrypted_weights_id,
            "Encrypted-Bias-Id": encrypted_bias_id,
            "Scale": str(scale),
            "N-Features": str(n_features),
            "Num-Chunks": str(num_chunks),
        }

        worker_response = worker.run(
            timeout=WORKER_TIMEOUT,
            headers=worker_headers,
        )
        worker_status = worker_response.status_code

        if worker_status != 200:
            raise HTTPException(status_code=500, detail="Worker error: {}".format(worker_response.content))

        worker_response.raise_for_status()
        jsonWorkerResponse = worker_response.json()
        encrypted_predictions_id = jsonWorkerResponse["encrypted_predictions_id"]
        worker_service_time = jsonWorkerResponse["service_time"]
        worker_end_time = time.time()
        worker_response_time = worker_end_time - worker_start_time

        encrypted_predictions_result = await storage_backend.get(
            bucket_id=BUCKET_ID,
            ball_id=encrypted_predictions_id,
            segment=True,
            encrypt=True,
            scheme=Scheme.CKKS,
        )

        if encrypted_predictions_result.is_err:
            logger.error(f"Failed to get encrypted predictions: {encrypted_predictions_result.unwrap_err()}")
            raise HTTPException(status_code=500, detail="Failed to get encrypted predictions")
        encrypted_predictions_response = encrypted_predictions_result.unwrap()
        encrypted_predictions = encrypted_predictions_response.raw_value
        logger.debug({
            "event": "GET",
            "experiment_id": experiment_id,
            "bucket_id": BUCKET_ID,
            "ball_id": encrypted_predictions_id,
            "matrix_id": encrypted_predictions_id,
            "shape": str(plaintext_matrix_test_response.shape),
            "dtype": str(plaintext_matrix_test_response.dtype),
            "read_time": encrypted_predictions_response.read_time,
        })

        start_time_decryption = time.time()
        predictions_plain_list = ckks.decrypt_list(encrypted_predictions, take=1)
        predictions_plain = np.array([p[0] for p in predictions_plain_list], dtype=np.float32)
        end_time_decryption = time.time() - start_time_decryption
        logger.debug({
            "event": "DECRYPT",
            "experiment_id": experiment_id,
            "decrypt_time": end_time_decryption,
        })

        label_vector = [1 if v >= 0.5 else 0 for v in predictions_plain]
        response_time = time.time() - arrivalTime

        logger.info(ExperimentLogEntry(
            event="COMPLETED",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=arrivalTime,
            end_time=time.time(),
            id=plaintext_matrix_train_id,
            worker_id=worker_id,
            security_level=security_level,
            dataowner_time=service_time_dataowner,
            manager_time=get_worker_service_time,
            worker_time=worker_response_time,
        ).model_dump())

        return {
            "label_vector": label_vector,
            "algorithm": algorithm,
            "worker_id": worker_id,
            "service_time_manager": get_worker_service_time,
            "service_time_worker": worker_response_time,
            "service_time_dataowner": service_time_dataowner,
            "service_time_predict": response_time,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error({
            "msg": str(e),
        })
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        del encrypted_predictions
        del encrypted_predictions_result
        del encrypted_predictions_response
        del predictions_plain_list
        del predictions_plain
        del ckks_params
        del storage_backend
        gc.collect()
