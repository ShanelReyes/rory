import time
import gc
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from rory.core.classification.secure.pqc.pplr import PPLR
from rory.core.classification.logistic_regression import LogisticRegression
from rory.core.utils.constants import Constants
from rorycommon import StorageBuilder, StorageParams, Scheme, CkksParams
from models import ExperimentLogEntry
from dependencies import get_logger, get_storage_client, get_ckks, get_settings
from config import Settings
from models.requests.machinelearning import (
    LRTrainWorkerRequest,
    LRPredictWorkerRequest,
    PPLRTrainWorkerRequest,
    PPLRPredictWorkerRequest,
)
from models.responses.machinelearning import (
    LRTrainResponse,
    LRPredictResponse,
    PPLRTrainResponse,
    PPLRPredictResponse,
)
from models.responses.clustering import HealthCheckResponse

router = APIRouter(prefix="/machine-learning", tags=["Machine Learning"])


@router.api_route("/test", methods=["GET", "POST"], response_model=HealthCheckResponse)
def test():
    return JSONResponse(
        content={"component_type": "worker"},
        status_code=200,
        headers={"Component-Type": "worker"},
    )


@router.post("/logistic-regression/train", response_model=LRTrainResponse)
async def logistic_regression_train(
    body: LRTrainWorkerRequest,
    logger=Depends(get_logger),
    storage_client=Depends(get_storage_client),
    settings: Settings = Depends(get_settings),
):
    local_start_time = time.time()
    worker_id = settings.node_id
    STORAGE_CLIENT = storage_client
    BUCKET_ID: str = settings.mictlanx_bucket_id
    MICTLANX_TIMEOUT = settings.mictlanx_timeout
    num_chunks = settings.num_chunks
    experiment_id = body.experiment_id
    algorithm = Constants.MachineLearningAlgorithms.LOGISTIC_REGRESSION_TRAIN
    plaintext_matrix_train_id = body.plaintext_matrix_train_id
    plaintext_label_vector_train_id = body.plaintext_label_vector_train_id
    weights_id = body.weights_id
    bias_id = body.bias_id
    epochs = int(body.epochs)
    learning_rate = float(body.learning_rate)

    if not all([plaintext_matrix_train_id, plaintext_label_vector_train_id, weights_id, bias_id]):
        raise HTTPException(status_code=400, detail="Missing mandatory IDs or shape parameters")

    storage_backend = (
        StorageBuilder(storage_client=STORAGE_CLIENT)
        .with_storage_params(StorageParams(num_chunks=num_chunks, timeout=MICTLANX_TIMEOUT))
        .build()
    )

    plaintext_matrix_train_result = await storage_backend.get(
        bucket_id=BUCKET_ID,
        ball_id=plaintext_matrix_train_id,
        segment=True,
        encrypt=False,
    )
    if plaintext_matrix_train_result.is_err:
        logger.error(f"Failed to get matrix train: {plaintext_matrix_train_result.unwrap_err()}")
        raise HTTPException(status_code=500, detail="Failed to get matrix train")

    plaintext_matrix_train_response = plaintext_matrix_train_result.unwrap()
    plaintext_matrix_train = plaintext_matrix_train_response.raw_value
    logger.debug({
        "event": "GET",
        "experiment_id": experiment_id,
        "bucket_id": BUCKET_ID,
        "ball_id": plaintext_matrix_train_id,
        "matrix_id": plaintext_matrix_train_id,
        "shape": str(plaintext_matrix_train.shape),
        "dtype": str(plaintext_matrix_train.dtype),
        "read_time": plaintext_matrix_train_response.read_time,
    })

    plaintext_label_vector_train_result = await storage_backend.get(
        bucket_id=BUCKET_ID,
        ball_id=plaintext_label_vector_train_id,
        segment=True,
        encrypt=False,
    )

    if plaintext_label_vector_train_result.is_err:
        logger.error(f"Failed to get label vector train: {plaintext_label_vector_train_result.unwrap_err()}")
        raise HTTPException(status_code=500, detail="Failed to get label vector train")

    plaintext_label_vector_train_response = plaintext_label_vector_train_result.unwrap()
    plaintext_label_vector_train = plaintext_label_vector_train_response.raw_value
    logger.debug({
        "event": "GET",
        "experiment_id": experiment_id,
        "bucket_id": BUCKET_ID,
        "ball_id": plaintext_label_vector_train_id,
        "matrix_id": plaintext_label_vector_train_id,
        "shape": str(plaintext_label_vector_train.shape),
        "dtype": str(plaintext_label_vector_train.dtype),
        "read_time": plaintext_label_vector_train_response.read_time,
    })

    start_time_train = time.time()
    weights, bias = LogisticRegression.fit(
        plaintext_matrix=plaintext_matrix_train,
        label_vector=plaintext_label_vector_train,
        epochs=epochs,
        learning_rate=learning_rate,
        weights=None,
        bias=0.0,
    )
    end_time_train = time.time() - start_time_train
    logger.debug({
        "event": "TRAIN",
        "experiment_id": experiment_id,
        "encrypted_matrix_id": plaintext_matrix_train_id,
        "encrypted_labelvector_id": plaintext_label_vector_train_id,
        "encrypted_weights_id": weights_id,
        "encrypted_bias_id": bias_id,
        "n_features": plaintext_matrix_train.shape[1],
        "n_samples": plaintext_matrix_train.shape[0],
        "train_time": end_time_train,
    })

    weight_result = await storage_backend.put(
        bucket_id=BUCKET_ID,
        data=weights,
        ball_id=weights_id,
        segment=True,
        encrypt=False,
        delete=True,
    )

    if weight_result.is_err:
        logger.error("Failed to put weights in cloud storage: {}".format(weight_result.unwrap_err()))
        raise HTTPException(status_code=500, detail="Failed to put weights in cloud storage")

    weight_response = weight_result.unwrap()
    logger.debug({
        "event": "PUT",
        "experiment_id": experiment_id,
        "bucket_id": BUCKET_ID,
        "ball_id": weights_id,
        "matrix_id": weights_id,
        "shape": str(weight_response.shape),
        "dtype": str(weight_response.dtype),
        "read_time": getattr(weight_response, "read_time", 0.0),
        "segment_time": getattr(weight_response, "segment_time", 0.0),
        "encrypt_time": getattr(weight_response, "encrypt_time", 0.0),
        "upload_time": getattr(weight_response, "upload_time", 0.0),
    })

    bias_result = await storage_backend.put(
        bucket_id=BUCKET_ID,
        data=[bias],
        ball_id=bias_id,
        segment=False,
        encrypt=False,
        delete=True,
    )

    if bias_result.is_err:
        logger.error("Failed to put bias in cloud storage: {}".format(bias_result.unwrap_err()))
        raise HTTPException(status_code=500, detail="Failed to put bias in cloud storage")

    bias_response = bias_result.unwrap()
    logger.debug({
        "event": "PUT",
        "experiment_id": experiment_id,
        "bucket_id": BUCKET_ID,
        "ball_id": bias_id,
        "matrix_id": bias_id,
        "shape": str(bias_response.shape),
        "dtype": str(bias_response.dtype),
        "read_time": getattr(bias_response, "read_time", 0.0),
        "segment_time": getattr(bias_response, "segment_time", 0.0),
        "upload_time": getattr(bias_response, "upload_time", 0.0),
    })

    end_time = time.time() - local_start_time

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
        worker_time=end_time,
    ).model_dump())

    return {
        "service_time": end_time,
        "train_time": end_time_train,
        "algorithm": algorithm,
    }


@router.post("/logistic-regression/predict", response_model=LRPredictResponse)
async def logistic_regression_predict(
    body: LRPredictWorkerRequest,
    logger=Depends(get_logger),
    storage_client=Depends(get_storage_client),
    settings: Settings = Depends(get_settings),
):
    local_start_time = time.time()
    logger = logger
    worker_id = settings.node_id
    STORAGE_CLIENT = storage_client
    BUCKET_ID: str = settings.mictlanx_bucket_id
    MICTLANX_TIMEOUT = settings.mictlanx_timeout
    num_chunks = settings.num_chunks
    experiment_id = body.experiment_id
    algorithm = Constants.MachineLearningAlgorithms.LOGISTIC_REGRESSION_PREDICT
    plaintext_matrix_train_id = body.plaintext_matrix_train_id
    plaintext_matrix_test_id = body.plaintext_matrix_test_id
    weights_id = body.weights_id
    bias_id = body.bias_id
    predictions_id = "{}predictions".format(plaintext_matrix_test_id)

    if not all([plaintext_matrix_test_id, weights_id, bias_id]):
        raise HTTPException(status_code=400, detail="Missing mandatory IDs or shape parameters")

    storage_backend = (
        StorageBuilder(storage_client=STORAGE_CLIENT)
        .with_storage_params(StorageParams(num_chunks=num_chunks, timeout=MICTLANX_TIMEOUT))
        .build()
    )

    plaintext_matrix_test_result = await storage_backend.get(
        bucket_id=BUCKET_ID,
        ball_id=plaintext_matrix_test_id,
        segment=True,
        encrypt=False,
    )
    if plaintext_matrix_test_result.is_err:
        logger.error(f"Failed to get matrix test: {plaintext_matrix_test_result.unwrap_err()}")
        raise HTTPException(status_code=500, detail="Failed to get matrix test")

    plaintext_matrix_test_response = plaintext_matrix_test_result.unwrap()
    plaintext_matrix_test = plaintext_matrix_test_response.raw_value
    logger.debug({
        "event": "GET",
        "experiment_id": experiment_id,
        "bucket_id": BUCKET_ID,
        "ball_id": plaintext_matrix_test_id,
        "matrix_id": plaintext_matrix_test_id,
        "shape": str(plaintext_matrix_test.shape),
        "dtype": str(plaintext_matrix_test.dtype),
        "read_time": plaintext_matrix_test_response.read_time,
    })

    weights_result = await storage_backend.get(
        bucket_id=BUCKET_ID,
        ball_id=weights_id,
        segment=True,
        encrypt=False,
    )
    if weights_result.is_err:
        logger.error(f"Failed to get weights: {weights_result.unwrap_err()}")
        raise HTTPException(status_code=500, detail="Failed to get weights")

    weights_response = weights_result.unwrap()
    weights = weights_response.raw_value
    logger.debug({
        "event": "GET",
        "experiment_id": experiment_id,
        "bucket_id": BUCKET_ID,
        "ball_id": weights_id,
        "matrix_id": weights_id,
        "shape": str(weights.shape),
        "dtype": str(weights.dtype),
        "read_time": weights_response.read_time,
    })

    bias_result = await storage_backend.get(
        bucket_id=BUCKET_ID,
        ball_id=bias_id,
        segment=False,
        encrypt=False,
    )
    if bias_result.is_err:
        logger.error(f"Failed to get bias: {bias_result.unwrap_err()}")
        raise HTTPException(status_code=500, detail="Failed to get bias")

    bias_response = bias_result.unwrap()
    bias = bias_response.raw_value
    logger.debug({
        "event": "GET",
        "experiment_id": experiment_id,
        "bucket_id": BUCKET_ID,
        "ball_id": bias_id,
        "matrix_id": bias_id,
        "dtype": "float64",
        "read_time": bias_response.read_time,
    })

    start_time_predict = time.time()
    predictions = LogisticRegression.predict(
        plaintext_matrix=plaintext_matrix_test,
        weights=weights,
        bias=bias,
    )
    end_time_predict = time.time() - start_time_predict
    logger.debug({
        "event": "PREDICT",
        "experiment_id": experiment_id,
        "encrypted_matrix_id": plaintext_matrix_test_id,
        "encrypted_weights_id": weights_id,
        "encrypted_bias_id": bias_id,
        "n_features": plaintext_matrix_test.shape[1],
        "predict_time": end_time_predict,
    })

    predictions_result = await storage_backend.put(
        bucket_id=BUCKET_ID,
        data=predictions,
        ball_id=predictions_id,
        segment=True,
        encrypt=False,
        delete=True,
    )

    if predictions_result.is_err:
        logger.error("Failed to put predictions in cloud storage: {}".format(predictions_result.unwrap_err()))
        raise HTTPException(status_code=500, detail="Failed to put predictions in cloud storage")

    predictions_response = predictions_result.unwrap()
    logger.debug({
        "event": "PUT",
        "experiment_id": experiment_id,
        "bucket_id": BUCKET_ID,
        "ball_id": predictions_id,
        "matrix_id": predictions_id,
        "shape": str(predictions_response.shape),
        "dtype": str(predictions_response.dtype),
        "read_time": getattr(predictions_response, "read_time", 0.0),
        "segment_time": getattr(predictions_response, "segment_time", 0.0),
        "encrypt_time": getattr(predictions_response, "encrypt_time", 0.0),
        "upload_time": getattr(predictions_response, "upload_time", 0.0),
    })

    end_time = time.time()
    service_time = end_time - local_start_time

    logger.info(ExperimentLogEntry(
        event="COMPLETED",
        experiment_id=experiment_id,
        algorithm=algorithm,
        start_time=local_start_time,
        end_time=time.time(),
        id=plaintext_matrix_train_id,
        worker_id=worker_id,
        worker_time=service_time,
    ).model_dump())

    return {
        "predictions_id": predictions_id,
        "predict_time": end_time_predict,
        "service_time": service_time,
    }


@router.post("/pplr/train", response_model=PPLRTrainResponse)
async def pplr_train(
    body: PPLRTrainWorkerRequest,
    logger=Depends(get_logger),
    storage_client=Depends(get_storage_client),
    ckks=Depends(get_ckks),
    settings: Settings = Depends(get_settings),
):
    encrypted_matrix_train = None
    encrypted_label_vector_train = None
    init_encrypted_weights = None
    init_encrypted_bias = None
    encrypted_weights = None
    encrypted_bias = None
    encrypted_weight_response = None
    encrypted_bias_response = None
    encrypted_matrix_train_result = None
    encrypted_matrix_train_response = None
    encrypted_label_vector_train_result = None
    encrypted_label_vector_train_response = None
    init_encrypted_weights_result = None
    init_encrypted_weights_response = None
    init_encrypted_bias_result = None
    init_encrypted_bias_response = None
    encrypted_weight_result = None
    encrypted_bias_result = None
    ckks_params = None
    storage_backend = None

    try:
        local_start_time = time.time()
        worker_id = settings.node_id
        STORAGE_CLIENT = storage_client
        BUCKET_ID: str = settings.mictlanx_bucket_id
        MICTLANX_TIMEOUT = settings.mictlanx_timeout
        algorithm = Constants.MachineLearningAlgorithms.PPLR_TRAIN
        experiment_id = body.experiment_id
        learning_rate = float(body.learning_rate)
        encrypted_matrix_train_id = body.encrypted_matrix_train_id
        encrypted_label_vector_train_id = body.encrypted_label_vector_train_id
        encrypted_weights_id = body.encrypted_weights_id
        encrypted_bias_id = body.encrypted_bias_id
        scale = int(body.scale)
        n_features = int(body.n_features)
        n_samples = int(body.n_samples)
        num_chunks = int(body.num_chunks)

        if not all([encrypted_matrix_train_id, encrypted_weights_id, encrypted_bias_id, encrypted_label_vector_train_id]):
            raise HTTPException(status_code=400, detail="Missing mandatory IDs or shape parameters")

        _round = settings.ckks_round
        decimals = settings.ckks_decimals
        keys_path = settings.keys_path
        ctx_filename = settings.ctx_filename
        pubkey_filename = settings.pubkey_filename
        secretkey_filename = settings.secret_key_filename
        relinkey_filename = settings.relinkey_filename
        rotatekey_filename = settings.rotatekey_filename
        logger.debug({
            "event": "PPLR_TRAIN_STARTED",
            "worker_id": worker_id,
            "num_chunks": num_chunks,
            "algorithm": algorithm,
            "experiment_id": experiment_id,
            "learning_rate": learning_rate,
            "encrypted_matrix_train_id": encrypted_matrix_train_id,
            "encrypted_label_vector_train_id": encrypted_label_vector_train_id,
            "encrypted_weights_id": encrypted_weights_id,
            "encrypted_bias_id": encrypted_bias_id,
            "scale": scale,
            "n_features": n_features,
            "n_samples": n_samples,
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
            StorageBuilder(storage_client=STORAGE_CLIENT, scheme=Scheme.CKKS)
            .with_ckks(ckks)
            .with_ckks_params(ckks_params=ckks_params)
            .with_storage_params(StorageParams(num_chunks=num_chunks, timeout=MICTLANX_TIMEOUT))
            .build()
        )

        encrypted_matrix_train_result = await storage_backend.get(
            bucket_id=BUCKET_ID,
            ball_id=encrypted_matrix_train_id,
            segment=True,
            encrypt=True,
            scheme=Scheme.CKKS,
        )
        if encrypted_matrix_train_result.is_err:
            logger.error(f"Failed to get encrypted matrix train: {encrypted_matrix_train_result.unwrap_err()}")
            raise HTTPException(status_code=500, detail="Failed to get encrypted matrix train")

        encrypted_matrix_train_response = encrypted_matrix_train_result.unwrap()
        encrypted_matrix_train = encrypted_matrix_train_response.raw_value
        logger.debug({
            "event": "GET",
            "experiment_id": experiment_id,
            "bucket_id": BUCKET_ID,
            "ball_id": encrypted_matrix_train_id,
            "matrix_id": encrypted_matrix_train_id,
            "shape": str((n_samples, n_features)),
            "read_time": encrypted_matrix_train_response.read_time,
        })

        encrypted_label_vector_train_result = await storage_backend.get(
            bucket_id=BUCKET_ID,
            ball_id=encrypted_label_vector_train_id,
            segment=True,
            encrypt=True,
            scheme=Scheme.CKKS,
        )

        if encrypted_label_vector_train_result.is_err:
            logger.error(f"Failed to get encrypted label vector train: {encrypted_label_vector_train_result.unwrap_err()}")
            raise HTTPException(status_code=500, detail="Failed to get encrypted label vector train")

        encrypted_label_vector_train_response = encrypted_label_vector_train_result.unwrap()
        encrypted_label_vector_train = encrypted_label_vector_train_response.raw_value
        logger.debug({
            "event": "GET",
            "experiment_id": experiment_id,
            "bucket_id": BUCKET_ID,
            "ball_id": encrypted_label_vector_train_id,
            "matrix_id": encrypted_label_vector_train_id,
            "shape": str((n_samples, 1)),
            "dtype": "PyCtxt",
            "read_time": encrypted_label_vector_train_response.read_time,
        })

        init_encrypted_weights_result = await storage_backend.get(
            bucket_id=BUCKET_ID,
            ball_id=encrypted_weights_id,
            segment=True,
            encrypt=True,
            scheme=Scheme.CKKS,
        )

        if init_encrypted_weights_result.is_err:
            logger.error(f"Failed to get init encrypted weights: {init_encrypted_weights_result.unwrap_err()}")
            raise HTTPException(status_code=500, detail="Failed to get init encrypted weights")

        init_encrypted_weights_response = init_encrypted_weights_result.unwrap()
        init_encrypted_weights = init_encrypted_weights_response.raw_value
        logger.debug({
            "event": "GET",
            "experiment_id": experiment_id,
            "bucket_id": BUCKET_ID,
            "ball_id": encrypted_weights_id,
            "matrix_id": encrypted_weights_id,
            "shape": str((1, n_features)),
            "read_time": init_encrypted_weights_response.read_time,
        })

        init_encrypted_bias_result = await storage_backend.get(
            bucket_id=BUCKET_ID,
            ball_id=encrypted_bias_id,
            segment=True,
            encrypt=True,
            scheme=Scheme.CKKS,
        )

        if init_encrypted_bias_result.is_err:
            logger.error(f"Failed to get init encrypted bias: {init_encrypted_bias_result.unwrap_err()}")
            raise HTTPException(status_code=500, detail="Failed to get init encrypted bias")

        init_encrypted_bias_response = init_encrypted_bias_result.unwrap()
        init_encrypted_bias = init_encrypted_bias_response.raw_value
        logger.debug({
            "event": "GET",
            "experiment_id": experiment_id,
            "bucket_id": BUCKET_ID,
            "ball_id": encrypted_bias_id,
            "matrix_id": encrypted_bias_id,
            "shape": str((1,)),
            "read_time": init_encrypted_bias_response.read_time,
        })

        start_time_train = time.time()
        encrypted_weights, encrypted_bias = PPLR.fit(
            HE=ckks.he_object,
            learning_rate=learning_rate,
            encrypted_weights=init_encrypted_weights[0],
            encrypted_bias=init_encrypted_bias[0],
            encrypted_matrix=encrypted_matrix_train,
            encrypted_labelvector=encrypted_label_vector_train,
            n_features=n_features,
            scale=scale,
            n_samples=n_samples,
        )
        end_time_train = time.time() - start_time_train
        logger.debug({
            "event": "TRAIN",
            "experiment_id": experiment_id,
            "encrypted_matrix_id": encrypted_matrix_train_id,
            "encrypted_labelvector_id": encrypted_label_vector_train_id,
            "encrypted_weights_id": encrypted_weights_id,
            "encrypted_bias_id": encrypted_bias_id,
            "n_features": n_features,
            "n_samples": n_samples,
            "scale": scale,
            "train_time": end_time_train,
        })

        encrypted_weight_result = await storage_backend.put(
            bucket_id=BUCKET_ID,
            data=[encrypted_weights],
            ball_id=encrypted_weights_id,
            delete=True,
            segment=True,
            encrypt=False,
            scheme=Scheme.CKKS,
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

        encrypted_bias_result = await storage_backend.put(
            bucket_id=BUCKET_ID,
            data=[encrypted_bias],
            ball_id=encrypted_bias_id,
            segment=False,
            encrypt=False,
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

        end_time = time.time() - local_start_time

        logger.info(ExperimentLogEntry(
            event="COMPLETED",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=local_start_time,
            end_time=time.time(),
            id=encrypted_matrix_train_id,
            learning_rate=learning_rate,
            worker_id=worker_id,
            worker_time=end_time,
        ).model_dump())

        return {
            "service_time": end_time,
            "train_time": end_time_train,
            "algorithm": algorithm,
        }
    finally:
        del encrypted_matrix_train
        del encrypted_label_vector_train
        del init_encrypted_weights
        del init_encrypted_bias
        del encrypted_weights
        del encrypted_bias
        del encrypted_weight_response
        del encrypted_bias_response
        del encrypted_matrix_train_result
        del encrypted_matrix_train_response
        del encrypted_label_vector_train_result
        del encrypted_label_vector_train_response
        del init_encrypted_weights_result
        del init_encrypted_weights_response
        del init_encrypted_bias_result
        del init_encrypted_bias_response
        del encrypted_weight_result
        del encrypted_bias_result
        del ckks_params
        del storage_backend
        gc.collect()


@router.post("/pplr/predict", response_model=PPLRPredictResponse)
async def pplr_predict(
    body: PPLRPredictWorkerRequest,
    logger=Depends(get_logger),
    storage_client=Depends(get_storage_client),
    ckks=Depends(get_ckks),
    settings: Settings = Depends(get_settings),
):
    encrypted_matrix_test = None
    encrypted_weights = None
    encrypted_bias = None
    encrypted_predictions = None
    encrypted_matrix_test_result = None
    encrypted_matrix_test_response = None
    encrypted_weights_result = None
    encrypted_weights_response = None
    encrypted_bias_result = None
    encrypted_bias_response = None
    encrypted_predictions_result = None
    encrypted_predictions_response = None
    ckks_params = None
    storage_backend = None
    sb_put = None

    try:
        local_start_time = time.time()
        worker_id = settings.node_id
        STORAGE_CLIENT = storage_client
        BUCKET_ID: str = settings.mictlanx_bucket_id
        algorithm = Constants.MachineLearningAlgorithms.PPLR_PREDICT
        experiment_id = body.experiment_id
        encrypted_matrix_test_id = body.encrypted_matrix_test_id
        encrypted_weights_id = body.encrypted_weights_id
        encrypted_bias_id = body.encrypted_bias_id
        scale = int(body.scale)
        n_features = int(body.n_features)
        encrypted_predictions_id = "{}encryptedpredictions".format(encrypted_matrix_test_id)
        num_chunks = int(body.num_chunks)

        if not all([encrypted_matrix_test_id, encrypted_weights_id, encrypted_bias_id]):
            raise HTTPException(status_code=400, detail="Missing mandatory IDs or shape parameters")

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
            StorageBuilder(storage_client=STORAGE_CLIENT, scheme=Scheme.CKKS)
            .with_ckks(ckks)
            .with_ckks_params(ckks_params=ckks_params)
            .with_storage_params(StorageParams(num_chunks=num_chunks, timeout=MICTLANX_TIMEOUT))
            .build()
        )

        encrypted_matrix_test_result = await storage_backend.get(
            bucket_id=BUCKET_ID,
            ball_id=encrypted_matrix_test_id,
            segment=True,
            encrypt=True,
            scheme=Scheme.CKKS,
        )

        if encrypted_matrix_test_result.is_err:
            logger.error(f"Failed to get encrypted matrix test: {encrypted_matrix_test_result.unwrap_err()}")
            raise HTTPException(status_code=500, detail="Failed to get encrypted matrix test")

        encrypted_matrix_test_response = encrypted_matrix_test_result.unwrap()
        encrypted_matrix_test = encrypted_matrix_test_response.raw_value
        logger.debug({
            "event": "GET",
            "experiment_id": experiment_id,
            "bucket_id": BUCKET_ID,
            "ball_id": encrypted_matrix_test_id,
            "matrix_id": encrypted_matrix_test_id,
            "shape": str((0, n_features)),
            "dtype": "PyCtxt",
            "read_time": encrypted_matrix_test_response.read_time,
        })

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
            "shape": str((1, n_features)),
            "read_time": encrypted_weights_response.read_time,
        })

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
            "read_time": encrypted_bias_response.read_time,
        })

        start_time_predict = time.time()
        encrypted_predictions = PPLR.predict(
            HE=ckks.he_object,
            encrypted_matrix=encrypted_matrix_test,
            encrypted_weights=encrypted_weights[0],
            encrypted_bias=encrypted_bias[0],
            scale=scale,
            n_features=n_features,
        )
        end_time_predict = time.time() - start_time_predict
        logger.debug({
            "event": "PREDICT",
            "experiment_id": experiment_id,
            "encrypted_matrix_id": encrypted_matrix_test_id,
            "encrypted_weights_id": encrypted_weights_id,
            "encrypted_bias_id": encrypted_bias_id,
            "n_features": n_features,
            "scale": scale,
            "predict_time": end_time_predict,
        })

        sb_put = storage_backend.as_builder().with_storage_params(StorageParams(num_chunks=1, timeout=MICTLANX_TIMEOUT)).build()
        encrypted_predictions_result = await sb_put.put(
            bucket_id=BUCKET_ID,
            data=encrypted_predictions,
            ball_id=encrypted_predictions_id,
            delete=True,
            segment=True,
            encrypt=False,
            scheme=Scheme.CKKS,
        )

        if encrypted_predictions_result.is_err:
            logger.error("Failed to put encrypted predictions in cloud storage: {}".format(encrypted_predictions_result.unwrap_err()))
            raise HTTPException(status_code=500, detail="Failed to put encrypted weights in cloud storage")

        encrypted_predictions_response = encrypted_predictions_result.unwrap()
        logger.debug({
            "event": "PUT",
            "experiment_id": experiment_id,
            "bucket_id": BUCKET_ID,
            "ball_id": encrypted_predictions_id,
            "matrix_id": encrypted_predictions_id,
            "shape": str(encrypted_predictions_response.shape),
            "dtype": str(encrypted_predictions_response.dtype),
            "read_time": getattr(encrypted_predictions_response, "read_time", 0.0),
            "segment_time": getattr(encrypted_predictions_response, "segment_time", 0.0),
            "encrypt_time": getattr(encrypted_predictions_response, "encrypt_time", 0.0),
            "upload_time": getattr(encrypted_predictions_response, "upload_time", 0.0),
        })

        end_time = time.time()
        service_time = end_time - local_start_time

        logger.info(ExperimentLogEntry(
            event="COMPLETED",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=local_start_time,
            end_time=time.time(),
            id=encrypted_matrix_test_id,
            worker_id=worker_id,
            worker_time=service_time,
        ).model_dump())

        return {
            "encrypted_predictions_id": encrypted_predictions_id,
            "predict_time": end_time_predict,
            "service_time": service_time,
            "algorithm": algorithm,
        }
    finally:
        del encrypted_matrix_test
        del encrypted_weights
        del encrypted_bias
        del encrypted_predictions
        del encrypted_matrix_test_result
        del encrypted_matrix_test_response
        del encrypted_weights_result
        del encrypted_weights_response
        del encrypted_bias_result
        del encrypted_bias_response
        del encrypted_predictions_result
        del encrypted_predictions_response
        del ckks_params
        del storage_backend
        del sb_put
        gc.collect()
