import time, json
from flask import Blueprint,current_app,request,Response
from rory.core.classification.secure.pqc.pplr import PPLR
from rory.core.classification.logistic_regression import LogisticRegression
from rory.core.utils.constants import Constants
from rory.core.security.cryptosystem.pqc.ckks import Ckks
from rorycommon import StorageBuilder, StorageParams, Scheme, CkksParams
from models import ExperimentLogEntry
from mictlanx import AsyncClient

machinelearning = Blueprint("machinelearning", __name__, url_prefix="/machine-learning")

@machinelearning.route("/test", methods=["GET", "POST"])
def test():
     """Health check and component identification endpoint for the Worker node.
    This method serves as a heartbeat signal for the Rory Manager, allowing the orchestrator to confirm 
    the node's availability and its specific role within the PPDMaaS ecosystem. It returns the component 
    type both in the JSON payload and the HTTP response headers to facilitate automated discovery and 
    load balancing.

    Note:
        **Infrastructure Check**: This endpoint does not require cryptographic parameters or session identifiers, making it the primary tool for connectivity troubleshooting.

    Returns:
        Response: A Flask Response object with a 200 status containing a JSON payload with:
            component_type (str): The identification string "worker".
        
        Headers:
            Component-Type (str): Metadata indicating the node's functional role.
    """
     return Response(
        response=json.dumps({"component_type": "worker"}),
        status=200,
        headers={"Component-Type": "worker"}
    )

@machinelearning.route("/logistic-regression/train", methods=["POST"])
async def logistic_regression_train():
    local_start_time            = time.time()
    logger                      = current_app.config["logger"]
    worker_id                   = current_app.config["NODE_ID"]
    STORAGE_CLIENT: AsyncClient = current_app.config["ASYNC_STORAGE_CLIENT"]
    BUCKET_ID: str              = current_app.config.get("BUCKET_ID", "rory")
    MICTLANX_TIMEOUT            = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
    num_chunks                  = current_app.config.get("NUM_CHUNKS",2)
    headers                     = request.headers
    experiment_id               = headers.get("Experiment-Id","")
    algorithm                   = Constants.MachineLearningAlgorithms.LOGISTIC_REGRESSION_TRAIN
    plaintext_matrix_train_id   = headers.get("Plaintext-Matrix-Train-Id","train_x")
    plaintext_label_vector_train_id = headers.get("Plaintext-Label-Vector-Train-Id","train_y")
    weights_id                  = headers.get("Weights-Id")
    bias_id                     = headers.get("Bias-Id")
    epochs                      = int(headers.get("Epochs", 1))
    learning_rate               = float(headers.get("Learning-Rate", "0.01"))
    if not all([plaintext_matrix_train_id, plaintext_label_vector_train_id]):
        return Response("Missing mandatory IDs or shape parameters", status=400)
    
    storage_backend = (
        StorageBuilder(storage_client = STORAGE_CLIENT)
        .with_storage_params(StorageParams(num_chunks=num_chunks, timeout=MICTLANX_TIMEOUT))
        .build()
    )

    plaintext_matrix_train_result = await storage_backend.get(
        bucket_id = BUCKET_ID,
        ball_id   = plaintext_matrix_train_id,
        segment   = True,
        encrypt   = False
    )
    if plaintext_matrix_train_result.is_err:
        logger.error(f"Failed to get matrix train: {plaintext_matrix_train_result.unwrap_err()}")
        return Response(status=500, response="Failed to get matrix train")

    plaintext_matrix_train_response = plaintext_matrix_train_result.unwrap()
    plaintext_matrix_train = plaintext_matrix_train_response.raw_value
    logger.debug({
        "event"        : "GET",
        "experiment_id": experiment_id,
        "bucket_id"    : BUCKET_ID,
        "ball_id"      : plaintext_matrix_train_id,
        "matrix_id"    : plaintext_matrix_train_id,
        "shape"        : str(plaintext_matrix_train.shape),
        "dtype"        : str(plaintext_matrix_train.dtype),
        "read_time"    : plaintext_matrix_train_response.read_time,
    })

    plaintext_label_vector_train_result = await storage_backend.get(
        bucket_id = BUCKET_ID,
        ball_id   = plaintext_label_vector_train_id,
        segment   = True,
        encrypt   = False
    )

    if plaintext_label_vector_train_result.is_err:
        logger.error(f"Failed to get label vector train: {plaintext_label_vector_train_result.unwrap_err()}")
        return Response(status=500, response="Failed to get label vector train") 
    plaintext_label_vector_train_response = plaintext_label_vector_train_result.unwrap()
    plaintext_label_vector_train = plaintext_label_vector_train_response.raw_value
    logger.debug({
        "event"        : "GET",
        "experiment_id": experiment_id,
        "bucket_id"    : BUCKET_ID,
        "ball_id"      : plaintext_label_vector_train_id,
        "matrix_id"    : plaintext_label_vector_train_id,
        "shape"        : str(plaintext_label_vector_train.shape),
        "dtype"        : str(plaintext_label_vector_train.dtype),
        "read_time"    : plaintext_label_vector_train_response.read_time,
    })

    start_time_train = time.time()
    weights, bias = LogisticRegression.fit(
        plaintext_matrix = plaintext_matrix_train,
        label_vector     = plaintext_label_vector_train,
        epochs           = epochs,
        learning_rate    = learning_rate,
        weights          = None,
        bias             = 0.0,
    )
    end_time_train = time.time() - start_time_train
    logger.debug({
        "event"                   : "TRAIN",
        "experiment_id"           : experiment_id,
        "encrypted_matrix_id"     : plaintext_matrix_train_id,
        "encrypted_labelvector_id": plaintext_label_vector_train_id,
        "encrypted_weights_id"    : weights_id,
        "encrypted_bias_id"       : bias_id,
        "n_features"              : plaintext_matrix_train.shape[1],
        "n_samples"               : plaintext_matrix_train.shape[0],
        "train_time"              : end_time_train,
    })

    weight_result = await storage_backend.put(
        bucket_id = BUCKET_ID,
        data      = weights,
        ball_id   = weights_id,
        segment   = True,
        encrypt   = False,
        delete    = True
    )

    if weight_result.is_err:
        logger.error("Failed to put weights in cloud storage: {}".format(weight_result.unwrap_err()))
        return Response(status=500, response="Failed to put weights in cloud storage")
    weight_response = weight_result.unwrap()
    logger.debug({
        "event"        : "PUT",
        "experiment_id": experiment_id,
        "bucket_id"    : BUCKET_ID,
        "ball_id"      : weights_id,
        "matrix_id"    : weights_id,
        "shape"        : str(weight_response.shape),
        "dtype"        : str(weight_response.dtype),
        "read_time"    : getattr(weight_response, "read_time", 0.0),
        "segment_time" : getattr(weight_response, "segment_time", 0.0),
        "encrypt_time" : getattr(weight_response, "encrypt_time", 0.0),
        "upload_time"  : getattr(weight_response, "upload_time", 0.0),
    })

    bias_result = await storage_backend.put(
        bucket_id = BUCKET_ID,
        data      = [bias],
        ball_id   = bias_id,
        segment   = False,
        encrypt   = False,
        delete    = True
    )
    
    if bias_result.is_err:
        logger.error("Failed to put bias in cloud storage: {}".format(bias_result.unwrap_err()))
        return Response(status=500, response="Failed to put bias in cloud storage")
    bias_response = bias_result.unwrap()
    logger.debug({
        "event"        : "PUT",
        "experiment_id": experiment_id,
        "bucket_id"    : BUCKET_ID,
        "ball_id"      : bias_id,
        "matrix_id"    : bias_id,
        "shape"        : str(bias_response.shape),
        "dtype"        : str(bias_response.dtype),
        "read_time"    : getattr(bias_response, "read_time", 0.0),
        "segment_time" : getattr(bias_response, "segment_time", 0.0),
        "upload_time"  : getattr(bias_response, "upload_time", 0.0),
    })

    end_time = time.time() - local_start_time

    logger.info(ExperimentLogEntry(
        event         = "COMPLETED",
        experiment_id = experiment_id,
        algorithm     = algorithm,
        start_time    = local_start_time,
        end_time      = time.time(),
        id            = plaintext_matrix_train_id,
        epochs        = epochs,
        learning_rate = learning_rate,
        worker_id     = worker_id,
        worker_time   = end_time,
    ).model_dump())
    return Response(
            response = json.dumps({
                "service_time":end_time,
                "train_time":end_time_train,   
                "algorithm":algorithm,  
            }),
            status   = 200,
            headers  = {}
            )

@machinelearning.route("/logistic-regression/predict", methods=["POST"])
async def logistic_regression_predict():
    local_start_time         = time.time()
    logger                   = current_app.config["logger"]
    worker_id                = current_app.config["NODE_ID"]
    STORAGE_CLIENT: AsyncClient = current_app.config["ASYNC_STORAGE_CLIENT"]
    BUCKET_ID: str           = current_app.config.get("BUCKET_ID", "rory")
    MICTLANX_TIMEOUT         = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
    num_chunks               = current_app.config.get("NUM_CHUNKS",2)
    headers                  = request.headers
    experiment_id            = headers.get("Experiment-Id","")
    algorithm                = Constants.MachineLearningAlgorithms.LOGISTIC_REGRESSION_PREDICT
    plaintext_matrix_train_id = headers.get("Plaintext-Matrix-Train-Id","train_x")
    plaintext_matrix_test_id = headers.get("Plaintext-Matrix-Test-Id","test_x")
    weights_id               = headers.get("Weights-Id")
    bias_id                  = headers.get("Bias-Id")
    predictions_id = "{}predictions".format(plaintext_matrix_test_id)
    if not all([plaintext_matrix_test_id, weights_id, bias_id]):
        return Response("Missing mandatory IDs or shape parameters", status=400)

    storage_backend = (
        StorageBuilder(storage_client = STORAGE_CLIENT)
        .with_storage_params(StorageParams(num_chunks=num_chunks, timeout=MICTLANX_TIMEOUT))
        .build()
    )

    plaintext_matrix_test_result = await storage_backend.get(
        bucket_id = BUCKET_ID,
        ball_id   = plaintext_matrix_test_id,
        segment   = True,
        encrypt   = False
    )
    if plaintext_matrix_test_result.is_err:
        logger.error(f"Failed to get matrix test: {plaintext_matrix_test_result.unwrap_err()}")
        return Response(status=500, response="Failed to get matrix test")
    plaintext_matrix_test_response = plaintext_matrix_test_result.unwrap()
    plaintext_matrix_test = plaintext_matrix_test_response.raw_value
    logger.debug({
        "event"        : "GET",
        "experiment_id": experiment_id,
        "bucket_id"    : BUCKET_ID,
        "ball_id"      : plaintext_matrix_test_id,
        "matrix_id"    : plaintext_matrix_test_id,
        "shape"        : str(plaintext_matrix_test.shape),
        "dtype"        : str(plaintext_matrix_test.dtype),
        "read_time"    : plaintext_matrix_test_response.read_time,
    })

    weights_result = await storage_backend.get(
        bucket_id = BUCKET_ID,
        ball_id   = weights_id,
        segment   = True,
        encrypt   = False
    )
    if weights_result.is_err:
        logger.error(f"Failed to get weights: {weights_result.unwrap_err()}")
        return Response(status=500, response="Failed to get weights")
    weights_response = weights_result.unwrap()
    weights = weights_response.raw_value
    logger.debug({
        "event"        : "GET",
        "experiment_id": experiment_id,
        "bucket_id"    : BUCKET_ID,
        "ball_id"      : weights_id,
        "matrix_id"    : weights_id,
        "shape"        : str(weights.shape),
        "dtype"        : str(weights.dtype),
        "read_time"    : weights_response.read_time,
    })

    bias_result = await storage_backend.get(
        bucket_id = BUCKET_ID,
        ball_id   = bias_id,
        segment   = False,
        encrypt   = False
    )
    if bias_result.is_err:
        logger.error(f"Failed to get bias: {bias_result.unwrap_err()}")
        return Response(status=500, response="Failed to get bias")
    bias_response = bias_result.unwrap()
    bias = bias_response.raw_value
    logger.debug({
        "event"        : "GET",
        "experiment_id": experiment_id,
        "bucket_id"    : BUCKET_ID,
        "ball_id"      : bias_id,
        "matrix_id"    : bias_id,
        "dtype"        : "float64",
        "read_time"    : bias_response.read_time,
    })

    start_time_predict = time.time()
    predictions = LogisticRegression.predict(
        plaintext_matrix = plaintext_matrix_test,
        weights               = weights,
        bias                  = bias
    )
    end_time_predict = time.time() - start_time_predict
    logger.debug({
        "event"               : "PREDICT",
        "experiment_id"       : experiment_id,
        "encrypted_matrix_id" : plaintext_matrix_test_id,
        "encrypted_weights_id": weights_id,
        "encrypted_bias_id"   : bias_id,
        "n_features"          : plaintext_matrix_test.shape[1],
        "predict_time"        : end_time_predict,
    })

    predictions_result = await storage_backend.put(
        bucket_id = BUCKET_ID,
        data      = predictions,
        ball_id   = predictions_id,
        segment   = True,
        encrypt   = False,
        delete    = True
    )

    if predictions_result.is_err:
        logger.error("Failed to put predictions in cloud storage: {}".format(predictions_result.unwrap_err()))
        return Response(status=500, response="Failed to put predictions in cloud storage")
    predictions_response = predictions_result.unwrap()
    logger.debug({
        "event"        : "PUT",
        "experiment_id": experiment_id,
        "bucket_id"    : BUCKET_ID,
        "ball_id"      : predictions_id,
        "matrix_id"    : predictions_id,
        "shape"        : str(predictions_response.shape),
        "dtype"        : str(predictions_response.dtype),
        "read_time"    : getattr(predictions_response, "read_time", 0.0),
        "segment_time" : getattr(predictions_response, "segment_time", 0.0),
        "encrypt_time" : getattr(predictions_response, "encrypt_time", 0.0),
        "upload_time"  : getattr(predictions_response, "upload_time", 0.0),
    })

    end_time     = time.time()
    service_time = end_time - local_start_time

    logger.info(ExperimentLogEntry(
        event         = "COMPLETED",
        experiment_id = experiment_id,
        algorithm     = algorithm,
        start_time    = local_start_time,
        end_time      = time.time(),
        id            = plaintext_matrix_train_id,
        worker_id     = worker_id,
        worker_time   = service_time,
    ).model_dump())

    return Response(
        response = json.dumps({
            "predictions_id": predictions_id,
            "predict_time": end_time_predict,
            "service_time":service_time
        }),
        status   = 200,
        headers  = {}
        )


@machinelearning.route("/pplr/train", methods=["POST"])
async def pplr_train():
    local_start_time            = time.time()
    logger                      = current_app.config["logger"]
    worker_id                   = current_app.config["NODE_ID"]
    STORAGE_CLIENT: AsyncClient = current_app.config["ASYNC_STORAGE_CLIENT"]
    BUCKET_ID: str              = current_app.config.get("BUCKET_ID", "rory")
    MICTLANX_TIMEOUT            = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
    num_chunks                  = current_app.config.get("NUM_CHUNKS",2)
    request_headers             = request.headers
    algorithm                   = Constants.MachineLearningAlgorithms.PPLR_TRAIN
    experiment_id               = request_headers.get("Experiment-Id", "")
    learning_rate               = float(request_headers.get("Learning-Rate", "0.01"))
    encrypted_matrix_train_id       = request_headers.get("Encrypted-Matrix-Train-Id")
    encrypted_label_vector_train_id = request_headers.get("Encrypted-Label-Vector-Train-Id")
    encrypted_weights_id            = request_headers.get("Encrypted-Weights-Id")
    encrypted_bias_id               = request_headers.get("Encrypted-Bias-Id")
    scale                           = int(request_headers.get("Scale", 40))
    n_features                      = int(request_headers.get("N-Features", 0))
    n_samples                       = int(request_headers.get("N-Samples", 0))
    num_chunks                      = int(request_headers.get("Num-Chunks",-1))
    
    if not all([encrypted_matrix_train_id,encrypted_weights_id,encrypted_bias_id,encrypted_label_vector_train_id]):
        return Response("Missing mandatory IDs or shape parameters", status=400)
    
    _round                  = bool(int(current_app.config.get("_round","0")))
    decimals                = int(current_app.config.get("DECIMALS","4"))
    keys_path               = current_app.config.get("KEYS_PATH","/rory/keys")
    ctx_filename            = current_app.config.get("CTX_FILENAME","ctx")
    pubkey_filename         = current_app.config.get("PUBKEY_FILENAME","pubkey")
    secretkey_filename      = current_app.config.get("SECRET_KEY_FILENAME","secretkey")
    relinkey_filename       = current_app.config.get("RELINKEY_FILENAME","relinkey")
    rotatekey_filename      = current_app.config.get("ROTATEKEY_FILENAME","rotatekey")
    
    
    ckks = Ckks.from_pyfhel_server(
        _round             = _round,
        decimals           = decimals,
        path               = keys_path,
        ctx_filename       = ctx_filename,
        pubkey_filename    = pubkey_filename,
        relinkey_filename  = relinkey_filename,
        rotatekey_filename = rotatekey_filename
    )
    
    ckks_params = CkksParams(
        keys_path          = keys_path,
        ctx_filename       = ctx_filename,
        pubkey_filename    = pubkey_filename,
        secretkey_filename = secretkey_filename,
        relinkey_filename  = relinkey_filename,
        rotatekey_filename = rotatekey_filename,
        decimals           = decimals,
        _round             = _round
    )

    storage_backend = (
        StorageBuilder(storage_client = STORAGE_CLIENT, scheme = Scheme.CKKS)
        .with_ckks(ckks)
        .with_ckks_params(ckks_params=ckks_params)
        .with_storage_params(StorageParams(num_chunks=2, timeout=300))
        .build()
    )

    encrypted_matrix_train_result = await storage_backend.get(
        bucket_id = BUCKET_ID,
        ball_id   = encrypted_matrix_train_id,
        segment   = True,
        encrypt   = True,
        scheme    = Scheme.CKKS
    )
    if encrypted_matrix_train_result.is_err:
        logger.error(f"Failed to get encrypted matrix train: {encrypted_matrix_train_result.unwrap_err()}")
        return Response(status=500, response="Failed to get encrypted matrix train")
    encrypted_matrix_train_response = encrypted_matrix_train_result.unwrap()
    encrypted_matrix_train = encrypted_matrix_train_response.raw_value
    logger.debug({
        "event"        : "GET",
        "experiment_id": experiment_id,
        "bucket_id"    : BUCKET_ID,
        "ball_id"      : encrypted_matrix_train_id,
        "matrix_id"    : encrypted_matrix_train_id,
        "shape"        : str((n_samples, n_features)),
        "read_time"    : encrypted_matrix_train_response.read_time,
    })

    encrypted_label_vector_train_result = await storage_backend.get(
        bucket_id = BUCKET_ID,
        ball_id   = encrypted_label_vector_train_id,
        segment   = True,
        encrypt   = True,
        scheme    = Scheme.CKKS
    )

    if encrypted_label_vector_train_result.is_err:
        logger.error(f"Failed to get encrypted label vector train: {encrypted_label_vector_train_result.unwrap_err()}")
        return Response(status=500, response="Failed to get encrypted label vector train") 
    encrypted_label_vector_train_response = encrypted_label_vector_train_result.unwrap()
    encrypted_label_vector_train = encrypted_label_vector_train_response.raw_value
    logger.debug({
        "event"        : "GET",
        "experiment_id": experiment_id,
        "bucket_id"    : BUCKET_ID,
        "ball_id"      : encrypted_label_vector_train_id,
        "matrix_id"    : encrypted_label_vector_train_id,
        "shape"        : str((n_samples, 1)),
        "dtype"        : "PyCtxt",
        "read_time"    : encrypted_label_vector_train_response.read_time,
    })

    init_encrypted_weights_result = await storage_backend.get(
        bucket_id = BUCKET_ID,
        ball_id   = encrypted_weights_id,
        segment   = True,
        encrypt   = True,
        scheme    = Scheme.CKKS
    )
  
    if init_encrypted_weights_result.is_err:
        logger.error(f"Failed to get init encrypted weights: {init_encrypted_weights_result.unwrap_err()}")
        return Response(status=500, response="Failed to get init encrypted weights")
    init_encrypted_weights_response = init_encrypted_weights_result.unwrap()
    init_encrypted_weights = init_encrypted_weights_response.raw_value
    logger.debug({
        "event"        : "GET",
        "experiment_id": experiment_id,
        "bucket_id"    : BUCKET_ID,
        "ball_id"      : encrypted_weights_id,
        "matrix_id"    : encrypted_weights_id,
        "shape"        : str((1, n_features)),
        "read_time"    : init_encrypted_weights_response.read_time,
    })

    init_encrypted_bias_result = await storage_backend.get(
        bucket_id = BUCKET_ID,
        ball_id   = encrypted_bias_id,
        segment   = True,
        encrypt   = True,
        scheme    = Scheme.CKKS
    )

    if init_encrypted_bias_result.is_err:
        logger.error(f"Failed to get init encrypted bias: {init_encrypted_bias_result.unwrap_err()}")
        return Response(status=500, response="Failed to get init encrypted bias")
    init_encrypted_bias_response = init_encrypted_bias_result.unwrap()
    init_encrypted_bias = init_encrypted_bias_response.raw_value
    logger.debug({
        "event"        : "GET",
        "experiment_id": experiment_id,
        "bucket_id"    : BUCKET_ID,
        "ball_id"      : encrypted_bias_id,
        "matrix_id"    : encrypted_bias_id,
        "shape"        : str((1,)),
        "read_time"    : init_encrypted_bias_response.read_time,
    })

    start_time_train = time.time()
    encrypted_weights, encrypted_bias = PPLR.fit(
        HE                     = ckks.he_object,
        learning_rate          = learning_rate,
        encrypted_weights      = init_encrypted_weights[0],
        encrypted_bias         = init_encrypted_bias[0],
        encrypted_matrix       = encrypted_matrix_train,
        encrypted_labelvector = encrypted_label_vector_train,
        n_features             = n_features,
        scale                  = scale,
        n_samples              = n_samples
    )
    end_time_train = time.time() - start_time_train
    logger.debug({
        "event"                   : "TRAIN",
        "experiment_id"           : experiment_id,
        "encrypted_matrix_id"     : encrypted_matrix_train_id,
        "encrypted_labelvector_id": encrypted_label_vector_train_id,
        "encrypted_weights_id"    : encrypted_weights_id,
        "encrypted_bias_id"       : encrypted_bias_id,
        "n_features"              : n_features,
        "n_samples"               : n_samples,
        "scale"                   : scale,
        "train_time"              : end_time_train,
    })

    del init_encrypted_weights
    del init_encrypted_bias

    encrypted_weight_result = await storage_backend.put(
        bucket_id = BUCKET_ID,
        data      = [encrypted_weights],
        ball_id   = encrypted_weights_id,
        delete    = True,
        segment   = True,
        encrypt   = False,
        scheme    = Scheme.CKKS,
    )
    if encrypted_weight_result.is_err:
        logger.error("Failed to put encrypted weights in cloud storage: {}".format(encrypted_weight_result.unwrap_err()))
        return Response(status=500, response="Failed to put encrypted weights in cloud storage")
    encrypted_weight_response = encrypted_weight_result.unwrap()
    logger.debug({
        "event"        : "PUT",
        "experiment_id": experiment_id,
        "bucket_id"    : BUCKET_ID,
        "ball_id"      : encrypted_weights_id,
        "matrix_id"    : encrypted_weights_id,
        "shape"        : str(encrypted_weight_response.shape),
        "dtype"        : str(encrypted_weight_response.dtype),
        "read_time"    : getattr(encrypted_weight_response, "read_time", 0.0),
        "segment_time" : getattr(encrypted_weight_response, "segment_time", 0.0),
        "encrypt_time" : getattr(encrypted_weight_response, "encrypt_time", 0.0),
        "upload_time"  : getattr(encrypted_weight_response, "upload_time", 0.0),
    })

    encrypted_bias_result = await storage_backend.put(
        bucket_id = BUCKET_ID,
        data      = [encrypted_bias],
        ball_id   = encrypted_bias_id,
        segment   = False,
        encrypt   = False,
        scheme    = Scheme.CKKS,
        delete    = True
    )
    
    if encrypted_bias_result.is_err:
        logger.error("Failed to put encrypted bias in cloud storage: {}".format(encrypted_bias_result.unwrap_err()))
        return Response(status=500, response="Failed to put encrypted bias in cloud storage")
    encrypted_bias_response = encrypted_bias_result.unwrap()
    logger.debug({
        "event"        : "PUT",
        "experiment_id": experiment_id,
        "bucket_id"    : BUCKET_ID,
        "ball_id"      : encrypted_bias_id,
        "matrix_id"    : encrypted_bias_id,
        "shape"        : str(encrypted_bias_response.shape),
        "dtype"        : str(encrypted_bias_response.dtype),
        "read_time"    : getattr(encrypted_bias_response, "read_time", 0.0),
        "segment_time" : getattr(encrypted_bias_response, "segment_time", 0.0),
        "encrypt_time" : getattr(encrypted_bias_response, "encrypt_time", 0.0),
        "upload_time"  : getattr(encrypted_bias_response, "upload_time", 0.0),
    })

    end_time = time.time() - local_start_time

    logger.info(ExperimentLogEntry(
        event         = "COMPLETED",
        experiment_id = experiment_id,
        algorithm     = algorithm,
        start_time    = local_start_time,
        end_time      = time.time(),
        id            = encrypted_matrix_train_id,
        learning_rate = learning_rate,
        worker_id     = worker_id,
        worker_time   = end_time,
    ).model_dump())

    return Response(
            response = json.dumps({
                "service_time":end_time,  
                "train_time":end_time_train, 
                "algorithm":algorithm,
            }),
            status   = 200,
            headers  = {}
            )
    
@machinelearning.route("/pplr/predict", methods=["POST"])
async def pplr_predict():
    local_start_time            = time.time()
    logger                      = current_app.config["logger"]
    worker_id                   = current_app.config["NODE_ID"]
    STORAGE_CLIENT: AsyncClient = current_app.config["ASYNC_STORAGE_CLIENT"]
    BUCKET_ID: str              = current_app.config.get("BUCKET_ID", "rory")
    request_headers             = request.headers
    algorithm                   = Constants.MachineLearningAlgorithms.PPLR_PREDICT
    experiment_id               = request_headers.get("Experiment-Id", "")
    encrypted_matrix_test_id    = request_headers.get("Encrypted-Matrix-Test-Id")
    encrypted_weights_id        = request_headers.get("Encrypted-Weights-Id")
    encrypted_bias_id           = request_headers.get("Encrypted-Bias-Id")
    scale                       = int(request_headers.get("Scale", 40)) # Escala para Pyfhel
    n_features                  = int(request_headers.get("N-Features", 0))
    encrypted_predictions_id    = "{}encryptedpredictions".format(encrypted_matrix_test_id)
    
    if not all([encrypted_matrix_test_id,encrypted_weights_id,encrypted_bias_id]):
        return Response("Missing mandatory IDs or shape parameters", status=400)
    
    MICTLANX_TIMEOUT   = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
    _round             = bool(int(current_app.config.get("_round","0")))            #False
    decimals           = int(current_app.config.get("DECIMALS","4"))
    keys_path          = current_app.config.get("KEYS_PATH","/rory/keys")
    ctx_filename       = current_app.config.get("CTX_FILENAME","ctx")
    pubkey_filename    = current_app.config.get("PUBKEY_FILENAME","pubkey")
    secretkey_filename = current_app.config.get("SECRET_KEY_FILENAME","secretkey")
    relinkey_filename  = current_app.config.get("RELINKEY_FILENAME","relinkey")
    rotatekey_filename = current_app.config.get("ROTATEKEY_FILENAME","rotatekey")
    
    ckks = Ckks.from_pyfhel_server(
        _round             = _round,
        decimals           = decimals,
        path               = keys_path,
        ctx_filename       = ctx_filename,
        pubkey_filename    = pubkey_filename,
        relinkey_filename  = relinkey_filename,
        rotatekey_filename = rotatekey_filename
    )
    
    ckks_params = CkksParams(
        keys_path          = keys_path,
        ctx_filename       = ctx_filename,
        pubkey_filename    = pubkey_filename,
        secretkey_filename = secretkey_filename,
        relinkey_filename  = relinkey_filename,
        rotatekey_filename = rotatekey_filename,
        decimals           = decimals,
        _round             = _round
    )

    storage_backend = (
        StorageBuilder(storage_client = STORAGE_CLIENT, scheme = Scheme.CKKS)
        .with_ckks(ckks)
        .with_ckks_params(ckks_params=ckks_params)
        .with_storage_params(StorageParams(num_chunks=2, timeout=300))
        .build()
    )

    encrypted_matrix_test_result = await storage_backend.get(
        bucket_id = BUCKET_ID,
        ball_id   = encrypted_matrix_test_id,
        segment   = True,
        encrypt   = True,
        scheme    = Scheme.CKKS
    )

    if encrypted_matrix_test_result.is_err:
        logger.error(f"Failed to get encrypted matrix test: {encrypted_matrix_test_result.unwrap_err()}")
        return Response(status=500, response="Failed to get encrypted matrix test") 
    encrypted_matrix_test_response = encrypted_matrix_test_result.unwrap()
    encrypted_matrix_test = encrypted_matrix_test_response.raw_value
    logger.debug({
        "event"        : "GET",
        "experiment_id": experiment_id,
        "bucket_id"    : BUCKET_ID,
        "ball_id"      : encrypted_matrix_test_id,
        "matrix_id"    : encrypted_matrix_test_id,
        "shape"        : str((0, n_features)),
        "dtype"        : "PyCtxt",
        "read_time"    : encrypted_matrix_test_response.read_time,
    })

    encrypted_weights_result = await storage_backend.get(
        bucket_id = BUCKET_ID,
        ball_id   = encrypted_weights_id,
        segment   = True,
        encrypt   = True,
        scheme    = Scheme.CKKS
    )

    if encrypted_weights_result.is_err:
        logger.error(f"Failed to get encrypted weights: {encrypted_weights_result.unwrap_err()}")
        return Response(status=500, response="Failed to get encrypted weights") 
    encrypted_weights_response = encrypted_weights_result.unwrap()
    encrypted_weights = encrypted_weights_response.raw_value
    logger.debug({
        "event"        : "GET",
        "experiment_id": experiment_id,
        "bucket_id"    : BUCKET_ID,
        "ball_id"      : encrypted_weights_id,
        "matrix_id"    : encrypted_weights_id,
        "shape"        : str((1, n_features)),
        "read_time"    : encrypted_weights_response.read_time,
    })

    encrypted_bias_result = await storage_backend.get(
        bucket_id = BUCKET_ID,
        ball_id   = encrypted_bias_id,
        segment   = True,
        encrypt   = True,
        scheme    = Scheme.CKKS
    )

    if encrypted_bias_result.is_err:
        logger.error(f"Failed to get encrypted bias: {encrypted_bias_result.unwrap_err()}")
        return Response(status=500, response="Failed to get encrypted bias") 
    encrypted_bias_response = encrypted_bias_result.unwrap()
    encrypted_bias = encrypted_bias_response.raw_value
    logger.debug({
        "event"        : "GET",
        "experiment_id": experiment_id,
        "bucket_id"    : BUCKET_ID,
        "ball_id"      : encrypted_bias_id,
        "matrix_id"    : encrypted_bias_id,
        "read_time"    : encrypted_bias_response.read_time,
    })

    start_time_predict = time.time()
    encrypted_predictions = PPLR.predict(
        HE                = ckks.he_object,
        encrypted_matrix  = encrypted_matrix_test, 
        encrypted_weights = encrypted_weights[0], 
        encrypted_bias    = encrypted_bias[0],
        scale             = scale,
        n_features        = n_features
    )
    end_time_predict = time.time() - start_time_predict
    logger.debug({
        "event"               : "PREDICT",
        "experiment_id"       : experiment_id,
        "encrypted_matrix_id" : encrypted_matrix_test_id,
        "encrypted_weights_id": encrypted_weights_id,
        "encrypted_bias_id"   : encrypted_bias_id,
        "n_features"          : n_features,
        "scale"               : scale,
        "predict_time"        : end_time_predict,
    })

    sb_put = storage_backend.as_builder().with_storage_params(StorageParams(num_chunks=1, timeout=MICTLANX_TIMEOUT)).build()
    encrypted_predictions_result = await sb_put.put(
        bucket_id = BUCKET_ID,
        data      = encrypted_predictions,
        ball_id   = encrypted_predictions_id,
        delete    = True,
        segment   = True,
        encrypt   = False,
        scheme    = Scheme.CKKS,
    )

    if encrypted_predictions_result.is_err:
        logger.error("Failed to put encrypted predictions in cloud storage: {}".format(encrypted_predictions_result.unwrap_err()))
        return Response(status=500, response="Failed to put encrypted weights in cloud storage")
    encrypted_predictions_response = encrypted_predictions_result.unwrap()
    logger.debug({
        "event"        : "PUT",
        "experiment_id": experiment_id,
        "bucket_id"    : BUCKET_ID,
        "ball_id"      : encrypted_predictions_id,
        "matrix_id"    : encrypted_predictions_id,
        "shape"        : str(encrypted_predictions_response.shape),
        "dtype"        : str(encrypted_predictions_response.dtype),
        "read_time"    : getattr(encrypted_predictions_response, "read_time", 0.0),
        "segment_time" : getattr(encrypted_predictions_response, "segment_time", 0.0),
        "encrypt_time" : getattr(encrypted_predictions_response, "encrypt_time", 0.0),
        "upload_time"  : getattr(encrypted_predictions_response, "upload_time", 0.0),
    })

    end_time     = time.time()
    service_time = end_time - local_start_time

    logger.info(ExperimentLogEntry(
        event         = "COMPLETED",
        experiment_id = experiment_id,
        algorithm     = algorithm,
        start_time    = local_start_time,
        end_time      = time.time(),
        id            = encrypted_matrix_test_id,
        worker_id     = worker_id,
        worker_time   = service_time,
    ).model_dump())

    return Response(
            response = json.dumps({
                "encrypted_predictions_id": encrypted_predictions_id,
                "predict_time": end_time_predict,
                "service_time":service_time,
                "algorithm":algorithm,
            }),
            status   = 200,
            headers  = {}
            )