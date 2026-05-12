import time, json
import numpy as np
import numpy.typing as npt
from typing import List,Tuple
from flask import Blueprint,current_app,request,Response
from rory.core.machine_learning.secure.pqc.lite_pplr import PPLR
from rory.core.machine_learning.logistic_regression import LogisticRegressionBaseline
from rory.core.utils.utils import Utils
from rory.core.utils.constants import Constants
from rory.core.security.cryptosystem.pqc.ckks import Ckks
from rorycommon import StorageBuilder, StorageParams, Scheme, CkksParams
from mictlanx import AsyncClient
from option import Result, Some
from mictlanx.utils.segmentation import Chunks
from option import Option,Some,NONE
from Pyfhel import PyCtxt,Pyfhel
from models import ExperimentLogEntry

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
async def logistic_regression():
    local_start_time            = time.time()
    logger                      = current_app.config["logger"]
    STORAGE_CLIENT: AsyncClient = current_app.config["ASYNC_STORAGE_CLIENT"]
    BUCKET_ID: str              = current_app.config.get("BUCKET_ID", "rory")
    headers                     = request.headers
    experiment_id               = headers.get("Experiment-Id","")
    iterations                  = int(headers.get("Iterations", 1))
    algorithm                   = Constants.MachineLearningAlgorithms.LOGISTIC_REGRESSION_TRAIN
    plaintext_matrix_train_id   = headers.get("Plaintext-Matrix-Train-Id","train_x")
    plaintext_matrix_train_label_id = headers.get("Plaintext-Matrix-Train-Label-Id","train_y")
    epochs                      = int(headers.get("Epochs", 1))
    learning_rate               = float(headers.get("Learning-Rate", "0.01"))
    if not all([plaintext_matrix_train_id, plaintext_matrix_train_label_id]):
        return Response("Missing mandatory IDs or shape parameters", status=400)
    MICTLANX_TIMEOUT            = int(current_app.config.get("MICTLANX_TIMEOUT", 3600))
    MICTLANX_DELAY              = int(current_app.config.get("MICTLANX_DELAY","2"))
    MICTLANX_BACKOFF_FACTOR     = float(current_app.config.get("MICTLANX_BACKOFF_FACTOR","0.5"))
    MICTLANX_MAX_RETRIES        = int(current_app.config.get("MICTLANX_MAX_RETRIES", 10))
    
    logger.debug({
        "algorithm" : algorithm,
        "experiment_id" : experiment_id,
        "plaintext_matrix_train_id": plaintext_matrix_train_id,
        "plaintext_matrix_train_label_id": plaintext_matrix_train_label_id,
        "epoch": epochs, 
        "learning_rate": learning_rate, 
    })

    return Response(
        response=json.dumps({
                "message": "Logistic regression training completed successfully",
                "algorithm": algorithm,
                "experiment_id": experiment_id,
        }),
        status=200,
        headers  = {}
        )


@machinelearning.route("/pplr/train", methods=["POST"])
async def pplr_train():
    local_start_time            = time.time()
    logger                      = current_app.config["logger"]
    worker_id                   = current_app.config["NODE_ID"]
    STORAGE_CLIENT: AsyncClient = current_app.config["ASYNC_STORAGE_CLIENT"]
    BUCKET_ID: str              = current_app.config.get("BUCKET_ID", "rory")
    request_headers             = request.headers
    algorithm                   = Constants.MachineLearningAlgorithms.PPLR_TRAIN
    experiment_id               = request_headers.get("Experiment-Id", "")
    epochs                      = int(request_headers.get("Epochs", 1))
    learning_rate               = float(request_headers.get("Learning-Rate", "0.01"))
    encrypted_matrix_train_id       = request_headers.get("Encrypted-Matrix-Train-Id")
    encrypted_label_vector_train_id = request_headers.get("Encrypted-Label-Vector-Train-Id")
    encrypted_weights_id            = request_headers.get("Encrypted-Weights-Id")
    encrypted_bias_id               = request_headers.get("Encrypted-Bias-Id")
    scale                           = int(request_headers.get("Scale", 40))                   # Escala para Pyfhel
    n_features                      = int(request_headers.get("N-Features", 0))
    n_samples                       = int(request_headers.get("N-Samples", 0))
    num_chunks                      = int(request_headers.get("Num-Chunks",-1))
    
    if not all([encrypted_matrix_train_id,encrypted_weights_id,encrypted_bias_id,encrypted_label_vector_train_id]):
        return Response("Missing mandatory IDs or shape parameters", status=400)
    
    MICTLANX_TIMEOUT        = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
    MICTLANX_DELAY          = int(current_app.config.get("MICTLANX_DELAY","2"))
    MICTLANX_BACKOFF_FACTOR = float(current_app.config.get("MICTLANX_BACKOFF_FACTOR","0.5"))
    MICTLANX_MAX_RETRIES    = int(current_app.config.get("MICTLANX_MAX_RETRIES","10"))
    _round                  = bool(int(current_app.config.get("_round","0")))                 #False
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
    #___________________

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
    encrypted_matrix_train = encrypted_matrix_train_result.unwrap().raw_value

    logger.debug({
        "msg": "encrypted matrix train get from storage",
        "encrypted_matrix_train_id": encrypted_matrix_train_id
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
    encrypted_label_vector_train = encrypted_label_vector_train_result.unwrap().raw_value
    # print("Encrypted label vector train retrieved successfully",encrypted_label_vector_train)

    
    logger.debug({
        "msg": "encrypted label vector train get from storage",
        "encrypted_label_vector_train_id": encrypted_label_vector_train_id,
        "type": str(type(encrypted_label_vector_train)),
        # "value":str(encrypted_label_vector_train)
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
    init_encrypted_weights = init_encrypted_weights_result.unwrap().raw_value

    logger.debug({
        "msg": "encrypted weight get from storage",
        "encrypted_weight_id": encrypted_weights_id,
        "type": str(type(init_encrypted_weights)),
        # "value":str(init_encrypted_weights)
    })

    
    init_encrypted_bias_result = await storage_backend.get(
        bucket_id = BUCKET_ID,
        ball_id   = encrypted_bias_id,
        segment   = True,
        encrypt   = True,
        scheme    = Scheme.CKKS
    )
    # logger.debug({
    #     "type":"BIAS",
    #     # "msg":str(init_encrypted_bias_result)
    # })
    
    if init_encrypted_bias_result.is_err:
        logger.error(f"Failed to get init encrypted bias: {init_encrypted_bias_result.unwrap_err()}")
        return Response(status=500, response="Failed to get init encrypted bias")
    init_encrypted_bias = init_encrypted_bias_result.unwrap().raw_value

    logger.debug({
        "msg": "encrypted bias get from storage",
        "encrypted_bias_id": encrypted_bias_id,
        "type": str(type(init_encrypted_bias)),
        # "value":str(init_encrypted_bias)
    })

    # time.sleep(1000)

    encrypted_weights, encrypted_bias = PPLR.train(
        HE                = ckks.he_object,
        epochs            = epochs,
        learning_rate     = learning_rate,
        encrypted_weights = init_encrypted_weights[0],
        encrypted_bias    = init_encrypted_bias[0],
        encrypted_X       = encrypted_matrix_train,
        encrypted_y       = encrypted_label_vector_train,
        n_features        = n_features,
        scale             = scale,
        n_samples         = n_samples
    )
    
    logger.debug({
            "msg": "Finish train",
            "type": str(type(encrypted_weights)),
            # "value":str(encrypted_weights),
            "type": str(type(encrypted_bias)),
            # "value":str(encrypted_bias),

    })
    # time.sleep(1000)
    del init_encrypted_weights
    del init_encrypted_bias

    sb_put = storage_backend.as_builder().with_storage_params(StorageParams(num_chunks=1, timeout=MICTLANX_TIMEOUT)).build()
    encrypted_weight_result = await sb_put.put(
        bucket_id = BUCKET_ID,
        data      = [encrypted_weights],
        ball_id   = encrypted_weights_id,
        delete    = True,
        segment   = True,
        encrypt   = False,
        scheme    = Scheme.CKKS,
    )
    # logger.debug({
    #     "msg":str(encrypted_weight_result)
    # })

    if encrypted_weight_result.is_err:
        logger.error("Failed to put encrypted weights in cloud storage: {}".format(encrypted_weight_result.unwrap_err()))
        return Response(status=500, response="Failed to put encrypted weights in cloud storage")
    encrypted_weight_response = encrypted_weight_result.unwrap()

    encrypted_bias_result = await sb_put.put(
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

    return Response(
            response = json.dumps({
                "encrypted_weights_id":encrypted_weights_id,
                "encrypted_bias_id":encrypted_bias_id,       
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
    iterations                  = int(request_headers.get("Iterations", 1))
    encrypted_matrix_test_id    = request_headers.get("Encrypted-Matrix-Test-Id")
    encrypted_weights_id        = request_headers.get("Encrypted-Weights-Id")
    encrypted_bias_id           = request_headers.get("Encrypted-Bias-Id")
    scale                       = int(request_headers.get("Scale", 40)) # Escala para Pyfhel
    n_features                  = int(request_headers.get("N-Features", 0))
    num_chunks                  = int(request_headers.get("Num-Chunks",-1))
    
    if not all([encrypted_matrix_test_id,encrypted_weights_id,encrypted_bias_id]):
        return Response("Missing mandatory IDs or shape parameters", status=400)
    
    MICTLANX_TIMEOUT             = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
    MICTLANX_DELAY               = int(current_app.config.get("MICTLANX_DELAY","2"))
    MICTLANX_BACKOFF_FACTOR      = float(current_app.config.get("MICTLANX_BACKOFF_FACTOR","0.5"))
    MICTLANX_MAX_RETRIES         = int(current_app.config.get("MICTLANX_MAX_RETRIES","10"))
    _round                       = bool(int(current_app.config.get("_round","0"))) #False
    decimals                     = int(current_app.config.get("DECIMALS","4"))
    keys_path                    = current_app.config.get("KEYS_PATH","/rory/keys")
    ctx_filename                 = current_app.config.get("CTX_FILENAME","ctx")
    pubkey_filename              = current_app.config.get("PUBKEY_FILENAME","pubkey")
    secretkey_filename           = current_app.config.get("SECRET_KEY_FILENAME","secretkey")
    relinkey_filename            = current_app.config.get("RELINKEY_FILENAME","relinkey")
    rotatekey_filename           = current_app.config.get("ROTATEKEY_FILENAME","rotatekey")
    
    logger.debug({
            "algorithm" : algorithm,
            "encrypted_matrix_test_id": encrypted_matrix_test_id,
            "encrypted_weight_matrix_id": encrypted_weights_id,
            "encrypted_bias_train_id": encrypted_bias_id,
            "scale": scale,
            "n_features": n_features,
            "max_iterations": iterations,
        })
    
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

    logger.debug({
            "msg": "storage_backend created"
        })  

    return Response(
            response = json.dumps({
                "message": "PPLR prediction completed successfully",
            }),
            status   = 200,
            headers  = {}
            )