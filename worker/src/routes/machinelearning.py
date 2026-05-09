import time, json
import numpy as np
import numpy.typing as npt
from typing import List,Tuple
from flask import Blueprint,current_app,request,Response
from rory.core.machine_learning.secure.pqc.pplr import LogisticRegressionFHE
from rory.core.machine_learning.secure.pqc.lite_pplr import PPLR
from rory.core.machine_learning.logistic_regression import LogisticRegressionBaseline
from rory.core.utils.utils import Utils
from rory.core.utils.constants import Constants
from rory.core.security.cryptosystem.pqc.ckks import Ckks
from mictlanx import AsyncClient
from option import Result, Some
from mictlanx.utils.segmentation import Chunks
from option import Option,Some,NONE
from rorycommon import Common as RoryCommon
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
    out_predictions_id          = f"predictions_{experiment_id}_{iterations}"
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
            "out_predictions_id": out_predictions_id
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
    accuracy_threshold          = float(request_headers.get("Accuracy-Threshold", "0.80"))
    iterations                  = int(request_headers.get("Iterations", 1))
    encrypted_matrix_train_id   = request_headers.get("Encrypted-Matrix-Train-Id")
    encrypted_label_vector_train_id = request_headers.get("Encrypted-Label-Vector-Train-Id")
    encrypted_weights_id        = request_headers.get("Encrypted-Weights-Id")
    encrypted_bias_id           = request_headers.get("Encrypted-Bias-Id")
    scale                       = int(request_headers.get("Scale", 40)) # Escala para Pyfhel
    n_features                  = int(request_headers.get("N-Features", 0))
    n_samples                   = int(request_headers.get("N-Samples", 0))
    num_chunks                  = int(request_headers.get("Num-Chunks",-1))
    
    if not all([encrypted_matrix_train_id,encrypted_weights_id,encrypted_bias_id,encrypted_label_vector_train_id]):
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
            "encrypted_matrix_train_id": encrypted_matrix_train_id,
            "encrypted_vextor_train_label_id": encrypted_label_vector_train_id,
            "encrypted_weights_matrix_id": encrypted_weights_id,
            "encrypted_bias_vector_id": encrypted_bias_id,
            "epoch": epochs, 
            "learning_rate": learning_rate, 
            "accuracy_threshold": accuracy_threshold,
            "n_features": n_features,
            "n_features": n_features, 
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
    logger.debug({
             "msg": "Created Context"
         })
    
    # Leer dataset train desde el sistema de almacenamiento
    encrypted_matrix_train_result = await RoryCommon.get_pyctxt(
        client         = STORAGE_CLIENT,
        bucket_id      = BUCKET_ID,
        key            = encrypted_matrix_train_id,
        ckks           = ckks,
        max_retries    = MICTLANX_MAX_RETRIES,
        delay          = MICTLANX_DELAY,
        backoff_factor = MICTLANX_BACKOFF_FACTOR,
        timeout        = MICTLANX_TIMEOUT
    )
    
    logger.debug({
            "msg": "encrypted matrix train get from storage"
        })
    
    logger.debug({
            "chunks": num_chunks
        })
    
    encrypted_label_vector_train_result = await RoryCommon.get_pyctxt(
        client         = STORAGE_CLIENT,
        bucket_id      = BUCKET_ID,
        key            = encrypted_label_vector_train_id,
        ckks           = ckks,
        max_retries    = MICTLANX_MAX_RETRIES,
        delay          = MICTLANX_DELAY,
        backoff_factor = MICTLANX_BACKOFF_FACTOR,
        timeout        = MICTLANX_TIMEOUT
    )
    
    logger.debug({
            "msg": "encrypted label vector train get from storage "
        })
    
    init_encrypted_weights = await RoryCommon.get_pyctxt(
        client         = STORAGE_CLIENT,
        bucket_id      = BUCKET_ID,
        key            = encrypted_weights_id,
        ckks           = ckks,
        max_retries    = MICTLANX_MAX_RETRIES,
        delay          = MICTLANX_DELAY,
        backoff_factor = MICTLANX_BACKOFF_FACTOR,
        timeout        = MICTLANX_TIMEOUT
    )
    
    logger.debug({
            "msg": "encrypted weights get from storage"
        })
    
    init_encrypted_bias = await RoryCommon.get_pyctxt(
        client         = STORAGE_CLIENT,
        bucket_id      = BUCKET_ID,
        key            = encrypted_bias_id,
        ckks           = ckks,
        max_retries    = MICTLANX_MAX_RETRIES,
        delay          = MICTLANX_DELAY,
        backoff_factor = MICTLANX_BACKOFF_FACTOR,
        timeout        = MICTLANX_TIMEOUT
    )
    
    logger.debug({
            "msg": "encrypted bias get from storage"
        })
    
    if isinstance(init_encrypted_weights, list):
        logger.debug({"msg": "init_encrypted_weights are a list"})
        init_encrypted_weights = init_encrypted_weights[0]

    if isinstance(init_encrypted_bias, list):
        logger.debug({"msg": "init_encrypted_bias are a list"})
        init_encrypted_bias = init_encrypted_bias[0]
        
    #encrypted_weights, encrypted_bias = LogisticRegressionFHE.train(
    encrypted_weights, encrypted_bias = PPLR.train(
        HE = ckks.he_object,
        epochs = epochs,
        learning_rate = learning_rate, 
        encrypted_weights = init_encrypted_weights, 
        encrypted_bias = init_encrypted_bias, 
        encrypted_X = encrypted_matrix_train_result, 
        encrypted_y = encrypted_label_vector_train_result, 
        n_features = n_features, 
        scale = scale, 
        n_samples = n_samples)
    logger.debug({
            "msg": "Finish train"
        })
    del init_encrypted_weights
    del init_encrypted_bias

    weights_chunks = RoryCommon.from_pyctxts_to_chunks(
            key        = encrypted_weights_id,
            xs         = encrypted_weights,
            num_chunks = num_chunks
        )

    encrypted_weights_put_chunk = await RoryCommon.delete_and_put_chunks(
            client    = STORAGE_CLIENT,
            bucket_id = BUCKET_ID,
            key       = encrypted_weights_id,
            chunks    = weights_chunks,
            timeout   = MICTLANX_TIMEOUT,
            max_tries = MICTLANX_MAX_RETRIES,
            tags = {
                "shape": str((1, n_features)),
                "dtype": "float32"
            }
        )
    if encrypted_weights_put_chunk.is_err:
            logger.error("Failed to process encrypted weights")
            return Response(status=500, response="Failed to process encrypted weights")

    logger.debug({
            "msg": "Put in storage weights"
        })    
    
    encrypted_bias_put_chunk = await RoryCommon.delete_and_put_chunks(
            client    = STORAGE_CLIENT,
            bucket_id = BUCKET_ID,
            key       = encrypted_bias_id,
            chunks    = encrypted_bias,
            timeout   = MICTLANX_TIMEOUT,
            max_tries = MICTLANX_MAX_RETRIES,
            tags = {
                "shape": str((1, 1)),
                "dtype": "float32"
            }
        )
    if encrypted_bias_put_chunk.is_err:
            logger.error("Failed to process encrypted bias")
            return Response(status=500, response="Failed to process encrypted bias")

    logger.debug({
            "msg": "Put in storage bias"
        })

    return Response(
            response = json.dumps({
                "encrypted_weight_id":encrypted_weights_id,
                "encrypted_bias_id":encrypted_bias_id,       
            }),
            status   = 200,
            headers  = {}
            )
    
@machinelearning.route("/pplr/predict", methods=["POST"])
async def pplr_predict():
    
    return Response(
            response = json.dumps({
                "encrypted_out_predictions_id": "predictions_id_placeholder"          
            }),
            status   = 200,
            headers  = {}
            )