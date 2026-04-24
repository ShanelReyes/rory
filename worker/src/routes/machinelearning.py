import time, json
import numpy as np
import numpy.typing as npt
from typing import List,Tuple
from flask import Blueprint,current_app,request,Response
from rory.core.machine_learning.secure.pqc.pplr import LogisticRegressionFHE
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

machinelearning = Blueprint("machinelearning", __name__, url_prefix="/machinelearning")

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

@machinelearning.route("/logisticregression", methods=["POST"])
async def logisticregression():
    local_start_time            = time.time()
    logger                      = current_app.config["logger"]
    STORAGE_CLIENT: AsyncClient = current_app.config["ASYNC_STORAGE_CLIENT"]
    BUCKET_ID: str              = current_app.config.get("BUCKET_ID", "rory")
    headers                     = request.headers
    to_remove_headers           = ["User-Agent","Accept-Encoding","Connection"]
    filtered_headers            = dict(list(filter(lambda x: not x[0] in to_remove_headers, headers.items())))
    experiment_id               = filtered_headers.get("Experiment-Id","")
    algorithm                   = Constants.MachineLearningAlgorithms.LOGISTIC_REGRESSION
    plaintext_matrix_train_id   = headers.get("Plaintext-Matrix-Train-Id","train_x")
    plaintext_matrix_test_id    = headers.get("Plaintext-Matrix-Test-Id","test_x")
    plaintext_matrix_train_label_id = headers.get("Plaintext-Matrix-Train-Label-Id","train_y")
    plaintext_matrix_test_label_id = headers.get("Plaintext-Matrix-Test-Label-Id","test_y")
    epochs                      = int(headers.get("Epochs", 1))
    learning_rate               = float(headers.get("Learning-Rate", "0.01"))
    matrix_train_id             = filtered_headers.get("Matrix-Train-Id")
    matrix_test_id              = filtered_headers.get("Matrix-Test-Id")
    MICTLANX_TIMEOUT            = int(current_app.config.get("MICTLANX_TIMEOUT", 3600))
    MICTLANX_DELAY              = int(current_app.config.get("MICTLANX_DELAY","2"))
    MICTLANX_BACKOFF_FACTOR     = float(current_app.config.get("MICTLANX_BACKOFF_FACTOR","0.5"))
    MICTLANX_MAX_RETRIES        = int(current_app.config.get("MICTLANX_MAX_RETRIES", 10))
    
    logger.debug({
        "algorithm" : algorithm,
        "plaintext_matrix_train_id": plaintext_matrix_train_id,
        "plaintext_matrix_test_id": plaintext_matrix_test_id,
        "plaintext_matrix_train_label_id": plaintext_matrix_train_label_id,
        "plaintext_matrix_test_label_id": plaintext_matrix_test_label_id,
        "epoch": epochs, 
        "learning_rate": learning_rate, 
    })

    return Response(
        response=json.dumps({
            "x": "This endpoint is under development. Please check back later."
        }),
        status=200
        )


@machinelearning.route("/pplr", methods=["POST"])
async def pplr():
    local_start_time            = time.time()
    logger                      = current_app.config["logger"]
    worker_id                   = current_app.config["NODE_ID"]
    STORAGE_CLIENT: AsyncClient = current_app.config["ASYNC_STORAGE_CLIENT"]
    BUCKET_ID: str              = current_app.config.get("BUCKET_ID", "rory")
    headers                     = request.headers
    algorithm                   = Constants.MachineLearningAlgorithms.PPLR
    experiment_id               = headers.get("Experiment-Id", "")
    epochs                      = int(headers.get("Epochs", 1))
    learning_rate               = float(headers.get("Learning-Rate", "0.01"))
    accuracy_threshold          = float(headers.get("Accuracy-Threshold", "0.80"))
    iterations                  = int(headers.get("Iterations", 1))
    encrypted_matrix_train_id   = headers.get("Encrypted-Matrix-Train-Id")
    encrypted_matrix_test_id    = headers.get("Encrypted-Matrix-Test-Id")
    encrypted_matrix_train_label_id   = headers.get("Encrypted-Matrix-Train-Label-Id")
    encrypted_matrix_test_label_id    = headers.get("Encrypted-Matrix-Test-Label-Id")
    encrypted_weights_id        = headers.get("Encrypted-Weights-Id")
    encrypted_bias_id           = headers.get("Encrypted-Bias-Id")
    scale                       = int(headers.get("Scale", 40)) # Escala para Pyfhel
    n_features                  = int(headers.get("N-Features", 0))
    n_samples                   = int(headers.get("N-Samples", 0))

    if not all([encrypted_matrix_train_id, encrypted_weights_id, encrypted_bias_id]):
        return Response("Missing mandatory IDs or shape parameters", status=400)
    # out_weights_id = f"w_{experiment_id}_{iterations}"
    # out_bias_id = f"b_{experiment_id}_{iterations}"
    # out_preds_id = f"preds_{experiment_id}_{iterations}"
    MICTLANX_TIMEOUT        = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
    MICTLANX_DELAY          = int(current_app.config.get("MICTLANX_DELAY","2"))
    MICTLANX_BACKOFF_FACTOR = float(current_app.config.get("MICTLANX_BACKOFF_FACTOR","0.5"))
    MICTLANX_MAX_RETRIES    = int(current_app.config.get("MICTLANX_MAX_RETRIES","10"))
    _round                  = bool(int(current_app.config.get("_round","0"))) #False
    decimals                = int(current_app.config.get("DECIMALS","4"))
    path                    = current_app.config.get("KEYS_PATH","/rory/keys")
    ctx_filename            = current_app.config.get("CTX_FILENAME","ctx")
    pubkey_filename         = current_app.config.get("PUBKEY_FILENAME","pubkey")
    secretkey_filename      = current_app.config.get("SECRET_KEY_FILENAME","secretkey")
    relinkey_filename       = current_app.config.get("RELINKEY_FILENAME","relinkey")
    rotatekey_filename      = current_app.config.get("ROTATEKEY_FILENAME","rotatekey")
    
    logger.debug({
            "algorithm" : algorithm,
            "encrypted_matrix_train_id": encrypted_matrix_train_id,
            "encrypted_matrix_test_id": encrypted_matrix_test_id,
            "encrypted_matrix_train_label_id": encrypted_matrix_train_label_id,
            "encrypted_matrix_test_label_id": encrypted_matrix_test_label_id,
            "epoch": epochs, 
            "learning_rate": learning_rate, 
            "accuracy_threshold": accuracy_threshold,
            "scale" : scale,
            "n_features" : n_features,
            "n_samples" : n_samples,
        })
    
    return Response(
            response = json.dumps({
                "x": "This endpoint is under development. Please check back later."                
            }),
            status   = 200,
            headers  = {}
            )
    