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
    # plaintext_matrix_test_id    = headers.get("Plaintext-Matrix-Test-Id","test_x")
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
        # "plaintext_matrix_test_id": plaintext_matrix_test_id,
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
    headers                     = request.headers
    algorithm                   = Constants.MachineLearningAlgorithms.PPLR_TRAIN
    experiment_id               = headers.get("Experiment-Id", "")
    epochs                      = int(headers.get("Epochs", 1))
    learning_rate               = float(headers.get("Learning-Rate", "0.01"))
    accuracy_threshold          = float(headers.get("Accuracy-Threshold", "0.80"))
    iterations                  = int(headers.get("Iterations", 1))
    encrypted_matrix_train_id   = headers.get("Encrypted-Matrix-Train-Id")
    # encrypted_matrix_test_id    = headers.get("Encrypted-Matrix-Test-Id")
    encrypted_matrix_train_label_id   = headers.get("Encrypted-Matrix-Train-Label-Id")
    encrypted_weights_id        = headers.get("Encrypted-Weights-Id")
    encrypted_bias_id           = headers.get("Encrypted-Bias-Id")
    scale                       = int(headers.get("Scale", 40)) # Escala para Pyfhel
    n_features                  = int(headers.get("N-Features", 0))
    n_samples_train             = int(headers.get("N-Samples-Train", 0))

    if not all([encrypted_matrix_train_id,encrypted_weights_id,encrypted_bias_id,encrypted_matrix_train_label_id]):
        return Response("Missing mandatory IDs or shape parameters", status=400)
    encrypted_out_weights_id     = f"weights_{experiment_id}_{iterations}"
    encrypted_out_bias_id        = f"bias_{experiment_id}_{iterations}"
    encrypted_out_predictions_id = f"predictions_{experiment_id}_{iterations}"
    MICTLANX_TIMEOUT             = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
    MICTLANX_DELAY               = int(current_app.config.get("MICTLANX_DELAY","2"))
    MICTLANX_BACKOFF_FACTOR      = float(current_app.config.get("MICTLANX_BACKOFF_FACTOR","0.5"))
    MICTLANX_MAX_RETRIES         = int(current_app.config.get("MICTLANX_MAX_RETRIES","10"))
    _round                       = bool(int(current_app.config.get("_round","0"))) #False
    decimals                     = int(current_app.config.get("DECIMALS","4"))
    path                         = current_app.config.get("KEYS_PATH","/rory/keys")
    ctx_filename                 = current_app.config.get("CTX_FILENAME","ctx")
    pubkey_filename              = current_app.config.get("PUBKEY_FILENAME","pubkey")
    secretkey_filename           = current_app.config.get("SECRET_KEY_FILENAME","secretkey")
    relinkey_filename            = current_app.config.get("RELINKEY_FILENAME","relinkey")
    rotatekey_filename           = current_app.config.get("ROTATEKEY_FILENAME","rotatekey")
    
    logger.debug({
            "algorithm" : algorithm,
            "encrypted_matrix_train_id": encrypted_matrix_train_id,
            # "encrypted_matrix_test_id": encrypted_matrix_test_id,
            "encrypted_matrix_train_label_id": encrypted_matrix_train_label_id,
            "encrypted_weight_matrix_id": encrypted_weights_id,
            "encrypted_bias_vector_id": encrypted_bias_id,
            "epoch": epochs, 
            "learning_rate": learning_rate, 
            "accuracy_threshold": accuracy_threshold,
            "n_features": n_features,
            "n_features": n_features, 
        })
    
    


    # Agregar al return response los headers que se enviaran al cliente
    # vector de pesos, bias, label_vector_train
    # service_time = time.time() - local_start_time
    return Response(
            response = json.dumps({
                "encrypted_out_weights_id": encrypted_out_weights_id,
                "encrypted_out_bias_id": encrypted_out_bias_id,
                "encrypted_out_predictions_id": encrypted_out_predictions_id          
            }),
            status   = 200,
            headers  = {}
            )
    
@machinelearning.route("/pplr/predict", methods=["POST"])
async def pplr_predict():
    # Similar a pplr_train pero con la logica de prediccion, se pueden compartir algunos parametros como el scale, n_features, etc.
    return Response(
            response = json.dumps({
                "encrypted_out_predictions_id": "predictions_id_placeholder"          
            }),
            status   = 200,
            headers  = {}
            )