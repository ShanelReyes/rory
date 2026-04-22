import time, json
import numpy as np
import numpy.typing as npt
from typing import List,Tuple
from flask import Blueprint,current_app,request,Response
from rory.core.machine_learning import LogisticRegressionBaseline
from rory.core.machine_learning.secure.pqc import LogisticRegressionFHE
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

logisticregression = Blueprint("logisticregression", __name__, url_prefix="/logisticregression")

@logisticregression.route("/test", methods=["GET", "POST"])
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

@logisticregression.route("/lr", methods=["POST"])
async def lr():
    """
    Standard Logistic Regression (Plaintext) execution endpoint.

    Retrieves plaintext datasets from the Cloud Storage System (CSS) and executes the 
    baseline manual logistic regression using a Maclaurin polynomial approximation. 
    It logs the execution time to serve as a benchmark for evaluating the computational 
    overhead introduced by Post-Quantum secure variants.

    Attributes:
        Experiment-Id (str): Tracking ID for performance auditing.
        Epochs (int): Number of training iterations. Defaults to 1.
        Learning-Rate (float): Step size for gradient descent. Defaults to 0.01.
        Matrix-Train-Id (str): Plaintext training dataset.
        Matrix-Test-Id (str): Plaintext testing dataset.

    Returns:
        label_vector (list): The predicted class assignments for the test records.
        training_time (float): Execution time for the manual model training phase.
        inference_time (float): Execution time for the manual model inference phase.
        service_time (float): Total time elapsed during the worker's processing flow.
    """
    local_start_time            = time.time()
    logger                      = current_app.config["logger"]
    STORAGE_CLIENT: AsyncClient = current_app.config["ASYNC_STORAGE_CLIENT"]
    BUCKET_ID: str              = current_app.config.get("BUCKET_ID", "rory")
    headers                     = request.headers
    to_remove_headers           = ["User-Agent","Accept-Encoding","Connection"]
    filtered_headers            = dict(list(filter(lambda x: not x[0] in to_remove_headers, headers.items())))
    experiment_id               = filtered_headers.get("Experiment-Id","")
    algorithm                   = Constants.MachinelearningAlgorithms.LR
    epochs                      = int(headers.get("Epochs", 1))
    learning_rate               = float(headers.get("Learning-Rate", 0.01))
    matrix_train_id             = filtered_headers.get("Matrix-Train-Id")
    matrix_test_id              = filtered_headers.get("Matrix-Test-Id")

    MICTLANX_TIMEOUT            = int(current_app.config.get("MICTLANX_TIMEOUT", 3600))
    MICTLANX_DELAY              = int(current_app.config.get("MICTLANX_DELAY","2"))
    MICTLANX_BACKOFF_FACTOR     = float(current_app.config.get("MICTLANX_BACKOFF_FACTOR","0.5"))
    MICTLANX_MAX_RETRIES        = int(current_app.config.get("MICTLANX_MAX_RETRIES", 10))

    try:
        matrix_train = await RoryCommon.get_and_merge(
            client         = STORAGE_CLIENT, 
            bucket_id      = BUCKET_ID, 
            key            = matrix_train_id,
            delay          = MICTLANX_DELAY,
            max_retries    = MICTLANX_MAX_RETRIES,
            timeout        = MICTLANX_TIMEOUT,
            backoff_factor = MICTLANX_BACKOFF_FACTOR
        )
        matrix_test = await RoryCommon.get_and_merge(
            client         = STORAGE_CLIENT, 
            bucket_id      = BUCKET_ID, 
            key            = matrix_train_id,
            delay          = MICTLANX_DELAY,
            max_retries    = MICTLANX_MAX_RETRIES,
            timeout        = MICTLANX_TIMEOUT,
            backoff_factor = MICTLANX_BACKOFF_FACTOR
        )

        # It is assumed that the last column contains the labels (y)
        X_train = matrix_train[:, :-1]
        y_train = matrix_train[:, -1]
        X_test = matrix_test[:, :-1]
        y_test = matrix_test[:, -1]

        del matrix_train
        del matrix_test

        result_dict = LogisticRegressionBaseline.run_manual_baseline(
            X_train         = X_train, 
            X_test          = X_test, 
            y_train         = y_train, 
            y_test          = y_test,
            epochs          = epochs, 
            learning_rate   = learning_rate
        )

        service_time = time.time() - local_start_time
        predictions = result_dict["predictions"].tolist()

        return Response(
            response=json.dumps({
                "label_vector": predictions,
                "training_time": result_dict["training_time"],
                "inference_time": result_dict["inference_time"],
                "service_time": service_time
            }),
            status=200
        )

    except Exception as e:
        logger.error({
            "msg":str(e)
        })
        return Response(
            response = None,
            status   = 503,
            headers  = {"Error-Message":str(e)}
        )


@logisticregression.route("/pplr", methods=["POST"])
async def pplr():
    """
    Interactive Privacy-Preserving Logistic Regression (PPLR) execution endpoint.
    This method implements a stateless, homomorphic training and inference round utilizing 
    the CKKS scheme for Post-Quantum security. The Worker receives encrypted parameters, 
    executes the requested number of epochs using encrypted gradients, and outputs 
    the updated weights and predictions to the Cloud Storage System (CSS). It delegates 
    convergence evaluation and noise budget management (decryption/re-encryption) entirely 
    to the Client.

    Attributes:
        Experiment-Id (str): Tracking ID.
        Epochs (int): Number of training iterations for this specific round.
        Learning-Rate (float): Step size for the homomorphic gradient descent.
        Iterations (int): Current round index.
        Scale (int): The scaling factor utilized by Pyfhel for CKKS operations.
        Encrypted-Matrix-Train-Id (str): Storage key for the CKKS-encrypted training dataset.
        Encrypted-Matrix-Test-Id (str): Storage key for the CKKS-encrypted testing dataset.
        Encrypted-Weights-Id (str): Storage key for the current encrypted model weights.
        Encrypted-Bias-Id (str): Storage key for the current encrypted model bias.
        N-Features (int): The number of features in the dataset.
        N-Samples (int): The number of training samples for gradient averaging.

    Returns:
        encrypted_weights_id (str): Storage key for the dynamically generated updated weights.
        encrypted_bias_id (str): Storage key for the dynamically generated updated bias.
        encrypted_predict_vector_id (str): Storage key for the CKKS-encrypted predictions list.
        training_time (float): Execution time for the FHE gradient descent loop.
        inference_time (float): Execution time for the FHE prediction loop.
        service_time (float): Total execution time for this specific worker round.
    """
    local_start_time            = time.time()
    logger                      = current_app.config["logger"]
    worker_id                   = current_app.config["NODE_ID"]
    STORAGE_CLIENT: AsyncClient = current_app.config["ASYNC_STORAGE_CLIENT"]
    BUCKET_ID: str              = current_app.config.get("BUCKET_ID", "rory")
    
    headers                     = request.headers
    algorithm                   = Constants.MachinelearningAlgorithms.PPLR
    experiment_id               = headers.get("Experiment-Id", "")
    epochs                      = int(headers.get("Epochs", 1))
    learning_rate               = float(headers.get("Learning-Rate", 0.01))
    iterations                  = int(headers.get("Iterations", 0))
    scale                       = int(headers.get("Scale", 40)) # Escala para Pyfhel
    
   
    encrypted_matrix_train_id   = headers.get("Encrypted-Matrix-Train-Id")
    encrypted_matrix_test_id    = headers.get("Encrypted-Matrix-Test-Id")
    encrypted_weights_id        = headers.get("Encrypted-Weights-Id")
    encrypted_bias_id           = headers.get("Encrypted-Bias-Id")
    n_features                  = int(headers.get("N-Features", 0))
    n_samples                   = int(headers.get("N-Samples", 0))


    if not all([encrypted_matrix_train_id, encrypted_weights_id, encrypted_bias_id, n_features, n_samples]):
        return Response("Missing mandatory IDs or shape parameters", status=400)

    out_weights_id = f"w_{experiment_id}_{iterations}"
    out_bias_id = f"b_{experiment_id}_{iterations}"
    out_preds_id = f"preds_{experiment_id}_{iterations}"

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
    rotatekey_filename       = current_app.config.get("ROTATEKEY_FILENAME","rotatekey")

    # Configuración CKKS
    ckks = Ckks.from_pyfhel(
        _round              = _round,
        decimals            = decimals,
        path                = path,
        ctx_filename        = ctx_filename,
        pubkey_filename     = pubkey_filename,
        secretkey_filename  = secretkey_filename,
        relinkey_filename   = relinkey_filename,
        rotatekey_filename  = rotatekey_filename
    )

    try:
        enc_X_train, enc_y_train = await RoryCommon.get_pyctxt_list_split(
            client      = STORAGE_CLIENT, 
            bucket_id   = BUCKET_ID, 
            key         = encrypted_matrix_train_id, 
            ckks        = ckks, 
            timeout     = MICTLANX_TIMEOUT
        )
        enc_X_test = await RoryCommon.get_pyctxt_list(
            client      = STORAGE_CLIENT, 
            bucket_id   = BUCKET_ID, 
            key         = encrypted_matrix_test_id, 
            ckks        = ckks, 
            timeout     = MICTLANX_TIMEOUT
        )
        
        enc_weights = await RoryCommon.get_pyctxt(
            client      = STORAGE_CLIENT, 
            bucket_id   = BUCKET_ID, 
            key         = encrypted_weights_id, 
            ckks        = ckks
        )
        enc_bias = await RoryCommon.get_pyctxt(
            client      = STORAGE_CLIENT, 
            bucket_id   = BUCKET_ID, 
            key         = encrypted_bias_id, 
            ckks        = ckks
        )

        updated_weights, updated_bias, train_time = LogisticRegressionFHE.train(
            HE                  = ckks.he_object, 
            epochs              = epochs, 
            learning_rate       = learning_rate, 
            encrypted_weights   = enc_weights, 
            encrypted_bias      = enc_bias, 
            encrypted_X         = enc_X_train, 
            encrypted_y         = enc_y_train, 
            n_features          = n_features, 
            scale               = scale, 
            n_samples           = n_samples
        )


        enc_predictions, inference_time = LogisticRegressionFHE.predict(
            HE                  = ckks.he_object,
            encrypted_X_test    = enc_X_test,
            encrypted_weights   = updated_weights,
            encrypted_bias      = updated_bias,
            scale               = scale,
            n_features          = n_features
        )

        w_chunks = RoryCommon.from_pyctxts_to_chunks(
            xs          = updated_weights, 
            key         = out_weights_id, 
            num_chunks  = 1
        )
        await RoryCommon.delete_and_put_chunks(
            client      = STORAGE_CLIENT, 
            bucket_id   = BUCKET_ID, 
            key         = out_weights_id, 
            chunks      = w_chunks.unwrap(),
            max_tries = MICTLANX_MAX_RETRIES,
            timeout   = MICTLANX_TIMEOUT
        )

        b_chunks = RoryCommon.from_pyctxts_to_chunks(
            xs          = updated_bias, 
            key         = out_bias_id, 
            num_chunks  = 1
        )
        put_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
            client      = STORAGE_CLIENT, 
            bucket_id   = BUCKET_ID, 
            key         = out_bias_id, 
            chunks      = b_chunks.unwrap(),
            max_tries   = MICTLANX_MAX_RETRIES,
            timeout     = MICTLANX_TIMEOUT
        )


        p_chunks = RoryCommon.from_pyctxt_list_to_chunks(
            xs          = enc_predictions, 
            key         = out_preds_id, 
            num_chunks  = 4
        )
        await RoryCommon.delete_and_put_chunks(
            client      = STORAGE_CLIENT, 
            bucket_id   = BUCKET_ID, 
            key         = out_preds_id, 
            chunks      = p_chunks.unwrap()
        )

        service_time = time.time() - local_start_time

        classification_completed_entry = ExperimentLogEntry(
            event          = "COMPLETED",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = local_start_time,
            end_time       = time.time(),
            id             = model_id,
            num_chunks     = num_chunks,
            time           = service_time
        )
        logger.info(classification_completed_entry.model_dump())

        return Response(
            response=json.dumps({
                "encrypted_weights_id": out_weights_id,
                "encrypted_bias_id": out_bias_id,
                "encrypted_predict_vector_id": out_preds_id,
                "training_time": train_time,
                "inference_time": inference_time,
                "service_time": service_time
            }),
            status=200
        )

    except Exception as e:
        logger.error({
            "msg":str(e)
        })
        return Response(
            response = None,
            status   = 503,
            headers  = {"Error-Message":str(e)}
        )