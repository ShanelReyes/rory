import os
import time, json
import numpy as np
from uuid import uuid4
from requests import Session
from flask import Blueprint,current_app,request,Response
from rory.core.interfaces.rorymanager import RoryManager
from rory.core.interfaces.roryworker import RoryWorker
from rory.core.utils.constants import Constants
from rorycommon import Common as RoryCommon
from rorycommon import StorageBuilder, StorageParams, Scheme, CkksParams
from rory.core.utils.utils import Utils
from mictlanx import AsyncClient
from mictlanx.utils.segmentation import Chunks
from concurrent.futures import ProcessPoolExecutor
from option import Some
from utils.utils import Utils
from rory.core.security.cryptosystem.pqc.ckks import Ckks, CkksModes

machinelearning = Blueprint("machinelearning",__name__,url_prefix = "/machine-learning")

@machinelearning.route("/test",methods=["GET","POST"])
def test():
    """Diagnostic and health check endpoint for the logisticregression component.

    This method provides a simple mechanism to verify that the 
    logisticregression routes are active and reachable. It is primarily used 
    by the Rory platform's orchestration layer to identify the node type 
    and ensure proper network synchronization before initiating machine 
    learning workflows.

    Returns:
        Response: A Flask Response object containing a JSON payload:
            component_type (str): "client".
            
        Headers:
            Component-Type: "client"
            
        Status Code:
            200: If the logisticregression service is operational.
    """
    return Response(
        response = json.dumps({
            "component_type":"client"
        }),
        status   = 200,
        headers  = {
            "Component-Type":"client"
        }
    )

@machinelearning.route("/logistic-regression/train",methods = ["POST"])
async def logistic_regression_train():
    try:
        arrivalTime                  = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        TESTING                      = current_app.config.get("TESTING",True)
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        STORAGE_CLIENT:AsyncClient   = current_app.config.get("ASYNC_STORAGE_CLIENT")
        max_workers                  = current_app.config.get("MAX_WORKERS",2)
        num_chunks                   = current_app.config.get("NUM_CHUNKS",4)
        np_random                    = current_app.config.get("np_random")
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        if executor == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm                       = Constants.MachineLearningAlgorithms.LOGISTIC_REGRESSION_TRAIN
        s                               = Session()
        request_headers                 = request.headers #Headers for the request
        experiment_id                   = request_headers.get("Experiment-Id",uuid4().hex[:10])
        experiment_iteration            = request_headers.get("Experiment-Iteration","0")
        plaintext_matrix_train_id       = request_headers.get("Plaintext-Matrix-Train-Id","train_x")
        plaintext_label_vector_train_id = request_headers.get("Plaintext-Label-Vector-Train-Id","train_y")
        plaintext_matrix_train_filename = request_headers.get("Plaintext-Matrix-Train-Filename","train_x")
        plaintext_label_vector_train_filename = request_headers.get("Plaintext-Label-Vector-Train-Filename","train_y")
        extension                       = request_headers.get("Extension","csv")
        plaintext_matrix_train_path     = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_train_filename, extension)    
        plaintext_label_vector_train_path     = "{}/{}.{}".format(SOURCE_PATH, plaintext_label_vector_train_filename, extension) 

        epochs               = int(request_headers.get("Epochs", "1"))
        learning_rate        = float(request_headers.get("Learning-Rate", "0.01"))
        weights_id = "{}weights".format(plaintext_matrix_train_id)
        bias_id    = "{}bias".format(plaintext_matrix_train_id)

        MAX_ITERATIONS          = int(request_headers.get("Max-Iterations",current_app.config.get("MAX_ITERATIONS",10)))
        WORKER_TIMEOUT          = int(current_app.config.get("WORKER_TIMEOUT",300))
        
        logger.debug({
            "algorithm" : algorithm,
            "plaintext_matrix_train_id": plaintext_matrix_train_id,
            "plaintext_matrix_train_path": plaintext_matrix_train_path,  
            "plaintext_matrix_train_filename": plaintext_matrix_train_filename,
            "plaintext_label_vector_train_id": plaintext_label_vector_train_id,
            "plaintext_label_vector_train_path": plaintext_label_vector_train_path,
            "plaintext_label_vector_train_filename": plaintext_label_vector_train_filename,
            "extension" : extension,
            "epoch": epochs, 
            "learning_rate": learning_rate, 
            "max_iterations": MAX_ITERATIONS,
            "Weights_Id": weights_id,
            "Bias_Id": bias_id
        })

        storage_backend = (
            StorageBuilder(storage_client = STORAGE_CLIENT)
            .with_storage_params(StorageParams(num_chunks=2, timeout=300))
            .build()
        )

        plaintext_matrix_train_result = await storage_backend.put_from_file(
            bucket_id = BUCKET_ID,
            ball_id   = plaintext_matrix_train_id,
            path      = plaintext_matrix_train_path,
            extension = extension,
            segment   = True,
            encrypt   = False,
            delete    = True
        )

        if plaintext_matrix_train_result.is_err:
            logger.error("Failed to process training dataset: {}".format(plaintext_matrix_train_result.unwrap_err()))
            return Response(status=500, response="Failed to process training dataset")

        logger.debug({
            "msg": "Read, segment and put in storage dataset train"
        })
        
        plaintext_label_vector_train = await storage_backend.put_from_file(
            bucket_id = BUCKET_ID,
            ball_id   = plaintext_label_vector_train_id,
            path      = plaintext_label_vector_train_path,
            extension = extension,
            segment   = True,
            encrypt   = False,
            delete    = True
        )

        if plaintext_label_vector_train.is_err:
            logger.error("Failed to process label vector: {}".format(plaintext_label_vector_train.unwrap_err()))
            return Response(status=500, response="Failed to process label vector")

        logger.debug({
            "msg": "Read, segment and put in storage label vector train"
        })

        get_worker_start_time       = time.time()
        managerResponse:RoryManager = current_app.config.get("manager")
        get_worker_result           = managerResponse.getWorker(
            headers = {
                "Algorithm"             : algorithm,
                "Start-Request-Time"    : str(arrivalTime),
                "Start-Get-Worker-Time" : str(get_worker_start_time) 
            }
        )
        if get_worker_result.is_err:
            error = get_worker_result.unwrap_err()
            logger.error(str(error))
            return Response(str(error), status=500)
        (worker_id,port) = get_worker_result.unwrap()

        worker = RoryWorker(
            workerId  = worker_id,
            port      = port,
            session   = s,
            algorithm = algorithm,
        )

        logger.debug({
            "msg": "Complete comunication",
            "worker id": worker_id
        })

        status = Constants.ClusteringStatus.START

        worker_headers = {
            "Clustering-Status"              : str(status),
            "Experiment-Id"                  : experiment_id,
            "Plaintext-Matrix-Train-Id"      : plaintext_matrix_train_id,
            "Plaintext-Label-Vector-Train-Id": plaintext_label_vector_train_id,
            "Epochs"                         : str(epochs),
            "Learning-Rate"                  : str(learning_rate),
            "Weights-Id"                     : weights_id,
            "Bias-Id"                        : bias_id
        }

        logger.debug({
            "msg": "Connection with the worker"
        })

        worker_response = worker.run(
                timeout = WORKER_TIMEOUT, 
                headers = worker_headers
            ) 
        worker_status = worker_response.status_code

        if worker_status !=200:
            return Response("Worker error: {}".format(worker_response.content),status=500)
        
        worker_response.raise_for_status()
        jsonWorkerResponse = worker_response.json()

        logger.debug({
            "worker_status": str(worker_response),
            "worker_id": worker_id,
            "worker_port" : port,
        })
        
        worker_response.raise_for_status()
        jsonWorkerResponse         = worker_response.json()
        
        return Response(
            response = json.dumps({
                "x": "This endpoint is under development. Please check back later."                
            }),
            status   = 200,
            headers  = {}
            )
    except Exception as e:
        logger.error("CLIENT_ERROR "+str(e))
        return Response(
            response = None, 
            status = 500, 
            headers={"Error-Message":str(e)})


@machinelearning.route("/logistic-regression/predict",methods = ["POST"])
async def logistic_regression_predict():
    try:
        arrivalTime                  = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        TESTING                      = current_app.config.get("TESTING",True)
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        STORAGE_CLIENT:AsyncClient   = current_app.config.get("ASYNC_STORAGE_CLIENT")
        max_workers                  = current_app.config.get("MAX_WORKERS",2)
        num_chunks                   = current_app.config.get("NUM_CHUNKS",4)
        np_random                    = current_app.config.get("np_random")
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        
        if executor == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm                       = Constants.MachineLearningAlgorithms.LOGISTIC_REGRESSION_PREDICT
        s                               = Session()
        request_headers                 = request.headers #Headers for the request
        experiment_id                   = request_headers.get("Experiment-Id",uuid4().hex[:10])
        experiment_iteration            = request_headers.get("Experiment-Iteration","0")
        plaintext_matrix_train_id       = request_headers.get("Plaintext-Matrix-Train-Id","train_x")
        plaintext_matrix_test_id        = request_headers.get("Plaintext-Matrix-Test-Id","test_x")
        plaintext_matrix_test_filename  = request_headers.get("Plaintext-Matrix-Test-Filename","test_x")
        extension                       = request_headers.get("Extension","csv")
        plaintext_matrix_test_path      = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_test_filename, extension)
        weights_id = "{}weights".format(plaintext_matrix_train_id)
        bias_id    = "{}bias".format(plaintext_matrix_train_id)

        MAX_ITERATIONS          = int(request_headers.get("Max-Iterations",current_app.config.get("MAX_ITERATIONS",10)))
        WORKER_TIMEOUT          = int(current_app.config.get("WORKER_TIMEOUT",300))
        MICTLANX_TIMEOUT        = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
        MICTLANX_DELAY          = int(current_app.config.get("MICTLANX_DELAY","2"))
        MICTLANX_BACKOFF_FACTOR = float(current_app.config.get("MICTLANX_BACKOFF_FACTOR","0.5"))
        MICTLANX_MAX_RETRIES    = int(current_app.config.get("MICTLANX_MAX_RETRIES","10"))

        storage_backend = (
            StorageBuilder(storage_client = STORAGE_CLIENT)
            .with_storage_params(StorageParams(num_chunks=2, timeout=300))
            .build()
        )

        plaintext_matrix_test_result = await storage_backend.put_from_file(
            bucket_id = BUCKET_ID,
            ball_id   = plaintext_matrix_test_id,
            path      = plaintext_matrix_test_path,
            extension = extension,
            segment   = True,
            encrypt   = False,
            delete    = True
        )

        if plaintext_matrix_test_result.is_err:
            logger.error("Failed to process test dataset: {}".format(plaintext_matrix_test_result.unwrap_err()))
            return Response(status=500, response="Failed to process test dataset")

        logger.debug({
            "msg": "Read, segment and put in storage dataset test"
        })

        get_worker_start_time       = time.time()
        managerResponse:RoryManager = current_app.config.get("manager")
        get_worker_result           = managerResponse.getWorker(
            headers = {
                "Algorithm"             : algorithm,
                "Start-Request-Time"    : str(arrivalTime),
                "Start-Get-Worker-Time" : str(get_worker_start_time) 
            }
        )
        if get_worker_result.is_err:
            error = get_worker_result.unwrap_err()
            logger.error(str(error))
            return Response(str(error), status=500)
        (worker_id,port) = get_worker_result.unwrap()

        worker = RoryWorker(
            workerId  = worker_id,
            port      = port,
            session   = s,
            algorithm = algorithm,
        )

        logger.debug({
            "msg": "Complete comunication",
            "worker id": worker_id
        })

        status = Constants.ClusteringStatus.START

        worker_headers = {
            "Clustering-Status"              : str(status),
            "Experiment-Id"                  : experiment_id,
            "Plaintext-Matrix-Test-Id"      : plaintext_matrix_test_id,
            "Weights-Id"                     : weights_id,
            "Bias-Id"                        : bias_id
        }

        logger.debug({
            "msg": "Connection with the worker"
        })

        worker_response = worker.run(
                timeout = WORKER_TIMEOUT, 
                headers = worker_headers
            ) 
        worker_status = worker_response.status_code

        if worker_status !=200:
            return Response("Worker error: {}".format(worker_response.content),status=500)
        
        worker_response.raise_for_status()
        jsonWorkerResponse = worker_response.json()

        # Extraer label vector de la respuesta del worker
      

        # Colocar label vector en el response
        return Response(
            response = json.dumps({
                "x": "This endpoint is under development. Please check back later."                
            }),
            status   = 200,
            headers  = {}
            )
    except Exception as e:
        logger.error("CLIENT_ERROR "+str(e))
        return Response(
            response = None, 
            status = 500, 
            headers={"Error-Message":str(e)})

@machinelearning.route("/pplr/train",methods = ["POST"])
async def pplr_train():
    try:
        arrivalTime                  = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        TESTING                      = current_app.config.get("TESTING",True)
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        STORAGE_CLIENT:AsyncClient   = current_app.config.get("ASYNC_STORAGE_CLIENT")
        max_workers                  = current_app.config.get("MAX_WORKERS",2)
        num_chunks                   = current_app.config.get("NUM_CHUNKS",2)
        np_random                    = current_app.config.get("np_random")
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        security_level               = current_app.config.get("LIU_SECURITY_LEVEL",128)
        if executor == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm                             = Constants.MachineLearningAlgorithms.PPLR_TRAIN
        MODE                                  = CkksModes.ML
        s                                     = Session()
        request_headers                       = request.headers                                                                   #Headers for the request
        experiment_id                         = request_headers.get("Experiment-Id",uuid4().hex[:10])
        plaintext_matrix_train_id             = request_headers.get("Plaintext-Matrix-Train-Id","train_x")
        encrypted_matrix_train_id             = "encrypted{}".format(plaintext_matrix_train_id)                                   # The id of the encrypted matrix is built
        plaintext_label_vector_train_id       = request_headers.get("Plaintext-Label-Vector-Train-Id","train_y")
        encrypted_label_vector_train_id       = "encrypted{}".format(plaintext_label_vector_train_id)
        plaintext_matrix_train_filename       = request_headers.get("Plaintext-Matrix-Train-Filename","train_x")
        plaintext_label_vector_train_filename = request_headers.get("Plaintext-Label-Vector-Train-Filename","train_y")
        extension                             = request_headers.get("Extension","csv")
        plaintext_matrix_train_path           = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_train_filename, extension)
        plaintext_label_vector_train_path     = "{}/{}.{}".format(SOURCE_PATH, plaintext_label_vector_train_filename, extension)
        epochs                                = int(request_headers.get("Epochs", "1"))
        learning_rate                         = float(request_headers.get("Learning-Rate", "0.01"))
        encrypted_weights_id                  = "{}encryptedweights".format(plaintext_matrix_train_id)
        encrypted_bias_id                     = "{}encryptedbias".format(plaintext_matrix_train_id)
        WORKER_TIMEOUT     = int(current_app.config.get("WORKER_TIMEOUT",300))
        _round             = bool(int(current_app.config.get("_round","0"))) 
        decimals           = int(current_app.config.get("DECIMALS","4"))
        keys_path          = current_app.config.get("KEYS_PATH","/rory/keys/keys128")
        ctx_filename       = current_app.config.get("CTX_FILENAME","ctx")
        pubkey_filename    = current_app.config.get("PUBKEY_FILENAME","pubkey")
        secretkey_filename = current_app.config.get("SECRET_KEY_FILENAME","secretkey")
        relinkey_filename  = current_app.config.get("RELINKEY_FILENAME","relinkey")
        rotatekey_filename = current_app.config.get("ROTATEKEY_FILENAME","rotatekey")

        ckks           = Ckks.from_pyfhel_client(
            _round             = _round,
            decimals           = decimals,
            path               = keys_path,
            ctx_filename       = ctx_filename,
            pubkey_filename    = pubkey_filename,
            secretkey_filename = secretkey_filename,
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

        plaintext_matrix_train_result = await storage_backend.put_from_file(
            bucket_id = BUCKET_ID,
            ball_id   = encrypted_matrix_train_id,
            path      = plaintext_matrix_train_path,
            extension = extension,
            segment   = True,
            encrypt   = True,
            delete    = True
        )
        
        if plaintext_matrix_train_result.is_err:
            logger.error("Failed to process training dataset: {}".format(plaintext_matrix_train_result.unwrap_err()))
            return Response(status=500, response="Failed to process training dataset")
        plaintext_matrix_train_respose = plaintext_matrix_train_result.unwrap()

        logger.debug({
            "msg": "Read, segment, encrypt and put in storage dataset train"
        })
        #_________________
        
        plaintext_label_vector_train = await storage_backend.put_from_file(
            bucket_id = BUCKET_ID,
            ball_id   = encrypted_label_vector_train_id,
            path      = plaintext_label_vector_train_path,
            extension = extension,
            segment   = True,
            encrypt   = True,
            delete    = True
        )

        if plaintext_label_vector_train.is_err:
            logger.error("Failed to process label vector: {}".format(plaintext_label_vector_train.unwrap_err()))
            return Response(status=500, response="Failed to process label vector")
        plaintext_label_vector_train_response = plaintext_label_vector_train.unwrap()

        logger.debug({
            "msg": "Read, segment, encrypt and put in storage label vector train"
        })
        #_________________
        
        scale            = ckks.SECURITY_LEVELS[MODE.value][security_level]["scale"]
        n_samples        = plaintext_matrix_train_respose.shape[0]
        n_features       = plaintext_matrix_train_respose.shape[1]
        plaintext_weight = np.zeros((1,n_features), dtype=np.float32)
        
        logger.debug({
            "scale"        : scale,
            "n_features"   : n_features,
            "n_samples"    : n_samples,
            "weights shape": plaintext_weight.shape,
        })

        encrypted_weight_result = await storage_backend.put(
            bucket_id = BUCKET_ID,
            data      = plaintext_weight,
            ball_id   = encrypted_weights_id,
            segment   = True,
            encrypt   = True,
            scheme    = Scheme.CKKS,
            delete    = True
        )

        if encrypted_weight_result.is_err:
            logger.error("Failed to put encrypted weights in cloud storage: {}".format(encrypted_weight_result.unwrap_err()))
            return Response(status=500, response="Failed to put encrypted weights in cloud storage")
        encrypted_weight_response = encrypted_weight_result.unwrap()
        
        logger.debug({
            "msg": "Read, segment, encrypt and put in storage encrypted weights"
        })
        
        plaintext_bias = np.array([0.0], dtype=np.float32)

        encrypted_bias_result = await storage_backend.put(
            bucket_id = BUCKET_ID,
            data      = plaintext_bias,
            ball_id   = encrypted_bias_id,
            segment   = True,
            encrypt   = True,
            scheme    = Scheme.CKKS,
            delete    = True
        )
        
        if encrypted_bias_result.is_err:
            logger.error("Failed to put encrypted bias in cloud storage: {}".format(encrypted_bias_result.unwrap_err()))
            return Response(status=500, response="Failed to put encrypted bias in cloud storage")
        encrypted_bias_response = encrypted_bias_result.unwrap()
        #__________________________


        get_worker_start_time       = time.time()
        managerResponse:RoryManager = current_app.config.get("manager") # Communicates with the manager
        get_worker_result           = managerResponse.getWorker( #Gets the worker from the manager
            headers = {
                "Algorithm"            : algorithm,
                "Start-Request-Time"   : str(arrivalTime),
                "Start-Get-Worker-Time": str(get_worker_start_time)
            }
        )
        if get_worker_result.is_err:
            error = get_worker_result.unwrap_err()
            logger.error(str(error))
            return Response(str(error), status=500)
        (worker_id,port) = get_worker_result.unwrap()
        
        worker = RoryWorker( #Allows to establish the connection with the worker
            workerId  = worker_id,
            port      = port,
            session   = s,
            algorithm = algorithm,
        )

        status = Constants.ClusteringStatus.START #Set the status to start
        iteration = 0

        worker_headers = {
            "Clustering-Status"   : str(status),
            "Experiment-Id"       : experiment_id,
            "Epochs"              : str(epochs),
            "Learning-Rate"       : str(learning_rate),
            "Encrypted-Matrix-Train-Id": encrypted_matrix_train_id,
            "Encrypted-Label-Vector-Train-Id": encrypted_label_vector_train_id,
            "Encrypted-Weights-Id": encrypted_weights_id,
            "Encrypted-Bias-Id"   : encrypted_bias_id,
            "Scale"               : str(scale),
            "N-Features"          : str(n_features),
            "N-Samples"           : str(n_samples),
            "Num-Chunks"          : str(num_chunks),
        }

        logger.debug({
            "msg": "Connection with the worker"
        })

        worker_response = worker.run(
                timeout = WORKER_TIMEOUT, 
                headers = worker_headers
            ) #Run 1 starts
        worker_status = worker_response.status_code

        if worker_status !=200:
            return Response("Worker error: {}".format(worker_response.content),status=500)
        
        worker_response.raise_for_status()
        jsonWorkerResponse         = worker_response.json()

        del encrypted_weight_response
        del encrypted_bias_response
        
        encrypted_weights_result = await storage_backend.get(
            bucket_id = BUCKET_ID,
            ball_id   = encrypted_weights_id,
            segment   = True,
            encrypt   = True,
            scheme    = Scheme.CKKS
        )
    
        if encrypted_weights_result.is_err:
            logger.error(f"Failed to get init encrypted weights: {encrypted_weights_result.unwrap_err()}")
            return Response(status=500, response="Failed to get init encrypted weights")
        encrypted_weights = encrypted_weights_result.unwrap().raw_value

        logger.debug({
            "msg": "encrypted weight get from storage",
            "encrypted_weight_id": encrypted_weights_id,
            "type": str(type(encrypted_weights)),
        })

        weights_plain_list = ckks.decrypt_list(encrypted_weights, take=n_features)
        weights_plain = weights_plain_list[0].reshape(1, -1).astype(np.float32)
        
        logger.debug({
            "msg": "Decrypted weights",
            "weights_plain_list": str(weights_plain_list),
            "weights_plain": str(weights_plain),
        })

        encrypted_weight_result = await storage_backend.put(
            bucket_id = BUCKET_ID,
            data      = weights_plain,
            ball_id   = encrypted_weights_id,
            segment   = True,
            encrypt   = True,
            scheme    = Scheme.CKKS,
            delete    = True
        )

        if encrypted_weight_result.is_err:
            logger.error("Failed to put encrypted weights in cloud storage: {}".format(encrypted_weight_result.unwrap_err()))
            return Response(status=500, response="Failed to put encrypted weights in cloud storage")
        encrypted_weight_response = encrypted_weight_result.unwrap()
        
        logger.debug({
            "msg": "Read, segment, encrypt and put in storage encrypted weights",
            "encrypted_weight_id": encrypted_weights_id,
            "type": str(type(encrypted_weight_response)),
        })

        encrypted_bias_result = await storage_backend.get(
            bucket_id = BUCKET_ID,
            ball_id   = encrypted_bias_id,
            segment   = True,
            encrypt   = True,
            scheme    = Scheme.CKKS
        )

        if encrypted_bias_result.is_err:
            logger.error(f"Failed to get init encrypted bias: {encrypted_bias_result.unwrap_err()}")
            return Response(status=500, response="Failed to get init encrypted bias")
        encrypted_bias = encrypted_bias_result.unwrap().raw_value

        logger.debug({
            "msg": "encrypted bias get from storage",
            "encrypted_bias_id": encrypted_bias_id,
            "type": str(type(encrypted_bias)),
            "value": str(encrypted_bias),
        })
        
        bias_plain_list = ckks.decrypt_list(encrypted_bias, take=1)
        bias_plain = bias_plain_list[0].reshape(1, -1).astype(np.float32)

        logger.debug({
            "msg": "Decrypted bias",
            "bias_plain_list": str(bias_plain_list),
            "bias_plain": str(bias_plain),
        })
        
        encrypted_bias_result = await storage_backend.put(
            bucket_id = BUCKET_ID,
            data      = bias_plain,
            ball_id   = encrypted_bias_id,
            segment   = True,
            encrypt   = True,
            scheme    = Scheme.CKKS,
            delete    = True
        )

        if encrypted_bias_result.is_err:
            logger.error("Failed to put encrypted bias in cloud storage: {}".format(encrypted_bias_result.unwrap_err()))
            return Response(status=500, response="Failed to put encrypted bias in cloud storage")
        encrypted_bias_response = encrypted_bias_result.unwrap()
        
        logger.debug({
            "msg": "Read, segment, encrypt and put in storage encrypted bias",
            "encrypted_bias_id": encrypted_bias_id,
            "type": str(type(encrypted_bias_response)),
        })

        return Response(
            response = json.dumps({
                "x": "This endpoint is under development. Please check back later.",
                "algorithm": algorithm,
                "worker_id": worker_id,
            }),
            status   = 200,
            headers  = {}
        )

    except Exception as e:
        logger.error({
            "msg":str(e)
        })
        return Response(
            response = None, 
            status = 500, 
            headers={"Error-Message":str(e)}
        )


@machinelearning.route("/pplr/predict",methods = ["POST"])
async def pplr_predict():
    try:
        arrivalTime                  = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        TESTING                      = current_app.config.get("TESTING",True)
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        STORAGE_CLIENT:AsyncClient   = current_app.config.get("ASYNC_STORAGE_CLIENT")
        max_workers                  = current_app.config.get("MAX_WORKERS",2)
        num_chunks                   = current_app.config.get("NUM_CHUNKS",2)
        np_random                    = current_app.config.get("np_random")
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        security_level               = current_app.config.get("LIU_SECURITY_LEVEL",128)
        
        if executor == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm            = Constants.MachineLearningAlgorithms.PPLR_PREDICT
        MODE                 = CkksModes.ML
        s                    = Session()
        request_headers      = request.headers                                        #Headers for the request
        experiment_id        = request_headers.get("Experiment-Id",uuid4().hex[:10])
        experiment_iteration = request_headers.get("Experiment-Iteration","0")

        plaintext_matrix_test_id       = request_headers.get("Plaintext-Matrix-Test-Id","test_x")
        encrypted_matrix_test_id       = "encrypted{}".format(plaintext_matrix_test_id)
        plaintext_matrix_test_filename = request_headers.get("Plaintext-Matrix-Test-Filename","test_x")
        extension                      = request_headers.get("Extension","csv")
        plaintext_matrix_train_id      = request_headers.get("Plaintext-Matrix-Train-Id","train_x")
        plaintext_matrix_test_path     = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_test_filename, extension)
        encrypted_weights_id           = "{}encryptedweights".format(plaintext_matrix_train_id)
        encrypted_bias_id              = "{}encryptedbias".format(plaintext_matrix_train_id)

        _round             = bool(int(current_app.config.get("_round","0")))            #False
        decimals           = int(current_app.config.get("DECIMALS","4"))
        keys_path          = current_app.config.get("KEYS_PATH","/rory/keys/keys128")
        ctx_filename       = current_app.config.get("CTX_FILENAME","ctx")
        pubkey_filename    = current_app.config.get("PUBKEY_FILENAME","pubkey")
        secretkey_filename = current_app.config.get("SECRET_KEY_FILENAME","secretkey")
        relinkey_filename  = current_app.config.get("RELINKEY_FILENAME","relinkey")
        rotatekey_filename = current_app.config.get("ROTATEKEY_FILENAME","rotatekey")
        
        WORKER_TIMEOUT          = int(current_app.config.get("WORKER_TIMEOUT",300))
        max_workers             = Utils.get_workers(num_chunks=num_chunks)

        ckks                   = Ckks.from_pyfhel_client(
            _round             = _round,
            decimals           = decimals,
            path               = keys_path,
            ctx_filename       = ctx_filename,
            pubkey_filename    = pubkey_filename,
            secretkey_filename = secretkey_filename,
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

        plaintext_matrix_test_result = await storage_backend.put_from_file(
            bucket_id = BUCKET_ID,
            ball_id   = encrypted_matrix_test_id,
            path      = plaintext_matrix_test_path,
            extension = extension,
            segment   = True,
            encrypt   = True,
            delete    = True
        )
        
        if plaintext_matrix_test_result.is_err:
            logger.error("Failed to process test dataset: {}".format(plaintext_matrix_test_result.unwrap_err()))
            return Response(status=500, response="Failed to process test dataset")
        plaintext_matrix_test_response = plaintext_matrix_test_result.unwrap()

        logger.debug({
            "msg": "Read, segment, encrypt and put in storage dataset test",
            "encrypted_matrix_test_id": encrypted_matrix_test_id
        })

        scale            = ckks.SECURITY_LEVELS[MODE.value][security_level]["scale"]
        n_features       = plaintext_matrix_test_response.shape[1]

        get_worker_start_time       = time.time()
        managerResponse:RoryManager = current_app.config.get("manager") # Communicates with the manager
        get_worker_result           = managerResponse.getWorker( #Gets the worker from the manager
            headers = {
                "Algorithm"            : algorithm,
                "Start-Request-Time"   : str(arrivalTime),
                "Start-Get-Worker-Time": str(get_worker_start_time)
            }
        )
        if get_worker_result.is_err:
            error = get_worker_result.unwrap_err()
            logger.error(str(error))
            return Response(str(error), status=500)
        (worker_id,port) = get_worker_result.unwrap()
        
        worker = RoryWorker( #Allows to establish the connection with the worker
            workerId  = worker_id,
            port      = port,
            session   = s,
            algorithm = algorithm,
        )

        iteration = 0

        worker_headers = {
            "Experiment-Id"       : experiment_id,
            "Encrypted-Matrix-Test-Id": encrypted_matrix_test_id,
            "Encrypted-Weights-Id": encrypted_weights_id,
            "Encrypted-Bias-Id"   : encrypted_bias_id,
            "Scale"               : str(scale),
            "N-Features"          : str(n_features),
            "Num-Chunks"          : str(num_chunks),
        }

        logger.debug({
            "msg": "Connection with the worker"
        })

        worker_response = worker.run(
                timeout = WORKER_TIMEOUT, 
                headers = worker_headers
            )
        worker_status = worker_response.status_code

        if worker_status !=200:
            return Response("Worker error: {}".format(worker_response.content),status=500)
        
        worker_response.raise_for_status()
        jsonWorkerResponse    = worker_response.json()
        encrypted_predictions_id = jsonWorkerResponse["encrypted_predictions_id"]

        # logger.debug({
        # "msg": "We are the champions",
        # "encrypted_predictions_result":str(encrypted_predictions)
        # })

        encrypted_predictions_result = await storage_backend.get(
            bucket_id = BUCKET_ID,
            ball_id   = encrypted_predictions_id,
            segment   = True,
            encrypt   = True,
            scheme    = Scheme.CKKS
        )

        if encrypted_predictions_result.is_err:
            logger.error(f"Failed to get encrypted predictions: {encrypted_predictions_result.unwrap_err()}")
            return Response(status=500, response="Failed to get encrypted predictions") 
        encrypted_predictions = encrypted_predictions_result.unwrap().raw_value
        
        logger.debug({
            "msg": "encrypted predictions get from storage",
            "encrypted_predictions_id": encrypted_predictions_id,
            "type": str(type(encrypted_predictions)),
        })

        predictions_plain_list = ckks.decrypt_list(encrypted_predictions, take=1)
        predictions_plain = np.array([p[0] for p in predictions_plain_list], dtype=np.float32)

        logger.debug({
            "msg": "Decrypt predictions",
        })

        label_predictions = [1 if v >= 0.5 else 0 for v in predictions_plain]


        return Response(
            response = json.dumps({
                "label_predictions":label_predictions,
                "worker_id":worker_id,
                "algorithm":algorithm,
            }),
            status   = 200,
            headers  = {}
        )

    except Exception as e:
        logger.error({
            "msg":str(e)
        })
        return Response(
            response = None, 
            status = 500, 
            headers={"Error-Message":str(e)}
        )
    

    