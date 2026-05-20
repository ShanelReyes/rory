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
from mictlanx import AsyncClient
from concurrent.futures import ProcessPoolExecutor
from models import ExperimentLogEntry
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
        local_start_time             = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        TESTING                      = current_app.config.get("TESTING",True)
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        STORAGE_CLIENT:AsyncClient   = current_app.config.get("ASYNC_STORAGE_CLIENT")
        num_chunks                   = current_app.config.get("NUM_CHUNKS",4)
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        if executor == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm                       = Constants.MachineLearningAlgorithms.LOGISTIC_REGRESSION_TRAIN
        s                               = Session()
        request_headers                 = request.headers #Headers for the request
        experiment_id                   = request_headers.get("Experiment-Id",uuid4().hex[:10])
        plaintext_matrix_train_id       = request_headers.get("Plaintext-Matrix-Train-Id","train_x")
        plaintext_label_vector_train_id = request_headers.get("Plaintext-Label-Vector-Train-Id","train_y")
        plaintext_matrix_train_filename = request_headers.get("Plaintext-Matrix-Train-Filename","train_x")
        plaintext_label_vector_train_filename = request_headers.get("Plaintext-Label-Vector-Train-Filename","train_y")
        extension                       = request_headers.get("Extension","csv")
        plaintext_matrix_train_path     = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_train_filename, extension)    
        plaintext_label_vector_train_path = "{}/{}.{}".format(SOURCE_PATH, plaintext_label_vector_train_filename, extension) 

        epochs           = int(request_headers.get("Epochs", "1"))
        learning_rate    = float(request_headers.get("Learning-Rate", "0.01"))
        weights_id       = "{}weights".format(plaintext_matrix_train_id)
        bias_id          = "{}bias".format(plaintext_matrix_train_id)
        WORKER_TIMEOUT   = int(current_app.config.get("WORKER_TIMEOUT",300))
        MICTLANX_TIMEOUT = int(current_app.config.get("MICTLANX_TIMEOUT",3600))

        storage_backend = (
            StorageBuilder(storage_client = STORAGE_CLIENT)
            .with_storage_params(StorageParams(num_chunks=num_chunks, timeout=MICTLANX_TIMEOUT))
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
        plaintext_matrix_train_response = plaintext_matrix_train_result.unwrap()
        logger.debug({
            "event"        : "PUT",
            "experiment_id": experiment_id,
            "bucket_id"    : BUCKET_ID,
            "ball_id"      : plaintext_matrix_train_id,
            "matrix_id"    : plaintext_matrix_train_id,
            "shape"        : str(plaintext_matrix_train_response.shape),
            "dtype"        : str(plaintext_matrix_train_response.dtype),
            "read_time"    : plaintext_matrix_train_response.read_time,
            "segment_time" : plaintext_matrix_train_response.segment_time,
            "upload_time"  : plaintext_matrix_train_response.upload_time,
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

        plaintext_label_vector_train_response = plaintext_label_vector_train.unwrap()
        logger.debug({
            "event"        : "PUT",
            "experiment_id": experiment_id,
            "bucket_id"    : BUCKET_ID,
            "ball_id"      : plaintext_label_vector_train_id,
            "matrix_id"    : plaintext_label_vector_train_id,
            "shape"        : str(plaintext_label_vector_train_response.shape),
            "dtype"        : str(plaintext_label_vector_train_response.dtype),
            "read_time"    : plaintext_label_vector_train_response.read_time,
            "segment_time" : plaintext_label_vector_train_response.segment_time,
            "upload_time"  : plaintext_label_vector_train_response.upload_time,
        })

        service_time_client         = time.time() - local_start_time
        get_worker_start_time       = time.time()
        managerResponse:RoryManager = current_app.config.get("manager")
        get_worker_result           = managerResponse.getWorker(
            headers = {
                "Algorithm"             : algorithm,
                "Start-Request-Time"    : str(local_start_time),
                "Start-Get-Worker-Time" : str(get_worker_start_time) 
            }
        )
        if get_worker_result.is_err:
            error = get_worker_result.unwrap_err()
            logger.error(str(error))
            return Response(str(error), status=500)
        (_worker_id,port) = get_worker_result.unwrap()

        get_worker_end_time     = time.time()
        get_worker_service_time = get_worker_end_time - get_worker_start_time
        worker_id               = "localhost" if TESTING else _worker_id
        worker_start_time       = time.time()

        worker = RoryWorker(
            workerId  = worker_id,
            port      = port,
            session   = s,
            algorithm = algorithm,
        )

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

        worker_response = worker.run(
                timeout = WORKER_TIMEOUT, 
                headers = worker_headers
            ) 
        worker_status = worker_response.status_code

        if worker_status !=200:
            return Response("Worker error: {}".format(worker_response.content),status=500)
        
        worker_response.raise_for_status()
        jsonWorkerResponse  = worker_response.json()
        worker_service_time = jsonWorkerResponse["service_time"]
        worker_end_time     = time.time()

        worker_response_time = worker_end_time - worker_start_time
        response_time        = time.time() - local_start_time

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
            client_time   = service_time_client,
            manager_time  = get_worker_service_time,
            worker_time   = worker_response_time,
        ).model_dump())
        
        return Response(
            response = json.dumps({
                "worker_id":worker_id,
                "service_time_manager":get_worker_service_time,
                "service_time_worker":worker_response_time,
                "service_time_client":service_time_client,
                "service_time_train":response_time,
                "algorithm":algorithm,                
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
        WORKER_TIMEOUT               = int(current_app.config.get("WORKER_TIMEOUT",300))
        MICTLANX_TIMEOUT             = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        num_chunks                   = current_app.config.get("NUM_CHUNKS",4)
        if executor == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm                      = Constants.MachineLearningAlgorithms.LOGISTIC_REGRESSION_PREDICT
        s                              = Session()
        request_headers                = request.headers                                                            #Headers for the request
        experiment_id                  = request_headers.get("Experiment-Id",uuid4().hex[:10])
        plaintext_matrix_train_id      = request_headers.get("Plaintext-Matrix-Train-Id","train_x")
        plaintext_matrix_test_id       = request_headers.get("Plaintext-Matrix-Test-Id","test_x")
        plaintext_matrix_test_filename = request_headers.get("Plaintext-Matrix-Test-Filename","test_x")
        extension                      = request_headers.get("Extension","csv")
        plaintext_matrix_test_path     = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_test_filename, extension)
        weights_id                     = "{}weights".format(plaintext_matrix_train_id)
        bias_id                        = "{}bias".format(plaintext_matrix_train_id)
        
        storage_backend                = (
            StorageBuilder(storage_client = STORAGE_CLIENT)
            .with_storage_params(StorageParams(num_chunks=num_chunks, timeout=MICTLANX_TIMEOUT))
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

        plaintext_matrix_test_response = plaintext_matrix_test_result.unwrap()
        logger.debug({
            "event"        : "PUT",
            "experiment_id": experiment_id,
            "bucket_id"    : BUCKET_ID,
            "ball_id"      : plaintext_matrix_test_id,
            "matrix_id"    : plaintext_matrix_test_id,
            "shape"        : str(plaintext_matrix_test_response.shape),
            "dtype"        : str(plaintext_matrix_test_response.dtype),
            "read_time"    : plaintext_matrix_test_response.read_time,
            "segment_time" : plaintext_matrix_test_response.segment_time,
            "upload_time"  : plaintext_matrix_test_response.upload_time,
        })

        service_time_client = time.time() - arrivalTime
        managerResponse:RoryManager = current_app.config.get("manager")
        get_worker_start_time       = time.time()
        get_worker_result           = managerResponse.getWorker(
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
        (_worker_id,port) = get_worker_result.unwrap()

        get_worker_end_time     = time.time()
        get_worker_service_time = get_worker_end_time - get_worker_start_time
        worker_id               = "localhost" if TESTING else _worker_id
        worker_start_time       = time.time()

        worker = RoryWorker(
            workerId  = worker_id,
            port      = port,
            session   = s,
            algorithm = algorithm,
        )

        status = Constants.ClusteringStatus.START
        worker_headers = {
            "Clustering-Status"       : str(status),
            "Experiment-Id"           : experiment_id,
            "Plaintext-Matrix-Test-Id": plaintext_matrix_test_id,
            "Plaintext-Matrix-Train-Id": plaintext_matrix_train_id,
            "Weights-Id"              : weights_id,
            "Bias-Id"                 : bias_id
        }

        worker_response = worker.run(
                timeout = WORKER_TIMEOUT, 
                headers = worker_headers
            ) 
        worker_status = worker_response.status_code

        if worker_status !=200:
            return Response("Worker error: {}".format(worker_response.content),status=500)
        
        worker_response.raise_for_status()
        jsonWorkerResponse   = worker_response.json()
        predictions_id       = jsonWorkerResponse["predictions_id"]
        worker_service_time  = jsonWorkerResponse["service_time"]
        worker_end_time      = time.time()
        worker_response_time = worker_end_time - worker_start_time

        predictions_result = await storage_backend.get(
            bucket_id = BUCKET_ID,
            ball_id   = predictions_id,
            segment   = True,
            encrypt   = False,
        )
        if predictions_result.is_err:
            logger.error(f"Failed to get predictions: {predictions_result.unwrap_err()}")
            return Response(status=500, response="Failed to get predictions") 
        predictions_response = predictions_result.unwrap()
        predictions  = predictions_response.raw_value
        label_vector = predictions.astype(int).tolist()

        logger.debug({
            "event"        : "GET",
            "experiment_id": experiment_id,
            "bucket_id"    : BUCKET_ID,
            "ball_id"      : predictions_id,
            "matrix_id"    : predictions_id,
            "shape"        : str(predictions.shape),
            "dtype"        : str(predictions.dtype),
            "read_time"    : predictions_response.read_time,
        })

        response_time = time.time() - arrivalTime

        logger.info(ExperimentLogEntry(
            event         = "COMPLETED",
            experiment_id = experiment_id,
            algorithm     = algorithm,
            start_time    = arrivalTime,
            end_time      = time.time(),
            id            = plaintext_matrix_train_id,
            worker_id     = worker_id,
            client_time   = service_time_client,
            manager_time  = get_worker_service_time,
            worker_time   = worker_response_time,
        ).model_dump())

        return Response(
            response = json.dumps({
                "label_vector":label_vector,
                "algorithm":algorithm,
                "worker_id":worker_id,
                "service_time_manager":get_worker_service_time,
                "service_time_worker":worker_response_time,
                "service_time_client":service_time_client,
                "service_time_predict":response_time,               
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
        num_chunks                   = current_app.config.get("NUM_CHUNKS",1)
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        security_level               = current_app.config.get("LIU_SECURITY_LEVEL",128)
        if executor == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm                             = Constants.MachineLearningAlgorithms.PPLR_TRAIN
        MODE                                  = CkksModes.ML
        s                                     = Session()
        request_headers                       = request.headers 
        experiment_id                         = request_headers.get("Experiment-Id",uuid4().hex[:10])
        plaintext_matrix_train_id             = request_headers.get("Plaintext-Matrix-Train-Id","train_x")
        encrypted_matrix_train_id             = "encrypted{}".format(plaintext_matrix_train_id)
        plaintext_label_vector_train_id       = request_headers.get("Plaintext-Label-Vector-Train-Id","train_y")
        encrypted_label_vector_train_id       = "encrypted{}".format(plaintext_label_vector_train_id)
        plaintext_matrix_train_filename       = request_headers.get("Plaintext-Matrix-Train-Filename","train_x")
        plaintext_label_vector_train_filename = request_headers.get("Plaintext-Label-Vector-Train-Filename","train_y")
        extension                             = request_headers.get("Extension","csv")
        plaintext_matrix_train_path           = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_train_filename, extension)
        plaintext_label_vector_train_path     = "{}/{}.{}".format(SOURCE_PATH, plaintext_label_vector_train_filename, extension)
        total_epochs                          = int(request_headers.get("Epochs", "1"))
        learning_rate                         = float(request_headers.get("Learning-Rate", "0.01"))
        encrypted_weights_id                  = "{}encryptedweights".format(plaintext_matrix_train_id)
        encrypted_bias_id                     = "{}encryptedbias".format(plaintext_matrix_train_id)

        WORKER_TIMEOUT     = int(current_app.config.get("WORKER_TIMEOUT",300))
        MICTLANX_TIMEOUT   = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
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
            .with_storage_params(StorageParams(num_chunks=num_chunks, timeout=MICTLANX_TIMEOUT))
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
            "event"        : "PUT",
            "experiment_id": experiment_id,
            "bucket_id"    : BUCKET_ID,
            "ball_id"      : encrypted_matrix_train_id,
            "matrix_id"    : encrypted_matrix_train_id,
            "shape"        : str(plaintext_matrix_train_respose.shape),
            "dtype"        : str(plaintext_matrix_train_respose.dtype),
            "read_time"    : plaintext_matrix_train_respose.read_time,
            "segment_time" : plaintext_matrix_train_respose.segment_time,
            "encrypt_time" : getattr(plaintext_matrix_train_respose, "encrypt_time", 0.0),
            "upload_time"  : plaintext_matrix_train_respose.upload_time,
        })

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
            "event"        : "PUT",
            "experiment_id": experiment_id,
            "bucket_id"    : BUCKET_ID,
            "ball_id"      : encrypted_label_vector_train_id,
            "matrix_id"    : encrypted_label_vector_train_id,
            "shape"        : str(plaintext_label_vector_train_response.shape),
            "dtype"        : str(plaintext_label_vector_train_response.dtype),
            "read_time"    : plaintext_label_vector_train_response.read_time,
            "segment_time" : plaintext_label_vector_train_response.segment_time,
            "encrypt_time" : getattr(plaintext_label_vector_train_response, "encrypt_time", 0.0),
            "upload_time"  : plaintext_label_vector_train_response.upload_time,
        })

        scale            = ckks.SECURITY_LEVELS[MODE.value][security_level]["scale"]
        n_samples        = plaintext_matrix_train_respose.shape[0]
        n_features       = plaintext_matrix_train_respose.shape[1]
        plaintext_weight = np.zeros((1,n_features), dtype=np.float32)
        
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

        service_time_client         = time.time() - arrivalTime
        get_worker_start_time       = time.time()
        managerResponse:RoryManager = current_app.config.get("manager")
        get_worker_result           = managerResponse.getWorker( 
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
        (_worker_id,port) = get_worker_result.unwrap()

        logger.debug({
            "event":"GET.WORKER",
            "worker_id":_worker_id,
            "port":port,
            "is_local": TESTING
        })

        get_worker_end_time     = time.time() 
        get_worker_service_time = get_worker_end_time - get_worker_start_time
        worker_id               =  "localhost" if TESTING else _worker_id
        
        worker_start_time = time.time()
        worker = RoryWorker(
            workerId  = worker_id,
            port      = port,
            session   = s,
            algorithm = algorithm,
        )

        current_epoch = 0
        status = Constants.ClusteringStatus.START

        while current_epoch < total_epochs:
            
            if current_epoch > 0:
                status = Constants.ClusteringStatus.WORK_IN_PROGRESS

            worker_headers = {
                "Clustering-Status"              : str(status),
                "Experiment-Id"                  : experiment_id,
                "Learning-Rate"                  : str(learning_rate),
                "Encrypted-Matrix-Train-Id"      : encrypted_matrix_train_id,
                "Encrypted-Label-Vector-Train-Id": encrypted_label_vector_train_id,
                "Encrypted-Weights-Id"           : encrypted_weights_id,
                "Encrypted-Bias-Id"              : encrypted_bias_id,
                "Scale"                          : str(scale),
                "N-Features"                     : str(n_features),
                "N-Samples"                      : str(n_samples),
                "Num-Chunks"                     : str(num_chunks),
            }
            logger.debug({
                "event":"WORKER.RUN",
                "worker_id":_worker_id,
                "status":str(status),
                "experiment_id":experiment_id,
                "learning_rate":learning_rate,
                "encrypted_matrix_train_id": encrypted_matrix_train_id,
                "encrypted_label_vector_train_id":encrypted_label_vector_train_id,
                "encrypted_weights_id":encrypted_weights_id,
                "encrypted_bais_id":encrypted_bias_id,
                "scale":scale,
                "n_features": n_features,
                "n_samples": n_samples,
                "num_chunks": num_chunks,
                "total_epochs":total_epochs,
                "current_epoch": current_epoch
            })
            worker_run_start_time = time.time()
            worker_response = worker.run(
                    timeout = WORKER_TIMEOUT,
                    headers = worker_headers
                )
            worker_status = worker_response.status_code

            if worker_status !=200:
                logger.error(f"Worker execution failed at epoch {current_epoch + 1}: {worker_response.content}")
                return Response("Worker error: {}".format(worker_response.content),status=500)

            worker_response.raise_for_status()
            jsonWorkerResponse  = worker_response.json()
            worker_service_time = jsonWorkerResponse["service_time"]
            worker_end_time     = time.time()

            logger.info({
                "event":"WORKER.RUN.COMPLETED",
                "total_epochs":total_epochs,
                "current_epoch":current_epoch,
                "response_time":worker_end_time - worker_run_start_time
            })
            current_epoch += 1

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
                "shape"        : str(encrypted_weight_response.shape),
                "dtype"        : str(encrypted_weight_response.dtype),
                "read_time"    : encrypted_weights_response.read_time,
            })

            start_time_decryption = time.time()
            weights_plain_list    = ckks.decrypt_list(encrypted_weights, take=n_features)
            weights_plain         = weights_plain_list[0].reshape(1, -1).astype(np.float32)
            end_time_decryption   = time.time() - start_time_decryption
            logger.debug({
                "event"        : "DECRYPT",
                "experiment_id": experiment_id,
                "decrypt_time" : end_time_decryption,
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

            del weights_plain
            del encrypted_weights
            del weights_plain_list

            if encrypted_weight_result.is_err:
                logger.error("Failed to put encrypted weights in cloud storage: {}".format(encrypted_weight_result.unwrap_err()))
                return Response(status=500, response="Failed to put encrypted weights in cloud storage")
            encrypted_weight_put_response = encrypted_weight_result.unwrap()
            logger.debug({
                "event"        : "PUT",
                "experiment_id": experiment_id,
                "bucket_id"    : BUCKET_ID,
                "ball_id"      : encrypted_weights_id,
                "matrix_id"    : encrypted_weights_id,
                "shape"        : str(encrypted_weight_put_response.shape),
                "dtype"        : str(encrypted_weight_put_response.dtype),
                "read_time"    : getattr(encrypted_weight_put_response, "read_time", 0.0),
                "segment_time" : getattr(encrypted_weight_put_response, "segment_time", 0.0),
                "encrypt_time" : getattr(encrypted_weight_put_response, "encrypt_time", 0.0),
                "upload_time"  : getattr(encrypted_weight_put_response, "upload_time", 0.0),
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
                "shape"        : str(encrypted_bias_response.shape if hasattr(encrypted_bias_response, 'shape') else (1,)),
                "dtype"        : "float32",
                "read_time"    : encrypted_bias_response.read_time,
            })

            start_time_decryption = time.time()
            bias_plain_list       = ckks.decrypt_list(encrypted_bias, take=1)
            bias_plain            = bias_plain_list[0].reshape(1, -1).astype(np.float32)
            end_time_decryption   = time.time() - start_time_decryption
            logger.debug({
                "event"        : "DECRYPT.BIAS",
                "experiment_id": experiment_id,
                "encrypted_bias_id": encrypted_bias_id,
                "decrypt_time" : end_time_decryption,
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
            encrypted_bias_put_response = encrypted_bias_result.unwrap()
            logger.debug({
                "event"        : "PUT",
                "experiment_id": experiment_id,
                "bucket_id"    : BUCKET_ID,
                "ball_id"      : encrypted_bias_id,
                "matrix_id"    : encrypted_bias_id,
                "shape"        : str(encrypted_bias_put_response.shape),
                "dtype"        : str(encrypted_bias_put_response.dtype),
                "read_time"    : getattr(encrypted_bias_put_response, "read_time", 0.0),
                "segment_time" : getattr(encrypted_bias_put_response, "segment_time", 0.0),
                "encrypt_time" : getattr(encrypted_bias_put_response, "encrypt_time", 0.0),
                "upload_time"  : getattr(encrypted_bias_put_response, "upload_time", 0.0),
            })
            endTime    = time.time() 

        worker_response_time = worker_end_time - worker_start_time
        response_time        = endTime - arrivalTime

        logger.info(ExperimentLogEntry(
            event          = "COMPLETED",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = arrivalTime,
            end_time       = time.time(),
            id             = plaintext_matrix_train_id,
            epochs         = total_epochs,
            learning_rate  = learning_rate,
            worker_id      = worker_id,
            security_level = security_level,
            client_time    = service_time_client,
            manager_time   = get_worker_service_time,
            worker_time    = worker_response_time,
        ).model_dump())

        return Response(
            response = json.dumps({
                "algorithm": algorithm,
                "worker_id": worker_id,
                "epochs": total_epochs,
                "service_time_manager":get_worker_service_time,
                "service_time_worker":worker_response_time,
                "service_time_client":service_time_client,
                "service_time_train":response_time,
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
        request_headers      = request.headers
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

        _round             = bool(int(current_app.config.get("_round","0")))
        decimals           = int(current_app.config.get("DECIMALS","4"))
        keys_path          = current_app.config.get("KEYS_PATH","/rory/keys/keys128")
        ctx_filename       = current_app.config.get("CTX_FILENAME","ctx")
        pubkey_filename    = current_app.config.get("PUBKEY_FILENAME","pubkey")
        secretkey_filename = current_app.config.get("SECRET_KEY_FILENAME","secretkey")
        relinkey_filename  = current_app.config.get("RELINKEY_FILENAME","relinkey")
        rotatekey_filename = current_app.config.get("ROTATEKEY_FILENAME","rotatekey")
        WORKER_TIMEOUT     = int(current_app.config.get("WORKER_TIMEOUT",300))
        MICTLANX_TIMEOUT   = int(current_app.config.get("MICTLANX_TIMEOUT",3600))

        logger.debug({
            "event":"PPLR.PREDICT.STARTED",
            "experiment_id":experiment_id,
            "num_chunks":num_chunks
        })

        ckks = Ckks.from_pyfhel_client(
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
            .with_storage_params(StorageParams(num_chunks=num_chunks, timeout=MICTLANX_TIMEOUT))
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
            "event"        : "PUT",
            "experiment_id": experiment_id,
            "bucket_id"    : BUCKET_ID,
            "ball_id"      : encrypted_matrix_test_id,
            "matrix_id"    : encrypted_matrix_test_id,
            "shape"        : str(plaintext_matrix_test_response.shape),
            "dtype"        : str(plaintext_matrix_test_response.dtype),
            "read_time"    : plaintext_matrix_test_response.read_time,
            "segment_time" : plaintext_matrix_test_response.segment_time,
            "encrypt_time" : getattr(plaintext_matrix_test_response, "encrypt_time", 0.0),
            "upload_time"  : plaintext_matrix_test_response.upload_time,
        })

        scale            = ckks.SECURITY_LEVELS[MODE.value][security_level]["scale"]
        n_features       = plaintext_matrix_test_response.shape[1]

        service_time_client = time.time() - arrivalTime
        managerResponse:RoryManager = current_app.config.get("manager")
        get_worker_start_time       = time.time()
        get_worker_result           = managerResponse.getWorker(
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
        (_worker_id,port) = get_worker_result.unwrap()

        get_worker_end_time     = time.time() 
        get_worker_service_time = get_worker_end_time - get_worker_start_time
        worker_id               =  "localhost" if TESTING else _worker_id
        
        worker_start_time = time.time()
        worker = RoryWorker(
            workerId  = worker_id,
            port      = port,
            session   = s,
            algorithm = algorithm,
        )

        worker_headers = {
            "Experiment-Id"       : experiment_id,
            "Encrypted-Matrix-Test-Id": encrypted_matrix_test_id,
            "Encrypted-Weights-Id": encrypted_weights_id,
            "Encrypted-Bias-Id"   : encrypted_bias_id,
            "Scale"               : str(scale),
            "N-Features"          : str(n_features),
            "Num-Chunks"          : str(num_chunks),
        }

        worker_response = worker.run(
                timeout = WORKER_TIMEOUT, 
                headers = worker_headers
            )
        worker_status = worker_response.status_code

        if worker_status !=200:
            return Response("Worker error: {}".format(worker_response.content),status=500)
        
        worker_response.raise_for_status()
        jsonWorkerResponse       = worker_response.json()
        encrypted_predictions_id = jsonWorkerResponse["encrypted_predictions_id"]
        worker_service_time      = jsonWorkerResponse["service_time"]
        worker_end_time          = time.time()
        worker_response_time     = worker_end_time - worker_start_time

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
        encrypted_predictions_response = encrypted_predictions_result.unwrap()
        encrypted_predictions = encrypted_predictions_response.raw_value
        logger.debug({
            "event"        : "GET",
            "experiment_id": experiment_id,
            "bucket_id"    : BUCKET_ID,
            "ball_id"      : encrypted_predictions_id,
            "matrix_id"    : encrypted_predictions_id,
            "shape"        : str(plaintext_matrix_test_response.shape),
            "dtype"        : str(plaintext_matrix_test_response.dtype),
            "read_time"    : encrypted_predictions_response.read_time,
        })

        start_time_decryption = time.time()
        predictions_plain_list = ckks.decrypt_list(encrypted_predictions, take=1)
        predictions_plain = np.array([p[0] for p in predictions_plain_list], dtype=np.float32)
        end_time_decryption = time.time() - start_time_decryption
        logger.debug({
            "event":"DECRYPT",
            "experiment_id":experiment_id,
            "decrypt_time":end_time_decryption,
        })

        label_vector = [1 if v >= 0.5 else 0 for v in predictions_plain]
        response_time = time.time() - arrivalTime

        logger.info(ExperimentLogEntry(
            event          = "COMPLETED",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = arrivalTime,
            end_time       = time.time(),
            id             = plaintext_matrix_train_id,
            worker_id      = worker_id,
            security_level = security_level,
            client_time    = service_time_client,
            manager_time   = get_worker_service_time,
            worker_time    = worker_response_time,
        ).model_dump())

        return Response(
            response = json.dumps({
                "label_vector":label_vector,
                "algorithm":algorithm,
                "worker_id":worker_id,
                "service_time_manager":get_worker_service_time,
                "service_time_worker":worker_response_time,
                "service_time_client":service_time_client,
                "service_time_predict":response_time,
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
    

    