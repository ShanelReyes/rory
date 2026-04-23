import os
import time, json
import numpy as np
from uuid import uuid4
from requests import Session
from flask import Blueprint,current_app,request,Response
from rory.core.interfaces.rorymanager import RoryManager
from rory.core.interfaces.roryworker import RoryWorker
from rory.core.security.dataowner import DataOwner
from rory.core.security.pqc.dataowner import DataOwner as DataOwnerPQC
from rory.core.security.cryptosystem.liu import Liu
from rory.core.utils.constants import Constants
from rorycommon import Common as RoryCommon
from rory.core.utils.utils import Utils
from mictlanx import AsyncClient
from mictlanx.utils.segmentation import Chunks
from concurrent.futures import ProcessPoolExecutor
from option import Some
from models import ExperimentLogEntry
from rory.core.security.cryptosystem.pqc.ckks import Ckks

machinelearning = Blueprint("machinelearning",__name__,url_prefix = "/machinelearning")

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

@machinelearning.route("/logisticregression",methods = ["POST"])
async def logisticregression():
    """
    This method implements an interactive, logistic regression protocol with raw data. The workflow is designed for 
    Machine Learning as a Service (MLaaS), where the Client (Data Owner) provides their information to get a result.
    
    Attributes:
        Plaintext-Matrix-Id (str): Unique ID for the matrix. Defaults to "matrix0".
        Plaintext-Matrix-Filename (str): Local filename for data reading. Defaults to "matrix0".
        Extension (str): File extension of the dataset. Defaults to "csv".
        E (int): Number of epochs to rounds. **Required**.
        Max-Iterations (int): Maximum number of protocol rounds. Defaults to 10.
        Experiment-Id (str): Tracking ID for performance auditing.
        Experiment-Iteration (str): Current loop index of the experiment.

    Returns:
        prediction (list): Final predictions.
        iterations (int): Actual number of iterations performed.
        worker_id (str): ID of the node that performed the secure computations.
        service_time_manager (float): Time spent in Worker allocation.
        service_time_worker (float): Cumulative time of remote computation.
        service_time_client (float): Total local time (Encryption/Decryption/IO).
        response_time_clustering (float): End-to-end execution time.
    """
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
        algorithm                       = Constants.MachineLearningAlgorithms.LOGISTIC_REGRESSION
        s                               = Session()
        request_headers                 = request.headers #Headers for the request
        experiment_id                   = request_headers.get("Experiment-Id",uuid4().hex[:10])
        plaintext_matrix_train_id       = request_headers.get("Plaintext-Matrix-Train-Id","matrix0")
        plaintext_matrix_test_id        = request_headers.get("Plaintext-Matrix-Test-Id","matrix1")
        plaintext_matrix_train_filename = request_headers.get("Plaintext-Matrix-Train-Filename","matrix0")
        plaintext_matrix_test_filename  = request_headers.get("Plaintext-Matrix-Test-Filename","matrix1")
        extension                       = request_headers.get("Extension","csv")
        epochs                          = int(request_headers.get("Epochs", 1))
        learning_rate                   = float(request_headers.get("Learning-Rate", 0.01))
        experiment_iteration            = request_headers.get("Experiment-Iteration","0")
        experiment_id                   = request_headers.get("Experiment-Id",uuid4().hex[:10])
        # requestId_train                 = "request-{}".format(plaintext_matrix_train_id)
        # requestId_test                  = "request-{}".format(plaintext_matrix_test_id)
        plaintext_matrix_train_path     = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_train_filename, extension)    
        plaintext_matrix_test_path     = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_test_filename, extension) 

        MAX_ITERATIONS          = int(request_headers.get("Max-Iterations",current_app.config.get("MAX_ITERATIONS",10)))
        WORKER_TIMEOUT          = int(current_app.config.get("WORKER_TIMEOUT",300))
        MICTLANX_TIMEOUT        = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
        MICTLANX_DELAY          = int(current_app.config.get("MICTLANX_DELAY","2"))
        MICTLANX_BACKOFF_FACTOR = float(current_app.config.get("MICTLANX_BACKOFF_FACTOR","0.5"))
        MICTLANX_MAX_RETRIES    = int(current_app.config.get("MICTLANX_MAX_RETRIES","10"))

        
        local_read_dataset_start_time = time.time()
        records_test_result = await RoryCommon.read_numpy_from(
            path=plaintext_matrix_test_path, 
            extension=extension
        )
        if records_test_result.is_err:
            return Response(status=500, response="Failed to local read the records")
        
        records_test = records_test_result.unwrap().astype(np.float32)

        local_read_entry = ExperimentLogEntry(
            event          = "LOCAL.READ",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = local_read_dataset_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_test_id,
            worker_id      = "",
            num_chunks     = num_chunks,
            e              = e,
            workers        = max_workers,
        )
        logger.debug(local_read_entry.model_dump()) #Debug en lugar de info

        try: 
            put_records_start_time    = time.time()
            maybe_records_test_chunks = Chunks.from_ndarray(
                ndarray      = records_test,
                group_id     = plaintext_matrix_test_id,
                chunk_prefix = Some(plaintext_matrix_test_id),
                num_chunks   = num_chunks,
            )

            if maybe_records_test_chunks.is_none:
                logger.error({
                    "error":"Failed to create chunks"
                })
                return Response(status=500,response="something went wrong creating the chunks")
                
            put_records_test_result = await RoryCommon.delete_and_put_chunks(
                client    = STORAGE_CLIENT,
                bucket_id = BUCKET_ID,
                key       = plaintext_matrix_test_id,
                chunks    = maybe_records_test_chunks.unwrap(),
                timeout   = MICTLANX_TIMEOUT,
                tags      = {
                    "full_shape": str(records_test.shape),
                    "full_dtype": str(records_test.dtype)
                }
            )

            if put_records_test_result.is_err:
                logger.error(str(put_records_test_result.unwrap_err()))
                return Response(status=500, response="Failed to put the records test")
        except Exception as e:
            logger.error(str(e))

        service_time_client_end = time.time()
        service_time_client = service_time_client_end - arrivalTime
            
        put_records_entry = ExperimentLogEntry(
            event          = "PUT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = put_records_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_test_id,
            worker_id      = "",
            num_chunks     = num_chunks,
        )
        logger.info(put_records_entry.model_dump())

        managerResponse:RoryManager = current_app.config.get("manager") # Communicates with the manager
        get_worker_start_time       = time.time()
        get_worker_result           = managerResponse.getWorker( #Gets the worker from the manager
            headers = {
                "Algorithm"             : algorithm,
                "Start-Request-Time"    : str(arrivalTime),
                "Start-Get-Worker-Time" : str(get_worker_start_time),
                "Matrix-Id"             : plaintext_matrix_test_id
            }
        )

        if get_worker_result.is_err:
            error = get_worker_result.unwrap_err()
            logger.error(str(error))
            return Response(response=str(error), status=500)
            
        (_worker_id,port) = get_worker_result.unwrap()

        get_worker_end_time     = time.time() 
        get_worker_service_time = get_worker_end_time - get_worker_start_time
        worker_id               =  "localhost" if TESTING else _worker_id

        get_worker_entry = ExperimentLogEntry(
            event          = "GET.WORKER",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_worker_start_time,
            end_time       = time.time(),
            worker_id      = worker_id,
            num_chunks     = num_chunks,
        )
        logger.info(get_worker_entry.model_dump())

        worker_start_time = time.time()
        worker            = RoryWorker( #Allows to establish the connection with the worker
            workerId  = worker_id,
            port      = port,
            session   = s,
            algorithm = algorithm,
        )

        workerResponse = worker.run(
            headers    = {
                "Model-Labels-Shape":request_headers["Model-Labels-Shape"]
            },
            timeout = WORKER_TIMEOUT
        )
        workerResponse.raise_for_status()
            
        worker_end_time      = time.time()
        worker_response_time = worker_end_time - worker_start_time 
        jsonWorkerResponse   = workerResponse.json()
        endTime              = time.time() # Get the time when it ends
        worker_service_time  = jsonWorkerResponse["service_time"]
        label_vector         = jsonWorkerResponse["label_vector"]
        response_time        = endTime - arrivalTime# Get the service time
            
        classification_completed_entry = ExperimentLogEntry(
            event          = "COMPLETED",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = arrivalTime,
            end_time       = time.time(),
            num_chunks     = num_chunks,
            workers        = max_workers,
            time           = response_time,
        )
        logger.debug(classification_completed_entry.model_dump())

        return Response(
            response = json.dumps({
                "label_vector":label_vector,
                "worker_id":worker_id,
                "service_time_manager":get_worker_service_time,
                "service_time_worker":worker_response_time,
                "service_time_client":service_time_client,
                "service_time_predict":response_time,
                "algorithm":algorithm,
            }),
            status   = 200,
            headers  = {}
            )
    except Exception as e:
        logger.error("CLIENT_ERROR "+str(e))
        return Response(response = None, status = 500, headers={"Error-Message":str(e)})


        


        

@machinelearning.route("/pplr",methods = ["POST"])
async def pplr():
    """
    This method implements an interactive, privacy-preserving logistic regression protocol 
    powered by CKKS homomorphic encryption scheme. The workflow is designed for 
    Privacy-Preserving Machine Learning as a Service (PPMLaaS), where the Client (Data Owner) 
    remains the only entity capable of decrypting intermediate computations.

    
    Attributes:
       
        Plaintext-Matrix-Id (str): Unique ID for the matrix. Defaults to "matrix0".
        Plaintext-Matrix-Filename (str): Local filename for data reading. Defaults to "matrix0".
        Extension (str): File extension of the dataset. Defaults to "csv".
        E (int): Number of epochs to rounds. **Required**.
        Max-Iterations (int): Maximum number of protocol rounds. Defaults to 10.
        Experiment-Id (str): Tracking ID for performance auditing.
        Experiment-Iteration (str): Current loop index of the experiment.

    Returns:
        prediction (list): Final predictions.
        iterations (int): Actual number of iterations performed.
        worker_id (str): ID of the node that performed the secure computations.
        service_time_manager (float): Time spent in Worker allocation.
        service_time_worker (float): Cumulative time of remote computation.
        service_time_client (float): Total local time (Encryption/Decryption/IO).
        response_time_clustering (float): End-to-end execution time.

    Raises:
        
    """
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
        security_level               = current_app.config.get("LIU_SECURITY_LEVEL",128)
        
        if executor == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm                       = Constants.MachineLearningAlgorithms.PPLR
        s                               = Session()
        request_headers                 = request.headers #Headers for the request
        experiment_id                   = request_headers.get("Experiment-Id",uuid4().hex[:10])
        experiment_iteration            = request_headers.get("Experiment-Iteration","0")

        plaintext_matrix_train_id       = request_headers.get("Plaintext-Matrix-Train-Id","matrix-train")
        encrypted_matrix_train_id       = "encrypted{}".format(plaintext_matrix_train_id) # The id of the encrypted matrix is built
        plaintext_matrix_test_id        = request_headers.get("Plaintext-Matrix-Test-Id","matrix-test")
        encrypted_matrix_test_id        = "encrypted{}".format(plaintext_matrix_test_id)
        plaintext_matrix_train_filename = request_headers.get("Plaintext-Matrix-Train-Filename","matrix-train")
        plaintext_matrix_test_filename  = request_headers.get("Plaintext-Matrix-Test-Filename","matrix-test")
        extension                       = request_headers.get("Extension","csv")
        plaintext_matrix_train_path     = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_train_filename, extension)    
        plaintext_matrix_test_path      = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_test_filename, extension) 

        epochs                          = int(request_headers.get("Epochs", "1"))
        learning_rate                   = float(request_headers.get("Learning-Rate", 0.01))
        accuracy_threshold              = float(request_headers.get("Accuracy-Threshold", 0.80))
        
       

        _round             = bool(int(current_app.config.get("_round","0"))) #False
        decimals           = int(current_app.config.get("DECIMALS","4"))
        path               = current_app.config.get("KEYS_PATH","/rory/keys")
        ctx_filename       = current_app.config.get("CTX_FILENAME","ctx")
        pubkey_filename    = current_app.config.get("PUBKEY_FILENAME","pubkey")
        secretkey_filename = current_app.config.get("SECRET_KEY_FILENAME","secretkey")
        relinkey_filename  = current_app.config.get("RELINKEY_FILENAME","relinkey")
        rotatekey_filename  = current_app.config.get("ROTATEKEY_FILENAME","rotatekey")
        # max_workers        = Utils.get_workers(num_chunks=num_chunks)

        MAX_ITERATIONS          = int(request_headers.get("Max-Iterations",current_app.config.get("MAX_ITERATIONS",10)))
        WORKER_TIMEOUT          = int(current_app.config.get("WORKER_TIMEOUT",300))
        MICTLANX_TIMEOUT        = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
        MICTLANX_DELAY          = int(current_app.config.get("MICTLANX_DELAY","2"))
        MICTLANX_BACKOFF_FACTOR = float(current_app.config.get("MICTLANX_BACKOFF_FACTOR","0.5"))
        MICTLANX_MAX_RETRIES    = int(current_app.config.get("MICTLANX_MAX_RETRIES","10"))

        logger.debug({"Debug":"Starting the PPLR protocol. The actual implementation is in progress."})

        
        return Response(
            response = json.dumps({
                "epochs" : epochs,
                "learning_rate" : learning_rate,
                "accuracy_threshold" :  accuracy_threshold,
            }),
            status   = 200,
            headers  = {}
        )

    except Exception as e:
        logger.error({
            "msg":str(e)
        })
        return Response(response = None, status = 500, headers={"Error-Message":str(e)})


        


        