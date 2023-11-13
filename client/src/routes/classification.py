import os
import time, json
import numpy as np
import pandas as pd
from typing import List,Generator,Awaitable,Iterator
import numpy.typing as npt
from requests import Session
from flask import Blueprint,current_app,request,Response
from option import Some
from option import Result
from rory.core.interfaces.rorymanager import RoryManager
from rory.core.interfaces.roryworker import RoryWorker
from rory.core.interfaces.metricsResult_external import MetricsResultExternal
from rory.core.interfaces.metricsResult_internal import MetricsResultInternal
from rory.core.security.dataowner import DataOwner
from rory.core.security.cryptosystem.liu import Liu
from rory.core.security.cryptosystem.FDHOpe import Fdhope
from rory.core.utils.constants import Constants
from rory.core.utils.Utils import Utils
from rory.core.utils.SegmentationUtils import Segmentation
from rory.core.validation_index.metrics import internal_validation_indexes,external_validation_indexes
from rory.core.interfaces.logger_metrics import LoggerMetrics
from mictlanx.v3.client import Client 
from mictlanx.v4.interfaces.responses import PutResponse
from mictlanx.v4.client import Client  as V4Client
from mictlanx.v4.interfaces.responses import GetNDArrayResponse
from mictlanx.utils.segmentation import Chunks,Chunk
from mictlanx.v3.interfaces.payloads import PutNDArrayPayload
from concurrent.futures import ProcessPoolExecutor

classification = Blueprint("classification",__name__,url_prefix = "/classification")

@classification.route("/test",methods=["GET","POST"])
def test():
    return Response(
        response = json.dumps({
            "component_type":"client"
        }),
        status   = 200,
        headers  = {
            "Component-Type":"client"
        }
    )

@classification.route("/sknn/train",methods = ["POST"])
def sknn_train():
    try:
        arrivalTime                  = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        TESTING                      = current_app.config.get("TESTING",True)
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        liu:Liu                      = current_app.config.get("liu")
        dataowner:DataOwner          = current_app.config.get("dataowner")
        STORAGE_CLIENT:V4Client      = current_app.config.get("STORAGE_CLIENT")
        num_chunks                   = current_app.config.get("NUM_CHUNKS",4)
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        logger.debug("INIT_TRAIN")
        if executor == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm          = Constants.ClassificationAlgorithms.SKNN_TRAIN
        s                  = Session()
        requestHeaders     = request.headers #Headers for the request
        model_id           = requestHeaders.get("Model-Id","matrix-0_model")        
        model_labels_id    = "{}_labels".format(model_id)
        encrypted_model_id = "encrypted-{}".format(model_id) #encrypted-iris_model
        extension          = requestHeaders.get("Extension","npy")
        m                  = requestHeaders.get("M","3")
        model_path         = "{}/{}.{}".format(SOURCE_PATH, model_id, extension)
        model_labels_path  = "{}/{}.{}".format(SOURCE_PATH, model_labels_id, extension)
        logger.debug("SKNN TRAIN algorithm={}, m={}, model_id={}, model_labels_id={}, encrypted_model_id={}".format(algorithm, m, model_id, model_labels_id, encrypted_model_id))

        model_path_exists = os.path.exists(model_path) 
        model_path_labels_exists = os.path.exists(model_labels_path)
        logger.debug("model_path_exists={}, MPLEAA={} Ext={}".format(model_path_exists,model_path_labels_exists,extension))
        if not model_path_exists or not model_path_labels_exists:
            return Response(response="Either model or label vector not found", status=500)
        else:
            
            logger.debug("Client starts to process {} at {}".format(model_id,arrivalTime))
            with open(model_path, "rb") as f:
                model = np.load(f)
            logger.debug("OPEN MODEL SUCCESSFULLY")

            with open(model_labels_path, "rb") as f:
                model_labels:npt.NDArray = np.load(f)
                model_labels = model_labels.astype(np.int16)
            logger.debug("OPEN MODEL_LABELS SUCCESSFULLY")
            
            X = STORAGE_CLIENT.put_ndarray(
                key       = model_labels_id,
                ndarray   = model_labels,
                tags      = {},
                bucket_id = BUCKET_ID
            ).result()
            logger.debug("MODEL LABELS PUT SUCCESSFULLY")

            encryption_start_time = time.time()
            r                     = model.shape[0]
            a                     = model.shape[1]
            encrypted_model_shape = "({},{},{})".format(r,a,m),

            encrypted_model_chunks:Chunks = Segmentation.segment_and_encrypt_liu_with_executor( #Encrypt 
                executor         = executor,
                key              = encrypted_model_id,
                plaintext_matrix = model,
                dataowner        = dataowner,
                n                  = a*r*m,
                num_chunks       = num_chunks
            )
            logger.debug("SEGMENT AND ENCRYPT WITH LIU")
            logger.debug("_"*35)
            chunks = encrypted_model_chunks.iter()

            logger.debug("{} {} {}".format(type(encrypted_model_id), encrypted_model_chunks,type(BUCKET_ID)))
            put_chunks_generator_results = STORAGE_CLIENT.put_chunks(
                key       = encrypted_model_id, 
                chunks    = encrypted_model_chunks, 
                bucket_id = BUCKET_ID,
                tags      = {}
            )

            for i,put_chunk_result in enumerate(put_chunks_generator_results):
                encryption_end_time    = time.time()
                encryption_time        = encryption_end_time - encryption_start_time
                encrypt_logger_metrics = LoggerMetrics( #Write times of encrypt in logger
                    operation_type = "ENCRYPT_CHUNK",
                    matrix_id      = model_id,
                    algorithm      = algorithm,
                    arrival_time   = encryption_start_time,
                    end_time       = encryption_end_time, 
                    service_time   = encryption_time,
                    m_value        = m
                ) 
                if put_chunk_result.is_err:
                    logger.error("Something went wrong storage and encrypt the chunk.")
                    return Response(
                        status   = 500,
                        response = "{}".format(str(put_chunk_result.unwrap_err()))
                    )
                logger.info(str(encrypt_logger_metrics)+","+str(i))

            encryption_end_time    = time.time()
            encryption_time        = encryption_end_time - encryption_start_time
            encrypt_logger_metrics = LoggerMetrics( #Write times of encrypt in logger
                operation_type = "ENCRYPT",
                matrix_id      = model_id,
                algorithm      = algorithm,
                arrival_time   = encryption_start_time,
                end_time       = encryption_end_time, 
                service_time   = encryption_time,
                m_value        = m
            ) 
            logger.info(str(encrypt_logger_metrics))
            
            endTime        = time.time() # Get the time when it ends
            response_time  = endTime - arrivalTime # Get the service time
            logger_metrics = LoggerMetrics(
                operation_type = algorithm, 
                matrix_id      = model_id, 
                algorithm      = algorithm, 
                arrival_time   = arrivalTime, 
                end_time       = endTime, 
                service_time   = response_time
            )
            logger.info(str(logger_metrics))
            logger.debug("_"*50)
            return Response(
                response = json.dumps({
                    "responseTime": str(response_time),
                    "algorithm"   : algorithm,
                }),
                status  = 200,
                headers = {
                    "Encrypted-Model-Shape":encrypted_model_shape,
                    "Encrypted-Model-Dtype":"float64"
                }
            )
    except Exception as e:
        logger.error("CLIENT_ERROR "+str(e))
        return Response(response = None, status = 500, headers={"Error-Message":str(e)})


@classification.route("/sknn/predict",methods = ["POST"])
def sknn_predict():
    try:
        arrivalTime                  = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        TESTING                      = current_app.config.get("TESTING",True)
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        liu:Liu                      = current_app.config.get("liu")
        dataowner:DataOwner          = current_app.config.get("dataowner")
        STORAGE_CLIENT:V4Client      = current_app.config.get("STORAGE_CLIENT")
        num_chunks                   = current_app.config.get("NUM_CHUNKS",4)
        max_workers                  = current_app.config.get("MAX_WORKERS",2)
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        WORKER_TIMEOUT               = int(current_app.config.get("WORKER_TIMEOUT",300))
        if executor == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm                 = Constants.ClassificationAlgorithms.SKNN_PREDICT
        s                         = Session()
        requestHeaders            = request.headers #Headers for the request
        model_id                  = requestHeaders.get("Model-Id","model-0") #iris
        records_test_id           = requestHeaders.get("Records-Test-Id","matrix-0_data")
        encrypted_records_test_id = "encrypted-{}".format(records_test_id) # The id of the encrypted matrix is built
        extension                 = requestHeaders.get("Extension","npy")
        m                         = requestHeaders.get("M","3")
        _encrypted_model_shape    = requestHeaders.get("Encrypted-Model-Shape",-1)
        _encrypted_model_dtype    = requestHeaders.get("Encrypted-Model-Dtype",-1)
        records_test_path         = "{}/{}.{}".format(SOURCE_PATH, records_test_id, extension)
        logger.debug("SKNN PREDICT algorithm={}, m={}, model_id={}, records_test_id={}".format(algorithm, m, model_id, records_test_id))
        
        if _encrypted_model_dtype == -1:
            return Response("Encrypted-Model-Dtype", status=500)
        if _encrypted_model_shape == -1 :
            return Response("Encrypted-Model-Shape header is required", status=500)
    

        with open(records_test_path, "rb") as f:
            records_test = np.load(f)    
        logger.debug("OPEN RECORDS SUCCESSFULLY")

        
        r           = records_test.shape[0]
        a           = records_test.shape[1]
        cores       = os.cpu_count()
        max_workers = num_chunks if max_workers > num_chunks else max_workers
        max_workers = cores if max_workers > cores else max_workers
        logger.debug("ENCRYPT_WORKERS {}".format(max_workers))        
        encryption_start_time = time.time()
        logger.debug("NUM_CHUNKS {}".format(num_chunks))
        logger.debug("RECORDS SHAPE {}".format(records_test.shape))
        encryption_start_time = time.time()

        encrypted_records_chunks:Chunks = Segmentation.segment_and_encrypt_liu_with_executor( #Encrypt 
            executor         = executor,
            key              = encrypted_records_test_id,
            plaintext_matrix = records_test,
            dataowner        = dataowner,
            n                = a*r*m,
            num_chunks       = num_chunks
        )

        logger.debug("SEGMENTATION AND ENCRYPT WITH LIU")
        
        chunks = encrypted_records_chunks.iter()

        logger.debug("{} {} {}".format(type(encrypted_records_test_id), encrypted_records_chunks,type(BUCKET_ID)))
        put_chunks_generator_results = STORAGE_CLIENT.put_chunks(
            key       = encrypted_records_test_id, 
            chunks    = encrypted_records_chunks, 
            bucket_id = BUCKET_ID,
            tags      = {}
        )

        for i,put_chunk_result in enumerate(put_chunks_generator_results):
            encryption_end_time    = time.time()
            encryption_time        = encryption_end_time - encryption_start_time
            encrypt_logger_metrics = LoggerMetrics( #Write times of encrypt in logger
                operation_type = "ENCRYPT_CHUNK",
                matrix_id      = records_test_id,
                algorithm      = algorithm,
                arrival_time   = encryption_start_time,
                end_time       = encryption_end_time, 
                service_time   = encryption_time,
                m_value        = m
            ) 
            if put_chunk_result.is_err:
                logger.error("Something went wrong storage and encrypt the chunk.")
                return Response(
                    status   = 500,
                    response = "{}".format(str(put_chunk_result.unwrap_err()))
                )
            logger.info(str(encrypt_logger_metrics)+","+str(i))

        encryption_end_time    = time.time()
        encryption_time        = encryption_end_time - encryption_start_time
        encrypt_logger_metrics = LoggerMetrics( #Write times of encrypt in logger
            operation_type = "ENCRYPT",
            matrix_id      = records_test_id,
            algorithm      = algorithm,
            arrival_time   = encryption_start_time,
            end_time       = encryption_end_time, 
            service_time   = encryption_time,
            m_value        = m
        ) 
        logger.info(str(encrypt_logger_metrics))

        managerResponse:RoryManager = current_app.config.get("manager") # Communicates with the manager
        get_worker_start_time = time.time()
        mr                    = managerResponse.getWorker( #Gets the worker from the manager
            headers = {
                "Algorithm"             : algorithm,
                "Start-Request-Time"    : str(arrivalTime),
                "Start-Get-Worker-Time" : str(get_worker_start_time) 
            }
        )
        logger.debug("GET WORKER SUCCESSFULLY")
        get_worker_end_time     = time.time() 
        get_worker_service_time = get_worker_end_time - get_worker_start_time
        stringResponse          = mr.content.decode("utf-8") #Decode the manager's response
        jsonResponse            = json.loads(stringResponse) # Pass the response to json
        workerId                =  "localhost" if TESTING else jsonResponse["workerId"]

        get_worker_logger_metrics = LoggerMetrics( # Write times of worker communication in logger
            operation_type = "GET_WORKER",
            matrix_id      = records_test_id,
            algorithm      = algorithm,
            arrival_time   = get_worker_start_time, 
            end_time       = get_worker_end_time, 
            service_time   = get_worker_service_time,
            m_value        = m,
            worker_id      = workerId
        )
        logger.info(str(get_worker_logger_metrics))

        worker         = RoryWorker( #Allows to establish the connection with the worker
            workerId   = workerId,
            port       = jsonResponse["workerPort"],
            session    = s,
            algorithm  = algorithm,
        )
        worker_arrival_time = time.time()
        logger.debug("RORY WORKER SUCCESSFULLY")
        run_headers = {
            "Records-Test-Id"         : records_test_id,
            "Model-Id"                : model_id,
            "Encrypted-Model-Shape"   : _encrypted_model_shape,
            "Encrypted-Records-Shape" : "({},{},{})".format(r,a,m),
            "Encrypted-Model-Dtype"   : _encrypted_model_dtype,
            "Encrypted-Records-Dtype" : "float64",
            "Num-Chunks"              : str(num_chunks),

        }
        workerResponse = worker.run(
            headers    = run_headers,
            timeout = WORKER_TIMEOUT
        )
        logger.debug("RUN_WORKER_RESPONSE {}".format(workerResponse))

        worker_end_time       = time.time()
        worker_service_time   = worker_end_time - worker_arrival_time 
        interaction_logger_metrics = LoggerMetrics(
            operation_type = "WORKER",
            matrix_id      = records_test_id,
            algorithm      = algorithm, 
            arrival_time   = worker_arrival_time, 
            end_time       = worker_end_time, 
            service_time   = worker_service_time,
            worker_id      = workerId
        )
        logger.info(str(interaction_logger_metrics))

        stringWorkerResponse = workerResponse.content.decode("utf-8") #Response from worker
        jsonWorkerResponse   = json.loads(stringWorkerResponse) #pass to json
        endTime              = time.time() # Get the time when it ends
        worker_service_time  = workerResponse.headers.get("Service-Time",0) # Extract the time at which it started]]
        response_time        = endTime - arrivalTime # Get the service time
        logger_metrics = LoggerMetrics(
            operation_type = algorithm, 
            matrix_id      = records_test_id, 
            algorithm      = algorithm, 
            arrival_time   = arrivalTime, 
            end_time       = endTime, 
            service_time   = response_time,
            worker_id      = workerId
        )
        logger.info(str(logger_metrics))
        logger.debug("_"*50)
        return Response(
            response = json.dumps({
                "labelVector" : jsonWorkerResponse.get("labelVector",[]),
                "serviceTime" : worker_service_time,
                "responseTime": str(response_time),
                "algorithm"   : algorithm,
            }),
            status   = 200,
            headers  = {}
        )
    except Exception as e:
        logger.error("CLIENT_ERROR "+str(e))
        return Response(response = None, status = 500, headers={"Error-Message":str(e)})


@classification.route("/knn/train", methods = ["POST"])
def knn_train():
    arrivalTime                  = time.time()
    logger                       = current_app.config["logger"]
    BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
    TESTING                      = current_app.config.get("TESTING",True)
    SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
    STORAGE_CLIENT:V4Client      = current_app.config.get("STORAGE_CLIENT")
    executor:ProcessPoolExecutor = current_app.config.get("executor")
    WORKER_TIMEOUT               = int(current_app.config.get("WORKER_TIMEOUT",300))
    if executor == None:
        raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
    algorithm         = Constants.ClassificationAlgorithms.KNN_TRAIN
    s                 = Session()
    requestHeaders    = request.headers #Headers for the request
    model_id          = requestHeaders.get("Model-Id","matrix-0_model")        
    model_labels_id   = "{}_labels".format(model_id)
    extension         = requestHeaders.get("Extension","npy")
    model_path        = "{}/{}.{}".format(SOURCE_PATH, model_id, extension)
    model_labels_path = "{}/{}.{}".format(SOURCE_PATH, model_labels_id, extension)

    logger.debug("_"*50)
    logger.debug("Client starts to process {} at {}".format(model_id,arrivalTime))
    with open(model_path, "rb") as f:
        model = np.load(f)
    logger.debug("OPEN MODEL SUCCESSFULLY")
    
    with open(model_labels_path, "rb") as f:
        model_labels:npt.NDArray = np.load(f)
        model_labels             = model_labels.astype(np.int16)
    logger.debug("OPEN MODEL_LABELS SUCCESSFULLY")

    model_result = STORAGE_CLIENT.put_ndarray(
        key       = model_id,
        ndarray   = model,
        tags      = {},
        bucket_id = BUCKET_ID
    ).result()
    logger.debug("MODEL RESULT PUT SUCCESSFULLY")

    model_labels_result = STORAGE_CLIENT.put_ndarray(
        key       = model_labels_id,
        ndarray   = model_labels,
        tags      = {},
        bucket_id = BUCKET_ID
    ).result()
    logger.debug("MODEL LABELS RESULT PUT SUCCESSFULLY")

    managerResponse:RoryManager = current_app.config.get("manager") # Communicates with the manager
    get_worker_start_time = time.time()
    mr                    = managerResponse.getWorker( #Gets the worker from the manager
        headers = {
            "Algorithm"             : algorithm,
            "Start-Request-Time"    : str(arrivalTime),
            "Start-Get-Worker-Time" : str(get_worker_start_time) 
        }
    )
    logger.debug("GET WORKER SUCCESSFULLY")
    get_worker_end_time     = time.time() 
    get_worker_service_time = get_worker_end_time - get_worker_start_time
    stringResponse          = mr.content.decode("utf-8") #Decode the manager's response
    jsonResponse            = json.loads(stringResponse) # Pass the response to json
    workerId                =  "localhost" if TESTING else jsonResponse["workerId"]

    get_worker_logger_metrics = LoggerMetrics( # Write times of worker communication in logger
        operation_type = "GET_WORKER",
        matrix_id      = model_id,
        algorithm      = algorithm,
        arrival_time   = get_worker_start_time, 
        end_time       = get_worker_end_time, 
        service_time   = get_worker_service_time,
        worker_id      = workerId
    )
    logger.info(str(get_worker_logger_metrics))

    worker         = RoryWorker( #Allows to establish the connection with the worker
        workerId   = workerId,
        port       = jsonResponse["workerPort"],
        session    = s,
        algorithm  = algorithm,
    )
    logger.debug("RORY WORKER SUCCESSFULLY")
    worker_arrival_time = time.time()
    workerResponse = worker.run(
        headers    = {
            "Model-Id": model_id,
        },
        timeout = WORKER_TIMEOUT
    )

    worker_end_time       = time.time()
    worker_service_time   = worker_end_time - worker_arrival_time 
    interaction_logger_metrics = LoggerMetrics(
        operation_type = "WORKER",
        matrix_id      = model_id,
        algorithm      = algorithm, 
        arrival_time   = worker_arrival_time, 
        end_time       = worker_end_time, 
        service_time   = worker_service_time,
        worker_id      = workerId
    )
    logger.info(str(interaction_logger_metrics))
    endTime             = time.time() # Get the time when it ends
    response_time       = endTime - arrivalTime # Get the service time
    
    logger_metrics = LoggerMetrics(
        operation_type = algorithm, 
        matrix_id      = model_id, 
        algorithm      = algorithm, 
        arrival_time   = arrivalTime, 
        end_time       = endTime, 
        service_time   = response_time,
        worker_id      = workerId
    )
    logger.info(str(logger_metrics))

    return Response(
        response = json.dumps({
            "serviceTime" : worker_service_time,
            "responseTime": str(response_time),
            "algorithm"   : algorithm,
        }),
        status   = 200,
        headers  = {}
    )


@classification.route("/knn/predict",methods = ["POST"])
def knn_predict():
    try:
        arrivalTime                  = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        TESTING                      = current_app.config.get("TESTING",True)
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        liu:Liu                      = current_app.config.get("liu")
        dataowner:DataOwner          = current_app.config.get("dataowner")
        STORAGE_CLIENT:V4Client      = current_app.config.get("STORAGE_CLIENT")
        num_chunks                   = current_app.config.get("NUM_CHUNKS",4)
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        WORKER_TIMEOUT               = int(current_app.config.get("WORKER_TIMEOUT",300))
        if executor == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm         = Constants.ClassificationAlgorithms.KNN_PREDICT
        s                 = Session()
        requestHeaders    = request.headers #Headers for the request
        model_id          = requestHeaders.get("Model-Id","model-0") #iris
        records_test_id   = requestHeaders.get("Records-Test-Id","matrix-0_data")
        extension         = requestHeaders.get("Extension","npy")
        records_test_path = "{}/{}.{}".format(SOURCE_PATH, records_test_id, extension)
        
        logger.debug("_"*50)
        logger.debug("Client starts to process {} at {}".format(records_test_id,arrivalTime))
        
        with open(records_test_path, "rb") as f:
            records_test = np.load(f)   
        logger.debug("OPEN RECORDS SUCCESSFULLY") 
        
        model_result  = STORAGE_CLIENT.put_ndarray(
            key       = records_test_id,
            ndarray   = records_test,
            tags      = {},
            bucket_id = BUCKET_ID
        ).result()
        logger.debug("MODEL_RESULT PUT SUCCESSFULLY")

        managerResponse:RoryManager = current_app.config.get("manager") # Communicates with the manager
        get_worker_start_time = time.time()
        mr                    = managerResponse.getWorker( #Gets the worker from the manager
            headers = {
                "Algorithm"             : algorithm,
                "Start-Request-Time"    : str(arrivalTime),
                "Start-Get-Worker-Time" : str(get_worker_start_time) 
            }
        )
        logger.debug("GET WORKER SUCCESSFULLY")
        get_worker_end_time     = time.time() 
        get_worker_service_time = get_worker_end_time - get_worker_start_time
        stringResponse          = mr.content.decode("utf-8") #Decode the manager's response
        jsonResponse            = json.loads(stringResponse) # Pass the response to json
        workerId                =  "localhost" if TESTING else jsonResponse["workerId"]

        get_worker_logger_metrics = LoggerMetrics( # Write times of worker communication in logger
            operation_type = "GET_WORKER",
            matrix_id      = records_test_id,
            algorithm      = algorithm,
            arrival_time   = get_worker_start_time, 
            end_time       = get_worker_end_time, 
            service_time   = get_worker_service_time,
            worker_id      = workerId
        )
        logger.info(str(get_worker_logger_metrics))

        worker         = RoryWorker( #Allows to establish the connection with the worker
            workerId   = workerId,
            port       = jsonResponse["workerPort"],
            session    = s,
            algorithm  = algorithm,
        )
        logger.debug("RORY WORKER SUCCESSFULLY")
        worker_arrival_time = time.time()

        workerResponse = worker.run(
            headers    = {
                "Records-Test-Id": records_test_id,
                "Model-Id": model_id
            },
            timeout = WORKER_TIMEOUT
        )
        logger.debug("RUN_WORKER_RESPONSE {}".format(workerResponse))
        
        worker_end_time     = time.time()
        worker_service_time = worker_end_time - worker_arrival_time 
        interaction_logger_metrics = LoggerMetrics(
            operation_type = "WORKER",
            matrix_id      = records_test_id,
            algorithm      = algorithm, 
            arrival_time   = worker_arrival_time, 
            end_time       = worker_end_time, 
            service_time   = worker_service_time,
            worker_id      = workerId
        )
        logger.info(str(interaction_logger_metrics))

        stringWorkerResponse = workerResponse.content.decode("utf-8") #Response from worker
        jsonWorkerResponse   = json.loads(stringWorkerResponse) #pass to json
        endTime              = time.time() # Get the time when it ends
        worker_service_time  = workerResponse.headers.get("Service-Time",0) # Extract the time at which it started]]
        response_time        = endTime - arrivalTime # Get the service time

        logger_metrics = LoggerMetrics(
            operation_type = algorithm, 
            matrix_id      = records_test_id, 
            algorithm      = algorithm, 
            arrival_time   = arrivalTime, 
            end_time       = endTime, 
            service_time   = response_time,
            worker_id      = workerId
        )
        logger.info(str(logger_metrics))

        return Response(
            response = json.dumps({
                "labelVector" : jsonWorkerResponse.get("labelVector",[]),
                "serviceTime" : worker_service_time,
                "responseTime": str(response_time),
                "algorithm"   : algorithm,
            }),
            status   = 200,
            headers  = {}
        )
    except Exception as e:
        logger.error("CLIENT_ERROR "+str(e))
        return Response(response = None, status = 500, headers={"Error-Message":str(e)})