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
# from 

clustering = Blueprint("clustering",__name__,url_prefix = "/clustering")


@clustering.route("/test",methods=["GET","POST"])
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

@clustering.route("/skmeans",methods = ["POST"])
def skmeans():
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
        if executor == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm               = Constants.ClusteringAlgorithms.SKMEANS
        s                       = Session()
        requestHeaders          = request.headers #Headers for the request
        plaintext_matrix_id     = requestHeaders.get("Plaintext-Matrix-Id","matrix-0")
        encrypted_matrix_id     = "encrypted-{}".format(plaintext_matrix_id) # The id of the encrypted matrix is built
        udm_id                  = "{}-UDM".format(plaintext_matrix_id) # The iudm id is built
        plainTextMatrixFilename = requestHeaders.get("Plaintext-Matrix-Filename","matrix-0")
        extension               = requestHeaders.get("Extension","csv")
        m                       = requestHeaders.get("M","3")
        k                       = requestHeaders.get("K")
        experiment_iteration    = requestHeaders.get("Experiment-Iteration","0")
        MAX_ITERATIONS          = int(requestHeaders.get("Max-Iterations",current_app.config.get("MAX_ITERATIONS",10)))
        WORKER_TIMEOUT          = int(current_app.config.get("WORKER_TIMEOUT",300))
        requestId               = "request-{}".format(plaintext_matrix_id)
        plaintext_matrix_path   = "{}/{}.{}".format(SOURCE_PATH, plainTextMatrixFilename, extension)
        logger.debug("_"*50)
        logger.debug("SKMEANS algorithm={}, m={}, k={}, plain_matrix_id={}, encrypted_matrix_id={}".format(algorithm,m,k,plaintext_matrix_id,encrypted_matrix_id))
        
        if extension == "csv":
            plaintext_matrix = pd.read_csv(
                plaintext_matrix_path, 
                header=None
            ).values
        elif extension == "npy":
            with open(plaintext_matrix_path, "rb") as f:
                plaintext_matrix = np.load(f)
        else:
            return Response(response = None, status = 500, headers={"Error-Message":"Extension invalida"})
        
        r           = plaintext_matrix.shape[0]
        a           = plaintext_matrix.shape[1]
        cores       = os.cpu_count()
        max_workers = num_chunks if max_workers > num_chunks else max_workers
        max_workers = cores if max_workers > cores else max_workers
        logger.debug("ENCRYPT_WORKERS {}".format(max_workers))        
        encryption_start_time = time.time()
        logger.debug("NUM_CHUNKS {}".format(num_chunks))
        logger.debug("SHAPE {}".format(plaintext_matrix.shape))

        encrypted_matrix_chunks = Segmentation.segment_and_encrypt_liu_with_executor( #Encrypt 
            executor         = executor,
            key              = encrypted_matrix_id,
            plaintext_matrix = plaintext_matrix,
            dataowner        = dataowner,
            n                = a*r*m,
            num_chunks       = num_chunks
        )
        logger.debug("SEGMENTATION AND ENCRYPT {}".format(plaintext_matrix_id))
        logger.debug("_"*35)
        chunks = encrypted_matrix_chunks.iter()

        logger.debug("{} {} {}".format(type(encrypted_matrix_id), encrypted_matrix_chunks,type(BUCKET_ID)))
        put_chunks_generator_results = STORAGE_CLIENT.put_chunks(
            key       = encrypted_matrix_id, 
            chunks    = encrypted_matrix_chunks, 
            bucket_id = BUCKET_ID,
            tags      = {}
        )
        
        for i,put_chunk_result in enumerate(put_chunks_generator_results):
            encryption_end_time    = time.time()
            encryption_time        = encryption_end_time - encryption_start_time
            encrypt_logger_metrics = LoggerMetrics( #Write times of encrypt in logger
                operation_type = "ENCRYPT_CHUNK",
                matrix_id      = plaintext_matrix_id,
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
        
        encryption_time        = encryption_end_time - encryption_start_time
        encrypt_logger_metrics = LoggerMetrics( #Write times of encrypt in logger
            operation_type = "ENCRYPT",
            matrix_id      = plaintext_matrix_id,
            algorithm      = algorithm,
            arrival_time   = encryption_start_time,
            end_time       = encryption_end_time, 
            service_time   = encryption_time,
            m_value        = m
        ) 
        logger.info(str(encrypt_logger_metrics))
        udm_start_time = time.time()
        udm            = dataowner.get_U(
            plaintext_matrix = plaintext_matrix,
            algorithm        = algorithm
        )
        
        udm_put_result:Result[PutResponse,Exception] = STORAGE_CLIENT.put_ndarray(
            key       = udm_id, 
            ndarray   = udm, 
            tags      = {},
            bucket_id = BUCKET_ID
        ).result()

        print("UDM_PUT_RESULT {}".format(udm_put_result))
        if udm_put_result.is_err:
            raise udm_put_result.unwrap_err()

        udm_end_time = time.time()
        udm_time     = udm_end_time - udm_start_time
       
        udm_logger_metrics = LoggerMetrics( #Write times of udm in logger
            operation_type = "UDM_GENERATION",
            matrix_id      = plaintext_matrix_id,
            algorithm      = algorithm,
            arrival_time   = udm_start_time,
            end_time       = udm_end_time,
            service_time   = udm_time,
            m_value        = m
        ) 
        logger.info(str(udm_logger_metrics))

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
            matrix_id      = plaintext_matrix_id,
            algorithm      = algorithm,
            arrival_time   = get_worker_start_time, 
            end_time       = get_worker_end_time, 
            service_time   = get_worker_service_time,
            m_value        = m,
            k_value        = k,
            worker_id      = workerId
        )
        logger.info(str(get_worker_logger_metrics))
        
        worker         = RoryWorker( #Allows to establish the connection with the worker
            workerId   = workerId,
            port       = jsonResponse["workerPort"],
            session    = s,
            algorithm  = algorithm,
        )

        status         = Constants.ClusteringStatus.START #Set the status to start
        workerResponse = None 
        interaction_arrival_time = time.time()
        iterations   = 0
        # init_headers = {
        #     "Plaintext-Matrix-Id": plaintext_matrix_id,
        #     "K":str(k),
        #     "M":str(m), 
        #     "Experiment-Iteration": str(experiment_iteration), 
        #     "Max-Iterations":str(MAX_ITERATIONS) 
        # }
        logger.debug("BEFORE WHILE")

        while (status != Constants.ClusteringStatus.COMPLETED): #While the status is not completed
            
            #print("ENTRA AL WHILE")
            inner_interaction_arrival_time = time.time()
            # extra_headers = init_headers if iterations == 0 else {}
            run1_headers  = {
                "Step-Index"             : "1",
                "Clustering-Status"      : str(status),
                "Plaintext-Matrix-Id"    : plaintext_matrix_id,
                "Request-Id"             : requestId,
                "Encrypted-Matrix-Id"    : encrypted_matrix_id,
                "Encrypted-Matrix-Shape" : "({},{},{})".format(r,a,m),
                "Encrypted-Matrix-Dtype" : "float64",
                "Encrypted-Udm-Dtype"    : "float64",
                "Num-Chunks"             : str(num_chunks),
                "Iterations"             : str(iterations),
                "K"                      : str(k),
                "M"                      : str(m), 
                "Experiment-Iteration"   : str(experiment_iteration), 
                "Max-Iterations"         : str(MAX_ITERATIONS) 
            }
            logger.debug(str(run1_headers))
            #s.headers.update(run1_headers)
            
            logger.debug("UPDATE RUN_1 HEADERS")
            workerResponse    = worker.run(
                timeout = WORKER_TIMEOUT, 
                headers = run1_headers
            ) #Run 1 starts

            logger.debug("RUN1_WORKER_RESPONSE {}".format(workerResponse))
            
            if workerResponse.status_code !=200:
                return Response("Worker error: {}".format(workerResponse.content),status=500)
            
            stringWorkerResponse              = workerResponse.content.decode("utf-8") #Response from worker
            jsonWorkerResponse                = json.loads(stringWorkerResponse) #pass to json
            encryptedShiftMatrixId            = workerResponse.headers.get("Encrypted-Shift-Matrix-Id") # Extract id from Shift matrix
            encryptedShiftMatrix_get_response = Segmentation.get_matrix_or_error(
                client    = STORAGE_CLIENT, 
                key       = encryptedShiftMatrixId,
                bucket_id = BUCKET_ID
            )
            
            logger.debug("ENCRYPTED_SHIFT_MATRIX GET SUCCESSFULLY")

            encryptedShiftMatrix           = encryptedShiftMatrix_get_response.value
            shiftMatrix_chipher_schema_res = liu.decryptMatrix( #Shift Matrix is decrypted
                ciphertext_matrix = encryptedShiftMatrix.tolist(),
                secret_key        = dataowner.sk,
                m                 = int(m)
            )
            shiftMatrix   = shiftMatrix_chipher_schema_res.matrix
            logger.debug("DECRYPT MATRIX WITH LIU SUCCESSFULLY")

            shiftMatrixId = "{}-ShiftMatrix".format(plaintext_matrix_id) # The id of the Shift matrix is formed
            x:Result[PutResponse,Exception] = STORAGE_CLIENT.put_ndarray(
                key       = shiftMatrixId,
                ndarray   = shiftMatrix,
                tags      = {},
                bucket_id = BUCKET_ID
            ).result() #Shift matrix is saved to the storage system
            
            logger.debug("SHIFT_MATRIX PUT SUCCESSFULLY")

            status       = Constants.ClusteringStatus.WORK_IN_PROGRESS #Status is updated
            run2_headers = {
                    "Step-Index"             : "2",
                    "Clustering-Status"      : str(status),
                    "Shift-Matrix-Id"        : shiftMatrixId,
                    "Plaintext-Matrix-Id": plaintext_matrix_id,
                    "Encrypted-Matrix-Id"    :encrypted_matrix_id,
                    "Encrypted-Matrix-Shape" : "({},{},{})".format(r,a,m),
                    "Encrypted-Matrix-Dtype" : "float64",
                    "Num-Chunks"             : str(num_chunks),
                    "Iterations"             :str(iterations),
                    "K":str(k),
                    "M":str(m), 
                    "Experiment-Iteration": str(experiment_iteration), 
                    "Max-Iterations":str(MAX_ITERATIONS) 
            }
            # s.headers.update(run2_headers)
            workerResponse      = worker.run(
                timeout = WORKER_TIMEOUT,
                headers = run2_headers
            ) #Start run 2

            service_time_worker = workerResponse.headers.get("Service-Time",0) 
            iterations+=1
            if (iterations >= MAX_ITERATIONS): #If the number of iterations is equal to the maximum
                status              = Constants.ClusteringStatus.COMPLETED #Change the status to complete
                startTime           = float(s.headers.get("Start-Time",0))
                service_time_worker = time.time() - startTime #The service time is calculated
            else: 
                status = int(workerResponse.headers.get("Clustering-Status",Constants.ClusteringStatus.WORK_IN_PROGRESS)) #Status is maintained
            endTime    = time.time() # Get the time when it ends
            inner_interaction_service_time   = endTime - inner_interaction_arrival_time
            inner_interaction_logger_metrics = LoggerMetrics(
                operation_type = "INNER_INTERACTION",
                matrix_id      = plaintext_matrix_id,
                algorithm      = algorithm,
                arrival_time   = inner_interaction_arrival_time,
                end_time       = endTime,
                service_time   = inner_interaction_service_time,
                m_value        = m,
                k_value        = k,
                worker_id      = workerId,
                n_iterations   = iterations
            )
            logger.info(str(inner_interaction_logger_metrics))
        
        interaction_end_time       = time.time()
        interaction_service_time   = interaction_end_time - interaction_arrival_time 
        interaction_logger_metrics = LoggerMetrics(
            operation_type = "INTERACTIONS",
            matrix_id      = plaintext_matrix_id,
            algorithm      = algorithm, 
            arrival_time   = interaction_arrival_time, 
            end_time       = interaction_end_time,
            service_time   = interaction_service_time,
            m_value        = m,
            k_value        = k,
            worker_id      = workerId,
            n_iterations   = iterations
        )
        logger.info(str(interaction_logger_metrics))

        response_time  = endTime - arrivalTime 
        logger_metrics = LoggerMetrics(
            operation_type = algorithm,
            matrix_id      = plaintext_matrix_id,
            algorithm      = algorithm,
            arrival_time   = arrivalTime, 
            end_time       = endTime, 
            service_time   = response_time,
            m_value        = m,
            k_value        = k,
            worker_id      = workerId,
            n_iterations   = iterations
        )
        logger.info(str(logger_metrics))

        return Response(
            response = json.dumps({
                "labelVector" : jsonWorkerResponse.get("labelVector",[]),
                "serviceTime" : service_time_worker,
                "responseTime": response_time,
                "algorithm"   : algorithm
            }),
            status   = 200,
            headers  = {}
        )
    except Exception as e:
        logger.error("CLIENT_ERROR "+str(e))
        return Response(response= str(e) , status= 500)
    
    
@clustering.route("/kmeans",methods = ["POST"])
def kmeans():
    try:
        arrivalTime             = time.time()
        logger                  = current_app.config["logger"]
        TESTING                 = current_app.config.get("TESTING",True)
        SOURCE_PATH             = current_app.config["SOURCE_PATH"]
        STORAGE_CLIENT:V4Client = current_app.config.get("STORAGE_CLIENT")
        BUCKET_ID:str           = current_app.config.get("BUCKET_ID","rory")
        WORKER_TIMEOUT               = int(current_app.config.get("WORKER_TIMEOUT",300))
        algorithm               = Constants.ClusteringAlgorithms.KMEANS
        s                       = Session()
        requestHeaders          = request.headers #Headers for the request
        plainTextMatrixId       = requestHeaders.get("Plaintext-Matrix-Id","matrix-0")
        plainTextMatrixFilename = requestHeaders.get("Plaintext-Matrix-Filename","matrix-0")

        extension               = requestHeaders.get("Extension","csv")
        k                       = requestHeaders.get("K","3")
        MAX_ITERATIONS          = int(requestHeaders.get("Max-Iterations",current_app.config.get("MAX_ITERATIONS",100)))
        requestId               = "request-{}".format(plainTextMatrixId)
        plaintextMatrix_path    = "{}/{}.{}".format(SOURCE_PATH, plainTextMatrixFilename, extension)
        logger.debug("KMEANS algorithm={}, k={}, plain_matrix_id={}".format(algorithm,k,plainTextMatrixId))

        if extension == "csv":
            plaintextMatrix = pd.read_csv(
                plaintextMatrix_path, 
                header=None
            ).values
        elif extension == "npy":
            with open(plaintextMatrix_path, "rb") as f:
                plaintextMatrix = np.load(f)
        else:
            return Response(response = None, status = 500, headers={"Error-Message":"Extension invalida"})
        

        #plaintextMatrix         = pd.read_csv(plaintextMatrix_path, header=None).values   
        # print(plaintextMatrix)
        X = STORAGE_CLIENT.put_ndarray(
            key     = plainTextMatrixId,
            ndarray = plaintextMatrix,
            tags      = {},
            bucket_id = BUCKET_ID
        ).result()
        
        logger.debug("PLAINTEXT_MATRIX PUT SUCCESSFULLY")
        
        get_worker_arrival_time     = time.time()
        managerResponse:RoryManager = current_app.config.get("manager") # Communicates with the manager
        mr          = managerResponse.getWorker( #Gets the worker from the manager
            headers = {
                "Algorithm"         : algorithm,
                "Start-Request-Time": str(arrivalTime)
            }
        )
        get_worker_end_time       = time.time()
        get_worker_service_time   = get_worker_end_time - get_worker_arrival_time 
        logger.debug("GET WORKER COMPLETED SUCCESSFULLY")

        stringResponse = mr.content.decode("utf-8") #Decode the manager's response
        jsonResponse   = json.loads(stringResponse) # Pass the response to json
        workerId       = "localhost" if TESTING else jsonResponse["workerId"]
        get_worker_logger_metrics = LoggerMetrics(
            operation_type = "GET_WORKER", 
            matrix_id      = plainTextMatrixId, 
            algorithm      = algorithm, 
            arrival_time   = get_worker_arrival_time, 
            end_time       = get_worker_end_time,
            service_time   = get_worker_service_time,
            worker_id      = workerId
            )
        logger.info(str(get_worker_logger_metrics) )

        worker         = RoryWorker( #Allows to establish the connection with the worker
            workerId   = workerId,
            port       = jsonResponse["workerPort"],
            session    = s,
            algorithm  = algorithm
        )   
        logger.debug("RORY WORKER COMPLETED SUCCESSFULLY")

        interaction_arrival_time = time.time()
        workerResponse           = worker.run(
            headers    = {
                "Plaintext-Matrix-Id": plainTextMatrixId,
                "K": str(k),
            },
            timeout = WORKER_TIMEOUT
        )
        logger.debug("WORKER_RUN COMPLETED SUCCESSFULLY")

        interaction_end_time       = time.time()
        interaction_service_time   = interaction_end_time - interaction_arrival_time 
        interaction_logger_metrics = LoggerMetrics(
            operation_type = "INTERACTIONS",
            matrix_id      = plainTextMatrixId,
            algorithm      = algorithm, 
            arrival_time   = interaction_arrival_time, 
            end_time       = interaction_end_time, 
            service_time   = interaction_service_time,
            worker_id      = workerId
        )
        logger.info(str(interaction_logger_metrics))

        stringWorkerResponse = workerResponse.content.decode("utf-8") #Response from worker
        jsonWorkerResponse   = json.loads(stringWorkerResponse) #pass to json
        endTime              = time.time() # Get the time when it ends
        worker_service_time  = workerResponse.headers.get("Service-Time",0) # Extract the time at which it started]]
        response_time        = endTime - arrivalTime # Get the service time
        iterations           = int(workerResponse.headers.get("Iterations",0)) # Extract the current number of iterations
        logger_metrics = LoggerMetrics(
            operation_type = algorithm, 
            matrix_id      = plainTextMatrixId, 
            algorithm      = algorithm, 
            arrival_time   = arrivalTime, 
            end_time       = endTime, 
            service_time   = response_time,
            worker_id      = workerId,
            n_iterations   = iterations
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
        logger.error(str(e))
        return Response(response = None, status= 500, headers = {"Error-Message":str(e)})


@clustering.route("/dbskmeans", methods = ["POST"])
def dbskmeans():
    try:
        arrivalTime             = time.time()
        logger                  = current_app.config["logger"]
        BUCKET_ID:str           = current_app.config.get("BUCKET_ID","rory")
        TESTING                 = current_app.config.get("TESTING",True)
        SOURCE_PATH             = current_app.config["SOURCE_PATH"]
        liu:Liu                 = current_app.config.get("liu")
        dataowner:DataOwner     = current_app.config.get("dataowner")
        STORAGE_CLIENT:V4Client = current_app.config.get("STORAGE_CLIENT")
        num_chunks              = current_app.config.get("NUM_CHUNKS",4)
        max_workers             = current_app.config.get("MAX_WORKERS",2)
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        if executor == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm               = Constants.ClusteringAlgorithms.DBSKMEANS
        s                       = Session()
        requestHeaders          = request.headers #Headers for the request
        plaintext_matrix_id     = requestHeaders.get("Plaintext-Matrix-Id","matrix-0")
        encrypted_matrix_id     = "encrypted-{}".format(plaintext_matrix_id) # The id of the encrypted matrix is built
        encrypted_udm_id        = "{}-encrypted-UDM".format(plaintext_matrix_id) # The iudm id is built
        plainTextMatrixFilename = requestHeaders.get("Plaintext-Matrix-Filename","matrix-0")
        extension               = requestHeaders.get("Extension","csv")
        m                       = requestHeaders.get("M","3")
        k                       = requestHeaders.get("K")
        experiment_iteration    = requestHeaders.get("Experiment-Iteration","0")
        MAX_ITERATIONS          = int(requestHeaders.get("Max-Iterations",current_app.config.get("MAX_ITERATIONS",10)))
        WORKER_TIMEOUT          = int(current_app.config.get("WORKER_TIMEOUT",300))
        requestId               = "request-{}".format(plaintext_matrix_id)
        plaintext_matrix_path   = "{}/{}.{}".format(SOURCE_PATH, plainTextMatrixFilename, extension)
        iterations   = 0
        logger.debug("DBSKMEANS algorithm={}, m={}, k={}, plain_matrix_id={}, encrypted_matrix_id={}, encrypted_udm_id={}".format(algorithm,m,k,plaintext_matrix_id,encrypted_matrix_id, encrypted_udm_id))

        if extension == "csv":
            plaintext_matrix = pd.read_csv(
                plaintext_matrix_path, 
                header=None
            ).values
        elif extension == "npy":
            with open(plaintext_matrix_path, "rb") as f:
                plaintext_matrix = np.load(f)
        else:
            return Response(response = None, status = 500, headers={"Error-Message":"Extension invalida"})
        
        r           = plaintext_matrix.shape[0]
        a           = plaintext_matrix.shape[1]
        cores       = os.cpu_count()
        max_workers = num_chunks if max_workers > num_chunks else max_workers
        max_workers = cores if max_workers > cores else max_workers
        logger.debug("ENCRYPT_WORKERS {}".format(max_workers))
        encryption_start_time   = time.time()
        logger.debug("NUM_CHUNKS {}".format(num_chunks))
        logger.debug("SHAPE {}".format(plaintext_matrix.shape))

        encrypt_arrival_time = time.time()
        encrypted_matrix_chunks = Segmentation.segment_and_encrypt_liu_with_executor( #Encrypt 
            key              = encrypted_matrix_id,
            plaintext_matrix = plaintext_matrix,
            dataowner        = dataowner,
            n                = a*r*m,
            num_chunks       = num_chunks,
            max_workers      = max_workers,
            executor         = executor
        )
        
        logger.debug("SEGMENTATION AND ENCRYPT {}".format(plaintext_matrix_id))
        logger.debug("_"*35)
        chunks = encrypted_matrix_chunks.iter()

        logger.debug("{} {} {}".format(type(encrypted_matrix_id), encrypted_matrix_chunks,type(BUCKET_ID)))
        put_chunks_generator_results = STORAGE_CLIENT.put_chunks(
            bucket_id = "{}-{}".format(BUCKET_ID,iterations),
            key       = encrypted_matrix_id, 
            chunks    = encrypted_matrix_chunks, 
            tags      = {}
        )
        
        for i,put_chunk_result in enumerate(put_chunks_generator_results):
            encryption_end_time = time.time()
            encryption_time     = encryption_end_time - encryption_start_time
            encrypt_logger_metrics = LoggerMetrics( #Write times of encrypt in logger
                operation_type = "ENCRYPT_CHUNK",
                matrix_id      = plaintext_matrix_id,
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
         
        encryption_time     = encryption_end_time - encryption_start_time
        encrypt_logger_metrics = LoggerMetrics( #Write times of encrypt in logger
            operation_type = "ENCRYPT",
            matrix_id      = plaintext_matrix_id,
            algorithm      = algorithm,
            arrival_time   = encryption_start_time,
            end_time       = encryption_end_time, 
            service_time   = encryption_time,
            m_value        = m
        ) 
     
        logger.info(str(encrypt_logger_metrics))
        udm_start_time = time.time()
        udm = dataowner.get_U(
            plaintext_matrix = plaintext_matrix,
            algorithm        = algorithm
        )

        encrypted_matrix_UDM_chunks = Segmentation.segment_and_encrypt_fdhope_with_executor( #Encrypt 
            key              = encrypted_udm_id,
            plaintext_matrix = udm,
            dataowner        = dataowner,
            n                = r*r*a*m,
            num_chunks       = num_chunks,
            max_workers      = max_workers,
            algorithm        = algorithm,
            threshold        = 0.0,
            executor         = executor
        )
        logger.debug("SEGMENT AND ENCRYPT WITH FDHOPE")

        put_chunks_generator_results = STORAGE_CLIENT.put_chunks(
            bucket_id = "{}-{}".format(BUCKET_ID,iterations),
            key       = encrypted_udm_id, 
            chunks    = encrypted_matrix_UDM_chunks, 
            tags      = {}
        )
        logger.debug("ENCRYPTED_UDM PUT CHUNKS")

        for i,put_chunk_result in enumerate(put_chunks_generator_results):
            udm_end_time = time.time()
            udm_time     = udm_end_time - udm_start_time
            encrypt_logger_metrics = LoggerMetrics( #Write times of encrypt in logger
                operation_type = "GENERATION_AND_ENCRYPT_UDM_CHUNK",
                matrix_id      = plaintext_matrix_id,
                algorithm      = algorithm,
                arrival_time   = udm_start_time,
                end_time       = udm_end_time, 
                service_time   = udm_time,
                m_value        = m
            ) 
            if put_chunk_result.is_err:
                logger.error("Something went wrong storage and encrypt the chunk.")
                return Response(
                    status   = 500,
                    response = "{}".format(str(put_chunk_result.unwrap_err()))
                )
            logger.info(str(encrypt_logger_metrics)+","+str(i))

        udm_logger_metrics = LoggerMetrics( #Write times of udm in logger
            operation_type = "UDM_GENERATION_COMPLETED",
            matrix_id      = plaintext_matrix_id,
            algorithm      = algorithm,
            arrival_time   = udm_start_time,
            end_time       = udm_end_time,
            service_time   = udm_time,
            m_value        = m
        ) 
        logger.info(str(udm_logger_metrics))
        managerResponse:RoryManager = current_app.config.get("manager") # Communicates with the manager
        
        get_worker_start_time = time.time()
        mr                    = managerResponse.getWorker( #Gets the worker from the manager
            headers = {
                "Algorithm"             : algorithm,
                "Start-Request-Time"    : str(arrivalTime),
                "Start-Get-Worker-Time" : str(get_worker_start_time) 
            }
        )
        logger.debug("GET WORKER")
        get_worker_end_time       = time.time() 
        get_worker_service_time   = get_worker_end_time - get_worker_start_time

        stringResponse = mr.content.decode("utf-8") #Decode the manager's response
        jsonResponse   = json.loads(stringResponse) # Pass the response to json
        workerId       = "localhost" if TESTING else jsonResponse["workerId"]

        get_worker_logger_metrics = LoggerMetrics(
            operation_type = "GET_WORKER",
            matrix_id      = plaintext_matrix_id,
            algorithm      = algorithm,
            arrival_time   = get_worker_start_time, 
            end_time       = get_worker_end_time, 
            service_time   = get_worker_service_time,
            m_value        = m,
            k_value        = k,
            worker_id      = workerId
        )
        logger.info(str(get_worker_logger_metrics))

        worker         = RoryWorker( #Allows to establish the connection with the worker
            workerId   = workerId,
            port       = jsonResponse["workerPort"],
            session    = s,
            algorithm  = algorithm
        )
        logger.debug("<<0>>")
        status           = Constants.ClusteringStatus.START #Set the status to start
        workerResponse2   = None 
        interaction_arrival_time = time.time()
        # init_headers = {
            # "Plaintext-Matrix-Id": plaintext_matrix_id,
        # }
        logger.debug("1. BEFORE WHILE")

        initial_udm_shape = (r,r,a)
        global_start_time = time.time()
        while (status   != Constants.ClusteringStatus.COMPLETED): #While the status is not completed
            inner_interaction_arrival_time = time.time()
            # extra_headers = init_headers if iterations == 0 else {}
            run1_headers  = {
                "Step-Index"             : "1",
                "K"                      : str(k),
                "Clustering-Status"      : str(status),
                "Plaintext-Matrix-Id"    : plaintext_matrix_id,
                "Request-Id"             : requestId,
                "Encrypted-Matrix-Id"    : encrypted_matrix_id,
                "Encrypted-Matrix-Shape" : "({},{},{})".format(r,a,m),
                "Encrypted-Matrix-Dtype" : "float64",
                "Encrypted-Udm-Dtype"    : "float64",
                "Encrypted-Udm-Shape"    : str(initial_udm_shape),
                "Num-Chunks"             : str(num_chunks),
                "Iterations"             : str(iterations),
                "K"                      : str(k),
                "M"                      : str(m), 
                "Experiment-Iteration"   : str(experiment_iteration), 
                "Max-Iterations"         : str(MAX_ITERATIONS) 
                # **extra_headers
            }
            # s.headers.update(run1_headers)
            logger.debug(str(run1_headers))
            logger.debug("2. UPDATE RUN_1 HEADERS")
            workerResponse1         = worker.run(timeout = WORKER_TIMEOUT,headers =run1_headers) #Run 1 starts

            logger.debug("<<2.1>>")
            logger.debug("RUN1_WORKER_RESPONSE {}".format(workerResponse1))

            if workerResponse1.status_code !=200:
                return Response("Worker error: {}".format(workerResponse1.content),status=500)
            stringWorkerResponse              = workerResponse1.content.decode("utf-8") #Response from worker
            jsonWorkerResponse                = json.loads(stringWorkerResponse) #pass to json
            logger.debug("JSON LOADS")

            encryptedShiftMatrixId            = workerResponse1.headers.get("Encrypted-Shift-Matrix-Id") # Extract id from Shift matrix
            encryptedShiftMatrix_get_response = Segmentation.get_matrix_or_error(
                bucket_id = "{}-{}".format(BUCKET_ID,iterations),
                client    = STORAGE_CLIENT, 
                key       = encryptedShiftMatrixId
            )
            logger.debug("ENCRYPTED_SHIFT_MATRIX GET SUCCESSFULLY")
            encryptedShiftMatrix   = encryptedShiftMatrix_get_response.value
            cipher_schema_res      = liu.decryptMatrix( #Shift Matrix is decrypted
                ciphertext_matrix  = encryptedShiftMatrix.tolist(),
                secret_key         = dataowner.sk,
                m                  = int(m)
            )
            logger.debug("DECRYPT SHIFT_MATRIX WITH LIU SUCCESSFULLY")

            shiftMatrixOpe_res     = Fdhope.encryptMatrix( #Re-encrypt shift matrix with the FDHOPE scheme
                plaintext_matrix   = cipher_schema_res.matrix, 
                messagespace       = dataowner.messageIntervals,
                cipherspace        = dataowner.cypherIntervals
            )
            shiftMatrixOpe   = shiftMatrixOpe_res.matrix
            logger.debug("ENCRYPT SHIFT_MATRIX WITH OPE SUCCESSFULLY")

            shiftMatrixId    = "{}-ShiftMatrix".format(plaintext_matrix_id) # The id of the Shift matrix is formed
            shiftMatrixOpeId = "{}-ShiftMatrixOpe".format(plaintext_matrix_id) # The id of the Shift matrix is formed
            
            y:Result[PutResponse,Exception]  = STORAGE_CLIENT.put_ndarray(
                key       = shiftMatrixOpeId,
                ndarray   = shiftMatrixOpe,
                tags      = {},
                bucket_id = "{}-{}".format(BUCKET_ID,iterations)
            ).result()#.unwrap() #Shift matrix is saved to the storage system
            logger.debug("SHIFT_MATRIX PUT SUCCESSFULLY")
            status       = Constants.ClusteringStatus.WORK_IN_PROGRESS #Status is updated
            run2_headers = {
                    "Step-Index"             : "2",
                    "Clustering-Status"      : str(status),
                    "Shift-Matrix-Id"        : shiftMatrixId,
                    "Shift-Matrix-Ope-Id"    : shiftMatrixOpeId,
                    "Plaintext-Matrix-Id"    : workerResponse1.headers.get("Plaintext-Matrix-Id",plaintext_matrix_id),
                    "Encrypted-Matrix-Id"    : workerResponse1.headers.get("Encrypted-Matrix-Id",encrypted_matrix_id),
                    "Encrypted-Matrix-Shape" : "({},{},{})".format(r,a,m),
                    "Encrypted-Matrix-Dtype" : "float64",
                    "Encrypted-Udm-Dtype"    : "float64",
                    "Encrypted-Udm-Shape"    : str(initial_udm_shape),
                    # "({},{},{})".format(r,r,a),
                    "Num-Chunks"             : str(num_chunks),
                    "Iterations"             : str(iterations),
                    "K":str(k),
                    "M":str(m), 
                    "Experiment-Iteration": str(experiment_iteration), 
                    "Max-Iterations":str(MAX_ITERATIONS) 
            }
            
            
            workerResponse2    = worker.run(
                timeout = WORKER_TIMEOUT,
                headers = run2_headers
            ) #Start run 2
            workerResponse2.raise_for_status()
            initial_udm_shape = workerResponse2.headers.get("Encrypted-Udm-Shape")
            service_time_worker = workerResponse2.headers.get("Service-Time",0) 
            iterations+=1
            if (iterations >= MAX_ITERATIONS): #If the number of iterations is equal to the maximum
                status              = Constants.ClusteringStatus.COMPLETED #Change the status to complete
                # start_time          = start_time
                startTime           = float(s.headers.get("Start-Time",0))
                service_time_worker = time.time() - startTime #The service time is calculated
            else: 
                status = int(workerResponse2.headers.get("Clustering-Status",Constants.ClusteringStatus.WORK_IN_PROGRESS)) #Status is maintained
            endTime                          = time.time() # Get the time when it ends
            inner_interaction_service_time   = endTime-inner_interaction_arrival_time
            inner_interaction_logger_metrics = LoggerMetrics(
                operation_type = "INNER_INTERACTION",
                matrix_id      = plaintext_matrix_id,
                algorithm      = algorithm,
                arrival_time   = inner_interaction_arrival_time,
                end_time       = endTime,
                service_time   = inner_interaction_service_time,
                m_value        = m,
                k_value        = k,
                worker_id      = workerId,
                n_iterations   = iterations
            )
            logger.info(str(inner_interaction_logger_metrics))

        interaction_end_time       = time.time()
        interaction_service_time   = interaction_end_time - interaction_arrival_time 
        interaction_logger_metrics = LoggerMetrics(
            operation_type = "INTERACTIONS",
            matrix_id      = plaintext_matrix_id,
            algorithm      = algorithm, 
            arrival_time   = interaction_arrival_time, 
            end_time       = interaction_end_time, 
            service_time   = interaction_service_time,
            m_value        = m,
            k_value        = k,
            worker_id      = workerId,
            n_iterations   = iterations
        )
        logger.info(str(interaction_logger_metrics))

        response_time  = endTime - arrivalTime 
        logger_metrics = LoggerMetrics(
            operation_type = algorithm,
            matrix_id      = plaintext_matrix_id,
            algorithm      = algorithm, 
            arrival_time   = arrivalTime, 
            end_time       = endTime, 
            service_time   = response_time,
            m_value        = m,
            k_value        = k,
            worker_id      = workerId,
            n_iterations   = iterations
        )
        logger.info(str(logger_metrics))

        return Response(
            response = json.dumps({
                "labelVector" : jsonWorkerResponse.get("labelVector",[]),
                "serviceTime" : service_time_worker,
                "responseTime": str(response_time),
                "algorithm"   : algorithm
            }),
            status   = 200,
            headers  = {}
        )
    except Exception as e:
        logger.error(str(e))
        return Response(response= str(e) , status= 500)
    

@clustering.route("/dbsnnc", methods = ["POST"])
def dbsnnc():
    try:
        arrivalTime                  = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        TESTING                      = current_app.config.get("TESTING",True)
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        dataowner:DataOwner          = current_app.config.get("dataowner")
        STORAGE_CLIENT:V4Client      = current_app.config.get("STORAGE_CLIENT")
        num_chunks                   = current_app.config.get("NUM_CHUNKS",4)
        max_workers                  = current_app.config.get("MAX_WORKERS",2)
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        if executor == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm               = Constants.ClusteringAlgorithms.DBSNNC
        s                       = Session()
        requestHeaders          = request.headers #Headers for the request
        plaintext_matrix_id     = requestHeaders.get("Plaintext-Matrix-Id","matrix-0")
        encrypted_matrix_id     = "encrypted-{}".format(plaintext_matrix_id) # The id of the encrypted matrix is built
        dm_id                   = "{}-DM".format(plaintext_matrix_id)
        encrypted_dm_id         = "{}-encrypted-DM".format(plaintext_matrix_id) # The iudm id is built
        plainTextMatrixFilename = requestHeaders.get("Plaintext-Matrix-Filename","matrix-0")
        extension               = requestHeaders.get("Extension","csv")
        m                       = int(requestHeaders.get("M","3"))
        threshold               = float(requestHeaders.get("Threshold","0.01"))
        requestId               = "request-{}".format(plaintext_matrix_id)
        plaintext_matrix_path   = "{}/{}.{}".format(SOURCE_PATH, plainTextMatrixFilename, extension)
        WORKER_TIMEOUT          = int(current_app.config.get("WORKER_TIMEOUT",300))
        logger.debug("DBSNNC algorithm={}, m={}, plain_matrix_id={}, encrypted_matrix_id={}, encrypted_dm_id={}".format(algorithm,m,plaintext_matrix_id,encrypted_matrix_id, encrypted_dm_id))

        if extension == "csv":
            plaintext_matrix = pd.read_csv(
                plaintext_matrix_path, 
                header=None
            ).values
        elif extension == "npy":
            with open(plaintext_matrix_path, "rb") as f:
                plaintext_matrix = np.load(f)
        else:
            return Response(response = None, status = 500, headers={"Error-Message":"Extension invalida"})
        
        r           = plaintext_matrix.shape[0]
        a           = plaintext_matrix.shape[1]

        
        cores       = os.cpu_count()
        max_workers = num_chunks if max_workers > num_chunks else max_workers
        max_workers = cores if max_workers > cores else max_workers
        logger.debug("ENCRYPT_WORKERS {}".format(max_workers))        
        encryption_start_time = time.time()
        logger.debug("NUM_CHUNKS {}".format(num_chunks))
        logger.debug("SHAPE {}".format(plaintext_matrix.shape))

        encrypt_arrival_time = time.time()
        encrypted_matrix_chunks = Segmentation.segment_and_encrypt_liu_with_executor( #Encrypt 
            executor         = executor,
            key              = encrypted_matrix_id,
            plaintext_matrix = plaintext_matrix,
            dataowner        = dataowner,
            n                = r*a*m,
            num_chunks       = num_chunks,
            max_workers      = max_workers,
        )
        
        logger.debug("SEGMENTATION AND ENCRYPT {}".format(plaintext_matrix_id))
        logger.debug("_"*35)
        chunks = encrypted_matrix_chunks.iter()

        logger.debug("{} {} {}".format(type(encrypted_matrix_id), encrypted_matrix_chunks,type(BUCKET_ID)))
        put_chunks_generator_results = STORAGE_CLIENT.put_chunks(
            key       = encrypted_matrix_id, 
            chunks    = encrypted_matrix_chunks, 
            bucket_id = BUCKET_ID,
            tags      = {}
        )
        
        for i,put_chunk_result in enumerate(put_chunks_generator_results):
            encryption_end_time    = time.time()
            encryption_time        = encryption_end_time - encryption_start_time
            encrypt_chunk_logger_metrics = LoggerMetrics( #Write times of encrypt in logger
                operation_type = "ENCRYPT_CHUNK",
                matrix_id      = plaintext_matrix_id,
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
            logger.info(
                str(encrypt_chunk_logger_metrics)+","+str(i)
            )
        
        encryption_total_time = encryption_end_time - encryption_start_time
        encrypt_logger_metrics = LoggerMetrics(
            operation_type = "ENCRYPT",
            matrix_id      = plaintext_matrix_id,
            algorithm      = algorithm,
            arrival_time   = encrypt_arrival_time, 
            end_time       = encryption_end_time, 
            service_time   = encryption_total_time,
            m_value        = m
        ) 
        logger.debug(str(encrypt_logger_metrics))

        dm_start_time = time.time()
        dm = dataowner.get_U(
            plaintext_matrix = plaintext_matrix,
            algorithm        = algorithm
        )

        encrypted_matrix_DM_chunks = Segmentation.segment_and_encrypt_fdhope_with_executor( #Encrypt 
            key              = encrypted_dm_id,
            plaintext_matrix = dm,
            dataowner        = dataowner,
            n                = r*r,
            num_chunks       = num_chunks,
            max_workers      = max_workers,
            algorithm        = algorithm,
            threshold        = threshold,
            executor         = executor
        )
        logger.debug("SEGMENT AND ENCRYPT WITH FDHOPE")

        put_chunks_generator_results = STORAGE_CLIENT.put_chunks(
            bucket_id = BUCKET_ID,
            key       = encrypted_dm_id, 
            chunks    = encrypted_matrix_DM_chunks, 
            tags      = {}
        )
        logger.debug("ENCRYPTED_DM PUT CHUNKS")

        for i,put_chunk_result in enumerate(put_chunks_generator_results):
            dm_end_time = time.time()
            dm_time     = dm_end_time - dm_start_time
            encrypt_logger_metrics = LoggerMetrics( #Write times of encrypt in logger
                operation_type = "GENERATION_AND_ENCRYPT_DM_CHUNK",
                matrix_id      = plaintext_matrix_id,
                algorithm      = algorithm,
                arrival_time   = dm_start_time,
                end_time       = dm_end_time, 
                service_time   = dm_time,
                m_value        = m
            ) 
            if put_chunk_result.is_err:
                logger.error("Something went wrong storage and encrypt the chunk.")
                return Response(
                    status   = 500,
                    response = "{}".format(str(put_chunk_result.unwrap_err()))
                )
            logger.info(str(encrypt_logger_metrics)+","+str(i))

        dm_logger_metrics = LoggerMetrics( #Write times of udm in logger
            operation_type = "UDM_GENERATION_COMPLETED",
            matrix_id      = plaintext_matrix_id,
            algorithm      = algorithm,
            arrival_time   = dm_start_time,
            end_time       = dm_end_time,
            service_time   = dm_time,
            m_value        = m
        ) 
        logger.info(str(dm_logger_metrics))
        logger.debug("GET_U SUCCESSFULLY")

        encrypted_threshold = dataowner.encrypted_threshold
        
        logger.debug("GET ENCRYPT_THRESHOLD")

        # dm_put_result:Result[PutResponse,Exception] = STORAGE_CLIENT.put_ndarray(
        #     key       = dm_id, 
        #     ndarray   = dm, 
        #     tags      = {},
        #     bucket_id = BUCKET_ID
        # ).result()

        # print("UDM_PUT_RESULT {}".format(dm_put_result))
        # if dm_put_result.is_err:
            # raise dm_put_result.unwrap_err()

        # dm_end_time = time.time()
        # dm_time     = dm_end_time - dm_start_time

        # udm_logger_metrics = LoggerMetrics(
        #     operation_type = "UDM_GENERATION",
        #     matrix_id      = plaintext_matrix_id,
        #     algorithm      = algorithm,
        #     arrival_time   = udm_start_time,
        #     end_time       = udm_end_time,
        #     service_time   = udm_time,
        #     m_value        = m
        # )         
        # logger.info(str(udm_logger_metrics))

        managerResponse:RoryManager = current_app.config.get("manager") # Communicates with the manager
        
        get_worker_start_time = time.time()
        mr = managerResponse.getWorker( #Gets the worker from the manager
            headers = {
                "Algorithm": algorithm,
                "Start-Request-Time": str(arrivalTime),
                "Start-Get-Worker-Time" : str(get_worker_start_time) 
            }
        )
        logger.debug("GET WORKER SUCCESSFULLY")
        get_worker_end_time     = time.time() 
        get_worker_service_time = get_worker_end_time - get_worker_start_time
        stringResponse          = mr.content.decode("utf-8") #Decode the manager's response
        jsonResponse            = json.loads(stringResponse) # Pass the response to json
        workerId                = "localhost" if TESTING else jsonResponse["workerId"]

        get_worker_logger_metrics = LoggerMetrics(
            operation_type = "GET_WORKER",
            matrix_id      = plaintext_matrix_id,
            algorithm      = algorithm,
            arrival_time   = get_worker_start_time, 
            end_time       = get_worker_end_time, 
            service_time   = get_worker_service_time,
            worker_id      = workerId,
            m              = m
        )
        logger.info(str(get_worker_logger_metrics))

        worker         = RoryWorker( #Allows to establish the connection with the worker
            workerId   = workerId,
            port       = jsonResponse["workerPort"],
            session    = s,
            algorithm  = algorithm
        )
        dm_shape = (r,r)
        interaction_arrival_time = time.time()
        logger.debug("RORY WORKER SUCCESSFULLY")

        run_headers = {
            "Plaintext-Matrix-Id"    : plaintext_matrix_id,
            "Request-Id"             : requestId,
            "Encrypted-Matrix-Id"    : encrypted_matrix_id,
            "Encrypted-Matrix-Shape" : "({},{},{})".format(r,a,m),
            "Encrypted-Matrix-Dtype" : "float64",
            "Encrypted-Dm-Dtype"     : "float64",
            "Encrypted-Dm-Shape"     : str(dm_shape),
            "Num-Chunks"             : str(num_chunks),
            "M"                      : str(m),
            "Encrypted-Threshold"    : str(encrypted_threshold),
            "Dm-Shape"               : str(dm_shape),
            "Dm-Dtype"               : "float64",
        }

        workerResponse  = worker.run(
            timeout = WORKER_TIMEOUT, 
            headers = run_headers
        )

        logger.debug("RUN1_WORKER_RESPONSE {}".format(workerResponse))
            
        if workerResponse.status_code !=200:
            return Response("Worker error: {}".format(workerResponse.content),status=500)
        
        stringWorkerResponse = workerResponse.content.decode("utf-8") #Response from worker
        jsonWorkerResponse   = json.loads(stringWorkerResponse) #pass to json
        endTime              = time.time() # Get the time when it ends
        worker_service_time  = workerResponse.headers.get("Service-Time",0) # Extract the time at which it started]]
        response_time        = endTime - arrivalTime # Get the service time
        logger_metrics       = LoggerMetrics(
            operation_type = algorithm,
            matrix_id      = plaintext_matrix_id,
            algorithm      = algorithm,
            arrival_time   = arrivalTime, 
            end_time       = endTime, 
            service_time   = response_time,
            m_value        = m,
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
        logger.error(str(e))
        return Response(response =None, status= 500, headers={"Error-Message":str(e)})
    

@clustering.route("/nnc", methods = ["POST"])
def nnc():
    try:
        arrivalTime                  = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        TESTING                      = current_app.config.get("TESTING",True)
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        dataowner:DataOwner          = current_app.config.get("dataowner")
        STORAGE_CLIENT:V4Client      = current_app.config.get("STORAGE_CLIENT")
        num_chunks                   = current_app.config.get("NUM_CHUNKS",4)
        max_workers                  = current_app.config.get("MAX_WORKERS",2)
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        if executor == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm               = Constants.ClusteringAlgorithms.NNC
        s                       = Session()
        requestHeaders          = request.headers #Headers for the request
        plaintext_matrix_id     = requestHeaders.get("Plaintext-Matrix-Id","matrix-0")
        dm_id                   = "{}-DM".format(plaintext_matrix_id)
        plainTextMatrixFilename = requestHeaders.get("Plaintext-Matrix-Filename","matrix-0")
        extension               = requestHeaders.get("Extension","csv")
        threshold               = float(requestHeaders.get("Threshold","0.5"))
        requestId               = "request-{}".format(plaintext_matrix_id)
        plaintext_matrix_path   = "{}/{}.{}".format(SOURCE_PATH, plainTextMatrixFilename, extension)
        WORKER_TIMEOUT          = int(current_app.config.get("WORKER_TIMEOUT",300))
        logger.debug("NNC algorithm={}, plain_matrix_id={}".format(algorithm,plaintext_matrix_id))

        if extension == "csv":
            plaintext_matrix = pd.read_csv(
                plaintext_matrix_path, 
                header=None
            ).values
        elif extension == "npy":
            with open(plaintext_matrix_path, "rb") as f:
                plaintext_matrix = np.load(f)
        else:
            return Response(response = None, status = 500, headers={"Error-Message":"Extension invalida"})
        
        r           = plaintext_matrix.shape[0]
        a           = plaintext_matrix.shape[1]

        X = STORAGE_CLIENT.put_ndarray(
            key     = plaintext_matrix_id,
            ndarray = plaintext_matrix,
            tags      = {},
            bucket_id = BUCKET_ID
        ).result()
        
        logger.debug("PLAINTEXT_MATRIX PUT SUCCESSFULLY")

        dm_start_time = time.time()
        dm            = dataowner.get_U(
            plaintext_matrix = plaintext_matrix,
            algorithm        = algorithm, 
            logger = logger
        )
        logger.debug("GET_U SUCCESSFULLY")

        dm_put_result:Result[PutResponse,Exception] = STORAGE_CLIENT.put_ndarray(
            key       = dm_id, 
            ndarray   = dm, 
            tags      = {},
            bucket_id = BUCKET_ID
        ).result()

        print("DM_PUT_RESULT {}".format(dm_put_result))
        if dm_put_result.is_err:
            raise dm_put_result.unwrap_err()

        dm_end_time = time.time()
        dm_time     = dm_end_time - dm_start_time

        udm_logger_metrics = LoggerMetrics(
            operation_type = "UDM_GENERATION",
            matrix_id      = plaintext_matrix_id,
            algorithm      = algorithm,
            arrival_time   = dm_start_time,
            end_time       = dm_end_time,
            service_time   = dm_time
        )         
        logger.info(str(udm_logger_metrics))

        managerResponse:RoryManager = current_app.config.get("manager") # Communicates with the manager
        
        get_worker_start_time = time.time()
        mr = managerResponse.getWorker( #Gets the worker from the manager
            headers = {
                "Algorithm": algorithm,
                "Start-Request-Time": str(arrivalTime),
                "Start-Get-Worker-Time" : str(get_worker_start_time) 
            }
        )
        logger.debug("GET WORKER SUCCESSFULLY")
        get_worker_end_time     = time.time() 
        get_worker_service_time = get_worker_end_time - get_worker_start_time
        stringResponse          = mr.content.decode("utf-8") #Decode the manager's response
        jsonResponse            = json.loads(stringResponse) # Pass the response to json
        workerId                = "localhost" if TESTING else jsonResponse["workerId"]

        get_worker_logger_metrics = LoggerMetrics(
            operation_type = "GET_WORKER",
            matrix_id      = plaintext_matrix_id,
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
            algorithm  = algorithm
        )
        dm_shape = (r,a)
        logger.debug("RORY WORKER SUCCESSFULLY")

        run_headers = {
            "Plaintext-Matrix-Id" : plaintext_matrix_id,
            "Request-Id"          : requestId,
            "Num-Chunks"          : str(num_chunks),
            "Threshold"           : str(threshold),
            "Dm-Shape"            : str(dm_shape),
            "Dm-Dtype"            : "float64",
        }

        workerResponse  = worker.run(
            timeout = WORKER_TIMEOUT, 
            headers = run_headers
        )

        logger.debug("RUN1_WORKER_RESPONSE {}".format(workerResponse))
            
        if workerResponse.status_code !=200:
            return Response("Worker error: {}".format(workerResponse.content),status=500)
        
        stringWorkerResponse = workerResponse.content.decode("utf-8") #Response from worker
        jsonWorkerResponse   = json.loads(stringWorkerResponse) #pass to json
        endTime              = time.time() # Get the time when it ends
        worker_service_time  = workerResponse.headers.get("Service-Time",0) # Extract the time at which it started]]
        response_time        = endTime - arrivalTime # Get the service time
        logger_metrics       = LoggerMetrics(
            operation_type = algorithm,
            matrix_id      = plaintext_matrix_id,
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
        logger.error(str(e))
        return Response(response =None, status= 500, headers={"Error-Message":str(e)})
 


@clustering.route("/metrics", methods = ["GET"])

def metrics():
    try:
        logger                     = current_app.config["metricslogger"]
        STORAGE_CLIENT:Client      = current_app.config.get("STORAGE_CLIENT")
        requestHeaders             = request.headers #Headers for the request
        plainTextMatrixId          = requestHeaders.get("Plaintext-Matrix-Id","matrix-0")
        k                          = requestHeaders.get("K","2")
        algorithm                  = requestHeaders.get("Algorithm","SKMEANS")
        plaintextMatrix_response = STORAGE_CLIENT.get_ndarray(key = plainTextMatrixId).unwrap()
        plainTextMatrix          = plaintextMatrix_response.value
        targetId                 = "{}_{}_k{}".format(plainTextMatrixId,algorithm,k)
        target_response      = STORAGE_CLIENT.get_ndarray(key = targetId).unwrap()
        target          = target_response.value
        target_reshape  = target.reshape(-1,1).flatten()
        result:MetricsResultInternal = internal_validation_indexes(
            plaintext_matrix = plainTextMatrix,
            target = target_reshape
        )
        logger.info("METRICS {} {} {}".format(#Show the final result in a logger
            plainTextMatrixId,
            k,
            result
        ))

        return Response(
            response = json.dumps({
                "plainTextMatrixId": plainTextMatrixId,
                "k": k,
                **result.toDict()
            }),
            status   = 200,
            headers  = {}
        )
    except Exception as e:
        logger.error(str(e))
        return Response(response = None, status = 500, headers={"Error-Message": str(e)})