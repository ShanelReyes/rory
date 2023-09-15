import os
import time, json
import numpy as np
import pandas as pd
from typing import List,Generator,Awaitable,Iterator
import numpy.typing as npt
from requests import Session
from flask import Blueprint,current_app,request,Response
from option import Some
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
from mictlanx.v4.client import Client  as V4Client
from mictlanx.v4.interfaces.responses import GetNDArrayResponse
from option import Result

from mictlanx.utils.segmentation import Chunks,Chunk
from mictlanx.v3.interfaces.payloads import PutNDArrayPayload
from concurrent.futures import ProcessPoolExecutor

clustering = Blueprint("clustering",__name__,url_prefix = "/clustering")
@clustering.route("/skmeans",methods = ["POST"])
def skmeans():
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
        requestId               = "request-{}".format(plaintext_matrix_id)
        plaintext_matrix_path   = "{}/{}.{}".format(SOURCE_PATH, plainTextMatrixFilename, extension)
        logger.debug("Client starts to process {} at {}".format(plaintext_matrix_id,arrivalTime))
        plaintext_matrix        = pd.read_csv(plaintext_matrix_path, header=None).values
        r                       = plaintext_matrix.shape[0]
        a                       = plaintext_matrix.shape[1]
        cores                   = os.cpu_count()
        max_workers             = num_chunks if max_workers > num_chunks else max_workers
        max_workers             = cores if max_workers > cores else max_workers
        logger.debug("ENCRYPT_WORKERS {}".format(max_workers))
        encryption_start_time   = time.time()
        encrypted_matrix_chunks = Segmentation.segment_and_encrypt( #Encrypt 
            key              = encrypted_matrix_id,
            plaintext_matrix = plaintext_matrix,
            dataowner        = dataowner,
            n                = a*r*m,
            num_chunks       = num_chunks,
            max_workers      = max_workers
        )
        logger.debug("SEGMENTATION")
        print("_"*35)

        put_chunks_generator_results = STORAGE_CLIENT.put_chunks(
            key       = encrypted_matrix_id, 
            chunks    = encrypted_matrix_chunks, 
            bucket_id = BUCKET_ID,
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
            logger.info(str(encrypt_logger_metrics)+","+str(i))
        
        encryption_end_time = time.time()
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
        # UDM 
        udm_start_time = time.time()
        udm,_ = dataowner.get_U(
            plaintext_matrix = plaintext_matrix,
            algorithm        = algorithm
        )
        udm_put_future = STORAGE_CLIENT.put_ndarray(
            key       = udm_id, 
            ndarray   = udm, 
            tags      = {},
            bucket_id = BUCKET_ID
        ).result()
        print(udm_put_future)
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
        get_worker_end_time     = time.time() 
        get_worker_service_time = get_worker_end_time - get_worker_start_time
        stringResponse          = mr.content.decode("utf-8") #Decode the manager's response
        jsonResponse            = json.loads(stringResponse) # Pass the response to json
        workerId                =  "localhost" if TESTING else jsonResponse["workerId"]
        print("WORKER_ID",workerId)

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
            algorithm  = algorithm
        )
        status                   = Constants.ClusteringStatus.START #Set the status to start
        workerResponse           = None 
        interaction_arrival_time = time.time()
        iterations =0 
        # udm_put_future.result()
        #print("START WHILE")
        init_headers = {"Plaintext-Matrix-Id": plaintext_matrix_id,"K":str(k),"M":str(m), "Experiment-Iteration": str(experiment_iteration), "Max-Iterations":str(MAX_ITERATIONS) }
        while (status != Constants.ClusteringStatus.COMPLETED): #While the status is not completed

            inner_interaction_arrival_time = time.time()
            # if iterations == 0
            extra_headers = init_headers if iterations == 0 else {}
            run1_headers = {
                "Step-Index"          : "1",
                "Clustering-Status"   : str(status),
                "Plaintext-Matrix-Id" : plaintext_matrix_id,
                "Request-Id"          : requestId,
                "Encrypted-Matrix-Id" : encrypted_matrix_id,
                "Encrypted-Matrix-Shape":"({},{},{})".format(r,a,m),
                # "Content-Type"        : "application/json",
                **extra_headers
            }
            
            s.headers.update(run1_headers)
            print(s.headers)
            print("Run 1 starts")
            workerResponse    = worker.run() #Run 1 starts


            workerResponse.raise_for_status()
            print("_"*50)
            # if not workerResponse.ok:
            #     raise Exception("Error in worker {} - {}".format(worker.workerId, workerResponse))

            run1_service_time = float(workerResponse.headers.get("Service-Time",0))
            # print("WORKER_HEADERS",workerResponse.headers)
            s.headers.update({
                "Clustering-Status":workerResponse.headers.get("Clustering-Status",str(status)),
                "Plaintext-Matrix-Id": workerResponse.headers.get("Plaintext-Matrix-Id",plaintext_matrix_id),
                "Request-Id":workerResponse.headers.get("Request-Id",requestId),
                "Encrypted-Matrix-Id": workerResponse.headers.get("Encrypted-Matrix-Id",encrypted_matrix_id),
                "Encrypted-Shift-Matrix-Id": workerResponse.headers.get("Encrypted-Shift-Matrix-Id","ENCRYPTED_SHIF_MATRIX"),
                "K":workerResponse.headers.get("K",str(k)),
                "M":workerResponse.headers.get("M",str(m)),
                "Experiment-Iteration":workerResponse.headers.get("Experiment-Iteration",str(experiment_iteration)),
                "Max-Iterations":workerResponse.headers.get("Max-Iterations",str(MAX_ITERATIONS))
            }) # the current headers are updated with the ones that come from the worker
            print(s.headers)


            stringWorkerResponse              = workerResponse.content.decode("utf-8") #Response from worker
            jsonWorkerResponse                = json.loads(stringWorkerResponse) #pass to json
            encryptedShiftMatrixId            = workerResponse.headers.get("Encrypted-Shift-Matrix-Id") # Extract id from Shift matrix
            print("GET ENCRYPTED SHIFT MATRIX")
            encryptedShiftMatrix_get_response = Segmentation.get_matrix_or_error(
                client = STORAGE_CLIENT, 
                key    = encryptedShiftMatrixId
            )
            encryptedShiftMatrix              = encryptedShiftMatrix_get_response.value
            shiftMatrix_chipher_schema_res    = liu.decryptMatrix( #Shift Matrix is decrypted
                ciphertext_matrix = encryptedShiftMatrix.tolist(),
                secret_key        = dataowner.sk,
                m                 = int(m)
            )
            shiftMatrix   = shiftMatrix_chipher_schema_res.matrix
            shiftMatrixId = "{}-ShiftMatrix".format(plaintext_matrix_id) # The id of the Shift matrix is formed
            #logger.debug("shiftmatrix:{}".format(shiftMatrix))
            print("PUT SHIFT MATRIX")
            x = STORAGE_CLIENT.put_ndarray(
                key     = shiftMatrixId,
                ndarray = shiftMatrix,
                tags={},
                bucket_id=BUCKET_ID
            ).result() #Shift matrix is saved to the storage system
            status       = Constants.ClusteringStatus.WORK_IN_PROGRESS #Status is updated
            run2_headers = {
                    "Step-Index"        : "2",
                    "Clustering-Status" : str(status),
                    "Shift-Matrix-Id"   : shiftMatrixId,
            }
            s.headers.update(run2_headers)
            #print("START RUN 2")
            workerResponse      = worker.run() #Start run 2
            workerResponse.raise_for_status()
            runw_service_time   = float(workerResponse.headers.get("Service-Time",0))
            print("WORKER_RESPONSE_HEADERS", workerResponse.headers)
            s.headers.update(workerResponse.headers) # The headers are updated
            service_time_worker = workerResponse.headers.get("Service-Time",0) 
            iterations          = int(s.headers.get("Iterations",0)) # Extract the current number of iterations
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
        return Response(response = None, status = 500, headers={"Error-Message":str(e)})
    
    
@clustering.route("/kmeans",methods = ["POST"])
def kmeans():
    try:
        arrivalTime           = time.time()
        logger                = current_app.config["logger"]
        TESTING               = current_app.config.get("TESTING",True)
        SOURCE_PATH           = current_app.config["SOURCE_PATH"]
        STORAGE_CLIENT:Client = current_app.config.get("STORAGE_CLIENT")
        algorithm             = Constants.ClusteringAlgorithms.KMEANS
        s                     = Session()
        requestHeaders        = request.headers #Headers for the request
        plainTextMatrixId     = requestHeaders.get("Plaintext-Matrix-Id","matrix-0")
        extension             = requestHeaders.get("Extension","csv")
        k                     = requestHeaders.get("K","3")
        MAX_ITERATIONS        = int(requestHeaders.get("Max-Iterations",current_app.config.get("MAX_ITERATIONS",100)))
        requestId             = "request-{}".format(plainTextMatrixId)
        plaintextMatrix_path  = "{}/{}.{}".format(SOURCE_PATH, plainTextMatrixId, extension)
        plaintextMatrix       = pd.read_csv(plaintextMatrix_path, header=None).values   
        
        _ = STORAGE_CLIENT.put_ndarray(
            key     = plainTextMatrixId,
            ndarray = plaintextMatrix,
            update  = True
        ).unwrap()
        
        get_worker_arrival_time = time.time()
        managerResponse:RoryManager = current_app.config.get("manager") # Communicates with the manager
        mr          = managerResponse.getWorker( #Gets the worker from the manager
            headers = {
                "Algorithm"         : algorithm,
                "Start-Request-Time": str(arrivalTime)
            }
        )
        get_worker_end_time       = time.time()
        get_worker_service_time   = get_worker_end_time - get_worker_arrival_time 

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

        interaction_arrival_time = time.time()
        workerResponse           = worker.run(
            headers    = {
                "Plaintext-Matrix-Id": plainTextMatrixId,
                "K": str(k),
            }
        )
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
        arrivalTime           = time.time()
        logger                = current_app.config["logger"]
        TESTING               = current_app.config.get("TESTING",True)
        SINK_PATH             = current_app.config["SINK_PATH"]
        SOURCE_PATH           = current_app.config["SOURCE_PATH"]
        liu:Liu               = current_app.config.get("liu")
        dataowner:DataOwner   = current_app.config.get("dataowner")
        STORAGE_CLIENT:Client = current_app.config.get("STORAGE_CLIENT")
        algorithm             = Constants.ClusteringAlgorithms.DBSKMEANS
        s                     = Session()
        requestHeaders        = request.headers #Headers for the request
        plainTextMatrixId     = requestHeaders.get("Plaintext-Matrix-Id","matrix-0")
        extension             = requestHeaders.get("Extension","csv")
        m                     = requestHeaders.get("M","3")
        k                     = requestHeaders.get("K")
        MAX_ITERATIONS        = int(requestHeaders.get("Max-Iterations",current_app.config.get("MAX_ITERATIONS",100)))
        requestId             = "request-{}".format(plainTextMatrixId)
        plaintextMatrix_path  = "{}/{}.{}".format(SOURCE_PATH, plainTextMatrixId, extension)
        plaintextMatrix       = pd.read_csv(plaintextMatrix_path, header=None).values
        
        encrypt_arrival_time = time.time()
        outsourced           = dataowner.outsourcedData(  # The data is sent to the dataowner to start the encryption
            plaintext_matrix = plaintextMatrix,
            algorithm        = algorithm
        )  
        encrypt_end_time       = time.time() 
        encrypt_service_time   = encrypt_end_time - encrypt_arrival_time
        encrypt_logger_metrics = LoggerMetrics(
            operation_type = "ENCRYPT",
            matrix_id      = plainTextMatrixId,
            algorithm      = algorithm,
            arrival_time   = encrypt_arrival_time, 
            end_time       = encrypt_end_time, 
            service_time   = outsourced.encrypted_matrix_time,
            m_value        = m
        ) 
        udm_logger_metrics = LoggerMetrics(
            operation_type = "UDM_GENERATION",
            matrix_id      = plainTextMatrixId,
            algorithm      = algorithm,
            arrival_time   = encrypt_arrival_time, 
            end_time       = encrypt_end_time, 
            service_time   = outsourced.udm_time,
            m_value        = m
        ) 
        logger.info(str(encrypt_logger_metrics))
        logger.info(str(udm_logger_metrics))

        encryptedMatrixId = "encrypted-{}".format(plainTextMatrixId) # The id of the encrypted matrix is built
        UDMId             = "{}-encrypted-UDM".format(plainTextMatrixId) # The iudm id is built
        
        _ = STORAGE_CLIENT.put_ndarray(
            key     = encryptedMatrixId,
            ndarray = outsourced.encrypted_matrix,
            update  = True
        ).unwrap() # The encrypted matrix is placed in the storage system

        _ = STORAGE_CLIENT.put_ndarray(
            key     = UDMId,
            ndarray = outsourced.UDM,
            update  = True
        ).unwrap() # The udm array is placed in the storage system

        managerResponse:RoryManager = current_app.config.get("manager") # Communicates with the manager
        
        get_worker_start_time = time.time()
        mr                    = managerResponse.getWorker( #Gets the worker from the manager
            headers = {
                "Algorithm"             : algorithm,
                "Start-Request-Time"    : str(arrivalTime),
                "Start-Get-Worker-Time" : str(get_worker_start_time) 
            }
        )
        get_worker_end_time       = time.time() 
        get_worker_service_time   = get_worker_end_time - get_worker_start_time

        stringResponse = mr.content.decode("utf-8") #Decode the manager's response
        jsonResponse   = json.loads(stringResponse) # Pass the response to json
        workerId       = "localhost" if TESTING else jsonResponse["workerId"]

        get_worker_logger_metrics = LoggerMetrics(
            operation_type = "GET_WORKER",
            matrix_id      = plainTextMatrixId,
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
        status           = Constants.ClusteringStatus.START #Set the status to start
        workerResponse   = None 
        interaction_arrival_time = time.time()
        while (status   != Constants.ClusteringStatus.COMPLETED): #While the status is not completed
            inner_interaction_arrival_time = time.time()
            run1_headers = {
                    "Step-Index"          : "1",
                    "Clustering-Status"   : str(status),
                    "Plaintext-Matrix-Id" : plainTextMatrixId,
                    "Request_Id"          : requestId,
                    "Encrypted-Matrix-Id" : encryptedMatrixId,
                    "Content-Type"        : "application/json",
                    **requestHeaders
            }
            s.headers.update(run1_headers)
            workerResponse         = worker.run() #Run 1 starts
            run1_service_time      = float(workerResponse.headers.get("Service-Time",0))
            s.headers.update(workerResponse.headers) # the current headers are updated with the ones that come from the worker
            stringWorkerResponse          = workerResponse.content.decode("utf-8") #Response from worker
            jsonWorkerResponse            = json.loads(stringWorkerResponse) #pass to json
            encryptedShiftMatrixId        = workerResponse.headers.get("Encrypted-Shift-Matrix-Id") # Extract id from Shift matrix
            encryptedShiftMatrix_response = STORAGE_CLIENT.get_ndarray(
                key = encryptedShiftMatrixId
            ).unwrap()
            encryptedShiftMatrix   = encryptedShiftMatrix_response.value
            cipher_schema_res      = liu.decryptMatrix( #Shift Matrix is decrypted
                ciphertext_matrix  = encryptedShiftMatrix.tolist(),
                secret_key         = dataowner.sk,
                m                  = int(m)
            )
            shiftMatrixOpe_res     = Fdhope.encryptMatrix( #Re-encrypt shift matrix with the FDHOPE scheme
                plaintext_matrix   = cipher_schema_res.matrix, 
                messagespace       = outsourced.messageIntervals,
                cipherspace        = outsourced.cypherIntervals
            )
            shiftMatrixOpe   = shiftMatrixOpe_res.matrix
            shiftMatrixId    = "{}-ShiftMatrix".format(plainTextMatrixId) # The id of the Shift matrix is formed
            shiftMatrixOpeId = "{}-ShiftMatrixOpe".format(plainTextMatrixId) # The id of the Shift matrix is formed
            
            _ = STORAGE_CLIENT.put_ndarray(
                key     = shiftMatrixId,
                ndarray = cipher_schema_res.matrix,
                update  = True
            ).unwrap() #Shift matrix is saved to the storage system
            
            _  = STORAGE_CLIENT.put_ndarray(
                key     = shiftMatrixOpeId,
                ndarray = shiftMatrixOpe,
                update  = True
            ).unwrap() #Shift matrix is saved to the storage system

            status       = Constants.ClusteringStatus.WORK_IN_PROGRESS #Status is updated
            run2_headers = {
                    "Step-Index"          : "2",
                    "Clustering-Status"   : str(status),
                    "Shift-Matrix-Id"     : shiftMatrixId,
                    "Shift-Matrix-Ope-Id" : shiftMatrixOpeId
            }
            s.headers.update(run2_headers)
            workerResponse    = worker.run() #Start run 2
            runw_service_time = float(workerResponse.headers.get("Service-Time",0))
            s.headers.update(workerResponse.headers) # The headers are updated
            service_time_worker = workerResponse.headers.get("Service-Time",0) 

            iterations          = int(s.headers.get("Iterations",0)) # Extract the current number of iterations
            if (iterations >= MAX_ITERATIONS): #If the number of iterations is equal to the maximum
                status              = Constants.ClusteringStatus.COMPLETED #Change the status to complete
                startTime           = float(s.headers.get("Start-Time",0))
                service_time_worker = time.time() - startTime #The service time is calculated
            else: 
                status = int(workerResponse.headers.get("Clustering-Status",Constants.ClusteringStatus.WORK_IN_PROGRESS)) #Status is maintained
            endTime                          = time.time() # Get the time when it ends
            inner_interaction_service_time   = endTime-inner_interaction_arrival_time
            inner_interaction_logger_metrics = LoggerMetrics(
                operation_type = "INNER_INTERACTION",
                matrix_id      = plainTextMatrixId,
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
            matrix_id      = plainTextMatrixId,
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
            matrix_id      = plainTextMatrixId,
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
        return Response(response = None, status= 500, headers={"Error-Message":str(e)})
    

@clustering.route("/dbsnnc", methods = ["POST"])
def dbsnnc():
    try:
        arrivalTime           = time.time()
        logger                = current_app.config["logger"]
        TESTING               = current_app.config.get("TESTING",True)
        SOURCE_PATH           = current_app.config["SOURCE_PATH"]
        dataowner:DataOwner   = current_app.config.get("dataowner")
        STORAGE_CLIENT:Client = current_app.config.get("STORAGE_CLIENT")
        algorithm             = Constants.ClusteringAlgorithms.DBSNNC
        s                     = Session()
        requestHeaders        = request.headers #Headers for the request
        plainTextMatrixId     = requestHeaders.get("Plaintext-Matrix-Id","matrix-0")
        extension             = requestHeaders.get("Extension","csv")
        m                     = requestHeaders.get("M","3")
        threshold             = float(requestHeaders.get("Threshold","0.5"))
        requestId             = "request-{}".format(plainTextMatrixId)
        plaintextMatrix_path  = "{}/{}.{}".format(SOURCE_PATH, plainTextMatrixId, extension)
        plaintextMatrix       = pd.read_csv(plaintextMatrix_path, header=None).values

        encrypt_arrival_time = time.time()
        outsourced           = dataowner.outsourcedData(  # The data is sent to the dataowner to start the encryption
            plaintext_matrix = plaintextMatrix,
            algorithm        = algorithm,
            threshold        = threshold
        )
        encrypt_end_time       = time.time() 
        encrypt_service_time   = encrypt_end_time - encrypt_arrival_time
        encrypt_logger_metrics = LoggerMetrics(
            operation_type = "ENCRYPT",
            matrix_id      = plainTextMatrixId,
            algorithm      = algorithm,
            arrival_time   = encrypt_arrival_time, 
            end_time       = encrypt_end_time, 
            service_time   = outsourced.encrypted_matrix_time
        ) 
        udm_logger_metrics = LoggerMetrics(
            operation_type = "UDM_GENERATION",
            matrix_id      = plainTextMatrixId,
            algorithm      = algorithm,
            arrival_time   = encrypt_arrival_time, 
            end_time       = encrypt_end_time, 
            service_time   = outsourced.udm_time
        ) 
        
        logger.info(str(encrypt_logger_metrics))
        logger.info(str(udm_logger_metrics))

        encryptedMatrixId = "encrypted-{}".format(plainTextMatrixId) # The id of the encrypted matrix is built
        UDMId             = "{}-encrypted-UDM".format(plainTextMatrixId) # The iudm id is built

        _ = STORAGE_CLIENT.put_ndarray(
            key     = encryptedMatrixId,
            ndarray = outsourced.encrypted_matrix,
            update  = True
        ).unwrap() # The encrypted matrix is placed in the storage system
        
        _ = STORAGE_CLIENT.put_ndarray(
            key     = UDMId,
            ndarray = outsourced.UDM,
            update  = True
        ).unwrap() # The udm array is placed in the storage system
        managerResponse:RoryManager = current_app.config.get("manager") # Communicates with the manager
        
        get_worker_start_time = time.time()
        mr = managerResponse.getWorker( #Gets the worker from the manager
            headers = {
                "Algorithm": algorithm,
                "Start-Request-Time": str(arrivalTime)
            }
        )
        get_worker_end_time     = time.time() 
        get_worker_service_time = get_worker_end_time - get_worker_start_time
        stringResponse          = mr.content.decode("utf-8") #Decode the manager's response
        jsonResponse            = json.loads(stringResponse) # Pass the response to json
        workerId                = "localhost" if TESTING else jsonResponse["workerId"]

        get_worker_logger_metrics = LoggerMetrics(
            operation_type = "GET_WORKER",
            matrix_id      = plainTextMatrixId,
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

        interaction_arrival_time = time.time()
        workerResponse           = worker.run(
            headers = {
                "Plaintext-Matrix-Id": plainTextMatrixId,
                "Encrypted-Matrix-Id": encryptedMatrixId,
                "Encrypted-Threshold": str(outsourced.encrypted_threshold),
            }
        )
        interaction_end_time       = time.time()
        interaction_service_time   = interaction_end_time - interaction_arrival_time 
        interaction_logger_metrics = LoggerMetrics(
            operation_type = "INTERACTIONS",
            matrix_id      = plainTextMatrixId,
            algorithm      = algorithm, 
            arrival_time   = interaction_arrival_time, 
            end_time       = interaction_end_time, 
            service_time   = interaction_service_time,
            m_value        = m,
            worker_id      = workerId
        )
        logger.info(str(interaction_logger_metrics))

        stringWorkerResponse = workerResponse.content.decode("utf-8") #Response from worker
        jsonWorkerResponse   = json.loads(stringWorkerResponse) #pass to json
        endTime              = time.time() # Get the time when it ends
        worker_service_time  = workerResponse.headers.get("Service-Time",0) # Extract the time at which it started]]
        response_time        = endTime - arrivalTime # Get the service time
        logger_metrics       = LoggerMetrics(
            operation_type = algorithm,
            matrix_id      = plainTextMatrixId,
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