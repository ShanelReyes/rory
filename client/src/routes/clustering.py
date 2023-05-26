import time, json
import numpy as np
import pandas as pd
from requests import Session
from flask import Blueprint,current_app,request,Response
# 
from rory.core.interfaces.secureclusteringmanager import SecureClusteringManager
from rory.core.interfaces.secureclusteringworker import SecureClusteringWorker
from rory.core.interfaces.metricsResult_external import MetricsResultExternal
from rory.core.interfaces.metricsResult_internal import MetricsResultInternal
from rory.core.security.dataowner import DataOwner
from rory.core.security.cryptosystem.liu import Liu
from rory.core.security.cryptosystem.FDHOpe import Fdhope
from rory.core.utils.constants import Constants
from rory.core.validation_index.metrics import internal_validation_indexes,external_validation_indexes
from rory.core.interfaces.logger_metrics import LoggerMetrics
from mictlanx.v3.client import Client 
from mictlanx.v3.interfaces.payloads import PutNDArrayPayload

clustering = Blueprint("clustering",__name__,url_prefix = "/clustering")

@clustering.route("/skmeans",methods = ["POST"])
def skmeans():
    try:
        arrivalTime           = time.time()
        logger                = current_app.config["logger"]
        TESTING               = current_app.config.get("TESTING",True),
        SINK_PATH             = current_app.config["SINK_PATH"]
        SOURCE_PATH           = current_app.config["SOURCE_PATH"]
        liu:Liu               = current_app.config.get("liu")
        dataowner:DataOwner   = current_app.config.get("dataowner")
        STORAGE_CLIENT:Client = current_app.config.get("STORAGE_CLIENT")
        algorithm             = Constants.ClusteringAlgorithms.SKMEANS
        s                     = Session()
        requestHeaders        = request.headers #Headers for the request
        plainTextMatrixId     = requestHeaders.get("Plaintext-Matrix-Id","matrix-0")
        extension             = requestHeaders.get("Extension","csv")
        m                     = requestHeaders.get("M","3")
        MAX_ITERATIONS        = int(requestHeaders.get("Max-Iterations",current_app.config.get("MAX_ITERATIONS",10)))
        requestId             = "request-{}".format(plainTextMatrixId)
        plaintextMatrix_path  = "{}/{}.{}".format(SOURCE_PATH, plainTextMatrixId, extension)
        plaintextMatrix       = pd.read_csv(plaintextMatrix_path, header=None).values

        encrypt_arrival_time  = time.time()
        outsourced            = dataowner.outsourcedData(  # The data is sent to the dataowner to start the encryption
            plaintext_matrix  = plaintextMatrix,
            algorithm         = algorithm
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
        UDMId             = "{}-UDM".format(plainTextMatrixId) # The iudm id is built

        encrypted_matrix_put_payload  = PutNDArrayPayload(key = encryptedMatrixId, ndarray = outsourced.encrypted_matrix)
        encrypted_matrix_put_response = STORAGE_CLIENT.put_ndarray(encrypted_matrix_put_payload,update=True).unwrap() # The encrypted matrix is placed in the storage system
        UDM_put_payload               = PutNDArrayPayload(key = UDMId, ndarray = outsourced.UDM)
        UDM_put_response              = STORAGE_CLIENT.put_ndarray(UDM_put_payload,update=True) # The udm array is placed in the storage system
        managerResponse:SecureClusteringManager = current_app.config.get("manager") # Communicates with the manager
        
        get_worker_start_time = time.time()
        mr                    = managerResponse.getWorker( #Gets the worker from the manager
            headers = {
                "Algorithm"             : algorithm,
                "Start-Request-Time"    : str(arrivalTime),
                "Start-Get-Worker-Time" : str(get_worker_start_time) # Si quieres lo quitas, pero mira...
            }
        )
        get_worker_end_time       = time.time() 
        get_worker_service_time   = get_worker_end_time - get_worker_start_time
        get_worker_logger_metrics = LoggerMetrics(
            operation_type = "GET_WORKER",
            matrix_id      = plainTextMatrixId,
            algorithm      = algorithm,
            arrival_time   = get_worker_start_time, 
            end_time       = get_worker_end_time, 
            service_time   = get_worker_service_time,
            m_value        = m
        )
        logger.info(str(get_worker_logger_metrics))

        stringResponse      = mr.content.decode("utf-8") #Decode the manager's response
        jsonResponse        = json.loads(stringResponse) # Pass the response to json
        worker              = SecureClusteringWorker( #Allows to establish the connection with the worker
            workerId   = "localhost" if TESTING else jsonResponse["workerId"],
            port       = jsonResponse["workerPort"],
            session    = s,
            algorithm  = algorithm
        )
        status                   = Constants.ClusteringStatus.START #Set the status to start
        workerResponse           = None 
        interaction_arrival_time = time.time()
        while (status != Constants.ClusteringStatus.COMPLETED): #While the status is not completed
            inner_interaction_arrival_time = time.time()
            run1_headers = {
                "Step-Index"          : "1",
                "Clustering-Status"   : str(status),
                "Plaintext-Matrix-Id" : plainTextMatrixId,
                "Request_Id"          : requestId,
                "Encrypted-Matrix-Id" : encryptedMatrixId,
                **requestHeaders
            }
            s.headers.update(run1_headers)
            workerResponse    = worker.run() #Run 1 starts
            run1_service_time = float(workerResponse.headers.get("Service-Time",0))
            s.headers.update(workerResponse.headers) # the current headers are updated with the ones that come from the worker
            stringWorkerResponse              = workerResponse.content.decode("utf-8") #Response from worker
            jsonWorkerResponse                = json.loads(stringWorkerResponse) #pass to json
            encryptedShiftMatrixId            = workerResponse.headers.get("Encrypted-Shift-Matrix-Id") # Extract id from Shift matrix
            encryptedShiftMatrix_get_response = STORAGE_CLIENT.get_ndarray(key = encryptedShiftMatrixId,cache=True).unwrap()
            encryptedShiftMatrix              = encryptedShiftMatrix_get_response.value
            shiftMatrix                       = liu.decryptMatrix( #Shift Matrix is decrypted
                ciphertext_matrix = encryptedShiftMatrix.tolist(),
                secret_key        = dataowner.sk,
                m                 = int(m)
            )
            shiftMatrixId            = "{}-ShiftMatrix".format(plainTextMatrixId) # The id of the Shift matrix is formed
            shiftmatrix_put_payload  = PutNDArrayPayload(key = shiftMatrixId, ndarray = np.array(shiftMatrix.matrix))
            shiftmatrix_put_response = STORAGE_CLIENT.put_ndarray(shiftmatrix_put_payload,update=True).unwrap() #Shift matrix is saved to the storage system
            status                   = Constants.ClusteringStatus.WORK_IN_PROGRESS #Status is updated
            run2_headers             = {
                    "Step-Index"        : "2",
                    "Clustering-Status" : str(status),
                    "Shift-Matrix-Id"   : shiftMatrixId,
            }
            s.headers.update(run2_headers)
            workerResponse      = worker.run() #Start run 2
            runw_service_time   = float(workerResponse.headers.get("Service-Time",0))
            s.headers.update(workerResponse.headers) # The headers are updated
            service_time_worker = workerResponse.headers.get("Service-Time",0) 
            iterations          = int(s.headers.get("Iterations",0)) # Extract the current number of iterations
            if (iterations >= MAX_ITERATIONS): #If the number of iterations is equal to the maximum
                status              = Constants.ClusteringStatus.COMPLETED #Change the status to complete
                startTime           = float(s.headers.get("Start-Time",0))
                service_time_worker = time.time() - startTime #The service time is calculated
            else: 
                status = int(workerResponse.headers.get("Status",Constants.ClusteringStatus.WORK_IN_PROGRESS)) #Status is maintained
            endTime       = time.time() # Get the time when it ends
            inner_interaction_service_time = endTime-inner_interaction_arrival_time
            inner_interaction_logger_metrics = LoggerMetrics(operation_type = "INNER_INTERACTION",
                                                             matrix_id      = plainTextMatrixId,
                                                             algorithm      = algorithm,
                                                             arrival_time   = inner_interaction_arrival_time,
                                                             end_time       = endTime,
                                                             service_time   = inner_interaction_service_time,
                                                             m_value        = m)
            logger.info(str(inner_interaction_logger_metrics))
            print("_"*10)
        
        interaction_end_time       = time.time()
        interaction_service_time   = interaction_end_time - interaction_arrival_time 
        interaction_logger_metrics = LoggerMetrics(
            operation_type = "INTERACTIONS",
            matrix_id      = plainTextMatrixId,
            algorithm      = algorithm, 
            arrival_time   = interaction_arrival_time, 
            end_time       = interaction_end_time,
            service_time   = interaction_service_time,
            m_value        = m
        )
        logger.info(str(interaction_logger_metrics))

        response_time = endTime - arrivalTime 
        logger_metrics = LoggerMetrics(
            operation_type = algorithm,
            matrix_id      = plainTextMatrixId,
            algorithm      = algorithm,
            arrival_time   = arrivalTime, 
            end_time       = endTime, 
            service_time   = response_time,
            m_value        = m
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
        return Response(response = None, status = 500, headers={"Error-Message":str(e)})
    
    
@clustering.route("/kmeans",methods = ["POST"])
def kmeans():
    try:
        arrivalTime           = time.time()
        logger                = current_app.config["logger"]
        TESTING               = current_app.config.get("TESTING",True),
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
        #
        put_payload                  = PutNDArrayPayload(key=plainTextMatrixId, ndarray=plaintextMatrix)
        plaintextMatrix_put_response = STORAGE_CLIENT.put_ndarray(payload=put_payload,update=True).unwrap()
        
        get_worker_arrival_time = time.time()
        managerResponse:SecureClusteringManager = current_app.config.get("manager") # Communicates with the manager
        mr          = managerResponse.getWorker( #Gets the worker from the manager
            headers = {
                "Algorithm"         : algorithm,
                "Start-Request-Time": str(arrivalTime)
            }
        )
        get_worker_end_time       = time.time()
        get_worker_service_time   = get_worker_end_time - get_worker_arrival_time 
        get_worker_logger_metrics = LoggerMetrics(
            operation_type = "GET_WORKER", 
            matrix_id      = plainTextMatrixId, 
            algorithm      = algorithm, 
            arrival_time   = get_worker_arrival_time, 
            end_time       = get_worker_end_time,
            service_time   = get_worker_service_time
            )
        logger.info(str(get_worker_logger_metrics) )

        stringResponse = mr.content.decode("utf-8") #Decode the manager's response
        jsonResponse   = json.loads(stringResponse) # Pass the response to json
        worker         = SecureClusteringWorker( #Allows to establish the connection with the worker
            workerId   = "localhost" if TESTING else jsonResponse["workerId"],
            port       = jsonResponse["workerPort"],
            session    = s,
            algorithm  = algorithm
        )

        interaction_arrival_time = time.time()
        workerResponse = worker.run(
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
            service_time   = interaction_service_time
        )
        logger.info(str(interaction_logger_metrics))

        stringWorkerResponse = workerResponse.content.decode("utf-8") #Response from worker
        jsonWorkerResponse   = json.loads(stringWorkerResponse) #pass to json
        endTime              = time.time() # Get the time when it ends
        worker_service_time  = workerResponse.headers.get("Service-Time",0) # Extract the time at which it started]]
        response_time        = endTime - arrivalTime # Get the service time

        logger_metrics = LoggerMetrics(
            operation_type = algorithm, 
            matrix_id      = plainTextMatrixId, 
            algorithm      = algorithm, 
            arrival_time   = arrivalTime, 
            end_time       = endTime, 
            service_time   = response_time)
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
        return Response(response = None, status= 500, headers = {"Error-Message":str(e)})


@clustering.route("/dbskmeans", methods = ["POST"])
def dbskmeans():
    try:
        arrivalTime           = time.time()
        logger                = current_app.config["logger"]
        TESTING               = current_app.config.get("TESTING",True),
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
        MAX_ITERATIONS        = int(requestHeaders.get("Max-Iterations",current_app.config.get("MAX_ITERATIONS",100)))
        requestId             = "request-{}".format(plainTextMatrixId)
        plaintextMatrix_path  = "{}/{}.{}".format(SOURCE_PATH, plainTextMatrixId, extension)
        plaintextMatrix       = pd.read_csv(plaintextMatrix_path, header=None).values
        
        encrypt_arrival_time = time.time()
        outsourced           = dataowner.outsourcedData(  # The data is sent to the dataowner to start the encryption
            plaintext_matrix = plaintextMatrix,
            algorithm        = algorithm
        )  
        encrypt_end_time     = time.time() 
        encrypt_service_time = encrypt_end_time - encrypt_arrival_time
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
            m_value        = m) 
        
        logger.info(str(encrypt_logger_metrics))
        logger.info(str(udm_logger_metrics))


        encryptedMatrixId = "encrypted-{}".format(plainTextMatrixId) # The id of the encrypted matrix is built
        UDMId             = "{}-encrypted-UDM".format(plainTextMatrixId) # The iudm id is built
        encrypted_matrix_put_payload = PutNDArrayPayload(key=encryptedMatrixId,ndarray=outsourced.encrypted_matrix)
        _                 = STORAGE_CLIENT.put_ndarray(payload=encrypted_matrix_put_payload,update=True).unwrap() # The encrypted matrix is placed in the storage system
        udm_put_payload   = PutNDArrayPayload(key=UDMId,ndarray=outsourced.UDM)
        _                 = STORAGE_CLIENT.put_ndarray(payload=udm_put_payload,update=True).unwrap() # The udm array is placed in the storage system
        managerResponse:SecureClusteringManager = current_app.config.get("manager") # Communicates with the manager
        
        get_worker_start_time = time.time()
        mr          = managerResponse.getWorker( #Gets the worker from the manager
            headers = {
                "Algorithm"         : algorithm,
                "Start-Request-Time": str(arrivalTime)
            }
        )
        get_worker_end_time       = time.time() 
        get_worker_service_time   = get_worker_end_time - get_worker_start_time
        get_worker_logger_metrics = LoggerMetrics(
            operation_type = "GET_WORKER",
            matrix_id      = plainTextMatrixId,
            algorithm      = algorithm,
            arrival_time   = get_worker_start_time, 
            end_time       = get_worker_end_time, 
            service_time   = get_worker_service_time,
            m_value        = m)
        logger.info(str(get_worker_logger_metrics))

        stringResponse      = mr.content.decode("utf-8") #Decode the manager's response
        jsonResponse        = json.loads(stringResponse) # Pass the response to json
        worker              = SecureClusteringWorker( #Allows to establish the connection with the worker
            workerId  = "localhost" if TESTING else jsonResponse["workerId"],
            port      = jsonResponse["workerPort"],
            session   = s,
            algorithm = algorithm
        )
        status           = Constants.ClusteringStatus.START #Set the status to start
        workerResponse   = None 
        interaction_arrival_time = time.time()

        while (status   != Constants.ClusteringStatus.COMPLETED): #While the status is not completed
            inner_interaction_arrival_time = time.time()
            run1_headers = {
                    "Step-Index"         : "1",
                    "Clustering-Status"  : str(status),
                    "Plaintext-Matrix-Id": plainTextMatrixId,
                    "Request_Id"         : requestId,
                    "Encrypted-Matrix-Id": encryptedMatrixId,
                    **requestHeaders
            }
            s.headers.update(run1_headers)
            workerResponse         = worker.run() #Run 1 starts
            run1_service_time      = float(workerResponse.headers.get("Service-Time",0))
            s.headers.update(workerResponse.headers) # the current headers are updated with the ones that come from the worker
            stringWorkerResponse   = workerResponse.content.decode("utf-8") #Response from worker
            jsonWorkerResponse     = json.loads(stringWorkerResponse) #pass to json
            encryptedShiftMatrixId = workerResponse.headers.get("Encrypted-Shift-Matrix-Id") # Extract id from Shift matrix
            encryptedShiftMatrix_response = STORAGE_CLIENT.get_ndarray(key = encryptedShiftMatrixId).unwrap()
            encryptedShiftMatrix   = encryptedShiftMatrix_response.value
            shiftMatrix            = liu.decryptMatrix( #Shift Matrix is decrypted
                ciphertext_matrix  = encryptedShiftMatrix.tolist(),
                secret_key         = dataowner.sk,
                m                  = int(m)
            )
            shiftMatrixOpe         = Fdhope.encryptMatrix( #Re-encrypt shift matrix with the FDHOPE scheme
                plaintext_matrix   = shiftMatrix.matrix, 
                messagespace       = outsourced.messageIntervals,
                cipherspace        = outsourced.cypherIntervals
                    )
            shiftMatrixId                = "{}-ShiftMatrix".format(plainTextMatrixId) # The id of the Shift matrix is formed
            shiftMatrixOpeId             = "{}-ShiftMatrixOpe".format(plainTextMatrixId) # The id of the Shift matrix is formed
            shift_matrix_put_payload     = PutNDArrayPayload(key=shiftMatrixId,ndarray=np.array(shiftMatrix.matrix))
            _                            = STORAGE_CLIENT.put_ndarray(payload=shift_matrix_put_payload,update=True).unwrap() #Shift matrix is saved to the storage system
            shift_matrix_ope_put_payload = PutNDArrayPayload(key = shiftMatrixOpeId, ndarray = np.array(shiftMatrixOpe.matrix))
            _                            = STORAGE_CLIENT.put_ndarray(payload=shift_matrix_ope_put_payload,update=True).unwrap() #Shift matrix is saved to the storage system
            status                       = Constants.ClusteringStatus.WORK_IN_PROGRESS #Status is updated
            run2_headers                 = {
                    "Step-Index"       :"2",
                    "Clustering-Status": str(status),
                    "Shift-Matrix-Id"  : shiftMatrixId,
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
                status = int(workerResponse.headers.get("Status",Constants.ClusteringStatus.WORK_IN_PROGRESS)) #Status is maintained
            endTime                          = time.time() # Get the time when it ends
            inner_interaction_service_time   = endTime-inner_interaction_arrival_time
            inner_interaction_logger_metrics = LoggerMetrics(operation_type = "INNER_INTERACTION",
                                                             matrix_id      = plainTextMatrixId,
                                                             algorithm      = algorithm,
                                                             arrival_time   = inner_interaction_arrival_time,
                                                             end_time       = endTime,
                                                             service_time   = inner_interaction_service_time,
                                                             m_value        = m)
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
            m_value        = m)
        logger.info(str(interaction_logger_metrics))

        response_time = endTime - arrivalTime 
        logger_metrics = LoggerMetrics(
            operation_type = algorithm,
            matrix_id      = plainTextMatrixId,
            algorithm      = algorithm, 
            arrival_time   = arrivalTime, 
            end_time       = endTime, 
            service_time   = response_time,
            m_value        = m)
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
        return Response(response = None, status= 500, headers={"Error-Message":str(e)})
    

@clustering.route("/dbsnnc", methods = ["POST"])
def dbsnnc():
    try:
        arrivalTime           = time.time()
        logger                = current_app.config["logger"]
        TESTING               = current_app.config.get("TESTING",True),
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
        encrypt_end_time     = time.time() 
        encrypt_service_time = encrypt_end_time - encrypt_arrival_time
        encrypt_logger_metrics = LoggerMetrics(operation_type="ENCRYPT",matrix_id=plainTextMatrixId,algorithm=algorithm,arrival_time=encrypt_arrival_time, end_time= encrypt_end_time, service_time=outsourced.encrypted_matrix_time) 
        udm_logger_metrics   = LoggerMetrics(operation_type="UDM_GENERATION",matrix_id=plainTextMatrixId,algorithm=algorithm,arrival_time=encrypt_arrival_time, end_time= encrypt_end_time, service_time=outsourced.udm_time) 
        
        logger.info(str(encrypt_logger_metrics))
        logger.info(str(udm_logger_metrics))

        encryptedMatrixId = "encrypted-{}".format(plainTextMatrixId) # The id of the encrypted matrix is built
        UDMId             = "{}-encrypted-UDM".format(plainTextMatrixId) # The iudm id is built

        encrypted_matrix_put_payload = PutNDArrayPayload(key = encryptedMatrixId, ndarray = outsourced.encrypted_matrix)
        _               = STORAGE_CLIENT.put_ndarray(payload=encrypted_matrix_put_payload,update=True).unwrap() # The encrypted matrix is placed in the storage system
        udm_put_payload = PutNDArrayPayload(key = UDMId, ndarray = outsourced.UDM)
        _               = STORAGE_CLIENT.put_ndarray(payload=udm_put_payload,update=True).unwrap() # The udm array is placed in the storage system
        managerResponse:SecureClusteringManager = current_app.config.get("manager") # Communicates with the manager
        
        get_worker_start_time = time.time()
        mr          = managerResponse.getWorker( #Gets the worker from the manager
            headers = {
                "Algorithm": algorithm,
                "Start-Request-Time": str(arrivalTime)
            }
        )
        get_worker_end_time       = time.time() 
        get_worker_service_time   = get_worker_end_time - get_worker_start_time
        get_worker_logger_metrics = LoggerMetrics(operation_type="GET_WORKER",matrix_id=plainTextMatrixId,algorithm=algorithm,arrival_time= get_worker_start_time, end_time= get_worker_end_time, service_time=get_worker_service_time)
        logger.info(str(get_worker_logger_metrics))

        stringResponse = mr.content.decode("utf-8") #Decode the manager's response
        jsonResponse   = json.loads(stringResponse) # Pass the response to json
        worker         = SecureClusteringWorker( #Allows to establish the connection with the worker
            workerId   = "localhost" if TESTING else jsonResponse["workerId"],
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
            m_value        = m)
        logger.info(str(interaction_logger_metrics))

        stringWorkerResponse = workerResponse.content.decode("utf-8") #Response from worker
        jsonWorkerResponse   = json.loads(stringWorkerResponse) #pass to json
        endTime              = time.time() # Get the time when it ends
        worker_service_time  = workerResponse.headers.get("Service-Time",0) # Extract the time at which it started]]
        response_time        = endTime - arrivalTime # Get the service time

        response_time = endTime - arrivalTime 

        logger_metrics = LoggerMetrics(
            operation_type = algorithm,
            matrix_id      = plainTextMatrixId,
            algorithm      = algorithm,
            arrival_time   = arrivalTime, 
            end_time       = endTime, 
            service_time   = response_time,
            m_value        = m)
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
        return Response(response = None, status = 500, headers={"Error-Message": str(e)})