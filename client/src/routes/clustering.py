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
# 
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

        outsourced = dataowner.outsourcedData(  # The data is sent to the dataowner to start the encryption
            plaintext_matrix = plaintextMatrix,
            algorithm        = algorithm
        )    
        encryptedMatrixId           = "encrypted-{}".format(plainTextMatrixId) # The id of the encrypted matrix is built
        UDMId                       = "{}-UDM".format(plainTextMatrixId) # The iudm id is built

        encrypted_matrix_put_payload = PutNDArrayPayload(key = encryptedMatrixId, ndarray = outsourced.encrypted_matrix)
        encrypted_matrix_put_response = STORAGE_CLIENT.put_ndarray(encrypted_matrix_put_payload,update=True).unwrap() # The encrypted matrix is placed in the storage system
        # print("UDM",outsourced.UDM)
        UDM_put_payload = PutNDArrayPayload(key = UDMId, ndarray = outsourced.UDM)
        UDM_put_response              = STORAGE_CLIENT.put_ndarray(UDM_put_payload,update=True) # The udm array is placed in the storage system
        managerResponse:SecureClusteringManager = current_app.config.get("manager") # Communicates with the manager
        get_worker_start_time = time.time()
        mr = managerResponse.getWorker( #Gets the worker from the manager
            headers = {
                "Algorithm": algorithm,
                "Start-Request-Time": str(arrivalTime),
                "Start-Get-Worker-Time":str(get_worker_start_time) # Si quieres lo quitas, pero mira...
            }
        )
        get_worker_end_time = time.time() - get_worker_start_time
        stringResponse      = mr.content.decode("utf-8") #Decode the manager's response
        jsonResponse        = json.loads(stringResponse) # Pass the response to json
        worker              = SecureClusteringWorker( #Allows to establish the connection with the worker
            workerId   = "localhost" if TESTING else jsonResponse["workerId"],
            port       = jsonResponse["workerPort"],
            session    = s,
            algorithm  = algorithm
        )
        status         = Constants.ClusteringStatus.START #Set the status to start
        workerResponse = None 
        # time.sleep(100)
        while (status != Constants.ClusteringStatus.COMPLETED): #While the status is not completed
            run1_headers = {
                "Step-Index":"1",
                "Clustering-Status": str(status),
                "Plaintext-Matrix-Id": plainTextMatrixId,
                "Request_Id": requestId,
                "Encrypted-Matrix-Id": encryptedMatrixId,
                **requestHeaders
            }
            s.headers.update(run1_headers)
            workerResponse = worker.run() #Run 1 starts
            run1_service_time = float(workerResponse.headers.get("Service-Time",0))
            s.headers.update(workerResponse.headers) # the current headers are updated with the ones that come from the worker
            stringWorkerResponse          = workerResponse.content.decode("utf-8") #Response from worker
            jsonWorkerResponse            = json.loads(stringWorkerResponse) #pass to json
            encryptedShiftMatrixId        = workerResponse.headers.get("Encrypted-Shift-Matrix-Id") # Extract id from Shift matrix
            # print("WORKER_HEADERS",workerResponse.headers)
            encryptedShiftMatrix_get_response = STORAGE_CLIENT.get_ndarray(key = encryptedShiftMatrixId,cache=True).unwrap()
            encryptedShiftMatrix          = encryptedShiftMatrix_get_response.value
            shiftMatrix                   = liu.decryptMatrix( #Shift Matrix is decrypted
                ciphertext_matrix = encryptedShiftMatrix.tolist(),
                secret_key        = dataowner.sk,
                m                 = int(m)
            )
            shiftMatrixId         = "{}-ShiftMatrix".format(plainTextMatrixId) # The id of the Shift matrix is formed
            shiftmatrix_put_payload = PutNDArrayPayload(key = shiftMatrixId, ndarray = np.array(shiftMatrix.matrix))
            shiftmatrix_put_response = STORAGE_CLIENT.put_ndarray(shiftmatrix_put_payload,update=True).unwrap() #Shift matrix is saved to the storage system
            status               = Constants.ClusteringStatus.WORK_IN_PROGRESS #Status is updated
            run2_headers         = {
                    "Step-Index":"2",
                    "Clustering-Status": str(status),
                    "Shift-Matrix-Id": shiftMatrixId,
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

            print("_"*10)
        response_time = endTime - arrivalTime 
        logger.info("SKMEANS {} {} {}".format(#Show the final result in a logger
            plainTextMatrixId,
            service_time_worker,
            response_time
        ))

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
        arrivalTime                = time.time()
        logger                     = current_app.config["logger"]
        TESTING                    = current_app.config.get("TESTING",True),
        SOURCE_PATH                = current_app.config["SOURCE_PATH"]
        STORAGE_CLIENT:Client      = current_app.config.get("STORAGE_CLIENT")
        algorithm                  = Constants.ClusteringAlgorithms.KMEANS
        s                          = Session()
        requestHeaders             = request.headers #Headers for the request
        plainTextMatrixId          = requestHeaders.get("Plaintext-Matrix-Id","matrix-0")
        extension                  = requestHeaders.get("Extension","csv")
        k                          = requestHeaders.get("K","3")
        MAX_ITERATIONS             = int(requestHeaders.get("Max-Iterations",current_app.config.get("MAX_ITERATIONS",100)))
        requestId                  = "request-{}".format(plainTextMatrixId)
        plaintextMatrix_path       = "{}/{}.{}".format(SOURCE_PATH, plainTextMatrixId, extension)
        plaintextMatrix            = pd.read_csv(plaintextMatrix_path, header=None).values
        
        put_payload = PutNDArrayPayload(key=plainTextMatrixId, ndarray=plaintextMatrix)
        plaintextMatrix_put_response = STORAGE_CLIENT.put_ndarray(payload=put_payload,update=True).unwrap()
        managerResponse:SecureClusteringManager = current_app.config.get("manager") # Communicates with the manager
        mr = managerResponse.getWorker( #Gets the worker from the manager
            headers = {
                "Algorithm": algorithm,
                "Start-Request-Time": str(arrivalTime)
            }
        )
        stringResponse = mr.content.decode("utf-8") #Decode the manager's response
        jsonResponse   = json.loads(stringResponse) # Pass the response to json
        worker         = SecureClusteringWorker( #Allows to establish the connection with the worker
            workerId   = "localhost" if TESTING else jsonResponse["workerId"],
            port       = jsonResponse["workerPort"],
            session    = s,
            algorithm  = algorithm
        )
        workerResponse = worker.run(
            headers={
                "Plaintext-Matrix-Id": plainTextMatrixId,
                "K": str(k),
            }
        )
        stringWorkerResponse = workerResponse.content.decode("utf-8") #Response from worker
        jsonWorkerResponse   = json.loads(stringWorkerResponse) #pass to json
        endTime              = time.time() # Get the time when it ends
        worker_service_time  = workerResponse.headers.get("Service-Time",0) # Extract the time at which it started]]
        response_time        = endTime - arrivalTime # Get the service time

        logger.info("KMEANS {} {} {}".format(#Show the final result in a logger
            plainTextMatrixId,
            worker_service_time,
            response_time
        ))
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
# 

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
        
        outsourced = dataowner.outsourcedData(  # The data is sent to the dataowner to start the encryption
            plaintext_matrix = plaintextMatrix,
            algorithm = algorithm
        )    
        encryptedMatrixId = "encrypted-{}".format(plainTextMatrixId) # The id of the encrypted matrix is built
        UDMId             = "{}-encrypted-UDM".format(plainTextMatrixId) # The iudm id is built
        encrypted_matrix_put_payload = PutNDArrayPayload(key=encryptedMatrixId,ndarray=outsourced.encrypted_matrix)
        _                 = STORAGE_CLIENT.put_ndarray(payload=encrypted_matrix_put_payload,update=True).unwrap() # The encrypted matrix is placed in the storage system
        udm_put_payload = PutNDArrayPayload(key=UDMId,ndarray=outsourced.UDM)
        _                 = STORAGE_CLIENT.put_ndarray(payload=udm_put_payload,update=True).unwrap() # The udm array is placed in the storage system
        managerResponse:SecureClusteringManager = current_app.config.get("manager") # Communicates with the manager
        get_worker_start_time = time.time()
        mr = managerResponse.getWorker( #Gets the worker from the manager
            headers = {
                "Algorithm": algorithm,
                "Start-Request-Time": str(arrivalTime)
            }
        )
        get_worker_end_time = time.time() - get_worker_start_time
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
        while (status   != Constants.ClusteringStatus.COMPLETED): #While the status is not completed
            run1_headers = {
                    "Step-Index":"1",
                    "Clustering-Status": str(status),
                    "Plaintext-Matrix-Id": plainTextMatrixId,
                    "Request_Id": requestId,
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
            shiftMatrixId          = "{}-ShiftMatrix".format(plainTextMatrixId) # The id of the Shift matrix is formed
            shiftMatrixOpeId       = "{}-ShiftMatrixOpe".format(plainTextMatrixId) # The id of the Shift matrix is formed
            shift_matrix_put_payload = PutNDArrayPayload(key=shiftMatrixId,ndarray=np.array(shiftMatrix.matrix))
            _                      = STORAGE_CLIENT.put_ndarray(payload=shift_matrix_put_payload,update=True).unwrap() #Shift matrix is saved to the storage system
            shift_matrix_ope_put_payload = PutNDArrayPayload(key = shiftMatrixOpeId, ndarray = np.array(shiftMatrixOpe.matrix))
            _                      = STORAGE_CLIENT.put_ndarray(payload=shift_matrix_ope_put_payload,update=True).unwrap() #Shift matrix is saved to the storage system
            status                 = Constants.ClusteringStatus.WORK_IN_PROGRESS #Status is updated
            run2_headers           = {
                    "Step-Index":"2",
                    "Clustering-Status": str(status),
                    "Shift-Matrix-Id": shiftMatrixId,
            }
            s.headers.update(run2_headers)
            workerResponse = worker.run() #Start run 2
            runw_service_time = float(workerResponse.headers.get("Service-Time",0))
            s.headers.update(workerResponse.headers) # The headers are updated
            service_time_worker = workerResponse.headers.get("Service-Time",0) 
            iterations = int(s.headers.get("Iterations",0)) # Extract the current number of iterations
            if (iterations >= MAX_ITERATIONS): #If the number of iterations is equal to the maximum
                status = Constants.ClusteringStatus.COMPLETED #Change the status to complete
                startTime           = float(s.headers.get("Start-Time",0))
                service_time_worker = time.time() - startTime #The service time is calculated
            else: 
                status = int(workerResponse.headers.get("Status",Constants.ClusteringStatus.WORK_IN_PROGRESS)) #Status is maintained
            endTime       = time.time() # Get the time when it ends
            service_time  = workerResponse.headers.get("Service-Time",0) 
            response_time = endTime - arrivalTime 

            logger.info("DBSKMEANS {} {} {}".format(#Show the final result in a logger
                plainTextMatrixId,
                service_time,
                response_time
            ))

            return Response(
                response = json.dumps({
                    "labelVector" : jsonWorkerResponse.get("labelVector",[]),
                    "serviceTime" : service_time,
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

        outsourced = dataowner.outsourcedData(  # The data is sent to the dataowner to start the encryption
            plaintext_matrix = plaintextMatrix,
            algorithm        = algorithm,
            threshold        = threshold
        )

        encryptedMatrixId = "encrypted-{}".format(plainTextMatrixId) # The id of the encrypted matrix is built
        UDMId             = "{}-encrypted-UDM".format(plainTextMatrixId) # The iudm id is built

        encrypted_matrix_put_payload = PutNDArrayPayload(key = encryptedMatrixId, ndarray = outsourced.encrypted_matrix)
        _ = STORAGE_CLIENT.put_ndarray(payload=encrypted_matrix_put_payload,update=True).unwrap() # The encrypted matrix is placed in the storage system
        udm_put_payload = PutNDArrayPayload(key = UDMId, ndarray = outsourced.UDM)
        _ = STORAGE_CLIENT.put_ndarray(payload=udm_put_payload,update=True).unwrap() # The udm array is placed in the storage system
        managerResponse:SecureClusteringManager = current_app.config.get("manager") # Communicates with the manager
        mr = managerResponse.getWorker( #Gets the worker from the manager
            headers = {
                "Algorithm": algorithm,
                "Start-Request-Time": str(arrivalTime)
            }
        )
        stringResponse = mr.content.decode("utf-8") #Decode the manager's response
        jsonResponse   = json.loads(stringResponse) # Pass the response to json
        worker         = SecureClusteringWorker( #Allows to establish the connection with the worker
            workerId   = "localhost" if TESTING else jsonResponse["workerId"],
            port       = jsonResponse["workerPort"],
            session    = s,
            algorithm  = algorithm
        )
        workerResponse = worker.run(
            headers={
                "Plaintext-Matrix-Id": plainTextMatrixId,
                "Encrypted-Matrix-Id": encryptedMatrixId,
                "Encrypted-Threshold": str(outsourced.encrypted_threshold),
            }
        )
        stringWorkerResponse = workerResponse.content.decode("utf-8") #Response from worker
        jsonWorkerResponse   = json.loads(stringWorkerResponse) #pass to json
        endTime              = time.time() # Get the time when it ends
        worker_service_time  = workerResponse.headers.get("Service-Time",0) # Extract the time at which it started]]
        response_time        = endTime - arrivalTime # Get the service time

        logger.info("DBSNNC {} {} {}".format(#Show the final result in a logger
            plainTextMatrixId,
            worker_service_time,
            response_time
        ))

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