import time, json
import numpy as np
from flask import Blueprint,current_app,request,Response
from rory.core.clustering.kmeans import kmeans as kMeans
from rory.core.clustering.secure.local.dbsnnc import Dbsnnc
from rory.core.utils.Utils import Utils
from rory.core.utils.constants import Constants
from rory.core.clustering.secure.distributed.skmeans import SKMeans
from rory.core.clustering.secure.distributed.dbskmeans import DBSKMeans
from mictlanx.v3.client import Client
from mictlanx.v3.interfaces.payloads import PutNDArrayPayload
from rory.core.interfaces.logger_metrics import LoggerMetrics
# 
clustering = Blueprint("clustering",__name__,url_prefix = "/clustering")

"""
Description:
    First part of the skmeans process. 
    It stops where client interaction is required and writes the centroids and matrix S to disk.
"""
def skmeans_1(requestHeaders) -> Response:
    arrivalTime            = time.time() #Worker start time
    logger                 = current_app.config["logger"]
    workerId               = current_app.config["NODE_ID"] # Get the node_id from the global configuration
    STORAGE_CLIENT:Client  = current_app.config["STORAGE_CLIENT"]
    status                 = int(requestHeaders.get("Clustering-Status", Constants.ClusteringStatus.START)) 
    isStartStatus          = status == Constants.ClusteringStatus.START #if status is start save it to isStartStatus
    k                      = int(requestHeaders.get("K",3)) # It is passed to integer because the headers are strings
    m                      = int(requestHeaders.get("M",3))
    algorithm              = Constants.ClusteringAlgorithms.SKMEANS
    plainTextMatrixID      = requestHeaders.get("Plaintext-Matrix-Id")
    encryptedMatrixId      = requestHeaders.get("Encrypted-Matrix-Id","")
    UDMId                  = "{}-UDM".format(plainTextMatrixID) 
    encryptedShiftMatrixId = "{}-EncryptedShiftMatrix".format(plainTextMatrixID) #Build the id of Encrypted Shift Matrix
    skmeans                = SKMeans()
    responseHeaders        = {} 
    logger.debug("headers:{}".format(requestHeaders))
    try:
        responseHeaders["Start-Time"] = str(arrivalTime)
        encryptedMatrix_response      = STORAGE_CLIENT.get_ndarray(
            key   = encryptedMatrixId,
            cache = True,
            force = isStartStatus
        ).unwrap() # Extract the encrypted dataset
        encryptedMatrix                           = encryptedMatrix_response.value
        responseHeaders["Encrypted-Matrix-Dtype"] = encryptedMatrix_response.metadata.tags.get("dtype",encryptedMatrix.dtype) #["tags"]["dtype"] #Save the data type
        responseHeaders["Encrypted-Matrix-Shape"] = encryptedMatrix_response.metadata.tags.get("shape",encryptedMatrix.shape) #Save the shape
        udm_matrix_response = STORAGE_CLIENT.get_ndarray(
            key = UDMId
        ).unwrap() #Gets the UDM of the storage system
        UDMMatrix                           = udm_matrix_response.value
        #logger.debug("get_UDM_1:{}".format(UDMMatrix))
        responseHeaders["Udm-Matrix-Dtype"] = udm_matrix_response.metadata.tags.get("dtype",UDMMatrix.dtype) # Extract the type
        responseHeaders["Udm-Matrix-Shape"] = udm_matrix_response.metadata.tags.get("shape",UDMMatrix.shape) # Extract the shape
        Cent_iId                            = "{}-Cent_i".format(plainTextMatrixID) #Build the id of Cent_i
        Cent_jId                            = "{}-Cent_j".format(plainTextMatrixID) #Build the id of Cent_j
            
        if(isStartStatus): #if the status is start
            __Cent_j = None #There is no Cent_j
        else: 
            Cent_j_response = STORAGE_CLIENT.get_ndarray(
                key = Cent_iId
            ).unwrap() #Cent_J is extracted from the storage system
            __Cent_j        = Cent_j_response.value
            status = Constants.ClusteringStatus.WORK_IN_PROGRESS
        S1,Cent_i,Cent_j,label_vector = skmeans.run1( # The first part of the skmeans is done
            status          = status,
            k               = k,
            m               = m,
            encryptedMatrix = encryptedMatrix, 
            UDM             = UDMMatrix,
            Cent_j          = __Cent_j
        )
        _ = STORAGE_CLIENT.put_ndarray( # Saving Cent_i to storage
            key     = Cent_iId, 
            ndarray = np.array(Cent_i),
            update  = True
        ).unwrap() 
        _ = STORAGE_CLIENT.put_ndarray( # Saving Cent_j to storage
            key     = Cent_jId, 
            ndarray = np.array(Cent_j),
            update  = True
        ).unwrap() 
        _ = STORAGE_CLIENT.put_ndarray( # Saving S1 matrix to storage
            key     = encryptedShiftMatrixId, 
            ndarray = np.array(S1),
            update  = True
        ).unwrap() 
        
        endTime                                      = time.time()
        serviceTime                                  = endTime - arrivalTime
        responseHeaders["Service-Time"]              = str(serviceTime)
        responseHeaders["Iterations"]                = str(int(requestHeaders.get("Iterations",0)) + 1) #Saves the number of iterations in the header
        responseHeaders["Encrypted-Shift-Matrix-Id"] = encryptedShiftMatrixId #Save the id of the encrypted shift matrix    
        
        run1_logger_metrics = LoggerMetrics( #Write skmeans 1 metrics in logger
            operation_type = "SKMEANS_1",
            matrix_id      = plainTextMatrixID,
            worker_id      = workerId,
            algorithm      = algorithm,
            arrival_time   = arrivalTime, 
            end_time       = endTime, 
            service_time   = serviceTime,
            k_value        = k,
            m_value        = m,
            n_iterations   = responseHeaders.get("Iterations",0)
        )
        logger.info(str(run1_logger_metrics))
                
        return Response( #Returns the final response as a label vector + the headers
            response = json.dumps({"labelVector":label_vector}),
            status   = 200,
            headers  = responseHeaders
        )
    except Exception as e:
        logger.error( encryptedMatrixId+" "+str(e) )
        return Response(None,status = 503,headers = {"Error-Message":str(e)} )

"""
Description:
    Second part of the skmeans process. 
    It starts when it receives S (decrypted matrix) from the client.
    If S is zero process ends
"""
def skmeans_2(requestHeaders):
    arrivalTime           = time.time()
    logger                = current_app.config["logger"]
    workerId              = current_app.config["NODE_ID"]
    STORAGE_CLIENT:Client = current_app.config["STORAGE_CLIENT"]
    status                = int(requestHeaders.get("Clustering-Status",Constants.ClusteringStatus.START))
    algorithm             = Constants.ClusteringAlgorithms.SKMEANS
    k                     = int(requestHeaders.get("K",3))
    m                     = int(requestHeaders.get("M",3))
    encryptedMatrixId     = requestHeaders["Encrypted-Matrix-Id"]
    plainTextMatrixID     = requestHeaders["Plaintext-Matrix-Id"]
    shiftMatrixId         = requestHeaders.get("Shift-Matrix-Id","{}-ShiftMatrix".format(plainTextMatrixID))
    logger.debug("headers:{}".format(requestHeaders))
    UDM_id                = "{}-UDM".format(plainTextMatrixID)
    Cent_iId              = "{}-Cent_i".format(plainTextMatrixID) #Build the id of Cent_i
    Cent_jId              = "{}-Cent_j".format(plainTextMatrixID) #Build the id of Cent_j
    iterations            = int(requestHeaders.get("Iterations",0))
    responseHeaders       = {}
    start_time            = requestHeaders.get("Start-Time","0.0")
    try:
        UDM_response = STORAGE_CLIENT.get_ndarray(
            key = UDM_id
        ).unwrap()
        UDM                      = UDM_response.value
        #logger.debug("get_UDM:{}".format(UDM))

        Cent_i_response = STORAGE_CLIENT.get_ndarray(
            key = Cent_iId
        ).unwrap()
        Cent_i                      = Cent_i_response.value

        Cent_j_response = STORAGE_CLIENT.get_ndarray(
            key = Cent_jId
        ).unwrap()
        Cent_j                      = Cent_j_response.value

        shiftMatrix_get_response = STORAGE_CLIENT.get_ndarray(
            key = shiftMatrixId
        ).unwrap()
        shiftMatrix              = shiftMatrix_get_response.value

        #isZero                   = Utils.verifyZero(shiftMatrix)
        isZero = Utils.verify_mean_error(old_matrix = Cent_i, new_matrix = Cent_j, min_error = 0.15)
        logger.debug("isZero={}".format(isZero))
        logger.debug("_"*20)
        if(isZero): #If Shift matrix is zero
            responseHeaders["Clustering-Status"]  = Constants.ClusteringStatus.COMPLETED #Change the status to COMPLETED
            endTime                               = time.time()
            serviceTime                           = endTime - arrivalTime #The service time is calculated
            responseHeaders["Total-Service-Time"] = str(serviceTime) #Save the service time

            run2_logger_metrics = LoggerMetrics(
            operation_type = "SKMEANS_2",
            matrix_id      = plainTextMatrixID,
            worker_id      = workerId,
            algorithm      = algorithm,
            arrival_time   = arrivalTime, 
            end_time       = endTime, 
            service_time   = serviceTime,
            k_value        = k,
            m_value        = m,
            n_iterations   = iterations
            )
            logger.info(str(run2_logger_metrics))

            return Response( #Return none and headers
                response = None, 
                status   = 204, 
                # headers  = {**requestHeaders, **responseHeaders}
                headers  = responseHeaders
            )
        else: #If Shift matrix is not zero
            skmeans         = SKMeans() 
            status          = Constants.ClusteringStatus.WORK_IN_PROGRESS
            responseHeaders["Clustering-Status"] = status #The status is changed to WORK IN PROGRESS
            attibutes_shape = eval(requestHeaders["Encrypted-Matrix-Shape"]) # extract the attributes of shape
            _UDM            = skmeans.run_2( # The second part of the skmeans starts
                k           = k,
                UDM         = UDM,
                attributes  = attibutes_shape[1],
                shiftMatrix = shiftMatrix,
            )
            UDM_array = np.array(_UDM)
            #logger.debug("put_UDM:{}".format(UDM_array))
            _ = STORAGE_CLIENT.put_ndarray(
                key     = UDM_id, 
                ndarray = UDM_array,
                update  = True
            ).unwrap() # UDM is extracted from the storage system
            endTime2                        = time.time()
            serviceTime2                    = endTime2 - arrivalTime  #Service time is calculated
            responseHeaders["End-Time"]     = str(endTime2)
            responseHeaders["Service-Time"] = str(serviceTime2)
            
            run2_logger_metrics = LoggerMetrics(
                operation_type = "SKMEANS_2",
                matrix_id      = plainTextMatrixID,
                worker_id      = workerId,
                algorithm      = algorithm,
                arrival_time   = arrivalTime, 
                end_time       = endTime2, 
                service_time   = serviceTime2,
                k_value        = k,
                m_value        = m,
                n_iterations   = iterations
            )
            logger.info(str(run2_logger_metrics))
            return Response( #Return none and headers
                response = None,
                status   = 204, 
                headers  = responseHeaders
            )
    except Exception as e:
        logger.error(encryptedMatrixId+" "+str(e))
        return Response(
            response = None,
            status   = 503,
            headers  = {"Error-Message":str(e)} 
        )

@clustering.route("/skmeans",methods = ["POST"])
def skmeans():
    headers         = request.headers
    head            = ["User-Agent","Accept-Encoding","Connection"]
    filteredHeaders = dict(list(filter(lambda x: not x[0] in head, headers.items())))
    step_index      = int(filteredHeaders.get("Step-Index",1))
    response        = Response()
    if step_index == 1:
        return skmeans_1(filteredHeaders)
    elif step_index == 2:
        return skmeans_2(filteredHeaders)
    else:
        return response

@clustering.route("/kmeans",methods = ["POST"])
def kmeans():
    arrivalTime           = time.time() #System startup time
    headers               = request.headers
    to_remove_headers     = ["User-Agent","Accept-Encoding","Connection"]
    filteredHeaders       = dict(list(filter(lambda x: not x[0] in to_remove_headers, headers.items())))
    algorithm             = Constants.ClusteringAlgorithms.KMEANS
    logger                = current_app.config["logger"]
    workerId              = current_app.config["NODE_ID"] # Get the node_id from the global configuration
    STORAGE_CLIENT:Client = current_app.config["STORAGE_CLIENT"]
    plainTextMatrixId     = filteredHeaders.get("Plaintext-Matrix-Id")
    k                     = int(filteredHeaders.get("K",3))
    responseHeaders       = {}
    try:
        plainTextMatrix_response        = STORAGE_CLIENT.get_ndarray(key = plainTextMatrixId, cache=True).unwrap()
        plainTextMatrix                 = plainTextMatrix_response.value
        result                          = kMeans(k = k, plaintext_matrix = plainTextMatrix)
        endTime                         = time.time()
        serviceTime                     = endTime - arrivalTime
        responseHeaders["Service-Time"] = str(serviceTime)
        responseHeaders["Iterations"]   = int(result.n_iterations)
        
        logger_metrics = LoggerMetrics(
            operation_type = algorithm, 
            matrix_id      = plainTextMatrixId, 
            worker_id      = workerId,
            algorithm      = algorithm, 
            arrival_time   = arrivalTime, 
            end_time       = endTime, 
            service_time   = serviceTime,
            k_value        = k,
            n_iterations   = result.n_iterations)
        logger.info(str(logger_metrics))

        return Response( #Returns the final response as a label vector + the headers
            response = json.dumps({"labelVector":result.label_vector.tolist()}),
            status   = 200,
            headers  = {**responseHeaders}
        )
    except Exception as e:
        print(e)
        return Response(
            response = None,
            status   = 503,
            headers  = {"Error-Message":str(e)}
        )


"""
Description:
    First part of the dbskmeans process. 
    It stops where client interaction is required and writes the centroids and matrix S to disk.
"""
def dbskmeans_1(requestHeaders) -> Response:
    arrivalTime            = time.time() #System startup time
    logger                 = current_app.config["logger"]
    workerId               = current_app.config["NODE_ID"] # Get the node_id from the global configuration
    STORAGE_CLIENT:Client  = current_app.config["STORAGE_CLIENT"]
    status                 = int(requestHeaders.get("Clustering-Status", Constants.ClusteringStatus.START)) 
    isStartStatus          = status == Constants.ClusteringStatus.START #if status is start save it to isStartStatus
    k                      = int(requestHeaders.get("K",3)) # It is passed to integer because the headers are strings
    m                      = int(requestHeaders.get("M",3))
    algorithm              = Constants.ClusteringAlgorithms.DBSKMEANS
    plainTextMatrixID      = requestHeaders.get("Plaintext-Matrix-Id")
    encryptedMatrixId      = requestHeaders.get("Encrypted-Matrix-Id","")
    UDMId                  = "{}-encrypted-UDM".format(plainTextMatrixID) 
    Cent_iId               = "{}-Cent_i".format(plainTextMatrixID) #Build the id of Cent_i
    Cent_jId               = "{}-Cent_j".format(plainTextMatrixID) #Build the id of Cent_j
    encryptedShiftMatrixId = "{}-EncryptedShiftMatrix".format(plainTextMatrixID) #Build the id of Encrypted Shift Matrix
    dbskmeans              = DBSKMeans()
    responseHeaders        = {}
    logger.debug("headers:{}".format(requestHeaders))
    try:
        #if(isStartStatus):
        responseHeaders["Start-Time"] = str(arrivalTime)
        encryptedMatrix_response      = STORAGE_CLIENT.get_ndarray(
            key   = encryptedMatrixId,
            cache = True, 
            force = isStartStatus
        ).unwrap() # Extract the encrypted dataset
        encryptedMatrix          = encryptedMatrix_response.value
        responseHeaders["Encrypted-Matrix-Dtype"] = encryptedMatrix_response.metadata.tags.get("dtype",encryptedMatrix.dtype) #Save the data type
        responseHeaders["Encrypted-Matrix-Shape"] = encryptedMatrix_response.metadata.tags.get("shape",encryptedMatrix.shape) #Save the shape
        UDMMatrix_response = STORAGE_CLIENT.get_ndarray(
            key= UDMId
        ).unwrap() #Gets the UDM of the storage system
        UDMMatrix                           = UDMMatrix_response.value
        responseHeaders["Udm-Matrix-Dtype"] = UDMMatrix_response.metadata.tags.get("dtype",UDMMatrix.dtype) # Extract the type
        responseHeaders["Udm-Matrix-Shape"] = UDMMatrix_response.metadata.tags.get("shape",UDMMatrix.shape) # Extract the shape
        if(isStartStatus): #if the status is start
            __Cent_j = None #There is no Cent_j
        else: 
            __Cent_j_response = STORAGE_CLIENT.get_ndarray(
                key = Cent_iId
            ).unwrap() #Cent_J is extracted from the storage system
            __Cent_j          = __Cent_j_response.value
            status = Constants.ClusteringStatus.WORK_IN_PROGRESS
        S1,Cent_i,Cent_j,label_vector = dbskmeans.run1(
            status           = status,
            k                = k,
            m                = m,
            encryptedMatrix  = encryptedMatrix,
            UDM              = UDMMatrix,
            Cent_j           = __Cent_j
        ) 
        _ = STORAGE_CLIENT.put_ndarray( # Saving Cent_i to storage
            key     = Cent_iId, 
            ndarray = np.array(Cent_i),
            update  = True
        ).unwrap() 
        _ = STORAGE_CLIENT.put_ndarray( # Saving Cent_j to storage
            key     = Cent_jId, 
            ndarray = np.array(Cent_j),
            update  = True
        ).unwrap() 
        _ = STORAGE_CLIENT.put_ndarray( # Saving S1 matrix to storage
            key     = encryptedShiftMatrixId,  
            ndarray = np.array(S1),
            update  = True
        ).unwrap() 
        
        endTime                                      = time.time()
        serviceTime                                  = endTime - arrivalTime
        responseHeaders["Service-Time"]              = str(serviceTime)
        responseHeaders["Iterations"]                = str(int(requestHeaders.get("Iterations",0)) + 1) #Saves the number of iterations in the header
        responseHeaders["Encrypted-Shift-Matrix-Id"] = encryptedShiftMatrixId #Save the id of the encrypted shift matrix
        responseHeaders["Service-Time"]              = str(serviceTime) #Save the service time

        logger_metrics = LoggerMetrics(
            operation_type = "DBSKMEANS_1",
            matrix_id      = plainTextMatrixID,
            worker_id      = workerId,
            algorithm      = algorithm, 
            arrival_time   = arrivalTime, 
            end_time       = endTime, 
            service_time   = serviceTime,
            m_value        = m,
            k_value        = k,
            n_iterations   = responseHeaders.get("Iterations",0))
        logger.info(str(logger_metrics))
        
        return Response( #Returns the final response as a label vector + the headers
            response = json.dumps({"labelVector":label_vector }),
            status   = 200,
            headers  = {**requestHeaders, **responseHeaders}
        )
    except Exception as e:
        print("ERROR {}".format(e))
        logger.error( encryptedMatrixId+" "+str(e) )
        return Response(None,status = 503,headers = {"error-Message":str(e)} )

"""
Description:
    Second part of the dbskmeans process. 
    It starts when it receives S (decrypted matrix) from the client.
    If S is zero process ends
"""
def dbskmeans_2(requestHeaders):
    arrivalTime           = time.time()
    logger                = current_app.config["logger"]
    workerId              = current_app.config["NODE_ID"]
    STORAGE_CLIENT:Client = current_app.config["STORAGE_CLIENT"]
    algorithm             = Constants.ClusteringAlgorithms.DBSKMEANS
    status                = int(requestHeaders.get("Clustering-Status",Constants.ClusteringStatus.START))
    k                     = requestHeaders.get("K",3)
    m                     = requestHeaders.get("M",3)
    iterations            = int(requestHeaders.get("Iterations",0))
    start_time            = requestHeaders.get("Start-Time","0.0")
    encryptedMatrixId     = requestHeaders["Encrypted-Matrix-Id"]
    plainTextMatrixID     = requestHeaders["Plaintext-Matrix-Id"]
    shiftMatrixId         = requestHeaders.get("Shift-Matrix-Id","{}-ShiftMatrix".format(plainTextMatrixID))
    shiftMatrixOpeId      = requestHeaders.get("Shift-Matrix-Ope-Id","{}-ShiftMatrixOpe".format(plainTextMatrixID))
    UDM_id                = "{}-encrypted-UDM".format(plainTextMatrixID)
    Cent_iId              = "{}-Cent_i".format(plainTextMatrixID) #Build the id of Cent_i
    Cent_jId              = "{}-Cent_j".format(plainTextMatrixID) #Build the id of Cent_j
    responseHeaders       = {}
    logger.debug("headers:{}".format(requestHeaders))
    try:
        UDMMatrix = STORAGE_CLIENT.get_ndarray(
                key = UDM_id,
        ).unwrap()
        UDM = UDMMatrix.value
        
        Cent_i_response = STORAGE_CLIENT.get_ndarray(
            key = Cent_iId
        ).unwrap()
        Cent_i = Cent_i_response.value

        Cent_j_response = STORAGE_CLIENT.get_ndarray(
            key = Cent_jId
        ).unwrap()
        Cent_j = Cent_j_response.value
        
        shiftMatrix_response = STORAGE_CLIENT.get_ndarray(
            key = shiftMatrixId
        ).unwrap()
        shiftMatrix = shiftMatrix_response.value
        #isZero     = Utils.verifyZero(shiftMatrix)
        isZero = Utils.verify_mean_error(old_matrix = Cent_i, new_matrix = Cent_j, min_error = 0.15)
        logger.debug("isZero={}".format(isZero))
        logger.debug("_"*20)

        if(isZero): #If Shift matrix is zero
            responseHeaders["Clustering-Status"]  = Constants.ClusteringStatus.COMPLETED #Change the status to COMPLETED
            endTime                               = time.time()
            totalServiceTime                      = endTime - float(start_time) #The service time is calculated
            responseHeaders["Total-Service-Time"] = str(totalServiceTime) #Save the service time
            
            logger_metrics = LoggerMetrics(
                operation_type = "DBSKMEANS_2",
                matrix_id      = plainTextMatrixID,
                worker_id      = workerId,
                algorithm      = algorithm, 
                arrival_time   = arrivalTime, 
                end_time       = endTime, 
                service_time   = totalServiceTime,
                m_value        = m,
                k_value        = k,
                n_iterations   = iterations)
            logger.info(str(logger_metrics))
            return Response( #Return none and headers
                response = None, 
                status   = 204, 
                headers  = {**requestHeaders, **responseHeaders}
            )
        else: #If Shift matrix is not zero
            dbskmeans               = DBSKMeans() 
            status                  = Constants.ClusteringStatus.WORK_IN_PROGRESS #The status is changed to WORK IN PROGRESS
            responseHeaders["Clustering-Status"] = status
            attibutes_shape         = eval(requestHeaders["Encrypted-Matrix-Shape"]) # extract the attributes of shape
            shiftMatrixOpe_response = STORAGE_CLIENT.get_ndarray(
                key = shiftMatrixOpeId
            ).unwrap()
            shiftMatrixOpe = shiftMatrixOpe_response.value
            _UDMMatrix     = dbskmeans.run_2( # The second part of the skmeans starts
                k                = k,
                UDM              = UDM,
                attributes       = attibutes_shape[1],
                shiftMatrix      = shiftMatrixOpe,
            )
            UDM_array = np.array(_UDMMatrix)
            #udm_put_payload                 = PutNDArrayPayload( key = UDM_id, ndarray= np.array(_UDMMatrix))
            _ = STORAGE_CLIENT.put_ndarray(
                key     = UDM_id, 
                ndarray = UDM_array,
                update  = True
            ).unwrap() # UDM is extracted from the storage system
            end_time                        = time.time()
            service_time                    = end_time - arrivalTime  #Service time is calculated
            responseHeaders["End-Time"]     = str(end_time)
            responseHeaders["Service-Time"] = str(service_time)

            logger_metrics = LoggerMetrics(
                operation_type = "DBSKMEANS_2",
                matrix_id      = plainTextMatrixID,
                worker_id      = workerId,
                algorithm      = algorithm, 
                arrival_time   = arrivalTime, 
                end_time       = endTime, 
                service_time   = service_time,
                m_value        = m,
                k_value        = k,
                n_iterations   = iterations
            )
            logger.info(str(logger_metrics))
            
            return Response( #Return none and headers
                response = None,
                status   = 204, 
                headers  = {**requestHeaders, **responseHeaders}
            )
    except Exception as e:
        logger.error(encryptedMatrixId+" "+str(e))
        return Response(
            response = None,
            status   = 503,
            headers  = {"Error-Message":str(e)} 
        )

"""
Description:
    DBSKMEANS algorithm
"""
@clustering.route("/dbskmeans", methods = ["POST"])
def dbskmeans():
    headers         = request.headers
    head            = ["User-Agent","Accept-Encoding","Connection"]
    filteredHeaders = dict(list(filter(lambda x: not x[0] in head, headers.items())))
    step_index      = int(filteredHeaders.get("Step-Index",1))
    response        = Response()
    if step_index == 1:
        return dbskmeans_1(filteredHeaders)
    elif step_index == 2:
        return dbskmeans_2(filteredHeaders)
    else:
        return response


@clustering.route("/dbsnnc", methods = ["POST"])
def dbsnnc():
    arrivalTime           = time.time() #System startup time
    headers               = request.headers
    head                  = ["User-Agent","Accept-Encoding","Connection"]
    filteredHeaders       = dict(list(filter(lambda x: not x[0] in head, headers.items())))
    algorithm             = Constants.ClusteringAlgorithms.DBSNNC
    logger                = current_app.config["logger"]
    STORAGE_CLIENT:Client = current_app.config["STORAGE_CLIENT"]
    workerId              = current_app.config["NODE_ID"] # Get the node_id from the global configuration
    plainTextMatrixId     = filteredHeaders.get("Plaintext-Matrix-Id")
    encryptedMatrixId     = filteredHeaders.get("Encrypted-Matrix-Id","")
    encrypted_threshold   = filteredHeaders.get("Encrypted-Threshold")
    UDMId                 = "{}-encrypted-UDM".format(plainTextMatrixId)  
    responseHeaders       = {}
    try:        
        UDMMatrix_response = STORAGE_CLIENT.get_ndarray(key = UDMId).unwrap()
        UDMMatrix          = UDMMatrix_response.value
        result             = Dbsnnc.run(
            EDM                 = UDMMatrix,
            encrypted_threshold = float(encrypted_threshold)
        )
        endTime                         = time.time()
        serviceTime                     = endTime - arrivalTime
        responseHeaders["Service-Time"] = str(serviceTime)
        
        logger_metrics = LoggerMetrics(
            operation_type = algorithm,
            matrix_id      = plainTextMatrixId,
            worker_id      = workerId,
            algorithm      = algorithm,
            arrival_time   = arrivalTime, 
            end_time       = endTime, 
            service_time   = serviceTime
        )
        logger.info(str(logger_metrics))

        return Response( #Returns the final response as a label vector + the headers
            response = json.dumps({"labelVector":result.label_vector}),
            status   = 200,
            headers  = responseHeaders
        )
    except Exception as e:
        print(e)
        return Response(
            response = None,
            status   = 503,
            headers  = {"Error-Message":e})