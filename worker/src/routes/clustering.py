import time, json
import numpy as np
from flask import Blueprint,current_app,request,Response
from rory.core.clustering.kmeans import kmeans as kMeans
from rory.core.clustering.secure.local.dbsnnc import Dbsnnc
from rory.core.utils.Utils import Utils
from rory.core.utils.SegmentationUtils import Segmentation
from rory.core.utils.constants import Constants
from rory.core.clustering.secure.distributed.skmeans import SKMeans
from rory.core.clustering.secure.distributed.dbskmeans import DBSKMeans
from mictlanx.v3.client import Client 
from mictlanx.v4.client import Client as V4Client
from option import Result
from mictlanx.v4.interfaces.responses import GetNDArrayResponse
from mictlanx.v3.interfaces.payloads import PutNDArrayPayload
from rory.core.interfaces.logger_metrics import LoggerMetrics
# 
clustering = Blueprint("clustering",__name__,url_prefix = "/clustering")


@clustering.route("/test",methods=["GET","POST"])
def test():
    return Response(
        response = json.dumps({
            "component_type":"worker"
        }),
        status   = 200,
        headers  = {
            "Component-Type":"worker"
        }
    )

"""
Description:
    First part of the skmeans process. 
    It stops where client interaction is required and writes the centroids and matrix S to disk.
"""
def skmeans_1(requestHeaders) -> Response:
    arrivalTime             = time.time() #Worker start time
    logger                  = current_app.config["logger"]
    workerId                = current_app.config["NODE_ID"] # Get the node_id from the global configuration
    STORAGE_CLIENT:V4Client = current_app.config["STORAGE_CLIENT"]
    BUCKET_ID:str           = current_app.config.get("BUCKET_ID","rory")
    status                  = int(requestHeaders.get("Clustering-Status", Constants.ClusteringStatus.START)) 
    isStartStatus           = status == Constants.ClusteringStatus.START #if status is start save it to isStartStatus
    k                       = int(requestHeaders.get("K",3)) # It is passed to integer because the headers are strings
    m                       = int(requestHeaders.get("M",3))
    algorithm               = Constants.ClusteringAlgorithms.SKMEANS
    plainTextMatrixID       = requestHeaders.get("Plaintext-Matrix-Id")
    encryptedMatrixId       = requestHeaders.get("Encrypted-Matrix-Id","")
    UDMId                   = "{}-UDM".format(plainTextMatrixID) 
    encryptedShiftMatrixId  = "{}-EncryptedShiftMatrix".format(plainTextMatrixID) #Build the id of Encrypted Shift Matrix
    skmeans                 = SKMeans()
    responseHeaders         = {}
    logger.debug("Worker starts SKMEANS_1 process -> {}".format(plainTextMatrixID))
    try:
        responseHeaders["Start-Time"] = str(arrivalTime)
        print("BEFORE")
        encryptedMatrix_response = STORAGE_CLIENT.get_and_merge_ndarray(key=encryptedMatrixId)
        x                        = encryptedMatrix_response.result()
        encryptedMatrix_response:GetNDArrayResponse = x.unwrap()
        encryptedMatrix                             = encryptedMatrix_response.value
        responseHeaders["Encrypted-Matrix-Dtype"]   = encryptedMatrix_response.metadata.tags.get("dtype",encryptedMatrix.dtype) #["tags"]["dtype"] #Save the data type
        responseHeaders["Encrypted-Matrix-Shape"]   = encryptedMatrix_response.metadata.tags.get("shape",encryptedMatrix.shape) #Save the shape
        
        udm_matrix_response = Segmentation.get_matrix_or_error(
            client = STORAGE_CLIENT,
            key    = UDMId
        )
                
        UDMMatrix                           = udm_matrix_response.value
        responseHeaders["Udm-Matrix-Dtype"] = udm_matrix_response.metadata.tags.get("dtype",UDMMatrix.dtype) # Extract the type
        responseHeaders["Udm-Matrix-Shape"] = udm_matrix_response.metadata.tags.get("shape",UDMMatrix.shape) # Extract the shape
        Cent_iId                            = "{}-Cent_i".format(plainTextMatrixID) #Build the id of Cent_i
        Cent_jId                            = "{}-Cent_j".format(plainTextMatrixID) #Build the id of Cent_j
            
        if(isStartStatus): #if the status is start
            __Cent_j = None #There is no Cent_j
        else: 
            Cent_j_response = Segmentation.get_matrix_or_error(
                client = STORAGE_CLIENT,
                key    = Cent_iId
            )
            __Cent_j        = Cent_j_response.value
            status          = Constants.ClusteringStatus.WORK_IN_PROGRESS

        S1,Cent_i,Cent_j,label_vector = skmeans.run1( # The first part of the skmeans is done
            status          = status,
            k               = k,
            m               = m,
            encryptedMatrix = encryptedMatrix, 
            UDM             = UDMMatrix,
            Cent_j          = __Cent_j
        )

        x = STORAGE_CLIENT.put_ndarray( # Saving Cent_i to storage
            key       = Cent_iId, 
            ndarray   = np.array(Cent_i),
            tags      = {},
            bucket_id = BUCKET_ID
        )

        y = STORAGE_CLIENT.put_ndarray( # Saving Cent_j to storage
            key       = Cent_jId, 
            ndarray   = np.array(Cent_j),
            tags      = {},
            bucket_id = BUCKET_ID
        )

        z = STORAGE_CLIENT.put_ndarray( # Saving S1 matrix to storage
            key       = encryptedShiftMatrixId, 
            ndarray   = np.array(S1),
            tags      = {},
            bucket_id = BUCKET_ID
        )
        
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

        x = x.result()
        print("X",x)
        y = y.result()
        print("y",y)
        z = z.result() 
        print("z",z)

        logger.info(str(run1_logger_metrics))
      
        return Response( #Returns the final response as a label vector + the headers
            response = json.dumps({"labelVector":label_vector}),
            status   = 200,
            headers  = responseHeaders
        )
    
    except Exception as e:

        logger.error("WORKER_SKMEANS_1 "+encryptedMatrixId+" "+str(e))

        return Response(
            None,
            status  = 500,
            headers = {"Error-Message":str(e)} 
        )


"""
Description:
    Second part of the skmeans process. 
    It starts when it receives S (decrypted matrix) from the client.
    If S is zero process ends
"""
def skmeans_2(requestHeaders):
    arrivalTime             = time.time()
    logger                  = current_app.config["logger"]
    workerId                = current_app.config["NODE_ID"]
    BUCKET_ID:str           = current_app.config.get("BUCKET_ID","rory")
    STORAGE_CLIENT:V4Client = current_app.config["STORAGE_CLIENT"]
    status                  = int(requestHeaders.get("Clustering-Status",Constants.ClusteringStatus.START))
    algorithm               = Constants.ClusteringAlgorithms.SKMEANS
    k                       = int(requestHeaders.get("K",3))
    m                       = int(requestHeaders.get("M",3))
    encryptedMatrixId       = requestHeaders["Encrypted-Matrix-Id"]
    plainTextMatrixID       = requestHeaders["Plaintext-Matrix-Id"]
    shiftMatrixId           = requestHeaders.get("Shift-Matrix-Id","{}-ShiftMatrix".format(plainTextMatrixID))
    UDM_id                  = "{}-UDM".format(plainTextMatrixID)
    Cent_iId                = "{}-Cent_i".format(plainTextMatrixID) #Build the id of Cent_i
    Cent_jId                = "{}-Cent_j".format(plainTextMatrixID) #Build the id of Cent_j
    iterations              = int(requestHeaders.get("Iterations",0))
    responseHeaders         = {}
    start_time              = requestHeaders.get("Start-Time","0.0")
    logger.debug("Worker starts SKMEANS_2 process -> {}".format(plainTextMatrixID))
    try:
        UDM_response = Segmentation.get_matrix_or_error(
            client = STORAGE_CLIENT, 
            key    = UDM_id
        )
        UDM             = UDM_response.value
        Cent_i_response = Segmentation.get_matrix_or_error(
            client = STORAGE_CLIENT,
            key    = Cent_iId
        )
        Cent_i          = Cent_i_response.value
        Cent_j_response = Segmentation.get_matrix_or_error(
            client = STORAGE_CLIENT,
            key    = Cent_jId
        ) 
        Cent_j                   = Cent_j_response.value
        shiftMatrix_get_response = Segmentation.get_matrix_or_error(
            client = STORAGE_CLIENT,
            key    = shiftMatrixId
        )
        shiftMatrix = shiftMatrix_get_response.value

        isZero = Utils.verify_mean_error(
            old_matrix = Cent_i, 
            new_matrix = Cent_j, 
            min_error  = 0.15
        )
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
                headers  = responseHeaders
            )
        else: #If Shift matrix is not zero
            skmeans         = SKMeans() 
            status          = Constants.ClusteringStatus.WORK_IN_PROGRESS
            print("STATUS",status)
            responseHeaders["Clustering-Status"] = status #The status is changed to WORK IN PROGRESS
            attibutes_shape = eval(requestHeaders["Encrypted-Matrix-Shape"]) # extract the attributes of shape
            _UDM            = skmeans.run_2( # The second part of the skmeans starts
                k           = k,
                UDM         = UDM,
                attributes  = int(attibutes_shape[1]),
                shiftMatrix = shiftMatrix,
            )
            print("RUN 2 FINISH")
            UDM_array = np.array(_UDM)
            x = STORAGE_CLIENT.put_ndarray(
                key       = UDM_id, 
                ndarray   = UDM_array,
                tags      = {},
                bucket_id = BUCKET_ID
            ).result() # UDM is extracted from the storage system
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
            # x_result = x.result()
            # print(x_result)
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
    arrivalTime             = time.time() #System startup time
    headers                 = request.headers
    to_remove_headers       = ["User-Agent","Accept-Encoding","Connection"]
    filteredHeaders         = dict(list(filter(lambda x: not x[0] in to_remove_headers, headers.items())))
    algorithm               = Constants.ClusteringAlgorithms.KMEANS
    logger                  = current_app.config["logger"]
    workerId                = current_app.config["NODE_ID"] # Get the node_id from the global configuration
    STORAGE_CLIENT:V4Client = current_app.config["STORAGE_CLIENT"]
    BUCKET_ID:str           = current_app.config.get("BUCKET_ID","rory")
    plainTextMatrixId       = filteredHeaders.get("Plaintext-Matrix-Id")
    k                       = int(filteredHeaders.get("K",3))
    responseHeaders         = {}
    try:
        plainTextMatrix_response = Segmentation.get_matrix_or_error(
            client = STORAGE_CLIENT,
            key    = plainTextMatrixId
        ) 
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
    arrivalTime             = time.time() #System startup time
    logger                  = current_app.config["logger"]
    workerId                = current_app.config["NODE_ID"] # Get the node_id from the global configuration
    STORAGE_CLIENT:V4Client = current_app.config["STORAGE_CLIENT"]
    BUCKET_ID:str           = current_app.config.get("BUCKET_ID","rory")
    status                  = int(requestHeaders.get("Clustering-Status", Constants.ClusteringStatus.START)) 
    isStartStatus           = status == Constants.ClusteringStatus.START #if status is start save it to isStartStatus
    k                       = int(requestHeaders.get("K",3)) # It is passed to integer because the headers are strings
    m                       = int(requestHeaders.get("M",3))
    algorithm               = Constants.ClusteringAlgorithms.DBSKMEANS
    plaintext_matrix_id     = requestHeaders.get("Plaintext-Matrix-Id")
    encrypted_matrix_id     = requestHeaders.get("Encrypted-Matrix-Id","")
    encrypted_udm_id        = "{}-encrypted-UDM".format(plaintext_matrix_id) 
    Cent_iId                = "{}-Cent_i".format(plaintext_matrix_id) #Build the id of Cent_i
    Cent_jId                = "{}-Cent_j".format(plaintext_matrix_id) #Build the id of Cent_j
    encryptedShiftMatrixId  = "{}-EncryptedShiftMatrix".format(plaintext_matrix_id) #Build the id of Encrypted Shift Matrix
    dbskmeans               = DBSKMeans()
    responseHeaders         = {}
    
    logger.debug("Worker starts SKMEANS_1 process -> {}".format(plaintext_matrix_id))
    try:
        responseHeaders["Start-Time"] = str(arrivalTime)
        print("BEFORE")
        encryptedMatrix_response = STORAGE_CLIENT.get_and_merge_ndarray(key=encrypted_matrix_id)
        x = encryptedMatrix_response.result()
        print("CHUNKS_RES",x)
        print("_"*20)
        encryptedMatrix_response:GetNDArrayResponse = x.unwrap()
        encryptedMatrix          = encryptedMatrix_response.value
        responseHeaders["Encrypted-Matrix-Dtype"] = encryptedMatrix_response.metadata.tags.get("dtype",encryptedMatrix.dtype) #Save the data type
        responseHeaders["Encrypted-Matrix-Shape"] = encryptedMatrix_response.metadata.tags.get("shape",encryptedMatrix.shape) #Save the shape
        
        udm_matrix_response:Result[GetNDArrayResponse,Exception] = STORAGE_CLIENT.get_and_merge_ndarray(key=encrypted_udm_id).result().unwrap()
        UDMMatrix                           = udm_matrix_response.value
        responseHeaders["Udm-Matrix-Dtype"] = udm_matrix_response.metadata.tags.get("dtype",UDMMatrix.dtype) # Extract the type
        responseHeaders["Udm-Matrix-Shape"] = udm_matrix_response.metadata.tags.get("shape",UDMMatrix.shape) # Extract the shape
        
        if(isStartStatus): #if the status is start
            __Cent_j = None #There is no Cent_j
        else: 
            Cent_j_response = Segmentation.get_matrix_or_error(
                client = STORAGE_CLIENT,
                key    = Cent_iId
            )
            __Cent_j          = Cent_j_response.value
            status = Constants.ClusteringStatus.WORK_IN_PROGRESS
        
        S1,Cent_i,Cent_j,label_vector = dbskmeans.run1(
            status           = status,
            k                = k,
            m                = m,
            encryptedMatrix  = encryptedMatrix,
            UDM              = UDMMatrix,
            Cent_j           = __Cent_j
        ) 
        x = STORAGE_CLIENT.put_ndarray( # Saving Cent_i to storage
            key       = Cent_iId, 
            ndarray   = np.array(Cent_i),
            tags      = {},
            bucket_id = BUCKET_ID
        )

        y = STORAGE_CLIENT.put_ndarray( # Saving Cent_j to storage
            key       = Cent_jId, 
            ndarray   = np.array(Cent_j),
            tags      = {},
            bucket_id = BUCKET_ID
        )

        z = STORAGE_CLIENT.put_ndarray( # Saving S1 matrix to storage
            key       = encryptedShiftMatrixId,  
            ndarray   = np.array(S1),
            tags      = {},
            bucket_id = BUCKET_ID
        )
        
        endTime                                      = time.time()
        serviceTime                                  = endTime - arrivalTime
        responseHeaders["Service-Time"]              = str(serviceTime)
        responseHeaders["Iterations"]                = str(int(requestHeaders.get("Iterations",0)) + 1) #Saves the number of iterations in the header
        responseHeaders["Encrypted-Shift-Matrix-Id"] = encryptedShiftMatrixId #Save the id of the encrypted shift matrix

        x = x.result()
        y = y.result()
        z = z.result() 
        
        logger_metrics = LoggerMetrics(
            operation_type = "DBSKMEANS_1",
            matrix_id      = plaintext_matrix_id,
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
        logger.error( encrypted_matrix_id+" "+str(e) )
        return Response(None,status = 503,headers = {"error-Message":str(e)} )

"""
Description:
    Second part of the dbskmeans process. 
    It starts when it receives S (decrypted matrix) from the client.
    If S is zero process ends
"""
def dbskmeans_2(requestHeaders):
    arrivalTime             = time.time()
    logger                  = current_app.config["logger"]
    workerId                = current_app.config["NODE_ID"]
    BUCKET_ID:str           = current_app.config.get("BUCKET_ID","rory")
    STORAGE_CLIENT:V4Client = current_app.config["STORAGE_CLIENT"]
    algorithm               = Constants.ClusteringAlgorithms.DBSKMEANS
    status                  = int(requestHeaders.get("Clustering-Status",Constants.ClusteringStatus.START))
    k                       = requestHeaders.get("K",3)
    m                       = requestHeaders.get("M",3)
    iterations              = int(requestHeaders.get("Iterations",0))
    start_time              = requestHeaders.get("Start-Time","0.0")
    encryptedMatrixId       = requestHeaders["Encrypted-Matrix-Id"]
    plainTextMatrixID       = requestHeaders["Plaintext-Matrix-Id"]
    shiftMatrixId           = requestHeaders.get("Shift-Matrix-Id","{}-ShiftMatrix".format(plainTextMatrixID))
    shiftMatrixOpeId        = requestHeaders.get("Shift-Matrix-Ope-Id","{}-ShiftMatrixOpe".format(plainTextMatrixID))
    UDM_id                  = "{}-encrypted-UDM".format(plainTextMatrixID)
    Cent_iId                = "{}-Cent_i".format(plainTextMatrixID) #Build the id of Cent_i
    Cent_jId                = "{}-Cent_j".format(plainTextMatrixID) #Build the id of Cent_j
    responseHeaders         = {}
    logger.debug("Worker starts SKMEANS_2 process -> {}".format(plainTextMatrixID))
    try:
        UDM_response:Result[GetNDArrayResponse,Exception] = STORAGE_CLIENT.get_and_merge_ndarray(key=UDM_id).result().unwrap()
        UDM             = UDM_response.value
        Cent_i_response = Segmentation.get_matrix_or_error(
            client = STORAGE_CLIENT,
            key    = Cent_iId
        )
        Cent_i          = Cent_i_response.value
        Cent_j_response = Segmentation.get_matrix_or_error(
            client = STORAGE_CLIENT,
            key    = Cent_jId
        ) 
        Cent_j                   = Cent_j_response.value

        shiftMatrix_get_response = Segmentation.get_matrix_or_error(
            client = STORAGE_CLIENT,
            key    = shiftMatrixId
        )
        shiftMatrix = shiftMatrix_get_response.value
        isZero = Utils.verify_mean_error(
            old_matrix = Cent_i, 
            new_matrix = Cent_j,
            min_error  = 0.15
        )
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
                n_iterations   = iterations
            )
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
            shiftMatrixOpe_response = Segmentation.get_matrix_or_error(
                client = STORAGE_CLIENT,
                key = shiftMatrixOpeId
            )
            shiftMatrixOpe = shiftMatrixOpe_response.value
            _UDMMatrix     = dbskmeans.run_2( # The second part of the skmeans starts
                k                = k,
                UDM              = UDM,
                attributes       = int(attibutes_shape[1]),
                shiftMatrix      = shiftMatrixOpe,
            )
            UDM_array = np.array(_UDMMatrix)
            _ = STORAGE_CLIENT.put_ndarray(
                key       = UDM_id, 
                ndarray   = UDM_array,
                ags       = {},
                bucket_id = BUCKET_ID
            ).result() # UDM is extracted from the storage system
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