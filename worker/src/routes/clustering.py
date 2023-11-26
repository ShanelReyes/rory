import time, json
import numpy as np
import numpy.typing as npt
from typing import Awaitable,Tuple
from flask import Blueprint,current_app,request,Response
from rory.core.clustering.kmeans import kmeans as kMeans
from rory.core.clustering.secure.local.dbsnnc import Dbsnnc
from rory.core.clustering.nnc import Nnc
from rory.core.utils.Utils import Utils
from rory.core.utils.SegmentationUtils import Segmentation
from rory.core.utils.constants import Constants
from rory.core.clustering.secure.distributed.skmeans import SKMeans
from rory.core.clustering.secure.distributed.dbskmeans import DBSKMeans
from mictlanx.v3.client import Client 
from mictlanx.v4.client import Client as V4Client
from option import Result
from mictlanx.utils.segmentation import Chunks
from mictlanx.v4.interfaces.responses import GetNDArrayResponse,GetBytesResponse,Metadata
from mictlanx.v3.interfaces.payloads import PutNDArrayPayload
from rory.core.interfaces.logger_metrics import LoggerMetrics
from option import Option,Some
from functools import reduce
import operator
from logging import Logger

from rory.core.logger import Dumblogger
# 
clustering = Blueprint("clustering",__name__,url_prefix = "/clustering")

def get_and_merge_ndarray(STORAGE_CLIENT:V4Client,bucket_id:str, key:str,num_chunks:int, shape:tuple,dtype:str,logger:Logger)->Tuple[npt.NDArray,Metadata]:
    # logger = kwargs.get("logger",Dumblogger())
    
    encryptedMatrix_result:Result[GetBytesResponse,Exception]                    = STORAGE_CLIENT.get_and_merge_with_num_chunks(bucket_id=bucket_id,key=key,num_chunks=num_chunks).result()
    if encryptedMatrix_result.is_err:
        raise Exception("{} not found".format(key))
    
    encryptedMatrix_response = encryptedMatrix_result.unwrap()
    _encryptedMatrix         = np.frombuffer(encryptedMatrix_response.value,dtype=dtype)
    logger.debug("SHAPE "+str(shape))
    logger.debug("RAW_SHAPE "+str(_encryptedMatrix.shape))
    expected_shape           = reduce(operator.mul,shape)
    logger.debug("EXPECTED_SHAPE "+str(expected_shape))
    if not _encryptedMatrix.size == expected_shape:
        raise Exception("Matrix sizes are not equal: calculated: {} != expected: {}".format(_encryptedMatrix.size, expected_shape ))
    
    encryptedMatrix = _encryptedMatrix.reshape(shape)
    return (encryptedMatrix,encryptedMatrix_response.metadata)

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
    arrivalTime              = time.time() #Worker start time
    logger                   = current_app.config["logger"]
    workerId                 = current_app.config["NODE_ID"] # Get the node_id from the global configuration
    STORAGE_CLIENT:V4Client  = current_app.config["STORAGE_CLIENT"]
    BUCKET_ID:str            = current_app.config.get("BUCKET_ID","rory")
    status                   = int(requestHeaders.get("Clustering-Status", Constants.ClusteringStatus.START)) 
    isStartStatus            = status == Constants.ClusteringStatus.START #if status is start save it to isStartStatus
    k                        = int(requestHeaders.get("K",3)) # It is passed to integer because the headers are strings
    m                        = int(requestHeaders.get("M",3))
    algorithm                = Constants.ClusteringAlgorithms.SKMEANS
    plainTextMatrixID        = requestHeaders.get("Plaintext-Matrix-Id")
    encryptedMatrixId        = requestHeaders.get("Encrypted-Matrix-Id",-1)
    UDMId                    = "{}-UDM".format(plainTextMatrixID) 
    _encrypted_matrix_shape  = requestHeaders.get("Encrypted-Matrix-Shape",-1)
    _encrypted_matrix_dtype  = requestHeaders.get("Encrypted-Matrix-Dtype",-1)

    logger.debug(str(requestHeaders))
    logger.debug("SKMEANS_1 algorithm={}, m={}, k={}, plain_matrix_id={}, ems={}, emd={}".format(algorithm,m,k,plainTextMatrixID,_encrypted_matrix_shape,_encrypted_matrix_dtype))

    if _encrypted_matrix_dtype == -1:
        return Response("Encrypted-Matrix-Dtype", status=500)
    if _encrypted_matrix_shape == -1 :
        return Response("Encrypted-Matrix-Shape header is required", status=500)
    
    encrypted_matrix_shape:tuple = eval(_encrypted_matrix_shape)

    encryptedShiftMatrixId = "{}-EncryptedShiftMatrix".format(plainTextMatrixID) #Build the id of Encrypted Shift Matrix
    Cent_iId               = "{}-Cent_i".format(plainTextMatrixID) #Build the id of Cent_i
    Cent_jId               = "{}-Cent_j".format(plainTextMatrixID) #Build the id of Cent_j
    num_chunks             = int(requestHeaders.get("Num-Chunks",-1))
    skmeans                = SKMeans()
    responseHeaders        = {}
    logger.debug("Worker starts SKMEANS_1 process -> {}".format(plainTextMatrixID))

    if num_chunks == -1:
        return Response("Num-Chunks header is required", status=503)
    try:
        logger.debug("get encrypted matrix {}".format(encryptedMatrixId))
        responseHeaders["Start-Time"] = str(arrivalTime)
        # encryptedMatrix_response = STORAGE_CLIENT.get_and_merge_ndarray(key=encryptedMatrixId)
        # x                        = encryptedMatrix_response.result()
        # encryptedMatrix_response:GetNDArrayResponse = x.unwrap()
        # encryptedMatrix                             = encryptedMatrix_response.value
        (encryptedMatrix, encrypted_matrix_metadata) = get_and_merge_ndarray(
            STORAGE_CLIENT = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID, 
            key            = encryptedMatrixId,
            num_chunks     = num_chunks, 
            shape          = encrypted_matrix_shape,
            dtype          = _encrypted_matrix_dtype,
            logger=logger
        )

        responseHeaders["Encrypted-Matrix-Dtype"] = encrypted_matrix_metadata.tags.get("dtype",encryptedMatrix.dtype) #["tags"]["dtype"] #Save the data type
        responseHeaders["Encrypted-Matrix-Shape"] = encrypted_matrix_metadata.tags.get("shape",encryptedMatrix.shape) #Save the shape
        logger.debug("ENCRYPTED_MATRIX GET SUCCESSFULLY")

        udm_matrix_response = Segmentation.get_matrix_or_error(
            client    = STORAGE_CLIENT,
            key       = UDMId,
            bucket_id = BUCKET_ID
        )
                
        UDMMatrix                           = udm_matrix_response.value
        responseHeaders["Udm-Matrix-Dtype"] = udm_matrix_response.metadata.tags.get("dtype",UDMMatrix.dtype) # Extract the type
        responseHeaders["Udm-Matrix-Shape"] = udm_matrix_response.metadata.tags.get("shape",UDMMatrix.shape) # Extract the shape
        logger.debug("ENCRYPTED_UDM_MATRIX GET SUCCESSFULLY")

        if(isStartStatus): #if the status is start
            __Cent_j = None #There is no Cent_j
        else: 
            Cent_j_response = Segmentation.get_matrix_or_error(
                client    = STORAGE_CLIENT,
                key       = Cent_iId,
                bucket_id = BUCKET_ID
            )
            __Cent_j        = Cent_j_response.value
            status          = Constants.ClusteringStatus.WORK_IN_PROGRESS
            logger.debug("CLIENT_J GET SUCCESSFULLY")

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
        logger.debug("CLIENT_I PUT SUCCESSFULLY")

        y = STORAGE_CLIENT.put_ndarray( # Saving Cent_j to storage
            key       = Cent_jId, 
            ndarray   = np.array(Cent_j),
            tags      = {},
            bucket_id = BUCKET_ID
        )
        logger.debug("CLIENT_J PUT SUCCESSFULLY")

        z = STORAGE_CLIENT.put_ndarray( # Saving S1 matrix to storage
            key       = encryptedShiftMatrixId, 
            ndarray   = np.array(S1),
            tags      = {},
            bucket_id = BUCKET_ID
        )
        logger.debug("ENCRYPTED_SHIFT_MATRIX_ID PUT SUCCESSFULLY")
        
        endTime                                      = time.time()
        serviceTime                                  = endTime - arrivalTime
        responseHeaders["Service-Time"]              = str(serviceTime)
        responseHeaders["Iterations"]                = str(int(requestHeaders.get("Iterations",0)) + 1) #Saves the number of iterations in the header
        responseHeaders["Encrypted-Shift-Matrix-Id"] = encryptedShiftMatrixId #Save the id of the encrypted shift matrix    
        
        x = x.result()
        y = y.result()
        z = z.result() 

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
        logger.error("WORKER_SKMEANS_1_ERROR "+encryptedMatrixId+" "+str(e))
        return Response(str(e),status = 503)


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
    algorithm               = Constants.ClusteringAlgorithms.SKMEANS
    status                  = int(requestHeaders.get("Clustering-Status",Constants.ClusteringStatus.START))
    k                       = int(requestHeaders.get("K",3))
    m                       = int(requestHeaders.get("M",3))
    num_chunks              = int(requestHeaders.get("NUM_CHUNKS",4))
    iterations              = int(requestHeaders.get("Iterations",0))
    encryptedMatrixId       = requestHeaders["Encrypted-Matrix-Id"]
    plainTextMatrixID       = requestHeaders["Plaintext-Matrix-Id"]
    shiftMatrixId           = requestHeaders.get("Shift-Matrix-Id","{}-ShiftMatrix".format(plainTextMatrixID))
    if encryptedMatrixId == -1 or plainTextMatrixID == -1:
        return Response("Either Encrypted-Matrix-Id or Plain-Matrix-Id is missing",status=500)
    UDM_id                       = "{}-UDM".format(plainTextMatrixID)
    Cent_iId                     = "{}-Cent_i".format(plainTextMatrixID) #Build the id of Cent_i
    Cent_jId                     = "{}-Cent_j".format(plainTextMatrixID) #Build the id of Cent_j
    responseHeaders              = {}
    start_time                   = requestHeaders.get("Start-Time","0.0")
    logger.debug("Worker starts SKMEANS_2 process -> {}".format(plainTextMatrixID))
    try:
        
        UDM_put_future:Awaitable[Result[GetNDArrayResponse,Exception]] =  STORAGE_CLIENT.get_ndarray(key=UDM_id)
        #logger.debug("UDM_FUTURE {}".format(UDM_put_future))
        UDM_result:Result[GetNDArrayResponse,Exception] = UDM_put_future.result()
        #logger.debug("UDM_RESPONSE {}".format(UDM_result))
        if UDM_result.is_err:
            return Response(None, status=500, headers={"Error-Message":str(UDM_result.unwrap_err())})
        UDM_response:GetNDArrayResponse = UDM_result.unwrap()
        UDM             = UDM_response.value
        
        Cent_i_response = Segmentation.get_matrix_or_error(
            client    = STORAGE_CLIENT,
            key       = Cent_iId,
            bucket_id = BUCKET_ID
        )
        Cent_i          = Cent_i_response.value
        Cent_j_response = Segmentation.get_matrix_or_error(
            client    = STORAGE_CLIENT,
            key       = Cent_jId,
            bucket_id = BUCKET_ID
        )
        Cent_j                   = Cent_j_response.value

        shiftMatrix_get_response = Segmentation.get_matrix_or_error(
            client    = STORAGE_CLIENT,
            key       = shiftMatrixId,
            bucket_id = BUCKET_ID
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
            responseHeaders["Clustering-Status"] = status #The status is changed to WORK IN PROGRESS
            attibutes_shape = eval(requestHeaders["Encrypted-Matrix-Shape"]) # extract the attributes of shape
            _UDM            = skmeans.run_2( # The second part of the skmeans starts
                k           = k,
                UDM         = UDM,
                attributes  = int(attibutes_shape[1]),
                shiftMatrix = shiftMatrix,
            )
            UDM_array = np.array(_UDM)
            logger.debug("_UDMMATRIX SHAPE{}".format(UDM_array.shape))

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
            return Response( #Return none and headers
                response = None,
                status   = 204, 
                headers  = responseHeaders
            )
    except Exception as e:
        logger.error("SKMEANS_2_ERROR: "+encryptedMatrixId+" "+str(e))
        return Response(str(e),status = 503)

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
            client    = STORAGE_CLIENT,
            key       = plainTextMatrixId,
            bucket_id = BUCKET_ID
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
    arrivalTime              = time.time() #System startup time
    logger                   = current_app.config["logger"]
    workerId                 = current_app.config["NODE_ID"] # Get the node_id from the global configuration
    STORAGE_CLIENT:V4Client  = current_app.config["STORAGE_CLIENT"]
    BUCKET_ID:str            = current_app.config.get("BUCKET_ID","rory")
    status                   = int(requestHeaders.get("Clustering-Status", Constants.ClusteringStatus.START)) 
    isStartStatus            = status == Constants.ClusteringStatus.START #if status is start save it to isStartStatus
    k                        = int(requestHeaders.get("K",3)) # It is passed to integer because the headers are strings
    m                        = int(requestHeaders.get("M",3))
    algorithm                = Constants.ClusteringAlgorithms.DBSKMEANS
    plainTextMatrixID        = requestHeaders.get("Plaintext-Matrix-Id")
    encryptedMatrixId        = requestHeaders.get("Encrypted-Matrix-Id","")
    _encrypted_matrix_shape  = requestHeaders.get("Encrypted-Matrix-Shape",-1)
    _encrypted_matrix_dtype  = requestHeaders.get("Encrypted-Matrix-Dtype",-1)

    _encrypted_udm_shape     = requestHeaders.get("Encrypted-Udm-Shape",-1)
    _encrypted_udm_dtype     = requestHeaders.get("Encrypted-Udm-Dtype",-1)
    iterations               = int(requestHeaders.get("Iterations",0))
    logger.debug(str(requestHeaders))

    logger.debug("DBSKMEANS_1 algorithm={}, m={}, k={}, plain_matrix_id={}, ems={}, emd={}, eus={}, eud={}".format(algorithm,m,k,plainTextMatrixID,_encrypted_matrix_shape,_encrypted_matrix_dtype,_encrypted_udm_shape,_encrypted_udm_dtype))
    if _encrypted_matrix_dtype == -1:
        return Response("Encrypted-Matrix-Dtype", status=500)
    if _encrypted_matrix_shape == -1 :
        return Response("Encrypted-Matrix-Shape header is required", status=500)

    if _encrypted_udm_dtype == -1:
        return Response("Encrypted-UDM-Dtype", status=500)
    if _encrypted_udm_shape == -1 :
        return Response("Encrypted-UDM-Shape header is required", status=500)
    
    encrypted_matrix_shape:tuple = eval(_encrypted_matrix_shape)
    encrypted_udm_shape:tuple    = eval(_encrypted_udm_shape)
    
    encryptedUdmId          = "{}-encrypted-UDM".format(plainTextMatrixID) 
    Cent_iId                = "{}-Cent_i".format(plainTextMatrixID) #Build the id of Cent_i
    Cent_jId                = "{}-Cent_j".format(plainTextMatrixID) #Build the id of Cent_j
    encryptedShiftMatrixId  = "{}-EncryptedShiftMatrix".format(plainTextMatrixID) #Build the id of Encrypted Shift Matrix
    num_chunks              = int(requestHeaders.get("Num-Chunks",-1))
    dbskmeans               = DBSKMeans()
    responseHeaders         = {}
    logger.debug("Worker starts DBSKMEANS_1 process -> {}".format(plainTextMatrixID))


    if num_chunks == -1:
        return Response("Num-Chunks header is required", status=503)
    try:
        logger.debug("get encrypted matrix {}".format(encryptedMatrixId))
        responseHeaders["Start-Time"]               = str(arrivalTime)
        # encryptedMatrix_response                    = STORAGE_CLIENT.get_and_merge_ndarray(key=encryptedMatrixId)
        # encryptedMatrix_result:Result[GetBytesResponse,Exception]                    = STORAGE_CLIENT.get_and_merge_with_num_chunks(bucket_id=BUCKET_ID,key=encryptedMatrixId,num_chunks=num_chunks).result()
        # if encryptedMatrix_result.is_err:
            # return Response("Get encrypted matrix error: {}".format(encryptedMatrix_result.unwrap_err()), 500 )
        # if en

            # return Ok()
        # encryptedMatrix_result:GetBytesResponse = x.unwrap() 
        # Deserialize bytes to NDMAtrix with shape encrypted_matrix_shape
        # encryptedMatrix                         = np.frombuffer(encryptedMatrix_result.value,dtype=_encrypted_matrix_dtype).reshape(encrypted_matrix_shape)
        # encrypted_matrix_metadata:Metadata      =  encryptedMatrix_result.metadata
        (encryptedMatrix, encrypted_matrix_metadata) = get_and_merge_ndarray(
            STORAGE_CLIENT = STORAGE_CLIENT,
            # Warning this is a bad practice and it must be change as soon as possible.....
            bucket_id      = "{}-0".format(BUCKET_ID), 
            # _______________________________________________________________________________
            key            = encryptedMatrixId,
            num_chunks     = num_chunks, 
            shape          = encrypted_matrix_shape,
            dtype          = _encrypted_matrix_dtype,
            logger=logger
            
        )
        logger.debug("ENCRYPTED_MATRIX GET AND MERGE SUCCESSFULLY")

        responseHeaders["Encrypted-Matrix-Dtype"] = encrypted_matrix_metadata.tags.get("dtype",encryptedMatrix.dtype) #Save the data type
        responseHeaders["Encrypted-Matrix-Shape"] = encrypted_matrix_metadata.tags.get("shape",encryptedMatrix.shape) #Save the shape
        # 
        # logger.debug("get udm before {}".format(encryptedUdmId))
        #logger.debug("ENCRYPTED_MATRIX GET SUCCESSFULLY")
        
        # udm_matrix_response:Result[GetNDArrayResponse,Exception] = STORAGE_CLIENT.get_and_merge_ndarray(key=encryptedUdmId)
        # udm_matrix_response:Result[GetNDArrayResponse,Exception] = STORAGE_CLIENT.get_and_merge_with_num_chunks(bucket_id=BUCKET_ID, key=encryptedUdmId, num_chunks=num_chunks)
        # y                                      = udm_matrix_response.result()
        # udm_matrix_response:GetNDArrayResponse = y.unwrap()
        # UDMMatrix                              = udm_matrix_response.value
        # udm_bucket_id = "{}-{}".format(BUCKET_ID,0) if iterations == 0  else  "{}-{}".format(BUCKET_ID,iterations)

        (UDMMatrix, udm_metadata) = get_and_merge_ndarray(
            STORAGE_CLIENT = STORAGE_CLIENT,
            bucket_id      = "{}-{}".format(BUCKET_ID,iterations),
            key            = encryptedUdmId,
            num_chunks     = num_chunks,
            shape          = encrypted_udm_shape,
            dtype          = _encrypted_udm_dtype,
            logger=logger
        )

        responseHeaders["Encrypted-Udm-Dtype"]    = str(udm_metadata.tags.get("dtype",UDMMatrix.dtype)) # Extract the type
        responseHeaders["Encrypted-Udm-Shape"]    = str(udm_metadata.tags.get("shape",UDMMatrix.shape)) # Extract the shape
        logger.debug("ENCRYPTED_UDM_MATRIX GET AND MERGE SUCCESSFULLY")

        if(isStartStatus): #if the status is start
            __Cent_j = None #There is no Cent_j

        else: 
            Cent_j_response = Segmentation.get_matrix_or_error(
                bucket_id = "{}-{}".format(BUCKET_ID,iterations-1),
                client    = STORAGE_CLIENT,
                key       = Cent_iId
            )
            __Cent_j          = Cent_j_response.value
            status = Constants.ClusteringStatus.WORK_IN_PROGRESS
            logger.debug("CLIENT_J GET SUCCESSFULLY")
            #logger.debug("CENT_J {}".format(__Cent_j.shape))
        
        #logger.debug("ENCRYPTED_MATRIX_SHAPE {}".format(encrypted_matrix_shape))
        #logger.debug("UDM_SHAPE {}".format(UDMMatrix.shape))
        

        S1,Cent_i,Cent_j,label_vector = dbskmeans.run1(
            status           = status,
            k                = k,
            m                = m,
            encryptedMatrix  = encryptedMatrix,
            UDM              = UDMMatrix,
            Cent_j           = __Cent_j
        ) 
        logger.debug("RUN_1 COMPLETED SUCCESSFULLY")

        x = STORAGE_CLIENT.put_ndarray( # Saving Cent_i to storage
            key       = Cent_iId, 
            ndarray   = np.array(Cent_i),
            tags      = {},
            # bucket_id = BUCKET_ID
            bucket_id = "{}-{}".format(BUCKET_ID,iterations),
        )
        logger.debug("CLIENT_I PUT SUCCESSFULLY")

        y = STORAGE_CLIENT.put_ndarray( # Saving Cent_j to storage
            key       = Cent_jId, 
            ndarray   = np.array(Cent_j),
            tags      = {},
            # bucket_id = BUCKET_ID
            bucket_id = "{}-{}".format(BUCKET_ID,iterations),
        )
        logger.debug("CLIENT_J PUT SUCCESSFULLY")

        z = STORAGE_CLIENT.put_ndarray( # Saving S1 matrix to storage
            key       = encryptedShiftMatrixId,  
            ndarray   = np.array(S1),
            tags      = {},
            # bucket_id = BUCKET_ID
            bucket_id = "{}-{}".format(BUCKET_ID,iterations),
        )
        logger.debug("ENCRYPTED_SHIFT_MATRIX PUT SUCCESSFULLY")
        #logger.debug("ENCRYPTED_SHIFT_MATRIX SHAPE {}".format(np.array(S1).shape))
        
        endTime                                      = time.time()
        serviceTime                                  = endTime - arrivalTime
        responseHeaders["Service-Time"]              = str(serviceTime)
        responseHeaders["Iterations"]                = str(int(requestHeaders.get("Iterations",0)) + 1) #Saves the number of iterations in the header
        responseHeaders["Encrypted-Shift-Matrix-Id"] = encryptedShiftMatrixId #Save the id of the encrypted shift matrix

        x = x.result()
        y = y.result()
        z = z.result() 
        #print(x,y,z)
        
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
        
        logger.debug("_"*100)
        return Response( #Returns the final response as a label vector + the headers
            response = json.dumps({"labelVector":label_vector }),
            status   = 200,
            headers  = {**requestHeaders, **responseHeaders}
        )
    except Exception as e:
        # print("ERROR {}".format(e))
        logger.error("DBSKMEANS_1_ERROR: "+encryptedMatrixId+" "+str(e) )
        return Response(str(e),status = 503)

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
    k                       = int(requestHeaders.get("K",3))
    m                       = int(requestHeaders.get("M",3))
    num_chunks              = int(requestHeaders.get("NUM_CHUNKS",4))
    iterations              = int(requestHeaders.get("Iterations",0))
    start_time              = requestHeaders.get("Start-Time","0.0")
    encryptedMatrixId       = requestHeaders.get("Encrypted-Matrix-Id",-1)
    plainTextMatrixID       = requestHeaders.get("Plaintext-Matrix-Id",-1)
    if encryptedMatrixId == -1 or plainTextMatrixID == -1:
        return Response("Either Encrypted-Matrix-Id or Plain-Matrix-Id is missing",status=500)
    
    shiftMatrixId           = requestHeaders.get("Shift-Matrix-Id","{}-ShiftMatrix".format(plainTextMatrixID))
    shiftMatrixOpeId        = requestHeaders.get("Shift-Matrix-Ope-Id","{}-ShiftMatrixOpe".format(plainTextMatrixID))
    
    # encryptedMatrixId       = requestHeaders.get("Encrypted-Matrix-Id","")
    _encrypted_matrix_shape  = requestHeaders.get("Encrypted-Matrix-Shape",-1)
    _encrypted_matrix_dtype  = requestHeaders.get("Encrypted-Matrix-Dtype",-1)

    _encrypted_udm_shape     = requestHeaders.get("Encrypted-Udm-Shape",-1)
    _encrypted_udm_dtype     = requestHeaders.get("Encrypted-Udm-Dtype",-1)
    # iterations               = int(requestHeaders.get("Iterations",0))

    logger.debug("DBSKMEANS_2 algorithm={}, m={}, k={}, plain_matrix_id={}, ems={}, emd={}, eus={}, eud={}".format(
        algorithm, m, k, plainTextMatrixID, _encrypted_matrix_shape, _encrypted_matrix_dtype, _encrypted_udm_shape, _encrypted_udm_dtype))
    
    if _encrypted_matrix_dtype == -1:
        return Response("Encrypted-Matrix-Dtype", status=500)
    if _encrypted_matrix_shape == -1 :
        return Response("Encrypted-Matrix-Shape header is required", status=500)

    if _encrypted_udm_dtype == -1:
        return Response("Encrypted-UDM-Dtype", status=500)
    if _encrypted_udm_shape == -1 :
        return Response("Encrypted-UDM-Shape header is required", status=500)


    encrypted_matrix_shape:tuple = eval(_encrypted_matrix_shape)
    encrypted_udm_shape:tuple    = eval(_encrypted_udm_shape)
    UDM_id                       = "{}-encrypted-UDM".format(plainTextMatrixID)
    Cent_iId                     = "{}-Cent_i".format(plainTextMatrixID) #Build the id of Cent_i
    Cent_jId                     = "{}-Cent_j".format(plainTextMatrixID) #Build the id of Cent_j
    responseHeaders              = {}
    logger.debug("Worker starts DBSKMEANS_2 process -> {}".format(plainTextMatrixID))
    try:
        (UDMMatrix, udm_metadata) = get_and_merge_ndarray(
            STORAGE_CLIENT = STORAGE_CLIENT,
            # Warning this is a bad practice and it must be change as soon as possible.....
            bucket_id      = "{}-{}".format(BUCKET_ID,iterations), 
            # _______________________________________________________________________________
            key            = UDM_id,
            num_chunks     = num_chunks, 
            shape          = encrypted_udm_shape,
            dtype          = _encrypted_udm_dtype,
            logger=logger
        )
        responseHeaders["Encrypted-Udm-Dtype"] = str(udm_metadata.tags.get("dtype",UDMMatrix.dtype)) # Extract the type
        responseHeaders["Encrypted-Udm-Shape"] = str(udm_metadata.tags.get("shape",UDMMatrix.shape)) # Extract the shape
        logger.debug("ENCRYPTED_UDM_MATRIX GET AND MERGE SUCCESSFULLY")

        #logger.debug(str(encrypted_matrix_metadata))
        # logger.debug("_"*50)
        # UDM_response:Result[GetNDArrayResponse,Exception] = STORAGE_CLIENT.get_and_merge_ndarray(key=UDM_id)
        # y                               = UDM_response.result()
        # UDM_response:GetNDArrayResponse = y.unwrap()``
        # UDM                             = UDM_response.value
        
        Cent_i_response = Segmentation.get_matrix_or_error(
            bucket_id = "{}-{}".format(BUCKET_ID,iterations),
            client    = STORAGE_CLIENT,
            key       = Cent_iId
        )
        Cent_i          = Cent_i_response.value

        logger.debug("CENT_I GET SUCCESSFULLY")
        Cent_j_response = Segmentation.get_matrix_or_error(
            bucket_id = "{}-{}".format(BUCKET_ID,iterations),
            client    = STORAGE_CLIENT,
            key       = Cent_jId
        ) 
        Cent_j  = Cent_j_response.value
        logger.debug("CENT_J GET SUCCESSFULLY")

        # shiftMatrix_get_response = Segmentation.get_matrix_or_error(
        #     client = STORAGE_CLIENT,
        #     key    = shiftMatrixId
        # )
        # shiftMatrix = shiftMatrix_get_response.value
        
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
            # attibutes_s/hape         = eval(requestHeaders["Encrypted-Matrix-Shape"]) # extract the attributes of shape
            #print("Get shiftMatrix")
            shiftMatrixOpe_response = Segmentation.get_matrix_or_error(
                bucket_id = "{}-{}".format(BUCKET_ID,iterations),
                client    = STORAGE_CLIENT,
                key       = shiftMatrixOpeId
            )
            shiftMatrixOpe:npt.NDArray = shiftMatrixOpe_response.value

            logger.debug("SHIFT_MATRIX_OPE GET SUCCESSFULLY")
            #print("UDM run 2")
            udm_start_time = time.time()
            logger.debug("ENCRYPTED_MATRIX_SHAPE {}".format(encrypted_matrix_shape))
            logger.debug("ENCRYPTED_UDM_SHAPE {}".format(UDMMatrix.shape))
            logger.debug("SHIF_MATRIX_OPE {}".format(shiftMatrixOpe.shape))
            # logger.debug("")
            _UDMMatrix     = dbskmeans.run_2( # The second part of the skmeans starts
                k                = k,
                UDM              = UDMMatrix,
                attributes       = int(encrypted_matrix_shape[1]),
                shiftMatrix      = shiftMatrixOpe,
            )
            
            responseHeaders["Encrypted-Udm-Shape"] = str(_UDMMatrix.shape)
            logger.debug("_UDMMATRIX SHAPE{}".format(_UDMMatrix.shape))
            logger.debug("RUN_2 COMPLETED SUCCESSFULLY")

            #print("Put UDM")
            # UDM_array = np.array(_UDMMatrix)
            # _ = STORAGE_CLIENT.put_ndarray(
            #     key       = UDM_id, 
            #     ndarray   = _UDMMatrix,
            #     tags      = {},
            #     bucket_id = BUCKET_ID
            # ).result() # UDM is extracted from the storage system
            
            maybe_UDM_chunks:Option[Chunks] = Chunks.from_ndarray(
                ndarray    = _UDMMatrix,
                group_id   = UDM_id,
                num_chunks = num_chunks,
                chunk_prefix = Some(UDM_id)
            )
            logger.debug("CHUNKS_FROM_NDARRAY COMPLETED SUCCESSFULLY")

            if maybe_UDM_chunks.is_none:
                logger.error("Something went wrong segment encrypted-UDM.")
                return Response(
                    status   = 500,
                    response = "{}".format(str(maybe_UDM_chunks.unwrap_err()))
                )
            
            UDM_chunks = maybe_UDM_chunks.unwrap()
            logger.debug("UDM_CHUNKS_LEN: {}".format(len(UDM_chunks.chunks)))
            
            next_bucket_id = "{}-{}".format(BUCKET_ID,iterations+1)
            put_chunks_generator_results = STORAGE_CLIENT.put_chunks(
                bucket_id= next_bucket_id,
                key       = UDM_id, 
                chunks    = UDM_chunks, 
                tags      = {}
            )
            
            logger.debug("ENCRYPTED_UDM_MATRIX PUT SUCCESSFULLY: bucket_id={} ball_id={}".format(next_bucket_id,UDM_id))

            for i,put_chunk_result in enumerate(put_chunks_generator_results):
                udm_end_time = time.time()
                udm_time     = udm_end_time - udm_start_time
                encrypt_logger_metrics = LoggerMetrics( #Write times of encrypt in logger
                    operation_type = "GENERATION_AND_ENCRYPT_UDM_CHUNK",
                    matrix_id      = plainTextMatrixID,
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
                end_time       = end_time, 
                service_time   = service_time,
                m_value        = m,
                k_value        = k,
                n_iterations   = iterations
            )
            logger.info(str(logger_metrics))
            logger.debug("_"*50)
            
            return Response( #Return none and headers
                response = None,
                status   = 204, 
                headers  = { **responseHeaders}
            )
    except Exception as e:
        logger.error("DBSKMEANS_2_ERROR: "+encryptedMatrixId+" "+str(e) )
        return Response(str(e),status = 503)

"""
Description:
    DBSKMEANS algorithm
"""
@clustering.route("/dbskmeans", methods = ["POST"])
def dbskmeans():
    headers         = request.headers
    head            = ["User-Agent","Accept-Encoding","Connection"]
    logger                  = current_app.config["logger"]
    filteredHeaders = dict(list(filter(lambda x: not x[0] in head, headers.items())))
    #logger.debug("DBSKMEANS "+str(filteredHeaders))
    
    step_index      = int(filteredHeaders.get("Step-Index",1))
    response        = Response()
    #logger.debug("STEP_INDEX {}".format(step_index))
    
    if step_index == 1:
        return dbskmeans_1(filteredHeaders)
    elif step_index == 2:
        return dbskmeans_2(filteredHeaders)
    else:
        return response


@clustering.route("/dbsnnc", methods = ["POST"])
def dbsnnc():
    arrivalTime             = time.time() #System startup time
    headers                 = request.headers
    to_remove_headers       = ["User-Agent","Accept-Encoding","Connection"]
    filteredHeaders         = dict(list(filter(lambda x: not x[0] in to_remove_headers, headers.items())))
    algorithm               = Constants.ClusteringAlgorithms.DBSNNC
    logger                  = current_app.config["logger"]
    STORAGE_CLIENT:Client   = current_app.config["STORAGE_CLIENT"]
    BUCKET_ID:str           = current_app.config.get("BUCKET_ID","rory")
    m                       = int(filteredHeaders.get("M",3))
    workerId                = current_app.config["NODE_ID"] # Get the node_id from the global configuration
    plainTextMatrixID       = filteredHeaders.get("Plaintext-Matrix-Id")
    encryptedMatrixId       = filteredHeaders.get("Encrypted-Matrix-Id",-1)
    encrypted_threshold     = float(filteredHeaders.get("Encrypted-Threshold"))
    dm_id                   = "{}-DM".format(plainTextMatrixID) 
    _encrypted_matrix_shape = filteredHeaders.get("Encrypted-Matrix-Shape",-1)
    _encrypted_matrix_dtype = filteredHeaders.get("Encrypted-Matrix-Dtype",-1)
    _encrypted_dm_shape     = filteredHeaders.get("Encrypted-Dm-Shape",-1)
    _encrypted_dm_dtype     = filteredHeaders.get("Encrypted-Dm-Dtype",-1)

    logger.debug(str(filteredHeaders))
    logger.debug("DBSNNC algorithm={}, m={}, plain_matrix_id={}, ems={}, emd={}, eus={}, eud={}".format(algorithm,m,plainTextMatrixID,_encrypted_matrix_shape,_encrypted_matrix_dtype, _encrypted_dm_shape,_encrypted_dm_dtype))

    if _encrypted_matrix_dtype == -1:
        return Response("Encrypted-Matrix-Dtype", status=500)
    if _encrypted_matrix_shape == -1 :
        return Response("Encrypted-Matrix-Shape header is required", status=500)
    
    if _encrypted_dm_dtype == -1:
        return Response("Encrypted-DM-Dtype", status=500)
    if _encrypted_dm_shape == -1 :
        return Response("Encrypted-DM-Shape header is required", status=500)
    
    encrypted_matrix_shape:tuple = eval(_encrypted_matrix_shape)
    encrypted_dm_shape:tuple     = eval(_encrypted_dm_shape)

    encryptedDmId   = "{}-encrypted-DM".format(plainTextMatrixID) 
    num_chunks      = int(filteredHeaders.get("Num-Chunks",-1))
    responseHeaders = {}
    logger.debug("Worker starts DBSNNC process -> {}".format(plainTextMatrixID))

    if num_chunks == -1:
        return Response("Num-Chunks header is required", status=503)
    
    try:      
        logger.debug("get encrypted matrix {}".format(encryptedMatrixId))
        responseHeaders["Start-Time"] = str(arrivalTime)
        
        logger.debug("Encrypted Matrix Id {}".format(encryptedMatrixId))
        logger.debug("Encrypted Matrix Shape {}".format(encrypted_matrix_shape))
        logger.debug("Encrypted Matrix Dtype {}".format(_encrypted_matrix_dtype))
        logger.debug("Bucket_id {}".format(BUCKET_ID))

        (encryptedMatrix, encrypted_matrix_metadata) = get_and_merge_ndarray(
            STORAGE_CLIENT = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID, 
            key            = encryptedMatrixId,
            num_chunks     = num_chunks, 
            shape          = encrypted_matrix_shape,
            dtype          = _encrypted_matrix_dtype,
            logger         = logger
        )

        responseHeaders["Encrypted-Matrix-Dtype"] = encrypted_matrix_metadata.tags.get("dtype",encryptedMatrix.dtype) #["tags"]["dtype"] #Save the data type
        responseHeaders["Encrypted-Matrix-Shape"] = encrypted_matrix_metadata.tags.get("shape",encryptedMatrix.shape) #Save the shape
        logger.debug("ENCRYPTED_MATRIX GET AND MERGE SUCCESSFULLY")

        # dm_matrix_response = Segmentation.get_matrix_or_error(
        #     client    = STORAGE_CLIENT,
        #     key       = dm_id,
        #     bucket_id = BUCKET_ID
        # )
        logger.debug("Encrypted Dm Id {}".format(encryptedDmId))
        logger.debug("Encrypted Dm Shape {}".format(encrypted_dm_shape))
        logger.debug("Encrypted Dm Dtype {}".format(_encrypted_dm_dtype))
        logger.debug("Bucket_id {}".format(BUCKET_ID))
        
        (DMMatrix, dm_metadata) = get_and_merge_ndarray(
            STORAGE_CLIENT = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = encryptedDmId,
            num_chunks     = num_chunks,
            shape          = encrypted_dm_shape,
            dtype          = _encrypted_dm_dtype,
            logger         = logger
        )

        responseHeaders["Encrypted-Dm-Dtype"] = str(dm_metadata.tags.get("dtype",DMMatrix.dtype)) # Extract the type
        responseHeaders["Encrypted-Dm-Shape"] = str(dm_metadata.tags.get("shape",DMMatrix.shape)) # Extract the shape
        logger.debug("ENCRYPTED_UDM_MATRIX GET AND MERGE SUCCESSFULLY")

        
        result = Dbsnnc.run(
            distance_matrix     = DMMatrix,
            encrypted_threshold = encrypted_threshold
        )
        endTime                         = time.time()
        serviceTime                     = endTime - arrivalTime
        responseHeaders["Service-Time"] = str(serviceTime)
        
        logger_metrics = LoggerMetrics(
            operation_type = algorithm,
            matrix_id      = plainTextMatrixID,
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
        logger.error(e)
        return Response(
            response = None,
            status   = 503,
            headers  = {"Error-Message":e})
    
@clustering.route("/nnc", methods = ["POST"])
def nnc():
    arrivalTime             = time.time() #System startup time
    headers                 = request.headers
    to_remove_headers       = ["User-Agent","Accept-Encoding","Connection"]
    filteredHeaders         = dict(list(filter(lambda x: not x[0] in to_remove_headers, headers.items())))
    algorithm               = Constants.ClusteringAlgorithms.NNC
    logger                  = current_app.config["logger"]
    STORAGE_CLIENT:Client   = current_app.config["STORAGE_CLIENT"]
    BUCKET_ID:str           = current_app.config.get("BUCKET_ID","rory")
    workerId                = current_app.config["NODE_ID"] # Get the node_id from the global configuration
    plainTextMatrixId       = filteredHeaders.get("Plaintext-Matrix-Id")
    threshold               = float(filteredHeaders.get("Threshold"))
    dm_id                   = "{}-DM".format(plainTextMatrixId) 
    responseHeaders         = {}

    logger.debug(str(filteredHeaders))
    logger.debug("NNC algorithm={}, plain_matrix_id={}".format(algorithm,plainTextMatrixId))

    responseHeaders = {}
    logger.debug("Worker starts NNC process -> {}".format(plainTextMatrixId))

    try:      
        logger.debug("get plaintext matrix {}".format(plainTextMatrixId))
        responseHeaders["Start-Time"] = str(arrivalTime)

        plainTextMatrix_response = Segmentation.get_matrix_or_error(
            client    = STORAGE_CLIENT,
            key       = plainTextMatrixId,
            bucket_id = BUCKET_ID
        ) 
        plainTextMatrix                           = plainTextMatrix_response.value
        responseHeaders["Plaintext-Matrix-Dtype"] = plainTextMatrix_response.metadata.tags.get("dtype",plainTextMatrix.dtype) #["tags"]["dtype"] #Save the data type
        responseHeaders["Plaintext-Matrix-Shape"] = plainTextMatrix_response.metadata.tags.get("shape",plainTextMatrix.shape) #Save the shape
        
        logger.debug("PLAINTEXT_MATRIX GET SUCCESSFULLY")

        endTime        = time.time()
        serviceTime    = endTime - arrivalTime

        dm_matrix_response = Segmentation.get_matrix_or_error(
            client    = STORAGE_CLIENT,
            key       = dm_id,
            bucket_id = BUCKET_ID
        )

        DMMatrix                           = dm_matrix_response.value
        responseHeaders["Dm-Matrix-Dtype"] = dm_matrix_response.metadata.tags.get("dtype",DMMatrix.dtype) # Extract the type
        responseHeaders["Dm-Matrix-Shape"] = dm_matrix_response.metadata.tags.get("shape",DMMatrix.shape) # Extract the shape
        logger.debug("ENCRYPTED_DM_MATRIX GET SUCCESSFULLY")
        
        result = Nnc.run(
            distance_matrix = DMMatrix,
            threshold       = threshold
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
        logger.error(e)
        return Response(
            response = None,
            status   = 503,
            headers  = {"Error-Message":e})