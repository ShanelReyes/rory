import time, json
import numpy as np
import numpy.typing as npt
from typing import Awaitable,List,Tuple
from flask import Blueprint,current_app,request,Response
from rory.core.clustering.kmeans import kmeans as kMeans
from rory.core.clustering.secure.local.dbsnnc import Dbsnnc
from rory.core.clustering.nnc import Nnc
from rory.core.utils.utils import Utils
from rory.core.utils.constants import Constants
from rory.core.clustering.secure.distributed.skmeans import SKMeans
from rory.core.clustering.secure.distributed.dbskmeans import DBSKMeans
from mictlanx.v3.client import Client 
from mictlanx.v4.client import Client as V4Client
from option import Result
from mictlanx.utils.segmentation import Chunks
from mictlanx.v4.interfaces.responses import GetNDArrayResponse,PutResponse
from option import Option,Some,NONE
from utils.utils import Utils as LocalUtils

# 
clustering = Blueprint("clustering",__name__,url_prefix = "/clustering")



def delete_and_put(STORAGE_CLIENT:V4Client,bucket_id:str,key:str,ndarray:npt.NDArray):
    _x = STORAGE_CLIENT.delete_by_ball_id(ball_id=key, bucket_id=bucket_id)
    return STORAGE_CLIENT.put_ndarray( # Saving Cent_i to storage
        key       = key, 
        ndarray   = ndarray,
        tags      = {},
        bucket_id = bucket_id
    )


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
    arrival_time            = time.time() #Worker start time
    # Config
    logger                  = current_app.config["logger"]
    worker_id               = current_app.config["NODE_ID"] # Get the node_id from the global configuration
    STORAGE_CLIENT:V4Client = current_app.config["STORAGE_CLIENT"]
    BUCKET_ID:str           = current_app.config.get("BUCKET_ID","rory")
    # Headers
    status                  = int(requestHeaders.get("Clustering-Status", Constants.ClusteringStatus.START)) 
    is_start_status         = status == Constants.ClusteringStatus.START #if status is start save it to isStartStatus
    k                       = int(requestHeaders.get("K",3)) # It is passed to integer because the headers are strings
    m                       = int(requestHeaders.get("M",3))
    algorithm               = Constants.ClusteringAlgorithms.SKMEANS
    plaintext_matrix_id     = requestHeaders.get("Plaintext-Matrix-Id")
    encrypted_matrix_id     = requestHeaders.get("Encrypted-Matrix-Id",-1)
    udm_id                  = "{}-udm".format(plaintext_matrix_id) 
    _encrypted_matrix_shape = requestHeaders.get("Encrypted-Matrix-Shape",-1)
    _encrypted_matrix_dtype = requestHeaders.get("Encrypted-Matrix-Dtype",-1)

    if _encrypted_matrix_dtype == -1:
        return Response("Encrypted-Matrix-Dtype", status=500)
    if _encrypted_matrix_shape == -1 :
        return Response("Encrypted-Matrix-Shape header is required", status=500)
    
    encrypted_matrix_shape:tuple = eval(_encrypted_matrix_shape)

    encrypted_shift_matrix_id = "{}-encrypted-shift-matrix".format(plaintext_matrix_id) #Build the id of Encrypted Shift Matrix
    cent_i_id                 = "{}-cent_i".format(plaintext_matrix_id) #Build the id of Cent_i
    cent_j_id                 = "{}-cent_j".format(plaintext_matrix_id) #Build the id of Cent_j
    num_chunks                = int(requestHeaders.get("Num-Chunks",-1))
    skmeans                   = SKMeans()
    responseHeaders           = {}
    
    logger.debug({
        "event":"SKMEANS1.STARTED",
        "plaintext_matrix_id":plaintext_matrix_id,
        "encrypted_matrix_id":encrypted_matrix_id,
        "UDM_id":udm_id,
        "encrypted_matrix_shape":_encrypted_matrix_shape,
        "encrypted_matrix_dtype":_encrypted_matrix_dtype,
        "status":status,
        "is_start_status":is_start_status,
        "k":k,
        "m":m,
        "algorithm":algorithm,
        "num_chunks":num_chunks,
        "encrypted_shift_matrix_id":encrypted_shift_matrix_id,
        "Cent_i_id":cent_i_id,
        "Cent_j_id":cent_j_id,
    })
    if num_chunks == -1:
        logger.error({
            "msg":"Num-Chunks header is required"
        })
        return Response("Num-Chunks header is required", status=503)
    try:
        responseHeaders["Start-Time"] = str(arrival_time)
        logger.debug({
            "event":"BEFORE.GET.MERGE.NDARRAY",
            "bucket_id":BUCKET_ID,
            "key":encrypted_matrix_id,
            "shape":encrypted_matrix_shape,
            "dtype":_encrypted_matrix_dtype
        })
        get_merge_encrypted_matrix_start_time = time.time()
        (encryptedMatrix, encrypted_matrix_metadata) = LocalUtils.get_and_merge_ndarray(
            STORAGE_CLIENT = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID, 
            key            = encrypted_matrix_id,
            num_chunks     = num_chunks, 
            shape          = encrypted_matrix_shape,
            dtype          = _encrypted_matrix_dtype,
        )

        responseHeaders["Encrypted-Matrix-Dtype"] = encrypted_matrix_metadata.tags.get("dtype",encryptedMatrix.dtype) #["tags"]["dtype"] #Save the data type
        responseHeaders["Encrypted-Matrix-Shape"] = encrypted_matrix_metadata.tags.get("shape",encryptedMatrix.shape) #Save the shape
        get_merge_encrypted_matrix_st = time.time()- get_merge_encrypted_matrix_start_time
        logger.info({
            "event":"GET.MERGE.NDARRAY",
            "bucket_id":BUCKET_ID,
            "key":encrypted_matrix_id,
            "num_chunks":num_chunks,
            "shape":encrypted_matrix_shape,
            "dtype":_encrypted_matrix_dtype,
            "service_time":get_merge_encrypted_matrix_st
        })
        
        logger.debug({
            "event":"GET.NDARRAY.BEFORE",
            "bucket_id":BUCKET_ID,
            "key":udm_id,
        })
        udm_get_start_time = time.time()
        udm_matrix_response = LocalUtils.get_matrix_or_error(
            client    = STORAGE_CLIENT,
            bucket_id = BUCKET_ID,
            key       = udm_id,
        )
        udm_get_st = time.time() - udm_get_start_time
        udm        = udm_matrix_response.value
        logger.debug({
            "event":"GET.NDARRAY",
            "bucket_id":BUCKET_ID,
            "key":udm_id,
            "shape":str(udm.shape), 
            "dtype":str(udm.dtype),
            "service_time":udm_get_st
        })
        responseHeaders["Udm-Matrix-Dtype"] = udm_matrix_response.metadata.tags.get("dtype",udm.dtype) # Extract the type
        responseHeaders["Udm-Matrix-Shape"] = udm_matrix_response.metadata.tags.get("shape",udm.shape) # Extract the shape
        
        if is_start_status: #if the status is start
            logger.debug({
                "event":"NO.CentJ.WORKER.RUN1.BEFORE",
                "status":status,
                "k":k,
                "m":m,
                "enctypted_matrix_shape":str(encryptedMatrix.shape),
                "enctypted_matrix_dtype":str(encryptedMatrix.dtype),
                "UDM_shape":str(udm.shape),
                "UDM_dtype":str(udm.dtype),
            })
            __Cent_j = NONE #There is no Cent_j
        else: 
            logger.debug({
                "event":"BEFORE.GET.NDARRAY",
                "bucket_id":BUCKET_ID,
                "key":cent_i_id,
            })
            cent_j_start_time = time.time()
            Cent_j_response   = LocalUtils.get_matrix_or_error(
                client    = STORAGE_CLIENT,
                key       = cent_i_id,
                bucket_id = BUCKET_ID
            )
            cent_j    = Cent_j_response.value
            __Cent_j  = Some(cent_j)
            status    = Constants.ClusteringStatus.WORK_IN_PROGRESS
            cent_j_st = time.time() - cent_j_start_time
            logger.debug({
                "event":"GET.NDARRAY",
                "bucket_id":BUCKET_ID,
                "key":cent_i_id,
                "shape":str(cent_j.shape), 
                "dtype":str(cent_j.dtype),
                "service_time":cent_j_st
            })
            logger.debug({
                "event":"WORKER.RUN1.BEFORE",
                "status":status,
                "k":k,
                "m":m,
                "enctypted_matrix_shape":str(encryptedMatrix.shape),
                "enctypted_matrix_dtype":str(encryptedMatrix.dtype),
                "UDM_shape":str(udm.shape),
                "UDM_dtype":str(udm.dtype),
                "cent_j_shape":str(cent_j.shape),
                "cent_j_dtype":str(cent_j.dtype),
            })

        run1_result:Result[
            Tuple[npt.NDArray, List[List[float]], List[List[float]], List[int]],
            Exception
        ] = skmeans.run1( # The first part of the skmeans is done
            status          = status,
            k               = k,
            m               = m,
            encryptedMatrix = encryptedMatrix, 
            UDM             = udm,
            Cent_j          = __Cent_j,
            num_attributes= encryptedMatrix.shape[1]
        )
        
        if run1_result.is_err:
            error = run1_result.unwrap_err()
            logger.error(str(error))
            return Response(str(error), status=500 )
        S1,Cent_i,Cent_j,label_vector = run1_result.unwrap()

        
        x = delete_and_put(
            STORAGE_CLIENT = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = cent_i_id,
            ndarray        = np.array(Cent_i)
        )
        
        y = delete_and_put(
            STORAGE_CLIENT = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = cent_j_id,
            ndarray        = np.array(Cent_j)
        )

        z = delete_and_put(
            STORAGE_CLIENT = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = encrypted_shift_matrix_id,
            ndarray        = np.array(S1)
        )
        
        _x:Result[PutResponse,Exception] = x.result()
        if _x.is_err:
            print("X")
            raise _x.unwrap_err()

        _y:Result[PutResponse,Exception] = y.result()
        if _y.is_err :
            print("Y")
            raise _y.unwrap_err()
        
        _z:Result[PutResponse,Exception] = z.result() 
        if _z.is_err:
            print("Z")
            raise _z.unwrap_err()

        end_time     = time.time()
        service_time = end_time - arrival_time
        n_iterations = int(requestHeaders.get("Iterations",0)) + 1
        logger.info({
            "event":"SKMEANS.1.COMPLETED",
            "plaintext_matrix_id":plaintext_matrix_id,
            "encrypted_matrix_id":encrypted_matrix_id,
            "encrypted_shift_matrix_id":encrypted_shift_matrix_id,
            "worker_id":worker_id,
            "algorithm":algorithm,
            "k":k,
            "m":m,
            "n_iterations":n_iterations,
            "service_time": service_time
        })
      
        return Response( #Returns the final response as a label vector + the headers
            response = json.dumps({
                "label_vector":label_vector,
                "service_time":service_time,
                "n_iterations":n_iterations,
                "encrypted_shift_matrix_id":encrypted_shift_matrix_id
            }),
            status   = 200,
            headers  = responseHeaders
        )
    except Exception as e:
        logger.error({
            "msg":str(e),
            "at":"worker_skmeans_1"
        })
        return Response(str(e),status = 500)


"""
Description:
    Second part of the skmeans process. 
    It starts when it receives S (decrypted matrix) from the client.
    If S is zero process ends
"""
def skmeans_2(requestHeaders):
    local_start_time        = time.time()
    logger                  = current_app.config["logger"]
    worker_id               = current_app.config["NODE_ID"]
    BUCKET_ID:str           = current_app.config.get("BUCKET_ID","rory")
    STORAGE_CLIENT:V4Client = current_app.config["STORAGE_CLIENT"]
    algorithm               = Constants.ClusteringAlgorithms.SKMEANS
    # HEADERS
    status                  = int(requestHeaders.get("Clustering-Status",Constants.ClusteringStatus.START))
    plaintext_matrix_id     = requestHeaders["Plaintext-Matrix-Id"]
    encrypted_matrix_id     = requestHeaders["Encrypted-Matrix-Id"]
    shift_matrix_id         = requestHeaders.get("Shift-Matrix-Id","{}-shift-matrix".format(plaintext_matrix_id))

    k                       = int(requestHeaders.get("K",3))
    m                       = int(requestHeaders.get("M",3))
    iterations              = int(requestHeaders.get("Iterations",0))
    
    if encrypted_matrix_id == -1 or plaintext_matrix_id == -1:
        return Response("Either Encrypted-Matrix-Id or Plain-Matrix-Id is missing",status=500)
    
    udm_id           = "{}-udm".format(plaintext_matrix_id)
    cent_i_id        = "{}-cent_i".format(plaintext_matrix_id) #Build the id of Cent_i
    cent_j_id        = "{}-cent_j".format(plaintext_matrix_id) #Build the id of Cent_j
    response_headers = {}

    logger.debug({
        "event":"SKMEANS.2.STARTED",
        "status":status,
        "shift_matrix_id":shift_matrix_id,
        "plaintext_matrix_id":plaintext_matrix_id,
        "encrypted_matrix_id":encrypted_matrix_id,
        "k":k,
        "m":m,
        "iterations":iterations
    })
    try:
        get_UDM_start_time = time.time()
        logger.debug({
            "event":"GET.NDARRAY.BEFORE",
            "key":udm_id, 
            "bucket_id":BUCKET_ID,
        })
        UDM_put_future:Awaitable[Result[GetNDArrayResponse,Exception]] =  STORAGE_CLIENT.get_ndarray(
            key       = udm_id,
            bucket_id = BUCKET_ID
        )
        UDM_result:Result[GetNDArrayResponse,Exception] = UDM_put_future.result()
        if UDM_result.is_err:
            return Response(None, status=500, headers={"Error-Message":str(UDM_result.unwrap_err())})
        get_UDM_st = time.time() - get_UDM_start_time
        
        UDM_response:GetNDArrayResponse = UDM_result.unwrap()
        UDM = UDM_response.value
        logger.info({
            "event":"GET.NDARRAY",
            "key":udm_id, 
            "bucket_id":BUCKET_ID,
            "shape":str(UDM.shape),
            "dtype":str(UDM.dtype),
            "service_time":get_UDM_st
        })
        
        logger.debug({
            "event":"GET.NDARRAY.BEFORE",
            "key":cent_i_id, 
            "bucket_id":BUCKET_ID,
        })
        get_cent_i_start_time = time.time()
        Cent_i_response = LocalUtils.get_matrix_or_error(
            client    = STORAGE_CLIENT,
            key       = cent_i_id,
            bucket_id = BUCKET_ID
        )
        Cent_i        = Cent_i_response.value
        get_cent_i_st = time.time() - get_cent_i_start_time
        logger.info({
            "event":"GET.NDARRAY",
            "key":cent_i_id, 
            "bucket_id":BUCKET_ID,
            "shape":str(Cent_i.shape),
            "dtype":str(Cent_i.dtype),
            "service_time":get_cent_i_st
        })

        logger.debug({
            "event":"GET.NDARRAY.BEFORE",
            "key":cent_j_id, 
            "bucket_id":BUCKET_ID,
        })
        get_cent_j_start_time = time.time()

        Cent_j_response = LocalUtils.get_matrix_or_error(
            client    = STORAGE_CLIENT,
            key       = cent_j_id,
            bucket_id = BUCKET_ID
        )
        Cent_j = Cent_j_response.value

        get_cent_j_st = time.time() - get_cent_j_start_time
        logger.info({
            "event":"GET.NDARRAY",
            "key":cent_j_id, 
            "bucket_id":BUCKET_ID,
            "shape":str(Cent_j.shape),
            "dtype":str(Cent_j.dtype),
            "service_time":get_cent_j_st
        })
        
        logger.debug({
            "event":"GET.NDARRAY.BEFORE",
            "key":shift_matrix_id, 
            "bucket_id":BUCKET_ID,
        })
        get_shift_matrix_start_time = time.time()

        shiftMatrix_get_response = LocalUtils.get_matrix_or_error(
            client    = STORAGE_CLIENT,
            key       = shift_matrix_id,
            bucket_id = BUCKET_ID
        )
        shiftMatrix = shiftMatrix_get_response.value
        get_shift_matrix_st = time.time() - get_cent_j_start_time
        logger.info({
            "event":"GET.NDARRAY",
            "key":shift_matrix_id, 
            "bucket_id":BUCKET_ID,
            "shape":str(shiftMatrix.shape),
            "dtype":str(shiftMatrix.dtype),
            "service_time":get_shift_matrix_st
        })
        min_error = 0.15
        isZero = Utils.verify_mean_error(
            old_matrix = Cent_i, 
            new_matrix = Cent_j, 
            min_error  = min_error
        )
        logger.debug({
            "event":"VERIFY.MEAN.ERROR",
            "min_error":min_error,
            "is_zero":int(isZero)
        })
        
        if(isZero): #If Shift matrix is zero
            response_headers["Clustering-Status"]  = Constants.ClusteringStatus.COMPLETED #Change the status to COMPLETED
            end_time                               = time.time()
            service_time                           = end_time - local_start_time #The service time is calculated
            response_headers["Total-Service-Time"] = str(service_time) #Save the service time

            logger.info({
                "event":"SKMEANS.2.COMPLETED",
                "plaintext_matrix_id":plaintext_matrix_id,
                "algorithm":algorithm,
                "worker_id":worker_id,
                "service_time":service_time,
                "n_iterations":iterations
            })
            return Response( #Return none and headers
                response = None, 
                status   = 204, 
                headers  = response_headers
            )
        else: #If Shift matrix is not zero
            skmeans = SKMeans() 
            status  = Constants.ClusteringStatus.WORK_IN_PROGRESS
            response_headers["Clustering-Status"] = status #The status is changed to WORK IN PROGRESS
            encrypted_matrix_shape = eval(requestHeaders["Encrypted-Matrix-Shape"]) # extract the attributes of shape
            logger.debug({
                "event":"NO.ZERO.RUN2",
                "status":status,
                "encrypted_matrix_shape":str(encrypted_matrix_shape),
                "shift_matrix_shape":str(shiftMatrix.shape),
                "shift_matrix_dtype":str(shiftMatrix.dtype)
            })
            _UDM = skmeans.run_2( # The second part of the skmeans starts
                k           = k,
                UDM         = UDM,
                attributes  = int(encrypted_matrix_shape[1]),
                shiftMatrix = shiftMatrix,
            )
            UDM_array = np.array(_UDM)
            logger.debug({
                "event":"SKMEANS.RUN2",
                "k":k,
                "udm_shape": str(UDM_array.shape),
                "udm_dtype":str(UDM_array.dtype),
            })
            
            udm_delete_result = STORAGE_CLIENT.delete(key=udm_id,bucket_id=BUCKET_ID)
            if udm_delete_result.is_err:
                error = udm_delete_result.unwrap_err()
                logger.error(str(error))
                return Response(str(error), status=500)
            logger.debug({
                "event":"UDM.DELETED",
                "key":udm_id,
                "bucket_id":BUCKET_ID
            })
            logger.debug({
                "event":"PUT.NDARRAY.BEFORE",
                "key":udm_id,
                "bucket_id":BUCKET_ID,
                "udm_shape": str(UDM_array.shape),
                "udm_dtype":str(UDM_array.dtype),
            })
            put_udm_start_time = time.time()
            x = STORAGE_CLIENT.put_ndarray(
                key       = udm_id, 
                ndarray   = UDM_array,
                tags      = {},
                bucket_id = BUCKET_ID
            ).result() # UDM is extracted from the storage system
            endTime2                        = time.time()
            put_udm_st= endTime2 - put_udm_start_time

            logger.info({
                "event":"PUT.NDARRAY",
                "key":udm_id,
                "bucket_id":BUCKET_ID,
                "udm_shape": str(UDM_array.shape),
                "udm_dtype":str(UDM_array.dtype),
                "service_time": put_udm_st
            })

            serviceTime2                    = endTime2 - local_start_time  #Service time is calculated
            response_headers["End-Time"]     = str(endTime2)
            response_headers["Service-Time"] = str(serviceTime2)
            
            logger.info({
                "event":"SKMEANS.2.UNCOMPLETED",
                "plaintext_matrix_id":plaintext_matrix_id,
                "algorithm":algorithm,
                "worker_id":worker_id,
                "service_time":serviceTime2,
                "n_iterations":iterations
            })
            return Response( #Return none and headers
                response = None,
                status   = 204, 
                headers  = response_headers
            )
    except Exception as e:
        logger.error("SKMEANS_2_ERROR: "+encrypted_matrix_id+" "+str(e))
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
    local_start_time        = time.time() #System startup time
    headers                 = request.headers
    to_remove_headers       = ["User-Agent","Accept-Encoding","Connection"]
    filtered_headers        = dict(list(filter(lambda x: not x[0] in to_remove_headers, headers.items())))
    algorithm               = Constants.ClusteringAlgorithms.KMEANS
    # Config
    logger                  = current_app.config["logger"]
    worker_id               = current_app.config["NODE_ID"] # Get the node_id from the global configuration
    STORAGE_CLIENT:V4Client = current_app.config["STORAGE_CLIENT"]
    BUCKET_ID:str           = current_app.config.get("BUCKET_ID","rory")

    plaintext_matrix_id     = filtered_headers.get("Plaintext-Matrix-Id")
    k                       = int(filtered_headers.get("K",3))
    response_headers        = {}
    logger.debug({
        "event":"KMEANS.STARTED",
        "algorithm":algorithm,
        "worker_id":worker_id,
        "bucket_id":BUCKET_ID,
        "plaintext_matrix_id":plaintext_matrix_id,
        "k":k
    })
    try:
        plaintext_matrix_response = LocalUtils.get_matrix_or_error(
            client    = STORAGE_CLIENT,
            key       = plaintext_matrix_id,
            bucket_id = BUCKET_ID
        ) 
        plaintext_matrix = plaintext_matrix_response.value
        result           = kMeans(
            k                = k, 
            plaintext_matrix = plaintext_matrix
        )
        end_time     = time.time()
        service_time = end_time - local_start_time

        response_headers["Service-Time"] = str(service_time)
        response_headers["Iterations"]   = int(result.n_iterations)
        
        logger.info({
            "event":"CLUSTERING",
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "worker_id":worker_id,
            "k":k,
            "n_iterations":result.n_iterations,
            "service_time":service_time
        })

        return Response( #Returns the final response as a label vector + the headers
            response = json.dumps({
                "label_vector":result.label_vector.tolist(),
                "iterations": result.n_iterations,
                "service_time": service_time
            }),
            status   = 200,
            headers  = {**response_headers}
        )
    except Exception as e:
        logger.error({
            "msg":str(e)
        })
        return Response(
            response = str(e),
            status   = 500
        )


"""
Description:
    First part of the dbskmeans process. 
    It stops where client interaction is required and writes the centroids and matrix S to disk.
"""
def dbskmeans_1(requestHeaders) -> Response:
    arrival_time             = time.time() #System startup time
    logger                   = current_app.config["logger"]
    worker_id                = current_app.config["NODE_ID"] # Get the node_id from the global configuration
    STORAGE_CLIENT:V4Client  = current_app.config["STORAGE_CLIENT"]
    BUCKET_ID:str            = current_app.config.get("BUCKET_ID","rory")
    # HEADERS
    status                   = int(requestHeaders.get("Clustering-Status", Constants.ClusteringStatus.START)) 
    is_start_status          = status == Constants.ClusteringStatus.START #if status is start save it to isStartStatus
    k                        = int(requestHeaders.get("K",3)) # It is passed to integer because the headers are strings
    m                        = int(requestHeaders.get("M",3))
    algorithm                = Constants.ClusteringAlgorithms.DBSKMEANS
    plaintext_matrix_id      = requestHeaders.get("Plaintext-Matrix-Id")
    encrypted_matrix_id      = requestHeaders.get("Encrypted-Matrix-Id","")
    _encrypted_matrix_shape  = requestHeaders.get("Encrypted-Matrix-Shape",-1)
    _encrypted_matrix_dtype  = requestHeaders.get("Encrypted-Matrix-Dtype",-1)
    _encrypted_udm_shape     = requestHeaders.get("Encrypted-Udm-Shape",-1)
    _encrypted_udm_dtype     = requestHeaders.get("Encrypted-Udm-Dtype",-1)
    iterations               = int(requestHeaders.get("Iterations",0))

    if _encrypted_matrix_dtype == -1:
        return Response("Encrypted-Matrix-Dtype", status=400)
    if _encrypted_matrix_shape == -1 :
        return Response("Encrypted-Matrix-Shape header is required", status=400)

    if _encrypted_udm_dtype == -1:
        return Response("Encrypted-UDM-Dtype", status=400)
    if _encrypted_udm_shape == -1 :
        return Response("Encrypted-UDM-Shape header is required", status=400)
    
    num_chunks = int(requestHeaders.get("Num-Chunks",-1))
    if num_chunks == -1:
        return Response("Num-Chunks header is required", status=503)
    encrypted_matrix_shape:tuple = eval(_encrypted_matrix_shape)
    encrypted_udm_shape:tuple    = eval(_encrypted_udm_shape)
    
    encrypted_udm_id           = "{}-encrypted-udm".format(plaintext_matrix_id) 
    cent_i_id                  = "{}-cent_i".format(plaintext_matrix_id) #Build the id of Cent_i
    cent_j_id                  = "{}-cent_j".format(plaintext_matrix_id) #Build the id of Cent_j
    encrypted_shift_matrix_id  = "{}-encrypted-shift-matrix".format(plaintext_matrix_id) #Build the id of Encrypted Shift Matrix
    dbskmeans                  = DBSKMeans()
    response_headers           = {}

    logger.debug({
        "event":"DBSKMEANS.1.STARTED",
        "algorithm":algorithm,
        "worker_id":worker_id,
        "status":status,
        "plaintext_matrix_id":plaintext_matrix_id,
        "encrypted_matrix_id":encrypted_matrix_id,
        "encrypted_matrix_shape":_encrypted_matrix_shape,
        "encrypted_udm_shape":_encrypted_udm_shape,
        "encrypted_shift_matrix_id":encrypted_shift_matrix_id,
        "num_chunks":num_chunks,
        "iterations":iterations,
        "k":k, 
        "m":m, 
    })

    try:
        response_headers["Start-Time"] = str(arrival_time)

        logger.debug({
            "event":"GET.NDARRAY.MERGE.BEFORE",
            "bucket_id":BUCKET_ID ,
            "key":encrypted_matrix_id,
            "num_chunks":num_chunks,
        })
        get_merge_start_time = time.time()
        (encrypted_matrix, encrypted_matrix_metadata) = LocalUtils.get_and_merge_ndarray(
            STORAGE_CLIENT = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = encrypted_matrix_id,
            num_chunks     = num_chunks, 
            shape          = encrypted_matrix_shape,
            dtype          = _encrypted_matrix_dtype,
        )
        get_merge_st = time.time() - get_merge_start_time
        logger.info({
            "event":"GET.NDARRAY.MERGE",
            "bucket_id":BUCKET_ID ,
            "key":encrypted_matrix_id,
            "num_chunks":num_chunks,
            "shape":str(encrypted_matrix.shape),
            "dtype":str(encrypted_matrix.dtype),
            "service_time":get_merge_st
        })
        response_headers["Encrypted-Matrix-Dtype"] = encrypted_matrix_metadata.tags.get("dtype",encrypted_matrix.dtype) #Save the data type
        response_headers["Encrypted-Matrix-Shape"] = encrypted_matrix_metadata.tags.get("shape",encrypted_matrix.shape) #Save the shape
        logger.debug({
            "event":"GET.NDARRAY.MERGE.BEFORE",
            "bucket_id":BUCKET_ID ,
            "key":encrypted_udm_id,
            "num_chunks":num_chunks,
        })
        get_merge_start_time = time.time()
        (encrypted_udm, udm_metadata) = LocalUtils.get_and_merge_ndarray(
            STORAGE_CLIENT = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = encrypted_udm_id,
            num_chunks     = num_chunks,
            shape          = encrypted_udm_shape,
            dtype          = _encrypted_udm_dtype
        )
        get_merge_st = time.time() - get_merge_start_time
        logger.info({
            "event":"GET.NDARRAY.MERGE",
            "bucket_id":BUCKET_ID ,
            "key":encrypted_udm_id,
            "num_chunks":num_chunks,
            "shape":str(encrypted_udm.shape),
            "dtype":str(encrypted_udm.dtype),
            "service_time":get_merge_st
        })
        
        response_headers["Encrypted-Udm-Dtype"]    = str(udm_metadata.tags.get("dtype",encrypted_udm.dtype)) # Extract the type
        response_headers["Encrypted-Udm-Shape"]    = str(udm_metadata.tags.get("shape",encrypted_udm.shape)) # Extract the shape
        
        if is_start_status: #if the status is start
            __Cent_j = NONE #There is no Cent_j
            logger.debug({
                "event":"NO.CentJ.WORKER.RUN1.BEFORE",
                "status":status,
                "k":k,
                "m":m,
                "enctypted_matrix_shape":str(encrypted_matrix.shape),
                "enctypted_matrix_dtype":str(encrypted_matrix.dtype),
                "encrypted_udm_shape":str(encrypted_udm.shape),
                "encrypted_udm_dtype":str(encrypted_udm.dtype),
            })

        else: 
            logger.debug({
                "event":"GET.MATRIX",
                "key":cent_i_id,
                "bucket_id":BUCKET_ID
            })
            get_matrix_cent_i_start_time = time.time()
            Cent_j_response = LocalUtils.get_matrix_or_error(
                bucket_id = BUCKET_ID,
                client    = STORAGE_CLIENT,
                key       = cent_i_id
            )
            cent_j_value = Cent_j_response.value
            __Cent_j     = Some(cent_j_value)
            status = Constants.ClusteringStatus.WORK_IN_PROGRESS
            logger.info({
                "event":"GET.MATRIX",
                "key":cent_i_id,
                "bucket_id":BUCKET_ID,
                "service_time":time.time() - get_matrix_cent_i_start_time
            })
        
            logger.debug({
                "event":"DBSKMEANS.RUN.1.BEFORE",
                "encrypted_matrix_shape":str(encrypted_matrix.shape),
                "encrypted_matrix_dtype":str(encrypted_matrix.dtype),
                "encrypted_udm_shape":str(encrypted_udm.shape),
                "encrypted_udm_dtype":str(encrypted_udm.dtype),
                "cent_j_shape":str(cent_j_value.shape),
                "cent_j_dtype":str(cent_j_value.dtype),
                "k":k,
                "m":m
            })
        run1_start_time = time.time()
        run1_result = dbskmeans.run1(
            encrypted_matrix = encrypted_matrix,
            UDM              = encrypted_udm,
            status           = status,
            k                = k,
            m                = m,
            Cent_j           = __Cent_j
        ) 

        if run1_result.is_err:
            error = run1_result.unwrap_err()
            logger.error({
                "msg":str(error)
            })
            return Response(str(error),status=500)
        S1,Cent_i,Cent_j,label_vector = run1_result.unwrap()
        run1_st = time.time()- run1_start_time
        logger.info({
            "event":"DBSKMEANS.RUN.1",
            "encrypted_matrix_shape":str(encrypted_matrix.shape),
            "encrypted_matrix_dtype":str(encrypted_matrix.dtype),
            "encrypted_udm_shape":str(encrypted_udm.shape),
            "encrypted_udm_dtype":str(encrypted_udm.dtype),
            "k":k,
            "m":m,
            "S1_shape":str(S1.shape),
            "S1_dtype":str(S1.dtype),
            "cent_i_shape":str(Cent_i.shape),
            "cent_i_dtype":str(Cent_i.dtype),
            "cent_j_shape":str(Cent_j.shape),
            "cent_j_dtype":str(Cent_j.dtype),
            "service_time":run1_st
        })

        logger.debug({
            "event":"PUT.NDARRAY.BEFORE",
            "key":cent_i_id,
            "bucket_id":BUCKET_ID
        })
        put_ndarray_start_time = time.time()
        xd = STORAGE_CLIENT.delete(
            key       = cent_i_id,
            bucket_id = BUCKET_ID
        )
        x:Result[PutResponse,Exception] = STORAGE_CLIENT.put_ndarray( # Saving Cent_i to storage
            key       = cent_i_id, 
            ndarray   = Cent_i,
            tags      = {},
            bucket_id = BUCKET_ID
        ).result()
        put_ndarray_st = time.time() - put_ndarray_start_time
        if x.is_err:
            error = str(x.unwrap_err())
            logger.error({
                "msg":error
            })
            return Response(error,status=500)
        logger.info({
            "event":"PUT.NDARRAY",
            "key":cent_i_id,
            "bucket_id":BUCKET_ID,
            "service_time":put_ndarray_st
        })
        
        logger.debug({
            "event":"PUT.NDARRAY.BEFORE",
            "key":cent_j_id,
            "bucket_id":BUCKET_ID
        })
        put_ndarray_start_time = time.time()
        yd = STORAGE_CLIENT.delete(
            key=cent_j_id,
            bucket_id=BUCKET_ID
        )
        y:Result[PutResponse,Exception] = STORAGE_CLIENT.put_ndarray( # Saving Cent_j to storage
            key       = cent_j_id, 
            ndarray   = Cent_j,
            tags      = {},
            bucket_id = BUCKET_ID
        ).result()
        if y.is_err:
            error = str(y.unwrap_err())
            logger.error({
                "msg":error
            })
            return Response(error,status=500)

        put_ndarray_st = time.time() - put_ndarray_start_time
        logger.info({
            "event":"PUT.NDARRAY",
            "key":cent_j_id,
            "bucket_id":BUCKET_ID,
            "service_time":put_ndarray_st
        })

        logger.debug({
            "event":"PUT.NDARRAY.BEFORE",
            "key":cent_j_id,
            "bucket_id":BUCKET_ID
        })
        put_ndarray_start_time = time.time()
        zd = STORAGE_CLIENT.delete(
            key=encrypted_shift_matrix_id,
            bucket_id=BUCKET_ID
        )
        z = STORAGE_CLIENT.put_ndarray( # Saving S1 matrix to storage
            key       = encrypted_shift_matrix_id,  
            ndarray   = S1,
            tags      = {},
            bucket_id = BUCKET_ID
        ).result()

        if z.is_err:
            error = str(z.unwrap_err())
            logger.error({
                "msg":error
            })
            return Response(error,status=500)

        put_ndarray_st = time.time() - put_ndarray_start_time
        logger.info({
            "event":"PUT.NDARRAY",
            "key":encrypted_shift_matrix_id,
            "bucket_id":BUCKET_ID,
            "service_time":put_ndarray_st
        })
        
        end_time                                      = time.time()
        service_time                                  = end_time - arrival_time
        response_headers["Service-Time"]              = str(service_time)
        response_headers["Iterations"]                = str(int(requestHeaders.get("Iterations",0)) + 1) #Saves the number of iterations in the header
        response_headers["Encrypted-Shift-Matrix-Id"] = encrypted_shift_matrix_id #Save the id of the encrypted shift matrix

        logger.info({
            "event":"DBSKMEANS.1.COMPLETED",
            "algorithm":algorithm, 
            "plaintext_matrix_id":plaintext_matrix_id,
            "encrypted_matrix_id":encrypted_matrix_id,
            "k":k,
            "m":m,
            "n_iterations":iterations,
            "service_time":service_time
        })
        
        return Response( #Returns the final response as a label vector + the headers
            response = json.dumps({
                "label_vector":label_vector,
                "encrypted_shift_matrix_id":encrypted_shift_matrix_id,
                "service_time":service_time
            }),
            status   = 200,
            headers  = {**requestHeaders, **response_headers}
        )
    except Exception as e:
        logger.error("DBSKMEANS_1_ERROR: "+encrypted_matrix_id+" "+str(e) )
        return Response(str(e),status = 503)

"""
Description:
    Second part of the dbskmeans process. 
    It starts when it receives S (decrypted matrix) from the client.
    If S is zero process ends
"""
def dbskmeans_2(requestHeaders):
    local_start_time        = time.time()
    logger                  = current_app.config["logger"]
    worker_id               = current_app.config["NODE_ID"]
    BUCKET_ID:str           = current_app.config.get("BUCKET_ID","rory")
    STORAGE_CLIENT:V4Client = current_app.config["STORAGE_CLIENT"]
    algorithm               = Constants.ClusteringAlgorithms.DBSKMEANS
    status                  = int(requestHeaders.get("Clustering-Status",Constants.ClusteringStatus.START))
    k                       = int(requestHeaders.get("K",3))
    m                       = int(requestHeaders.get("M",3))
    num_chunks              = int(requestHeaders.get("Num-Chunks",4))
    iterations              = int(requestHeaders.get("Iterations",0))
    global_start_time       = float(requestHeaders.get("Start-Time","0.0"))
    encrypted_matrix_id     = requestHeaders.get("Encrypted-Matrix-Id",-1)
    plaintext_matrix_id     = requestHeaders.get("Plaintext-Matrix-Id",-1)
    if encrypted_matrix_id  == -1 or plaintext_matrix_id == -1:
        return Response("Either Encrypted-Matrix-Id or Plain-Matrix-Id is missing",status=500)
    
    shift_matrix_id         = requestHeaders.get("Shift-Matrix-Id","{}-shift-matrix".format(plaintext_matrix_id))
    shift_matrix_ope_id     = requestHeaders.get("Shift-Matrix-Ope-Id","{}-shift-matrix-ope".format(plaintext_matrix_id))
    _encrypted_matrix_shape = requestHeaders.get("Encrypted-Matrix-Shape",-1)
    _encrypted_matrix_dtype = requestHeaders.get("Encrypted-Matrix-Dtype",-1)
    _encrypted_udm_shape    = requestHeaders.get("Encrypted-Udm-Shape",-1)
    _encrypted_udm_dtype    = requestHeaders.get("Encrypted-Udm-Dtype",-1)
    
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
    encrypted_udm_id             = "{}-encrypted-udm".format(plaintext_matrix_id)
    cent_i_id                    = "{}-cent_i".format(plaintext_matrix_id) #Build the id of Cent_i
    cent_j_id                    = "{}-cent_j".format(plaintext_matrix_id) #Build the id of Cent_j
    response_headers             = {}
    
    logger.debug({
        "event":"DBSKMEANS.2.STARTED",
        "algorithm":algorithm,
        "worker_id":worker_id,
        "status":status,
        "k":k,
        "m":m,
        "num_chunks":num_chunks,
        "n_iterations":iterations,
        "plaintext_matrix_id":plaintext_matrix_id,
        "encrypted_matrid_id":encrypted_matrix_id,
        "udm_id":encrypted_udm_id,
        "shift_matrix_id":shift_matrix_id,
        "shift_matrix_ope_id":shift_matrix_ope_id,
        "cent_i_id":cent_i_id,
        "cent_j_id":cent_j_id,
        "encrypted_matrix_shape":_encrypted_matrix_shape,
        "encrypted_matrix_dtype":_encrypted_matrix_dtype,
        "encrypted_udm_shape":_encrypted_udm_shape,
        "encrypted_udm_dtype":_encrypted_udm_dtype,
    })
    
    try:
        logger.debug({
            "event":"GET.NDARRAY.MERGE.BEFORE",
            "key":encrypted_udm_id,
            "bucket_id":BUCKET_ID,
            "num_chunks":num_chunks,
            "shape":_encrypted_udm_shape,
            "dtype":_encrypted_udm_dtype
        })
        get_merge_start_time = time.time()
        (prev_encrypted_udm, encrypted_udm_metadata) = LocalUtils.get_and_merge_ndarray(
            STORAGE_CLIENT = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = encrypted_udm_id,
            num_chunks     = num_chunks, 
            shape          = encrypted_udm_shape,
            dtype          = _encrypted_udm_dtype 
        )
        logger.info({
            "event":"GET.NDARRAY.MERGE",
            "key":encrypted_udm_id,
            "bucket_id":BUCKET_ID,
            "num_chunks":num_chunks,
            "shape":_encrypted_udm_shape,
            "dtype":_encrypted_udm_dtype,
            "service_time":time.time() - get_merge_start_time
        })
        
        logger.debug({
            "event":"GET.MATRIX.BEFORE",
            "key":cent_i_id,
            "bucket_id":BUCKET_ID,
        })
        get_matrix_start_time = time.time()
        Cent_i_response = LocalUtils.get_matrix_or_error(
            bucket_id = BUCKET_ID,
            client    = STORAGE_CLIENT,
            key       = cent_i_id
        )
        cent_i = Cent_i_response.value
        logger.info({
            "event":"GET.MATRIX",
            "key":cent_i_id,
            "bucket_id":BUCKET_ID,
            "service_time":time.time() - get_matrix_start_time
        })
        logger.debug({
            "event":"GET.MATRIX.BEFORE",
            "key":cent_j_id,
            "bucket_id":BUCKET_ID,
        })
        get_matrix_start_time = time.time()
        Cent_j_response = LocalUtils.get_matrix_or_error(
            bucket_id = BUCKET_ID,
            client    = STORAGE_CLIENT,
            key       = cent_j_id
        ) 
        cent_j = Cent_j_response.value
        logger.info({
            "event":"GET.MATRIX",
            "key":cent_j_id,
            "bucket_id":BUCKET_ID,
            "service_time":time.time() - get_matrix_start_time
        })
        
        min_error = 0.15
        isZero = Utils.verify_mean_error(
            old_matrix = cent_i, 
            new_matrix = cent_j,
            min_error  = min_error
        )
        logger.debug({
            "event":"VERIFY.MEAN.ERROR",
            "min_error":min_error,
            "is_zero":int(isZero)
        })
        
        if(isZero): #If Shift matrix is zero
            response_headers["Clustering-Status"]  = Constants.ClusteringStatus.COMPLETED #Change the status to COMPLETED
            end_time                               = time.time()
            service_time                           = end_time - local_start_time
            response_time                          = end_time - float(global_start_time) #The service time is calculated
            response_headers["Total-Service-Time"] = str(response_time) #Save the service time
            
            logger.info({
                "event":"DBSKMEANS.2.COMPLETED",
                "algorithm":algorithm,
                "n_iterations":iterations,
                "plaintext_matrix_id":plaintext_matrix_id,
                "encrypted_matrid_id":encrypted_matrix_id,
                "udm_id":encrypted_udm_id,
                "shift_matrix_id":shift_matrix_id,
                "shift_matrix_ope_id":shift_matrix_ope_id,
                "cent_i_id":cent_i_id,
                "cent_j_id":cent_j_id,
                "worker_id":worker_id,
                "m":m,
                "k":k,
                "shift_matrix_op_shape":str(shift_matrix_ope.shape),
                "shift_matrix_op_dtype":str(shift_matrix_ope.dtype),
                "prev_udm_shape":str(prev_encrypted_udm.shape),
                "prev_udm_dtype":str(prev_encrypted_udm.dtype),
                "current_udm_shape":str(current_udm.shape),
                "current_udm_dtype":str(current_udm.dtype),
                "num_chunks":num_chunks,
                "service_time":service_time,
                "response_time":response_time
            })
            return Response( #Return none and headers
                response = json.dumps({
                    "response_time":str(response_time),
                    "end_time":end_time,
                    "service_time":service_time,
                    "encrypted_udm_shape":str(current_udm.shape),
                    "encrypted_udm_dtype":str(current_udm.dtype),
                }), 
                status   = 200, 
                headers  = {**requestHeaders, **response_headers}
            )
        else: #If Shift matrix is not zero
            dbskmeans = DBSKMeans() 
            status    = Constants.ClusteringStatus.WORK_IN_PROGRESS #The status is changed to WORK IN PROGRESS
            response_headers["Clustering-Status"] = status
            logger.debug({
                "event":"GET.MATRIX.BEFORE",
                "key":shift_matrix_ope_id,
                "bucket_id":BUCKET_ID
            })
            get_matrix_start_time = time.time()
            shift_matrix_ope_response = LocalUtils.get_matrix_or_error(
                bucket_id = BUCKET_ID,
                client    = STORAGE_CLIENT,
                key       = shift_matrix_ope_id
            )
            shift_matrix_ope:npt.NDArray = shift_matrix_ope_response.value

            logger.info({
                "event":"GET.MATRIX",
                "key":shift_matrix_ope_id,
                "bucket_id":BUCKET_ID,
                "service_time":time.time() -get_matrix_start_time
            })
            
            udm_start_time = time.time()
            logger.debug({
                "event":"DBSKMEANS.RUN.2.BEFORE",
                "k":k,
                "shift_matrix_op_shape":str(shift_matrix_ope.shape),
                "shift_matrix_op_dtype":str(shift_matrix_ope.dtype),
                "prev_udm_shape":str(prev_encrypted_udm.shape),
                "prev_udm_dtype":str(prev_encrypted_udm.dtype),
            })
            current_udm = dbskmeans.run_2( # The second part of the skmeans starts
                k           = k,
                UDM         = prev_encrypted_udm,
                attributes  = int(encrypted_matrix_shape[1]),
                shiftMatrix = shift_matrix_ope,
            )
            
            response_headers["Encrypted-Udm-Dtype"] = str(current_udm.dtype)
            response_headers["Encrypted-Udm-Shape"] = str(current_udm.shape) # Extract the shape
            logger.info({
                "event":"DBSKMEANS.RUN.2",
                "k":k,
                "shift_matrix_op_shape":str(shift_matrix_ope.shape),
                "shift_matrix_op_dtype":str(shift_matrix_ope.dtype),
                "prev_udm_shape":str(prev_encrypted_udm.shape),
                "prev_udm_dtype":str(prev_encrypted_udm.dtype),
                "current_udm_shape":str(current_udm.shape),
                "current_udm_dtype":str(current_udm.dtype),
                "service_time":time.time()- udm_start_time
            })
            
            maybe_udm_chunks:Option[Chunks] = Chunks.from_ndarray(
                ndarray    = current_udm,
                group_id   = encrypted_udm_id,
                num_chunks = num_chunks,
                chunk_prefix = Some(encrypted_udm_id)
            )
            
            if maybe_udm_chunks.is_none:
                logger.error({
                    "msg":"Something went wrong segment encrypted udm."
                })
                return Response(
                    status   = 500,
                    response = "Something went wrong segment encrypted udm."
                )
            
            udm_chunks = maybe_udm_chunks.unwrap()
            
            logger.debug({
                "event":"PUT.CHUNKS.BEFORE",
                "key":encrypted_udm_id,
                "bucket_id":BUCKET_ID,
                "num_chunks":num_chunks
            })
            put_chunks_start_time = time.time()
            put_chunks_generator_results = STORAGE_CLIENT.put_chunks(
                bucket_id= BUCKET_ID,
                key       = encrypted_udm_id, 
                chunks    = udm_chunks, 
                tags      = {},
                update=True

            )
            
            for i,put_chunk_result in enumerate(put_chunks_generator_results):
                udm_end_time = time.time()
                udm_time     = udm_end_time - udm_start_time
                if put_chunk_result.is_err:
                    error = put_chunk_result.unwrap_err()
                    logger.error({
                        "msg":str(error)
                    })
                    return Response(
                        status   = 500,
                        response = "{}".format(str(error))
                    )

            logger.info({
                "event":"PUT.CHUNKS",
                "key":encrypted_udm_id,
                "bucket_id":BUCKET_ID,
                "num_chunks":num_chunks,
                "service_time":time.time() - put_chunks_start_time
            })
            end_time                         = time.time()
            service_time                     = end_time - local_start_time  #Service time is calculated
            response_headers["End-Time"]     = str(end_time)
            response_headers["Service-Time"] = str(service_time)

            logger.info({
                "event":"DBSKMEANS.2.UNCOMPLETED",
                "algorithm":algorithm,
                "n_iterations":iterations,
                "plaintext_matrix_id":plaintext_matrix_id,
                "encrypted_matrid_id":encrypted_matrix_id,
                "udm_id":encrypted_udm_id,
                "shift_matrix_id":shift_matrix_id,
                "shift_matrix_ope_id":shift_matrix_ope_id,
                "cent_i_id":cent_i_id,
                "cent_j_id":cent_j_id,
                "worker_id":worker_id,
                "m":m,
                "k":k,
                "shift_matrix_op_shape":str(shift_matrix_ope.shape),
                "shift_matrix_op_dtype":str(shift_matrix_ope.dtype),
                "prev_udm_shape":str(prev_encrypted_udm.shape),
                "prev_udm_dtype":str(prev_encrypted_udm.dtype),
                "current_udm_shape":str(current_udm.shape),
                "current_udm_dtype":str(current_udm.dtype),
                "num_chunks":num_chunks,
                "service_time":service_time
            })
            
            return Response( #Return none and headers
                response = json.dumps({
                    "end_time":end_time,
                    "service_time":service_time,
                    "encrypted_udm_shape":str(current_udm.shape),
                    "encrypted_udm_dtype":str(current_udm.dtype),
                }),
                status   = 200, 
                headers  = { **response_headers}
            )
    except Exception as e:
        logger.error({
            "msg":str(e)
        })
        return Response(str(e),status = 503)

"""
Description:
    DBSKMEANS algorithm
"""
@clustering.route("/dbskmeans", methods = ["POST"])
def dbskmeans():
    headers         = request.headers
    head            = ["User-Agent","Accept-Encoding","Connection"]
    logger          = current_app.config["logger"]
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
    local_start_time             = time.time() #System startup time
    headers                 = request.headers
    to_remove_headers       = ["User-Agent","Accept-Encoding","Connection"]
    filtered_headers        = dict(list(filter(lambda x: not x[0] in to_remove_headers, headers.items())))
    algorithm               = Constants.ClusteringAlgorithms.DBSNNC
    logger                  = current_app.config["logger"]
    STORAGE_CLIENT:Client   = current_app.config["STORAGE_CLIENT"]
    BUCKET_ID:str           = current_app.config.get("BUCKET_ID","rory")
    worker_id               = current_app.config["NODE_ID"] # Get the node_id from the global configuration
    plaintext_matrix_id     = filtered_headers.get("Plaintext-Matrix-Id")
    encrypted_matrix_id     = filtered_headers.get("Encrypted-Matrix-Id",-1)
    encrypted_dm_id         = filtered_headers.get("Encrypted-Dm-Id")
    encrypted_threshold     = float(filtered_headers.get("Encrypted-Threshold"))
    _encrypted_matrix_shape = filtered_headers.get("Encrypted-Matrix-Shape",-1)
    _encrypted_matrix_dtype = filtered_headers.get("Encrypted-Matrix-Dtype",-1)
    _encrypted_dm_shape     = filtered_headers.get("Encrypted-Dm-Shape",-1)
    _encrypted_dm_dtype     = filtered_headers.get("Encrypted-Dm-Dtype",-1)
    m                       = int(filtered_headers.get("M",3))

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

    num_chunks      = int(filtered_headers.get("Num-Chunks",-1))
    responseHeaders = {}
    
    if num_chunks == -1:
        return Response("Num-Chunks header is required", status=503)
    
    logger.debug({
        "event":"DBSNNC.1.STARTED",
        "algorithm":algorithm,
        "worker_id":worker_id,
        "plaintext_matrix_id":plaintext_matrix_id,
        "encrypted_matrix_id":encrypted_matrix_id,
        "encrypted_dm_id":encrypted_dm_id,
        "encrypted_matrix_shape":_encrypted_matrix_shape,
        "encrypted_matrix_dtype":_encrypted_matrix_dtype,
        "encrypted_threshold":encrypted_threshold,
        "num_chunks":num_chunks,
        "m":m, 
    })

    try:      
        responseHeaders["Start-Time"] = str(local_start_time)
        
        logger.debug({
            "event":"GET.NDARRAY.MERGE.BEFORE",
            "encrypted_matrix_id":encrypted_matrix_id,
            "num_chunks":num_chunks,
            "encrypted_matrix_shape":_encrypted_matrix_shape,
            "encrypted_matrix_dtype":_encrypted_matrix_dtype,
        })
        get_merge_encrypted_matrix_start_time = time.time()
        (encryptedMatrix, encrypted_matrix_metadata) = LocalUtils.get_and_merge_ndarray(
            STORAGE_CLIENT = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID, 
            key            = encrypted_matrix_id,
            num_chunks     = num_chunks, 
            shape          = encrypted_matrix_shape,
            dtype          = _encrypted_matrix_dtype
        )
        get_merge_encrypted_matrix_st = time.time() - get_merge_encrypted_matrix_start_time
        logger.info({
            "event":"GET.NDARRAY.MERGE",
            "encrypted_matrix_id":encrypted_matrix_id,
            "num_chunks":num_chunks,
            "encrypted_matrix_shape":_encrypted_matrix_shape,
            "encrypted_matrix_dtype":_encrypted_matrix_dtype,
            "service_time":get_merge_encrypted_matrix_st
        })

        responseHeaders["Encrypted-Matrix-Dtype"] = encrypted_matrix_metadata.tags.get("dtype",encryptedMatrix.dtype) #["tags"]["dtype"] #Save the data type
        responseHeaders["Encrypted-Matrix-Shape"] = encrypted_matrix_metadata.tags.get("shape",encryptedMatrix.shape) #Save the shape

        logger.debug({
            "event":"GET.NDARRAY.MERGE.BEFORE",
            "encrypted_matrix_id":encrypted_dm_id,
            "num_chunks":num_chunks,
            "encrypted_matrix_shape":_encrypted_dm_shape,
            "encrypted_matrix_dtype":_encrypted_dm_dtype,
        })
        get_merge_encrypted_dm_start_time = time.time()
        (distance_matrix, dm_metadata) = LocalUtils.get_and_merge_ndarray(
            STORAGE_CLIENT = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = encrypted_dm_id,
            num_chunks     = num_chunks,
            shape          = encrypted_dm_shape,
            dtype          = _encrypted_dm_dtype,
        )
        get_merge_encrypted_dm_st = time.time() - get_merge_encrypted_dm_start_time
        logger.info({
            "event":"GET.NDARRAY.MERGE",
            "encrypted_matrix_id":encrypted_dm_id,
            "num_chunks":num_chunks,
            "encrypted_matrix_shape":_encrypted_dm_shape,
            "encrypted_matrix_dtype":_encrypted_dm_dtype,
            "service_time": get_merge_encrypted_dm_st
        })

        responseHeaders["Encrypted-Dm-Dtype"] = str(dm_metadata.tags.get("dtype",distance_matrix.dtype)) # Extract the type
        responseHeaders["Encrypted-Dm-Shape"] = str(dm_metadata.tags.get("shape",distance_matrix.shape)) # Extract the shape
        
        logger.debug({
            "event":"DBSNNC.RUN.BEFORE",
            "algorithm":algorithm,
            "distance_matrix_shape":str(distance_matrix.shape),
            "distance_matrix_dtype":str(distance_matrix.dtype),
        })
        result = Dbsnnc.run(
            distance_matrix     = distance_matrix,
            encrypted_threshold = encrypted_threshold
        )
        end_time                         = time.time()
        service_time                     = end_time - local_start_time

        logger.info({
            "event":"DBSNNC.RUN",
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "encrypted_matrix_id":encrypted_matrix_id,
            "encrypted_dm_id":encrypted_dm_id,
            "distance_matrix_shape":str(distance_matrix.shape),
            "distance_matrix_dtype":str(distance_matrix.dtype),
            "num_chunks":num_chunks,
            "service_time":service_time
        })
        responseHeaders["Service-Time"] = str(service_time)
        
        return Response( #Returns the final response as a label vector + the headers
            response = json.dumps({
                "label_vector":result.label_vector,
                "service_time":service_time
            }),
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
    local_start_time      = time.time() #System startup time
    headers               = request.headers
    to_remove_headers     = ["User-Agent","Accept-Encoding","Connection"]
    filtered_headers      = dict(list(filter(lambda x: not x[0] in to_remove_headers, headers.items())))
    algorithm             = Constants.ClusteringAlgorithms.NNC
    logger                = current_app.config["logger"]
    STORAGE_CLIENT:Client = current_app.config["STORAGE_CLIENT"]
    BUCKET_ID:str         = current_app.config.get("BUCKET_ID","rory")
    worker_id             = current_app.config["NODE_ID"] # Get the node_id from the global configuration
    plaintext_matrix_id   = filtered_headers.get("Plaintext-Matrix-Id")
    threshold             = float(filtered_headers.get("Threshold"))
    dm_id                 = "{}-dm".format(plaintext_matrix_id) 
    response_headers      = {}
    response_headers      = {}
    logger.debug({
        "event":"NNC.STARTED",
        "algorithm":algorithm,
        "worker_id":worker_id,
        "bucket_id":BUCKET_ID,
        "plaintext_matrix_id":plaintext_matrix_id,
        "threshold":threshold,
        "dm_id":dm_id,
    })

    try:      
        response_headers["Start-Time"] = str(local_start_time)
        
        logger.debug({
            "event":"GET.MATRIX.BEFORE",
            "key":plaintext_matrix_id,
            "bucket_id":BUCKET_ID,
        })
        get_ptm_start_time = time.time()
        plainTextMatrix_response = LocalUtils.get_matrix_or_error(
            client    = STORAGE_CLIENT,
            key       = plaintext_matrix_id,
            bucket_id = BUCKET_ID
        ) 
        plaintext_matrix                           = plainTextMatrix_response.value
        response_headers["Plaintext-Matrix-Dtype"] = plainTextMatrix_response.metadata.tags.get("dtype",plaintext_matrix.dtype) #["tags"]["dtype"] #Save the data type
        response_headers["Plaintext-Matrix-Shape"] = plainTextMatrix_response.metadata.tags.get("shape",plaintext_matrix.shape) #Save the shape
        
        get_ptm_st = time.time() - get_ptm_start_time
        logger.info({
            "event":"GET.MATRIX",
            "key":plaintext_matrix_id,
            "bucket_id":BUCKET_ID,
            "service_time":get_ptm_st
        })
        
        logger.debug({
            "event":"GET.MATRIX.BEFORE",
            "key":dm_id,
            "bucket_id":BUCKET_ID,
        })
        get_ptm_start_time = time.time()
        dm_matrix_response = LocalUtils.get_matrix_or_error(
            client    = STORAGE_CLIENT,
            key       = dm_id,
            bucket_id = BUCKET_ID
        )

        distance_matrix                     = dm_matrix_response.value
        response_headers["Dm-Matrix-Dtype"] = dm_matrix_response.metadata.tags.get("dtype",distance_matrix.dtype) # Extract the type
        response_headers["Dm-Matrix-Shape"] = dm_matrix_response.metadata.tags.get("shape",distance_matrix.shape) # Extract the shape

        get_dm_st = time.time() - get_ptm_start_time
        logger.info({
            "event":"GET.MATRIX",
            "key":dm_id,
            "bucket_id":BUCKET_ID,
            "service_time":get_dm_st
        })
        
        logger.debug({
            "event":"RUN.1.BEFORE",
            "algorithm":algorithm,
            "distance_matrix_shape":str(distance_matrix.shape),
            "distance_matrix_dtype":str(distance_matrix.dtype),
            "threshold":threshold
        })
        result = Nnc.run(
            distance_matrix = distance_matrix,
            threshold       = threshold
        )

        end_time     = time.time()
        service_time = end_time - local_start_time
        response_headers["Service-Time"] = str(service_time)
        
        logger.info({
            "event":"NNC.COMPLETED",
            "algorithm":algorithm,
            "worker_id":worker_id,
            "algorithm":algorithm,
            "distance_matrix_shape":str(distance_matrix.shape),
            "distance_matrix_dtype":str(distance_matrix.dtype),
            "threshold":threshold,
            "service_time":service_time
        })

        return Response( #Returns the final response as a label vector + the headers
            response = json.dumps({
                "label_vector":result.label_vector,
                "service_time":service_time
            }),
            status   = 200,
            headers  = response_headers
        )
    except Exception as e:
        logger.error(e)
        return Response(
            response = None,
            status   = 503,
            headers  = {"Error-Message":e})