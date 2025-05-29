import time, json
import numpy as np
import numpy.typing as npt
import os
from typing import Awaitable,List,Tuple
from flask import Blueprint,current_app,request,Response
from rory.core.clustering.kmeans import kmeans as kMeans
from rory.core.clustering.secure.conventional.dbsnnc import Dbsnnc
from rory.core.clustering.nnc import Nnc
from rory.core.utils.utils import Utils
from rory.core.utils.constants import Constants
from rory.core.clustering.secure.conventional.skmeans import SKMeans
from rory.core.clustering.secure.conventional.dbskmeans import DBSKMeans
from rory.core.clustering.secure.pqc.skmeans import Skmeans as SkmeansPQC
from rory.core.clustering.secure.pqc.dbskmeans import DBSKMeans as DbskmeansPQC
from rory.core.security.cryptosystem.pqc.ckks import Ckks
from mictlanx.v4.client import Client as V4Client
from mictlanx import AsyncClient
from mictlanx.utils.index import Utils as MictlanXUtils
from option import Result, Some
from mictlanx.utils.segmentation import Chunks
from mictlanx.v4.interfaces.responses import GetNDArrayResponse,PutResponse
from option import Option,Some,NONE
from utils.utils import Utils as LocalUtils
from rorycommon import Common as RoryCommon
from Pyfhel import PyCtxt,Pyfhel
from models import ExperimentLogEntry
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
async def skmeans_1(requestHeaders) -> Response:
    arrival_time               = time.time() #Worker start time
    logger                     = current_app.config["logger"]
    worker_id                  = current_app.config["NODE_ID"] # Get the node_id from the global configuration
    STORAGE_CLIENT:AsyncClient = current_app.config["ASYNC_STORAGE_CLIENT"]
    BUCKET_ID:str              = current_app.config.get("BUCKET_ID","rory")
    status                     = int(requestHeaders.get("Clustering-Status", Constants.ClusteringStatus.START)) 
    is_start_status            = status == Constants.ClusteringStatus.START #if status is start save it to isStartStatus
    k                          = int(requestHeaders.get("K",3)) # It is passed to integer because the headers are strings
    m                          = int(requestHeaders.get("M",3))
    algorithm                  = Constants.ClusteringAlgorithms.SKMEANS
    plaintext_matrix_id        = requestHeaders.get("Plaintext-Matrix-Id")
    encrypted_matrix_id        = requestHeaders.get("Encrypted-Matrix-Id",-1)
    udm_id                     = "{}udm".format(plaintext_matrix_id) 
    _encrypted_matrix_shape    = requestHeaders.get("Encrypted-Matrix-Shape",-1)
    _encrypted_matrix_dtype    = requestHeaders.get("Encrypted-Matrix-Dtype",-1)
    experiment_id              = requestHeaders.get("Experiment-Id","")
    MICTLANX_TIMEOUT           = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
    MICTLANX_DELAY             = int(current_app.config.get("MICTLANX_DELAY","2"))
    MICTLANX_BACKOFF_FACTOR    = float(current_app.config.get("MICTLANX_BACKOFF_FACTOR","0.5"))
    MICTLANX_MAX_RETRIES       = int(current_app.config.get("MICTLANX_MAX_RETRIES","10")) 

    if _encrypted_matrix_dtype == -1:
        return Response("Encrypted-Matrix-Dtype", status=500)
    if _encrypted_matrix_shape == -1 :
        return Response("Encrypted-Matrix-Shape header is required", status=500)
    
    encrypted_matrix_shape:tuple = eval(_encrypted_matrix_shape)

    encrypted_shift_matrix_id = "{}encryptedshiftmatrix".format(plaintext_matrix_id) #Build the id of Encrypted Shift Matrix
    cent_i_id                 = "{}centi".format(plaintext_matrix_id) #Build the id of Cent_i
    cent_j_id                 = "{}centj".format(plaintext_matrix_id) #Build the id of Cent_j
    num_chunks                = int(requestHeaders.get("Num-Chunks",-1))
    skmeans                   = SKMeans()
    responseHeaders           = {}
    
    if num_chunks == -1:
        logger.error({"msg":"Num-Chunks header is required"})
        return Response("Num-Chunks header is required", status=503)
    try:
        responseHeaders["Start-Time"] = str(arrival_time)

        get_merge_encrypted_matrix_start_time  = time.time()
        encryptedMatrix:npt.NDArray = await RoryCommon.get_and_merge(
            client         = STORAGE_CLIENT,
            key            = encrypted_matrix_id,
            bucket_id      = BUCKET_ID,
            delay          = MICTLANX_DELAY,
            max_retries    = MICTLANX_MAX_RETRIES,
            timeout        = MICTLANX_TIMEOUT,
            backoff_factor = MICTLANX_BACKOFF_FACTOR
        )
        responseHeaders["Encrypted-Matrix-Dtype"] = str(encryptedMatrix.dtype)
        responseHeaders["Encrypted-Matrix-Shape"] = str(encryptedMatrix.shape)
        
        get_encrypted_ptm_entry = ExperimentLogEntry(
            event          = "GET",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_merge_encrypted_matrix_start_time,
            end_time       = time.time(),
            id             = encrypted_matrix_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k,
            m              = m,
        )
        logger.info(get_encrypted_ptm_entry.model_dump())
        
        udm_get_start_time  = time.time()
        udm = await RoryCommon.get_and_merge(
            client         = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = udm_id,
            delay          = MICTLANX_DELAY,
            max_retries    = MICTLANX_MAX_RETRIES,
            timeout        = MICTLANX_TIMEOUT,
            backoff_factor = MICTLANX_BACKOFF_FACTOR
        )
        
        get_udm_entry = ExperimentLogEntry(
                event          = "GET",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = udm_get_start_time,
                end_time       = time.time(),
                id             = udm_id,
                worker_id      = worker_id,
                num_chunks     = num_chunks,
                k              = k,
                m              = m,
        )
        logger.info(get_udm_entry.model_dump())
 
        responseHeaders["Udm-Matrix-Dtype"] = str(udm.dtype) # Extract the type
        responseHeaders["Udm-Matrix-Shape"] = str(udm.shape) # Extract the shape
        if is_start_status: #if the status is start
            __Cent_j = NONE #There is no Cent_j
        else: 
            cent_j_start_time = time.time()
            cent_j   = await RoryCommon.get_and_merge(
                client         = STORAGE_CLIENT,
                key            = cent_i_id,
                bucket_id      = BUCKET_ID,
                delay          = MICTLANX_DELAY,
                max_retries    = MICTLANX_MAX_RETRIES,
                timeout        = MICTLANX_TIMEOUT,
                backoff_factor = MICTLANX_BACKOFF_FACTOR
            )
            # __Cent_j = Some(cent_j)
            __Cent_j = cent_j.copy()

            status   = Constants.ClusteringStatus.WORK_IN_PROGRESS
            get_cent_i_entry = ExperimentLogEntry(
                event          = "GET",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = cent_j_start_time,
                end_time       = time.time(),
                id             = cent_i_id,
                worker_id      = worker_id,
                num_chunks     = num_chunks,
                k              = k,
                m              = m,
            ) 
            logger.info(get_cent_i_entry.model_dump())

        run1_start_time = time.time()
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
        )
        
        if run1_result.is_err:
            error = run1_result.unwrap_err()
            logger.error({
                "event":"SKMEANS.RUN1.FAILED",
                "raw_error":str(error)
            })
            return Response(str(error), status=500 )
        S1,Cent_i,Cent_j,label_vector = run1_result.unwrap()

        run1_entry = ExperimentLogEntry(
            event          = "RUN1",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = run1_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k,
            m              = m
        ) 
        logger.info(run1_entry.model_dump())

        Cent_i = np.array(Cent_i)
        Cent_j = np.array(Cent_j)
        t1 = time.time()
        maybe_cent_i_chunks = Chunks.from_ndarray(
            ndarray      = Cent_i,
            group_id     = cent_i_id,
            chunk_prefix = Some(cent_i_id),
            num_chunks   = k,
        )
        if maybe_cent_i_chunks.is_none:
            raise Exception("something went wrong creating the chunks")
        
        x = await RoryCommon.delete_and_put_chunks(
            client    = STORAGE_CLIENT,
            bucket_id = BUCKET_ID,
            key       = cent_i_id,
            chunks    = maybe_cent_i_chunks.unwrap(),
            timeout   = MICTLANX_TIMEOUT,
            max_tries = MICTLANX_MAX_RETRIES,
            tags      = {
                "full_shape": str(Cent_i.shape),
                "full_dtype": str(Cent_i.dtype)
            }
        )
        if x.is_err:
            return Response(status=500, response="Put cent_i failed.")

        put_cent_i_entry = ExperimentLogEntry(
            event          = "PUT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = t1,
            end_time       = time.time(),
            id             = cent_i_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k,
            m              = m,
        ) 
        logger.info(put_cent_i_entry.model_dump())
        
        t1 = time.time()
        maybe_cent_j_chunks = Chunks.from_ndarray(
            ndarray      = Cent_j,
            group_id     = cent_j_id,
            chunk_prefix = Some(cent_j_id),
            num_chunks   = k,
        )
        if maybe_cent_j_chunks.is_none:
            raise "something went wrong creating the chunks"
       
        y = await RoryCommon.delete_and_put_chunks(
            client    = STORAGE_CLIENT,
            bucket_id = BUCKET_ID,
            key       = cent_j_id,
            chunks    = maybe_cent_j_chunks.unwrap(),
            timeout   = MICTLANX_TIMEOUT,
            max_tries = MICTLANX_MAX_RETRIES,
            tags      = {
                "full_shape": str(Cent_j.shape),
                "full_dtype": str(Cent_j.dtype)
            }
        )
        if y.is_err:
            return Response(status=500, response="Put cent_j failed.")
        
        put_cent_j_entry = ExperimentLogEntry(
            event          = "PUT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = t1,
            end_time       = time.time(),
            id             = cent_j_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k,
            m              = m,
        ) 
        logger.info(put_cent_j_entry.model_dump())
        
        maybe_s1_chunks = Chunks.from_ndarray(
            ndarray      = S1,
            group_id     = encrypted_shift_matrix_id,
            chunk_prefix = Some(encrypted_shift_matrix_id),
            num_chunks   = num_chunks,
        )
        if maybe_s1_chunks.is_none:
            raise "something went wrong creating the chunks"
        
        t1 = time.time()
        z = await RoryCommon.delete_and_put_chunks(
            client    = STORAGE_CLIENT,
            bucket_id = BUCKET_ID,
            key       = encrypted_shift_matrix_id,
            chunks    = maybe_s1_chunks.unwrap(),
            timeout   = MICTLANX_TIMEOUT,
            max_tries = MICTLANX_MAX_RETRIES,
            tags = {
                "full_shape": str(S1.shape),
                "full_dtype": str(S1.dtype)
            }
        )

        put_encrypted_sm_entry = ExperimentLogEntry(
            event          = "PUT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = t1,
            end_time       = time.time(),
            id             = encrypted_shift_matrix_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k,
            m              = m,
        ) 
        logger.info(put_encrypted_sm_entry.model_dump())
 
        end_time     = time.time()
        service_time = end_time - arrival_time
        n_iterations = int(requestHeaders.get("Iterations",0)) + 1
        
        clustering_entry = ExperimentLogEntry(
            event          = "COMPLETED",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = arrival_time,
            end_time       = time.time(),
            id             = encrypted_matrix_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k,
            workers        = 0,
            security_level = 0,
            m              = m,
            iterations     = n_iterations,
            time           = service_time
        )
        logger.info(clustering_entry.model_dump())

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
async def skmeans_2(requestHeaders):
    local_start_time           = time.time()
    logger                     = current_app.config["logger"]
    worker_id                  = current_app.config["NODE_ID"]
    BUCKET_ID:str              = current_app.config.get("BUCKET_ID","rory")
    STORAGE_CLIENT:AsyncClient = current_app.config["ASYNC_STORAGE_CLIENT"]
    algorithm                  = Constants.ClusteringAlgorithms.SKMEANS
    status                     = int(requestHeaders.get("Clustering-Status",Constants.ClusteringStatus.START))
    plaintext_matrix_id        = requestHeaders["Plaintext-Matrix-Id"]
    encrypted_matrix_id        = requestHeaders["Encrypted-Matrix-Id"]
    shift_matrix_id            = requestHeaders.get("Shift-Matrix-Id","{}shiftmatrix".format(plaintext_matrix_id))
    k                          = int(requestHeaders.get("K",3))
    m                          = int(requestHeaders.get("M",3))
    iterations                 = int(requestHeaders.get("Iterations",0))
    experiment_id              = requestHeaders.get("Experiment-Id","")
    
    if encrypted_matrix_id == -1 or plaintext_matrix_id == -1:
        return Response("Either Encrypted-Matrix-Id or Plain-Matrix-Id is missing",status=500)
    num_chunks       = int(requestHeaders.get("Num-Chunks",-1))
    udm_id           = "{}udm".format(plaintext_matrix_id)
    cent_i_id        = "{}centi".format(plaintext_matrix_id) #Build the id of Cent_i
    cent_j_id        = "{}centj".format(plaintext_matrix_id) #Build the id of Cent_j
    response_headers = {}
    min_error               = float(current_app.config.get("MIN_ERROR", 0.015))
    MICTLANX_TIMEOUT        = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
    MICTLANX_DELAY          = int(current_app.config.get("MICTLANX_DELAY","2"))
    MICTLANX_BACKOFF_FACTOR = float(current_app.config.get("MICTLANX_BACKOFF_FACTOR","0.5"))
    MICTLANX_MAX_RETRIES    = int(current_app.config.get("MICTLANX_MAX_RETRIES","10")) 

    try:
        get_UDM_start_time = time.time()
  
        UDM =  await RoryCommon.get_and_merge(
            client         = STORAGE_CLIENT, 
            bucket_id      = BUCKET_ID,
            key            = udm_id,
            delay          = MICTLANX_DELAY,
            max_retries    = MICTLANX_MAX_RETRIES,
            timeout        = MICTLANX_TIMEOUT,
            backoff_factor = MICTLANX_BACKOFF_FACTOR
        )

        get_udm_entry = ExperimentLogEntry(
            event          = "GET",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_UDM_start_time,
            end_time       = time.time(),
            id             = udm_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k,
            m              = m,
            iterations     = iterations
        )
        logger.info(get_udm_entry.model_dump())

        get_cent_i_start_time = time.time()
        Cent_i = await RoryCommon.get_and_merge(
            client         = STORAGE_CLIENT,
            key            = cent_i_id,
            bucket_id      = BUCKET_ID,
            delay          = MICTLANX_DELAY,
            max_retries    = MICTLANX_MAX_RETRIES,
            timeout        = MICTLANX_TIMEOUT,
            backoff_factor = MICTLANX_BACKOFF_FACTOR
        )

        get_cent_i_entry = ExperimentLogEntry(
            event          = "GET",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_cent_i_start_time,
            end_time       = time.time(),
            id             = cent_i_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k,
            m              = m,
            iterations     = iterations
        )
        logger.info(get_cent_i_entry.model_dump())
 
        get_cent_j_start_time = time.time()
        Cent_j = await RoryCommon.get_and_merge(
            client         = STORAGE_CLIENT,
            key            = cent_j_id,
            bucket_id      = BUCKET_ID,
            delay          = MICTLANX_DELAY,
            max_retries    = MICTLANX_MAX_RETRIES,
            timeout        = MICTLANX_TIMEOUT,
            backoff_factor = MICTLANX_BACKOFF_FACTOR
        )

        get_cent_j_entry = ExperimentLogEntry(
            event          = "GET",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_cent_j_start_time,
            end_time       = time.time(),
            id             = cent_j_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k,
            m              = m,
            iterations     = iterations
        )
        logger.info(get_cent_j_entry.model_dump())
      
        get_shift_matrix_start_time = time.time()
        shiftMatrix = await RoryCommon.get_and_merge(
            client         = STORAGE_CLIENT,
            key            = shift_matrix_id,
            bucket_id      = BUCKET_ID,
            delay          = MICTLANX_DELAY,
            max_retries    = MICTLANX_MAX_RETRIES,
            timeout        = MICTLANX_TIMEOUT,
            backoff_factor = MICTLANX_BACKOFF_FACTOR
        )

        get_sm_entry = ExperimentLogEntry(
            event          = "GET",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_shift_matrix_start_time,
            end_time       = time.time(),
            id             = shift_matrix_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k,
            m              = m,
            iterations     = iterations
        )
        logger.info(get_sm_entry.model_dump())

        isZero = Utils.verify_mean_error(
            old_matrix = Cent_i, 
            new_matrix = Cent_j, 
            min_error  = min_error
        )
        
        if(isZero): #If Shift matrix is zero
            response_headers["Clustering-Status"]  = Constants.ClusteringStatus.COMPLETED #Change the status to COMPLETED
            end_time                               = time.time()
            service_time                           = end_time - local_start_time #The service time is calculated
            response_headers["Total-Service-Time"] = str(service_time) #Save the service time

            clustering_completed_entry = ExperimentLogEntry(
                event          = "COMPLETED",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = local_start_time,
                end_time       = time.time(),
                id             = plaintext_matrix_id,
                worker_id      = worker_id,
                num_chunks     = num_chunks,
                k              = k,
                m              = m,
                iterations     = iterations
            )
            logger.info(clustering_completed_entry.model_dump())
        
            return Response( #Return none and headers
                response = None, 
                status   = 204, 
                headers  = response_headers
            )
        else: #If Shift matrix is not zero
            run2_start_time = time.time()
            skmeans         = SKMeans() 
            status          = Constants.ClusteringStatus.WORK_IN_PROGRESS
            response_headers["Clustering-Status"] = status #The status is changed to WORK IN PROGRESS
            encrypted_matrix_shape = eval(requestHeaders["Encrypted-Matrix-Shape"]) # extract the attributes of shape
            _UDM = skmeans.run_2( # The second part of the skmeans starts
                k           = k,
                UDM         = UDM,
                attributes  = int(encrypted_matrix_shape[1]),
                shiftMatrix = shiftMatrix,
            )
            UDM_array = np.array(_UDM)

            run_2_entry = ExperimentLogEntry(
                    event          = "RUN2",
                    experiment_id  = experiment_id,
                    algorithm      = algorithm,
                    start_time     = run2_start_time,
                    end_time       = time.time(),
                    id             = plaintext_matrix_id,
                    worker_id      = worker_id,
                    num_chunks     = num_chunks,
                    k              = k,
                    m              = m,
                    iterations     = iterations
            )
            logger.info(run_2_entry.model_dump())

            put_udm_start_time = time.time()

            udm_chunks = Chunks.from_ndarray(
                ndarray      = UDM_array,
                group_id     = udm_id,
                chunk_prefix = Some(udm_id),
                num_chunks   = num_chunks,
            )

            if udm_chunks.is_none:
                return Response(status= 500, response = "Failed to put udm chunks")
            
            put_udm_result = await RoryCommon.delete_and_put_chunks(
                client    = STORAGE_CLIENT,
                bucket_id = BUCKET_ID,
                key       = udm_id,
                chunks    = udm_chunks.unwrap(),
                timeout   = MICTLANX_TIMEOUT,
                max_tries = MICTLANX_MAX_RETRIES,
                tags      = {
                    "full_shape":str(UDM_array.shape),
                    "full_dtype":str(UDM_array.dtype)
                }
            )
            
            if put_udm_result.is_err:
                error = str(put_udm_result.unwrap_err())
                logger.error({
                    "msg":error
                })
                return Response(error,status=500)
            
            put_udm_entry = ExperimentLogEntry(
                    event          = "PUT",
                    experiment_id  = experiment_id,
                    algorithm      = algorithm,
                    start_time     = put_udm_start_time,
                    end_time       = time.time(),
                    id             = udm_id,
                    worker_id      = worker_id,
                    num_chunks     = num_chunks,
                    k              = k,
                    m              = m,
                    iterations     = iterations
            )
            logger.info(put_udm_entry.model_dump())

            clutering_uncompleted_entry = ExperimentLogEntry(
                    event          = "UNCOMPLETED",
                    experiment_id  = experiment_id,
                    algorithm      = algorithm,
                    start_time     = local_start_time,
                    end_time       = time.time(),
                    id             = plaintext_matrix_id,
                    worker_id      = worker_id,
                    num_chunks     = num_chunks,
                    k              = k,
                    m              = m,
                    iterations     = iterations
            )
            logger.info(clutering_uncompleted_entry.model_dump())
            response_headers["End-Time"]     = str(clutering_uncompleted_entry.end_time)
            response_headers["Service-Time"] = str(clutering_uncompleted_entry.time)

            return Response( #Return none and headers
                response = None,
                status   = 204, 
                headers  = response_headers
            )
    except Exception as e:
        logger.error("SKMEANS_2_ERROR: "+encrypted_matrix_id+" "+str(e))
        return Response(str(e),status = 503)

@clustering.route("/skmeans",methods = ["POST"])
async def skmeans():
    headers         = request.headers
    head            = ["User-Agent","Accept-Encoding","Connection"]
    filteredHeaders = dict(list(filter(lambda x: not x[0] in head, headers.items())))
    step_index      = int(filteredHeaders.get("Step-Index",1))
    response        = Response()
    if step_index == 1:
        return await skmeans_1(filteredHeaders)
    elif step_index == 2:
        return await skmeans_2(filteredHeaders)
    else:
        return response

@clustering.route("/kmeans",methods = ["POST"])
async def kmeans():
    local_start_time        = time.time() #System startup time
    headers                 = request.headers
    to_remove_headers       = ["User-Agent","Accept-Encoding","Connection"]
    filtered_headers        = dict(list(filter(lambda x: not x[0] in to_remove_headers, headers.items())))
    experiment_id           = filtered_headers.get("Experiment-Id","")
    algorithm               = Constants.ClusteringAlgorithms.KMEANS
    logger                  = current_app.config["logger"]
    worker_id               = current_app.config["NODE_ID"] # Get the node_id from the global configuration
    STORAGE_CLIENT:AsyncClient = current_app.config["ASYNC_STORAGE_CLIENT"]
    BUCKET_ID:str           = current_app.config.get("BUCKET_ID","rory")
    plaintext_matrix_id     = filtered_headers.get("Plaintext-Matrix-Id")
    k                       = int(filtered_headers.get("K",3))
    response_headers        = {}
    MICTLANX_TIMEOUT        = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
    MICTLANX_DELAY          = int(current_app.config.get("MICTLANX_DELAY","2"))
    MICTLANX_BACKOFF_FACTOR = float(current_app.config.get("MICTLANX_BACKOFF_FACTOR","0.5"))
    MICTLANX_MAX_RETRIES    = int(current_app.config.get("MICTLANX_MAX_RETRIES","10")) 

    try:
        t1 = time.time()
        plaintext_matrix = await RoryCommon.get_matrix_or_error(
            client         = STORAGE_CLIENT,
            key            = plaintext_matrix_id,
            bucket_id      = BUCKET_ID,
            delay          = MICTLANX_DELAY,
            max_retries    = MICTLANX_MAX_RETRIES,
            timeout        = MICTLANX_TIMEOUT,
            backoff_factor = MICTLANX_BACKOFF_FACTOR
        )

        get_ptm_entry = ExperimentLogEntry(
            event         = "GET",
            experiment_id = experiment_id,
            start_time    = t1,
            end_time      = time.time(),
            algorithm     = algorithm,
            id            = plaintext_matrix_id,
            k             = k,
            iterations    = 0,
            num_chunks    = 0,
            worker_id     = worker_id,
            workers       = 0
        )
        logger.info(get_ptm_entry.model_dump())

        t1 = time.time()
        result = kMeans(
            k                = k, 
            plaintext_matrix = plaintext_matrix
        )

        clustering_entry = ExperimentLogEntry(
            event         = "COMPLETED",
            experiment_id = experiment_id,
            start_time    = local_start_time,
            end_time      = time.time(),
            algorithm     = algorithm,
            id            = plaintext_matrix_id,
            k             = k,
            iterations    = result.n_iterations,
            num_chunks    = 0,
            worker_id     = worker_id,
            workers       = 0
        )
        logger.info(clustering_entry.model_dump())
        
        response_headers["Service-Time"] = str(clustering_entry.time)
        response_headers["Iterations"]   = int(result.n_iterations)

        return Response( #Returns the final response as a label vector + the headers
            response = json.dumps({
                "label_vector":result.label_vector.tolist(),
                "iterations": result.n_iterations,
                "service_time": clustering_entry.time
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
async def dbskmeans_1(requestHeaders) -> Response:
    arrival_time            = time.time() #System startup time
    logger                  = current_app.config["logger"]
    worker_id               = current_app.config["NODE_ID"] # Get the node_id from the global configuration
    STORAGE_CLIENT:AsyncClient = current_app.config["ASYNC_STORAGE_CLIENT"]
    BUCKET_ID:str           = current_app.config.get("BUCKET_ID","rory")
    status                  = int(requestHeaders.get("Clustering-Status", Constants.ClusteringStatus.START)) 
    is_start_status         = status == Constants.ClusteringStatus.START #if status is start save it to isStartStatus
    k                       = int(requestHeaders.get("K",3)) # It is passed to integer because the headers are strings
    m                       = int(requestHeaders.get("M",3))
    algorithm               = Constants.ClusteringAlgorithms.DBSKMEANS
    plaintext_matrix_id     = requestHeaders.get("Plaintext-Matrix-Id")
    encrypted_matrix_id     = requestHeaders.get("Encrypted-Matrix-Id","")
    _encrypted_matrix_shape = requestHeaders.get("Encrypted-Matrix-Shape",-1)
    _encrypted_matrix_dtype = requestHeaders.get("Encrypted-Matrix-Dtype",-1)
    _encrypted_udm_shape    = requestHeaders.get("Encrypted-Udm-Shape",-1)
    _encrypted_udm_dtype    = requestHeaders.get("Encrypted-Udm-Dtype",-1)
    iterations              = int(requestHeaders.get("Iterations",0))
    max_iterations          = int(requestHeaders.get("Max-Iterations",0))
    experiment_id           = requestHeaders.get("Experiment-Id","")
    MICTLANX_TIMEOUT        = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
    MICTLANX_DELAY          = int(current_app.config.get("MICTLANX_DELAY","2"))
    MICTLANX_BACKOFF_FACTOR = float(current_app.config.get("MICTLANX_BACKOFF_FACTOR","0.5"))
    MICTLANX_MAX_RETRIES    = int(current_app.config.get("MICTLANX_MAX_RETRIES","10")) 

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
    
    encrypted_udm_id           = "{}encryptedudm".format(plaintext_matrix_id) 
    cent_i_id                  = "{}centi".format(plaintext_matrix_id) #Build the id of Cent_i
    cent_j_id                  = "{}centj".format(plaintext_matrix_id) #Build the id of Cent_j
    encrypted_shift_matrix_id  = "{}encryptedshiftmatrix".format(plaintext_matrix_id) #Build the id of Encrypted Shift Matrix
    dbskmeans                  = DBSKMeans()
    response_headers           = {}

    try:
        response_headers["Start-Time"] = str(arrival_time)
        get_merge_start_time = time.time()
   
        encryptedMatrix = await RoryCommon.get_and_merge(
            client         = STORAGE_CLIENT, 
            key            = encrypted_matrix_id,
            force          = False, 
            bucket_id      = BUCKET_ID,
            max_retries    = MICTLANX_MAX_RETRIES,
            timeout        = MICTLANX_TIMEOUT,
            backoff_factor = MICTLANX_BACKOFF_FACTOR,
            delay          = MICTLANX_DELAY, 
        )
        response_headers["Encrypted-Matrix-Dtype"] = str(encryptedMatrix.dtype)
        response_headers["Encrypted-Matrix-Shape"] = str(encryptedMatrix.shape)
        
        get_encrypted_ptm_entry = ExperimentLogEntry(
            event          = "GET",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_merge_start_time,
            end_time       = time.time(),
            id             = encrypted_matrix_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k,
            m              = m,
        )
        logger.info(get_encrypted_ptm_entry.model_dump())

        get_merge_start_time = time.time()
        encrypted_udm = await RoryCommon.get_and_merge(
            bucket_id      = BUCKET_ID,
            key            = encrypted_udm_id,
            force          = True,
            max_retries    = MICTLANX_MAX_RETRIES,
            delay          = MICTLANX_DELAY,
            timeout        = MICTLANX_TIMEOUT,
            backoff_factor = MICTLANX_BACKOFF_FACTOR,
            client         = STORAGE_CLIENT,
        )
        
        get_encrypted_ptm_entry = ExperimentLogEntry(
            event          = "GET",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_merge_start_time,
            end_time       = time.time(),
            id             = encrypted_udm_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k,
            m              = m,
        )
        logger.info(get_encrypted_ptm_entry.model_dump())

        response_headers["Encrypted-Udm-Dtype"] = str(encrypted_udm.dtype) # Extract the type
        response_headers["Encrypted-Udm-Shape"] = str(encrypted_udm.shape) # Extract the shape
        
        if is_start_status: #if the status is start
            __Cent_j = NONE #There is no Cent_j
        else: 
            get_matrix_cent_i_start_time = time.time()
            cent_j_value= await RoryCommon.get_and_merge(
                bucket_id      = BUCKET_ID,
                key            = cent_i_id,
                force          = True, 
                client         = STORAGE_CLIENT,
                max_retries    = MICTLANX_MAX_RETRIES,
                backoff_factor = MICTLANX_BACKOFF_FACTOR,
                delay          = MICTLANX_DELAY,
                timeout        = MICTLANX_TIMEOUT,
            )
            __Cent_j = cent_j_value.copy()
            status   = Constants.ClusteringStatus.WORK_IN_PROGRESS

            get_cent_entry = ExperimentLogEntry(
                event          = "GET",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = get_matrix_cent_i_start_time,
                end_time       = time.time(),
                id             = cent_i_id,
                worker_id      = worker_id,
                num_chunks     = num_chunks,
                k              = k,
                m              = m,
            )
            logger.info(get_cent_entry.model_dump())

        run1_start_time = time.time()
        run1_result     = dbskmeans.run1(
            encrypted_matrix = encryptedMatrix,
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
            return Response(f"Failed dbskmeans.run1(): {error}",status=500)
        
        S1, Cent_i, Cent_j, label_vector = run1_result.unwrap()
        run1_st = time.time() - run1_start_time
        
        run1_entry = ExperimentLogEntry(
            event          = "RUN1",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = run1_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k,
            m              = m
        ) 
        logger.info(run1_entry.model_dump())
    
        put_ndarray_start_time = time.time()
        maybe_cent_i_chunks = Chunks.from_ndarray(
            ndarray      = Cent_i,
            group_id     = cent_i_id,
            chunk_prefix = Some(cent_i_id),
            num_chunks   = k,
        )
        if maybe_cent_i_chunks.is_none:
            raise "Something went wrong creating the chunks: Cent_i"       

        del_put_result_cent_i = await RoryCommon.delete_and_put_chunks(
            client    = STORAGE_CLIENT,
            bucket_id = BUCKET_ID,
            key       = cent_i_id,
            chunks    = maybe_cent_i_chunks.unwrap(),
            timeout   = MICTLANX_TIMEOUT,
            max_tries = MICTLANX_MAX_RETRIES,
            tags      = {
                "full_shape": str(Cent_i.shape),
                "full_dtype": str(Cent_i.dtype)
            }
        )
        
        if del_put_result_cent_i.is_err:
            error = str(del_put_result_cent_i.unwrap_err())
            logger.error({"msg":error})
            return Response(error,status=500)      
        put_ndarray_st = time.time() - put_ndarray_start_time

        put_cent_i_entry = ExperimentLogEntry(
            event          = "PUT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = put_ndarray_start_time,
            end_time       = time.time(),
            id             = cent_i_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k,
            m              = m,
        ) 
        logger.info(put_cent_i_entry.model_dump())

        del maybe_cent_i_chunks
        del Cent_i

        put_ndarray_start_time = time.time()

        t1 = time.time()
        maybe_cent_j_chunks = Chunks.from_ndarray(
            ndarray      = Cent_j,
            group_id     = cent_j_id,
            chunk_prefix = Some(cent_j_id),
            num_chunks   = k,
        )
        if maybe_cent_j_chunks.is_none:
            raise "Something went wrong creating the chunks: Cent_j"

        y = await RoryCommon.delete_and_put_chunks(
            client    = STORAGE_CLIENT,
            bucket_id = BUCKET_ID,
            key       = cent_j_id,
            chunks    = maybe_cent_j_chunks.unwrap(),
            timeout   = MICTLANX_TIMEOUT,
            max_tries = MICTLANX_MAX_RETRIES,
            tags      = {
                "full_shape": str(Cent_j.shape),
                "full_dtype": str(Cent_j.dtype)
            }
        )

        if y.is_err:
            return Response("Failed put cent_j",status=500)
        
        put_ndarray_st = time.time() - put_ndarray_start_time

        put_cent_j_entry = ExperimentLogEntry(
                event          = "PUT",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = t1,
                end_time       = time.time(),
                id             = cent_j_id,
                worker_id      = worker_id,
                num_chunks     = num_chunks,
                k              = k,
                m              = m,
        ) 
        logger.info(put_cent_j_entry.model_dump())

        del maybe_cent_j_chunks
        del Cent_j

        put_ndarray_start_time = time.time()
        maybe_s1_chunks = Chunks.from_ndarray(
            ndarray      = S1,
            group_id     = encrypted_shift_matrix_id,
            chunk_prefix = Some(encrypted_shift_matrix_id),
            num_chunks   = k,
        )

        if maybe_s1_chunks.is_none:
            raise "Something went wrong creating the chunks: S1"

        z = await RoryCommon.delete_and_put_chunks(
            client    = STORAGE_CLIENT,
            bucket_id = BUCKET_ID,
            key       = encrypted_shift_matrix_id,
            chunks    = maybe_s1_chunks.unwrap(),
            timeout   = MICTLANX_TIMEOUT,
            max_tries = MICTLANX_MAX_RETRIES,
            tags      = {
                "full_shape": str(S1.shape),
                "full_dtype": str(S1.dtype)
            }
        )

        if z.is_err:
            logger.error({
                "msg":str(z.unwrap_err())
            })
            return Response("Failed to put: Encryted shift matrix",status=500)

        put_encrypted_sm_entry = ExperimentLogEntry(
            event          = "PUT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = put_ndarray_start_time,
            end_time       = time.time(),
            id             = encrypted_shift_matrix_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k,
            m              = m,
        ) 
        logger.info(put_encrypted_sm_entry.model_dump())
        
        del maybe_s1_chunks
        del S1

        end_time                                      = time.time()
        service_time                                  = end_time - arrival_time
        response_headers["Service-Time"]              = str(service_time)
        response_headers["Iterations"]                = str(int(requestHeaders.get("Iterations",0)) + 1) #Saves the number of iterations in the header
        response_headers["Encrypted-Shift-Matrix-Id"] = encrypted_shift_matrix_id #Save the id of the encrypted shift matrix

        clustering_entry = ExperimentLogEntry(
            event          = "COMPLETED",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = arrival_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k,
            workers        = 0,
            security_level = 0,
            m              = m,
        )
        logger.info(clustering_entry.model_dump())
        
        return Response( #Returns the final response as a label vector + the headers
            response = json.dumps({
                "label_vector":label_vector,
                "encrypted_shift_matrix_id":encrypted_shift_matrix_id,
                "n_iterations":iterations,
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
async def dbskmeans_2(requestHeaders):
    local_start_time        = time.time()
    logger                  = current_app.config["logger"]
    worker_id               = current_app.config["NODE_ID"]
    BUCKET_ID:str           = current_app.config.get("BUCKET_ID","rory")
    STORAGE_CLIENT:AsyncClient = current_app.config["ASYNC_STORAGE_CLIENT"]
    algorithm               = Constants.ClusteringAlgorithms.DBSKMEANS
    status                  = int(requestHeaders.get("Clustering-Status",Constants.ClusteringStatus.START))
    k                       = int(requestHeaders.get("K",3))
    m                       = int(requestHeaders.get("M",3))
    num_chunks              = int(requestHeaders.get("Num-Chunks",4))
    iterations              = int(requestHeaders.get("Iterations",0))
    global_start_time       = float(requestHeaders.get("Start-Time","0.0"))
    experiment_id           = requestHeaders.get("Experiment-Id","")
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

    min_error               = float(current_app.config.get("MIN_ERROR", 0.015))
    encrypted_matrix_shape:tuple = eval(_encrypted_matrix_shape)
    encrypted_udm_shape:tuple    = eval(_encrypted_udm_shape)
    encrypted_udm_id             = "{}encryptedudm".format(plaintext_matrix_id)
    cent_i_id                    = "{}centi".format(plaintext_matrix_id) #Build the id of Cent_i
    cent_j_id                    = "{}centj".format(plaintext_matrix_id) #Build the id of Cent_j
    response_headers             = {}
    
    MICTLANX_TIMEOUT        = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
    MICTLANX_DELAY          = int(current_app.config.get("MICTLANX_DELAY","2"))
    MICTLANX_BACKOFF_FACTOR = float(current_app.config.get("MICTLANX_BACKOFF_FACTOR","0.5"))
    MICTLANX_MAX_RETRIES    = int(current_app.config.get("MICTLANX_MAX_RETRIES","10")) 
    
    try:
        get_merge_start_time = time.time()
        prev_encrypted_udm = await RoryCommon.get_and_merge(
            client         = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = encrypted_udm_id,
            force          = True,
            max_retries    = MICTLANX_MAX_RETRIES,
            delay          = MICTLANX_DELAY,
            timeout        = MICTLANX_TIMEOUT, 
            backoff_factor = MICTLANX_BACKOFF_FACTOR,
        )

        get_udm_entry = ExperimentLogEntry(
            event          = "GET",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_merge_start_time,
            end_time       = time.time(),
            id             = encrypted_udm_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k,
            m              = m,
            iterations     = iterations
        )
        logger.info(get_udm_entry.model_dump())
      
        get_matrix_start_time = time.time()
        cent_i = await RoryCommon.get_and_merge(
            client         = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            force          = True,
            key            = cent_i_id,
            max_retries    = MICTLANX_MAX_RETRIES,
            delay          = MICTLANX_DELAY,
            timeout        = MICTLANX_TIMEOUT, 
            backoff_factor = MICTLANX_BACKOFF_FACTOR,     
        )

        get_cent_i_entry = ExperimentLogEntry(
            event          = "GET",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_matrix_start_time,
            end_time       = time.time(),
            id             = cent_i_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k,
            m              = m,
            iterations     = iterations
        )
        logger.info(get_cent_i_entry.model_dump())

        get_matrix_start_time = time.time()
        cent_j = await RoryCommon.get_and_merge(
            bucket_id      = BUCKET_ID,
            key            = cent_j_id,
            force          = True,
            client         = STORAGE_CLIENT,
            backoff_factor = MICTLANX_BACKOFF_FACTOR,
            delay          = MICTLANX_DELAY,
            max_retries    = MICTLANX_MAX_RETRIES,
            timeout        = MICTLANX_TIMEOUT
        ) 
        
        get_cent_j_entry = ExperimentLogEntry(
            event          = "GET",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_matrix_start_time,
            end_time       = time.time(),
            id             = cent_j_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k,
            m              = m,
            iterations     = iterations
        )
        logger.info(get_cent_j_entry.model_dump())

        isZero = Utils.verify_mean_error(
            old_matrix = cent_i, 
            new_matrix = cent_j,
            min_error  = min_error
        )
        
        if(isZero): #If Shift matrix is zero
            response_headers["Clustering-Status"]  = Constants.ClusteringStatus.COMPLETED #Change the status to COMPLETED
            end_time                               = time.time()
            service_time                           = end_time - local_start_time
            response_time                          = end_time - float(global_start_time) #The service time is calculated
            response_headers["Total-Service-Time"] = str(response_time) #Save the service time
            
            clustering_completed_entry = ExperimentLogEntry(
                event          = "COMPLETED",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = local_start_time,
                end_time       = time.time(),
                id             = plaintext_matrix_id,
                worker_id      = worker_id,
                num_chunks     = num_chunks,
                k              = k,
                m              = m,
                iterations     = iterations
            )
            logger.info(clustering_completed_entry.model_dump())

            return Response( #Return none and headers
                response = json.dumps({
                    "response_time":str(response_time),
                    "end_time":end_time,
                    "service_time":service_time,
                    "encrypted_udm_shape":str(prev_encrypted_udm.shape),
                    "encrypted_udm_dtype":str(prev_encrypted_udm.dtype),
                }), 
                status   = 200, 
                headers  = {**requestHeaders, **response_headers}
            )
        else: #If Shift matrix is not zero
            dbskmeans = DBSKMeans() 
            status    = Constants.ClusteringStatus.WORK_IN_PROGRESS #The status is changed to WORK IN PROGRESS
            response_headers["Clustering-Status"] = status

            get_matrix_start_time = time.time()
            shift_matrix_ope = await RoryCommon.get_and_merge(
                bucket_id      = BUCKET_ID,
                key            = shift_matrix_ope_id,
                client         = STORAGE_CLIENT,
                force          = True,
                timeout        = MICTLANX_TIMEOUT,
                backoff_factor = MICTLANX_BACKOFF_FACTOR,
                delay          = MICTLANX_DELAY,
                max_retries    = MICTLANX_MAX_RETRIES,
            )
            # shift_matrix_ope:npt.NDArray = shift_matrix_ope_response

            get_sm_entry = ExperimentLogEntry(
                event          = "GET",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = get_matrix_start_time,
                end_time       = time.time(),
                id             = shift_matrix_ope_id,
                worker_id      = worker_id,
                num_chunks     = num_chunks,
                k              = k,
                m              = m,
                iterations     = iterations
            )
            logger.info(get_sm_entry.model_dump())
            
            udm_start_time = time.time()
            current_udm = dbskmeans.run_2( # The second part of the skmeans starts
                k           = k,
                UDM         = prev_encrypted_udm,
                attributes  = int(encrypted_matrix_shape[1]),
                shiftMatrix = shift_matrix_ope,
            )
            
            response_headers["Encrypted-Udm-Dtype"] = str(current_udm.dtype)
            response_headers["Encrypted-Udm-Shape"] = str(current_udm.shape) # Extract the shape
            
            run_2_entry = ExperimentLogEntry(
                event          = "RUN2",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = udm_start_time,
                end_time       = time.time(),
                id             = plaintext_matrix_id,
                worker_id      = worker_id,
                num_chunks     = num_chunks,
                k              = k,
                m              = m,
                iterations     = iterations
            )
            logger.info(run_2_entry.model_dump())

            maybe_udm_chunks:Option[Chunks] = Chunks.from_ndarray(
                ndarray      = current_udm,
                group_id     = encrypted_udm_id,
                num_chunks   = num_chunks,
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
            cm_shape = str(current_udm.shape)
            cm_dtype = str(current_udm.dtype)
            del current_udm
         
            put_chunks_start_time = time.time()  
            put_chunks_udm_generator_results = await RoryCommon.delete_and_put_chunks(
                client    = STORAGE_CLIENT,
                bucket_id = BUCKET_ID,
                key       = encrypted_udm_id, 
                chunks    = udm_chunks, 
                timeout   = MICTLANX_TIMEOUT,
                max_tries = MICTLANX_MAX_RETRIES,
                tags      = {
                    "full_shape":cm_shape,
                    "full_dtype":cm_dtype,
                }
            )
            if put_chunks_udm_generator_results.is_err:
                return Response(
                    status=500,
                    response="Failed to put encrypted udm"
                )
            
            put_udm_entry = ExperimentLogEntry(
                event          = "PUT",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = put_chunks_start_time,
                end_time       = time.time(),
                id             = encrypted_udm_id,
                worker_id      = worker_id,
                num_chunks     = num_chunks,
                k              = k,
                m              = m,
                iterations     = iterations
            )
            logger.info(put_udm_entry.model_dump())

            del udm_chunks            
            end_time                         = time.time()
            service_time                     = end_time - local_start_time  #Service time is calculated
            response_headers["End-Time"]     = str(end_time)
            response_headers["Service-Time"] = str(service_time)

            clutering_uncompleted_entry = ExperimentLogEntry(
                event          = "UNCOMPLETED",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = local_start_time,
                end_time       = time.time(),
                id             = plaintext_matrix_id,
                worker_id      = worker_id,
                num_chunks     = num_chunks,
                k              = k,
                m              = m,
                iterations     = iterations
            )
            logger.info(clutering_uncompleted_entry.model_dump())

            del prev_encrypted_udm
            
            return Response( #Return none and headers
                response = json.dumps({
                    "end_time":end_time,
                    "service_time":service_time,
                    "encrypted_udm_shape":cm_shape,
                    "encrypted_udm_dtype":str(cm_dtype),
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
async def dbskmeans():
    headers         = request.headers
    head            = ["User-Agent","Accept-Encoding","Connection"]
    logger          = current_app.config["logger"]
    filteredHeaders = dict(list(filter(lambda x: not x[0] in head, headers.items())))
    step_index      = int(filteredHeaders.get("Step-Index",1))
    response        = Response("Failed invalid step_index", status=400)
    if step_index == 1:
        return await dbskmeans_1(filteredHeaders)
    elif step_index == 2:
        return await dbskmeans_2(filteredHeaders)
    else:
        return response


@clustering.route("/dbsnnc", methods = ["POST"])
async def dbsnnc():
    local_start_time           = time.time() #System startup time
    headers                    = request.headers
    to_remove_headers          = ["User-Agent","Accept-Encoding","Connection"]
    filtered_headers           = dict(list(filter(lambda x: not x[0] in to_remove_headers, headers.items())))
    algorithm                  = Constants.ClusteringAlgorithms.DBSNNC
    logger                     = current_app.config["logger"]
    STORAGE_CLIENT:AsyncClient = current_app.config["ASYNC_STORAGE_CLIENT"]
    BUCKET_ID:str              = current_app.config.get("BUCKET_ID","rory")
    worker_id                  = current_app.config["NODE_ID"] # Get the node_id from the global configuration
    plaintext_matrix_id        = filtered_headers.get("Plaintext-Matrix-Id")
    encrypted_matrix_id        = filtered_headers.get("Encrypted-Matrix-Id",-1)
    encrypted_dm_id            = filtered_headers.get("Encrypted-Dm-Id")
    encrypted_threshold        = float(filtered_headers.get("Encrypted-Threshold"))
    _encrypted_matrix_shape    = filtered_headers.get("Encrypted-Matrix-Shape",-1)
    _encrypted_matrix_dtype    = filtered_headers.get("Encrypted-Matrix-Dtype",-1)
    _encrypted_dm_shape        = filtered_headers.get("Encrypted-Dm-Shape",-1)
    _encrypted_dm_dtype        = filtered_headers.get("Encrypted-Dm-Dtype",-1)
    m                          = int(filtered_headers.get("M",3))
    experiment_id              = filtered_headers.get("Experiment-Id","")
    MICTLANX_TIMEOUT           = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
    MICTLANX_DELAY             = int(current_app.config.get("MICTLANX_DELAY","2"))
    MICTLANX_BACKOFF_FACTOR    = float(current_app.config.get("MICTLANX_BACKOFF_FACTOR","0.5"))
    MICTLANX_MAX_RETRIES       = int(current_app.config.get("MICTLANX_MAX_RETRIES","10")) 

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
    
    try:      
        responseHeaders["Start-Time"] = str(local_start_time)

        get_merge_encrypted_matrix_start_time = time.time()
        encryptedMatrix = await RoryCommon.get_and_merge(
            client         = STORAGE_CLIENT,
            key            = encrypted_matrix_id,
            bucket_id      = BUCKET_ID,
            max_retries    = MICTLANX_MAX_RETRIES,
            delay          = MICTLANX_DELAY,
            backoff_factor = MICTLANX_BACKOFF_FACTOR,
            timeout        = MICTLANX_TIMEOUT
        )
        
        get_encrypted_ptm_entry = ExperimentLogEntry(
            event          = "GET",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_merge_encrypted_matrix_start_time,
            end_time       = time.time(),
            id             = encrypted_matrix_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            m              = m,
        )
        logger.info(get_encrypted_ptm_entry.model_dump())

        responseHeaders["Encrypted-Matrix-Dtype"] = encryptedMatrix.dtype #["tags"]["dtype"] #Save the data type
        responseHeaders["Encrypted-Matrix-Shape"] = encryptedMatrix.shape #Save the shape
  
        get_merge_encrypted_dm_start_time = time.time()
        distance_matrix:npt.NDArray = await RoryCommon.get_and_merge(
            client         = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = encrypted_dm_id,
            max_retries    = MICTLANX_MAX_RETRIES,
            delay          = MICTLANX_DELAY,
            backoff_factor = MICTLANX_BACKOFF_FACTOR,
            timeout        = MICTLANX_TIMEOUT
        )
        
        get_encrypted_ptm_entry = ExperimentLogEntry(
            event          = "GET",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_merge_encrypted_dm_start_time,
            end_time       = time.time(),
            id             = encrypted_dm_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            m              = m,
        )
        logger.info(get_encrypted_ptm_entry.model_dump())

        responseHeaders["Encrypted-Dm-Dtype"] = distance_matrix.dtype # Extract the type
        responseHeaders["Encrypted-Dm-Shape"] = distance_matrix.shape # Extract the shape
        
        dbsnnc_run_start_time = time.time()
        result = Dbsnnc.run(
            distance_matrix     = distance_matrix,
            encrypted_threshold = encrypted_threshold
        )
        end_time            = time.time()
        dbsnnc_service_time = end_time - dbsnnc_run_start_time
        service_time        = end_time - local_start_time

        clustering_entry = ExperimentLogEntry(
            event          = "COMPLETED",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = local_start_time,
            end_time       = time.time(),
            id             = encrypted_matrix_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            workers        = 0,
            security_level = 0,
            m              = m,
        )
        logger.info(clustering_entry.model_dump())
        
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
async def nnc():
    local_start_time           = time.time() #System startup time
    headers                    = request.headers
    to_remove_headers          = ["User-Agent","Accept-Encoding","Connection"]
    filtered_headers           = dict(list(filter(lambda x: not x[0] in to_remove_headers, headers.items())))
    algorithm                  = Constants.ClusteringAlgorithms.NNC
    logger                     = current_app.config["logger"]
    STORAGE_CLIENT:AsyncClient = current_app.config["ASYNC_STORAGE_CLIENT"]
    BUCKET_ID:str              = current_app.config.get("BUCKET_ID","rory")
    worker_id                  = current_app.config["NODE_ID"] # Get the node_id from the global configuration
    plaintext_matrix_id        = filtered_headers.get("Plaintext-Matrix-Id")
    threshold                  = float(filtered_headers.get("Threshold"))
    _plaintext_matrix_shape    = filtered_headers.get("Plaintext-Matrix-Shape",-1)
    _plaintext_matrix_dtype    = filtered_headers.get("Plaintext-Matrix-Dtype",-1)
    _dm_shape                  = filtered_headers.get("Dm-Shape",-1)
    _dm_dtype                  = filtered_headers.get("Dm-Dtype",-1)
    dm_id                      = "{}dm".format(plaintext_matrix_id) 
    response_headers           = {}
    experiment_id              = filtered_headers.get("Experiment-Id","")
    MICTLANX_TIMEOUT           = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
    MICTLANX_DELAY             = int(current_app.config.get("MICTLANX_DELAY","2"))
    MICTLANX_BACKOFF_FACTOR    = float(current_app.config.get("MICTLANX_BACKOFF_FACTOR","0.5"))
    MICTLANX_MAX_RETRIES       = int(current_app.config.get("MICTLANX_MAX_RETRIES","10")) 

    if _plaintext_matrix_dtype == -1:
        return Response("Encrypted-Matrix-Dtype", status=500)
    if _plaintext_matrix_shape == -1 :
        return Response("Encrypted-Matrix-Shape header is required", status=500)
    
    if _dm_dtype == -1:
        return Response("Encrypted-DM-Dtype", status=500)
    if _dm_shape == -1 :
        return Response("Encrypted-DM-Shape header is required", status=500)
    
    plaintext_matrix_shape:tuple = eval(_plaintext_matrix_shape)
    dm_shape:tuple               = eval(_dm_shape)

    num_chunks      = int(filtered_headers.get("Num-Chunks",-1))
    responseHeaders = {}
    
    if num_chunks == -1:
        return Response("Num-Chunks header is required", status=503)

    try:      
        response_headers["Start-Time"] = str(local_start_time)

        get_merge_plaintext_matrix_start_time = time.time()

        plaintextMatrix = await RoryCommon.get_and_merge(
            client         = STORAGE_CLIENT,
            key            = plaintext_matrix_id,
            bucket_id      = BUCKET_ID,
            max_retries    = MICTLANX_MAX_RETRIES,
            delay          = MICTLANX_DELAY,
            backoff_factor = MICTLANX_BACKOFF_FACTOR,
            timeout        = MICTLANX_TIMEOUT
        )

        get_merge_plaintext_matrix_st = time.time() - get_merge_plaintext_matrix_start_time
        
        get_encrypted_ptm_entry = ExperimentLogEntry(
            event          = "GET",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_merge_plaintext_matrix_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
        )
        logger.info(get_encrypted_ptm_entry.model_dump())
        
        responseHeaders["Plaintext-Matrix-Dtype"] = plaintextMatrix.dtype #["tags"]["dtype"] #Save the data type
        responseHeaders["Plaintext-Matrix-Shape"] = plaintextMatrix.shape #Save the shape

 
        get_merge_dm_start_time = time.time()
        
        distance_matrix = await RoryCommon.get_and_merge(
            client         = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = dm_id,
            max_retries    = MICTLANX_MAX_RETRIES,
            delay          = MICTLANX_DELAY,
            backoff_factor = MICTLANX_BACKOFF_FACTOR,
            timeout        = MICTLANX_TIMEOUT
        )

        get_merge_dm_st = time.time() - get_merge_dm_start_time
        
        get_encrypted_ptm_entry = ExperimentLogEntry(
            event          = "GET",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_merge_dm_start_time,
            end_time       = time.time(),
            id             = dm_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
        )
        logger.info(get_encrypted_ptm_entry.model_dump())

        responseHeaders["Dm-Dtype"] = distance_matrix.dtype # Extract the type
        responseHeaders["Dm-Shape"] = distance_matrix.shape # Extract the shape
        
        nnc_run_start_time = time.time()

        result = Nnc.run(
            distance_matrix = distance_matrix,
            threshold       = threshold
        )
        end_time         = time.time()
        nnc_run_end_time = end_time - nnc_run_start_time
        service_time     = end_time - local_start_time
        
        clustering_entry = ExperimentLogEntry(
            event          = "COMPLETED",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = local_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
        )
        logger.info(clustering_entry.model_dump())

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
  

async def pqc_skmeans_1(requestHeaders) -> Response:
    try:
        arrival_time               = time.time() #Worker start time
        logger                     = current_app.config["logger"]
        worker_id                  = current_app.config["NODE_ID"] # Get the node_id from the global configuration
        STORAGE_CLIENT:AsyncClient = current_app.config["ASYNC_STORAGE_CLIENT"]
        BUCKET_ID:str              = current_app.config.get("BUCKET_ID","rory")
        status                     = int(requestHeaders.get("Clustering-Status", Constants.ClusteringStatus.START)) 
        is_start_status            = status == Constants.ClusteringStatus.START #if status is start save it to isStartStatus
        k                          = int(requestHeaders.get("K",3)) # It is passed to integer because the headers are strings
        algorithm                  = Constants.ClusteringAlgorithms.SKMEANS_PQC
        plaintext_matrix_id        = requestHeaders.get("Plaintext-Matrix-Id")
        encrypted_matrix_id        = requestHeaders.get("Encrypted-Matrix-Id",-1)
        udm_id                     = "{}udm".format(plaintext_matrix_id) 
        _encrypted_matrix_shape    = requestHeaders.get("Encrypted-Matrix-Shape",-1)
        _encrypted_matrix_dtype    = requestHeaders.get("Encrypted-Matrix-Dtype",-1)
        experiment_id              = requestHeaders.get("Experiment-Id","")
        _round                     = bool(int(current_app.config.get("_round","0"))) #False
        decimals                   = int(current_app.config.get("DECIMALS","2"))
        path                       = current_app.config.get("KEYS_PATH","/rory/keys")
        ctx_filename               = current_app.config.get("CTX_FILENAME","ctx")
        pubkey_filename            = current_app.config.get("PUBKEY_FILENAME","pubkey")
        secretkey_filename         = current_app.config.get("SECRET_KEY_FILENAME","secretkey")
        relinkey_filename          = current_app.config.get("RELINKEY_FILENAME","relinkey")
        MICTLANX_TIMEOUT           = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
        MICTLANX_DELAY             = int(current_app.config.get("MICTLANX_DELAY","2"))
        MICTLANX_BACKOFF_FACTOR    = float(current_app.config.get("MICTLANX_BACKOFF_FACTOR","0.5"))
        MICTLANX_MAX_RETRIES       = int(current_app.config.get("MICTLANX_MAX_RETRIES","10")) 

        if _encrypted_matrix_dtype == -1:
            return Response("Encrypted-Matrix-Dtype", status=500)
        if _encrypted_matrix_shape == -1 :
            return Response("Encrypted-Matrix-Shape header is required", status=500)

        encrypted_shift_matrix_id = "{}encryptedshiftmatrix".format(plaintext_matrix_id) #Build the id of Encrypted Shift Matrix
        init_sm_id                = "{}initsm".format(plaintext_matrix_id)
        cent_i_id                 = "{}centi".format(plaintext_matrix_id) #Build the id of Cent_i
        cent_j_id                 = "{}centj".format(plaintext_matrix_id) #Build the id of Cent_j
        num_chunks                = int(requestHeaders.get("Num-Chunks",-1))
        responseHeaders           = {}        

        # _______________________________________________________________________________
        ckks = Ckks.from_pyfhel(
            _round             = _round,
            decimals           = decimals,
            path               = path,
            ctx_filename       = ctx_filename,
            pubkey_filename    = pubkey_filename,
            secretkey_filename = secretkey_filename,
            relinkey_filename  = relinkey_filename
        )
        # _______________________________________________________________________________
        
        if num_chunks == -1:
            logger.error({
                "msg":"Num-Chunks header is required"
            })
            return Response("Num-Chunks header is required", status=503)

        responseHeaders["Start-Time"] = str(arrival_time)
        
   
        get_merge_encrypted_matrix_start_time  = time.time()
        init_shiftmatrix = await RoryCommon.get_pyctxt(
            client         = STORAGE_CLIENT, 
            bucket_id      = BUCKET_ID, 
            key            = init_sm_id, 
            ckks           = ckks,
            delay          = MICTLANX_DELAY,
            max_retries    = MICTLANX_MAX_RETRIES,
            timeout        = MICTLANX_TIMEOUT,
            backoff_factor = MICTLANX_BACKOFF_FACTOR
        )

        get_init_sm_entry = ExperimentLogEntry(
            event          = "GET",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_merge_encrypted_matrix_start_time,
            end_time       = time.time(),
            id             = init_sm_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k,
        )
        logger.info(get_init_sm_entry.model_dump())
        
        skmeans = SkmeansPQC(he_object=ckks.he_object, init_shiftmatrix=init_shiftmatrix)
        get_merge_encrypted_matrix_start_time = time.time()
        encryptedMatrix = await RoryCommon.get_pyctxt(
            client         = STORAGE_CLIENT, 
            bucket_id      = BUCKET_ID, 
            key            = encrypted_matrix_id, 
            ckks           = ckks,
            delay          = MICTLANX_DELAY,
            backoff_factor = MICTLANX_BACKOFF_FACTOR,
            max_retries    = MICTLANX_MAX_RETRIES,
            timeout        = MICTLANX_TIMEOUT,
        )

        get_encrypted_matrix_entry = ExperimentLogEntry(
            event          = "GET",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_merge_encrypted_matrix_start_time,
            end_time       = time.time(),
            id             = encrypted_matrix_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k,
        )
        logger.info(get_encrypted_matrix_entry.model_dump())

        udm_get_start_time  = time.time()
        udm = await RoryCommon.get_and_merge(
            client         = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = udm_id,
            force          = True,
            delay          = MICTLANX_DELAY,
            backoff_factor = MICTLANX_BACKOFF_FACTOR,
            max_retries    = MICTLANX_MAX_RETRIES,
            timeout        = MICTLANX_TIMEOUT,
        )

        get_udm_entry = ExperimentLogEntry(
            event          = "GET",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = udm_get_start_time,
            end_time       = time.time(),
            id             = udm_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k,
        )
        logger.info(get_udm_entry.model_dump())

        responseHeaders["Udm-Matrix-Dtype"] = str(udm.dtype) # Extract the type
        responseHeaders["Udm-Matrix-Shape"] = str(udm.shape) # Extract the shape
        
        if is_start_status: #if the status is start
            __Cent_j = init_shiftmatrix #There is no Cent_j
        else: 
            cent_j_start_time = time.time()

            __Cent_j = await RoryCommon.get_pyctxt(
                client         = STORAGE_CLIENT, 
                bucket_id      = BUCKET_ID, 
                key            = cent_i_id, 
                ckks           = ckks,
                force          = True,
                delay          = MICTLANX_DELAY,
                backoff_factor = MICTLANX_BACKOFF_FACTOR,
                max_retries    = MICTLANX_MAX_RETRIES,
                timeout        = MICTLANX_TIMEOUT,
            )
            status    = Constants.ClusteringStatus.WORK_IN_PROGRESS
            cent_j_st = time.time() - cent_j_start_time

            get_udm_entry = ExperimentLogEntry(
                event          = "GET",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = udm_get_start_time,
                end_time       = time.time(),
                id             = cent_i_id,
                worker_id      = worker_id,
                num_chunks     = num_chunks,
                k              = k,
            )
            logger.info(get_udm_entry.model_dump())

        _encrypted_matrix_shape = eval(_encrypted_matrix_shape)
        run1_start_time = time.time()
        run1_result = skmeans.run1( # The first part of the skmeans is done
            status          = status,
            k               = k,
            encryptedMatrix = encryptedMatrix, 
            UDM             = udm,
            Cent_j          = __Cent_j,
            num_attributes  = _encrypted_matrix_shape[1]
        )
        
        if run1_result.is_err:
            error = run1_result.unwrap_err()
            logger.error(str(error))
            return Response(response=str(error), status=500 )
        S1,Cent_i,Cent_j,label_vector = run1_result.unwrap()

        run1_entry = ExperimentLogEntry(
            event          = "RUN1",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = run1_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k
        ) 
        logger.info(run1_entry.model_dump())

        t1 = time.time()
        maybe_cent_i_chunks = RoryCommon.from_pyctxts_to_chunks(
            key        = cent_i_id,
            xs         = Cent_i,
            num_chunks = num_chunks)
        
        if maybe_cent_i_chunks.is_none:
            return Response(status=500, response="Failed to create the Cent_i chunks")

        x = await RoryCommon.delete_and_put_chunks(
            client    = STORAGE_CLIENT,
            bucket_id = BUCKET_ID,
            key       = cent_i_id,
            chunks    = maybe_cent_i_chunks.unwrap(),
            timeout   = MICTLANX_TIMEOUT,
            max_tries = MICTLANX_MAX_RETRIES
        )

        put_cent_i_entry = ExperimentLogEntry(
            event          = "PUT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = t1,
            end_time       = time.time(),
            id             = cent_i_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k,
        ) 
        logger.info(put_cent_i_entry.model_dump())

        t1 = time.time()
        maybe_cent_j_chunks = RoryCommon.from_pyctxts_to_chunks(
            key        = cent_j_id,
            xs         = Cent_j,
            num_chunks = num_chunks
        )
        if maybe_cent_j_chunks.is_none:
            return Response(status=500, response="Failed to create the Cent_j chunks")
        y = await RoryCommon.delete_and_put_chunks(
            client    = STORAGE_CLIENT,
            bucket_id = BUCKET_ID,
            key       = cent_j_id,
            chunks    = maybe_cent_j_chunks.unwrap(),
            timeout   = MICTLANX_TIMEOUT,
            max_tries = MICTLANX_MAX_RETRIES
        )

        put_cent_j_entry = ExperimentLogEntry(
            event          = "PUT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = t1,
            end_time       = time.time(),
            id             = cent_j_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k,
        ) 
        logger.info(put_cent_j_entry.model_dump())

        t1 = time.time()
        maybe_encrypted_shift_matrix_chunks = RoryCommon.from_pyctxts_to_chunks(
            xs         = S1, 
            key        = encrypted_shift_matrix_id,
            num_chunks = num_chunks
        )
        if maybe_encrypted_shift_matrix_chunks.is_none:
            return Response(status=500, response="Failed to create the encrypted shift matrix chunks")       
        S1_chunks = maybe_encrypted_shift_matrix_chunks.unwrap()
        z = await RoryCommon.delete_and_put_chunks(
            client    = STORAGE_CLIENT,
            bucket_id = BUCKET_ID,
            key       = encrypted_shift_matrix_id,
            chunks    = S1_chunks,
            timeout   = MICTLANX_TIMEOUT,
            max_tries = MICTLANX_MAX_RETRIES
        )

        put_encrypted_sm_entry = ExperimentLogEntry(
            event          = "PUT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = t1,
            end_time       = time.time(),
            id             = encrypted_shift_matrix_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k,
        ) 
        logger.info(put_encrypted_sm_entry.model_dump())

        end_time     = time.time()
        service_time = end_time - arrival_time
        n_iterations = int(requestHeaders.get("Iterations",0)) + 1
        
        clustering_entry = ExperimentLogEntry(
            event          = "COMPLETED",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = arrival_time,
            end_time       = time.time(),
            id             = encrypted_matrix_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k,
            iterations     = n_iterations
        )
        logger.info(clustering_entry.model_dump())

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


async def pqc_skmeans_2(requestHeaders):
    local_start_time        = time.time()
    logger                  = current_app.config["logger"]
    worker_id               = current_app.config["NODE_ID"]
    BUCKET_ID:str           = current_app.config.get("BUCKET_ID","rory")
    STORAGE_CLIENT:AsyncClient = current_app.config["ASYNC_STORAGE_CLIENT"]
    algorithm               = Constants.ClusteringAlgorithms.SKMEANS_PQC
    status                  = int(requestHeaders.get("Clustering-Status",Constants.ClusteringStatus.START))
    plaintext_matrix_id     = requestHeaders["Plaintext-Matrix-Id"]
    encrypted_matrix_id     = requestHeaders["Encrypted-Matrix-Id"]
    shift_matrix_id         = requestHeaders.get("Shift-Matrix-Id","{}shiftmatrix".format(plaintext_matrix_id))
    k                       = int(requestHeaders.get("K",3))
    isZero                  = bool(int(requestHeaders.get("Is-Zero")))
    iterations              = int(requestHeaders.get("Iterations",0))
    experiment_id           = requestHeaders.get("Experiment-Id","")
    init_sm_id              = "{}initsm".format(plaintext_matrix_id)

    MICTLANX_TIMEOUT        = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
    MICTLANX_DELAY          = int(current_app.config.get("MICTLANX_DELAY","2"))
    MICTLANX_BACKOFF_FACTOR = float(current_app.config.get("MICTLANX_BACKOFF_FACTOR","0.5"))
    MICTLANX_MAX_RETRIES    = int(current_app.config.get("MICTLANX_MAX_RETRIES","10"))
    _round                  = bool(int(current_app.config.get("_round","0"))) #False
    decimals                = int(current_app.config.get("DECIMALS","2"))
    path                    = current_app.config.get("KEYS_PATH","/rory/keys")
    ctx_filename            = current_app.config.get("CTX_FILENAME","ctx")
    pubkey_filename         = current_app.config.get("PUBKEY_FILENAME","pubkey")
    secretkey_filename      = current_app.config.get("SECRET_KEY_FILENAME","secretkey")
    relinkey_filename       = current_app.config.get("RELINKEY_FILENAME","relinkey")
    
    if encrypted_matrix_id == -1 or plaintext_matrix_id == -1:
        return Response("Either Encrypted-Matrix-Id or Plain-Matrix-Id is missing",status=500)
    num_chunks       = int(requestHeaders.get("Num-Chunks",-1))
    udm_id           = "{}udm".format(plaintext_matrix_id)
    cent_i_id        = "{}centi".format(plaintext_matrix_id) #Build the id of Cent_i
    cent_j_id        = "{}centj".format(plaintext_matrix_id) #Build the id of Cent_j
    response_headers = {}

    ckks = Ckks.from_pyfhel(
        _round   = _round,
        decimals = decimals,
        path               = path,
        ctx_filename       = ctx_filename,
        pubkey_filename    = pubkey_filename,
        secretkey_filename = secretkey_filename,
        relinkey_filename  = relinkey_filename
    )

    try:
        get_UDM_start_time = time.time()
        UDM =  await RoryCommon.get_and_merge(
            client         = STORAGE_CLIENT,
            key            = udm_id,
            bucket_id      = BUCKET_ID,
            force          = True,
            delay          = MICTLANX_DELAY,
            timeout        = MICTLANX_TIMEOUT,
            backoff_factor = MICTLANX_BACKOFF_FACTOR,
            max_retries    = MICTLANX_MAX_RETRIES
        )

        get_udm_entry = ExperimentLogEntry(
            event          = "GET",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_UDM_start_time,
            end_time       = time.time(),
            id             = init_sm_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k,
        )
        logger.info(get_udm_entry.model_dump())
        
        get_UDM_st = time.time() - get_UDM_start_time
        get_shift_matrix_start_time = time.time()

        shiftMatrix = await RoryCommon.get_and_merge(
            client         = STORAGE_CLIENT,
            key            = shift_matrix_id,
            bucket_id      = BUCKET_ID,
            force          = True,
            delay          = MICTLANX_DELAY,
            backoff_factor = MICTLANX_BACKOFF_FACTOR,
            max_retries    = MICTLANX_MAX_RETRIES,
            timeout        = MICTLANX_TIMEOUT,
        )

        get_sm_entry = ExperimentLogEntry(
            event          = "GET",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_shift_matrix_start_time,
            end_time       = time.time(),
            id             = encrypted_matrix_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k,
        )
        logger.info(get_sm_entry.model_dump())
        
        if(isZero): #If Shift matrix is zero
            response_headers["Clustering-Status"]  = Constants.ClusteringStatus.COMPLETED #Change the status to COMPLETED
            end_time                               = time.time()
            service_time                           = end_time - local_start_time #The service time is calculated
            response_headers["Total-Service-Time"] = str(service_time) #Save the service time

            clustering_entry = ExperimentLogEntry(
                event          = "COMPLETED",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = local_start_time,
                end_time       = time.time(),
                id             = encrypted_matrix_id,
                worker_id      = worker_id,
                num_chunks     = num_chunks,
                k              = k,
                iterations     = iterations
            )
            logger.info(clustering_entry.model_dump())

            return Response( #Return none and headers
                response = None, 
                status   = 204, 
                headers  = response_headers
            )
        
        else: #If Shift matrix is not zero
            t1 = time.time()
            init_shiftmatrix = await RoryCommon.get_pyctxt(
                client         = STORAGE_CLIENT, 
                bucket_id      = BUCKET_ID, 
                key            = init_sm_id, 
                ckks           = ckks,
                delay          = MICTLANX_DELAY,
                backoff_factor = MICTLANX_BACKOFF_FACTOR,
                max_retries    = MICTLANX_MAX_RETRIES,
                timeout        = MICTLANX_TIMEOUT,
            )

            get_init_sm_entry = ExperimentLogEntry(
                event          = "GET",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = t1,
                end_time       = time.time(),
                id             = init_sm_id,
                worker_id      = worker_id,
                num_chunks     = num_chunks,
                k              = k,
            )
            logger.info(get_init_sm_entry.model_dump())

            skmeans = SkmeansPQC(he_object=ckks.he_object, init_shiftmatrix=init_shiftmatrix) 
            status  = Constants.ClusteringStatus.WORK_IN_PROGRESS

            response_headers["Clustering-Status"] = status #The status is changed to WORK IN PROGRESS
            encrypted_matrix_shape = eval(requestHeaders["Encrypted-Matrix-Shape"])

            run2_start_time = time.time()
            _UDM = skmeans.run_2( # The second part of the skmeans starts
                k              = k,
                UDM            = UDM,
                num_attributes = int(encrypted_matrix_shape[1]),
                shiftMatrix    = shiftMatrix,
            )
            UDM_array = np.array(_UDM)

            run2_entry = ExperimentLogEntry(
                event          = "RUN2",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = run2_start_time,
                end_time       = time.time(),
                id             = plaintext_matrix_id,
                worker_id      = worker_id,
                num_chunks     = num_chunks,
                k              = k
            ) 
            logger.info(run2_entry.model_dump())

            put_udm_start_time = time.time()
            maybe_udm_chunks = Chunks.from_ndarray(
                ndarray      = UDM_array,
                group_id     = udm_id,
                chunk_prefix = Some(udm_id),
                num_chunks   = num_chunks,
            )
            if maybe_udm_chunks.is_none:
                return Response(status=500, response="something went wrong creating the chunks")
            
            put_udm_result = await RoryCommon.delete_and_put_chunks(
                client    = STORAGE_CLIENT,
                bucket_id = BUCKET_ID,
                key       = udm_id,
                chunks    = maybe_udm_chunks.unwrap(),
                timeout   = MICTLANX_TIMEOUT,
                max_tries = MICTLANX_MAX_RETRIES,
                tags      = {
                    "full_shape": str(UDM_array.shape),
                    "full_dtype": str(UDM_array.dtype)
                }
            )
            if put_udm_result.is_err:
                error = str(put_udm_result.unwrap_err())
                logger.error({
                    "msg":error
                })
                return Response(error,status=500)
            endTime2   = time.time()

            put_udm_entry = ExperimentLogEntry(
                event          = "PUT",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = put_udm_start_time,
                end_time       = time.time(),
                id             = udm_id,
                worker_id      = worker_id,
                num_chunks     = num_chunks,
                k              = k,
            ) 
            logger.info(put_udm_entry.model_dump())

            serviceTime2                     = endTime2 - local_start_time  #Service time is calculated
            response_headers["End-Time"]     = str(endTime2)
            response_headers["Service-Time"] = str(serviceTime2)
            
            clutering_uncompleted_entry = ExperimentLogEntry(
                event          = "UNCOMPLETED",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = local_start_time,
                end_time       = time.time(),
                id             = plaintext_matrix_id,
                worker_id      = worker_id,
                num_chunks     = num_chunks,
                k              = k,
                iterations     = iterations
            )
            logger.info(clutering_uncompleted_entry.model_dump())

            return Response( #Return none and headers
                response = None,
                status   = 204, 
                headers  = response_headers
            )

    except Exception as e:
        logger.error("SKMEANS_2_ERROR: "+encrypted_matrix_id+" "+str(e))
        return Response(str(e),status = 503)


@clustering.route("/pqc/skmeans",methods = ["POST"])
async def pqc_skmeans():
    headers         = request.headers
    head            = ["User-Agent","Accept-Encoding","Connection"]
    filteredHeaders = dict(list(filter(lambda x: not x[0] in head, headers.items())))
    step_index      = int(filteredHeaders.get("Step-Index",1))
    response        = Response(response="Invalid step index", status=500)
    logger                  = current_app.config["logger"]

    logger.info({
        "X":1,
        "step_index":step_index
    })
    if step_index == 1:
        return await pqc_skmeans_1(filteredHeaders)
    elif step_index == 2:
        return await pqc_skmeans_2(filteredHeaders)
    else:
        return response


async def pqc_dbskmeans_1(requestHeaders):
    try:
        arrival_time            = time.time() #Worker start time
        logger                  = current_app.config["logger"]
        worker_id               = current_app.config["NODE_ID"] # Get the node_id from the global configuration
        STORAGE_CLIENT:AsyncClient = current_app.config["ASYNC_STORAGE_CLIENT"]
        BUCKET_ID:str           = current_app.config.get("BUCKET_ID","rory")
        status                  = int(requestHeaders.get("Clustering-Status", Constants.ClusteringStatus.START)) 
        is_start_status         = status == Constants.ClusteringStatus.START #if status is start save it to isStartStatus
        k                       = int(requestHeaders.get("K",3)) # It is passed to integer because the headers are strings
        algorithm               = Constants.ClusteringAlgorithms.DBSKMEANS_PQC
        plaintext_matrix_id     = requestHeaders.get("Plaintext-Matrix-Id")
        encrypted_matrix_id     = requestHeaders.get("Encrypted-Matrix-Id",-1)
        udm_id                  = "{}udm".format(plaintext_matrix_id) 
        _encrypted_matrix_shape = requestHeaders.get("Encrypted-Matrix-Shape",-1)
        _encrypted_matrix_dtype = requestHeaders.get("Encrypted-Matrix-Dtype",-1)
        _encrypted_udm_shape    = requestHeaders.get("Encrypted-Udm-Shape",-1)
        _encrypted_udm_dtype    = requestHeaders.get("Encrypted-Udm-Dtype",-1)
        iterations              = int(requestHeaders.get("Iterations",0))
        experiment_id           = requestHeaders.get("Experiment-Id","")

        _round             = bool(int(current_app.config.get("_round","0"))) #False
        decimals           = int(current_app.config.get("DECIMALS","2"))
        path               = current_app.config.get("KEYS_PATH","/rory/keys")
        ctx_filename       = current_app.config.get("CTX_FILENAME","ctx")
        pubkey_filename    = current_app.config.get("PUBKEY_FILENAME","pubkey")
        secretkey_filename = current_app.config.get("SECRET_KEY_FILENAME","secretkey")
        relinkey_filename  = current_app.config.get("RELINKEY_FILENAME","relinkey")
        
        if _encrypted_matrix_dtype == -1:
            return Response("Encrypted-Matrix-Dtype", status=400)
        if _encrypted_matrix_shape == -1 :
            return Response("Encrypted-Matrix-Shape header is required", status=400)
        # logger.info("AQUI WORKER")
        if _encrypted_udm_dtype == -1:
            return Response("Encrypted-UDM-Dtype", status=400)
        if _encrypted_udm_shape == -1 :
            return Response("Encrypted-UDM-Shape header is required", status=400)
        
        num_chunks                   = int(requestHeaders.get("Num-Chunks",-1))
        encrypted_matrix_shape:tuple = eval(_encrypted_matrix_shape)
        encrypted_udm_shape:tuple    = eval(_encrypted_udm_shape)

        encrypted_shift_matrix_id = "{}encryptedshiftmatrix".format(plaintext_matrix_id) #Build the id of Encrypted Shift Matrix
        encrypted_udm_id          = "{}encryptedudm".format(plaintext_matrix_id) 
        init_sm_id                = "{}initsm".format(plaintext_matrix_id)
        cent_i_id                 = "{}centi".format(plaintext_matrix_id) #Build the id of Cent_i
        cent_j_id                 = "{}centj".format(plaintext_matrix_id) #Build the id of Cent_j
        responseHeaders           = {}

        MICTLANX_TIMEOUT        = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
        MICTLANX_DELAY          = int(current_app.config.get("MICTLANX_DELAY","2"))
        MICTLANX_BACKOFF_FACTOR = float(current_app.config.get("MICTLANX_BACKOFF_FACTOR","0.5"))
        MICTLANX_MAX_RETRIES    = int(current_app.config.get("MICTLANX_MAX_RETRIES","10"))

        # _______________________________________________________________________________
        ckks = Ckks.from_pyfhel(
            _round   = _round,
            decimals = decimals,
            path               = path,
            ctx_filename       = ctx_filename,
            pubkey_filename    = pubkey_filename,
            secretkey_filename = secretkey_filename,
            relinkey_filename  = relinkey_filename
        )
        # _______________________________________________________________________________
        
        if num_chunks == -1:
            logger.error({
                "msg":"Num-Chunks header is required"
            })
            return Response("Num-Chunks header is required", status=503)

        responseHeaders["Start-Time"] = str(arrival_time)     
        get_merge_encrypted_matrix_start_time  = time.time()
        init_shiftmatrix = await RoryCommon.get_pyctxt(
            client         = STORAGE_CLIENT, 
            bucket_id      = BUCKET_ID, 
            key            = init_sm_id, 
            ckks           = ckks,
            backoff_factor = MICTLANX_BACKOFF_FACTOR, 
            delay          = MICTLANX_DELAY,
            force          = False,
            max_retries    = MICTLANX_MAX_RETRIES,
            timeout        = MICTLANX_TIMEOUT,
        )

        get_init_sm_entry = ExperimentLogEntry(
            event          = "GET",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_merge_encrypted_matrix_start_time,
            end_time       = time.time(),
            id             = init_sm_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k,
        )
        logger.info(get_init_sm_entry.model_dump())

        dbskmeans = DbskmeansPQC(he_object=ckks.he_object, init_shiftmatrix=init_shiftmatrix)
        get_merge_encrypted_matrix_start_time  = time.time()
        
        encryptedMatrix = await RoryCommon.get_pyctxt(
            client         = STORAGE_CLIENT, 
            bucket_id      = BUCKET_ID, 
            key            = encrypted_matrix_id, 
            ckks           = ckks,
            backoff_factor = MICTLANX_BACKOFF_FACTOR, 
            delay          = MICTLANX_DELAY,
            force          = False,
            max_retries    = MICTLANX_MAX_RETRIES,
            timeout        = MICTLANX_TIMEOUT,
        )

        get_encrypted_matrix_entry = ExperimentLogEntry(
            event          = "GET",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_merge_encrypted_matrix_start_time,
            end_time       = time.time(),
            id             = encrypted_matrix_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k,
        )
        logger.info(get_encrypted_matrix_entry.model_dump())
                
        get_merge_start_time = time.time()
        encrypted_udm = await RoryCommon.get_and_merge(
            client         = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = encrypted_udm_id,
            max_retries    = MICTLANX_MAX_RETRIES,
            delay          = MICTLANX_DELAY, 
            backoff_factor = MICTLANX_BACKOFF_FACTOR,
            timeout        = MICTLANX_TIMEOUT,
        )
        
        get_udm_entry = ExperimentLogEntry(
            event          = "GET",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_merge_start_time,
            end_time       = time.time(),
            id             = encrypted_udm_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k,
        )
        logger.info(get_udm_entry.model_dump())

        responseHeaders["Encrypted-Udm-Dtype"] = str(encrypted_udm.dtype) # Extract the type
        responseHeaders["Encrypted-Udm-Shape"] = str(encrypted_udm.shape) # Extract the shape
        if is_start_status: #if the status is start
            __Cent_j = init_shiftmatrix #There is no Cent_j
        else: 
            cent_j_start_time = time.time()
            __Cent_j = LocalUtils.get_pyctxt_with_retry(
                STORAGE_CLIENT = STORAGE_CLIENT, 
                bucket_id      = BUCKET_ID, 
                num_chunks     = num_chunks,
                key            = cent_i_id, 
                ckks           = ckks
            )
            status = Constants.ClusteringStatus.WORK_IN_PROGRESS

            get_udm_entry = ExperimentLogEntry(
                event          = "GET",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = cent_j_start_time,
                end_time       = time.time(),
                id             = cent_i_id,
                worker_id      = worker_id,
                num_chunks     = num_chunks,
                k              = k,
            )
            logger.info(get_udm_entry.model_dump())

        run1_start_time = time.time()
        run1_result = dbskmeans.run1( # The first part of the skmeans is done
            status          = status,
            k               = k,
            encryptedMatrix = encryptedMatrix, 
            UDM             = encrypted_udm,
            Cent_j          = __Cent_j,
            num_attributes  = encrypted_matrix_shape[1]
        )
        if run1_result.is_err:
            error = run1_result.unwrap_err()
            logger.error(str(error))
            return Response(str(error), status=500 )
        S1,Cent_i,Cent_j,label_vector = run1_result.unwrap()

        run1_entry = ExperimentLogEntry(
            event          = "RUN1",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = run1_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k
        ) 
        logger.info(run1_entry.model_dump())

        t1 = time.time()
        maybe_cent_i_chunks = RoryCommon.from_pyctxts_to_chunks(
            key        = cent_i_id,
            num_chunks = num_chunks,
            xs         = Cent_i
        )
        if maybe_cent_i_chunks.is_none:
            return Response(status=500, response="Failed to create chunks from cent_i")
        
        x = await RoryCommon.delete_and_put_chunks(
            client    = STORAGE_CLIENT,
            bucket_id = BUCKET_ID,
            key       = cent_i_id,
            chunks    = maybe_cent_i_chunks.unwrap(),
            timeout   = MICTLANX_TIMEOUT,
            max_tries = MICTLANX_MAX_RETRIES,
        )
        if x.is_err:
            return Response(status =500, response="Failed to put cent i")

        put_cent_i_entry = ExperimentLogEntry(
            event          = "PUT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = t1,
            end_time       = time.time(),
            id             = cent_i_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k,
        ) 
        logger.info(put_cent_i_entry.model_dump())

        t1 = time.time()
        maybe_cent_j_chunks = RoryCommon.from_pyctxts_to_chunks(
            key        = cent_j_id,
            num_chunks = num_chunks,
            xs         = Cent_j
        )

        if maybe_cent_j_chunks.is_none:
            return Response(status=500, response="Failed to create chunks from cent_j")
        y = await RoryCommon.delete_and_put_chunks(
            client    = STORAGE_CLIENT,
            bucket_id = BUCKET_ID,
            key       = cent_j_id,
            chunks    = maybe_cent_j_chunks.unwrap(),
            timeout   = MICTLANX_TIMEOUT,
            max_tries = MICTLANX_MAX_RETRIES,
        )
        if y.is_err:
            return Response(status =500, response="Failed to put cent j")

        put_cent_j_entry = ExperimentLogEntry(
            event          = "PUT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = t1,
            end_time       = time.time(),
            id             = cent_j_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k,
        ) 
        logger.info(put_cent_j_entry.model_dump())
        
        t1 = time.time()
        maybe_s1_chunks = RoryCommon.from_pyctxts_to_chunks(
            key        = encrypted_shift_matrix_id, 
            num_chunks = num_chunks,
            xs         = S1
        )
        if maybe_s1_chunks.is_none:
            return Response(status=500, response="Failed to create chunks from encrypted shiftmatrix")
        z = await RoryCommon.delete_and_put_chunks(
            client    = STORAGE_CLIENT,
            bucket_id = BUCKET_ID,
            key       = encrypted_shift_matrix_id,
            chunks    = maybe_s1_chunks.unwrap(),
            timeout   = MICTLANX_TIMEOUT,
            max_tries = MICTLANX_MAX_RETRIES,
        )
        if z.is_err:
            return Response(status =500, response="Failed to put encrypted shift matrix")

        put_encrypted_sm_entry = ExperimentLogEntry(
            event          = "PUT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = t1,
            end_time       = time.time(),
            id             = encrypted_shift_matrix_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k,
        ) 
        logger.info(put_encrypted_sm_entry.model_dump())

        end_time     = time.time()
        service_time = end_time - arrival_time
        n_iterations = int(requestHeaders.get("Iterations",0)) + 1
        
        clustering_entry = ExperimentLogEntry(
            event          = "COMPLETED",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = arrival_time,
            end_time       = time.time(),
            id             = encrypted_matrix_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k,
            iterations     = n_iterations
        )
        logger.info(clustering_entry.model_dump())

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
            "at":"worker_dbskmeans_1"
        })
        return Response(str(e),status = 500)


async def pqc_dbskmeans_2(requestHeaders):
    local_start_time           = time.time()
    logger                     = current_app.config["logger"]
    worker_id                  = current_app.config["NODE_ID"]
    BUCKET_ID:str              = current_app.config.get("BUCKET_ID","rory")
    STORAGE_CLIENT:AsyncClient = current_app.config["ASYNC_STORAGE_CLIENT"]
    algorithm                  = Constants.ClusteringAlgorithms.SKMEANS_PQC
    status                     = int(requestHeaders.get("Clustering-Status",Constants.ClusteringStatus.START))
    plaintext_matrix_id        = requestHeaders["Plaintext-Matrix-Id"]
    encrypted_matrix_id        = requestHeaders["Encrypted-Matrix-Id"]
    shift_matrix_id            = requestHeaders.get("Shift-Matrix-Id","{}shiftmatrix".format(plaintext_matrix_id))
    k                          = int(requestHeaders.get("K",3))
    isZero                     = bool(int(requestHeaders.get("Is-Zero")))
    iterations                 = int(requestHeaders.get("Iterations",0))
    experiment_id              = requestHeaders.get("Experiment-Id","")
    _round                     = bool(int(current_app.config.get("_round","0"))) #False
    decimals                   = int(current_app.config.get("DECIMALS","2"))

    path               = current_app.config.get("KEYS_PATH","/rory/keys")
    ctx_filename       = current_app.config.get("CTX_FILENAME","ctx")
    pubkey_filename    = current_app.config.get("PUBKEY_FILENAME","pubkey")
    secretkey_filename = current_app.config.get("SECRET_KEY_FILENAME","secretkey")
    relinkey_filename  = current_app.config.get("RELINKEY_FILENAME","relinkey")
    
    shift_matrix_ope_id     = requestHeaders.get("Shift-Matrix-Ope-Id","{}-shift-matrix-ope".format(plaintext_matrix_id))
    _encrypted_matrix_shape = requestHeaders.get("Encrypted-Matrix-Shape",-1)
    _encrypted_matrix_dtype = requestHeaders.get("Encrypted-Matrix-Dtype",-1)
    _encrypted_udm_shape    = requestHeaders.get("Encrypted-Udm-Shape",-1)
    _encrypted_udm_dtype    = requestHeaders.get("Encrypted-Udm-Dtype",-1)
    
    MICTLANX_TIMEOUT        = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
    MICTLANX_DELAY          = int(current_app.config.get("MICTLANX_DELAY","2"))
    MICTLANX_BACKOFF_FACTOR = float(current_app.config.get("MICTLANX_BACKOFF_FACTOR","0.5"))
    MICTLANX_MAX_RETRIES    = int(current_app.config.get("MICTLANX_MAX_RETRIES","10"))

    if encrypted_matrix_id == -1 or plaintext_matrix_id == -1:
        return Response("Either Encrypted-Matrix-Id or Plain-Matrix-Id is missing",status=500)
    num_chunks       = int(requestHeaders.get("Num-Chunks",-1))
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
    encrypted_udm_id             = "{}encryptedudm".format(plaintext_matrix_id)
    init_sm_id       = "{}initsm".format(plaintext_matrix_id)
    cent_i_id        = "{}centi".format(plaintext_matrix_id) #Build the id of Cent_i
    cent_j_id        = "{}centj".format(plaintext_matrix_id) #Build the id of Cent_j
    response_headers = {}

    ckks = Ckks.from_pyfhel(
        _round   = _round,
        decimals = decimals,
        path               = path,
        ctx_filename       = ctx_filename,
        pubkey_filename    = pubkey_filename,
        secretkey_filename = secretkey_filename,
        relinkey_filename  = relinkey_filename
    )

    try:
        get_merge_start_time = time.time()
        prev_encrypted_udm = await RoryCommon.get_and_merge(
            client         = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = encrypted_udm_id,
            timeout        = MICTLANX_TIMEOUT,
            max_retries    = MICTLANX_MAX_RETRIES,
            delay          = MICTLANX_DELAY,
            backoff_factor = MICTLANX_BACKOFF_FACTOR,
            force          = True
        )

        get_udm_entry = ExperimentLogEntry(
            event          = "GET",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_merge_start_time,
            end_time       = time.time(),
            id             = encrypted_udm_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k,
        )
        logger.info(get_udm_entry.model_dump())
        

        if(isZero): #If Shift matrix is zero
            response_headers["Clustering-Status"]  = Constants.ClusteringStatus.COMPLETED #Change the status to COMPLETED
            end_time                               = time.time()
            service_time                           = end_time - local_start_time #The service time is calculated
            response_headers["Total-Service-Time"] = str(service_time) #Save the service time

            clustering_entry = ExperimentLogEntry(
                event          = "COMPLETED",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = local_start_time,
                end_time       = time.time(),
                id             = encrypted_matrix_id,
                worker_id      = worker_id,
                num_chunks     = num_chunks,
                k              = k,
                iterations     = iterations
            )
            logger.info(clustering_entry.model_dump())

            return Response( #Return none and headers
                response = None, 
                status   = 204, 
                headers  = response_headers
            )
        
        else:
            t1 = time.time()
            init_shiftmatrix = await RoryCommon.get_pyctxt(
                client         = STORAGE_CLIENT, 
                bucket_id      = BUCKET_ID, 
                key            = init_sm_id, 
                ckks           = ckks,
                backoff_factor = MICTLANX_BACKOFF_FACTOR,
                delay          = MICTLANX_DELAY,
                max_retries    = MICTLANX_MAX_RETRIES,
                timeout        = MICTLANX_TIMEOUT,
            )

            get_init_sm_entry = ExperimentLogEntry(
                event          = "GET",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = t1,
                end_time       = time.time(),
                id             = init_sm_id,
                worker_id      = worker_id,
                num_chunks     = num_chunks,
                k              = k,
            )
            logger.info(get_init_sm_entry.model_dump())

            dbskmeans = DbskmeansPQC(he_object=ckks.he_object, init_shiftmatrix=init_shiftmatrix)
            status    = Constants.ClusteringStatus.WORK_IN_PROGRESS

            response_headers["Clustering-Status"] = status #The status is changed to WORK IN PROGRESS
            get_matrix_start_time = time.time()
            shift_matrix_ope_response = await RoryCommon.get_and_merge(
                client         = STORAGE_CLIENT,
                bucket_id      = BUCKET_ID,
                key            = shift_matrix_ope_id,
                max_retries    = MICTLANX_MAX_RETRIES,
                backoff_factor = MICTLANX_BACKOFF_FACTOR,
                delay          = MICTLANX_DELAY,
                force          = True,
                timeout        = MICTLANX_TIMEOUT,
            )
            shift_matrix_ope:npt.NDArray = shift_matrix_ope_response.value
            response_headers["Clustering-Status"] = status

            get_init_sm_entry = ExperimentLogEntry(
                event          = "GET",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = get_matrix_start_time,
                end_time       = time.time(),
                id             = shift_matrix_ope_id,
                worker_id      = worker_id,
                num_chunks     = num_chunks,
                k              = k,
            )
            logger.info(get_init_sm_entry.model_dump())

            run2_start_time = time.time()
            current_udm = dbskmeans.run_2( # The second part of the skmeans starts
                k           = k,
                UDM         = prev_encrypted_udm,
                attributes  = int(encrypted_matrix_shape[1]),
                shiftMatrix = shift_matrix_ope,
            )

            run2_entry = ExperimentLogEntry(
                event          = "RUN2",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = run2_start_time,
                end_time       = time.time(),
                id             = plaintext_matrix_id,
                worker_id      = worker_id,
                num_chunks     = num_chunks,
                k              = k
            ) 
            logger.info(run2_entry.model_dump())
            put_udm_start_time = time.time()
            maybe_udm_chunks:Option[Chunks] = Chunks.from_ndarray(
                ndarray      = current_udm,
                group_id     = encrypted_udm_id,
                num_chunks   = num_chunks,
                chunk_prefix = Some(encrypted_udm_id)
            )
            if maybe_udm_chunks.is_none:
                logger.error({"msg":"Something went wrong segment encrypted udm."})
                return Response(
                    status   = 500,
                    response = "Something went wrong segment udm."
                )
            udm_chunks = maybe_udm_chunks.unwrap()
            cm_shape   = str(current_udm.shape)
            cm_dtype   = str(current_udm.dtype)
            del current_udm

            put_chunks_udm_generator_results = await RoryCommon.delete_and_put_chunks(
                client    = STORAGE_CLIENT,
                bucket_id = BUCKET_ID,
                key       = encrypted_udm_id, 
                chunks    = udm_chunks, 
                timeout   = MICTLANX_TIMEOUT,
                max_tries = MICTLANX_MAX_RETRIES,
                tags      = {
                    "full_shape":cm_shape,
                    "full_dtype":cm_dtype,
                }
            )
            del udm_chunks

            put_udm_entry = ExperimentLogEntry(
                event          = "PUT",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = put_udm_start_time,
                end_time       = time.time(),
                id             = encrypted_udm_id,
                worker_id      = worker_id,
                num_chunks     = num_chunks,
                k              = k,
            ) 
            logger.info(put_udm_entry.model_dump())

            end_time                         = time.time()
            service_time                     = end_time - local_start_time  #Service time is calculated
            response_headers["End-Time"]     = str(end_time)
            response_headers["Service-Time"] = str(service_time)
            
            clutering_uncompleted_entry = ExperimentLogEntry(
                event          = "UNCOMPLETED",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = local_start_time,
                end_time       = time.time(),
                id             = plaintext_matrix_id,
                worker_id      = worker_id,
                num_chunks     = num_chunks,
                k              = k,
                iterations     = iterations
            )
            logger.info(clutering_uncompleted_entry.model_dump())

            del prev_encrypted_udm
            return Response( #Return none and headers
                response = None,
                status   = 204, 
                headers  = response_headers
            )
    except Exception as e:
        logger.error("DBSKMEANS_2_ERROR: "+encrypted_matrix_id+" "+str(e))
        return Response(str(e),status = 503)


@clustering.route("/pqc/dbskmeans",methods = ["POST"])
async def pqc_dbskmeans():
    headers         = request.headers
    head            = ["User-Agent","Accept-Encoding","Connection"]
    filteredHeaders = dict(list(filter(lambda x: not x[0] in head, headers.items())))
    step_index      = int(filteredHeaders.get("Step-Index",1))
    response        = Response()
    logger          = current_app.config["logger"]
    logger.info({
        "X":1,
        "step_index":step_index
    })
    if step_index == 1:
        return await pqc_dbskmeans_1(filteredHeaders)
    elif step_index == 2:
        return await pqc_dbskmeans_2(filteredHeaders)
    else:
        return response