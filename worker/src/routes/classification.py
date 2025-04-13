import time, json, os
from flask import Blueprint,current_app,request,Response
from rory.core.utils.constants import Constants
from rory.core.classification.secure.distributed.sknn import SecureKNearestNeighbors as SKNN
from rory.core.classification.secure.pqc.sknn import SecureKNearestNeighbors as SKNNPQC
from rory.core.security.cryptosystem.pqc.ckks import Ckks
from rory.core.classification.knn import KNearestNeighbors as KNN
from mictlanx.v4.client import Client as V4Client
from rory.core.interfaces.logger_metrics import LoggerMetrics
from mictlanx.utils.index import Utils as MictlanXUtils
from utils.utils import Utils
from mictlanx.v4.interfaces.responses import GetNDArrayResponse
from mictlanx.utils.segmentation import Chunks
import numpy.typing as npt
from option import Result, Some
from rorycommon import Common as RoryCommon


import numpy as np
classification = Blueprint("classification",__name__,url_prefix = "/classification")

@classification.route("/test",methods=["GET","POST"])
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


async def sknn_pedict_1(requestHeaders):
    local_start_time         = time.time() #Worker start time
    headers                  = request.headers
    logger                   = current_app.config["logger"]
    worker_id                = current_app.config["NODE_ID"] # Get the node_id from the global configuration
    STORAGE_CLIENT           = current_app.config["ASYNC_STORAGE_CLIENT"]
    BUCKET_ID:str            = current_app.config.get("BUCKET_ID","rory")
    model_id                 = requestHeaders.get("Model-Id","model0") #iris
    encrypted_model_id       = "encrypted{}".format(model_id) #encrypted-iris_model
    model_labels_id          = "{}labels".format(model_id) #iris_model_labels
    records_test_id          = requestHeaders.get("Records-Test-Id","matrix0")
    encrypted_records_id     = "encrypted{}".format(records_test_id) # The id of the encrypted matrix is built
    algorithm                = Constants.ClassificationAlgorithms.SKNN_PREDICT
    _encrypted_model_shape   = requestHeaders.get("Encrypted-Model-Shape",-1)
    _encrypted_model_dtype   = requestHeaders.get("Encrypted-Model-Dtype",-1)
    _encrypted_records_shape = requestHeaders.get("Encrypted-Records-Shape",-1)
    _encrypted_records_dtype = requestHeaders.get("Encrypted-Records-Dtype",-1)
    distance:str             = current_app.config.get("DISTANCE","MANHATHAN")
    MICTLANX_TIMEOUT          = int(current_app.config.get("MICTLANX_TIMEOUT",120))
    backoff_factor = 1.5
    delay          = 1
    max_retries    = 10 
    
    if _encrypted_model_dtype == -1:
        return Response("Encrypted-Model-Dtype", status=500)
    if _encrypted_model_shape == -1 :
        return Response("Encrypted-Model-Shape header is required", status=500)
    
    if _encrypted_records_dtype == -1:
        return Response("Encrypted-Records-Dtype", status=500)
    if _encrypted_records_shape == -1 :
        return Response("Encrypted-Records-Shape header is required", status=500)
    
    encrypted_model_shape:tuple   = eval(_encrypted_model_shape)
    encrypted_records_shape:tuple = eval(_encrypted_records_shape)
    num_chunks                    = int(requestHeaders.get("Num-Chunks",-1))
    response_headers              = {}

    if num_chunks == -1:
        return Response("Num-Chunks header is required", status=503)
    try:
        response_headers["Start-Time"] = str(local_start_time)
   
        get_merge_encrypted_model_start_time = time.time()

        encrypted_model = await RoryCommon.get_and_merge(
            client= STORAGE_CLIENT,
            bucket_id = BUCKET_ID,
            key       = encrypted_model_id,
            max_retries = 20,
            delay = 2
        )
        # res = Utils.get_and_merge_ndarray()

        get_merge_encrypted_model_st = time.time()- get_merge_encrypted_model_start_time
        logger.info({
            "event":"GET.MERGE.NDARRAY",
            "bucket_id":BUCKET_ID,
            "key":encrypted_model_id,
            "algorithm":algorithm,
            "num_chunks":num_chunks,
            "shape":_encrypted_model_shape,
            "dtype":_encrypted_model_dtype,
            "service_time":get_merge_encrypted_model_st
        })

        response_headers["Encrypted-Model-Dtype"] = encrypted_model.dtype #["tags"]["dtype"] #Save the data type
        response_headers["Encrypted-Model-Shape"] = encrypted_model.shape #Save the shape
        
        get_merge_encrypted_records_start_time = time.time()
        
        encrypted_records = await RoryCommon.get_and_merge(
            client         = STORAGE_CLIENT,
            key            = encrypted_records_id,
            bucket_id      = BUCKET_ID,
            max_retries    = max_retries,
            delay          = delay,
            backoff_factor = backoff_factor,
            timeout        = MICTLANX_TIMEOUT,
        )
        

        response_headers["Encrypted-Records-Test-Dtype"] = encrypted_records.dtype #["tags"]["dtype"] #Save the data type
        response_headers["Encrypted-Records-Test-Shape"] = encrypted_records.shape #Save the shape
        get_merge_encrypted_records_st = time.time()- get_merge_encrypted_records_start_time

        logger.info({
            "event":"GET.MERGE.NDARRAY",
            "bucket_id":BUCKET_ID,
            "key":encrypted_records_id,
            "algorithm":algorithm,
            "num_chunks":num_chunks,
            "shape":_encrypted_model_shape,
            "dtype":_encrypted_model_dtype,
            "service_time":get_merge_encrypted_records_st
        })


        all_distances = SKNN.calculate_distances(
			dataset  = encrypted_records,
			model    = encrypted_model,
            distance = distance,
		)

        logger.info({
            "event":"CALCULATE.DISTANCES",
            "model_id":model_id,
            "algorithm":algorithm,
            "encrypted_records_id":encrypted_records_id,
            "encrypted_records_shape":str(encrypted_records.shape),
            "encrypted_records_dtype":str(encrypted_records.dtype),
            "encrypted_model_id":encrypted_model_id,
            "encrypted_model_shape":str(encrypted_model.shape),
            "encrypted_model_dtype":str(encrypted_model.dtype),
            "distance":distance
        })

        distances_id = "distances{}".format(records_test_id) 
        distances_shape = all_distances.shape
        distances_dtype = all_distances.dtype

  
                
        maybe_all_distances_chunks = Chunks.from_ndarray(
            ndarray = all_distances,
            group_id = distances_id,
            chunk_prefix = Some(distances_id),
            num_chunks = num_chunks
        )

        if maybe_all_distances_chunks.is_none:
            raise "something went wrong creating the chunks"
        print("SHAPE", all_distances.shape)

        put_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
            client         = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = distances_id,
            chunks         = maybe_all_distances_chunks.unwrap(),
            tags = {
                "full_shape":str(distances_shape),
                "full_dtype":"float64"
            }
        )
        

        end_time     = time.time()
        service_time = end_time - local_start_time
        
        logger.info({
            "event":"SKNN.PREDICT.1.COMPLETED",
            "model_id":model_id,
            "algorithm":algorithm,
            "service_time":service_time
        })
  
        return Response( #Returns the final response as a label vector + the headers
            response = json.dumps({
                "distances_id":distances_id,
                "distances_shape":str(distances_shape),
                "distances_dtype":str(distances_dtype),
                "service_time":service_time
            }),
            status   = 200,
        )
    except Exception as e:
        logger.error({
            "msg":str(e)
        })
        return Response(
            response = None,
            status   = 503,
            headers  = {"Error-Message":str(e)}
        )


async def sknn_predict_2(requestHeaders):
    local_start_time        = time.time() #Worker start time
    headers                 = request.headers
    logger                  = current_app.config["logger"]
    worker_id               = current_app.config["NODE_ID"] # Get the node_id from the global configuration
    STORAGE_CLIENT:V4Client = current_app.config["ASYNC_STORAGE_CLIENT"]
    BUCKET_ID:str           = current_app.config.get("BUCKET_ID","rory")
    model_id                = requestHeaders.get("Model-Id","model0") #iris
    model_labels_id         = "{}labels".format(model_id) #iris_model_labels
    records_test_id         = requestHeaders.get("Records-Test-Id","matrix0")
    _model_labels_shape     = requestHeaders.get("Model-Labels-Shape",-1)
    if _model_labels_shape == -1:
        return Response("Model-Labels-Shape header is required", status=500)
    model_labels_shape = eval(_model_labels_shape)
    min_distances_index_id  = "distancesindex{}".format(records_test_id)
    algorithm               = Constants.ClassificationAlgorithms.SKNN_PREDICT
    MICTLANX_TIMEOUT          = int(current_app.config.get("MICTLANX_TIMEOUT",120))
    backoff_factor = 1.5
    delay          = 1
    max_retries    = 10 
    try:

        model_labels_get_start_time = time.time()
        model_labels = await RoryCommon.get_and_merge(
            client         = STORAGE_CLIENT,
            key            = model_labels_id,
            bucket_id      = BUCKET_ID,
            backoff_factor = backoff_factor,
            delay          = delay,
            max_retries    = max_retries,
            timeout        = MICTLANX_TIMEOUT

        )

        model_labels_get_st = time.time() - model_labels_get_start_time

        logger.info({
            "event":"GET.NDARRAY",
            "bucket_id":BUCKET_ID,
            "model_id":model_id,
            "algorithm":algorithm,
            "key":model_labels_id,
            "shape":str(model_labels.shape), 
            "dtype":str(model_labels.dtype),
            "service_time":model_labels_get_st
        })


        min_distances_index = await RoryCommon.get_and_merge(
            client         = STORAGE_CLIENT,
            key            = min_distances_index_id,
            bucket_id      = BUCKET_ID,
            backoff_factor = backoff_factor,
            delay          = delay,
            max_retries    = max_retries,
            timeout        = MICTLANX_TIMEOUT
        )

        logger.info({
            "event":"GET.MATRIX",
            "model_id":model_id,
            "algorithm":algorithm,
            "models_labels_id":model_labels_id,
            "min_distances_index_id":min_distances_index_id
        })
        label_vector = SKNN.get_label_vector(
            model_labels = model_labels.reshape((model_labels_shape[1],)),
            min_indexes = min_distances_index
        )
        label_vector = label_vector.reshape((label_vector.shape[0],))
        end_time                       = time.time()
        service_time                   = end_time - local_start_time
        requestHeaders["Service-Time"] = str(service_time)

        logger.info({
            "event":"SKNN.PREDICT.2.COMPLETED",
            "model_id":model_id,
            "algorithm":algorithm,
            "service_time":service_time
        })

        return Response( #Returns the final response as a label vector + the headers
            response = json.dumps({
                "label_vector":label_vector.tolist(),
                "service_time":service_time
            }),
            status   = 200,
        )
    except Exception as e:
        logger.error({
            "msg":str(e)
        })
        return Response(
            response = None,
            status   = 503,
            headers  = {"Error-Message":str(e)}
        )


@classification.route("/sknn/predict",methods = ["POST"])
async def sknn_predict():
    headers         = request.headers
    head            = ["User-Agent","Accept-Encoding","Connection"]
    filteredHeaders = dict(list(filter(lambda x: not x[0] in head, headers.items())))
    step_index      = int(filteredHeaders.get("Step-Index",1))
    response        = Response()
    if step_index == 1:
        return await sknn_pedict_1(filteredHeaders)
    elif step_index == 2:
        return await sknn_predict_2(filteredHeaders)
    else:
        return response

@classification.route("/knn/predict",methods = ["POST"])
async def knn_predict():
    local_start_time        = time.time() #Worker start time
    headers                 = request.headers
    headers                 = request.headers
    to_remove_headers       = ["User-Agent","Accept-Encoding","Connection"]
    filtered_headers        = dict(list(filter(lambda x: not x[0] in to_remove_headers, headers.items())))
    logger                  = current_app.config["logger"]
    worker_id               = current_app.config["NODE_ID"] # Get the node_id from the global configuration
    BUCKET_ID:str           = current_app.config.get("BUCKET_ID","rory")
    STORAGE_CLIENT:V4Client = current_app.config["ASYNC_STORAGE_CLIENT"]
    model_id                = filtered_headers.get("Model-Id","model0") #iris
    model_labels_id         = "{}labels".format(model_id) #iris_model_labels
    records_test_id         = filtered_headers.get("Records-Test-Id","matrix0")
    algorithm               = Constants.ClassificationAlgorithms.KNN_PREDICT
    response_headers        = {}
    distance                = current_app.config["DISTANCE"]
    MICTLANX_TIMEOUT          = int(current_app.config.get("MICTLANX_TIMEOUT",120))
    backoff_factor = 1.5
    delay          = 1
    max_retries    = 10     
    _model_labels_shape     = filtered_headers.get("Model-Labels-Shape",-1)
    if _model_labels_shape == -1:
        error ="Model-Labels-Shape header is required"
        logger.error(error)
        return Response(error, status=500)
    model_labels_shape = eval(_model_labels_shape)
    try:
        response_headers["Start-Time"] = str(local_start_time)
     
        get_model_start_time = time.time()
        model = await RoryCommon.get_and_merge(
            client         = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = model_id,
            max_retries    = max_retries,
            delay          = delay,
            backoff_factor = backoff_factor,
            timeout        = MICTLANX_TIMEOUT
        )
        
        get_model_st = time.time() - get_model_start_time
        logger.info({
            "event":"GET",
            "bucket_id":BUCKET_ID,
            "key":model_id,
            "algorithm":algorithm,
            "service_time":get_model_st
        })

        get_model_labels_start_time = time.time()
        model_labels = await RoryCommon.get_and_merge(
            client         = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = model_labels_id,
            max_retries    = max_retries,
            delay          = delay,
            backoff_factor = backoff_factor,
            timeout        = MICTLANX_TIMEOUT,
        )
        get_model_labels_st = time.time() - get_model_labels_start_time
        logger.info({
            "event":"GET",
            "bucket_id":BUCKET_ID,
            "key":model_labels_id,
            "algorithm":algorithm,
            "service_time":get_model_labels_st
        })

     
        get_records_start_time = time.time()
        records = await RoryCommon.get_and_merge(
            client         = STORAGE_CLIENT,
            key            = records_test_id,
            bucket_id      = BUCKET_ID,
            max_retries    = max_retries,
            delay          = delay,
            backoff_factor = backoff_factor,
            timeout        = MICTLANX_TIMEOUT
        )
        get_records_st = time.time() - get_records_start_time
        logger.info({
            "event":"GET",
            "bucket_id":BUCKET_ID,
            "key":records_test_id,
            "algorithm":algorithm,
            "service_time":get_records_st
        })

        knn_predict_start_time = time.time()
        label_vector:npt.NDArray = KNN.predict(
            dataset      = records,
            model        = model,
            model_labels = model_labels.reshape((model_labels_shape[1],)),
            distance     = distance
        )
        knn_predict_st = time.time() - knn_predict_start_time
        logger.info({
            "event":"KNN.PREDICT",
            "model_id":model_id,
            "algorithm":algorithm,
            "records_shape":str(records.shape),
            "records_dtype":str(records.dtype),
            "models_labels_shape":str(model_labels.shape),
            "models_labels_shape":str(model_labels.dtype),
            "service_time":knn_predict_st
        })
        
        end_time                         = time.time()
        service_time                     = end_time - local_start_time
        response_headers["Service-Time"] = str(service_time)
        
        logger.info({
            "event":"KNN.PREDICT.COMPLETED",
            "model_id":model_id,
            "algorithm":algorithm,
            "service_time":time.time() - local_start_time
        })
        return Response( #Returns the final response as a label vector + the headers
            response = json.dumps({
                "label_vector":label_vector.flatten().tolist(),
                "service_time":service_time
            }),
            status   = 200,
            headers  = {**response_headers}
        )
    except Exception as e:
        logger.error({
            "msg":str(e)
        })
        return Response(
            response = None,
            status   = 503,
            headers  = {"Error-Message":str(e)}
        )
    
async def sknn_pqc_pedict_1(requestHeaders):
    local_start_time         = time.time() #Worker start time
    headers                  = request.headers
    logger                   = current_app.config["logger"]
    worker_id                = current_app.config["NODE_ID"] # Get the node_id from the global configuration
    STORAGE_CLIENT:V4Client  = current_app.config["STORAGE_CLIENT"]
    BUCKET_ID:str            = current_app.config.get("BUCKET_ID","rory")
    model_id                 = requestHeaders.get("Model-Id","model0") #iris
    encrypted_model_id       = "encrypted{}".format(model_id) #encrypted-iris_model
    model_labels_id          = "{}labels".format(model_id) #iris_model_labels
    records_test_id          = requestHeaders.get("Records-Test-Id","matrix0")
    encrypted_records_id     = "encrypted{}".format(records_test_id) # The id of the encrypted matrix is built
    algorithm                = Constants.ClassificationAlgorithms.SKNN_PQC_PREDICT
    _encrypted_model_shape   = requestHeaders.get("Encrypted-Model-Shape",-1)
    _encrypted_model_dtype   = requestHeaders.get("Encrypted-Model-Dtype",-1)
    _encrypted_records_shape = requestHeaders.get("Encrypted-Records-Shape",-1)
    _encrypted_records_dtype = requestHeaders.get("Encrypted-Records-Dtype",-1)
    _round   = False
    decimals = 2
    
    if _encrypted_model_dtype == -1:
        return Response("Encrypted-Model-Dtype", status=500)
    if _encrypted_model_shape == -1 :
        return Response("Encrypted-Model-Shape header is required", status=500)
    
    if _encrypted_records_dtype == -1:
        return Response("Encrypted-Records-Dtype", status=500)
    if _encrypted_records_shape == -1 :
        return Response("Encrypted-Records-Shape header is required", status=500)
    
    encrypted_model_shape:tuple   = eval(_encrypted_model_shape)
    encrypted_records_shape:tuple = eval(_encrypted_records_shape)
    num_chunks                    = int(requestHeaders.get("Num-Chunks",-1))
    response_headers              = {}

    path               = os.environ.get("KEYS_PATH","/rory/keys")
    ctx_filename       = os.environ.get("CTX_FILENAME","ctx")
    pubkey_filename    = os.environ.get("PUBKEY_FILENAME","pubkey")
    secretkey_filename = os.environ.get("SECRET_KEY_FILENAME","secretkey")

    # _______________________________________________________________________________
    ckks = Ckks.from_pyfhel(
        _round   = _round,
        decimals = decimals,
        path               = path,
        ctx_filename       = ctx_filename,
        pubkey_filename    = pubkey_filename,
        secretkey_filename = secretkey_filename
    )
    # _______________________________________________________________________________

    logger.debug({
        "event":"SKNN.PQC.PREDICT.1.STARTED",
        "worker_id":worker_id,
        "model_id":model_id,
        "encrypted_model_id":encrypted_model_id,
        "models_labels_id":model_labels_id,
        "algorithm":algorithm,
        "encrypted_records_id":encrypted_records_id,
        "encrypted_model_shape":_encrypted_model_shape,
        "encrypted_model_dtype":_encrypted_model_dtype,
        "encrypted_records_shape":_encrypted_records_shape,
        "encrypted_records_dtype":_encrypted_records_dtype,
        "num_chunks":num_chunks
    })
    if num_chunks == -1:
        return Response("Num-Chunks header is required", status=503)
    try:
        response_headers["Start-Time"] = str(local_start_time)
        logger.debug({
            "event":"GET.PYCTXT.MODEL.BEFORE",
            "bucket_id":BUCKET_ID,
            "encrypted_model_id":encrypted_model_id,
            "shape":_encrypted_model_shape,
            "dtype":_encrypted_model_dtype,
            "num_chunks":num_chunks,
            "model_id":model_id,
            "algorithm":algorithm,
        })
        get_merge_encrypted_model_start_time = time.time()

        encrypted_model = Utils.get_pyctxt_matrix_with_retry(
            STORAGE_CLIENT = STORAGE_CLIENT, 
            bucket_id      = BUCKET_ID, 
            num_chunks     = num_chunks,
            key            = encrypted_model_id, 
            ckks           = ckks,
            chunk_size     = "10MB"
        )
        get_merge_encrypted_model_st = time.time()- get_merge_encrypted_model_start_time
        logger.info({
            "event":"GET.PYCTXT.MODEL",
            "bucket_id":BUCKET_ID,
            "model_id":model_id,
            "algorithm":algorithm,
            "encrypted_model_id":encrypted_model_id,
            "num_chunks":num_chunks,
            "shape":_encrypted_model_shape,
            "dtype":_encrypted_model_dtype,
            "service_time":get_merge_encrypted_model_st
        })
        
        logger.debug({
            "event":"GET.PYCTXT.DATA.BEFORE",
            "bucket_id":BUCKET_ID,
            "encrypted_records_id":encrypted_records_id,
            "shape":_encrypted_records_shape,
            "dtype":_encrypted_records_dtype,
            "num_chunks":num_chunks,
            "model_id":model_id,
            "algorithm":algorithm,
        })

        encrypted_records = Utils.get_pyctxt_matrix_with_retry(
            STORAGE_CLIENT = STORAGE_CLIENT, 
            bucket_id      = BUCKET_ID, 
            num_chunks     = num_chunks,
            key            = encrypted_records_id, 
            ckks           = ckks
        )
        logger.info({
            "event":"GET.PYCTXT.DATA",
            "bucket_id":BUCKET_ID,
            "model_id":model_id,
            "algorithm":algorithm,
            "encrypted_records_id":encrypted_records_id,
            "shape":_encrypted_records_shape,
            "dtype":_encrypted_records_dtype,
            "num_chunks":num_chunks,
            "model_id":model_id,
            "algorithm":algorithm,
        })

        logger.debug({
            "event":"CALCULATE.DISTANCES.BEFORE",
            "model_id":model_id,
            "algorithm":algorithm,
            "encrypted_records_id":encrypted_records_id,
            "encrypted_model_id":encrypted_model_id,
            "encrypted_model_shape":str(encrypted_model_shape),
            "encrypted_model_dtype":str(encrypted_records_shape)
        })
        all_distances = SKNNPQC.calculate_distances(
			dataset = encrypted_records,
			model   = encrypted_model,
            model_shape   = encrypted_model_shape,
            dataset_shape = encrypted_records_shape
		)
        
        logger.info({
            "event":"CALCULATE.DISTANCES",
            "model_id":model_id,
            "algorithm":algorithm,
            "encrypted_records_id":encrypted_records_id,
            "encrypted_records_shape":str(encrypted_records_shape),
            "encrypted_model_id":encrypted_model_id,
            "encrypted_model_shape":str(encrypted_model_shape),
        })
        
        distances_id = "distances{}".format(records_test_id) 
        distances_shape = all_distances.shape
        distances_dtype = all_distances.dtype

        logger.debug({
            "event":"PUT.CHUNKED",
            "model_id":model_id,
            "algorithm":algorithm,
            "encrypted_records_id":encrypted_records_id,
            "encrypted_model_id":encrypted_model_id,
            "distances_shape":str(distances_shape),
            "distances_dtype":str(distances_dtype)
        })    

        distances_chunks = MictlanXUtils.to_gen_bytes(data=Utils.pyctxt_matrix_to_bytes(ciphertext=all_distances))

        z = Utils.delete_and_put_chunked(
            STORAGE_CLIENT = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            ball_id        = distances_id,
            key            = distances_id,
            chunks         = distances_chunks,
        )

        logger.debug({
            "event":"PUT.CHUNKED",
            "model_id":model_id,
            "algorithm":algorithm,
            "encrypted_records_id":encrypted_records_id,
            "encrypted_model_id":encrypted_model_id,
            "distances_shape":str(distances_shape),
            "distances_dtype":str(distances_dtype)
        })

        end_time     = time.time()
        service_time = end_time - local_start_time

        logger.info({
            "event":"SKNN.PREDICT.1.COMPLETED",
            "model_id":model_id,
            "algorithm":algorithm,
            "service_time":service_time
        })
        return Response( #Returns the final response as a label vector + the headers
            response = json.dumps({
                "distances_id":distances_id,
                "distances_shape":str(distances_shape),
                "distances_dtype":str(distances_dtype),
                "service_time":service_time
            }),
            status   = 200,
        )

    except Exception as e:
        logger.error({
            "msg":str(e)
        })
        return Response(
            response = None,
            status   = 503,
            headers  = {"Error-Message":str(e)}
        )

async def sknn_pqc_predict_2(requestHeaders):
    local_start_time        = time.time() #Worker start time
    headers                 = request.headers
    logger                  = current_app.config["logger"]
    worker_id               = current_app.config["NODE_ID"] # Get the node_id from the global configuration
    STORAGE_CLIENT:V4Client = current_app.config["STORAGE_CLIENT"]
    BUCKET_ID:str           = current_app.config.get("BUCKET_ID","rory")
    model_id                = requestHeaders.get("Model-Id","model0") #iris
    model_labels_id         = "{}labels".format(model_id) #iris_model_labels
    records_test_id         = requestHeaders.get("Records-Test-Id","matrix0")
    min_distances_index_id  = "distancesindex{}".format(records_test_id)
    algorithm               = Constants.ClassificationAlgorithms.SKNN_PQC_PREDICT

    try:
        logger.debug({
            "event":"SKNN.PQC.PREDICT.2.STARTED",
            "worker_id":worker_id,
            "model_id":model_id,
            "models_labels_id":model_labels_id,
            "records_test_id":records_test_id,
            "min_distances_index_id":min_distances_index_id,
            "algorithm":algorithm,
        })

        logger.debug({
            "event":"GET.NDARRAY.BEFORE",
            "model_id":model_id,
            "algorithm":algorithm,
            "key":model_labels_id,
            "bucket_id":BUCKET_ID,
        })
        model_labels_get_start_time = time.time()
        model_labels = Utils.get_matrix_or_error(
            client    = STORAGE_CLIENT,
            key       = model_labels_id,
            bucket_id = BUCKET_ID
        ).value

        model_labels_get_st = time.time() - model_labels_get_start_time

        logger.info({
            "event":"GET.NDARRAY",
            "bucket_id":BUCKET_ID,
            "model_id":model_id,
            "algorithm":algorithm,
            "key":model_labels_id,
            "shape":str(model_labels.shape), 
            "dtype":str(model_labels.dtype),
            "service_time":model_labels_get_st
        })

        logger.debug({
            "event":"GET.MATRIX.BEFORE",
            "bucket_id":BUCKET_ID,
            "model_id":model_id,
            "algorithm":algorithm,
            "models_labels_id":model_labels_id,
            "min_distances_index_id":min_distances_index_id
        })

        min_distances_index = Utils.get_matrix_or_error(
            client    = STORAGE_CLIENT,
            key       = min_distances_index_id,
            bucket_id = BUCKET_ID
        ).value

        logger.info({
            "event":"GET.MATRIX",
            "model_id":model_id,
            "algorithm":algorithm,
            "models_labels_id":model_labels_id,
            "min_distances_index_id":min_distances_index_id
        })

        label_vector = SKNNPQC.get_label_vector(
            model_labels = model_labels,
            min_indexes = min_distances_index
        )
        end_time                       = time.time()
        service_time                   = end_time - local_start_time
        requestHeaders["Service-Time"] = str(service_time)

        logger.info({
            "event":"SKNN.PREDICT.2.COMPLETED",
            "model_id":model_id,
            "algorithm":algorithm,
            "service_time":service_time
        })

        return Response( #Returns the final response as a label vector + the headers
            response = json.dumps({
                "label_vector":label_vector.tolist(),
                "service_time":service_time
            }),
            status   = 200,
        )
    except Exception as e:
        logger.error({
            "msg":str(e)
        })
        return Response(
            response = None,
            status   = 503,
            headers  = {"Error-Message":str(e)}
        )

@classification.route("/pqc/sknn/predict",methods = ["POST"])
async def sknn_pqc_predict():
    headers         = request.headers
    head            = ["User-Agent","Accept-Encoding","Connection"]
    filteredHeaders = dict(list(filter(lambda x: not x[0] in head, headers.items())))
    step_index      = int(filteredHeaders.get("Step-Index",1))
    response        = Response()
    if step_index == 1:
        return await sknn_pqc_pedict_1(filteredHeaders)
    elif step_index == 2:
        return await sknn_pqc_predict_2(filteredHeaders)
    else:
        return response