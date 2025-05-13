import time, json, os
from flask import Blueprint,current_app,request,Response
from rory.core.utils.constants import Constants
from rory.core.classification.secure.distributed.sknn import SecureKNearestNeighbors as SKNN
from rory.core.classification.secure.pqc.sknn import SecureKNearestNeighbors as SKNNPQC
from rory.core.security.cryptosystem.pqc.ckks import Ckks
from rory.core.classification.knn import KNearestNeighbors as KNN
# from mictlanx.v4.client import Client as V4Client
from mictlanx import AsyncClient
from rory.core.interfaces.logger_metrics import LoggerMetrics
# from mictlanx.utils.index import Utils as MictlanXUtils
from utils.utils import Utils
# from mictlanx.v4.interfaces.responses import GetNDArrayResponse
from mictlanx.utils.segmentation import Chunks
import numpy.typing as npt
from option import Result, Some
from models import ExperimentLogEntry
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
    STORAGE_CLIENT:AsyncClient = current_app.config["ASYNC_STORAGE_CLIENT"]
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
    experiment_id            = requestHeaders.get("Experiment-Id","")
    MICTLANX_TIMEOUT        = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
    MICTLANX_DELAY          = int(os.environ.get("MICTLANX_DELAY","2"))
    MICTLANX_BACKOFF_FACTOR = float(os.environ.get("MICTLANX_BACKOFF_FACTOR","0.5"))
    MICTLANX_MAX_RETRIES    = int(os.environ.get("MICTLANX_MAX_RETRIES","10"))  
    # print("",requestHeaders)
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
        print("HEADERS",requestHeaders)
        encrypted_model = await RoryCommon.get_and_merge(
            client         = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = encrypted_model_id,
            delay          = MICTLANX_DELAY,
            max_retries    = MICTLANX_MAX_RETRIES,
            timeout        = MICTLANX_TIMEOUT,
            backoff_factor = MICTLANX_BACKOFF_FACTOR
        )
        
        get_encrypted_ptm_entry = ExperimentLogEntry(
            event          = "GET",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_merge_encrypted_model_start_time,
            end_time       = time.time(),
            id             = encrypted_model_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
        )
        logger.info(get_encrypted_ptm_entry.model_dump())

        response_headers["Encrypted-Model-Dtype"] = encrypted_model.dtype #["tags"]["dtype"] #Save the data type
        response_headers["Encrypted-Model-Shape"] = encrypted_model.shape #Save the shape
        
        encrypted_records = await RoryCommon.get_and_merge(
            client         = STORAGE_CLIENT,
            key            = encrypted_records_id,
            bucket_id      = BUCKET_ID,
            max_retries    = MICTLANX_MAX_RETRIES,
            delay          = MICTLANX_DELAY,
            backoff_factor = MICTLANX_BACKOFF_FACTOR,
            timeout        = MICTLANX_TIMEOUT,
        )
        
        get_encrypted_ptm_entry = ExperimentLogEntry(
            event          = "GET",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_merge_encrypted_model_start_time,
            end_time       = time.time(),
            id             = encrypted_records_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
        )
        logger.info(get_encrypted_ptm_entry.model_dump())

        response_headers["Encrypted-Records-Test-Dtype"] = encrypted_records.dtype #["tags"]["dtype"] #Save the data type
        response_headers["Encrypted-Records-Test-Shape"] = encrypted_records.shape #Save the shape

        all_distances = SKNN.calculate_distances(
			dataset  = encrypted_records,
			model    = encrypted_model,
            distance = distance,
		)

        distances_id = "distances{}".format(records_test_id) 
        distances_shape = all_distances.shape
        distances_dtype = all_distances.dtype

        calculate_distances_entry = ExperimentLogEntry(
            event          = "CALCULATE.DISTANCES",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_merge_encrypted_model_start_time,
            end_time       = time.time(),
            id             = distances_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
        )
        logger.info(calculate_distances_entry.model_dump())
                
        maybe_all_distances_chunks = Chunks.from_ndarray(
            ndarray = all_distances,
            group_id = distances_id,
            chunk_prefix = Some(distances_id),
            num_chunks = num_chunks
        )
        if maybe_all_distances_chunks.is_none:
            raise "something went wrong creating the chunks"
        put_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
            client    = STORAGE_CLIENT,
            bucket_id = BUCKET_ID,
            key       = distances_id,
            chunks    = maybe_all_distances_chunks.unwrap(),
            max_tries = MICTLANX_MAX_RETRIES,
            timeout   = MICTLANX_TIMEOUT,
            tags      = {
                "full_shape":str(distances_shape),
                "full_dtype":"float64"
            }
        )
        end_time     = time.time()
        service_time = end_time - local_start_time
        
        classification_completed_entry = ExperimentLogEntry(
            event          = "COMPLETED",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = local_start_time,
            end_time       = time.time(),
            id             = model_id,
            num_chunks     = num_chunks,
            time           = service_time
        )
        logger.info(classification_completed_entry.model_dump())
  
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
    STORAGE_CLIENT:AsyncClient = current_app.config["ASYNC_STORAGE_CLIENT"]
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
    experiment_id            = requestHeaders.get("Experiment-Id","")
    MICTLANX_TIMEOUT        = int(current_app.config.get("MICTLANX_TIMEOUT",120))
    MICTLANX_DELAY          = int(os.environ.get("MICTLANX_DELAY","2"))
    MICTLANX_BACKOFF_FACTOR = float(os.environ.get("MICTLANX_BACKOFF_FACTOR","0.5"))
    MICTLANX_MAX_RETRIES    = int(os.environ.get("MICTLANX_MAX_RETRIES","10"))

    try:
        model_labels_get_start_time = time.time()
        model_labels = await RoryCommon.get_and_merge(
            client         = STORAGE_CLIENT,
            key            = model_labels_id,
            bucket_id      = BUCKET_ID,
            backoff_factor = MICTLANX_BACKOFF_FACTOR,
            delay          = MICTLANX_DELAY,
            max_retries    = MICTLANX_MAX_RETRIES,
            timeout        = MICTLANX_TIMEOUT
        )

        get_encrypted_ptm_entry = ExperimentLogEntry(
            event          = "GET",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = model_labels_get_start_time,
            end_time       = time.time(),
            id             = model_labels_id,
            worker_id      = worker_id,
        )
        logger.info(get_encrypted_ptm_entry.model_dump())

        min_distances_start_time = time.time()
        min_distances_index = await RoryCommon.get_and_merge(
            client         = STORAGE_CLIENT,
            key            = min_distances_index_id,
            bucket_id      = BUCKET_ID,
            backoff_factor = MICTLANX_BACKOFF_FACTOR,
            delay          = MICTLANX_DELAY,
            max_retries    = MICTLANX_MAX_RETRIES,
            timeout        = MICTLANX_TIMEOUT
        )
        get_encrypted_ptm_entry = ExperimentLogEntry(
            event          = "GET",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = min_distances_start_time,
            end_time       = time.time(),
            id             = min_distances_index_id,
            worker_id      = worker_id,
        )
        logger.info(get_encrypted_ptm_entry.model_dump())
        
        label_vector = SKNN.get_label_vector(
            model_labels = model_labels.reshape((model_labels_shape[1],)),
            min_indexes  = min_distances_index
        )
        label_vector = label_vector.reshape((label_vector.shape[0],))
        end_time     = time.time()
        service_time = end_time - local_start_time
        requestHeaders["Service-Time"] = str(service_time)

        classification_completed_entry = ExperimentLogEntry(
            event          = "COMPLETED",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = local_start_time,
            end_time       = time.time(),
            id             = model_id,
            time           = service_time
        )
        logger.info(classification_completed_entry.model_dump())

        return Response( #Returns the final response as a label vector + the headers
            response = json.dumps({
                "label_vector":list(map(int, label_vector.flatten().tolist())),
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
    STORAGE_CLIENT:AsyncClient = current_app.config["ASYNC_STORAGE_CLIENT"]
    model_id                = filtered_headers.get("Model-Id","model0") #iris
    model_labels_id         = "{}labels".format(model_id) #iris_model_labels
    records_test_id         = filtered_headers.get("Records-Test-Id","matrix0")
    algorithm               = Constants.ClassificationAlgorithms.KNN_PREDICT
    response_headers        = {}
    distance                = current_app.config["DISTANCE"]
    experiment_id            = filtered_headers.get("Experiment-Id","")
    MICTLANX_TIMEOUT        = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
    MICTLANX_DELAY          = int(os.environ.get("MICTLANX_DELAY","2"))
    MICTLANX_BACKOFF_FACTOR = float(os.environ.get("MICTLANX_BACKOFF_FACTOR","0.5"))
    MICTLANX_MAX_RETRIES    = int(os.environ.get("MICTLANX_MAX_RETRIES","10"))
  
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
            max_retries    = MICTLANX_MAX_RETRIES,
            delay          = MICTLANX_DELAY,
            backoff_factor = MICTLANX_BACKOFF_FACTOR,
            timeout        = MICTLANX_TIMEOUT
        )

        get_model_entry = ExperimentLogEntry(
            event          = "GET",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_model_start_time,
            end_time       = time.time(),
            id             = model_id,
            worker_id      = worker_id,
        )
        logger.info(get_model_entry.model_dump())

        get_model_labels_start_time = time.time()
        model_labels = await RoryCommon.get_and_merge(
            client         = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = model_labels_id,
            max_retries    = MICTLANX_MAX_RETRIES,
            delay          = MICTLANX_DELAY,
            backoff_factor = MICTLANX_BACKOFF_FACTOR,
            timeout        = MICTLANX_TIMEOUT,
        )
        
        get_model_labels_entry = ExperimentLogEntry(
            event          = "GET",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_model_labels_start_time,
            end_time       = time.time(),
            id             = model_labels_id,
            worker_id      = worker_id,
        )
        logger.info(get_model_labels_entry.model_dump())

        get_records_start_time = time.time()
        records = await RoryCommon.get_and_merge(
            client         = STORAGE_CLIENT,
            key            = records_test_id,
            bucket_id      = BUCKET_ID,
            max_retries    = MICTLANX_MAX_RETRIES,
            delay          = MICTLANX_DELAY,
            backoff_factor = MICTLANX_BACKOFF_FACTOR,
            timeout        = MICTLANX_TIMEOUT
        )
        
        get_records_entry = ExperimentLogEntry(
            event          = "GET",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_records_start_time,
            end_time       = time.time(),
            id             = records_test_id,
            worker_id      = worker_id,
        )
        logger.info(get_records_entry.model_dump())

        knn_predict_start_time = time.time()
        label_vector:npt.NDArray = KNN.predict(
            dataset      = records,
            model        = model,
            model_labels = model_labels.reshape((model_labels_shape[1],)),
            distance     = distance
        )
        knn_predict_st                   = time.time() - knn_predict_start_time        
        end_time                         = time.time()
        service_time                     = end_time - local_start_time
        response_headers["Service-Time"] = str(service_time)
        
        classification_completed_entry = ExperimentLogEntry(
            event          = "COMPLETED",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = local_start_time,
            end_time       = time.time(),
            id             = model_id,
            time           = service_time
        )
        logger.info(classification_completed_entry.model_dump())

        return Response( #Returns the final response as a label vector + the headers
            response = json.dumps({
                "label_vector":list(map(int, label_vector.flatten().tolist())),
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
    STORAGE_CLIENT:AsyncClient  = current_app.config["ASYNC_STORAGE_CLIENT"]
    BUCKET_ID:str            = current_app.config.get("BUCKET_ID","rory")
    model_id                 = requestHeaders.get("Model-Id","model0") #iris
    encrypted_model_id       = "encrypted{}".format(model_id) #encrypted-iris_model
    model_labels_id          = "{}labels".format(model_id) #iris_model_labels
    records_test_id          = requestHeaders.get("Records-Test-Id","matrix0")
    distances_id             = "distances{}".format(records_test_id) 
    encrypted_records_id     = "encrypted{}".format(records_test_id) # The id of the encrypted matrix is built
    algorithm                = Constants.ClassificationAlgorithms.SKNN_PQC_PREDICT
    _encrypted_model_shape   = requestHeaders.get("Encrypted-Model-Shape",-1)
    _encrypted_model_dtype   = requestHeaders.get("Encrypted-Model-Dtype",-1)
    _encrypted_records_shape = requestHeaders.get("Encrypted-Records-Shape",-1)
    _encrypted_records_dtype = requestHeaders.get("Encrypted-Records-Dtype",-1)
    _round                   = bool(int(os.environ.get("_round","0"))) #False
    decimals                 = int(os.environ.get("DECIMALS","2"))
    experiment_id            = requestHeaders.get("Experiment-Id","")
    MICTLANX_TIMEOUT         = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
    MICTLANX_DELAY           = int(os.environ.get("MICTLANX_DELAY","2"))
    MICTLANX_BACKOFF_FACTOR  = float(os.environ.get("MICTLANX_BACKOFF_FACTOR","0.5"))
    MICTLANX_MAX_RETRIES     = int(os.environ.get("MICTLANX_MAX_RETRIES","10")) 

    MICTLANX_CHUNK_SIZE        = os.environ.get("MICTLANX_CHUNK_SIZE","256kb")
    MICTLANX_MAX_PARALELL_GETS = int(os.environ.get("MICTLANX_MAX_PARALELL_GETS","2"))
    
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
    relinkey_filename  = os.environ.get("RELINKEY_FILENAME","relinkey")

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
        return Response("Num-Chunks header is required", status=503)
    try:
        response_headers["Start-Time"] = str(local_start_time)

        get_merge_encrypted_model_start_time = time.time()
        encrypted_model = await RoryCommon.get_pyctxt_matrix(
            client            = STORAGE_CLIENT,
            bucket_id         = BUCKET_ID,
            key               = encrypted_model_id,
            ckks              = ckks,
            backoff_factor    = MICTLANX_BACKOFF_FACTOR,
            chunk_size        = MICTLANX_CHUNK_SIZE,
            delay             = MICTLANX_DELAY,
            max_paralell_gets = MICTLANX_MAX_PARALELL_GETS,
            max_retries       = MICTLANX_MAX_RETRIES,
            timeout           = MICTLANX_TIMEOUT
        )
        
        get_encrypted_ptm_entry = ExperimentLogEntry(
            event          = "GET",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_merge_encrypted_model_start_time,
            end_time       = time.time(),
            id             = encrypted_model_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
        )
        logger.info(get_encrypted_ptm_entry.model_dump())
        
        t1 = time.time()
        encrypted_records = await RoryCommon.get_pyctxt_matrix(
            client            = STORAGE_CLIENT,
            bucket_id         = BUCKET_ID,
            key               = encrypted_records_id,
            ckks              = ckks,
            backoff_factor    = MICTLANX_BACKOFF_FACTOR,
            chunk_size        = MICTLANX_CHUNK_SIZE,
            delay             = MICTLANX_DELAY,
            headers           = {},
            max_paralell_gets = MICTLANX_MAX_PARALELL_GETS,
            max_retries       = MICTLANX_MAX_RETRIES,
            timeout           = MICTLANX_TIMEOUT
        )
        encrypted_records_get_rt = time.time() - t1
        
        get_encrypted_ptm_entry = ExperimentLogEntry(
            event          = "GET",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = t1,
            end_time       = time.time(),
            id             = encrypted_records_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
        )
        logger.info(get_encrypted_ptm_entry.model_dump())

        t1 = time.time()
        all_distances = SKNNPQC.calculate_distances(
			dataset = encrypted_records,
			model   = encrypted_model,
            model_shape   = encrypted_model_shape,
            dataset_shape = encrypted_records_shape
		)
        
        calculate_distances_entry = ExperimentLogEntry(
            event          = "CALCULATE.DISTANCES",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = t1,
            end_time       = time.time(),
            id             = distances_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
        )
        logger.info(calculate_distances_entry.model_dump())

        distances_shape = all_distances.shape
        distances_dtype = all_distances.dtype
        maybe_distances_chunks = RoryCommon.from_pyctxt_matrix_to_chunks(
            key        = distances_id,
            num_chunks = num_chunks,
            xs         = all_distances
        )
        if maybe_distances_chunks.is_none:
            return Response(status =500,response = "Failed to create distances chunks")
        
        t1 = time.time()
        z = await RoryCommon.put_chunks(
            client = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = distances_id,
            chunks         = maybe_distances_chunks.unwrap()
        )   

        put_encrypted_ptm_entry = ExperimentLogEntry(
            event          = "PUT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = t1,
            end_time       = time.time(),
            id             = distances_id,
            worker_id      = "",
            num_chunks     = num_chunks,
        )
        logger.info(put_encrypted_ptm_entry.model_dump())
        
        end_time     = time.time()
        service_time = end_time - local_start_time

        classification_completed_entry = ExperimentLogEntry(
            event          = "COMPLETED",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = local_start_time,
            end_time       = time.time(),
            id             = model_id,
            num_chunks     = num_chunks,
            time           = service_time
        )
        logger.info(classification_completed_entry.model_dump())

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
    STORAGE_CLIENT:AsyncClient = current_app.config["ASYNC_STORAGE_CLIENT"]
    BUCKET_ID:str           = current_app.config.get("BUCKET_ID","rory")
    model_id                = requestHeaders.get("Model-Id","model0") #iris
    model_labels_id         = "{}labels".format(model_id) #iris_model_labels
    records_test_id         = requestHeaders.get("Records-Test-Id","matrix0")
    min_distances_index_id  = "distancesindex{}".format(records_test_id)
    algorithm               = Constants.ClassificationAlgorithms.SKNN_PQC_PREDICT

    experiment_id            = requestHeaders.get("Experiment-Id","")
    MICTLANX_TIMEOUT         = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
    MICTLANX_DELAY           = int(os.environ.get("MICTLANX_DELAY","2"))
    MICTLANX_BACKOFF_FACTOR  = float(os.environ.get("MICTLANX_BACKOFF_FACTOR","0.5"))
    MICTLANX_MAX_RETRIES     = int(os.environ.get("MICTLANX_MAX_RETRIES","10")) 

    MICTLANX_CHUNK_SIZE        = os.environ.get("MICTLANX_CHUNK_SIZE","256kb")
    MICTLANX_MAX_PARALELL_GETS = int(os.environ.get("MICTLANX_MAX_PARALELL_GETS","2"))

    try:
        model_labels_get_start_time = time.time()
        model_labels = await RoryCommon.get_and_merge(
            client            = STORAGE_CLIENT,
            key               = model_labels_id,
            bucket_id         = BUCKET_ID,
            backoff_factor    = MICTLANX_BACKOFF_FACTOR,
            chunk_size        = MICTLANX_CHUNK_SIZE,
            delay             = MICTLANX_DELAY,
            max_paralell_gets = MICTLANX_MAX_PARALELL_GETS,
            max_retries       = MICTLANX_MAX_RETRIES,
            timeout           = MICTLANX_TIMEOUT
        )

        get_encrypted_ptm_entry = ExperimentLogEntry(
            event          = "GET",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = model_labels_get_start_time,
            end_time       = time.time(),
            id             = model_labels_id,
            worker_id      = worker_id,
        )
        logger.info(get_encrypted_ptm_entry.model_dump())

        min_distances_start_time = time.time()
        min_distances_index = await RoryCommon.get_and_merge(
            client            = STORAGE_CLIENT,
            key               = min_distances_index_id,
            bucket_id         = BUCKET_ID,
            backoff_factor    = MICTLANX_BACKOFF_FACTOR,
            chunk_size        = MICTLANX_CHUNK_SIZE,
            delay             = MICTLANX_DELAY,
            max_paralell_gets = MICTLANX_MAX_PARALELL_GETS,
            max_retries       = MICTLANX_MAX_RETRIES,
            timeout           = MICTLANX_TIMEOUT
        )

        get_encrypted_ptm_entry = ExperimentLogEntry(
            event          = "GET",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = min_distances_start_time,
            end_time       = time.time(),
            id             = min_distances_index_id,
            worker_id      = worker_id,
        )
        logger.info(get_encrypted_ptm_entry.model_dump())

        label_vector = SKNNPQC.get_label_vector(
            model_labels = model_labels.flatten(),
            min_indexes = min_distances_index.flatten()
        )
        end_time                       = time.time()
        service_time                   = end_time - local_start_time
        requestHeaders["Service-Time"] = str(service_time)

        classification_completed_entry = ExperimentLogEntry(
            event          = "COMPLETED",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = local_start_time,
            end_time       = time.time(),
            id             = model_id,
            time           = service_time
        )
        logger.info(classification_completed_entry.model_dump())

        return Response( #Returns the final response as a label vector + the headers
            response = json.dumps({
                "label_vector":list(map(int, label_vector.flatten().tolist())),
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