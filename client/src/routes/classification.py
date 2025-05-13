import os
import pickle as PK
import time, json
import numpy as np
import numpy.typing as npt
from uuid import uuid4
from requests import Session
from flask import Blueprint,current_app,request,Response
from rory.core.interfaces.rorymanager import RoryManager
from rory.core.interfaces.roryworker import RoryWorker
from rory.core.security.dataowner import DataOwner
from rory.core.security.pqc.dataowner import DataOwner as DataOwnerPQC
from rory.core.security.cryptosystem.liu import Liu
from rory.core.utils.constants import Constants
from rorycommon import Common as RoryCommon
# from mictlanx.v4.client import Client  as V4Client
from mictlanx import AsyncClient
from mictlanx.utils.segmentation import Chunks
from concurrent.futures import ProcessPoolExecutor
from utils.utils import Utils
from option import Some
from models import ExperimentLogEntry
from rory.core.security.cryptosystem.pqc.ckks import Ckks

classification = Blueprint("classification",__name__,url_prefix = "/classification")

@classification.route("/test",methods=["GET","POST"])
def test():
    return Response(
        response = json.dumps({
            "component_type":"client"
        }),
        status   = 200,
        headers  = {
            "Component-Type":"client"
        }
    )

@classification.route("/sknn/train",methods = ["POST"])
async def sknn_train():
    try:
        local_start_time             = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        liu:Liu                      = current_app.config.get("liu")
        dataowner:DataOwner          = current_app.config.get("dataowner")
        STORAGE_CLIENT:AsyncClient   = current_app.config.get("ASYNC_STORAGE_CLIENT")
        _num_chunks                  = current_app.config.get("NUM_CHUNKS",4)
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        np_random                    = current_app.config.get("np_random")
        security_level               = current_app.config.get("LIU_SECURITY_LEVEL",128)
        if executor == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm             = Constants.ClassificationAlgorithms.SKNN_TRAIN
        s                     = Session()
        request_headers       = request.headers #Headers for the request
        experiment_id         = request_headers.get("Experiment-Id",uuid4().hex[:10])
        model_id              = request_headers.get("Model-Id","matrix-0_model") #fertility-0_kjkk
        model_filename        = request_headers.get("Model-Filename",model_id)   #fertility_model
        model_labels_id       = "{}labels".format(model_id) #fertility_model_labels
        model_labels_filename = request_headers.get("Model-Labels-Filename",model_labels_id)   #fertility_model_labels     
        encrypted_model_id    = "encrypted{}".format(model_id) #encrypted-fertility_model
        extension             = request_headers.get("Extension","npy")
        m                     = dataowner.m
        num_chunks            = int(request_headers.get("Num-Chunks",_num_chunks))
        model_path            = "{}/{}.{}".format(SOURCE_PATH, model_filename, extension)
        model_labels_path     = "{}/{}.{}".format(SOURCE_PATH, model_labels_filename, extension)
        MICTLANX_TIMEOUT      = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
        max_workers           = Utils.get_workers(num_chunks=num_chunks)
        model_path_exists        = os.path.exists(model_path) 
        model_path_labels_exists = os.path.exists(model_labels_path)
        if not model_path_exists or not model_path_labels_exists:
            return Response(response="Either model or label vector not found", status=500)
        else:
            
            model_result = await RoryCommon.read_numpy_from(
                path      = model_path,
                extension = "npy"
            )
            if model_result.is_err:
                return Response(status=500,response="Something went wrong reading the model")
            model = model_result.unwrap()
            read_local_model_labels_start_time = time.time()

            model_labels_result = await RoryCommon.read_numpy_from(
                extension = "npy",
                path      = model_labels_path
            )
            if model_labels_result.is_err:
                return Response(status=500,response="Something went wrong reading the model labels")
            model_labels = model_labels_result.unwrap()
            model_labels = model_labels.reshape((1,model_labels.shape[0]))
            
            local_read_entry = ExperimentLogEntry(
                event          = "LOCAL.READ",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = read_local_model_labels_start_time,
                end_time       = time.time(),
                id             = model_id,
                worker_id      = "",
                num_chunks     = num_chunks,
                security_level = security_level,
                m              = m
            )
            logger.info(local_read_entry.model_dump())
            
            put_model_labels_start_time = time.time()
            maybe_models_labels_chunks  = Chunks.from_ndarray(
                ndarray      = model_labels, 
                group_id     = model_labels_id,
                num_chunks   = 1,
                chunk_prefix = Some(model_labels_id)
                )
            if maybe_models_labels_chunks.is_none:
                return Response(status=500, response="Something went wrong generating the chunks of model labels")
            
            put_model_labels_result = await RoryCommon.delete_and_put_chunks(
                client    = STORAGE_CLIENT, 
                bucket_id = BUCKET_ID, 
                key       = model_labels_id,
                chunks    = maybe_models_labels_chunks.unwrap(), 
                timeout   = MICTLANX_TIMEOUT,
                tags      = {
                    "full_dtype":str(model_labels.dtype),
                    "full_shape":str(model_labels.shape)
                }
            )
            
            put_encrypted_ptm_entry = ExperimentLogEntry(
                event          = "PUT",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = put_model_labels_start_time,
                end_time       = time.time(),
                id             = model_id,
                worker_id      = "",
                num_chunks     = num_chunks,
                security_level = security_level,
                m              = m
            )
            logger.info(put_encrypted_ptm_entry.model_dump())

            r:int = model.shape[0]
            a:int = model.shape[1]
            encrypted_model_shape = "({},{},{})".format(r,a,m)
            n = a*r*int(m)

            segment_encrypt_model_start_time = time.time()
            encrypted_model_chunks = RoryCommon.segment_and_encrypt_liu_with_executor( #Encrypt 
                executor         = executor,
                key              = encrypted_model_id,
                plaintext_matrix = model,
                dataowner        = dataowner,
                n                = n,
                num_chunks       = num_chunks,
                np_random        = np_random
            )
            
            segment_encrypt_entry = ExperimentLogEntry(
                event          = "SEGMENT.ENCRYPT",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = segment_encrypt_model_start_time,
                end_time       = time.time(),
                id             = model_id,
                worker_id      = "",
                num_chunks     = num_chunks,
                security_level = security_level,
                m              = m
            )
            logger.info(segment_encrypt_entry.model_dump())
            
            put_chunked_start_time = time.time()
            encrypted_model_put_chunks = await RoryCommon.delete_and_put_chunks(
                client    = STORAGE_CLIENT,
                bucket_id = BUCKET_ID,
                key       = encrypted_model_id,
                chunks    = encrypted_model_chunks,
                timeout   = MICTLANX_TIMEOUT,
                tags      = {
                    "full_shape": str(encrypted_model_shape),
                    "full_dtype":"float64"
                }
            )
            
            put_encrypted_ptm_entry = ExperimentLogEntry(
                event          = "PUT",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = put_chunked_start_time,
                end_time       = time.time(),
                id             = model_id,
                worker_id      = "",
                num_chunks     = num_chunks,
                security_level = security_level,
                m              = m
            )
            logger.info(put_encrypted_ptm_entry.model_dump())

            endTime       = time.time() # Get the time when it ends
            response_time = endTime - local_start_time # Get the service time

            classification_completed_entry = ExperimentLogEntry(
                event          = "COMPLETED",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = local_start_time,
                end_time       = time.time(),
                id             = model_id,
                num_chunks     = num_chunks,
                security_level = security_level,
                workers        = max_workers,
                time           = response_time
            )
            logger.info(classification_completed_entry.model_dump())

            return Response(
                response = json.dumps({
                    "response_time": response_time,
                    "encrypted_model_shape":str(encrypted_model_shape),
                    "encrypted_model_dtype":"float64",
                    "algorithm":algorithm,
                    "model_labels_shape":list(model_labels.shape)
                }),
                status  = 200,
            )
    except Exception as e:
        logger.error({
            "msg":str(e)
        })
        return Response(response = str(e), status = 500, headers={"Error-Message":str(e)})


@classification.route("/sknn/predict",methods = ["POST"])
async def sknn_predict():
    try:
        local_start_time             = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        TESTING                      = current_app.config.get("TESTING",True)
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        liu:Liu                      = current_app.config.get("liu")
        dataowner:DataOwner          = current_app.config.get("dataowner")
        STORAGE_CLIENT:AsyncClient   = current_app.config.get("ASYNC_STORAGE_CLIENT")
        _num_chunks                  = current_app.config.get("NUM_CHUNKS",4)
        np_random:bool               = current_app.config.get("np_random",False)
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        WORKER_TIMEOUT               = int(current_app.config.get("WORKER_TIMEOUT",300))
        security_level               = current_app.config.get("LIU_SECURITY_LEVEL",128)
        if executor == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm                 = Constants.ClassificationAlgorithms.SKNN_PREDICT
        s                         = Session()
        request_headers           = request.headers #Headers for the request
        num_chunks                = int(request_headers.get("Num-Chunks",_num_chunks))
        model_id                  = request_headers.get("Model-Id","model0") ##fertility-0
        model_filename            = request_headers.get("Model-Filename",model_id) #fertility_model
        records_test_id           = request_headers.get("Records-Test-Id","matrix0data") #fertility_data
        records_test_filename     = request_headers.get("Records-Test-Filename",records_test_id)
        encrypted_records_test_id = "encrypted{}".format(records_test_id) # The id of the encrypted matrix is built
        extension                 = request_headers.get("Extension","npy")
        m                         = dataowner.m
        model_labels_id           = "{}labels".format(model_id)
        _encrypted_model_shape    = request_headers.get("Encrypted-Model-Shape",-1)
        _encrypted_model_dtype    = request_headers.get("Encrypted-Model-Dtype",-1)
        _model_labels_shape       = request_headers.get("Model-Labels-Shape",-1)
        experiment_id             = request_headers.get("Experiment-Id",uuid4().hex[:10])
        records_test_path       = "{}/{}.{}".format(SOURCE_PATH, records_test_filename, extension)
        max_workers             = Utils.get_workers(num_chunks=num_chunks)
        MICTLANX_TIMEOUT        = int(current_app.config.get("MICTLANX_TIMEOUT",120))
        MICTLANX_DELAY          = int(os.environ.get("MICTLANX_DELAY","2"))
        MICTLANX_BACKOFF_FACTOR = float(os.environ.get("MICTLANX_BACKOFF_FACTOR","0.5"))
        MICTLANX_MAX_RETRIES    = int(os.environ.get("MICTLANX_MAX_RETRIES","10"))
        
        if _encrypted_model_dtype == -1:
            return Response("Encrypted-Model-Dtype", status=500)
        if _encrypted_model_shape == -1 :
            return Response("Encrypted-Model-Shape header is required", status=500)
        if _model_labels_shape == -1:
            return Response("Model-Labels-Shape header is required", status=500)
        

        read_local_start_time = time.time()
        records_test_ext    = "npy"
        records_test_result = await RoryCommon.read_numpy_from(
            path      = records_test_path, 
            extension = records_test_ext
        )
        read_local_st = time.time() - read_local_start_time
        if records_test_result.is_err:
            return Response(status=500, response=f"Failed to read {records_test_path}")
        records_test = records_test_result.unwrap()

        local_read_entry = ExperimentLogEntry(
            event          = "LOCAL.READ",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = read_local_st,
            end_time       = time.time(),
            id             = model_id,
            worker_id      = "",
            num_chunks     = num_chunks,
            security_level = security_level,
            m              = m
        )
        logger.info(local_read_entry.model_dump())

        r:int = records_test.shape[0]
        a:int = records_test.shape[1]
        n     = a*r*int(m)

        segment_encrypt_start_time = time.time()
        encrypted_records_chunks =  RoryCommon.segment_and_encrypt_liu_with_executor( #Encrypt 
            executor         = executor,
            key              = encrypted_records_test_id,
            dataowner        = dataowner,
            plaintext_matrix = records_test,
            n                = n,
            num_chunks       = num_chunks,
            np_random        = np_random
        )
        
        segment_encrypt_entry = ExperimentLogEntry(
            event          = "SEGMENT.ENCRYPT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = segment_encrypt_start_time,
            end_time       = time.time(),
            id             = model_id,
            worker_id      = "",
            num_chunks     = num_chunks,
            security_level = security_level,
            m              = m
        )
        logger.info(segment_encrypt_entry.model_dump())
        
        put_chunks_start_time = time.time()
        encrypted_records_shape = (r,a,int(m))
        
        put_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
            client    = STORAGE_CLIENT,
            bucket_id = BUCKET_ID,
            key       = encrypted_records_test_id,
            chunks    = encrypted_records_chunks,
            timeout   = MICTLANX_TIMEOUT,
            tags      = {
                "full_shape": str(encrypted_records_shape),
                "full_dtype":"float64"
            }
        )
        if put_chunks_generator_results.is_err:
            return Response(status=500, response="Failed to put encrypted records test in the storage")
        service_time_client = time.time() - local_start_time
        
        put_encrypted_ptm_entry = ExperimentLogEntry(
            event          = "PUT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = put_chunks_start_time,
            end_time       = time.time(),
            id             = model_id,
            worker_id      = "",
            num_chunks     = num_chunks,
            security_level = security_level,
            m              = m
        )
        logger.info(put_encrypted_ptm_entry.model_dump())
        
        managerResponse:RoryManager = current_app.config.get("manager") # Communicates with the manager

        get_worker_start_time = time.time()
        get_worker_result     = managerResponse.getWorker( #Gets the worker from the manager
            headers = {
                "Algorithm"             : algorithm,
                "Start-Request-Time"    : str(local_start_time),
                "Start-Get-Worker-Time" : str(get_worker_start_time),
                "Matrix-Id"             : model_id
            }
        )

        if get_worker_result.is_err:
            error = get_worker_result.unwrap_err()
            logger.error(str(error))
            return Response(str(error), status=500)
        (_worker_id,port) = get_worker_result.unwrap()

        get_worker_end_time     = time.time() 
        get_worker_service_time = get_worker_end_time - get_worker_start_time
        worker_id               =  "localhost" if TESTING else _worker_id

        get_worker_entry = ExperimentLogEntry(
            event          = "GET.WORKER",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_worker_start_time,
            end_time       = time.time(),
            id             = model_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            workers        = max_workers,
            security_level = security_level,
            m              = m
        )
        logger.info(get_worker_entry.model_dump())

        worker_start_time = time.time()
        worker = RoryWorker( #Allows to establish the connection with the worker
            workerId  = worker_id,
            port      = port,
            session   = s,
            algorithm = algorithm,
        )
        
        encrypted_records_dtype = "float64"
        run1_time = time.time()
        run1_headers = {
            "Step-Index"              : "1",
            "Records-Test-Id"         : records_test_id,
            "Model-Id"                : model_id,
            "Encrypted-Model-Shape"   : _encrypted_model_shape,
            "Encrypted-Model-Dtype"   : _encrypted_model_dtype,
            "Encrypted-Records-Shape" : str(encrypted_records_shape),
            "Encrypted-Records-Dtype" : str(encrypted_records_dtype),
            "Num-Chunks"              : str(num_chunks),
            "Model-Labels-Shape"      : _model_labels_shape
        }

        worker_run1_response = worker.run(
            headers = run1_headers,
            timeout = WORKER_TIMEOUT
        )
        worker_run1_response.raise_for_status()

        jsonWorkerResponse   = worker_run1_response.json()
        endTime              = time.time() # Get the time when it ends
        distances_id         = jsonWorkerResponse["distances_id"]
        distances_shape      = jsonWorkerResponse["distances_shape"]
        distances_dtype      = jsonWorkerResponse["distances_dtype"]
        worker_service_time  = jsonWorkerResponse["service_time"]
         
        run1_worker_entry = ExperimentLogEntry(
            event          = "RUN1",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = run1_time,
            end_time       = time.time(),
            id             = model_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            m              = m,
            workers        = max_workers,
            security_level = security_level
        )
        logger.info(run1_worker_entry.model_dump())

        get_all_distances_start_time = time.time()
        all_distances = await RoryCommon.get_and_merge(
            client         = STORAGE_CLIENT,
            key            = distances_id,
            bucket_id      = BUCKET_ID,
            max_retries    = MICTLANX_MAX_RETRIES,
            delay          = MICTLANX_DELAY,
            backoff_factor = MICTLANX_BACKOFF_FACTOR,
            timeout        = MICTLANX_TIMEOUT
        )
         
        get_encrypted_entry = ExperimentLogEntry(
            event          = "GET",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_all_distances_start_time,
            end_time       = time.time(),
            id             = model_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            m              = m,
            workers        = max_workers,
            security_level = security_level
        )
        logger.info(get_encrypted_entry.model_dump())
        
        decrypt_matrix_start_time = time.time()
        matrix_distances_plain = liu.decryptMatrix(
            ciphertext_matrix = all_distances,
            secret_key        = dataowner.sk,
        )

        min_distances_index         = np.argmin(matrix_distances_plain.matrix,axis=1)
        min_distances_index_id      = "distancesindex{}".format(records_test_id)
        decrypt_matrix_end_time     = time.time()
        decrypt_matrix_service_time = decrypt_matrix_start_time - decrypt_matrix_end_time

        decrypt_entry = ExperimentLogEntry(
            event          = "DECRYPT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = decrypt_matrix_start_time,
            end_time       = time.time(),
            id             = model_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            m              = m,
            workers        = max_workers,
            security_level = security_level
        )
        logger.info(decrypt_entry.model_dump())
        
        maybe_min_distances_chunks = Chunks.from_ndarray(
            ndarray      = min_distances_index.reshape(-1,1),
            group_id     = min_distances_index_id,
            chunk_prefix = Some(min_distances_index_id),
            num_chunks   = num_chunks,
        )

        if maybe_min_distances_chunks.is_none:
            raise "something went wrong creating the chunks"

        t_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
            client = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = min_distances_index_id,
            chunks         = maybe_min_distances_chunks.unwrap(),
            tags = {
                "full_shape": str(min_distances_index.shape),
                "full_dtype": str(min_distances_index.dtype)
            }
        )

        run2_time = time.time()
        run2_headers = {
            "Step-Index"              : "2",
            "Records-Test-Id"         : records_test_id,
            "Model-Id"                : model_id,
            "Encrypted-Model-Shape"   : _encrypted_model_shape,
            "Encrypted-Model-Dtype"   : _encrypted_model_dtype,
            "Encrypted-Records-Shape" : str(encrypted_records_shape),
            "Encrypted-Records-Dtype" : str(encrypted_records_dtype),
            "Num-Chunks"              : str(num_chunks),
            "Min_Distances_Index_Id"  : min_distances_index_id,
            "Model-Labels-Shape"      : _model_labels_shape
        }

        worker_run2_response = worker.run(
            headers = run2_headers,
            timeout = WORKER_TIMEOUT
        )
        worker_run2_response.raise_for_status()
        jsonWorkerResponse2   = worker_run2_response.json()
        service_time_worker   = worker_run2_response.headers.get("Service-Time",0)
        worker_end_time       = time.time()
        worker_response_time  = worker_end_time - worker_start_time

        run2_worker_entry = ExperimentLogEntry(
            event          = "RUN2",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = run2_time,
            end_time       = time.time(),
            id             = model_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            m              = m,
            workers        = max_workers,
            security_level = security_level,
        )
        logger.info(run2_worker_entry.model_dump())
        
        response_time = endTime - local_start_time # Get the service time

        classification_completed_entry = ExperimentLogEntry(
            event          = "COMPLETED",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = local_start_time,
            end_time       = time.time(),
            id             = model_id,
            num_chunks     = num_chunks,
            security_level = security_level,
            workers        = max_workers,
            time           = response_time,
            m              = m
        )
        logger.info(classification_completed_entry.model_dump())
        
        label_vector = jsonWorkerResponse2["label_vector"]
        return Response(
            response = json.dumps({
                "label_vector":label_vector,
                "worker_id":worker_id,
                "service_time_manager":get_worker_service_time,
                "service_time_worker":worker_response_time,
                "service_time_client":service_time_client,
                "service_time_predict":response_time,
                "algorithm":algorithm,
                
            }),
            status   = 200,
            headers  = {}
        )
    except Exception as e:
        logger.error({
            "msg":str(e)
        })
        return Response(response = None, status = 500, headers={"Error-Message":str(e)})


@classification.route("/knn/train", methods = ["POST"])
async def knn_train():
    local_start_time             = time.time()
    logger                       = current_app.config["logger"]
    BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
    SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
    STORAGE_CLIENT:AsyncClient   = current_app.config.get("ASYNC_STORAGE_CLIENT")
    executor:ProcessPoolExecutor = current_app.config.get("executor")
    if executor == None:
        raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
    algorithm             = Constants.ClassificationAlgorithms.KNN_TRAIN
    s                     = Session()
    request_headers       = request.headers #Headers for the request
    num_chunks            = int(request_headers.get("Num-Chunks",1))
    model_id              = request_headers.get("Model-Id","matrix0model")        
    model_filename        = request_headers.get("Model-Filename",model_id)        
    model_labels_id       = "{}labels".format(model_id)
    model_labels_filename = request_headers.get("Model-Labels-Filename",model_labels_id)        
    extension             = request_headers.get("Extension","npy")
    experiment_id         = request_headers.get("Experiment-Id",uuid4().hex[:10])
    model_path            = "{}/{}.{}".format(SOURCE_PATH, model_filename, extension)
    model_labels_path     = "{}/{}.{}".format(SOURCE_PATH, model_labels_filename, extension)

    get_model_start_time = time.time()
    model_ext    = "npy"
    model_result = await RoryCommon.read_numpy_from(
        path      = model_path,
        extension = model_ext
    )

    if model_result.is_err:
        return Response(status=500, response="Failed to read model")

    model        = model_result.unwrap()

    local_read_entry = ExperimentLogEntry(
        event          = "LOCAL.READ",
        experiment_id  = experiment_id,
        algorithm      = algorithm,
        start_time     = get_model_start_time,
        end_time       = time.time(),
        id             = model_id,
        worker_id      = "",
        num_chunks     = num_chunks,
    )
    logger.info(local_read_entry.model_dump())
    
    model_labels_ext    = "npy"
    model_labels_result = await RoryCommon.read_numpy_from(
        path      = model_labels_path,
        extension = model_labels_ext
    )

    if model_labels_result.is_err:
        return Response(status=500, response="Failed to read model labels")
    
    model_labels = model_labels_result.unwrap()
    model_labels = model_labels.reshape(1,-1)

    put_model_start_time = time.time()
    maybe_model_chunks   = Chunks.from_ndarray(
        ndarray      = model,
        group_id     = model_id,
        chunk_prefix = Some(model_id),
        num_chunks   = num_chunks,
    )

    if maybe_model_chunks.is_none:
        return Response(status=500, response="something went wrong creating the chunks")

    put_model_result = await RoryCommon.delete_and_put_chunks(
        client         = STORAGE_CLIENT,
        bucket_id      = BUCKET_ID,
        key            = model_id,
        chunks         = maybe_model_chunks.unwrap(),
        tags = {
            "full_shape": str(model.shape),
            "full_dtype": str(model.dtype)
        }
    )
    
    put_encrypted_model_entry = ExperimentLogEntry(
        event          = "PUT",
        experiment_id  = experiment_id,
        algorithm      = algorithm,
        start_time     = put_model_start_time,
        end_time       = time.time(),
        id             = model_id,
        worker_id      = "",
        num_chunks     = num_chunks
    )
    logger.info(put_encrypted_model_entry.model_dump())

    put_model_labels_start_time = time.time()
    maybe_model_labels_chunks = Chunks.from_ndarray(
        ndarray      = model_labels,
        group_id     = model_labels_id,
        chunk_prefix = Some(model_labels_id),
        num_chunks   = num_chunks,
    )

    if maybe_model_labels_chunks.is_none:
        raise "something went wrong creating the chunks"

    model_labels_results = await RoryCommon.delete_and_put_chunks(
        client         = STORAGE_CLIENT,
        bucket_id      = BUCKET_ID,
        key            = model_labels_id,
        chunks         = maybe_model_labels_chunks.unwrap(),
        tags = {
            "full_shape": str(model_labels.shape),
            "full_dtype": str(model_labels.dtype)
        }
    )
    
    put_encrypted_model_labels_entry = ExperimentLogEntry(
        event          = "PUT",
        experiment_id  = experiment_id,
        algorithm      = algorithm,
        start_time     = put_model_labels_start_time,
        end_time       = time.time(),
        id             = model_id,
        worker_id      = "",
        num_chunks     = num_chunks
    )
    logger.info(put_encrypted_model_labels_entry.model_dump())

    
    end_time      = time.time() # Get the time when it ends
    response_time = end_time - local_start_time # Get the service time
    
    classification_completed_entry = ExperimentLogEntry(
        event          = "COMPLETED",
        experiment_id  = experiment_id,
        algorithm      = algorithm,
        start_time     = local_start_time,
        end_time       = time.time(),
        id             = model_id,
        num_chunks     = num_chunks,
        time           = response_time
    )
    logger.info(classification_completed_entry.model_dump())

    return Response(
        response = json.dumps({
            "response_time": response_time,
            "algorithm"   : algorithm,
            "model_labels_shape":list(model_labels.shape)
        }),
        status   = 200,
        headers  = {}
    )


@classification.route("/knn/predict",methods = ["POST"])
async def knn_predict():
    try:
        local_start_time             = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        TESTING                      = current_app.config.get("TESTING",True)
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        STORAGE_CLIENT:AsyncClient   = current_app.config.get("ASYNC_STORAGE_CLIENT")
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        WORKER_TIMEOUT               = int(current_app.config.get("WORKER_TIMEOUT",300))
        MICTLANX_TIMEOUT             = int(current_app.config.get("MICTLANX_TIMEOUT",120))
        if executor == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm             = Constants.ClassificationAlgorithms.KNN_PREDICT
        s                     = Session()
        request_headers       = request.headers #Headers for the request
        num_chunks            = int(request_headers.get("Num-Chunks",1))
        model_id              = request_headers.get("Model-Id","model-0") #iris
        records_test_id       = request_headers.get("Records-Test-Id","matrix0data")
        records_test_filename = request_headers.get("Records-Test-Filename",records_test_id)
        extension             = request_headers.get("Extension","npy")
        experiment_id         = request_headers.get("Experiment-Id",uuid4().hex[:10])
        records_test_path     = "{}/{}.{}".format(SOURCE_PATH, records_test_filename, extension)
        _model_labels_shape   = request_headers.get("Model-Labels-Shape",-1)
        max_workers           = Utils.get_workers(num_chunks=num_chunks)
        
        if _model_labels_shape == -1:
            error ="Model-Labels-Shape header is required"
            logger.error(error)
            return Response(error, status=500) 
        
        local_read_start_time = time.time()
        records_test_ext            = "npy"
        records_test_result         = await RoryCommon.read_numpy_from(
            path      = records_test_path, 
            extension = records_test_ext)
        if records_test_result.is_err:
            return Response(status=500, response="Failed to local read the records")
        records_test = records_test_result.unwrap()
        
        local_read_entry = ExperimentLogEntry(
            event          = "LOCAL.READ",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = local_read_start_time,
            end_time       = time.time(),
            id             = model_id,
            worker_id      = "",
            num_chunks     = num_chunks,
        )
        logger.info(local_read_entry.model_dump())
     
        put_records_start_time    = time.time()
        maybe_records_test_chunks = Chunks.from_ndarray(
            ndarray      = records_test,
            group_id     = records_test_id,
            chunk_prefix = Some(records_test_id),
            num_chunks   = num_chunks,
        )

        if maybe_records_test_chunks.is_none:
            return Response(status=500,response="something went wrong creating the chunks")
        
        put_records_test_result = await RoryCommon.delete_and_put_chunks(
            client    = STORAGE_CLIENT,
            bucket_id = BUCKET_ID,
            key       = records_test_id,
            chunks    = maybe_records_test_chunks.unwrap(),
            timeout   = MICTLANX_TIMEOUT,
            tags      = {
                "full_shape": str(records_test.shape),
                "full_dtype": str(records_test.dtype)
            }
        )

        if put_records_test_result.is_err:
            return Response(status=500, response="Failed to put the records test")

        service_time_client_end = time.time()
        service_time_client = service_time_client_end - local_start_time
        
        put_records_entry = ExperimentLogEntry(
            event          = "PUT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = put_records_start_time,
            end_time       = time.time(),
            id             = model_id,
            worker_id      = "",
            num_chunks     = num_chunks,
        )
        logger.info(put_records_entry.model_dump())

        managerResponse:RoryManager = current_app.config.get("manager") # Communicates with the manager
        get_worker_start_time       = time.time()
        get_worker_result           = managerResponse.getWorker( #Gets the worker from the manager
            headers = {
                "Algorithm"             : algorithm,
                "Start-Request-Time"    : str(local_start_time),
                "Start-Get-Worker-Time" : str(get_worker_start_time),
                "Matrix-Id"             : model_id
            }
        )

        if get_worker_result.is_err:
            error = get_worker_result.unwrap_err()
            logger.error(str(error))
            return Response(response=str(error), status=500)
        
        (_worker_id,port) = get_worker_result.unwrap()

        get_worker_end_time     = time.time() 
        get_worker_service_time = get_worker_end_time - get_worker_start_time
        worker_id               =  "localhost" if TESTING else _worker_id

        get_worker_entry = ExperimentLogEntry(
            event          = "GET.WORKER",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_worker_start_time,
            end_time       = time.time(),
            id             = model_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
        )
        logger.info(get_worker_entry.model_dump())

        worker_start_time = time.time()
        worker            = RoryWorker( #Allows to establish the connection with the worker
            workerId  = worker_id,
            port      = port,
            session   = s,
            algorithm = algorithm,
        )

        workerResponse = worker.run(
            headers    = {
                "Records-Test-Id": records_test_id,
                "Model-Id": model_id,
                "Model-Labels-Shape":request_headers["Model-Labels-Shape"]
            },
            timeout = WORKER_TIMEOUT
        )
        workerResponse.raise_for_status()
        
        worker_end_time      = time.time()
        worker_response_time = worker_end_time - worker_start_time 
        jsonWorkerResponse   = workerResponse.json()
        endTime              = time.time() # Get the time when it ends
        worker_service_time  = jsonWorkerResponse["service_time"]
        label_vector         = jsonWorkerResponse["label_vector"]
        response_time        = endTime - local_start_time # Get the service time
        
        classification_completed_entry = ExperimentLogEntry(
            event          = "COMPLETED",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = local_start_time,
            end_time       = time.time(),
            id             = model_id,
            num_chunks     = num_chunks,
            workers        = max_workers,
            time           = response_time,
        )
        logger.info(classification_completed_entry.model_dump())

        return Response(
            response = json.dumps({
                "label_vector":label_vector,
                "worker_id":worker_id,
                "service_time_manager":get_worker_service_time,
                "service_time_worker":worker_response_time,
                "service_time_client":service_time_client,
                "service_time_predict":response_time,
                "algorithm":algorithm,
            }),
            status   = 200,
            headers  = {}
        )
    except Exception as e:
        logger.error("CLIENT_ERROR "+str(e))
        return Response(response = None, status = 500, headers={"Error-Message":str(e)})
    
@classification.route("/pqc/sknn/train", methods = ["POST"])
async def sknn_pqc_train():
    try:
        local_start_time             = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        STORAGE_CLIENT:AsyncClient   = current_app.config.get("ASYNC_STORAGE_CLIENT")
        _num_chunks                  = current_app.config.get("NUM_CHUNKS",4)
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        np_random                    = current_app.config.get("np_random")
        security_level               = current_app.config.get("LIU_SECURITY_LEVEL",128)
        if executor == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm             = Constants.ClassificationAlgorithms.SKNN_PQC_TRAIN
        s                     = Session()
        request_headers       = request.headers #Headers for the request
        model_id              = request_headers.get("Model-Id","matrix-0_model") #fertility-0_kjkk
        model_filename        = request_headers.get("Model-Filename",model_id)   #fertility_model
        model_labels_id       = "{}labels".format(model_id) #fertility_model_labels
        model_labels_filename = request_headers.get("Model-Labels-Filename",model_labels_id)   #fertility_model_labels     
        encrypted_model_id    = "encrypted{}".format(model_id) #encrypted-fertility_model
        extension             = request_headers.get("Extension","npy")
        num_chunks            = int(request_headers.get("Num-Chunks",_num_chunks))
        model_path            = "{}/{}.{}".format(SOURCE_PATH, model_filename, extension)
        model_labels_path     = "{}/{}.{}".format(SOURCE_PATH, model_labels_filename, extension)
        max_workers           = Utils.get_workers(num_chunks=num_chunks)
        experiment_id         = request_headers.get("Experiment-Id",uuid4().hex[:10])        
        _round             = bool(int(os.environ.get("_round","0"))) #False
        decimals           = int(os.environ.get("DECIMALS","2"))
        path               = os.environ.get("KEYS_PATH","/rory/keys")
        ctx_filename       = os.environ.get("CTX_FILENAME","ctx")
        pubkey_filename    = os.environ.get("PUBKEY_FILENAME","pubkey")
        secretkey_filename = os.environ.get("SECRET_KEY_FILENAME","secretkey")
        relinkey_filename  = os.environ.get("RELINKEY_FILENAME","relinkey")
        MICTLANX_TIMEOUT   = int(current_app.config.get("MICTLANX_TIMEOUT",3600))

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
        dataowner = DataOwnerPQC(scheme = ckks)

        model_path_exists        = os.path.exists(model_path) 
        model_path_labels_exists = os.path.exists(model_labels_path)
        if not model_path_exists or not model_path_labels_exists:
            return Response(response="Either model or label vector not found", status=500)
        else:

            read_local_model_start_time = time.time()
            model_result = await RoryCommon.read_numpy_from(
                path      = model_path,
                extension = "npy"
            )
            if model_result.is_err:
                return Response(status=500, response = "Failed to read the model")
            model = model_result.unwrap()
            
            local_read_entry = ExperimentLogEntry(
                event          = "LOCAL.READ",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = read_local_model_start_time,
                end_time       = time.time(),
                id             = model_id,
                worker_id      = "",
                num_chunks     = num_chunks,
                security_level = security_level,
                workers        = max_workers,
            )
            logger.info(local_read_entry.model_dump())                
          
            read_local_model_labels_start_time = time.time()
            model_labels_result = await RoryCommon.read_numpy_from(
                path      = model_labels_path,
                extension = "npy"
            )
            if model_labels_result.is_err:
                return Response(status= 500, response="Failed to read model labels")
            model_labels = model_labels_result.unwrap()
            model_labels = model_labels.reshape((1,model_labels.shape[0]))
            
            local_read_entry = ExperimentLogEntry(
                event          = "LOCAL.READ",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = read_local_model_labels_start_time,
                end_time       = time.time(),
                id             = model_id,
                worker_id      = "",
                num_chunks     = num_chunks,
                security_level = security_level,
                workers        = max_workers,
            )
            logger.info(local_read_entry.model_dump())   

            put_model_labels_start_time = time.time()
            maybe_model_labels_chunks = Chunks.from_ndarray(
                ndarray      = model_labels, 
                group_id     = model_labels_id, 
                chunk_prefix = Some(model_labels_id),
                num_chunks   = num_chunks
            )
            if maybe_model_labels_chunks.is_none:
                return Response(status=500, response="Failed to convert into chunks the model labels")
            
            ptm_result = await RoryCommon.delete_and_put_chunks(
                client    = STORAGE_CLIENT, 
                bucket_id = BUCKET_ID, 
                key       = model_labels_id,
                chunks    = maybe_model_labels_chunks.unwrap(),
                timeout   = MICTLANX_TIMEOUT,
                tags      = {
                    "full_shape":str(model_labels.shape), 
                    "full_dtype":str(model_labels.dtype)
                }
            )

            put_encrypted_ptm_entry = ExperimentLogEntry(
                event          = "PUT",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = put_model_labels_start_time,
                end_time       = time.time(),
                id             = model_id,
                worker_id      = "",
                num_chunks     = num_chunks,
                security_level = security_level,
                workers        = max_workers,
            )
            logger.info(put_encrypted_ptm_entry.model_dump())
            
            r:int = model.shape[0]
            a:int = model.shape[1]
            encrypted_model_shape = "({},{})".format(r,a)
            n = a*r

            segment_encrypt_model_start_time = time.time()
            encrypted_model_chunks = RoryCommon.segment_and_encrypt_ckks_with_executor_v2( #Encrypt 
                executor           = executor,
                key                = encrypted_model_id,
                plaintext_matrix   = model,
                n                  = n,
                num_chunks         = num_chunks,
                _round             = _round,
                decimals           = decimals,
                path               = path,
                ctx_filename       = ctx_filename,
                pubkey_filename    = pubkey_filename,
                secretkey_filename = secretkey_filename,
                relinkey_filename  = relinkey_filename,
            )
            
            segment_encrypt_entry = ExperimentLogEntry(
                event          = "SEGMENT.ENCRYPT",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = segment_encrypt_model_start_time,
                end_time       = time.time(),
                id             = model_id,
                worker_id      = "",
                num_chunks     = num_chunks,
                workers        = max_workers,
                security_level = security_level,
            )
            logger.info(segment_encrypt_entry.model_dump())

            put_chunked_start_time = time.time()            
            put_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
                client    = STORAGE_CLIENT,
                bucket_id = BUCKET_ID,
                key       = encrypted_model_id,
                chunks    = encrypted_model_chunks,
                timeout   = MICTLANX_TIMEOUT,
                tags      = {
                    "full_shape": str(encrypted_model_shape),
                    "full_dtype":"float64"
                }
            )
            if put_chunks_generator_results.is_err:
                return Response(status = 500, response="Failed to put encrypted model")

            put_encrypted_ptm_entry = ExperimentLogEntry(
                event          = "PUT",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = put_chunked_start_time,
                end_time       = time.time(),
                id             = model_id,
                worker_id      = "",
                num_chunks     = num_chunks,
                workers        = max_workers,
                security_level = security_level,
            )
            logger.info(put_encrypted_ptm_entry.model_dump())

            endTime       = time.time() # Get the time when it ends
            response_time = endTime - local_start_time # Get the service time

            classification_completed_entry = ExperimentLogEntry(
                event          = "COMPLETED",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = local_start_time,
                end_time       = time.time(),
                id             = model_id,
                num_chunks     = num_chunks,
                security_level = security_level,
                workers        = max_workers,
                time           = response_time
            )
            logger.info(classification_completed_entry.model_dump())

            return Response(
                response = json.dumps({
                    "response_time": str(response_time),
                    "encrypted_model_shape":str(encrypted_model_shape),
                    "encrypted_model_dtype":"float64",
                    "algorithm":algorithm,
                    "model_labels_shape":list(model_labels.shape)
                }),
                status  = 200,
            )

    except Exception as e:
        logger.error({
            "msg":str(e)
        })
        return Response(response = None, status = 500, headers={"Error-Message":str(e)})

@classification.route("/pqc/sknn/predict", methods = ["POST"])
async def sknn_pqc_predict():
    try:
        local_start_time             = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        TESTING                      = current_app.config.get("TESTING",True)
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        STORAGE_CLIENT:AsyncClient   = current_app.config.get("ASYNC_STORAGE_CLIENT")
        _num_chunks                  = current_app.config.get("NUM_CHUNKS",4)
        max_workers                  = current_app.config.get("MAX_WORKERS",2)
        np_random                    = current_app.config.get("np_random")
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        security_level               = current_app.config.get("LIU_SECURITY_LEVEL",128)
        WORKER_TIMEOUT               = int(current_app.config.get("WORKER_TIMEOUT",300))
        if executor == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm                 = Constants.ClassificationAlgorithms.SKNN_PQC_PREDICT
        s                         = Session()
        request_headers           = request.headers #Headers for the request
        num_chunks                = int(request_headers.get("Num-Chunks",_num_chunks))
        model_id                  = request_headers.get("Model-Id","model0") ##fertility-0
        model_filename            = request_headers.get("Model-Filename",model_id) #fertility_model
        records_test_id           = request_headers.get("Records-Test-Id","matrix0data") #fertility_data
        records_test_filename     = request_headers.get("Records-Test-Filename",records_test_id)
        records_test_extension    = request_headers.get("Records-Test-Extension","npy")
        encrypted_records_test_id = "encrypted{}".format(records_test_id) # The id of the encrypted matrix is built
        extension                 = request_headers.get("Extension","npy")
        model_labels_id           = "{}labels".format(model_id)
        _encrypted_model_shape    = request_headers.get("Encrypted-Model-Shape",-1)
        _encrypted_model_dtype    = request_headers.get("Encrypted-Model-Dtype",-1)
        experiment_id             = request_headers.get("Experiment-Id",uuid4().hex[:10])
        records_test_path         = "{}/{}.{}".format(SOURCE_PATH, records_test_filename, extension)
        max_workers               = Utils.get_workers(num_chunks=num_chunks)

        if _encrypted_model_dtype == -1:
            return Response("Encrypted-Model-Dtype", status=500)
        if _encrypted_model_shape == -1 :
            return Response("Encrypted-Model-Shape header is required", status=500)

        _round             = bool(int(os.environ.get("_round","0"))) #False
        decimals           = int(os.environ.get("DECIMALS","2"))
        path               = os.environ.get("KEYS_PATH","/rory/keys")
        ctx_filename       = os.environ.get("CTX_FILENAME","ctx")
        pubkey_filename    = os.environ.get("PUBKEY_FILENAME","pubkey")
        secretkey_filename = os.environ.get("SECRET_KEY_FILENAME","secretkey")
        relinkey_filename  = os.environ.get("RELINKEY_FILENAME","relinkey")

        MICTLANX_TIMEOUT        = int(current_app.config.get("MICTLANX_TIMEOUT",120))
        MICTLANX_DELAY          = int(os.environ.get("MICTLANX_DELAY","2"))
        MICTLANX_BACKOFF_FACTOR = float(os.environ.get("MICTLANX_BACKOFF_FACTOR","0.5"))
        MICTLANX_MAX_RETRIES    = int(os.environ.get("MICTLANX_MAX_RETRIES","10"))

        # _______________________________________________________________________________
        ckks = Ckks.from_pyfhel(
            _round   = _round,
            decimals = decimals,
            path               = path,
            ctx_filename       = ctx_filename,
            pubkey_filename    = pubkey_filename,
            secretkey_filename = secretkey_filename,
            relinkey_filename  = relinkey_filename,
        )
        # _______________________________________________________________________________
        dataowner = DataOwnerPQC(scheme = ckks)

        read_local_model_start_time = time.time()
        records_test_result   = await RoryCommon.read_numpy_from(
            path      = records_test_path, 
            extension = records_test_extension
        )
        if records_test_result.is_err:
            return Response(status =500, response ="Failed to read local records")
        records_test = records_test_result.unwrap()

        local_read_entry = ExperimentLogEntry(
            event          = "LOCAL.READ",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = read_local_model_start_time,
            end_time       = time.time(),
            id             = model_id,
            worker_id      = "",
            num_chunks     = num_chunks,
            security_level = security_level,
            workers        = max_workers,
        )
        logger.info(local_read_entry.model_dump())  

        r:int = records_test.shape[0]
        a:int = records_test.shape[1]
        n     = a*r

        segment_encrypt_start_time = time.time()
        encrypted_records_chunks = RoryCommon.segment_and_encrypt_ckks_with_executor_v2( #Encrypt 
            executor           = executor,
            key                = encrypted_records_test_id,
            plaintext_matrix   = records_test,
            n                  = n,
            num_chunks         = num_chunks,
            _round             = _round,
            decimals           = decimals,
            path               = path,
            ctx_filename       = ctx_filename,
            pubkey_filename    = pubkey_filename,
            secretkey_filename = secretkey_filename,
            relinkey_filename  = relinkey_filename,
        )
        
        segment_encrypt_entry = ExperimentLogEntry(
            event          = "SEGMENT.ENCRYPT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = segment_encrypt_start_time,
            end_time       = time.time(),
            id             = model_id,
            worker_id      = "",
            num_chunks     = num_chunks,
            workers        = max_workers,
            security_level = security_level,
        )
        logger.info(segment_encrypt_entry.model_dump())

        put_chunks_start_time        = time.time()
        encrypted_records_shape      = records_test.shape
        put_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
            client    = STORAGE_CLIENT,
            bucket_id = BUCKET_ID,
            key       = encrypted_records_test_id,
            chunks    = encrypted_records_chunks,
            timeout   = MICTLANX_TIMEOUT,
            tags      = {
                "full_shape": str(encrypted_records_shape),
                "full_dtype":"float64"
            }
        )
        if put_chunks_generator_results.is_err:
            return Response(status =500, response="Failed to put encrypted records")

        service_time_client = time.time() - local_start_time

        put_encrypted_ptm_entry = ExperimentLogEntry(
            event          = "PUT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = put_chunks_start_time,
            end_time       = time.time(),
            id             = model_id,
            worker_id      = "",
            num_chunks     = num_chunks,
            workers        = max_workers,
            security_level = security_level,
        )
        logger.info(put_encrypted_ptm_entry.model_dump())
        
        managerResponse:RoryManager = current_app.config.get("manager") # Communicates with the manager

        get_worker_start_time = time.time()
        get_worker_result     = managerResponse.getWorker( #Gets the worker from the manager
            headers = {
                "Algorithm"             : algorithm,
                "Start-Request-Time"    : str(local_start_time),
                "Start-Get-Worker-Time" : str(get_worker_start_time),
                "Matrix-Id"             : model_id
            }
        )

        if get_worker_result.is_err:
            error = get_worker_result.unwrap_err()
            logger.error(str(error))
            return Response(str(error), status=500)
        (_worker_id,port) = get_worker_result.unwrap()

        get_worker_end_time     = time.time() 
        get_worker_service_time = get_worker_end_time - get_worker_start_time
        worker_id               =  "localhost" if TESTING else _worker_id

        get_worker_entry = ExperimentLogEntry(
            event          = "GET.WORKER",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_worker_start_time,
            end_time       = time.time(),
            id             = model_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            workers        = max_workers,
            security_level = security_level,
        )
        logger.info(get_worker_entry.model_dump())

        worker_start_time = time.time()
        worker = RoryWorker( #Allows to establish the connection with the worker
            workerId   = worker_id,
            port       = port,
            session    = s,
            algorithm  = algorithm,
        )
        
        inner_interaction_arrival_time = time.time()
        encrypted_records_dtype = "float64"
        run1_headers = {
            "Step-Index"              : "1",
            "Records-Test-Id"         : records_test_id,
            "Model-Id"                : model_id,
            "Encrypted-Model-Shape"   : _encrypted_model_shape,
            "Encrypted-Model-Dtype"   : _encrypted_model_dtype,
            "Encrypted-Records-Shape" : str(encrypted_records_shape),
            "Encrypted-Records-Dtype" : str(encrypted_records_dtype),
            "Num-Chunks"              : str(num_chunks),
        }

        worker_run1_response = worker.run(
            headers = run1_headers,
            timeout = WORKER_TIMEOUT
        )
        worker_run1_response.raise_for_status()

        jsonWorkerResponse   = worker_run1_response.json()
        endTime              = time.time() # Get the time when it ends
        distances_id         = jsonWorkerResponse["distances_id"]
        distances_shape      = jsonWorkerResponse["distances_shape"]
        distances_dtype      = jsonWorkerResponse["distances_dtype"]
        worker_service_time  = jsonWorkerResponse["service_time"]
         
        run1_worker_entry = ExperimentLogEntry(
            event          = "RUN1",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = inner_interaction_arrival_time,
            end_time       = time.time(),
            id             = model_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            workers        = max_workers,
            security_level = security_level,
        )
        logger.info(run1_worker_entry.model_dump())
        
        get_all_distances_start_time = time.time()
        all_distances = await RoryCommon.get_pyctxt_matrix(
            client         = STORAGE_CLIENT, 
            bucket_id      = BUCKET_ID, 
            key            = distances_id, 
            ckks           = ckks,
            backoff_factor = MICTLANX_BACKOFF_FACTOR,
            delay          = MICTLANX_DELAY,
            force          = True,
            max_retries    = MICTLANX_MAX_RETRIES,
            timeout        = MICTLANX_TIMEOUT
        )
        
        get_encrypted_sm_entry = ExperimentLogEntry(
            event          = "GET",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_all_distances_start_time,
            end_time       = time.time(),
            id             = model_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            workers        = max_workers,
            security_level = security_level,
        )
        logger.info(get_encrypted_sm_entry.model_dump())
        
        decrypt_matrix_start_time = time.time()
        matrix_distances_plain    = ckks.decrypt_matrix_list(
            xs   = all_distances, 
            take = 1
        )
        _x = np.array(matrix_distances_plain).reshape(all_distances.shape)
        
        min_distances_index    = np.argmin(matrix_distances_plain,axis=1).reshape(1,-1)
        min_distances_index_id = "distancesindex{}".format(records_test_id)
        decrypt_entry = ExperimentLogEntry(
            event          = "DECRYPT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = decrypt_matrix_start_time,
            end_time       = time.time(),
            id             = model_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            workers        = max_workers,
            security_level = security_level,
        )
        logger.info(decrypt_entry.model_dump())

        t1 = time.time()
        maybe_min_distances_chunks = Chunks.from_ndarray(
            ndarray      = min_distances_index,
            group_id     = min_distances_index_id,
            chunk_prefix = Some(min_distances_index_id),
            num_chunks   = num_chunks,
        )

        if maybe_min_distances_chunks.is_none:
            raise "something went wrong creating the chunks"
        
        min_distances_put_result = await RoryCommon.delete_and_put_chunks(
            client    = STORAGE_CLIENT,
            bucket_id = BUCKET_ID,
            key       = min_distances_index_id,
            chunks    = maybe_min_distances_chunks.unwrap(),
            timeout   = MICTLANX_TIMEOUT,
            tags      = {
                "full_shape": str(min_distances_index.shape),
                "full_dtype": str(min_distances_index.dtype)
            }
        )
        if min_distances_put_result.is_err:
            return Response(status=500, response="Failed to put min distances")

        put_sm_entry = ExperimentLogEntry(
            event          = "PUT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = t1,
            end_time       = time.time(),
            id             = model_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            workers        = max_workers,
            security_level = security_level,
        )
        logger.info(put_sm_entry.model_dump())

        run2_headers = {
            "Step-Index"              : "2",
            "Records-Test-Id"         : records_test_id,
            "Model-Id"                : model_id,
            "Encrypted-Model-Shape"   : _encrypted_model_shape,
            "Encrypted-Model-Dtype"   : _encrypted_model_dtype,
            "Encrypted-Records-Shape" : str(encrypted_records_shape),
            "Encrypted-Records-Dtype" : str(encrypted_records_dtype),
            "Num-Chunks"              : str(num_chunks),
            "Min_Distances_Index_Id"  : min_distances_index_id
        }

        worker_run2_response = worker.run(
            headers = run2_headers,
            timeout = WORKER_TIMEOUT
        )
        worker_run2_response.raise_for_status()
        jsonWorkerResponse2  = worker_run2_response.json()
        service_time_worker  = worker_run2_response.headers.get("Service-Time",0)
        worker_end_time      = time.time()
        worker_response_time = worker_end_time - worker_start_time

        run2_worker_entry = ExperimentLogEntry(
            event          = "RUN2",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = inner_interaction_arrival_time,
            end_time       = time.time(),
            id             = model_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            workers        = max_workers,
            security_level = security_level,
        )
        logger.info(run2_worker_entry.model_dump())
        
        response_time = endTime - local_start_time # Get the service time

        classification_completed_entry = ExperimentLogEntry(
            event          = "COMPLETED",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = local_start_time,
            end_time       = time.time(),
            id             = model_id,
            num_chunks     = num_chunks,
            security_level = security_level,
            workers        = max_workers,
            time           = response_time
        )
        logger.info(classification_completed_entry.model_dump())
        
        label_vector = jsonWorkerResponse2["label_vector"]
        return Response(
            response = json.dumps({
                "label_vector":label_vector,
                "worker_id":worker_id,
                "service_time_manager":get_worker_service_time,
                "service_time_worker":worker_response_time,
                "service_time_client":service_time_client,
                "service_time_predict":response_time,
                "algorithm":algorithm,
            }),
            status   = 200,
            headers  = {}
        )

    except Exception as e:
        logger.error({
            "msg":str(e)
        })
        return Response(response = None, status = 500, headers={"Error-Message":str(e)})