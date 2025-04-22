import os
import pickle as PK
import time, json
import numpy as np
import numpy.typing as npt
from requests import Session
from flask import Blueprint,current_app,request,Response
from rory.core.interfaces.rorymanager import RoryManager
from rory.core.interfaces.roryworker import RoryWorker
from rory.core.security.dataowner import DataOwner
from rory.core.security.pqc.dataowner import DataOwner as DataOwnerPQC
from rory.core.security.cryptosystem.liu import Liu
from rory.core.utils.constants import Constants
from rorycommon import Common as RoryCommon
from mictlanx.v4.client import Client  as V4Client
from mictlanx.utils.segmentation import Chunks
from concurrent.futures import ProcessPoolExecutor
from utils.utils import Utils
from option import Some
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
        STORAGE_CLIENT:V4Client      = current_app.config.get("ASYNC_STORAGE_CLIENT")
        _num_chunks                  = current_app.config.get("NUM_CHUNKS",4)
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        np_random                    = current_app.config.get("np_random")
        securitylevel                = current_app.config.get("LIU_SECURITY_LEVEL",128)
        if executor == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm             = Constants.ClassificationAlgorithms.SKNN_TRAIN
        s                     = Session()
        request_headers       = request.headers #Headers for the request
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
        
        logger.debug({
            "event":"SKNN.TRAIN.STARTED",
            "algorithm":algorithm,
            "bucket_id":BUCKET_ID,
            "source_path":SOURCE_PATH,
            "num_chunks":num_chunks,
            "model_id":model_id,
            "model_labels_id":model_labels_id,
            "encrypted_model_id":encrypted_model_id,
            "model_filename":model_filename,
            "model_path":model_path,
            "model_labels_filename":model_labels_filename,
            "model_labels_path":model_labels_path,
            "extension":extension,
            "security_level":securitylevel,
            "m":int(m),
            "liu_round":liu.round
        })        

        model_path_exists        = os.path.exists(model_path) 
        model_path_labels_exists = os.path.exists(model_labels_path)
        if not model_path_exists or not model_path_labels_exists:
            return Response(response="Either model or label vector not found", status=500)
        else:
            
            model_result = await RoryCommon.read_numpy_from(
                path      =model_path,
                extension = "npy"
            )
            if model_result.is_err:
                return Response(status=500,response="Something went wrong reading the model")
            model = model_result.unwrap()
            read_local_model_labels_start_time = time.time()
            model_labels_result = await RoryCommon.read_numpy_from(
                extension="npy",
                path=model_labels_path
            )
            if model_labels_result.is_err:
                return Response(status=500,response="Something went wrong reading the model labels")
            model_labels = model_labels_result.unwrap()
            model_labels = model_labels.reshape((1,model_labels.shape[0]))
            # with open(model_labels_path, "rb") as f:
            #     model_labels:npt.NDArray = np.load(f)
            #     model_labels = model_labels.astype(np.int16)
            
            read_local_model_labels_st = time.time() - read_local_model_labels_start_time
            logger.info({
                "event":"READ.LOCAL",
                "model_id":model_id,
                "algorithm":algorithm,
                "model_labels_path":model_labels_path,
                "model_labels_filename":model_labels_filename,
                "service_time":read_local_model_labels_st
            })
            

            put_model_labels_start_time = time.time()
            maybe_models_labels_chunks = Chunks.from_ndarray(ndarray=model_labels, group_id=model_labels_id,num_chunks=1,chunk_prefix=Some(model_labels_id))
            if maybe_models_labels_chunks.is_none:
                return Response(status=500, response="Something went wrong generating the chunks of model labels")
            put_model_labels_result = await RoryCommon.delete_and_put_chunks(
                client         = STORAGE_CLIENT, 
                bucket_id      = BUCKET_ID, 
                key            = model_labels_id,
                chunks        = maybe_models_labels_chunks.unwrap(), 
                tags           = {
                    "full_dtype":str(model_labels.dtype),
                    "full_shape":str(model_labels.shape)
                }
            )
            put_model_labels_st = time.time() - put_model_labels_start_time
            logger.info({
                "event":"PUT.CHUNKS",
                "bucket_id":BUCKET_ID,
                "key":model_labels_id,
                "algorithm":algorithm,
                "shape":str(model_labels.shape),
                "dtype":str(model_labels.dtype),
                'ok':put_model_labels_result.is_ok,
                "service_time":put_model_labels_st
            })

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
            segment_encrypt_model_st = time.time() - segment_encrypt_model_start_time
            logger.info({
                "event":"SEGMENT.ENCRYPT.LIU",
                "model_id":model_id,
                "algorithm":algorithm,
                "encrypted_model_id":encrypted_model_id,
                "model_shape":str(model.shape),
                "model_dtype":str(model.dtype),
                "n":n,
                "num_chunks":num_chunks,
                "service_time":segment_encrypt_model_st
            })

            put_chunked_start_time = time.time()
            
         
            
            encrypted_model_put_chunks = await RoryCommon.delete_and_put_chunks(
            client        = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = encrypted_model_id,
            chunks         = encrypted_model_chunks,
            tags = {
                "full_shape": str(encrypted_model_shape),
                "full_dtype":"float64"
            }
        )

            put_chunked_st = time.time() - put_chunked_start_time
            logger.info({
                "event":"PUT.CHUNKED",
                "key":encrypted_model_id,
                "num_chunks":num_chunks,
                "algorithm":algorithm,
                "shape":encrypted_model_shape,
                "ok":encrypted_model_put_chunks.is_ok,
                "service_time":put_chunked_st
            })

            endTime       = time.time() # Get the time when it ends
            response_time = endTime - local_start_time # Get the service time

            logger.info({
                "event":"SKNN.TRAIN.COMPLETED",
                "model_id":model_id,
                "algorithm":algorithm,
                "response_time":response_time
            })
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
        STORAGE_CLIENT:V4Client      = current_app.config.get("ASYNC_STORAGE_CLIENT")
        _num_chunks                  = current_app.config.get("NUM_CHUNKS",4)
        max_workers                  = current_app.config.get("MAX_WORKERS",2)
        np_random:bool               = current_app.config.get("np_random",False)
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        WORKER_TIMEOUT               = int(current_app.config.get("WORKER_TIMEOUT",300))
        securitylevel                = current_app.config.get("LIU_SECURITY_LEVEL",128)
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

        records_test_path         = "{}/{}.{}".format(SOURCE_PATH, records_test_filename, extension)
        max_workers = Utils.get_workers(num_chunks=num_chunks)
        MICTLANX_TIMEOUT          = int(current_app.config.get("MICTLANX_TIMEOUT",120))
        backoff_factor = 1.5
        delay          = 1
        max_retries    = 10  
        if _encrypted_model_dtype == -1:
            return Response("Encrypted-Model-Dtype", status=500)
        if _encrypted_model_shape == -1 :
            return Response("Encrypted-Model-Shape header is required", status=500)
        if _model_labels_shape == -1:
            return Response("Model-Labels-Shape header is required", status=500)
        logger.debug({
            "event":"SKNN.1.PREDICT.STARTED",
            "bucket_id":BUCKET_ID,
            "testing":TESTING,
            "source_path":SOURCE_PATH, 
            "num_chunks":num_chunks,
            "max_workers":max_workers,
            "algorithm":algorithm,
            "model_id":model_id,
            "model_filename":model_filename,
            "records_test_id":records_test_id,
            "records_test_filename":records_test_filename,
            "extension":extension,
            "m":m,
            "encrypted_model_shape":_encrypted_model_shape,
            "encrypted_model_dtype":_encrypted_model_dtype,
            "records_test_path":records_test_path,
            "liu_round":liu.round,
            "security_level":securitylevel
        })        

        read_local_start_time = time.time()
        # Hay que parametrizar por envs
        records_test_ext = "npy"
        records_test_result = await RoryCommon.read_numpy_from(path=records_test_path, extension=records_test_ext)
        read_local_st = time.time() - read_local_start_time
        if records_test_result.is_err:
            return Response(status=500, response=f"Failed to read {records_test_path}")
        records_test = records_test_result.unwrap()

        logger.info({
            "event":"READ.LOCAL",
            "model_id":model_id,
            "records_path":records_test_path,
            "records_filename":model_id,
            "service_time":read_local_st,
            "algorithm":algorithm,
        })

        r:int = records_test.shape[0]
        a:int = records_test.shape[1]
        # Esto siempre se utiliza seria bueno colocarlo en una funcion de UTILS, unicamente del cliente por que es donde se utiliza(Ya lo hice). 
       
        n = a*r*int(m)

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
        encryption_service_time = time.time() - segment_encrypt_start_time
        logger.info({
            "event":"SEGMENT.ENCRYPT.LIU",
            "model_id":model_id,
            "records_id":encrypted_records_test_id,
            "records_shape":str(records_test.shape),
            "records_dtype":str(records_test.dtype),
            "algorithm":algorithm,
            "n":n,
            "num_chunks":num_chunks,
            "max_workers":max_workers,
            "service_time":encryption_service_time
        })

        put_chunks_start_time = time.time()
        encrypted_records_shape = (r,a,int(m))
        
        put_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
            client         = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = encrypted_records_test_id,
            chunks         = encrypted_records_chunks,
            tags = {
                "full_shape": str(encrypted_records_shape),
                "full_dtype":"float64"
            }
        )
        if put_chunks_generator_results.is_err:
            return Response(status=500, response="Failed to put encrypted records test in the storage")

        put_chunks_st = time.time() - put_chunks_start_time
        service_time_client = time.time() - local_start_time
        logger.info({
            "event":"PUT.CHUNKED",
            "model_id":model_id,
            "key":encrypted_records_test_id,
            "num_chunks":num_chunks,
            "algorithm":algorithm,
            "ok":put_chunks_generator_results.is_ok,
            "service_time":put_chunks_st
        })
        
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

        logger.info({
            "event":"MANAGER.GET.WORKER",
            "model_id":model_id,
            "worker_id":_worker_id,
            "port":port,
            "algorithm":algorithm,
            "service_time":get_worker_service_time,
            "m":m
        })
        worker_start_time = time.time()
        worker = RoryWorker( #Allows to establish the connection with the worker
            workerId   = worker_id,
            port       = port,
            session    = s,
            algorithm  = algorithm,
        )
        
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
            "Model-Labels-Shape"      : _model_labels_shape
        }

        worker_run1_response = worker.run(
            headers = run1_headers,
            timeout = WORKER_TIMEOUT
        )
        worker_run1_response.raise_for_status()

        # stringWorkerResponse = worker_run1_response.content.decode("utf-8") #Response from worker
        jsonWorkerResponse   = worker_run1_response.json()
        # json.loads(stringWorkerResponse) #pass to json
        endTime              = time.time() # Get the time when it ends
        distances_id         = jsonWorkerResponse["distances_id"]
        distances_shape      = jsonWorkerResponse["distances_shape"]
        distances_dtype      = jsonWorkerResponse["distances_dtype"]
        worker_service_time  = jsonWorkerResponse["service_time"]
         
        logger.info({
            "event":"WORKER.RUN.1",
            "model_id":model_id,
            "records_test_id":records_test_id,
            "encrypted_model_shape":_encrypted_model_shape,
            "encrypted_model_dtype":_encrypted_model_dtype,
            "encrypted_records_shape":str(encrypted_records_shape),
            "encrypted_records_dtype":str(encrypted_records_dtype),
            "num_chunks":num_chunks,
            "algorithm":algorithm,
        })

        get_all_distances_start_time = time.time()

        all_distances = await RoryCommon.get_and_merge(
            client         = STORAGE_CLIENT,
            key            = distances_id,
            bucket_id      = BUCKET_ID,
            max_retries    = max_retries,
            delay          = delay,
            backoff_factor = backoff_factor,
            timeout        = MICTLANX_TIMEOUT
        )
        
        get_all_distances_end_time     = time.time()
        get_all_distances_service_time = get_all_distances_end_time - get_all_distances_start_time
        logger.info({
            "event":"GET.NDARRAY",
            "model_id":model_id,
            "records_test_id":records_test_id,
            "distances_id":distances_id,
            "distances_shape":distances_shape,
            "distances_dtype":distances_dtype,
            "num_chunks":num_chunks,
            "service_time":get_all_distances_service_time,
            "algorithm":algorithm,
        }) 
        
        decrypt_matrix_start_time = time.time()
        matrix_distances_plain = liu.decryptMatrix(
            ciphertext_matrix = all_distances,
            secret_key        = dataowner.sk,
        )

        min_distances_index         = np.argmin(matrix_distances_plain.matrix,axis=1)
        min_distances_index_id      = "distancesindex{}".format(records_test_id)
        decrypt_matrix_end_time     = time.time()
        decrypt_matrix_service_time = decrypt_matrix_start_time - decrypt_matrix_end_time

        logger.info({
            "event":"DECRYPT.MIN",
            "model_id":model_id,
            "records_test_id":records_test_id,
            "service_time":decrypt_matrix_service_time,
            "algorithm":algorithm,
        })
        

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

        logger.info({
            "event":"WORKER.RUN.2",
            "model_id":model_id,
            "records_test_id":records_test_id,
            "encrypted_model_shape":_encrypted_model_shape,
            "encrypted_model_dtype":_encrypted_model_dtype,
            "encrypted_records_shape":str(encrypted_records_shape),
            "encrypted_records_dtype":str(encrypted_records_dtype),
            "num_chunks":num_chunks,
            "service_time":service_time_worker,
            "response_time": worker_response_time
        })
        
        response_time = endTime - local_start_time # Get the service time

        logger.info({
            "event":"SKNN.PREDICT.COMPLETED",
            "algorithm":algorithm,
            "model_id":model_id,
            "worker_service_time":worker_service_time,
            "worker_response_time":worker_response_time,
            "response_time":response_time
        })
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
    STORAGE_CLIENT:V4Client      = current_app.config.get("ASYNC_STORAGE_CLIENT")
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
    model_path            = "{}/{}.{}".format(SOURCE_PATH, model_filename, extension)
    model_labels_path     = "{}/{}.{}".format(SOURCE_PATH, model_labels_filename, extension)
    
    logger.debug({
        "event":"KNN.TRAIN.STARTED",
        "algorithm":algorithm,
        "model_id":model_id,
        "model_labels_id":model_labels_id,
        "extension":extension,
        "model_filename":model_filename,
        "model_labels_filename":model_labels_filename,
        "model_path":model_path,
        "model_labels_path":model_labels_path,
    })

    get_model_start_time = time.time()
    model_ext            = "npy"
    model_result         = await RoryCommon.read_numpy_from(
        path= model_path,
        extension=model_ext
    )

    if model_result.is_err:
        return Response(status=500, response="Failed to read model")

    model        = model_result.unwrap()
    get_model_st = time.time()- get_model_start_time

    logger.info({
        "event":"GET.MODEL.LOCAL",
        "path":model_path,
        "algorithm":algorithm,
        "model_id":model_id,
        "service_time":get_model_st,
    })
    
 
    get_model_labels_start_time = time.time()
    model_labels_ext            = "npy"

    model_labels_result         = await RoryCommon.read_numpy_from(
        path      = model_labels_path,
        extension = model_labels_ext
    )

    if model_labels_result.is_err:
        return Response(status=500, response="Failed to read model labels")
    
    model_labels        = model_labels_result.unwrap()
    model_labels        = model_labels.reshape(1,-1)
    get_model_labels_st = time.time()- get_model_labels_start_time

    logger.info({
        "event":"GET.MODEL.LABELS",
        "path":model_labels_path,
        "service_time":get_model_labels_st,
        "algorithm":algorithm,
        "model_id":model_id,
        "model_labels_id":model_labels_id,
        "model_labels_shape": str(model_labels.shape),
        "model_labels_dtype":str(model_labels.dtype)
    })


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

    put_model_st = time.time() - put_model_start_time
    logger.info({
        "event":"DELETE.AND.PUT.CHUNKED",
        "key":model_id,
        "algorithm":algorithm,
        "model_id":model_id,
        "bucket_id":BUCKET_ID,
        "shape":str(model.shape),
        "dtype":str(model.dtype),
        "ok":put_model_result.is_ok,
        "service_time":put_model_st
    })

    maybe_model_labels_chunks = Chunks.from_ndarray(
        ndarray      = model_labels,
        group_id     = model_labels_id,
        chunk_prefix = Some(model_labels_id),
        num_chunks   = num_chunks,
    )

    if maybe_model_labels_chunks.is_none:
        raise "something went wrong creating the chunks"

    logger.info({
        "event":"CHUNKS.FROM.NDARRAY",
        "key":model_labels_id,
        "algorithm":algorithm,
        "model_id":model_id,
        "bucket_id":BUCKET_ID,
        "shape":str(model_labels.shape),
        "dtype":str(model_labels.dtype),
    })



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

    put_model_labels_st = time.time() - put_model_start_time
    logger.info({
        "event":"DELETE.AND.PUT.CHUNKED",
        "key":model_labels_id,
        "algorithm":algorithm,
        "model_id":model_id,
        "bucket_id":BUCKET_ID,
        "shape":str(model_labels.shape),
        "dtype":str(model_labels.dtype),
        "service_time":put_model_labels_st
    })
    
    end_time      = time.time() # Get the time when it ends
    response_time = end_time - local_start_time # Get the service time
    logger.info({
        "event":"KNN.TRAIN.COMPLETED",
        "model_id":model_id,
        "algorithm":algorithm,
        "model_labels_id":model_labels_id,
        "service_time":response_time
    })

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
        STORAGE_CLIENT:V4Client      = current_app.config.get("ASYNC_STORAGE_CLIENT")
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        WORKER_TIMEOUT               = int(current_app.config.get("WORKER_TIMEOUT",300))
        if executor == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm             = Constants.ClassificationAlgorithms.KNN_PREDICT
        s                     = Session()
        request_headers       = request.headers #Headers for the request
        num_chunks                = int(request_headers.get("Num-Chunks",1))
        model_id              = request_headers.get("Model-Id","model-0") #iris
        records_test_id       = request_headers.get("Records-Test-Id","matrix0data")
        records_test_filename = request_headers.get("Records-Test-Filename",records_test_id)
        extension             = request_headers.get("Extension","npy")
        records_test_path     = "{}/{}.{}".format(SOURCE_PATH, records_test_filename, extension)
        _model_labels_shape     = request_headers.get("Model-Labels-Shape",-1)
        if _model_labels_shape == -1:
            error ="Model-Labels-Shape header is required"
            logger.error(error)
            return Response(error, status=500) 
        
        logger.debug({
            "event":"KNN.PREDICT.STARTED",
            "algorithm":algorithm, 
            "model_id":model_id,
            "records_test_id":records_test_id,
            "recors_test_filename":records_test_filename,
            "records_test_path":records_test_path,
            "extension":extension,
        })


        get_records_test_start_time = time.time()
        records_test_ext            = "npy"
        records_test_result         = await RoryCommon.read_numpy_from(path=records_test_path, extension=records_test_ext)
        get_recors_test_st          = time.time() -get_records_test_start_time
        if records_test_result.is_err:
            return Response(status=500, response="Failed to local read the records")
        records_test = records_test_result.unwrap()
        logger.info({
            "event":"GET.RECORDS",
            "path":records_test_path,
            "service_time":get_recors_test_st,
            "algorithm":algorithm,
            "model_id":model_id,
        })

     
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
            client         = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = records_test_id,
            chunks         = maybe_records_test_chunks.unwrap(),
            tags = {
                "full_shape": str(records_test.shape),
                "full_dtype": str(records_test.dtype)
            }
        )

        if put_records_test_result.is_err:
            return Response(status=500, response="Failed to put the records test")

        service_time_client_end = time.time()
        service_time_client = service_time_client_end - local_start_time
        put_records_st = time.time() - put_records_start_time
        logger.info({
            "event":"PUT.NDARRAY",
            "bucket_id":BUCKET_ID,
            "key":records_test_id,
            "algorithm":algorithm,
            "model_id":model_id,
            "shape":str(records_test.shape),
            "dtype":str(records_test.dtype),
            "ok":put_records_test_result.is_ok,
            "service_time":put_records_st
        })

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

        logger.info({
            "event":"MANAGER.GET.WORKER",
            "algorithm":algorithm,
            "model_id":model_id,
            "worker_id":_worker_id,
            "port":port,
            "service_time":get_worker_service_time,
        })

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
        logger.info({
            "event":"WORKER.PREDICT",
            "algorithm":algorithm,
            "model_id":model_id,
            "records_test_id":records_test_id,
            "service_time":worker_response_time,
            "worker_service_time":worker_service_time
        })

        logger.info({
            "event":"KNN.PREDICT.COMPLETED",
            "model_id":model_id,
            "algorithm":algorithm,
            "service_time_manager":get_worker_service_time,
            "service_time_worker":worker_response_time,
            "service_time_client":service_time_client,
            "service_time_predict":response_time
        })
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
        STORAGE_CLIENT:V4Client      = current_app.config.get("ASYNC_STORAGE_CLIENT")
        _num_chunks                  = current_app.config.get("NUM_CHUNKS",4)
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        np_random                    = current_app.config.get("np_random")
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
        
        _round   = False
        decimals = 2

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
        dataowner = DataOwnerPQC(scheme = ckks)  ##

        model_path_exists        = os.path.exists(model_path) 
        model_path_labels_exists = os.path.exists(model_labels_path)
        if not model_path_exists or not model_path_labels_exists:
            return Response(response="Either model or label vector not found", status=500)
        else:

            read_local_model_start_time = time.time()
            #     model:npt.NDArray = np.load(f)
            model_result = await RoryCommon.read_numpy_from(path=model_path,extension="npy")
            if model_result.is_err:
                return Response(status=500, response = "Failed to read the model")
            model = model_result.unwrap()
            read_local_model_st = time.time() - read_local_model_start_time
            
            logger.info({
                "event":"READ.LOCAL",
                "model_id":model_id,
                "algorithm":algorithm,
                "model_path":model_path,
                "service_time":read_local_model_st
            })
                
          
            read_local_model_labels_start_time = time.time()

            # with open(model_labels_path, "rb") as f:
            #     model_labels:npt.NDArray = np.load(f)
            #     model_labels = model_labels.astype(np.int16)
            model_labels_result = await RoryCommon.read_numpy_from(path = model_labels_path,extension="npy")
            if model_labels_result.is_err:
                return Response(status= 500, response="Failed to read model labels")
            model_labels = model_labels_result.unwrap()
            model_labels = model_labels.reshape((1,model_labels.shape[0]))
            read_local_model_labels_st = time.time() - read_local_model_labels_start_time
            logger.info({
                "event":"READ.LOCAL",
                "model_id":model_id,
                "algorithm":algorithm,
                "model_labels_path":model_labels_path,
                "model_labels_filename":model_labels_filename,
                "service_time":read_local_model_labels_st
            })
            maybe_model_labels_chunks = Chunks.from_ndarray(ndarray=model_labels, group_id=model_labels_id, chunk_prefix=Some(model_labels_id),num_chunks=num_chunks)
            if maybe_model_labels_chunks.is_none:
                return Response(status=500, response="Failed to convert into chunks the model labels")
            
            put_model_labels_start_time = time.time()
            ptm_result = await RoryCommon.delete_and_put_chunks(
                client = STORAGE_CLIENT, 
                bucket_id      = BUCKET_ID, 
                key            = model_labels_id,
                chunks       = maybe_model_labels_chunks.unwrap(), 
                tags           = {
                    "full_shape":str(model_labels.shape), 
                    "full_dtype":str(model_labels.dtype)
                }
            )
            put_model_labels_st = time.time() - put_model_labels_start_time

            logger.info({
                "event":"PUT",
                "bucket_id":BUCKET_ID,
                "key":model_labels_id,
                "response_time":put_model_labels_st
            })

            
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
                secretkey_filename = secretkey_filename
            )
            
            segment_encrypt_model_st = time.time() - segment_encrypt_model_start_time

            logger.info({
                "event":"SEGMENT.ENCRYPT.CKKS",
                "model_id":model_id,
                "algorithm":algorithm,
                "encrypted_model_id":encrypted_model_id,
                "model_shape":str(model.shape),
                "model_dtype":str(model.dtype),
                "n":n,
                "num_chunks":num_chunks,
                "service_time":segment_encrypt_model_st
            })


            put_chunked_start_time = time.time()

            
            put_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
                client = STORAGE_CLIENT,
                bucket_id      = BUCKET_ID,
                key            = encrypted_model_id,
                chunks         = encrypted_model_chunks,
                tags = {
                    "full_shape": str(encrypted_model_shape),
                    "full_dtype":"float64"
                }
            )
            if put_chunks_generator_results.is_err:
                return Response(status = 500, response="Failed to put encrypted model")

            put_chunked_st = time.time() - put_chunked_start_time
            logger.info({
                "event":"PUT.CHUNKED",
                "model_id":model_id,
                "key":encrypted_model_id,
                "num_chunks":num_chunks,
                "algorithm":algorithm,
                "service_time":put_chunked_st
            })

            endTime       = time.time() # Get the time when it ends
            response_time = endTime - local_start_time # Get the service time

            logger.info({
                "event":"SKNN.PQC.TRAIN.COMPLETED",
                "model_id":model_id,
                "algorithm":algorithm,
                "response_time":response_time
            })
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
        return Response(response = None, status = 500, headers={"Error-Message":str(e)})

@classification.route("/pqc/sknn/predict", methods = ["POST"])
async def sknn_pqc_predict():
    try:
        local_start_time             = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        TESTING                      = current_app.config.get("TESTING",True)
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        STORAGE_CLIENT:V4Client      = current_app.config.get("ASYNC_STORAGE_CLIENT")
        _num_chunks                  = current_app.config.get("NUM_CHUNKS",4)
        max_workers                  = current_app.config.get("MAX_WORKERS",2)
        np_random                    = current_app.config.get("np_random")
        executor:ProcessPoolExecutor = current_app.config.get("executor")
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
        records_test_extension = request_headers.get("Records-Test-Extension","npy")
        encrypted_records_test_id = "encrypted{}".format(records_test_id) # The id of the encrypted matrix is built
        extension                 = request_headers.get("Extension","npy")
        model_labels_id           = "{}labels".format(model_id)
        _encrypted_model_shape    = request_headers.get("Encrypted-Model-Shape",-1)
        _encrypted_model_dtype    = request_headers.get("Encrypted-Model-Dtype",-1)
        records_test_path         = "{}/{}.{}".format(SOURCE_PATH, records_test_filename, extension)
        backoff_factor=0.5
        delay=1
        max_retries=10
        timeout=120
        if _encrypted_model_dtype == -1:
            return Response("Encrypted-Model-Dtype", status=500)
        if _encrypted_model_shape == -1 :
            return Response("Encrypted-Model-Shape header is required", status=500)
    
        _round = False
        decimals = 2

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
        dataowner = DataOwnerPQC(scheme = ckks)  ##

        read_local_start_time = time.time()
        # with open(records_test_path, "rb") as f:
        #     records_test:npt.NDArray = np.load(f)    
        records_test_result = await RoryCommon.read_numpy_from(path=records_test_path, extension=records_test_extension)
        if records_test_result.is_err:
            return Response(status =500, response ="Failed to read local records")
        records_test = records_test_result.unwrap()
        read_local_st = time.time() - read_local_start_time

        logger.info({
            "event":"READ.LOCAL",
            "recrods_test_id":records_test_id,
            "path":records_test_path,
            "extension":records_test_extension,
            "service_time":read_local_st,
        })

        r:int = records_test.shape[0]
        a:int = records_test.shape[1]
        # cores = os.cpu_count()
        # max_workers = num_chunks if max_workers > num_chunks else max_workers
        # max_workers = cores if max_workers > cores else max_workers
        max_workers = Utils.get_workers(num_chunks=num_chunks)
        n = a*r

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
                secretkey_filename = secretkey_filename
        )
        # for c in encrypted_records_chunks:
            # print(c)
        # raise Exception("BOOM!")
        # print("SHAPE", records_test.shape)
        # raise Exception("BoOOM!")
        
        encryption_service_time = time.time() - segment_encrypt_start_time
        logger.info({
            "event":"SEGMENT.ENCRYPT.CKKS",
            "model_id":model_id,
            "records_id":encrypted_records_test_id,
            "records_shape":str(records_test.shape),
            "records_dtype":str(records_test.dtype),
            "algorithm":algorithm,
            "n":n,
            "num_chunks":num_chunks,
            "max_workers":max_workers,
            "service_time":encryption_service_time
        })

        put_chunks_start_time = time.time()
        encrypted_records_shape = records_test.shape

        
        put_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
            client = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = encrypted_records_test_id,
            chunks         = encrypted_records_chunks,
            tags = {
                "full_shape": str(encrypted_records_shape),
                "full_dtype":"float64"
            }
        )
        if put_chunks_generator_results.is_err:
            return Response(status =500, response="Failed to put encrypted records")

        put_chunks_st = time.time() - put_chunks_start_time
        service_time_client = time.time() - local_start_time
        logger.info({
            "event":"PUT.CHUNKED",
            "model_id":model_id,
            "key":encrypted_records_test_id,
            "num_chunks":num_chunks,
            "algorithm":algorithm,
            "service_time":put_chunks_st
        })
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

        logger.info({
            "event":"MANAGER.GET.WORKER",
            "worker_id":_worker_id,
            "service_time":get_worker_service_time,
        })
        worker_start_time = time.time()
        worker = RoryWorker( #Allows to establish the connection with the worker
            workerId   = worker_id,
            port       = port,
            session    = s,
            algorithm  = algorithm,
        )
        
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

        # stringWorkerResponse = worker_run1_response.content.decode("utf-8") #Response from worker
        jsonWorkerResponse   = worker_run1_response.json()
        # json.loads(stringWorkerResponse) #pass to json
        endTime              = time.time() # Get the time when it ends
        distances_id         = jsonWorkerResponse["distances_id"]
        distances_shape      = jsonWorkerResponse["distances_shape"]
        distances_dtype      = jsonWorkerResponse["distances_dtype"]
        worker_service_time  = jsonWorkerResponse["service_time"]
         
        logger.info({
            "event":"WORKER.RUN.1",
            "model_id":model_id,
            "records_test_id":records_test_id,
            "encrypted_model_shape":_encrypted_model_shape,
            "encrypted_model_dtype":_encrypted_model_dtype,
            "encrypted_records_shape":str(encrypted_records_shape),
            "encrypted_records_dtype":str(encrypted_records_dtype),
            "num_chunks":num_chunks,
            "algorithm":algorithm,
        })
        # raise Exception("BOOM!")
        get_all_distances_start_time = time.time()

        print("BEGFORE")
        all_distances = await RoryCommon.get_pyctxt_matrix(
            client = STORAGE_CLIENT, 
            bucket_id      = BUCKET_ID, 
            key            = distances_id, 
            ckks           = ckks,
            backoff_factor=backoff_factor,
            delay=delay,
            force=True,
            max_retries=max_retries,
            timeout=timeout
            # backoff_factor=back
        )
        print("ALL_DISTANCES", all_distances.shape)
        get_all_distances_end_time     = time.time()
        get_all_distances_service_time = get_all_distances_end_time - get_all_distances_start_time
        
        logger.info({
            "event":"GET",
            "bucket_id":BUCKET_ID,
            "key":distances_id,
            "algorithm":algorithm,
            "model_id":model_id,
            "response_time":get_all_distances_service_time,
        }) 
        decrypt_matrix_start_time = time.time()
        matrix_distances_plain = ckks.decrypt_matrix_list(
            xs   = all_distances, 
            take = 1
        )
        _x = np.array(matrix_distances_plain).reshape(all_distances.shape)
        print(_x)
        print(_x.shape)
        # _x = list(map(lambda x: x, matrix_distances_plain))

        min_distances_index         = np.argmin(matrix_distances_plain,axis=1).reshape(1,-1)
        min_distances_index_id      = "distancesindex{}".format(records_test_id)
        decrypt_matrix_end_time     = time.time()
        decrypt_matrix_service_time = decrypt_matrix_start_time - decrypt_matrix_end_time

        logger.info({
            "event":"DECRYPT.MIN",
            "model_id":model_id,
            "algorithm":algorithm,
            "service_time":decrypt_matrix_service_time,
        })
        # print("MIN_DISTANCE",min_distances_index, min_distances_index.shape)
        # raise Exception("BOOM!")
        # raise Exception("BOOM!")

        maybe_min_distances_chunks = Chunks.from_ndarray(
            ndarray      = min_distances_index,
            group_id     = min_distances_index_id,
            chunk_prefix = Some(min_distances_index_id),
            num_chunks   = num_chunks,
        )

        if maybe_min_distances_chunks.is_none:
            raise "something went wrong creating the chunks"
        
        t1 = time.time()
        min_distances_put_result = await RoryCommon.delete_and_put_chunks(
            client = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = min_distances_index_id,
            chunks         = maybe_min_distances_chunks.unwrap(),
            tags = {
                "full_shape": str(min_distances_index.shape),
                "full_dtype": str(min_distances_index.dtype)
            }
        )
        if min_distances_put_result.is_err:
            return Response(status=500, response="Failed to put min distances")

        logger.info({
            "event":"PUT.NDARRAY",
            "bucket_id":BUCKET_ID,
            "key":min_distances_index_id,
            "model_id":model_id,
            "algorithm":algorithm,
            "response_time": time.time() - t1
        })

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
        # stringWorkerResponse2 = 
        # worker_run2_response.content.decode("utf-8") #Response from worker
        jsonWorkerResponse2   = worker_run2_response.json()
        # json.loads(stringWorkerResponse2) #pass to json
        service_time_worker   = worker_run2_response.headers.get("Service-Time",0)
        worker_end_time       = time.time()
        worker_response_time  = worker_end_time - worker_start_time

        logger.info({
            "event":"WORKER.RUN.2",
            "model_id":model_id,
            "records_test_id":records_test_id,
            "encrypted_model_shape":_encrypted_model_shape,
            "encrypted_model_dtype":_encrypted_model_dtype,
            "encrypted_records_shape":str(encrypted_records_shape),
            "encrypted_records_dtype":str(encrypted_records_dtype),
            "num_chunks":num_chunks,
            "service_time":service_time_worker,
            "response_time": worker_response_time
        })
        
        response_time = endTime - local_start_time # Get the service time

        logger.info({
            "event":"SKNN.PQC.PREDICT.COMPLETED",
            "algorithm":algorithm,
            "model_id":model_id,
            "worker_service_time":worker_service_time,
            "worker_response_time":worker_response_time,
            "response_time":response_time
        })
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