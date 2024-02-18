import os
import time, json
import numpy as np
import numpy.typing as npt
from requests import Session
from flask import Blueprint,current_app,request,Response
from rory.core.interfaces.rorymanager import RoryManager
from rory.core.interfaces.roryworker import RoryWorker
from rory.core.security.dataowner import DataOwner
from rory.core.security.cryptosystem.liu import Liu
from rory.core.utils.constants import Constants
from rory.core.interfaces.logger_metrics import LoggerMetrics
from mictlanx.v4.client import Client  as V4Client
from mictlanx.utils.segmentation import Chunks
from concurrent.futures import ProcessPoolExecutor
from utils.utils import Utils

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
def sknn_train():
    try:
        local_start_time                  = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        # TESTING                      = current_app.config.get("TESTING",True)
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        liu:Liu                      = current_app.config.get("liu")
        dataowner:DataOwner          = current_app.config.get("dataowner")
        STORAGE_CLIENT:V4Client      = current_app.config.get("STORAGE_CLIENT")
        num_chunks                   = current_app.config.get("NUM_CHUNKS",4)
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        # logger.debug("INIT_TRAIN")
        if executor == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm          = Constants.ClassificationAlgorithms.SKNN_TRAIN
        s                  = Session()
        request_headers     = request.headers #Headers for the request
        model_id           = request_headers.get("Model-Id","matrix-0_model")
        model_filename     = request_headers.get("Model-Filename",model_id)        
        model_labels_id    = "{}_labels".format(model_id)
        model_labels_filename     = request_headers.get("Model-Labels-Filename",model_labels_id)        
        encrypted_model_id = "encrypted-{}".format(model_id) #encrypted-iris_model
        
        extension          = request_headers.get("Extension","npy")
        m                  = request_headers.get("M","3")
        model_path         = "{}/{}.{}".format(SOURCE_PATH, model_filename, extension)
        model_labels_path  = "{}/{}.{}".format(SOURCE_PATH, model_labels_filename, extension)
        # logger.debug("SKNN TRAIN algorithm={}, m={}, model_id={}, model_labels_id={}, encrypted_model_id={}".format(algorithm, m, model_id, model_labels_id, encrypted_model_id))

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
            "m":m,
        })        



        model_path_exists        = os.path.exists(model_path) 
        model_path_labels_exists = os.path.exists(model_labels_path)
        logger.debug("model_path_exists={}, MPLEAA={} Ext={}".format(model_path_exists,model_path_labels_exists,extension))
        if not model_path_exists or not model_path_labels_exists:
            return Response(response="Either model or label vector not found", status=500)
        else:
            
            logger.debug({
                "event":"READ.LOCAL.BEFORE",
                "model_path":model_path
            })
            read_local_model_start_time = time.time()
            with open(model_path, "rb") as f:
                model:npt.NDArray = np.load(f)
            read_local_model_st = time.time() - read_local_model_start_time
            logger.info({
                "event":"READ.LOCAL",
                "model_path":model_path,
                "service_time":read_local_model_st
            })
                
            logger.debug({
                "event":"READ.LOCAL.BEFORE",
                "model_labels_path":model_labels_path
            })
            read_local_model_labels_start_time = time.time()
            with open(model_labels_path, "rb") as f:
                model_labels:npt.NDArray = np.load(f)
                model_labels = model_labels.astype(np.int16)
            
            read_local_model_labels_st = time.time() - read_local_model_start_time
            logger.info({
                "event":"READ.LOCAL",
                "model_labels_path":model_labels_path,
                "service_time":read_local_model_labels_st
            })
            
            logger.debug({
                "event":"PUT.NDARRAY.BEFORE",
                "key":model_labels_id,
                "bucket_id":BUCKET_ID,
                "shape":str(model_labels.shape),
                "dtype":str(model_labels.dtype)
            })
            put_model_labels_start_time = time.time()
            X = STORAGE_CLIENT.put_ndarray(
                key       = model_labels_id,
                ndarray   = model_labels,
                tags      = {},
                bucket_id = BUCKET_ID
            ).result()
            put_model_labels_st = time.time() - put_model_labels_start_time
            logger.info({
                "event":"PUT.NDARRAY",
                "key":model_labels_id,
                "bucket_id":BUCKET_ID,
                "shape":str(model_labels.shape),
                "dtype":str(model_labels.dtype),
                "service_time":put_model_labels_st
            })
            # encryption_start_time = time.time()
            r                     = model.shape[0]
            a                     = model.shape[1]
            encrypted_model_shape = "({},{},{})".format(r,a,m)

            n =  a*r*m
            logger.debug({
                "event":"SEGMENT.ENCRYPT.LIU.BEFORE",
                "encrypted_model_id":encrypted_model_id,
                "model_shape":str(model.shape),
                "model_dtype":str(model.dtype),
                "n":n,
                "num_chunks":num_chunks,
            })
            segment_encrypt_model_start_time = time.time()
            encrypted_model_chunks:Chunks = Utils.segment_and_encrypt_liu_with_executor( #Encrypt 
                executor         = executor,
                key              = encrypted_model_id,
                plaintext_matrix = model,
                dataowner        = dataowner,
                n                  =n,
                num_chunks       = num_chunks
            )

            segment_encrypt_model_st = time.time() - segment_encrypt_model_start_time
            # chunks = encrypted_model_chunks.iter()
            logger.info({
                "event":"SEGMENT.ENCRYPT.LIU",
                "encrypted_model_id":encrypted_model_id,
                "model_shape":str(model.shape),
                "model_dtype":str(model.dtype),
                "n":n,
                "num_chunks":num_chunks,
                "service_time":segment_encrypt_model_st
            })

            # logger.debug("{} {} {}".format(type(encrypted_model_id), encrypted_model_chunks,type(BUCKET_ID)))

            logger.debug({
                "event":"PUT.CHUNKS.BEFORE",
                "algorithm":algorithm,
                "key":encrypted_model_id,
                "num_chunks":num_chunks
            })
            put_chunks_start_time = time.time()
            STORAGE_CLIENT.delete_by_ball_id(
                ball_id=encrypted_model_id, 
                bucket_id=BUCKET_ID
            )
            put_chunks_generator_results = STORAGE_CLIENT.put_chunks(
                key       = encrypted_model_id, 
                chunks    = encrypted_model_chunks, 
                bucket_id = BUCKET_ID,
                tags      = {}
            )
            put_chunks_st = time.time() - put_chunks_start_time
            logger.info({
                "event":"PUT.CHUNKS",
                "key":encrypted_model_id,
                "num_chunks":num_chunks,
                "algorithm":algorithm,
                "service_time":put_chunks_st
            })

            for i,put_chunk_result in enumerate(put_chunks_generator_results):
                # encryption_end_time    = time.time()
                if put_chunk_result.is_err:
                    logger.error({
                        "msg":str(put_chunk_result.unwrap_err())
                    })
                    return Response(
                        status   = 500,
                        response = "{}".format(str(put_chunk_result.unwrap_err()))
                    )
                # logger.info(str(encrypt_logger_metrics)+","+str(i))


            segment_encrypt_model_st = time.time() - segment_encrypt_model_start_time
            logger.info({
                "event":"PUT.SEGMENT.ENCRYPT", 
                "algorithm":algorithm,
                "model_id":model_id,
                "encrypted_model_id":encrypted_model_id,
                "service_time":segment_encrypt_model_st
            })
            endTime        = time.time() # Get the time when it ends
            response_time  = endTime - local_start_time # Get the service time
            logger.info({
                "event":"SKNN.TRAIN.COMPLETED",
                "algorithm":algorithm,
                
                "response_time":response_time
            })
            return Response(
                response = json.dumps({
                    "response_time": response_time,
                    "encrypted_model_shape":str(encrypted_model_shape),
                    "encrypted_model_dtype":"float64",
                    "algorithm"   : algorithm,
                }),
                status  = 200,
                # headers = {
                #     "Encrypted-Model-Shape":encrypted_model_shape,
                #     "Encrypted-Model-Dtype":"float64"
                # }
            )
    except Exception as e:
        logger.error({
            "msg":str(e)
        })
        return Response(response = None, status = 500, headers={"Error-Message":str(e)})


@classification.route("/sknn/predict",methods = ["POST"])
def sknn_predict():
    try:
        local_start_time                  = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        TESTING                      = current_app.config.get("TESTING",True)
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        liu:Liu                      = current_app.config.get("liu")
        dataowner:DataOwner          = current_app.config.get("dataowner")
        STORAGE_CLIENT:V4Client      = current_app.config.get("STORAGE_CLIENT")
        _num_chunks                   = current_app.config.get("NUM_CHUNKS",4)
        max_workers                  = current_app.config.get("MAX_WORKERS",2)
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        WORKER_TIMEOUT               = int(current_app.config.get("WORKER_TIMEOUT",300))
        if executor == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm                 = Constants.ClassificationAlgorithms.SKNN_PREDICT
        s                         = Session()
        request_headers            = request.headers #Headers for the request
        num_chunks                = int(request_headers.get("Num-Chunks",_num_chunks))
        model_id                  = request_headers.get("Model-Id","model-0") #iris
        records_test_id           = request_headers.get("Records-Test-Id","matrix-0_data")
        records_test_filename     = request_headers.get("Records-Test-Filename",records_test_id)
        encrypted_records_test_id = "encrypted-{}".format(records_test_id) # The id of the encrypted matrix is built
        extension                 = request_headers.get("Extension","npy")
        m                         = request_headers.get("M","3")
        _encrypted_model_shape    = request_headers.get("Encrypted-Model-Shape",-1)
        _encrypted_model_dtype    = request_headers.get("Encrypted-Model-Dtype",-1)
        records_test_path         = "{}/{}.{}".format(SOURCE_PATH, records_test_filename, extension)
        # logger.debug("SKNN PREDICT algorithm={}, m={}, model_id={}, records_test_id={}".format(algorithm, m, model_id, records_test_id))
        
        if _encrypted_model_dtype == -1:
            return Response("Encrypted-Model-Dtype", status=500)
        if _encrypted_model_shape == -1 :
            return Response("Encrypted-Model-Shape header is required", status=500)
    

        logger.debug({
            "event":"SKNN.PREDICT.STARTED",
            "bucket_id":BUCKET_ID,
            "testing":TESTING,
            "source_path":SOURCE_PATH, 
            "num_chunks":num_chunks,
            "max_workers":max_workers,
            "algorithm":algorithm,
            "model_id":model_id,
            "records_test_id":records_test_id,
            "records_test_filename":records_test_filename,
            "extension":extension,
            "m":m,
            "encrypted_model_shape":_encrypted_model_shape,
            "encrypted_model_dtype":_encrypted_model_dtype,
            "records_test_path":records_test_path,

        })        


        logger.debug({
            "event":"READ.LOCAL.BEFORE",
            "path":records_test_path
        })
        read_local_start_time = time.time()
        with open(records_test_path, "rb") as f:
            records_test:npt.NDArray = np.load(f)    
        read_local_st = time.time() - read_local_start_time
        logger.info({
            "event":"READ.LOCAL",
            "path":records_test_path,
            "service_time":read_local_st
        })
        r           = records_test.shape[0]
        a           = records_test.shape[1]
        cores       = os.cpu_count()
        max_workers = num_chunks if max_workers > num_chunks else max_workers
        max_workers = cores if max_workers > cores else max_workers
        n = a*r*m
        logger.debug({
            "event":"SEGMENT.ENCRYPT.LIU.BEFORE",
            "key":encrypted_records_test_id,
            "plaintext_matrix_shape":str(records_test.shape),
            "plaintext_matrix_dtype":str(records_test.dtype),
            "algorithm":algorithm,
            "n":n,
            "num_chunks":num_chunks,
            "max_workers":max_workers,
        })

        segment_encrypt_start_time = time.time()
        encrypted_records_chunks:Chunks = Utils.segment_and_encrypt_liu_with_executor( #Encrypt 
            executor         = executor,
            key              = encrypted_records_test_id,
            plaintext_matrix = records_test,
            dataowner        = dataowner,
            n                = n,
            num_chunks       = num_chunks
        )
        encryption_service_time = time.time() - segment_encrypt_start_time
        logger.info({
            "event":"SEGMENT.ENCRYPT.LIU",
            "key":encrypted_records_test_id,
            "plaintext_matrix_shape":str(records_test.shape),
            "plaintext_matrix_dtype":str(records_test.dtype),
            "algorithm":algorithm,
            "n":n,
            "num_chunks":num_chunks,
            "max_workers":max_workers,
            "service_time":encryption_service_time
        })

        


        logger.debug({
            "event":"PUT.CHUNKS.BEFORE",
            "key":encrypted_records_test_id,
            "algorithm":algorithm,
            "num_chunks":num_chunks
        })
        put_chunks_start_time = time.time()
        STORAGE_CLIENT.delete_by_ball_id(
            ball_id=encrypted_records_test_id, 
            bucket_id=BUCKET_ID
        )
        put_chunks_generator_results = STORAGE_CLIENT.put_chunks(
            key       = encrypted_records_test_id, 
            chunks    = encrypted_records_chunks, 
            bucket_id = BUCKET_ID,
            tags      = {}
        )

        put_chunks_st = time.time() - put_chunks_start_time
        logger.info({
            "event":"PUT.CHUNKS",
            "key":encrypted_records_test_id,
            "num_chunks":num_chunks,
            "algorithm":algorithm,
            "service_time":put_chunks_st
        })

        for i,put_chunk_result in enumerate(put_chunks_generator_results):
            encryption_end_time    = time.time()
            if put_chunk_result.is_err:
                logger.error({
                    "msg":str(put_chunk_result.unwrap_err())
                })
                return Response(
                    status   = 500,
                    response = "{}".format(str(put_chunk_result.unwrap_err()))
                )

        put_segment_encrypt_st        = time.time() - segment_encrypt_start_time

        logger.info({
            "event":"PUT.SEGMENT.ENCRYPT", 
            "records_test_id":records_test_id,
            "algorithm":algorithm,
            "service_time":put_segment_encrypt_st
        })
        managerResponse:RoryManager = current_app.config.get("manager") # Communicates with the manager

        get_worker_start_time = time.time()
        get_worker_result          = managerResponse.getWorker( #Gets the worker from the manager
            headers = {
                "Algorithm"             : algorithm,
                "Start-Request-Time"    : str(local_start_time),
                "Start-Get-Worker-Time" : str(get_worker_start_time) 
            }
        )

        if get_worker_result.is_err:
            error = get_worker_result.unwrap_err()
            logger.error(str(error))
            return Response(str(error), status=500)
        (_worker_id,port) = get_worker_result.unwrap()

        get_worker_end_time         = time.time() 
        get_worker_service_time     = get_worker_end_time - get_worker_start_time
        worker_id                    =  "localhost" if TESTING else _worker_id

        logger.info({
            "event":"MANAGER.GET.WORKER",
            "worker_id":_worker_id,
            "port":port,
            "algorithm":algorithm,
            "service_time":get_worker_service_time,
            # "k":k,
            "m":m
        })

        worker         = RoryWorker( #Allows to establish the connection with the worker
            workerId   = worker_id,
            port       = port,
            session    = s,
            algorithm  = algorithm,
        )
        worker_start_time = time.time()
        # logger.debug("RORY WORKER SUCCESSFULLY")
        encrypted_records_shape = (r,a,int(m))
        encrypted_records_dtype = "float64"
        run_headers = {
            "Records-Test-Id"         : records_test_id,
            "Model-Id"                : model_id,
            "Encrypted-Model-Shape"   : _encrypted_model_shape,
            "Encrypted-Model-Dtype"   : _encrypted_model_dtype,

            "Encrypted-Records-Shape" : str(encrypted_records_shape),
            "Encrypted-Records-Dtype" : str(encrypted_records_dtype),
            "Num-Chunks"              : str(num_chunks),

        }
        logger.debug({
            "event":"RUN.1.BEFORE",
            "model_id":model_id,
            "records_test_id":records_test_id,
            "encrypted_model_shape":_encrypted_model_shape,
            "encrypted_model_dtype":_encrypted_model_dtype,

            "encrypted_records_shape":str(encrypted_records_shape),
            "encrypted_records_dtype":str(encrypted_records_dtype),
            "num_chunks":num_chunks
        })
        worker_response = worker.run(
            headers    = run_headers,
            timeout = WORKER_TIMEOUT
        )
        worker_response.raise_for_status()

        stringWorkerResponse = worker_response.content.decode("utf-8") #Response from worker
        jsonWorkerResponse   = json.loads(stringWorkerResponse) #pass to json
        endTime              = time.time() # Get the time when it ends
        worker_service_time  = jsonWorkerResponse["service_time"]
        worker_end_time       = time.time()
        worker_response_time   = worker_end_time - worker_start_time 
        logger.info({
            "event":"RUN.1",
            "model_id":model_id,
            "records_test_id":records_test_id,
            "encrypted_model_shape":_encrypted_model_shape,
            "encrypted_model_dtype":_encrypted_model_dtype,

            "encrypted_records_shape":str(encrypted_records_shape),
            "encrypted_records_dtype":str(encrypted_records_dtype),
            "num_chunks":num_chunks,
            "service_time":worker_service_time,
            "response_time": worker_response_time
        })
        # worker_response.headers.get("Service-Time",0) # Extract the time at which it started]]
        response_time        = endTime - local_start_time # Get the service time
        logger.info({
            "event":"SKNN.PREDICT.COMPLETED",
            "algorithm":algorithm,
            "worker_service_time":worker_service_time,
            "worker_response_time":worker_response_time,
            "response_time":response_time
        })
        label_vector = jsonWorkerResponse["label_vector"]
        return Response(
            response = json.dumps({
                "label_vector" : label_vector,
                "worker_service_time" : worker_response_time,
                "worker_response_time":worker_response_time,
                "response_time": response_time,
                "algorithm"   : algorithm,
            }),
            status   = 200,
            headers  = {}
        )
    except Exception as e:
        logger.error({
            "msg":str(e)
        })
        # logger.error("CLIENT_ERROR "+str(e))
        return Response(response = None, status = 500, headers={"Error-Message":str(e)})


@classification.route("/knn/train", methods = ["POST"])
def knn_train():
    local_start_time                  = time.time()
    logger                       = current_app.config["logger"]
    BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
    # TESTING                      = current_app.config.get("TESTING",True)
    SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
    STORAGE_CLIENT:V4Client      = current_app.config.get("STORAGE_CLIENT")
    executor:ProcessPoolExecutor = current_app.config.get("executor")
    # WORKER_TIMEOUT               = int(current_app.config.get("WORKER_TIMEOUT",300))
    if executor == None:
        raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
    algorithm         = Constants.ClassificationAlgorithms.KNN_TRAIN
    s                 = Session()
    request_headers    = request.headers #Headers for the request
    model_id          = request_headers.get("Model-Id","matrix-0_model")        
    model_filename    = request_headers.get("Model-Filename",model_id)        
    
    model_labels_id   = "{}_labels".format(model_id)
    model_labels_filename    = request_headers.get("Model-Labels-Filename",model_labels_id)        

    extension         = request_headers.get("Extension","npy")
    model_path        = "{}/{}.{}".format(SOURCE_PATH, model_filename, extension)
    model_labels_path = "{}/{}.{}".format(SOURCE_PATH, model_labels_filename, extension)

    # logger.debug("_"*50)
    # logger.debug("Client starts to process {} at {}".format(model_id,local_start_time))
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

    logger.debug({
        "event":"GET.MODEL.BEFORE",
        "path":model_path,
    })
    get_model_start_time = time.time()
    with open(model_path, "rb") as f:
        model:npt.NDArray = np.load(f)
    get_model_st = time.time()- get_model_start_time
    logger.info({
        "event":"GET.MODEL.BEFORE",
        "path":model_path,
        "service_time":get_model_st
    })
    
    logger.debug({
        "event":"GET.MODEL.LABELS.BEFORE",
        "path":model_labels_path,
    })
    get_model_labels_start_time = time.time()
    with open(model_labels_path, "rb") as f:
        model_labels:npt.NDArray = np.load(f)
        model_labels             = model_labels.astype(np.int16)

    get_model_labels_st = time.time()- get_model_start_time
    logger.info({
        "event":"GET.MODEL.LABELS.BEFORE",
        "path":model_labels_path,
        "service_time":get_model_labels_st
    })
    # logger.debug("OPEN MODEL_LABELS SUCCESSFULLY")

    logger.debug({
        "event":"PUT.NDARRAY.BEFORE",
        "key":model_id,
        "bucket_id":BUCKET_ID,
        "shape":str(model.shape),
        "dtype":str(model.dtype),
    })
    put_model_start_time = time.time()
    _ = STORAGE_CLIENT.delete(key=model_id,bucket_id=BUCKET_ID)
    model_result = STORAGE_CLIENT.put_ndarray(
        key       = model_id,
        ndarray   = model,
        tags      = {},
        bucket_id = BUCKET_ID
    ).result()
    put_model_st = time.time() - put_model_start_time
    logger.info({
        "event":"PUT.NDARRAY",
        "key":model_id,
        "bucket_id":BUCKET_ID,
        "shape":str(model.shape),
        "dtype":str(model.dtype),
        "service_time":put_model_st
    })
    
    # logger.debug("MODEL RESULT PUT SUCCESSFULLY")

    logger.debug({
        "event":"PUT.NDARRAY.BEFORE",
        "key":model_labels_id,
        "bucket_id":BUCKET_ID,
        "shape":str(model_labels.shape),
        "dtype":str(model_labels.dtype),
    })
    put_model_labels_start_time = time.time()
    _ = STORAGE_CLIENT.delete(key=model_labels_id,bucket_id=BUCKET_ID)
    model_labels_result = STORAGE_CLIENT.put_ndarray(
        key       = model_labels_id,
        ndarray   = model_labels,
        tags      = {},
        bucket_id = BUCKET_ID
    ).result()

    put_model_labels_st = time.time() - put_model_start_time
    logger.info({
        "event":"PUT.NDARRAY",
        "key":model_labels_id,
        "bucket_id":BUCKET_ID,
        "shape":str(model_labels.shape),
        "dtype":str(model_labels.dtype),
        "service_time":put_model_labels_st
    })
    end_time             = time.time() # Get the time when it ends
    service_time       = end_time - local_start_time # Get the service time
    logger.info({
        "event":"KNN.TRAIN.COMPLETED",
        "model_id":model_id,
        "model_labels_id":model_labels_id,
        "service_time":service_time
    })

    return Response(
        response = json.dumps({
            "service_time": service_time,
            "algorithm"   : algorithm,
        }),
        status   = 200,
        headers  = {}
    )


@classification.route("/knn/predict",methods = ["POST"])
def knn_predict():
    try:
        local_start_time                  = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        TESTING                      = current_app.config.get("TESTING",True)
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        # liu:Liu                      = current_app.config.get("liu")
        # dataowner:DataOwner          = current_app.config.get("dataowner")
        STORAGE_CLIENT:V4Client      = current_app.config.get("STORAGE_CLIENT")
        # num_chunks                   = current_app.config.get("NUM_CHUNKS",4)
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        WORKER_TIMEOUT               = int(current_app.config.get("WORKER_TIMEOUT",300))
        if executor == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm         = Constants.ClassificationAlgorithms.KNN_PREDICT
        s                 = Session()
        request_headers    = request.headers #Headers for the request
        model_id          = request_headers.get("Model-Id","model-0") #iris
        records_test_id   = request_headers.get("Records-Test-Id","matrix-0_data")
        records_test_filename   = request_headers.get("Records-Test-Filename",records_test_id)
        extension         = request_headers.get("Extension","npy")
        records_test_path = "{}/{}.{}".format(SOURCE_PATH, records_test_filename, extension)
        
        logger.debug({
            "event":"KNN.PREDICT.STARTED",
            "algorithm":algorithm, 
            "model_id":model_id,
            "records_test_id":records_test_id,
            "recors_test_filename":records_test_filename,
            "records_test_path":records_test_path,
            "extension":extension,
        })
        logger.debug({
            "event":"GET.RECORDS.BEFORE",
            "path":records_test_path
        })
        get_records_test_start_time = time.time()
        with open(records_test_path, "rb") as f:
            records_test = np.load(f)   
        get_recors_test_st= time.time() -get_records_test_start_time
        logger.info({
            "event":"GET.RECORDS",
            "path":records_test_path,
            "service_time":get_recors_test_st
        })

        # 

        logger.debug({
            "event":"PUT.NDARRAY.BEFORE",
            "key":records_test_id,
            "bucket_id":BUCKET_ID,
            "shape":str(records_test.shape),
            "dtype":str(records_test.dtype),
        })
        put_records_start_time = time.time()
        _ = STORAGE_CLIENT.delete(key=records_test_id,bucket_id=BUCKET_ID)
        records_result  = STORAGE_CLIENT.put_ndarray(
            key       = records_test_id,
            ndarray   = records_test,
            tags      = {},
            bucket_id = BUCKET_ID
        ).result()
        put_records_st = time.time() - put_records_start_time
        logger.info({
            "event":"PUT.NDARRAY",
            "key":records_test_id,
            "bucket_id":BUCKET_ID,
            "shape":str(records_test.shape),
            "dtype":str(records_test.dtype),
            "service_time":put_records_st
        })
        # logger.debug("RECORDS_TEST PUT SUCCESSFULLY")

        
        managerResponse:RoryManager = current_app.config.get("manager") # Communicates with the manager

        get_worker_start_time = time.time()
        get_worker_result          = managerResponse.getWorker( #Gets the worker from the manager
            headers = {
                "Algorithm"             : algorithm,
                "Start-Request-Time"    : str(local_start_time),
                "Start-Get-Worker-Time" : str(get_worker_start_time) 
            }
        )

        if get_worker_result.is_err:
            error = get_worker_result.unwrap_err()
            logger.error(str(error))
            return Response(str(error), status=500)
        (_worker_id,port) = get_worker_result.unwrap()

        get_worker_end_time         = time.time() 
        get_worker_service_time     = get_worker_end_time - get_worker_start_time
        worker_id                    =  "localhost" if TESTING else _worker_id

        logger.info({
            "event":"MANAGER.GET.WORKER",
            "algorithm":algorithm,
            "worker_id":_worker_id,
            "port":port,
            "service_time":get_worker_service_time,
        })

        worker         = RoryWorker( #Allows to establish the connection with the worker
            workerId   = worker_id,
            port       = port,
            session    = s,
            algorithm  = algorithm,
        )
        
        logger.debug({
            "event":"WORKER.PREDICT.BEFORE",
            "model_id":model_id,
            "records_test_id":records_test_id
        })
        worker_start_time = time.time()
        workerResponse = worker.run(
            headers    = {
                "Records-Test-Id": records_test_id,
                "Model-Id": model_id
            },
            timeout = WORKER_TIMEOUT
        )
        workerResponse.raise_for_status()
        # logger.debug("RUN_WORKER_RESPONSE {}".format(workerResponse))
        
        worker_end_time     = time.time()
        worker_response_time = worker_end_time - worker_start_time 
        stringWorkerResponse = workerResponse.content.decode("utf-8") #Response from worker
        jsonWorkerResponse   = json.loads(stringWorkerResponse) #pass to json
        endTime              = time.time() # Get the time when it ends
        worker_service_time  = jsonWorkerResponse["service_time"]
        label_vector         = jsonWorkerResponse["label_vector"]
        # workerResponse.headers.get("Service-Time",0) # Extract the time at which it started]]
        response_time        = endTime - local_start_time # Get the service time
        logger.info({
            "event":"WORKER.PREDICT",
            "model_id":model_id,
            "records_test_id":records_test_id,
            "service_time":worker_response_time,
            "worker_service_time":worker_service_time
        })

        logger.info({
            "event":"KNN.COMPLETED",
            "worker_response_time":worker_response_time,
            "worker_service_time":worker_service_time,
            "response_time":response_time
        })
        return Response(
            response = json.dumps({
                "label_vector" : label_vector,
                "worker_service_time" : worker_service_time,
                "worker_response_time" : worker_response_time,
                "response_time": response_time,
                "algorithm"   : algorithm,
            }),
            status   = 200,
            headers  = {}
        )
    except Exception as e:
        logger.error("CLIENT_ERROR "+str(e))
        return Response(response = None, status = 500, headers={"Error-Message":str(e)})