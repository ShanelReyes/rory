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
from mictlanx.v4.interfaces.responses import PutResponse, GetNDArrayResponse
from concurrent.futures import ProcessPoolExecutor
from utils.utils import Utils
from option import Result, Some

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
        local_start_time             = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        liu:Liu                      = current_app.config.get("liu")
        dataowner:DataOwner          = current_app.config.get("dataowner")
        STORAGE_CLIENT:V4Client      = current_app.config.get("STORAGE_CLIENT")
        _num_chunks                  = current_app.config.get("NUM_CHUNKS",4)
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        np_random                    = current_app.config.get("np_random")
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
            "m":int(m),
            "liu_round":liu.round
        })        

        model_path_exists        = os.path.exists(model_path) 
        model_path_labels_exists = os.path.exists(model_labels_path)
        if not model_path_exists or not model_path_labels_exists:
            return Response(response="Either model or label vector not found", status=500)
        else:
            
            logger.debug({
                "event":"READ.LOCAL.BEFORE",
                "model_id":model_id,
                "algorithm":algorithm,
                "model_path":model_path,
                "model_filename":model_filename,
            })
            read_local_model_start_time = time.time()
            with open(model_path, "rb") as f:
                model:npt.NDArray = np.load(f)
            read_local_model_st = time.time() - read_local_model_start_time
            
            logger.info({
                "event":"READ.LOCAL",
                "model_id":model_id,
                "algorithm":algorithm,
                "model_path":model_path,
                "service_time":read_local_model_st
            })
                
            logger.debug({
                "event":"READ.LOCAL.BEFORE",
                "model_id":model_id,
                "algorithm":algorithm,
                "model_labels_path":model_labels_path,
                "model_labels_filename":model_labels_filename,
            })
            read_local_model_labels_start_time = time.time()

            with open(model_labels_path, "rb") as f:
                model_labels:npt.NDArray = np.load(f)
                model_labels = model_labels.astype(np.int16)
            
            read_local_model_labels_st = time.time() - read_local_model_labels_start_time
            logger.info({
                "event":"READ.LOCAL",
                "model_id":model_id,
                "algorithm":algorithm,
                "model_labels_path":model_labels_path,
                "model_labels_filename":model_labels_filename,
                "service_time":read_local_model_labels_st
            })
            
            logger.debug({
                "event":"PUT.NDARRAY.BEFORE",
                "model_id":model_id,
                "algorithm":algorithm,
                "key":model_labels_id,
                "bucket_id":BUCKET_ID,
                "shape":str(model_labels.shape),
                "dtype":str(model_labels.dtype)
            })

            put_model_labels_start_time = time.time()
            # _delete_result = Utils.while_not_delete(STORAGE_CLIENT=STORAGE_CLIENT, bucket_id=BUCKET_ID, key=model_labels_id)
            # del_result = STORAGE_CLIENT.delete(key= model_labels_id, bucket_id=BUCKET_ID)
            # X = STORAGE_CLIENT.put_ndarray(
            #     key       = model_labels_id,
            #     ndarray   = model_labels,
            #     tags      = {},
            #     bucket_id = BUCKET_ID
            # ).result()
            ptm_result = Utils.delete_and_put_ndarray(
            STORAGE_CLIENT = STORAGE_CLIENT, 
            bucket_id      = BUCKET_ID, 
            ball_id        = model_labels_id, 
            key            = model_labels_id,
            ndarray        = model_labels, 
            tags           = {}
        )
            
            put_model_labels_st = time.time() - put_model_labels_start_time

            logger.info({
                "event":"PUT.NDARRAY",
                "model_id":model_id,
                "key":model_labels_id,
                "algorithm":algorithm,
                "bucket_id":BUCKET_ID,
                "shape":str(model_labels.shape),
                "dtype":str(model_labels.dtype),
                "service_time":put_model_labels_st
            })

            r:int = model.shape[0]
            a:int = model.shape[1]
            encrypted_model_shape = "({},{},{})".format(r,a,m)
            n = a*r*int(m)

            logger.debug({
                "event":"SEGMENT.ENCRYPT.LIU.BEFORE",
                "model_id":model_id,
                "algorithm":algorithm,
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

            logger.debug({
                "event":"PUT.CHUNKED.BEFORE",
                "model_id":model_id,
                "algorithm":algorithm,
                "key":encrypted_model_id,
                "num_chunks":num_chunks
            })
            put_chunked_start_time = time.time()
            # _delete_result = Utils.while_not_delete_ball_id(STORAGE_CLIENT=STORAGE_CLIENT, bucket_id=BUCKET_ID, ball_id=encrypted_model_id)
            # STORAGE_CLIENT.delete_by_ball_id(
            #     ball_id   = encrypted_model_id, 
            #     bucket_id = BUCKET_ID
            # )
            chunks_bytes = Utils.chunks_to_bytes_gen(
                chs = encrypted_model_chunks
            )
            # put_chunks_generator_results = STORAGE_CLIENT.put_chunked(
            #     key       = encrypted_model_id, 
            #     chunks    = chunks_bytes, 
            #     bucket_id = BUCKET_ID,
            #     tags      = {
            #         "shape": str(encrypted_model_shape),
            #         "dtype":"float64"
            #     }
            # )

            put_chunks_generator_results = Utils.delete_and_put_chunked(
            STORAGE_CLIENT = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            ball_id        = encrypted_model_id,
            key            = encrypted_model_id,
            chunks         = chunks_bytes,
            tags = {
                "shape": str(encrypted_model_shape),
                "dtype":"float64"
            }
        )

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
                }),
                status  = 200,
            )
    except Exception as e:
        logger.error({
            "msg":str(e)
        })
        return Response(response = None, status = 500, headers={"Error-Message":str(e)})


@classification.route("/sknn/predict",methods = ["POST"])
def sknn_predict():
    try:
        local_start_time             = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        TESTING                      = current_app.config.get("TESTING",True)
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        liu:Liu                      = current_app.config.get("liu")
        dataowner:DataOwner          = current_app.config.get("dataowner")
        STORAGE_CLIENT:V4Client      = current_app.config.get("STORAGE_CLIENT")
        _num_chunks                  = current_app.config.get("NUM_CHUNKS",4)
        max_workers                  = current_app.config.get("MAX_WORKERS",2)
        np_random                    = current_app.config.get("np_random")
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        WORKER_TIMEOUT               = int(current_app.config.get("WORKER_TIMEOUT",300))
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
        records_test_path         = "{}/{}.{}".format(SOURCE_PATH, records_test_filename, extension)
        
        if _encrypted_model_dtype == -1:
            return Response("Encrypted-Model-Dtype", status=500)
        if _encrypted_model_shape == -1 :
            return Response("Encrypted-Model-Shape header is required", status=500)
    
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
        })        

        logger.debug({
            "event":"READ.LOCAL.BEFORE",
            "model_id":model_id,
            "records_path":records_test_path,
            "records_filename":model_id,
            "algorithm":algorithm,
        })
        read_local_start_time = time.time()
        with open(records_test_path, "rb") as f:
            records_test:npt.NDArray = np.load(f)    
        read_local_st = time.time() - read_local_start_time

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
        cores = os.cpu_count()
        max_workers = num_chunks if max_workers > num_chunks else max_workers
        max_workers = cores if max_workers > cores else max_workers
        n = a*r*int(m)

        logger.debug({
            "event":"SEGMENT.ENCRYPT.LIU.BEFORE",
            "model_id":model_id,
            "key":encrypted_records_test_id,
            "records_shape":str(records_test.shape),
            "records_dtype":str(records_test.dtype),
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

        logger.debug({
            "event":"PUT.CHUNKED.BEFORE",
            "model_id":model_id,
            "encrypted_records_text_id":encrypted_records_test_id,
            "algorithm":algorithm,
            "num_chunks":num_chunks
        })
        put_chunks_start_time = time.time()
        encrypted_records_shape = (r,a,int(m))

        # _delete_result = Utils.while_not_delete_ball_id(STORAGE_CLIENT=STORAGE_CLIENT, bucket_id=BUCKET_ID, ball_id=encrypted_records_test_id)

        # STORAGE_CLIENT.delete_by_ball_id(
        #     ball_id   = encrypted_records_test_id, 
        #     bucket_id = BUCKET_ID
        # )
        chunks_bytes = Utils.chunks_to_bytes_gen(
            chs = encrypted_records_chunks
        )
        # put_chunks_generator_results = STORAGE_CLIENT.put_chunked(
        #     key       = encrypted_records_test_id, 
        #     chunks    = chunks_bytes, 
        #     bucket_id = BUCKET_ID,
        #     tags      = {
        #         "shape": str(encrypted_records_shape),
        #         "dtype":"float64"
        #     }
        # )
        put_chunks_generator_results = Utils.delete_and_put_chunked(
            STORAGE_CLIENT = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            ball_id        = encrypted_records_test_id,
            key            = encrypted_records_test_id,
            chunks         = chunks_bytes,
            tags = {
                "shape": str(encrypted_records_shape),
                "dtype":"float64"
            }
        )

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
        }

        logger.debug({
            "event":"WORKER.RUN.1.BEFORE",
            "model_id":model_id,
            "records_test_id":records_test_id,
            "encrypted_model_shape":_encrypted_model_shape,
            "encrypted_model_dtype":_encrypted_model_dtype,
            "encrypted_records_shape":str(encrypted_records_shape),
            "encrypted_records_dtype":str(encrypted_records_dtype),
            "num_chunks":num_chunks,
            "algorithm":algorithm,
        })
        worker_run1_response = worker.run(
            headers = run1_headers,
            timeout = WORKER_TIMEOUT
        )
        worker_run1_response.raise_for_status()

        stringWorkerResponse = worker_run1_response.content.decode("utf-8") #Response from worker
        jsonWorkerResponse   = json.loads(stringWorkerResponse) #pass to json
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
        logger.debug({
            "event":"GET.NDARRAY.BEFORE",
            "model_id":model_id,
            "records_test_id":records_test_id,
            "distances_id":distances_id,
            "distances_shape":distances_shape,
            "distances_dtype":distances_dtype,
            "num_chunks":num_chunks,
            "algorithm":algorithm,
        }) 

        x:Result[GetNDArrayResponse,Exception] = STORAGE_CLIENT.get_ndarray_with_retry(
            key         = distances_id,
            bucket_id   = BUCKET_ID,
            max_retries = 20,
            delay       = 2
            ).result()
        
        if x.is_err:
            raise Exception("{} not found".format(distances_id))
        response = x.unwrap()
        all_distances                  = response.value
        all_distances_metadata         = response.metadata 
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
        logger.debug({
            "event":"DECRYPT.MIN.BEFORE",
            "model_id":model_id,
            "records_test_id":records_test_id,
            "algorithm":algorithm,
        })

        matrix_distances_plain = liu.decryptMatrix(
            ciphertext_matrix = all_distances,
            secret_key        = liu.sk,
            m                 = int(m)
        )

        min_distances_index         = np.argmin(matrix_distances_plain.matrix,axis=1)
        print("MIN_DISTANCE_INDEX",min_distances_index.shape)
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
        
        logger.debug({
            "event":"PUT.NDARRAY.BEFORE",
            "model_id":model_id,
            "distances_id":min_distances_index_id,
            "algorithm":algorithm,
            "num_chunks":num_chunks
        })

        min_distances_chunks = Chunks.from_ndarray(
            ndarray      = min_distances_index.reshape(-1,1),
            group_id     = min_distances_index_id,
            chunk_prefix = Some(min_distances_index_id),
            num_chunks   = num_chunks,
        )

        if min_distances_chunks.is_none:
            raise "something went wrong creating the chunks"
        
        chunks_bytes = Utils.chunks_to_bytes_gen(
            chs = min_distances_chunks.unwrap()
        )

        t_chunks_generator_results = Utils.delete_and_put_chunked(
            STORAGE_CLIENT = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            ball_id        = min_distances_index_id,
            key            = min_distances_index_id,
            chunks         = chunks_bytes,
            tags = {
                "shape": str(min_distances_index.shape),
                "dtype": str(min_distances_index.dtype)
            }
        )

        # _delete_result = Utils.while_not_delete(STORAGE_CLIENT=STORAGE_CLIENT, bucket_id=BUCKET_ID, key=min_distances_index_id)
        # yd = STORAGE_CLIENT.delete(
        #     key       = min_distances_index_id,
        #     bucket_id = BUCKET_ID
        # )
        # y:Result[PutResponse,Exception]  = STORAGE_CLIENT.put_ndarray(
        #     key       = min_distances_index_id,
        #     ndarray   = min_distances_index,
        #     tags      = {},
        #     bucket_id = BUCKET_ID
        # ).result()
        # ptm_result = Utils.delete_and_put_ndarray(
        #     STORAGE_CLIENT = STORAGE_CLIENT, 
        #     bucket_id      = BUCKET_ID, 
        #     ball_id        = min_distances_index_id, 
        #     key            = min_distances_index_id,
        #     ndarray        = min_distances_index, 
        #     tags           = {}
        # )

        logger.info({
            "event":"PUT.NDARRAY",
            "model_id":model_id,
            "distances_id":min_distances_index_id,
            "algorithm":algorithm,
            "num_chunks":num_chunks
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
        
        logger.debug({
            "event":"WORKER.RUN.2.BEFORE",
            "model_id":model_id,
            "records_test_id":records_test_id,
            "encrypted_model_shape":_encrypted_model_shape,
            "encrypted_model_dtype":_encrypted_model_dtype,
            "encrypted_records_shape":str(encrypted_records_shape),
            "encrypted_records_dtype":str(encrypted_records_dtype),
            "num_chunks":num_chunks,
            "min_distances_index_id":min_distances_index_id,
            "algorithm":algorithm,
        })

        worker_run2_response = worker.run(
            headers = run2_headers,
            timeout = WORKER_TIMEOUT
        )
        worker_run2_response.raise_for_status()
        stringWorkerResponse2 = worker_run2_response.content.decode("utf-8") #Response from worker
        jsonWorkerResponse2   = json.loads(stringWorkerResponse2) #pass to json
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
        # logger.error("CLIENT_ERROR "+str(e))
        return Response(response = None, status = 500, headers={"Error-Message":str(e)})


@classification.route("/knn/train", methods = ["POST"])
def knn_train():
    local_start_time             = time.time()
    logger                       = current_app.config["logger"]
    BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
    SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
    STORAGE_CLIENT:V4Client      = current_app.config.get("STORAGE_CLIENT")
    executor:ProcessPoolExecutor = current_app.config.get("executor")
    if executor == None:
        raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
    algorithm             = Constants.ClassificationAlgorithms.KNN_TRAIN
    s                     = Session()
    request_headers       = request.headers #Headers for the request
    num_chunks                = int(request_headers.get("Num-Chunks",1))
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

    logger.debug({
        "event":"GET.MODEL.BEFORE",
        "path":model_path,
        "algorithm":algorithm,
        "model_id":model_id,
    })
    
    get_model_start_time = time.time()
    with open(model_path, "rb") as f:
        model:npt.NDArray = np.load(f)
    get_model_st = time.time()- get_model_start_time

    logger.info({
        "event":"GET.MODEL",
        "path":model_path,
        "service_time":get_model_st,
        "algorithm":algorithm,
        "model_id":model_id,
    })
    
    logger.debug({
        "event":"GET.MODEL.LABELS.BEFORE",
        "path":model_labels_path,
        "algorithm":algorithm,
        "model_id":model_id,
    })
    get_model_labels_start_time = time.time()
    with open(model_labels_path, "rb") as f:
        model_labels:npt.NDArray = np.load(f)
        model_labels             = model_labels.astype(np.int16)

    model_labels = model_labels.reshape(-1,1)
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

    logger.debug({
        "event":"CHUNKS.FROM.NDARRAY.BEFORE",
        "key":model_id,
        "algorithm":algorithm,
        "model_id":model_id,
        "bucket_id":BUCKET_ID,
        "shape":str(model.shape),
        "dtype":str(model.dtype),
    })
    put_model_start_time = time.time()
    # print("tres")
    model_chunks = Chunks.from_ndarray(
        ndarray      = model,
        group_id     = model_id,
        chunk_prefix = Some(model_id),
        num_chunks   = num_chunks,
    )

    if model_chunks.is_none:
        raise "something went wrong creating the chunks"
    
    logger.info({
        "event":"CHUNKS.FROM.NDARRAY",
        "key":model_id,
        "algorithm":algorithm,
        "model_id":model_id,
        "bucket_id":BUCKET_ID,
        "shape":str(model.shape),
        "dtype":str(model.dtype),
    })

    logger.debug({
        "event":"DELETE.AND.PUT.CHUNKED.BEFORE",
        "key":model_id,
        "algorithm":algorithm,
        "model_id":model_id,
        "bucket_id":BUCKET_ID,
        "shape":str(model.shape),
        "dtype":str(model.dtype)
    })

    chunks_bytes = Utils.chunks_to_bytes_gen(
        chs = model_chunks.unwrap()
    
    )

    t_chunks_generator_results = Utils.delete_and_put_chunked(
        STORAGE_CLIENT = STORAGE_CLIENT,
        bucket_id      = BUCKET_ID,
        ball_id        = model_id,
        key            = model_id,
        chunks         = chunks_bytes,
        tags = {
            "shape": str(model.shape),
            "dtype": str(model.dtype)
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
        "service_time":put_model_st
    })

    logger.debug({
        "event":"CHUNKS.FROM.NDARRAY.BEFORE",
        "key":model_labels_id,
        "algorithm":algorithm,
        "model_id":model_id,
        "model_label_id":model_labels_id,
        "bucket_id":BUCKET_ID,
        "shape":str(model_labels.shape),
        "dtype":str(model_labels.dtype),
    })
    put_model_labels_start_time = time.time()

    model_labels_chunks = Chunks.from_ndarray(
        ndarray      = model_labels,
        group_id     = model_labels_id,
        chunk_prefix = Some(model_labels_id),
        num_chunks   = num_chunks,
    )

    if model_labels_chunks.is_none:
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

    logger.debug({
        "event":"DELETE.AND.PUT.CHUNKED.BEFORE",
        "key":model_labels_id,
        "algorithm":algorithm,
        "model_id":model_id,
        "bucket_id":BUCKET_ID,
        "shape":str(model_labels.shape),
        "dtype":str(model_labels.dtype)
    })

    chunks_bytes = Utils.chunks_to_bytes_gen(
        chs = model_labels_chunks.unwrap()
    
    )

    model_labels_results = Utils.delete_and_put_chunked(
        STORAGE_CLIENT = STORAGE_CLIENT,
        bucket_id      = BUCKET_ID,
        ball_id        = model_labels_id,
        key            = model_labels_id,
        chunks         = chunks_bytes,
        tags = {
            "shape": str(model_labels.shape),
            "dtype": str(model_labels.dtype)
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
        }),
        status   = 200,
        headers  = {}
    )


@classification.route("/knn/predict",methods = ["POST"])
def knn_predict():
    try:
        local_start_time             = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        TESTING                      = current_app.config.get("TESTING",True)
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        STORAGE_CLIENT:V4Client      = current_app.config.get("STORAGE_CLIENT")
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
            "path":records_test_path,
            "algorithm":algorithm,
            "model_id":model_id,
        })
        get_records_test_start_time = time.time()
        with open(records_test_path, "rb") as f:
            records_test = np.load(f)   
        get_recors_test_st = time.time() -get_records_test_start_time

        logger.info({
            "event":"GET.RECORDS",
            "path":records_test_path,
            "service_time":get_recors_test_st,
            "algorithm":algorithm,
            "model_id":model_id,
        })

        logger.debug({
            "event":"PUT.NDARRAY.BEFORE",
            "key":records_test_id,
            "algorithm":algorithm,
            "model_id":model_id,
            "bucket_id":BUCKET_ID,
            "shape":str(records_test.shape),
            "dtype":str(records_test.dtype),
        })
        put_records_start_time = time.time()

        records_test_chunks = Chunks.from_ndarray(
            ndarray      = records_test,
            group_id     = records_test_id,
            chunk_prefix = Some(records_test_id),
            num_chunks   = num_chunks,
        )

        if records_test_chunks.is_none:
            raise "something went wrong creating the chunks"
        
        chunks_bytes = Utils.chunks_to_bytes_gen(
            chs = records_test_chunks.unwrap()
        )

        records_test_results = Utils.delete_and_put_chunked(
            STORAGE_CLIENT = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            ball_id        = records_test_id,
            key            = records_test_id,
            chunks         = chunks_bytes,
            tags = {
                "shape": str(records_test.shape),
                "dtype": str(records_test.dtype)
            }
        )

        service_time_client_end = time.time()
        service_time_client = service_time_client_end - local_start_time
        put_records_st = time.time() - put_records_start_time
        logger.info({
            "event":"PUT.NDARRAY",
            "key":records_test_id,
            "algorithm":algorithm,
            "model_id":model_id,
            "bucket_id":BUCKET_ID,
            "shape":str(records_test.shape),
            "dtype":str(records_test.dtype),
            "service_time":put_records_st
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
            "algorithm":algorithm,
            "model_id":model_id,
            "worker_id":_worker_id,
            "port":port,
            "service_time":get_worker_service_time,
        })

        worker_start_time = time.time()
        worker        = RoryWorker( #Allows to establish the connection with the worker
            workerId  = worker_id,
            port      = port,
            session   = s,
            algorithm = algorithm,
        )
        
        logger.debug({
            "event":"WORKER.PREDICT.BEFORE",
            "algorithm":algorithm,
            "model_id":model_id,
            "records_test_id":records_test_id
        })

        workerResponse = worker.run(
            headers    = {
                "Records-Test-Id": records_test_id,
                "Model-Id": model_id
            },
            timeout = WORKER_TIMEOUT
        )
        workerResponse.raise_for_status()
        
        worker_end_time      = time.time()
        worker_response_time = worker_end_time - worker_start_time 
        stringWorkerResponse = workerResponse.content.decode("utf-8") #Response from worker
        jsonWorkerResponse   = json.loads(stringWorkerResponse) #pass to json
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