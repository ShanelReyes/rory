import os
import time, json
from option import Some
from requests import Session
from flask import Blueprint,current_app,request,Response
from option import Result
from rory.core.interfaces.rorymanager import RoryManager
from rory.core.interfaces.roryworker import RoryWorker
from rory.core.security.dataowner import DataOwner
from rory.core.security.cryptosystem.liu import Liu
from rory.core.security.cryptosystem.fdhope import Fdhope
from rory.core.utils.constants import Constants
from rory.core.utils.utils import Utils as RoryUtils
from mictlanx.v4.interfaces.responses import PutResponse
from mictlanx.v4.client import Client  as V4Client
from mictlanx.utils.segmentation import Chunks,Chunk
from concurrent.futures import ProcessPoolExecutor
from utils.utils import Utils

clustering = Blueprint("clustering",__name__,url_prefix = "/clustering")

@clustering.route("/test",methods=["GET","POST"])
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


# KMEANS
@clustering.route("/kmeans",methods   = ["POST"])
def kmeans():
    try:
        arrivalTime                   = time.time()
        logger                        = current_app.config["logger"]
        TESTING                       = current_app.config.get("TESTING",True)
        SOURCE_PATH                   = current_app.config["SOURCE_PATH"]
        STORAGE_CLIENT:V4Client       = current_app.config.get("STORAGE_CLIENT")
        BUCKET_ID:str                 = current_app.config.get("BUCKET_ID","rory")
        WORKER_TIMEOUT                = int(current_app.config.get("WORKER_TIMEOUT",300))
        algorithm                     = Constants.ClusteringAlgorithms.KMEANS
        s                             = Session()
        # Headers
        request_headers               = request.headers #Headers for the request
        plaintext_matrix_id           = request_headers.get("Plaintext-Matrix-Id","matrix-0")
        plaintext_matrix_filename     = request_headers.get("Plaintext-Matrix-Filename","matrix-0")
        extension                     = request_headers.get("Extension","csv")
        k                             = request_headers.get("K","3")
        # 
        plaintext_matrix_path         = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_filename, extension)
        local_read_dataset_start_time = time.time()

        logger.debug({
            "event":"KMEANS.STARTED",
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "plaintext_matrix_filename":plaintext_matrix_filename,
            "extension":extension,
            "k":k,
            "plaintext_matrix_path":plaintext_matrix_path
        })

        logger.debug({
            "event":"LOCAL.READ.DATASET.BEFORE",
            "path":plaintext_matrix_path,
            "filename":plaintext_matrix_filename,
        })

        plaintext_matrix_result = Utils.read_numpy_from(
            client    = STORAGE_CLIENT,
            path      = plaintext_matrix_path,
            extension = extension
        )

        if plaintext_matrix_result.is_ok:
            plaintextMatrix = plaintext_matrix_result.unwrap()
        else:
            raise plaintext_matrix_result.unwrap_err()
  
        local_read_dataset_st = time.time() - local_read_dataset_start_time
        
        logger.debug({
            "event":"LOCAL.READ.DATASET",
            "path":plaintext_matrix_path,
            "key":plaintext_matrix_id,
            "filename":plaintext_matrix_filename,
            "service_time":local_read_dataset_st
        })

        # DELETE FIRST
        _delete_result = STORAGE_CLIENT.delete(bucket_id=BUCKET_ID,key=plaintext_matrix_id)

        put_pm_start_time = time.time()

        logger.debug({
            "event":"PUT.PLAINTEXT_MATRIX.BEFORE",
            "key":plaintext_matrix_id,
            "shape":str(plaintextMatrix.shape),
            "dtype":str(plaintextMatrix.dtype)
        })

        ptm_result:Result[PutResponse,Exception] = STORAGE_CLIENT.put_ndarray(
            key       = plaintext_matrix_id,
            ndarray   = plaintextMatrix,
            tags      = {},
            bucket_id = BUCKET_ID
        ).result()

        print("PTM",ptm_result)
        if ptm_result.is_err:
            error = ptm_result.unwrap_err()
            logger.error({
                "msg":str(error)
            })
            return Response(response=str(error), status=500)

        put_pm_service_time = time.time()- put_pm_start_time
        logger.info({
            "event":"PUT.PLAINTEXT_MATRIX",
            "key":plaintext_matrix_id,
            "shape":str(plaintextMatrix.shape),
            "dtype":str(plaintextMatrix.dtype),
            "service_time":put_pm_service_time
        })
        
        logger.debug({
            "event":"MANAGER.GET.WORKER.BEFORE",
            "algorithm":algorithm
        })

        get_worker_arrival_time = time.time()
        manager:RoryManager     = current_app.config.get("manager") # Communicates with the manager
        get_worker_result       = manager.getWorker( #Gets the worker from the manager
            headers = {
                "Algorithm"         : algorithm,
                "Start-Request-Time": str(arrivalTime)
            }
        )
        if get_worker_result.is_err:
            error = get_worker_result.unwrap_err()
            logger.error(str(error))
            return Response(str(error), status=500)
        (worker_id, worker_port) = get_worker_result.unwrap()
        
        get_worker_end_time     = time.time()
        get_worker_service_time = get_worker_end_time - get_worker_arrival_time 
        worker_id               = "localhost" if TESTING else worker_id

        logger.info({
            "event":"GET.WORKER",
            "service_time":get_worker_service_time,
            "worker_id":worker_id,
            "worker_port":worker_port
        })

        worker = RoryWorker( #Allows to establish the connection with the worker
            workerId  = worker_id,
            port      = worker_port,
            session   = s,
            algorithm = algorithm
        )   

        interaction_arrival_time = time.time()
        logger.debug({
            "event":"WORKER.RUN.BEFORE",
            "algorithm":algorithm,
            "k":k,
            "plaintext_matrix_id":plaintext_matrix_id,
            "worker_id":worker_id,
            "service_time":time.time() - interaction_arrival_time
        })

        workerResponse = worker.run(
            headers = {
                "Plaintext-Matrix-Id": plaintext_matrix_id,
                "K": str(k),
            },
            timeout = WORKER_TIMEOUT
        )
        logger.info({
            "event":"WORKER.RUN",
            "algorithm":algorithm,
            "k":k,
            "plaintext_matrix_id":plaintext_matrix_id,
            "worker_id":worker_id,
            "service_time":time.time() - interaction_arrival_time
        })

        stringWorkerResponse = workerResponse.content.decode("utf-8") #Response from worker
        jsonWorkerResponse   = json.loads(stringWorkerResponse) #pass to json
        worker_service_time  =  jsonWorkerResponse["service_time"]
        iterations           = int(jsonWorkerResponse["iterations"]) # Extract the current number of iterations
        endTime              = time.time() # Get the time when it ends
        response_time        = endTime - arrivalTime # Get the service time

        logger.info({
            "event":"CLUSTERING",
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "worker_service_time": worker_service_time,
            "response_time":response_time,
            "iterations":iterations,
            "k":k
        })

        return Response(
            response = json.dumps({
                "label_vector" : jsonWorkerResponse.get("label_vector",[]),
                "iterations":iterations,
                "worker_service_time" : worker_service_time,
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
        return Response(response = None, status= 500, headers = {"Error-Message":str(e)})


@clustering.route("/skmeans",methods = ["POST"])
def skmeans():
    try:
        arrivalTime                  = time.time()
        # Config
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
        
        if executor == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm               = Constants.ClusteringAlgorithms.SKMEANS
        s                       = Session()
        # Headers
        request_headers           = request.headers #Headers for the request
        num_chunks                = int(request_headers.get("Num-Chunks",_num_chunks))
        plaintext_matrix_id       = request_headers.get("Plaintext-Matrix-Id","matrix0")
        encrypted_matrix_id       = "encrypted{}".format(plaintext_matrix_id) # The id of the encrypted matrix is built
        udm_id                    = "{}udm".format(plaintext_matrix_id) # The iudm id is built
        plaintext_matrix_filename = request_headers.get("Plaintext-Matrix-Filename","matrix-0")
        extension                 = request_headers.get("Extension","csv")
        k                         = int(request_headers.get("K"))
        experiment_iteration      = request_headers.get("Experiment-Iteration","0")
        MAX_ITERATIONS            = int(request_headers.get("Max-Iterations",current_app.config.get("MAX_ITERATIONS",10)))
        WORKER_TIMEOUT            = int(current_app.config.get("WORKER_TIMEOUT",300))
        requestId                 = "request-{}".format(plaintext_matrix_id)
        m                         = dataowner.m
        plaintext_matrix_path     = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_filename, extension)
        
        logger.debug({
            "event":"SKMEANS.STARTED",
            "plaintext_matrix_id":plaintext_matrix_id,
            "encrypted_matrix_id":encrypted_matrix_id,
            "udm_id":udm_id,
            "plaintext_matrix_filename":plaintext_matrix_filename,
            "plaintext_matrix_path":plaintext_matrix_path,
            "m":m,
            "k":k,
            "num_chunks":num_chunks,
            "max_workers":max_workers,
            "bucket_id":BUCKET_ID,
            "testing":TESTING,
            "experiment_iteration":experiment_iteration,
            "max_iterations":MAX_ITERATIONS,
            "request_id":requestId,
            "worker_timeout":WORKER_TIMEOUT,
            "source_path":SOURCE_PATH,
        })
        
        logger.debug({
            "event":"LOCAL.READ.DATASET.BEFORE",
            "path":plaintext_matrix_path,
            "filename":plaintext_matrix_filename,
        })
        local_read_dataset_start_time = time.time()
        plaintext_matrix_result  = Utils.read_numpy_from(
            client    = STORAGE_CLIENT,
            path      = plaintext_matrix_path,
            extension = extension,
        )
        if plaintext_matrix_result.is_ok:
            plaintext_matrix = plaintext_matrix_result.unwrap()
        else:
            raise plaintext_matrix_result.unwrap_err()

        local_read_dataset_st = time.time() - local_read_dataset_start_time
        
        r = plaintext_matrix.shape[0]
        a = plaintext_matrix.shape[1]
        logger.debug({
            "event":"LOCAL.READ.DATASET",
            "path":plaintext_matrix_path,
            "filename":plaintext_matrix_filename,
            "records":r,
            "attributes":a,
            "service_time":local_read_dataset_st
        })
        cores       = os.cpu_count()
        max_workers = num_chunks if max_workers > num_chunks else max_workers
        max_workers = cores if max_workers > cores else max_workers

        encryption_start_time = time.time()
        n = a*r*m
        logger.debug({
            "event":"SEGMENT.ENCRYPT.LIU.BEFORE",
            "key":encrypted_matrix_id,
            "plaintext_matrix_shape":str(plaintext_matrix.shape),
            "plaintext_matrix_dtype":str(plaintext_matrix.dtype),
            "algorithm":algorithm,
            "n":n,
            "num_chunks":num_chunks,
            "max_workers":max_workers,
        })
        encrypted_matrix_chunks = Utils.segment_and_encrypt_liu_with_executor( #Encrypt 
            executor         = executor,
            key              = encrypted_matrix_id,
            plaintext_matrix = plaintext_matrix,
            dataowner        = dataowner,
            n                = n,
            num_chunks       = num_chunks,
            np_random        = np_random
        )
        segment_encrypt_service_time = time.time() - encryption_start_time
        logger.info({
            "event":"SEGMENT.ENCRYPT.LIU",
            "key":encrypted_matrix_id,
            "plaintext_matrix_shape":str(plaintext_matrix.shape),
            "plaintext_matrix_dtype":str(plaintext_matrix.dtype),
            "algorithm":algorithm,
            "n":n,
            "num_chunks":num_chunks,
            "max_workers":max_workers,
            "service_time":segment_encrypt_service_time
        })
        
        logger.debug({
            "event":"PUT.CHUNKS.BEFORE",
            "key":encrypted_matrix_id,
            "algorithm":algorithm,
            "num_chunks":num_chunks
        })
        put_chunks_start_time = time.time()
        STORAGE_CLIENT.delete_by_ball_id(
            ball_id   = encrypted_matrix_id, 
            bucket_id = BUCKET_ID
        )
        # put_chunks_generator_results = STORAGE_CLIENT.put_chunks(
        #     key       = encrypted_matrix_id, 
        #     chunks    = encrypted_matrix_chunks, 
        #     bucket_id = BUCKET_ID,
        #     tags      = {}
        # )
        
        chunks_bytes = Utils.chunks_to_bytes_gen(
            chs = encrypted_matrix_chunks
        )

        put_chunks_generator_results = STORAGE_CLIENT.put_chunked(
            key       = encrypted_matrix_id, 
            chunks    = chunks_bytes, 
            bucket_id = BUCKET_ID,
            tags      = {
                "shape": str((r,a,m)),
                "dtype":"float64"
            }
        )
        print("PUT RESPONSE",put_chunks_generator_results)
        # time.sleep(100)


        put_chunks_st = time.time() - put_chunks_start_time
        logger.info({
            "event":"PUT.CHUNKS",
            "key":encrypted_matrix_id,
            "num_chunks":num_chunks,
            "algorithm":algorithm,
            "service_time":put_chunks_st
        })
        
        # for i,put_chunk_result in enumerate(
        # ):
        #     if put_chunk_result.is_err:
        #         logger.error("Something went wrong storage and encrypt the chunk.")
        #         return Response(
        #             status   = 500,
        #             response = "{}".format(str(put_chunk_result.unwrap_err()))
        #         )
        encryption_end_time      = time.time()
        segment_encrypt_put_time = encryption_end_time - encryption_start_time

        logger.info({
            "event":"PUT.SEGMENT.ENCRYPT", 
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "encrypted_matrid_id":encrypted_matrix_id,
            "service_time":segment_encrypt_put_time
        })

        logger.debug({
            "event":"UDM.GENERATION.BEFORE",
            "algorithm":algorithm,
            "plaintext_matrix_shape":str(plaintext_matrix.shape),
            "plaintext_matrix_dtype":str(plaintext_matrix.dtype),
        })
        udm_start_time = time.time()
        udm            = dataowner.get_U(
            plaintext_matrix = plaintext_matrix,
            algorithm        = algorithm
        )
        udm_gen_st = time.time()- udm_start_time
        logger.info({
            "event":"UDM.GENERATION",
            "shape":str(udm.shape),
            "udm_id":udm_id,
            "service_time":udm_gen_st,
        })

        logger.debug({
            "event":"PUT.NDARRAY.BEFORE",
            "key":udm_id,
            "bucket_id":BUCKET_ID,
            "udm_shape":str(udm.shape),
            "udm_dtype":str(udm.dtype),
        })
        udm_put_start_time = time.time()
        yd = STORAGE_CLIENT.delete_by_ball_id(
            ball_id   = udm_id,
            bucket_id = BUCKET_ID
        )

        udm_put_result:Result[PutResponse,Exception] = STORAGE_CLIENT.put_ndarray(
            key       = udm_id, 
            ndarray   = udm, 
            tags      = {},
            bucket_id = BUCKET_ID
        ).result()

        if udm_put_result.is_err:
            raise udm_put_result.unwrap_err()
        udm_put_st = time.time() - udm_put_start_time
        logger.info({            
            "event":"PUT.NDARRAY",
            "key":udm_id,
            "bucket_id":BUCKET_ID,
            "udm_shape":str(udm.shape),
            "udm_dtype":str(udm.dtype),
            "service_time":udm_put_st
        })

        managerResponse:RoryManager = current_app.config.get("manager") # Communicates with the manager
        get_worker_start_time       = time.time()
        get_worker_result           = managerResponse.getWorker( #Gets the worker from the manager
            headers = {
                "Algorithm"             : algorithm,
                "Start-Request-Time"    : str(arrivalTime),
                "Start-Get-Worker-Time" : str(get_worker_start_time) 
            }
        )
        if get_worker_result.is_err:
            error = get_worker_result.unwrap_err()
            logger.error(str(error))
            return Response(str(error), status=500)
        (worker_id,port) = get_worker_result.unwrap()

        get_worker_end_time     = time.time() 
        get_worker_service_time = get_worker_end_time - get_worker_start_time
        worker_id               =  "localhost" if TESTING else worker_id

        logger.info({
            "event":"MANAGER.GET.WORKER",
            "worker_id":worker_id,
            "port":port,
            "algorithm":algorithm,
            "service_time":get_worker_service_time,
            "k":k,
            "m":m
        })
        
        worker = RoryWorker( #Allows to establish the connection with the worker
            workerId  = worker_id,
            port      = port,
            session   = s,
            algorithm = algorithm,
        )

        status                   = Constants.ClusteringStatus.START #Set the status to start
        worker_run1_response     = None 
        interaction_arrival_time = time.time()
        iterations               = 0
        


        label_vector = None
        while (status != Constants.ClusteringStatus.COMPLETED): #While the status is not completed
            
            inner_interaction_arrival_time = time.time()
            run1_headers  = {
                "Step-Index"             : "1",
                "Clustering-Status"      : str(status),
                "Plaintext-Matrix-Id"    : plaintext_matrix_id,
                "Request-Id"             : requestId,
                "Encrypted-Matrix-Id"    : encrypted_matrix_id,
                "Encrypted-Matrix-Shape" : "({},{},{})".format(r,a,m),
                "Encrypted-Matrix-Dtype" : "float64",
                "Encrypted-Udm-Dtype"    : "float64",
                "Num-Chunks"             : str(num_chunks),
                "Iterations"             : str(iterations),
                "K"                      : str(k),
                "M"                      : str(m), 
                "Experiment-Iteration"   : str(experiment_iteration), 
                "Max-Iterations"         : str(MAX_ITERATIONS) 
            }
            
            logger.debug({
                "event":"WORKER.RUN1.BEFORE",
                "plaintext_matrix_id":plaintext_matrix_id,
                "step_index":"1",
                "clustering_status":str(status),
                "request_id":requestId,
                "encrypted_matrix_id":encrypted_matrix_id,
                "encrypted_matrix_shape":"({},{},{})".format(r,a,m),
                "encrypted_matrix_dtype":"float64", 
                "num_chunks":num_chunks,
                "iterations":iterations,
                "k":k, 
                "m":m, 
                "experiment_iteration":experiment_iteration,
                "max_iterations":MAX_ITERATIONS
            })
            
            worker_run1_response = worker.run(
                timeout = WORKER_TIMEOUT, 
                headers = run1_headers
            ) #Run 1 starts
            worker_run1_status = worker_run1_response.status_code

            if worker_run1_status !=200:
                return Response("Worker error: {}".format(worker_run1_response.content),status=500)
            
            worker_run1_response.raise_for_status()
            stringWorkerResponse      = worker_run1_response.content.decode("utf-8") #Response from worker
            jsonWorkerResponse        = json.loads(stringWorkerResponse) #pass to json
            encrypted_shift_matrix_id = jsonWorkerResponse["encrypted_shift_matrix_id"]
            run1_service_time         = jsonWorkerResponse["service_time"]
            run1_n_iterations         = jsonWorkerResponse["n_iterations"]
            label_vector              = jsonWorkerResponse["label_vector"]

            logger.info({
                "event":"SKMEANS.RUN1",
                "plaintext_matrix_id":plaintext_matrix_id,
                "step_index":"1",
                "clustering_status":str(status),
                "request_id":requestId,
                "encrypted_matrix_id":encrypted_matrix_id,
                "encrypted_matrix_shape":"({},{},{})".format(r,a,m),
                "encrypted_matrix_dtype":"float64", 
                "num_chunks":num_chunks,
                "iterations":iterations,
                "k":k, 
                "m":m, 
                "experiment_iteration":experiment_iteration,
                "max_iterations":MAX_ITERATIONS,
                "status":worker_run1_status,
                "worker_service_time": run1_service_time,
                "n_iterations":run1_n_iterations,
                "response_time":time.time() - inner_interaction_arrival_time
            })
            
            logger.debug({
                "event":"GET.MATRIX.BEFORE",
                "key":encrypted_shift_matrix_id,
                "bucket_id":BUCKET_ID
            })
            encrypted_shift_matrix_start_time = time.time()
            encryptedShiftMatrix_get_response = Utils.get_matrix_or_error(
                client    = STORAGE_CLIENT, 
                key       = encrypted_shift_matrix_id,
                bucket_id = BUCKET_ID
            )
            logger.info({
                "event":"GET.MATRIX",
                "key":encrypted_shift_matrix_id,
                "bucket_id":BUCKET_ID,
                "service_time": time.time() - encrypted_shift_matrix_start_time
            })
            
            encrypted_shift_matrix = encryptedShiftMatrix_get_response.value
            # time.sleep(50)
            logger.debug({
                "event":"DECRYPT.BEFORE",
                "m":m,
                "shape":str(encrypted_shift_matrix.shape),
                "dtype":str(encrypted_shift_matrix.dtype)
            })

            decrypt_start_time = time.time()
            shiftMatrix_chipher_schema_res = liu.decryptMatrix( #Shift Matrix is decrypted
                ciphertext_matrix = encrypted_shift_matrix.tolist(),
                secret_key        = dataowner.sk,
                m                 = int(m)
            )
            logger.info({
                "event":"DECRYPT",
                "m":m,
                "shape":str(encrypted_shift_matrix.shape),
                "dtype":str(encrypted_shift_matrix.dtype),
                "service_time":time.time() - decrypt_start_time
            })

            shift_matrix    = shiftMatrix_chipher_schema_res.matrix
            shift_matrix_id = "{}shiftmatrix".format(plaintext_matrix_id) # The id of the Shift matrix is formed

            
            logger.debug({
                "event":"PUT.NDARRAY.BEFORE",
                "key":shift_matrix_id,
                "shape":str(shift_matrix.shape),
                "dtype":str(shift_matrix.dtype)
            })
            put_shift_matrix_start_time     = time.time()
            _ = STORAGE_CLIENT.delete_by_ball_id(ball_id=shift_matrix_id, bucket_id=BUCKET_ID)
            # print("DEL_RESULT",del_resul)
            # if del_resul.is_err:
            #     return Response(response="Error deleting {}".format(shift_matrix_id))
            # print("DEL_RESULT",del_resul)
            _:Result[PutResponse,Exception] = STORAGE_CLIENT.put_ndarray(
                key       = shift_matrix_id,
                ndarray   = shift_matrix,
                tags      = {},
                bucket_id = BUCKET_ID
            ).result() #Shift matrix is saved to the storage system
            
            logger.info({
                "event":"PUT.NDARRAY",
                "key":shift_matrix_id,
                "shape":str(shift_matrix.shape),
                "dtype":str(shift_matrix.dtype),
                "service_time":time.time() -  put_shift_matrix_start_time

            })

            status       = Constants.ClusteringStatus.WORK_IN_PROGRESS #Status is updated
            run2_headers = {
                    "Step-Index"             : "2",
                    "Clustering-Status"      : str(status),
                    "Shift-Matrix-Id"        : shift_matrix_id,
                    "Plaintext-Matrix-Id"    : plaintext_matrix_id,
                    "Encrypted-Matrix-Id"    :encrypted_matrix_id,
                    "Encrypted-Matrix-Shape" : "({},{},{})".format(r,a,m),
                    "Encrypted-Matrix-Dtype" : "float64",
                    "Num-Chunks"             : str(num_chunks),
                    "Iterations"             :str(iterations),
                    "K":str(k),
                    "M":str(m), 
                    "Experiment-Iteration": str(experiment_iteration), 
                    "Max-Iterations":str(MAX_ITERATIONS) 
            }
            
            worker_run2_response      = worker.run(
                timeout = WORKER_TIMEOUT,
                headers = run2_headers
            ) #Start run 2
            worker_run2_response.raise_for_status()
            service_time_worker = worker_run2_response.headers.get("Service-Time",0) 
            iterations+=1
            if (iterations >= MAX_ITERATIONS): #If the number of iterations is equal to the maximum
                status              = Constants.ClusteringStatus.COMPLETED #Change the status to complete
                startTime           = float(s.headers.get("Start-Time",0))
                service_time_worker = time.time() - startTime #The service time is calculated
            else: 
                status = int(worker_run2_response.headers.get("Clustering-Status",Constants.ClusteringStatus.WORK_IN_PROGRESS)) #Status is maintained
            endTime    = time.time() # Get the time when it ends
            inner_interaction_service_time   = endTime - inner_interaction_arrival_time
            
            logger.info({
                "event":"SKMEANS.ITERATION.COMPLETED",
                "plaintext_matrix_id"   :plaintext_matrix_id,
                "worker_id":worker_id,
                "k":k,
                "m":m,
                "iterations":iterations,
                "service_time":inner_interaction_service_time,
            })

        interaction_end_time     = time.time()
        interaction_service_time = interaction_end_time - interaction_arrival_time 
        response_time  = endTime - arrivalTime 
        logger.info({
            "event":"SKMEANS.COMPLETED",
            "plaintext_matrix_id":plaintext_matrix_id,
            "encrypted_matrix_id":encrypted_matrix_id,
            "num_chunks":num_chunks,
            "algorithm":algorithm,
            "worker_id":worker_id,
            "k":k,
            "m":m,
            "n_iterations":iterations, 
            "max_iterations":MAX_ITERATIONS,
            "iterations_service_time":interaction_service_time,
            "response_time":response_time,
        })

        return Response(
            response = json.dumps({
                "label_vector" : label_vector,
                # "service_time" : service_time_worker,
                "response_time": response_time,
                "algorithm"   : algorithm
            }),
            status   = 200,
            headers  = {}
        )
    except Exception as e:
        logger.error({
            "msg":str(e)
        })
        return Response(response= str(e) , status= 500)
    

@clustering.route("/dbskmeans", methods = ["POST"])
def dbskmeans():
    try:
        # CONFIG
        local_start_time             = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        TESTING                      = current_app.config.get("TESTING",True)
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        liu:Liu                      = current_app.config.get("liu")
        dataowner:DataOwner          = current_app.config.get("dataowner")
        STORAGE_CLIENT:V4Client      = current_app.config.get("STORAGE_CLIENT")
        max_workers                  = int(current_app.config.get("MAX_WORKERS",2))
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        _num_chunks                  = current_app.config.get("NUM_CHUNKS",4)
        np_random                    = current_app.config.get("np_random")
        # 
        if executor               == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm                 = Constants.ClusteringAlgorithms.DBSKMEANS
        s                         = Session()
        # HEADERS
        request_headers           = request.headers #Headers for the request
        num_chunks                = int(request_headers.get("Num-Chunks",_num_chunks ) )
        plaintext_matrix_id       = request_headers.get("Plaintext-Matrix-Id","matrix0")
        encrypted_matrix_id       = "encrypted{}".format(plaintext_matrix_id) # The id of the encrypted matrix is built
        encrypted_udm_id          = "{}encryptedudm".format(plaintext_matrix_id) # The iudm id is built
        plaintext_matrix_filename = request_headers.get("Plaintext-Matrix-Filename","matrix-0")
        extension                 = request_headers.get("Extension","csv")
        # m                         = request_headers.get("M","3")
        m                         = dataowner.m
        k                         = request_headers.get("K","3")
        sens                      = float(request_headers.get("Sens","0.00000001"))
        experiment_iteration      = request_headers.get("Experiment-Iteration","0")
        MAX_ITERATIONS            = int(request_headers.get("Max-Iterations",current_app.config.get("MAX_ITERATIONS",10)))
        WORKER_TIMEOUT            = int(current_app.config.get("WORKER_TIMEOUT",300))
        request_id                = "request-{}".format(plaintext_matrix_id)
        plaintext_matrix_path     = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_filename, extension)
        iterations                = 0
        logger.debug({
            "event":"DBSKMEANS.STARTED",
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "encrypted_matrix_id":encrypted_matrix_id,
            "plaintext_matrix_filename":plaintext_matrix_filename,
            "plaintext_matrix_path":plaintext_matrix_path,
            "m":m,
            "k":k,
            "num_chunks":num_chunks,
            "max_workers":max_workers,
            "bucket_id":BUCKET_ID,
            "testing":TESTING,
            "experiment_iteration":experiment_iteration,
            "max_iterations":MAX_ITERATIONS,
            "request_id":request_id,
            "worker_timeout":WORKER_TIMEOUT,
            "source_path":SOURCE_PATH,
        })

        logger.debug({
            "event":"LOCAL.READ.DATASET.BEFORE",
            "path":plaintext_matrix_path,
            "key":plaintext_matrix_id,
            "filename":plaintext_matrix_filename,
            "algorithm":algorithm,
        })

        local_read_dataset_start_time = time.time()
        plaintext_matrix_result       = Utils.read_numpy_from(
            client    = STORAGE_CLIENT,
            path      = plaintext_matrix_path,
            extension = extension,
        )
        if plaintext_matrix_result.is_ok:
            plaintext_matrix = plaintext_matrix_result.unwrap()
        else:
            raise plaintext_matrix_result.unwrap_err()

        local_read_dataset_st = time.time() - local_read_dataset_start_time
        
        r = plaintext_matrix.shape[0]
        a = plaintext_matrix.shape[1]
        logger.debug({
            "event":"LOCAL.READ.DATASET",
            "path":plaintext_matrix_path,
            "key":plaintext_matrix_id,
            "filename":plaintext_matrix_filename,
            "algorithm":algorithm,
            "records":r,
            "attributes":a,
            "service_time":local_read_dataset_st
        })
        cores       = os.cpu_count()
        max_workers = num_chunks if max_workers > num_chunks else max_workers
        max_workers = cores if max_workers > cores else max_workers
        encryption_start_time = time.time()
        
        encrypt_segment_start_time = time.time()
        n = a*r*int(m)
        logger.debug({
            "event":"SEGMENT_ENCRYPT.BEFORE",
            "key":encrypted_matrix_id,
            "plaintext_matrix_shape":str(plaintext_matrix.shape),
            "plaintext_matrix_dtype":str(plaintext_matrix.dtype),
            "algorithm":algorithm,
            "n":n,
            "num_chunks":num_chunks,
            "max_workers":max_workers
        })
        encrypted_matrix_chunks = Utils.segment_and_encrypt_liu_with_executor( #Encrypt 
            key              = encrypted_matrix_id,
            plaintext_matrix = plaintext_matrix,
            dataowner        = dataowner,
            n                = n,
            num_chunks       = num_chunks,
            max_workers      = max_workers,
            executor         = executor,
            np_random        = np_random
        )
        encrypt_segment_service_time = time.time() - encrypt_segment_start_time
        logger.info({
            "event":"SEGMENT.ENCRYPT",
            "key":encrypted_matrix_id,
            "plaintext_matrix_shape":str(plaintext_matrix.shape),
            "plaintext_matrix_dtype":str(plaintext_matrix.dtype),
            "algorithm":algorithm,
            "n":n,
            "num_chunks":num_chunks,
            "max_workers":max_workers,
            "service_time":encrypt_segment_service_time
        })
        
        logger.debug({
            "event":"PUT.CHUNKS.BEFORE",
            "key":encrypted_matrix_id,
            "algorithm":algorithm,
            "num_chunks":num_chunks
        })
        put_chunks_start_time = time.time()
        STORAGE_CLIENT.delete_by_ball_id(
            ball_id   = encrypted_matrix_id, 
            bucket_id = BUCKET_ID
        )
        # put_chunks_generator_results = STORAGE_CLIENT.put_chunks(
        #     bucket_id = BUCKET_ID,
        #     key       = encrypted_matrix_id, 
        #     chunks    = encrypted_matrix_chunks, 
        #     tags      = {}
        # )
        chunks_bytes = Utils.chunks_to_bytes_gen(
            chs = encrypted_matrix_chunks
        )

        put_chunks_generator_results = STORAGE_CLIENT.put_chunked(
            key       = encrypted_matrix_id, 
            chunks    = chunks_bytes, 
            bucket_id = BUCKET_ID,
            tags      = {
                "shape": str((r,a,m)),
                "dtype":"float64"
            }
        )
        
        put_chunks_st = time.time() - put_chunks_start_time
        logger.info({
            "event":"PUT.CHUNKS",
            "key":encrypted_matrix_id,
            "num_chunks":num_chunks,
            "algorithm":algorithm,
            "service_time":put_chunks_st
        })
        
        # for i,put_chunk_result in enumerate(put_chunks_generator_results):
            
        #     if put_chunk_result.is_err:
        #         error = put_chunk_result.unwrap_err()
        #         logger.error({
        #             "msg":"Something went wrong storage and encrypt the chunk. {}".format(error)
        #         })
        #         return Response(
        #             status   = 500,
        #             response = "{}".format(str(error))
        #         )
        encryption_end_time = time.time()
        encryption_time     = encryption_end_time - encryption_start_time
     
        logger.info({
            "event":"PUT.SEGMENT.ENCRYPT", 
            "plaintext_matrix_id":plaintext_matrix_id,
            "algorithm":algorithm,
            "service_time":encryption_time
        })
        logger.debug({
            "event":"UDM.GENERATION.BEFORE",
            "algorithm":algorithm,
            "plaintext_matrix_shape":str(plaintext_matrix.shape),
            "plaintext_matrix_dtype":str(plaintext_matrix.dtype),
        })
        udm_start_time = time.time()
        print("BEFORE GET_U")
        udm            = dataowner.get_U(
            plaintext_matrix = plaintext_matrix,
            algorithm        = algorithm
        )
        logger.info({
            "event":"UDM.GENERATION",
            "algorithm":algorithm,
            "plaintext_matrix_shape":str(plaintext_matrix.shape),
            "plaintext_matrix_dtype":str(plaintext_matrix.dtype),
            "service_time":time.time() - udm_start_time
        })

        n         = r*r*a*int(m)
        threshold = 0.0
        segment_encrypt_fdhope_start_time = time.time()
        logger.debug({
            "event":"SEGMENT.ENCRYPT.FDHOPE.BEFORE",
            "key":encrypted_udm_id,
            "udm_shape":str(udm.shape),
            "udm_dtype":str(udm.dtype),
            "n":n,
            "num_chunks":num_chunks,
            "max_workers":max_workers,
            "algorithm":algorithm,
            "threshold":threshold
        })

        encrypted_matrix_UDM_chunks = Utils.segment_and_encrypt_fdhope_with_executor( #Encrypt 
            key              = encrypted_udm_id,
            plaintext_matrix = udm,
            dataowner        = dataowner,
            n                = n,
            num_chunks       = num_chunks,
            algorithm        = algorithm,
            sens             = sens,
            executor         = executor
        )
        segment_encrypt_fdhope_st = time.time() - segment_encrypt_fdhope_start_time
        logger.debug({
            "event":"SEGMENT.ENCRYPT.FDHOPE.BEFORE",
            "key":encrypted_udm_id,
            "udm_shape":str(udm.shape),
            "udm_dtype":str(udm.dtype),
            "n":n,
            "num_chunks":num_chunks,
            "max_workers":max_workers,
            "algorithm":algorithm,
            "threshold":threshold,
            "service_time":segment_encrypt_fdhope_st
        })
        
        logger.debug({
            "event":"PUT.CHUNKS.BEFORE",
            "key":encrypted_udm_id,
            "num_chunks":num_chunks,
        })
        put_chunks_start_time = time.time()
        STORAGE_CLIENT.delete_by_ball_id(
            ball_id   = encrypted_udm_id,
            bucket_id = BUCKET_ID
        )

        # put_chunks_generator_results = STORAGE_CLIENT.put_chunks(
        #     bucket_id = BUCKET_ID,
        #     key       = encrypted_udm_id, 
        #     chunks    = encrypted_matrix_UDM_chunks, 
        #     tags      = {}
        # )

        # for i,put_chunk_result in enumerate(put_chunks_generator_results):
        #     if put_chunk_result.is_err:
        #         logger.error("Something went wrong storage and encrypt the chunk.")
        #         return Response(
        #             status   = 500,
        #             response = "{}".format(str(put_chunk_result.unwrap_err()))
        #         )
        chunks_udm_bytes = Utils.chunks_to_bytes_gen(
            chs = encrypted_matrix_UDM_chunks
        )

        # print("shape", str((r,r,a)))
        print("udm_shape",str(udm.shape))
        put_chunks_udm_generator_results = STORAGE_CLIENT.put_chunked(
            key       = encrypted_udm_id, 
            chunks    = chunks_udm_bytes, 
            bucket_id = BUCKET_ID,
            tags      = {
                "shape": str(udm.shape),
                "dtype":"float64"
            }
        )
        logger.info({
            "event":"PUT.CHUNKS",
            "key":encrypted_udm_id,
            "num_chunks":num_chunks,
            "service_time":time.time() - put_chunks_start_time
        })

        managerResponse:RoryManager = current_app.config.get("manager") # Communicates with the manager
        
        get_worker_start_time = time.time()
        get_worker_result     = managerResponse.getWorker( #Gets the worker from the manager
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

        get_worker_end_time     = time.time() 
        get_worker_service_time = get_worker_end_time - get_worker_start_time
        worker_id               = "localhost" if TESTING else _worker_id

        logger.info({
            "event":"MANAGER.GET.WORKER",
            "worker_id":_worker_id,
            "port":port,
            "algorithm":algorithm,
            "service_time":get_worker_service_time,
            "k":k,
            "m":m
        })

        worker         = RoryWorker( #Allows to establish the connection with the worker
            workerId   = worker_id,
            port       = port,
            session    = s,
            algorithm  = algorithm
        )
        status           = Constants.ClusteringStatus.START #Set the status to start
        worker_run2_response = None
        initial_udm_shape = (r,r,a)
        global_start_time = time.time()
        label_vector = []
        
        while (status != Constants.ClusteringStatus.COMPLETED): #While the status is not completed
            inner_interaction_start_time = time.time()
            
            run1_headers = {
                "Start-Time":str(global_start_time),
                "Step-Index"             : "1",
                "Clustering-Status"      : str(status),
                "Plaintext-Matrix-Id"    : plaintext_matrix_id,
                "Encrypted-Matrix-Id"    : encrypted_matrix_id,
                "Encrypted-Matrix-Shape" : "({},{},{})".format(r,a,m),
                "Encrypted-Matrix-Dtype" : "float64",
                "Encrypted-Udm-Shape"    : str(initial_udm_shape),
                "Encrypted-Udm-Dtype"    : "float64",
                "Num-Chunks"             : str(num_chunks),
                "Iterations"             : str(iterations),
                "K"                      : str(k),
                "M"                      : str(m), 
                "Experiment-Iteration"   : str(experiment_iteration), 
                "Max-Iterations"         : str(MAX_ITERATIONS) 
            }

            logger.debug({
                "event":"WORKER.RUN.1.BEFORE",
                "step_index":"1",
                "clustering_status":str(status),
                "plaintext_matrix_id":plaintext_matrix_id,
                "encrypted_matrix_id":encrypted_matrix_id,
                "encrypted_matrix_shape":"({},{},{})".format(r,a,m),
                "encrypted_matrix_dtype":"float64", 
                "encrypted_udm_shape":str(initial_udm_shape),
                "encrypted_udm_dtype":"float64",
                "num_chunks":num_chunks,
                "iterations":iterations,
                "k":k, 
                "m":m, 
                "experiment_iteration":experiment_iteration,
                "max_iterations":MAX_ITERATIONS,
            })
            workerResponse1 = worker.run(timeout = WORKER_TIMEOUT,headers =run1_headers) #Run 1 starts
            workerResponse1.raise_for_status()
            
            stringWorkerResponse = workerResponse1.content.decode("utf-8") #Response from worker
            jsonWorkerResponse   = json.loads(stringWorkerResponse) #pass to json
            
            encrypted_shift_matrix_id = jsonWorkerResponse["encrypted_shift_matrix_id"]
            run1_service_time         = jsonWorkerResponse['service_time']
            label_vector              = jsonWorkerResponse["label_vector"]
            run1_response_time        = time.time() - inner_interaction_start_time
            logger.info({
                "event":"WORKER.RUN.1",
                "step_index":"1",
                "clustering_status":str(status),
                "plaintext_matrix_id":plaintext_matrix_id,
                "encrypted_matrix_id":encrypted_matrix_id,
                "encrypted_matrix_shape":"({},{},{})".format(r,a,m),
                "encrypted_matrix_dtype":"float64", 
                "encrypted_udm_shape":str(initial_udm_shape),
                "encrypted_udm_dtype":"float64",
                "num_chunks":num_chunks,
                "iterations":iterations,
                "k":k, 
                "m":m, 
                "experiment_iteration":experiment_iteration,
                "max_iterations":MAX_ITERATIONS,
                "service_time":run1_service_time,
                "response_time":run1_response_time
            })
            logger.debug({
                "event":"GET.MATRIX.BEFORE",
                "key":encrypted_shift_matrix_id,
                "bucket_id":BUCKET_ID
            })
            get_matrix_start_time = time.time()
            encryptedShiftMatrix_get_response = Utils.get_matrix_or_error(
                bucket_id = BUCKET_ID,
                client    = STORAGE_CLIENT, 
                key       = encrypted_shift_matrix_id
            )
            get_matrix_st = time.time() - get_matrix_start_time
            encryptedShiftMatrix = encryptedShiftMatrix_get_response.value
            logger.info({
                "event":"GET.MATRIX",
                "key":encrypted_shift_matrix_id,
                "bucket_id":BUCKET_ID,
                "service_time":get_matrix_st
            })
            logger.debug({
                "event":"DECRYPT.BEFORE",
                "m":int(m),
                "encrypted_shift_matrix_shape":str(encryptedShiftMatrix.shape),
                "encrypted_shift_matrix_dtype":str(encryptedShiftMatrix.dtype),
            })
            decrypt_start_time = time.time()
            cipher_schema_res  = liu.decryptMatrix( #Shift Matrix is decrypted
                ciphertext_matrix = encryptedShiftMatrix.tolist(),
                secret_key        = dataowner.sk,
                m                 = int(m)
            )
            descrypy_st = time.time() - decrypt_start_time
            logger.info({
                "event":"DECRYPT",
                "m":int(m),
                "encrypted_shift_matrix_shape":str(encryptedShiftMatrix.shape),
                "encrypted_shift_matrix_dtype":str(encryptedShiftMatrix.dtype),
                "service_time":descrypy_st
            })
            
            logger.debug({
                "event":"ENCRYPT.FDHOPE.BEFORE",
                "m":int(m),
                "shift_matrix_shape":str(cipher_schema_res.matrix.shape),
                "shift_matrix_dtype":str(cipher_schema_res.matrix.dtype),
            })
            encrypted_start_time = time.time()
            shift_matrix_ope_res = Fdhope.encryptMatrix( #Re-encrypt shift matrix with the FDHOPE scheme
                plaintext_matrix = cipher_schema_res.matrix, 
                messagespace     = dataowner.messageIntervals,
                cipherspace      = dataowner.cypherIntervals
            )
            
            shift_matrix_ope = shift_matrix_ope_res.matrix
            logger.info({
                "event":"ENCRYPT.FDHOPE",
                "m":int(m),
                "shift_matrix_shape":str(shift_matrix_ope.shape),
                "shift_matrix_dtype":str(shift_matrix_ope.dtype),
                "service_time":time.time() - encrypted_start_time
            })
            shift_matrix_id     = "{}shiftmatrix".format(plaintext_matrix_id) # The id of the Shift matrix is formed
            shift_matrix_ope_id = "{}shiftmatrixope".format(plaintext_matrix_id) # The id of the Shift matrix is formed
            
            put_matrix_start_time = time.time()
            logger.debug({
                "event":"PUT.NDARRAY.BEFORE",
                "key":shift_matrix_ope_id,
                "bucket_id":BUCKET_ID,
                "shift_matrix_shape":str(shift_matrix_ope.shape),
                "shift_matrix_dtype":str(shift_matrix_ope.dtype)
            })
            yd = STORAGE_CLIENT.delete(
                key       = shift_matrix_ope_id,
                bucket_id = BUCKET_ID
            )
            y:Result[PutResponse,Exception]  = STORAGE_CLIENT.put_ndarray(
                key       = shift_matrix_ope_id,
                ndarray   = shift_matrix_ope,
                tags      = {},
                bucket_id = BUCKET_ID
            ).result()#.unwrap() #Shift matrix is saved to the storage system
            
            status = Constants.ClusteringStatus.WORK_IN_PROGRESS #Status is updated
            logger.info({
                "event":"PUT.NDARRAY",
                "key":shift_matrix_ope_id,
                "bucket_id":BUCKET_ID,
                "shift_matrix_shape":str(shift_matrix_ope.shape),
                "shift_matrix_dtype":str(shift_matrix_ope.dtype),
                "service_time":time.time() - put_matrix_start_time
            })
            run2_headers = {
                    "Start-Time":str(global_start_time),
                    "Step-Index"             : "2",
                    "Clustering-Status"      : str(status),
                    "Shift-Matrix-Id"        : shift_matrix_id,
                    "Shift-Matrix-Ope-Id"    : shift_matrix_ope_id,
                    "Plaintext-Matrix-Id"    : plaintext_matrix_id,
                    "Encrypted-Matrix-Id"    : encrypted_matrix_id,
                    "Encrypted-Matrix-Shape" : "({},{},{})".format(r,a,m),
                    "Encrypted-Matrix-Dtype" : "float64",
                    "Encrypted-Udm-Dtype"    : "float64",
                    "Encrypted-Udm-Shape"    : str(initial_udm_shape),
                    "Num-Chunks"             : str(num_chunks),
                    "Iterations"             : str(iterations),
                    "K"                      : str(k),
                    "M"                      : str(m), 
                    "Experiment-Iteration"   : str(experiment_iteration), 
                    "Max-Iterations"         : str(MAX_ITERATIONS) 
            }
            
            logger.debug({
                "event":"WORKER.RUN.2.BEFORE",
                "step_index":"2",
                "clustering_status":str(status),
                "plaintext_matrix_id":plaintext_matrix_id,
                "encrypted_matrix_id":encrypted_matrix_id,
                "shift_matrix_id":shift_matrix_id,
                "shift_matrix_ope_id":shift_matrix_ope_id,
                "encrypted_matrix_shape":"({},{},{})".format(r,a,m),
                "encrypted_matrix_dtype":"float64", 
                "encrypted_udm_shape":str(initial_udm_shape),
                "encrypted_udm_dtype":"float64",
                "num_chunks":num_chunks,
                "iterations":iterations,
                "k":k, 
                "m":m, 
                "experiment_iteration":experiment_iteration,
                "max_iterations":MAX_ITERATIONS,
            })
            run2_start_time = time.time()
            worker_run2_response = worker.run(
                timeout = WORKER_TIMEOUT,
                headers = run2_headers
            ) #Start run 2
            worker_run2_response.raise_for_status()
            str_run2_response  = worker_run2_response.content.decode("utf-8") #Response from worker
            run2_json          = json.loads(str_run2_response) #pass to json
            initial_udm_shape  = eval(run2_json["encrypted_udm_shape"])
            run2_service_time  = run2_json["service_time"]
            run2_response_time = time.time() - run2_start_time
            
            logger.info({
                "event":"WORKER.RUN.2",
                "step_index":"2",
                "clustering_status":str(status),
                "plaintext_matrix_id":plaintext_matrix_id,
                "encrypted_matrix_id":encrypted_matrix_id,
                "shift_matrix_id":shift_matrix_id,
                "shift_matrix_ope_id":shift_matrix_ope_id,
                "encrypted_matrix_shape":"({},{},{})".format(r,a,m),
                "encrypted_matrix_dtype":"float64", 
                "encrypted_udm_shape":str(initial_udm_shape),
                "encrypted_udm_dtype":"float64",
                "num_chunks":num_chunks,
                "iterations":iterations,
                "k":k, 
                "m":m, 
                "experiment_iteration":experiment_iteration,
                "max_iterations":MAX_ITERATIONS,
                "worker_service_time":run2_service_time,
                "response_time": run2_response_time
            })
            iterations+=1
            if (iterations >= MAX_ITERATIONS): #If the number of iterations is equal to the maximum
                status = Constants.ClusteringStatus.COMPLETED #Change the status to complete
            else: 
                status = int(worker_run2_response.headers.get("Clustering-Status",Constants.ClusteringStatus.WORK_IN_PROGRESS)) #Status is maintained
            end_time                       = time.time() # Get the time when it ends
            inner_interaction_service_time = end_time-inner_interaction_start_time
            logger.info({
                "event":"DBSKMEANS.ITERATION.COMPLETED",
                "plaintext_matrix_id" :plaintext_matrix_id,
                "worker_id":worker_id,
                "k":k,
                "m":m,
                "iterations":iterations,
                "run1_service_time":run1_service_time,
                "run1_response_time":run1_response_time,
                "run2_service_time":run2_service_time,
                "run2_response_time":run2_response_time,
                "iteration_service_time":inner_interaction_service_time,
            })

        interaction_end_time = time.time()
        service_time         = interaction_end_time - global_start_time
        response_time        = end_time - local_start_time 
        logger.info({
            "event":"DBSKMEANS.COMPLETED",
            "algorithm":algorithm,
            "worker_id":_worker_id,
            "k":k,
            "m":m,
            "plaintext_matrix_id":plaintext_matrix_id,
            "encrypted_matrix_id":encrypted_matrix_id,
            "service_time":service_time,
            "response_time":response_time
        })

        return Response(
            response = json.dumps({
                "label_vector" : label_vector,
                "service_time" : service_time,
                "response_time": response_time,
                "algorithm"    : algorithm,
                "worker_id"    : _worker_id,
                "n_iterations" : iterations
            }),
            status   = 200,
            headers  = {}
        )
    except Exception as e:
        logger.error({
            "msg":str(e)
        })
        return Response(response= str(e) , status= 500)
    

@clustering.route("/dbsnnc", methods      = ["POST"])
def dbsnnc():
    try:
        local_start_time             = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        TESTING                      = current_app.config.get("TESTING",True)
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        dataowner:DataOwner          = current_app.config.get("dataowner")
        STORAGE_CLIENT:V4Client      = current_app.config.get("STORAGE_CLIENT")
        _num_chunks                  = current_app.config.get("NUM_CHUNKS",4)
        max_workers                  = current_app.config.get("MAX_WORKERS",2)
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        np_random                    = current_app.config.get("np_random")
        if executor                  == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm                 = Constants.ClusteringAlgorithms.DBSNNC
        s                         = Session()
        request_headers           = request.headers #Headers for the request
        num_chunks                = int(request_headers.get("Num-Chunks",_num_chunks))
        plaintext_matrix_id       = request_headers.get("Plaintext-Matrix-Id","matrix0")
        encrypted_matrix_id       = "encrypted-{}".format(plaintext_matrix_id) # The id of the encrypted matrix is built
        dm_id                     = "{}dm".format(plaintext_matrix_id)
        encrypted_dm_id           = "{}encrypteddm".format(plaintext_matrix_id) # The iudm id is built
        plaintext_matrix_filename = request_headers.get("Plaintext-Matrix-Filename","matrix-0")
        extension                 = request_headers.get("Extension","csv")
        # m                         = int(request_headers.get("M","3"))
        m                         = dataowner.m
        sens                      = float(request_headers.get("Sens","0.00000001"))
        threshold                 = float(request_headers.get("Threshold",-1))
        request_id                = "request{}".format(plaintext_matrix_id)
        plaintext_matrix_path     = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_filename, extension)
        experiment_iteration      = request_headers.get("Experiment-Iteration","0")
        WORKER_TIMEOUT            = int(current_app.config.get("WORKER_TIMEOUT",300))
        logger.debug({
            "event":"DBSNNC.STARTED",
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "encrypted_matrix_id":encrypted_matrix_id,
            "plaintext_matrix_filename":plaintext_matrix_filename,
            "extension":extension,
            "plaintext_matrix_path":plaintext_matrix_path,
            "m":m,
            "num_chunks":num_chunks,
            "max_workers":max_workers,
            "bucket_id":BUCKET_ID,
            "testing":TESTING,
            "experiment_iteration":experiment_iteration,
            "request_id":request_id,
            "worker_timeout":WORKER_TIMEOUT,
            "source_path":SOURCE_PATH,
        })

        logger.debug({
            "event":"LOCAL.READ.DATASET.BEFORE",
            "path":plaintext_matrix_path,
            "key":plaintext_matrix_id,
            "filename":plaintext_matrix_filename,
            "algorithm":algorithm
        })

        local_read_dataset_start_time = time.time()
        plaintext_matrix_result  = Utils.read_numpy_from(
            client    = STORAGE_CLIENT,
            path      = plaintext_matrix_path,
            extension = extension,
        )
        if plaintext_matrix_result.is_ok:
            plaintext_matrix = plaintext_matrix_result.unwrap()
        else:
            raise plaintext_matrix_result.unwrap_err()

        local_read_dataset_st = time.time() - local_read_dataset_start_time
        
        r = plaintext_matrix.shape[0]
        a = plaintext_matrix.shape[1]
        logger.debug({
            "event":"LOCAL.READ.DATASET",
            "path":plaintext_matrix_path,
            "key":plaintext_matrix_id,
            "filename":plaintext_matrix_filename,
            "algorithm":algorithm,
            "records":r,
            "attributes":a,
            "service_time":local_read_dataset_st
        })
        
        cores       = os.cpu_count()
        max_workers = num_chunks if max_workers > num_chunks else max_workers
        max_workers = cores if max_workers > cores else max_workers
        encryption_start_time = time.time()

        n = r*a*m
        logger.debug({
            "event":"SEGMENT.ENCRYPT.LIU.BEFORE",
            "max_workers": max_workers,
            "plaintext_matrix_id":plaintext_matrix_id,
            "encrypted_matrix_id":encrypted_matrix_id,
            "num_chunks":num_chunks,
            "n":n
        })
        segment_encrypt_start_time = time.time()
        encrypted_matrix_chunks = Utils.segment_and_encrypt_liu_with_executor( #Encrypt 
            executor         = executor,
            key              = encrypted_matrix_id,
            plaintext_matrix = plaintext_matrix,
            dataowner        = dataowner,
            n                = n,
            num_chunks       = num_chunks,
            max_workers      = max_workers,
            np_random        = np_random
        )
        segment_encrypt_st = time.time() - segment_encrypt_start_time
        logger.info({
            "event":"SEGMENT.ENCRYPT.LIU",
            "max_workers": max_workers,
            "plaintext_matrix_id":plaintext_matrix_id,
            "encrypted_matrix_id":encrypted_matrix_id,
            "num_chunks":num_chunks,
            "n":n,
            "service_time":segment_encrypt_st
        })
        
        logger.debug({
            "event":"PUT.CHUNKS.BEFORE",
            "key":encrypted_matrix_id,
            "algorithm":algorithm,
            "num_chunks":num_chunks
        })
        put_chunks_start_time = time.time()
        STORAGE_CLIENT.delete_by_ball_id(
            ball_id   = encrypted_matrix_id, 
            bucket_id = BUCKET_ID
        )
        # put_chunks_generator_results = STORAGE_CLIENT.put_chunks(
        #     key       = encrypted_matrix_id, 
        #     chunks    = encrypted_matrix_chunks, 
        #     bucket_id = BUCKET_ID,
        #     tags      = {}
        # )
        chunks_bytes = Utils.chunks_to_bytes_gen(
            chs = encrypted_matrix_chunks
        )

        put_chunks_generator_results = STORAGE_CLIENT.put_chunked(
            key       = encrypted_matrix_id, 
            chunks    = chunks_bytes, 
            bucket_id = BUCKET_ID,
            tags      = {
                "shape": str((r,a,m)),
                "dtype":"float64"
            }
        )

        put_chunks_st = time.time() - put_chunks_start_time
        logger.info({
            "event":"PUT.CHUNKS",
            "key":encrypted_matrix_id,
            "num_chunks":num_chunks,
            "algorithm":algorithm,
            "service_time":put_chunks_st
        })
        
        # for i,put_chunk_result in enumerate(put_chunks_generator_results):
        #     if put_chunk_result.is_err:
        #         logger.error("Something went wrong storage and encrypt the chunk.")
        #         return Response(
        #             status   = 500,
        #             response = "{}".format(str(put_chunk_result.unwrap_err()))
        #         )
        encryption_time = time.time() - encryption_start_time
        logger.info({
            "event":"PUT.SEGMENT.ENCRYPT", 
            "plaintext_matrix_id":plaintext_matrix_id,
            "encrypted_matrix_id":encrypted_matrix_id,
            "algorithm":algorithm,
            "service_time":encryption_time
        })

        logger.debug({
            "event":"DM.GENERATION.BEFORE",
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "plaintext_matrix_shape":str(plaintext_matrix.shape),
            "plaintext_matrix_dtype":str(plaintext_matrix.dtype),
        })
        dm_start_time = time.time()
        dm = dataowner.get_U(
            plaintext_matrix = plaintext_matrix,
            algorithm        = algorithm
        )
        
        generate_dm_st = time.time() - dm_start_time
        logger.info({
            "event":"DM.GENERATION",
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "plaintext_matrix_shape":str(plaintext_matrix.shape),
            "plaintext_matrix_dtype":str(plaintext_matrix.dtype),
            "service_time":generate_dm_st
        })

        if threshold==-1:
            threshold = RoryUtils.get_threshold(
                distance_matrix = dm
            )
        
        logger.debug({
            "event":"THRESHOLD.GENERATE",
            "key":dm_id,
            "bucket_id":BUCKET_ID,
            "threshold":threshold,
        })
        
        n = r*r
        logger.debug({
            "event":"SEGMENT.ENCRYPT.FDHOPE.BEFORE",
            "algorithm":algorithm,
            "encrypted_dm_id":encrypted_dm_id,
            "n":n,
            "num_chunks":num_chunks,
            "max_wokers":max_workers,
            "threshold":threshold,
            "dm_shape":str(dm.shape),
            "dm_dtype":str(dm.dtype),
        })
        segment_encrypt_fdhope_start_time = time.time()
        encrypted_matrix_DM_chunks = Utils.segment_and_encrypt_fdhope_with_executor( #Encrypt 
            key              = encrypted_dm_id,
            plaintext_matrix = dm,
            dataowner        = dataowner,
            n                = n,
            num_chunks       = num_chunks,
            algorithm        = algorithm,
            sens             = sens,
            executor         = executor
        )
        segment_encrypt_fdhope_st = time.time() - segment_encrypt_fdhope_start_time
        logger.info({
            "event":"SEGMENT.ENCRYPT.FDHOPE",
            "algorithm":algorithm,
            "encrypted_dm_id":encrypted_dm_id,
            "n":n,
            "num_chunks":num_chunks,
            "max_wokers":max_workers,
            "threshold":threshold,
            "dm_shape":str(dm.shape),
            "dm_dtype":str(dm.dtype),
            "service_time":segment_encrypt_fdhope_st
        })
        
        logger.debug({
            "event":"PUT.CHUNKS.BEFORE",
            "key":encrypted_dm_id,
            "algorithm":algorithm,
            "num_chunks":num_chunks
        })
        put_chunks_start_time = time.time()
        STORAGE_CLIENT.delete_by_ball_id(
            ball_id=encrypted_dm_id, 
            bucket_id=BUCKET_ID
        )
        # put_chunks_generator_results = STORAGE_CLIENT.put_chunks(
        #     bucket_id = BUCKET_ID,
        #     key       = encrypted_dm_id, 
        #     chunks    = encrypted_matrix_DM_chunks, 
        #     tags      = {}
        # )
        chunks_dm_bytes = Utils.chunks_to_bytes_gen(
            chs = encrypted_matrix_DM_chunks
        )

        put_chunks_dm_generator_results = STORAGE_CLIENT.put_chunked(
            key       = encrypted_dm_id, 
            chunks    = chunks_dm_bytes, 
            bucket_id = BUCKET_ID,
            tags      = {
                "shape": str((r,r)), ##r,r
                "dtype":"float64"
            }
        )

        put_chunks_st = time.time() - put_chunks_start_time
        logger.info({
            "event":"PUT.CHUNKS",
            "key":encrypted_matrix_id,
            "num_chunks":num_chunks,
            "algorithm":algorithm,
            "service_time":put_chunks_st
        })

        # for i,put_chunk_result in enumerate(put_chunks_generator_results):
        #     if put_chunk_result.is_err:
        #         logger.error({
        #             "msg":str(put_chunk_result.unwrap_err())
        #         })
        #         return Response(
        #             status   = 500,
        #             response = "{}".format(str(put_chunk_result.unwrap_err()))
        #         )


        segment_encrypt_fdhope_st = time.time() - segment_encrypt_start_time
        logger.info({
            "event":"PUT.SEGMENT.ENCRYPT", 
            "plaintext_matrix_id":plaintext_matrix_id,
            "encrypted_dm_id":encrypted_dm_id,
            "algorithm":algorithm,
            "service_time":segment_encrypt_fdhope_st
        })
        # encrypted_threshold = dataowner.encrypted_threshold
        logger.debug({
            "event":"ENCRYPTED.THRESHOLD.BEFORE", 
            "plaintext_matrix_id":plaintext_matrix_id,
            "threshold":threshold,
            "algorithm":algorithm,
        })
        encrypted_threshold = Fdhope.encrypt( #Threshold is encrypted
				plaintext    = threshold,
				messagespace = dataowner.messageIntervals, 
				cipherspace  = dataowner.cypherIntervals,
                sens = sens,
			)
        logger.debug({
            "event":"ENCRYPTED.THRESHOLD", 
            "plaintext_matrix_id":plaintext_matrix_id,
            "threshold":threshold,
            "encrypted_threshold":encrypted_threshold,
            "algorithm":algorithm,
        })
        managerResponse:RoryManager = current_app.config.get("manager") # Communicates with the manager
        
        get_worker_start_time = time.time()
        get_worker_result     = managerResponse.getWorker( #Gets the worker from the manager
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

        get_worker_end_time     = time.time() 
        get_worker_service_time = get_worker_end_time - get_worker_start_time
        worker_id               = "localhost" if TESTING else _worker_id

        logger.info({
            "event":"MANAGER.GET.WORKER",
            "algorithm":algorithm,
            "worker_id":_worker_id,
            "port":port,
            "m":m,
            "service_time":get_worker_service_time
        })

        worker = RoryWorker( #Allows to establish the connection with the worker
            workerId  = worker_id,
            port      = port,
            session   = s,
            algorithm = algorithm
        )
        dm_shape = (r,r)

        encrypted_matrix_shape = (r,a,m)
        encrypted_matrix_dtype = "float64"
        run_headers = {
            "Plaintext-Matrix-Id"    : plaintext_matrix_id,
            "Request-Id"             : request_id,
            "Encrypted-Matrix-Id"    : encrypted_matrix_id,
            "Encrypted-Matrix-Shape" : str(encrypted_matrix_shape),
            "Encrypted-Matrix-Dtype" : encrypted_matrix_dtype,
            "Encrypted-Dm-Id"        : encrypted_dm_id,
            "Encrypted-Dm-Shape"     : str(dm_shape),
            "Encrypted-Dm-Dtype"     : "float64",
            "Num-Chunks"             : str(num_chunks),
            "M"                      : str(m),
            "Encrypted-Threshold"    : str(encrypted_threshold),
            "Dm-Shape"               : str(dm_shape),
            "Dm-Dtype"               : "float64",
        }

        logger.debug({
            "event":"DBSNNC.BEFORE",
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "plaintext_matrix_shape":str(plaintext_matrix.shape),
            "plaintext_matrix_dtype":str(plaintext_matrix.dtype),
            "encrypted_matrix_id":encrypted_matrix_id,
            "encrypted_matrix_shape":str(encrypted_matrix_shape),
            "encrypted_matrix_dtype":encrypted_matrix_dtype,
            "encrypted_dm_id":encrypted_dm_id,
            "threshold":threshold,
            "encrypted_threshold":encrypted_threshold,
            "m":m,
            "num_chunks":num_chunks,
        })
        run1_response = worker.run(
            timeout = WORKER_TIMEOUT, 
            headers = run_headers
        )
        run1_response.raise_for_status()
        
        stringWorkerResponse = run1_response.content.decode("utf-8") #Response from worker
        jsonWorkerResponse   = json.loads(stringWorkerResponse) #pass to json
        endTime              = time.time() # Get the time when it ends
        worker_service_time  = jsonWorkerResponse["service_time"]
        label_vector         = jsonWorkerResponse["label_vector"]
        client_service_time  = endTime - local_start_time # Get the service time
        logger.info({
            "event":"DBSNNC.COMPLETED",
            "plaintext_matrix_id":plaintext_matrix_id,
            "encrypted_matrix_id":encrypted_matrix_id,
            "encrypted_distance_matrix_id":encrypted_dm_id,
            "num_chunks":num_chunks,
            "m":m,
            "threshold":threshold,
            "worker_id":_worker_id,
            "worker_service_time":worker_service_time,
            "service_time":client_service_time,
        })
        return Response(
            response = json.dumps({
                "label_vector" :label_vector,
                "worker_id"    :_worker_id,
                "algorithm"    : algorithm,
                "worker_service_time" : worker_service_time,
                "service_time": str(client_service_time),
                
            }),
            status   = 200,
            headers  = {}
        )
    except Exception as e:
        logger.error({
            "msg":str(e)
        })
        return Response(response =None, status= 500, headers={"Error-Message":str(e)})
    

@clustering.route("/nnc", methods = ["POST"])
def nnc():
    try:
        local_start_time             = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        TESTING                      = current_app.config.get("TESTING",True)
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        dataowner:DataOwner          = current_app.config.get("dataowner")
        STORAGE_CLIENT:V4Client      = current_app.config.get("STORAGE_CLIENT")
        # _num_chunks                  = current_app.config.get("NUM_CHUNKS",1)
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        if executor == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm                 = Constants.ClusteringAlgorithms.NNC
        s                         = Session()
        request_headers           = request.headers #Headers for the request
        num_chunks                = int(request_headers.get("Num-Chunks",1))
        plaintext_matrix_id       = request_headers.get("Plaintext-Matrix-Id","matrix0")
        dm_id                     = "{}dm".format(plaintext_matrix_id)
        plaintext_matrix_filename = request_headers.get("Plaintext-Matrix-Filename","matrix-0")
        extension                 = request_headers.get("Extension","csv")
        request_id                = "request-{}".format(plaintext_matrix_id)
        threshold                 = float(request_headers.get("Threshold",-1))
        plaintext_matrix_path     = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_filename, extension)
        WORKER_TIMEOUT            = int(current_app.config.get("WORKER_TIMEOUT",300))
        
        logger.debug({
            "event":"NNC.STARTED",
            "algorithm":algorithm,
            "num_chunks":num_chunks,
            "plaintext_matrix_id":plaintext_matrix_id,
            "plaintext_matrix_filename":plaintext_matrix_filename,
            "plaintext_matrix_path":plaintext_matrix_path,
            "extension":extension,
            "dm_id":dm_id
        })

        logger.debug({
            "event":"LOCAL.READ.DATASET.BEFORE",
            "path":plaintext_matrix_path,
            "key":plaintext_matrix_id,
            "filename":plaintext_matrix_filename,
            "algorithm":algorithm
        })

        local_read_dataset_start_time = time.time()
        plaintext_matrix_result = Utils.read_numpy_from( 
            client    = STORAGE_CLIENT,
            path      = plaintext_matrix_path,
            extension = extension,
        )

        if plaintext_matrix_result.is_ok:
            plaintext_matrix = plaintext_matrix_result.unwrap()
        else:
            raise plaintext_matrix_result.unwrap_err()

        local_read_dataset_st = time.time() - local_read_dataset_start_time
        
        r = plaintext_matrix.shape[0]
        a = plaintext_matrix.shape[1]
        logger.debug({
            "event":"LOCAL.READ.DATASET",
            "path":plaintext_matrix_path,
            "key":plaintext_matrix_id,
            "filename":plaintext_matrix_filename,
            "algorithm":algorithm,
            "records":r,
            "attributes":a,
            "service_time":local_read_dataset_st
        })
        
        logger.debug({
            "event":"PUT.CHUNKS.BEFORE",
            "key":plaintext_matrix_id,
            "bucket_id":BUCKET_ID,
            "plaintext_matrix_shape":str(plaintext_matrix.shape),
            "plaintext_matrix_dtype":str(plaintext_matrix.dtype),
        })
        put_ptm_start_time = time.time()

        plaintext_matrix_chunks = Chunks.from_ndarray(
            ndarray   = plaintext_matrix,
            group_id  = plaintext_matrix_id,
            chunk_prefix = Some(plaintext_matrix_id),
            num_chunks= num_chunks,
        )

        if plaintext_matrix_chunks.is_none:
            raise "something went wrong creating the chunks"
        

        # _ = STORAGE_CLIENT.delete(key=plaintext_matrix_id, bucket_id=BUCKET_ID)
        _ = STORAGE_CLIENT.delete_by_ball_id(
            ball_id   = plaintext_matrix_id, 
            bucket_id = BUCKET_ID
        )
        
        # put_chunks_generator_results = STORAGE_CLIENT.put_chunks(
        #     key       = plaintext_matrix_id, 
        #     chunks    = plaintext_matrix_chunks.unwrap(), 
        #     bucket_id = BUCKET_ID,
        #     tags      = {}
        # )

        # for i,put_chunk_result in enumerate(put_chunks_generator_results):
        #     if put_chunk_result.is_err:
        #         logger.error("Something went wrong storage the chunk.")
        #         return Response(
        #             status   = 500,
        #             response = "{}".format(str(put_chunk_result.unwrap_err()))
        #         )
        chunks_bytes = Utils.chunks_to_bytes_gen(
            chs = plaintext_matrix_chunks.unwrap()
        )

        put_chunks_generator_results = STORAGE_CLIENT.put_chunked(
            key       = plaintext_matrix_id, 
            chunks    = chunks_bytes, 
            bucket_id = BUCKET_ID,
            tags      = {
                "shape": str(plaintext_matrix.shape),
                "dtype":"float64"
            }
        )
        # time.sleep(100)


        put_ptm_st = time.time() - put_ptm_start_time
        logger.info({
            "event":"PUT.CHUNKS",
            "key":plaintext_matrix_id,
            "bucket_id":BUCKET_ID,
            "plaintext_matrix_shape":str(plaintext_matrix.shape),
            "plaintext_matrix_dtype":str(plaintext_matrix.dtype),
            "service_time":put_ptm_st
        })

        logger.debug({
            "event":"DM.GENERATION.BEFORE",
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "plaintext_matrix_shape":str(plaintext_matrix.shape),
            "plaintext_matrix_dtype":str(plaintext_matrix.dtype),
        })
        dm_start_time = time.time()
        dm = dataowner.get_U(
            plaintext_matrix = plaintext_matrix,
            algorithm        = algorithm
        )
        generate_dm_st = time.time() - dm_start_time
        logger.info({
            "event":"DM.GENERATION",
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "plaintext_matrix_shape":str(plaintext_matrix.shape),
            "plaintext_matrix_dtype":str(plaintext_matrix.dtype),
            "service_time":generate_dm_st
        })

        if threshold==-1:
            threshold = RoryUtils.get_threshold(
                distance_matrix = dm
            )

        logger.debug({
            "event":"THRESHOLD.GENERATE",
            "key":dm_id,
            "bucket_id":BUCKET_ID,
            "threshold":threshold,
        })

        logger.debug({
            "event":"CHUNKS.FROM.NDARRAY.BEFORE",
            "key":dm_id,
            "bucket_id":BUCKET_ID,
            "dm_shape":str(dm.shape),
            "dm_dtype":str(dm.dtype),
        })
        put_ptm_start_time = time.time()
        dm_chunks = Chunks.from_ndarray(
            ndarray   = dm,
            group_id  = dm_id,
            chunk_prefix = Some(dm_id),
            num_chunks= num_chunks
        )

        if dm_chunks.is_none:
            raise "something went wrong creating the chunks"
        
        logger.debug({
            "event":"CHUNKS.FROM.NDARRAY",
            "dm_id":dm_id,
            "bucket_id":BUCKET_ID,
            "dm_shape":str(dm.shape),
            "dm_dtype":str(dm.dtype),
        })

        logger.debug({
            "event":"PUT.CHUNKS.BEFORE",
            "dm_id":dm_id,
            "bucket_id":BUCKET_ID,
            "dm_shape":str(dm.shape),
            "dm_dtype":str(dm.dtype),
        })

        STORAGE_CLIENT.delete_by_ball_id(
            ball_id   = dm_id, 
            bucket_id = BUCKET_ID
        )
        # put_chunks_generator_results = STORAGE_CLIENT.put_chunks(
        #     key       = dm_id, 
        #     chunks    = dm_chunks.unwrap(), 
        #     bucket_id = BUCKET_ID,
        #     tags      = {}
        # )

        # for i,put_chunk_result in enumerate(put_chunks_generator_results):
        #     if put_chunk_result.is_err:
        #         logger.error("Something went wrong storage the chunk.")
        #         return Response(
        #             status   = 500,
        #             response = "{}".format(str(put_chunk_result.unwrap_err()))
        #         )
        
        chunks_dm_bytes = Utils.chunks_to_bytes_gen(
            chs = dm_chunks.unwrap()
        )

        put_chunks_dm_generator_results = STORAGE_CLIENT.put_chunked(
            key       = dm_id, 
            chunks    = chunks_dm_bytes, 
            bucket_id = BUCKET_ID,
            tags      = {
                "shape": str(dm.shape),
                "dtype":"float64"
            }
        )

        put_dm_st = time.time() - put_ptm_start_time
        logger.info({
            "event":"PUT.CHUNKS",
            "key":dm_id,
            "bucket_id":BUCKET_ID,
            "shape":str(dm.shape),
            "dtype":str(dm.dtype),
            "service_time":put_dm_st
        })

        managerResponse:RoryManager = current_app.config.get("manager") # Communicates with the manager

        get_worker_start_time = time.time()
        get_worker_result     = managerResponse.getWorker( #Gets the worker from the manager
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

        get_worker_end_time     = time.time() 
        get_worker_service_time = get_worker_end_time - get_worker_start_time
        worker_id               =  "localhost" if TESTING else _worker_id

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
            algorithm  = algorithm
        )
        pm_shape = (r,a)
        dm_shape = (r,r)
        dm_type  = "float64"
        run_headers = {
            "Plaintext-Matrix-Id"    : plaintext_matrix_id,
            "Request-Id"             : request_id,
            "Num-Chunks"             : str(num_chunks),
            "Threshold"              : str(threshold),
            "Plaintext-Matrix-Shape" : str(pm_shape),
            "Plaintext-Matrix-Dtype" : "float64",
            "Dm-Shape"               : str(dm_shape),
            "Dm-Dtype"               : dm_type,
        }

        logger.debug({
            "event":"NNC.BEFORE",
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "threshold":threshold,
            "dm_shape":str(dm_shape),
            "dm_dtype":dm_type
        })
        worker_response  = worker.run(
            timeout = WORKER_TIMEOUT, 
            headers = run_headers
        )
        
        worker_response.raise_for_status()
        stringWorkerResponse = worker_response.content.decode("utf-8") #Response from worker
        jsonWorkerResponse   = json.loads(stringWorkerResponse) #pass to json
        end_time             = time.time() # Get the time when it ends
        worker_service_time  = jsonWorkerResponse["service_time"]
        label_vector         = jsonWorkerResponse["label_vector"]
        client_service_time  = end_time - local_start_time # Get the service time
        logger.info({
            "event":"NNC.COMPLETED",
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "worker_id":_worker_id,
            "worker_service_time":worker_service_time,
            "service_time":client_service_time
        })

        return Response(
            response = json.dumps({
                "label_vector" : label_vector,
                "worker_id"    :_worker_id,
                "algorithm"    : algorithm,
                "worker_service_time" : worker_service_time,
                "service_time": str(client_service_time),
            }),
            status   = 200,
            headers  = {}
        )
    except Exception as e:
        logger.error({
            "msg":str(e)
        })
        return Response(response =None, status= 500, headers={"Error-Message":str(e)})
 