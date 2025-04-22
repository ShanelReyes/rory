import os
import time, json
import numpy as np
from option import Some
from requests import Session
from flask import Blueprint,current_app,request,Response
from rory.core.interfaces.rorymanager import RoryManager
from rory.core.interfaces.roryworker import RoryWorker
from rory.core.security.dataowner import DataOwner
from rory.core.security.pqc.dataowner import DataOwner as DataOwnerPQC
from rory.core.security.cryptosystem.liu import Liu
from rory.core.security.cryptosystem.fdhope import Fdhope
from rory.core.security.cryptosystem.pqc.ckks import Ckks
from rory.core.utils.constants import Constants
from rory.core.utils.utils import Utils as RoryUtils
from mictlanx.v4.client import Client  as V4Client
from mictlanx import AsyncClient
from mictlanx.utils.segmentation import Chunks
from concurrent.futures import ProcessPoolExecutor
from utils.utils import Utils
from rorycommon import Common as RoryCommon
from uuid import uuid4
from models import ExperimentLogEntry
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
@clustering.route("/kmeans",methods = ["POST"])
async def kmeans():
    try:
        arrivalTime                = time.time()
        logger                     = current_app.config["logger"]
        TESTING                    = current_app.config.get("TESTING",True)
        SOURCE_PATH                = current_app.config["SOURCE_PATH"]
        STORAGE_CLIENT:AsyncClient = current_app.config.get("ASYNC_STORAGE_CLIENT")
        BUCKET_ID:str              = current_app.config.get("BUCKET_ID","rory")
        WORKER_TIMEOUT             = int(current_app.config.get("WORKER_TIMEOUT",300))
        algorithm                  = Constants.ClusteringAlgorithms.KMEANS
        s                          = Session()
        request_headers            = request.headers #Headers for the request
        num_chunks                 = int(request_headers.get("Num-Chunks",1))
        plaintext_matrix_id        = request_headers.get("Plaintext-Matrix-Id","matrix-0")
        plaintext_matrix_filename  = request_headers.get("Plaintext-Matrix-Filename","matrix-0")
        extension                  = request_headers.get("Extension","csv")
        k                          = request_headers.get("K","3")
        experiment_id              = request_headers.get("Experiment-Id",uuid4().hex[:10])
        plaintext_matrix_path      = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_filename, extension)

 
        read_dataset_start_time = time.time()
        plaintext_matrix_result = await RoryCommon.read_numpy_from(
            path      = plaintext_matrix_path,
            extension = extension
        )

        if plaintext_matrix_result.is_ok:
            plaintextMatrix = plaintext_matrix_result.unwrap()
        else:
            raise plaintext_matrix_result.unwrap_err()
  
        # read_dataset_st = time.time() - read_dataset_start_time
        
        local_read_entry = ExperimentLogEntry(
            event="LOCAL.READ", 
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=read_dataset_start_time,
            end_time= time.time(),
            id= plaintext_matrix_id,
            worker_id= "",
            num_chunks=num_chunks,
            k=k,
            workers= 0, 
        )
        logger.info(local_read_entry.model_dump())
        put_pm_start_time = time.time()
        put_ptm_result = await RoryCommon.put_ndarray(
            client=STORAGE_CLIENT, 
            key=plaintext_matrix_id, 
            matrix=plaintextMatrix,
            num_chunks=num_chunks, 
            tags={}, 
            bucket_id=BUCKET_ID
        )
        if put_ptm_result.is_err:
            error = put_ptm_result.unwrap_err()
            logger.error({
                "msg":str(error)
            })
            return Response(response=str(error), status=500)

        put_ptm_entry = ExperimentLogEntry(
            event="PUT",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=put_pm_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id="",
            num_chunks=num_chunks,
            k = k,
            workers=0,
        )
        logger.info(put_ptm_entry.model_dump())
        
        service_time_client = time.time() - arrivalTime

        get_worker_start_time = time.time()
        manager:RoryManager     = current_app.config.get("manager") # Communicates with the manager
        get_worker_result       = manager.getWorker( #Gets the worker from the manager
            headers = {
                "Algorithm"         : algorithm,
                "Start-Request-Time": str(arrivalTime)
            }
        )
        if get_worker_result.is_err:
            error = get_worker_result.unwrap_err()
            logger.error({
                "error":"GET.WORKER.FAILED",
                "message":str(error)
            })
            return Response(str(error), status=500)
        (worker_id, worker_port) = get_worker_result.unwrap()
        
        get_worker_end_time     = time.time()
        # get_worker_service_time = get_worker_end_time - get_worker_start_time 
        worker_id               = "localhost" if TESTING else worker_id

        get_worker_entry = ExperimentLogEntry(
            event="GET.WORKER",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=get_worker_start_time,
            end_time=get_worker_end_time,
            id=plaintext_matrix_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            k =k,
            workers=0
        )
        logger.info(get_worker_entry.model_dump())
        # raise Exception("BOOM!")
        worker_run_1_start_time = time.time()
        worker = RoryWorker( #Allows to establish the connection with the worker
            workerId  = worker_id,
            port      = worker_port,
            session   = s,
            algorithm = algorithm
        )   

        interaction_arrival_time = time.time()

        workerResponse = worker.run(
            headers = {
                "Plaintext-Matrix-Id": plaintext_matrix_id,
                "K": str(k),
                "Experiment-Id": experiment_id,
            },
            timeout = WORKER_TIMEOUT
        )
        worker_run_1_entry = ExperimentLogEntry(
            event="WORKER.RUN.1",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=worker_run_1_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            k=k,
            workers=0
        )
        logger.info(worker_run_1_entry.model_dump())

        # stringWorkerResponse = workerResponse.content.decode("utf-8") #Response from worker
        jsonWorkerResponse   = workerResponse.json()
        # json.loads(stringWorkerResponse) #pass to json
        # worker_service_time  =  jsonWorkerResponse["service_time"]
        iterations           = int(jsonWorkerResponse["iterations"]) # Extract the current number of iterations
        endTime              = time.time() # Get the time when it ends
        worker_response_time = endTime - worker_run_1_start_time
        response_time        = endTime - arrivalTime # Get the service time

        kmeans_completed_entry = ExperimentLogEntry(
            event="KMEANS.COMPLETED",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=worker_run_1_start_time,
            end_time=time.time(),
            id=plaintext_matrix_id,
            worker_id=worker_id,
            num_chunks=num_chunks,
            k=k,
            workers=0,
            iterations=iterations
        )
        
        logger.info(kmeans_completed_entry.model_dump())
        #     "event":"KMEANS.COMPLETED",
        #     "algorithm":algorithm,
        #     "plaintext_matrix_id":plaintext_matrix_id,
        #     "worker_service_time": worker_service_time,
        #     "worker_response_time":worker_response_time,
        #     "response_time":response_time,
        #     "iterations":iterations,
        #     "k":k,
        #     "service_time_manager":get_worker_service_time,
        # })

        return Response(
            response = json.dumps({
                "label_vector" : jsonWorkerResponse.get("label_vector",[]),
                "iterations":iterations,
                "algorithm":algorithm,
                "worker_id":worker_id,
                "service_time_manager":get_worker_entry.time,
                "service_time_worker":worker_response_time,
                "service_time_client":service_time_client,
                "response_time_clustering":response_time,
            }),
            status   = 200,
            headers  = {}
        )       
    except Exception as e:
        logger.error({
            "msg":str(e)
        })
        return Response(response = None, status= 500, headers = {"Error-Message":str(e)})

#SKMEANS
@clustering.route("/skmeans",methods = ["POST"])
async def skmeans():
    try:
        arrivalTime                  = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        TESTING                      = current_app.config.get("TESTING",True)
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        liu:Liu                      = current_app.config.get("liu")
        dataowner:DataOwner          = current_app.config.get("dataowner")
        STORAGE_CLIENT:AsyncClient      = current_app.config.get("ASYNC_STORAGE_CLIENT")
        _num_chunks                  = current_app.config.get("NUM_CHUNKS",4)
        max_workers                  = current_app.config.get("MAX_WORKERS",2)
        security_level                = current_app.config.get("LIU_SECURITY_LEVEL",128)
        np_random:bool                    = current_app.config.get("np_random")
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        
        if executor == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm               = Constants.ClusteringAlgorithms.SKMEANS
        s                       = Session()
        request_headers           = request.headers #Headers for the request
        num_chunks                = int(request_headers.get("Num-Chunks",_num_chunks))
        plaintext_matrix_id       = request_headers.get("Plaintext-Matrix-Id","matrix0")
        encrypted_matrix_id       = "encrypted{}".format(plaintext_matrix_id) # The id of the encrypted matrix is built
        udm_id                    = "{}udm".format(plaintext_matrix_id) # The iudm id is built
        plaintext_matrix_filename = request_headers.get("Plaintext-Matrix-Filename","matrix0")
        extension                 = request_headers.get("Extension","csv")
        experiment_id             = request_headers.get("Experiment-Id",uuid4().hex[:10])
        k                         = int(request_headers.get("K"))
        experiment_iteration      = request_headers.get("Experiment-Iteration","0")
        MAX_ITERATIONS            = int(request_headers.get("Max-Iterations",current_app.config.get("MAX_ITERATIONS",10)))
        WORKER_TIMEOUT            = int(current_app.config.get("WORKER_TIMEOUT",300))
        requestId                 = "request-{}".format(plaintext_matrix_id)
        m                         = dataowner.m
        plaintext_matrix_path     = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_filename, extension)
        cores       = os.cpu_count()
        max_workers = num_chunks if max_workers > num_chunks else max_workers
        max_workers = cores if max_workers > cores else max_workers
        
        # logger.debug({
        #     "event":"SKMEANS.STARTED",
        #     "algorithm":algorithm,
        #     "plaintext_matrix_id":plaintext_matrix_id,
        #     "encrypted_matrix_id":encrypted_matrix_id,
        #     "udm_id":udm_id,
        #     "plaintext_matrix_filename":plaintext_matrix_filename,
        #     "plaintext_matrix_path":plaintext_matrix_path,
        #     "security_level":securitylevel,
        #     "m":m,
        #     "k":k,
        #     "num_chunks":num_chunks,
        #     "max_workers":max_workers,
        #     "bucket_id":BUCKET_ID,
        #     "testing":TESTING,
        #     "experiment_iteration":experiment_iteration,
        #     "max_iterations":MAX_ITERATIONS,
        #     "request_id":requestId,
        #     "worker_timeout":WORKER_TIMEOUT,
        #     "source_path":SOURCE_PATH,
        # })
        

        local_read_dataset_start_time = time.time()
        plaintext_matrix_result  = await RoryCommon.read_numpy_from(
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
        local_read_entry = ExperimentLogEntry(
            event="LOCAL.READ",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=local_read_dataset_start_time,
            end_time= time.time(),
            id= plaintext_matrix_id,
            worker_id= "",
            num_chunks=num_chunks,
            k=k,
            workers= max_workers, 
            security_level=security_level,
            m = m
        )
        logger.info(local_read_entry.model_dump())
            # "event":"LOCAL.READ.DATASET",
            # "algorithm":algorithm,
            # "plaintext_matrix_id":plaintext_matrix_id,
            # "path":plaintext_matrix_path,
            # "filename":plaintext_matrix_filename,
            # "records":r,
            # "attributes":a,
            # "service_time":local_read_dataset_st
        # })


        n = a*r*m

        encryption_start_time = time.time()
        encrypted_ptm_chunks = RoryCommon.segment_and_encrypt_liu_with_executor( #Encrypt 
            executor         = executor,
            key              = encrypted_matrix_id,
            plaintext_matrix = plaintext_matrix,
            dataowner        = dataowner,
            n                = n,
            num_chunks       = num_chunks,
            np_random        = np_random
        )
        segment_encrypt_entry = ExperimentLogEntry(
            event="SEGMENT.ENCRYPT",
            experiment_id=experiment_id,
            algorithm=algorithm,
            start_time=encryption_start_time,
            end_time= time.time(),
            id= plaintext_matrix_id,
            worker_id= "",
            num_chunks=num_chunks,
            k=k,
            workers= max_workers, 
            security_level=security_level,
            m = m
        )
        logger.info(segment_encrypt_entry.model_dump())
        
        put_ptm_start_time = time.time()
        put_ptm_chunks_results = await RoryCommon.put_chunks(
            client = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = encrypted_matrix_id,
            chunks         = encrypted_ptm_chunks,
            tags= {
                "full_shape": str((r,a,m)),
                "full_dtype":"float64"

            }

        )
        if put_ptm_chunks_results.is_err:
            return Response(status=500, response = "Put encrypted matrix failed")
        put_encrypted_ptm_entry = ExperimentLogEntry(
            event          = "PUT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = put_ptm_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = "",
            num_chunks     = num_chunks,
            k              = k,
            workers        = max_workers,
            security_level = security_level,
            m              = m
        )
        logger.info(put_encrypted_ptm_entry.model_dump())

        udm_start_time = time.time()
        udm            = dataowner.get_U(
            plaintext_matrix = plaintext_matrix,
            algorithm        = algorithm
        )
        
        udm_gen_st = time.time()- udm_start_time
        udm_gen_entry = ExperimentLogEntry(
            event          = "UDM.GENERATION",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = udm_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = "",
            num_chunks     = num_chunks,
            k              = k,
            workers        = max_workers,
            security_level = security_level,
            m              = m
        )
        logger.info(udm_gen_entry.model_dump())
        #     "event":"UDM.GENERATION",
        #     "algorithm":algorithm,
        #     "plaintext_matrix_id":plaintext_matrix_id,
        #     "shape":str(udm.shape),
        #     "udm_id":udm_id,
        #     "service_time":udm_gen_st,
        # })

    
        udm_put_start_time = time.time()
        
        maybe_udm_matrix_chunks = Chunks.from_ndarray(
            ndarray      = udm,
            group_id     = udm_id,
            chunk_prefix = Some(udm_id),
            num_chunks   = num_chunks,
        )

        if maybe_udm_matrix_chunks.is_none:
            raise "something went wrong creating the chunks"


        udm_put_result = await RoryCommon.put_chunks(
            client = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = udm_id,
            chunks         = maybe_udm_matrix_chunks.unwrap(),
            tags = {
                "full_shape": str(udm.shape),
                "full_dtype": str(udm.dtype)
            }
        )

        if udm_put_result.is_err:
            e= udm_put_result.unwrap_err()
            raise Exception(f"Put UDM failed: {str(e)}")
        
        service_time_client = time.time() - arrivalTime
        udm_put_entry = ExperimentLogEntry(
            event          = "PUT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = udm_put_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = "",
            num_chunks     = num_chunks,
            k              = k,
            workers        = max_workers,
            security_level = security_level,
            m              = m
        )
        logger.info(udm_put_entry.model_dump())

        get_worker_start_time       = time.time()
        manager:RoryManager         = current_app.config.get("manager") # Communicates with the manager
        get_worker_result           = manager.getWorker( #Gets the worker from the manager
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

        get_worker_entry = ExperimentLogEntry(
            event          = "GET.WORKER",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_worker_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            k              = k,
            workers        = max_workers,
            security_level = security_level,
            m              = m
        )
        logger.info(get_worker_entry.model_dump())
        #     "event":"MANAGER.GET.WORKER",
        #     "worker_id":worker_id,
        #     "port":port,
        #     "algorithm":algorithm,
        #     "plaintext_matrix_id":plaintext_matrix_id,
        #     "service_time":get_worker_service_time,
        #     "k":k,
        #     "m":m
        # })
        
        worker_start_time = time.time()
        worker = RoryWorker( #Allows to establish the connection with the worker
            workerId  = worker_id,
            port      = port,
            session   = s,
            algorithm = algorithm,
        )
        status               = Constants.ClusteringStatus.START #Set the status to start
        worker_run1_response = None

        interaction_arrival_time = time.time()
        iterations               = 0
        label_vector             = None
        endTime = 0 
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
                "Max-Iterations"         : str(MAX_ITERATIONS),
                "Experiment-Id":experiment_id
            }
            
          
            
            worker_run1_response = worker.run(
                timeout = WORKER_TIMEOUT, 
                headers = run1_headers
            ) #Run 1 starts
            worker_run1_status = worker_run1_response.status_code

            if worker_run1_status !=200:
                return Response("Worker error: {}".format(worker_run1_response.content),status=500)
            
            worker_run1_response.raise_for_status()
            jsonWorkerResponse        = worker_run1_response.json()
            encrypted_shift_matrix_id = jsonWorkerResponse["encrypted_shift_matrix_id"]
            run1_service_time         = jsonWorkerResponse["service_time"]
            run1_n_iterations         = jsonWorkerResponse["n_iterations"]
            label_vector              = jsonWorkerResponse["label_vector"]

            run1_worker_entry = ExperimentLogEntry(
                event          = "SKMEANS.RUN1",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = inner_interaction_arrival_time,
                end_time       = time.time(),
                id             = plaintext_matrix_id,
                worker_id      = worker_id,
                num_chunks     = num_chunks,
                k              = k,
                workers        = max_workers,
                security_level = security_level,
                m              = m,
                iterations     = run1_n_iterations
            )
            logger.info(run1_worker_entry.model_dump())

            
       
            encrypted_shift_matrix_start_time = time.time()
            encrypted_shift_matrix = await RoryCommon.get_and_merge(
                client    = STORAGE_CLIENT, 
                key       = encrypted_shift_matrix_id,
                bucket_id = BUCKET_ID
            )
            get_encrypted_sm_entry = ExperimentLogEntry(
                event          = "GET",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = encrypted_shift_matrix_start_time,
                end_time       = time.time(),
                id             = plaintext_matrix_id,
                worker_id      = worker_id,
                num_chunks     = num_chunks,
                k              = k,
                workers        = max_workers,
                security_level = security_level,
                m              = m,
                iterations     = 0
            )
            logger.info(get_encrypted_sm_entry.model_dump())
     

            decrypt_start_time = time.time()
            shiftMatrix_chipher_schema_res = liu.decryptMatrix( #Shift Matrix is decrypted
                ciphertext_matrix = encrypted_shift_matrix.tolist(),
                secret_key        = dataowner.sk,
            )
            decrypt_entry = ExperimentLogEntry(
                event          = "DECRYPT",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = decrypt_start_time,
                end_time       = time.time(),
                id             = plaintext_matrix_id,
                worker_id      = worker_id,
                num_chunks     = num_chunks,
                k              = k,
                workers        = max_workers,
                security_level = security_level,
                m              = m,
                iterations     = 0
            )
            logger.info(decrypt_entry.model_dump())
     

            shift_matrix    = shiftMatrix_chipher_schema_res.matrix
            shift_matrix_id = "{}shiftmatrix".format(plaintext_matrix_id) # The id of the Shift matrix is formed
            put_shift_matrix_start_time     = time.time()

            maybe_shift_matrix_chunks = Chunks.from_ndarray(
                ndarray      = shift_matrix,
                group_id     = shift_matrix_id,
                chunk_prefix = Some(shift_matrix_id),
                num_chunks   = num_chunks,
                )

            if maybe_shift_matrix_chunks.is_none:
                raise "something went wrong creating the chunks"
            

         
            
            put_shift_matrix_result = await RoryCommon.delete_and_put_chunks(
                client = STORAGE_CLIENT,
                bucket_id      = BUCKET_ID,
                key            = shift_matrix_id,
                chunks         = maybe_shift_matrix_chunks.unwrap(),
                tags = {
                    "full_shape": str(shift_matrix.shape),
                    "full_dtype": str(shift_matrix.dtype)
                }
            )

            put_sm_entry = ExperimentLogEntry(
                event          = "PUT",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = put_shift_matrix_start_time,
                end_time       = time.time(),
                id             = plaintext_matrix_id,
                worker_id      = worker_id,
                num_chunks     = num_chunks,
                k              = k,
                workers        = max_workers,
                security_level = security_level,
                m              = m,
                iterations     = 0
            )
            logger.info(put_sm_entry.model_dump())

            status       = Constants.ClusteringStatus.WORK_IN_PROGRESS #Status is updated
            run2_headers = {
                    "Step-Index"             : "2",
                    "Clustering-Status"      : str(status),
                    "Shift-Matrix-Id"        : shift_matrix_id,
                    "Plaintext-Matrix-Id"    : plaintext_matrix_id,
                    "Encrypted-Matrix-Id"    : encrypted_matrix_id,
                    "Encrypted-Matrix-Shape" : "({},{},{})".format(r,a,m),
                    "Encrypted-Matrix-Dtype" : "float64",
                    "Num-Chunks"             : str(num_chunks),
                    "Iterations"             :str(iterations),
                    "K":str(k),
                    "M":str(m), 
                    "Experiment-Iteration": str(experiment_iteration), 
                    "Max-Iterations":str(MAX_ITERATIONS),
                    "Experiment-Id":experiment_id
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
            
            skmeans_iteration_completed_entry = ExperimentLogEntry(
                event          = "ITERATION.COMPLETED",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = inner_interaction_arrival_time,
                end_time       = time.time(),
                id             = plaintext_matrix_id,
                worker_id      = worker_id,
                num_chunks     = num_chunks,
                k              = k,
                workers        = max_workers,
                security_level = security_level,
                m              = m,
                iterations     = 0
            )
            logger.info(skmeans_iteration_completed_entry.model_dump())
            # logger.info({
            #     "event":"SKMEANS.ITERATION.COMPLETED",
            #     "algorithm":algorithm,
            #     "plaintext_matrix_id":plaintext_matrix_id,
            #     "worker_id":worker_id,
            #     "k":k,
            #     "m":m,
            #     "iterations":iterations,
            #     "service_time":inner_interaction_service_time,
            # })

        interaction_end_time     = time.time()
        interaction_service_time = interaction_end_time - interaction_arrival_time 
        worker_response_time     = endTime - worker_start_time
        response_time            = endTime - arrivalTime 

        clustering_completed_entry = ExperimentLogEntry(
                event          = "CLUSTERING",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = arrivalTime,
                end_time       = time.time(),
                id             = plaintext_matrix_id,
                worker_id      = worker_id,
                num_chunks     = num_chunks,
                k              = k,
                workers        = max_workers,
                security_level = security_level,
                m              = m,
                iterations     = iterations
            )
        logger.info(clustering_completed_entry.model_dump())
        #     "event":"SKMEANS.COMPLETED",
        #     "algorithm":algorithm,
        #     "plaintext_matrix_id":plaintext_matrix_id,
        #     "encrypted_matrix_id":encrypted_matrix_id,
        #     "num_chunks":num_chunks,
        #     "worker_id":worker_id,
        #     "k":k,
        #     "m":m,
        #     "n_iterations":iterations, 
        #     "max_iterations":MAX_ITERATIONS,
        #     "service_time_encrypted_matrix":seg_encry_rt,
        #     "service_time_dm_generation":udm_gen_st,
        #     "service_time_manager":get_worker_service_time,
        #     "service_time_worker":worker_response_time,
        #     "service_time_client":service_time_client,
        #     "response_time_clustering":response_time,
        #     "iterations_service_time":interaction_service_time
        # })

        return Response(
            response = json.dumps({
                "label_vector" : label_vector,
                "iterations":iterations,
                "algorithm":algorithm,
                "worker_id":worker_id,
                "service_time_manager":get_worker_service_time,
                "service_time_worker":worker_response_time,
                "service_time_client":service_time_client,
                "response_time_clustering":response_time,
            }),
            status   = 200,
            headers  = {}
        )
    except Exception as e:
        logger.error({
            "msg":str(e)
        })
        return Response(response= str(e) , status= 500)

#DBSKMEANS
@clustering.route("/dbskmeans", methods = ["POST"])
async def dbskmeans():
    try:
        local_start_time             = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        TESTING                      = current_app.config.get("TESTING",True)
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        liu:Liu                      = current_app.config.get("liu")
        dataowner:DataOwner          = current_app.config.get("dataowner")
        STORAGE_CLIENT:V4Client      = current_app.config.get("ASYNC_STORAGE_CLIENT")
        max_workers                  = int(current_app.config.get("MAX_WORKERS",2))
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        _num_chunks                  = current_app.config.get("NUM_CHUNKS",4)
        np_random                    = current_app.config.get("np_random")
        securitylevel                = current_app.config.get("LIU_SECURITY_LEVEL",128)
        if executor               == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm                 = Constants.ClusteringAlgorithms.DBSKMEANS
        s                         = Session()
        request_headers           = request.headers #Headers for the request
        num_chunks                = int(request_headers.get("Num-Chunks",_num_chunks ) )
        plaintext_matrix_id       = request_headers.get("Plaintext-Matrix-Id","matrix0")
        encrypted_matrix_id       = "encrypted{}".format(plaintext_matrix_id) # The id of the encrypted matrix is built
        encrypted_udm_id          = "{}encryptedudm".format(plaintext_matrix_id) # The iudm id is built
        plaintext_matrix_filename = request_headers.get("Plaintext-Matrix-Filename","matrix0")
        extension                 = request_headers.get("Extension","csv")
        m                         = dataowner.m
        k                         = request_headers.get("K","3")
        sens                      = float(request_headers.get("Sens","0.00000001"))
        experiment_iteration      = request_headers.get("Experiment-Iteration","0")
        MAX_ITERATIONS            = int(request_headers.get("Max-Iterations",current_app.config.get("MAX_ITERATIONS",10)))
        WORKER_TIMEOUT            = int(current_app.config.get("WORKER_TIMEOUT",3600))
        MICTLANX_TIMEOUT          = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
        backoff_factor = 1.5
        delay          = 1
        max_retries    = 10
        
        request_id                = "request{}".format(plaintext_matrix_id)
        plaintext_matrix_path     = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_filename, extension)
        iterations                = 0
        # Hay que sacar estos valores desde las variables de entorno (assigned to Shanel)

        logger.debug({
            "event":"DBSKMEANS.STARTED",
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "encrypted_matrix_id":encrypted_matrix_id,
            "plaintext_matrix_filename":plaintext_matrix_filename,
            "plaintext_matrix_path":plaintext_matrix_path,
            "security_level":securitylevel,
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

      

        local_read_dataset_start_time = time.time()
        plaintext_matrix_result       = await RoryCommon.read_numpy_from(
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
        plaintext_matrix_dtype = plaintext_matrix.dtype
        plaintext_matrix_shape = plaintext_matrix.shape

        cores       = os.cpu_count()
        max_workers = num_chunks if max_workers > num_chunks else max_workers
        max_workers = cores if max_workers > cores else max_workers
        
        encrypt_segment_start_time = time.time()
        n = a*r*int(m)
     
        encrypted_matrix_chunks = RoryCommon.segment_and_encrypt_liu_with_executor( #Encrypt 
            executor         = executor,
            key              = encrypted_matrix_id,
            dataowner        = dataowner,
            plaintext_matrix = plaintext_matrix,
            n                = n,
            np_random        = np_random,
            num_chunks       = num_chunks,
        )
        encrypt_segment_service_time = time.time() - encrypt_segment_start_time
        logger.info({
            "event":"SEGMENT.ENCRYPT",
            "key":encrypted_matrix_id,
            "plaintext_matrix_shape":str(plaintext_matrix_shape),
            "plaintext_matrix_dtype":str(plaintext_matrix_dtype),
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "n":n,
            "num_chunks":num_chunks,
            "max_workers":max_workers,
            "service_time":encrypt_segment_service_time
        })
        
        put_chunks_start_time = time.time()

     
        put_encrypted_matrix_result = await RoryCommon.delete_and_put_chunks(
            client = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = encrypted_matrix_id,
            chunks         = encrypted_matrix_chunks,
            tags = {
                "full_shape": str((r,a,m)),
                "full_dtype":"float64"
            },
            timeout=MICTLANX_TIMEOUT
        )
       
        put_chunks_st = time.time() - put_chunks_start_time
        logger.info({
            "event":"DELETE.AND.PUT.CHUNKED",
            "bucket_id":BUCKET_ID,
            "key":encrypted_matrix_id,
            "num_chunks":num_chunks,
            "algorithm":algorithm,
            "ok":put_encrypted_matrix_result.is_ok,
            "service_time":put_chunks_st
        })
        
        udm_start_time = time.time()
        udm            = dataowner.get_U(
            plaintext_matrix = plaintext_matrix,
            algorithm        = algorithm
        )
        udm_shape = udm.shape
        udm_dtype = udm.dtype

        # Plaintext matrix is useless from here to bottom. Free some memory:
        del plaintext_matrix

        udm_st = time.time() - udm_start_time
        logger.info({
            "event":"UDM.GENERATION",
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "shape":str(udm_shape),
            "dtype":str(udm_dtype),
            "service_time":udm_st
        })

        n         = r*r*a

        # threshold = 0.0
        segment_encrypt_fdhope_start_time = time.time()

        encrypted_matrix_UDM_chunks = RoryCommon.segment_and_encrypt_fdhope_with_executor( #Encrypt 
            executor         = executor,
            algorithm        = algorithm,
            key              = encrypted_udm_id,
            dataowner        = dataowner,
            matrix           = udm,
            n                = n,
            num_chunks       = num_chunks,
            sens             = sens
        )
        
        segment_encrypt_fdhope_st = time.time() - segment_encrypt_fdhope_start_time

        put_chunks_start_time = time.time()
    
        put_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
            client         = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = encrypted_udm_id,
            chunks         = encrypted_matrix_UDM_chunks,
            tags = {
                "full_shape": str((r,r,a)),
                "full_dtype":"float64"
            },
            timeout=MICTLANX_TIMEOUT
        )

        if put_chunks_generator_results.is_err:
            return Response("Put chunks failed: UDM",status=500)

        logger.info({
            "event":"DELETE.AND.PUT.CHUNKED",
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "key":encrypted_udm_id,
            "num_chunks":num_chunks,
            "service_time":time.time() - put_chunks_start_time
        })
        service_time_client = time.time() - local_start_time
        del udm 
        del encrypted_matrix_UDM_chunks
        
        # Manager
        get_worker_start_time = time.time()
        managerResponse:RoryManager = current_app.config.get("manager") # Communicates with the manager
        
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
            "plaintext_matrix_id":plaintext_matrix_id,
            "service_time":get_worker_service_time,
        })

        worker_start_time = time.time()
        worker         = RoryWorker( #Allows to establish the connection with the worker
            workerId   = worker_id,
            port       = port,
            session    = s,
            algorithm  = algorithm
        )
        status           = Constants.ClusteringStatus.START #Set the status to start
        worker_run2_response = None
        initial_encrypted_udm_shape = (r,r,a)
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
                "Encrypted-Udm-Shape"    : str(initial_encrypted_udm_shape),
                "Encrypted-Udm-Dtype"    : "float64",
                "Num-Chunks"             : str(num_chunks),
                "Iterations"             : str(iterations),
                "K"                      : str(k),
                "M"                      : str(m), 
                "Experiment-Iteration"   : str(experiment_iteration), 
                "Max-Iterations"         : str(MAX_ITERATIONS) 
            }
            workerResponse1 = worker.run(timeout = WORKER_TIMEOUT,headers =run1_headers) #Run 1 starts
            workerResponse1.raise_for_status()
            
            # stringWorkerResponse = workerResponse1.content.decode("utf-8") #Response from worker
            jsonWorkerResponse   = workerResponse1.json()
            # json.loads(stringWorkerResponse) #pass to json
            
            encrypted_shift_matrix_id = jsonWorkerResponse["encrypted_shift_matrix_id"]
            run1_service_time         = jsonWorkerResponse['service_time']
            label_vector              = jsonWorkerResponse["label_vector"]
            run1_response_time        = time.time() - inner_interaction_start_time
            logger.info({
                "event":"WORKER.RUN.1",
                "step_index":"1",
                "clustering_status":str(status),
                "algorithm":algorithm,
                "plaintext_matrix_id":plaintext_matrix_id,
                "encrypted_matrix_id":encrypted_matrix_id,
                "encrypted_matrix_shape":"({},{},{})".format(r,a,m),
                "encrypted_matrix_dtype":"float64", 
                "encrypted_udm_shape":str(initial_encrypted_udm_shape),
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
            get_matrix_start_time = time.time()
            encryptedShiftMatrix = await RoryCommon.get_and_merge(
                bucket_id = BUCKET_ID,
                key       = encrypted_shift_matrix_id,
                client    = STORAGE_CLIENT, 
                timeout   = MICTLANX_TIMEOUT,
                backoff_factor=backoff_factor,
                delay=delay,
                max_retries=max_retries,
            )

            get_matrix_st              = time.time() - get_matrix_start_time
            # encryptedShiftMatrix       = encryptedShiftMatrix_get_response.value
            encryptedShiftMatrix_shape = encryptedShiftMatrix.shape
            encryptedShiftMatrix_dtype = encryptedShiftMatrix.dtype

            logger.info({
                "event":"GET.MATRIX.OR.ERROR",
                "algorithm":algorithm,
                "plaintext_matrix_id":plaintext_matrix_id,
                "key":encrypted_shift_matrix_id,
                "bucket_id":BUCKET_ID,
                "service_time":get_matrix_st
            })


            decrypt_start_time = time.time()
            cipher_schema_res  = liu.decryptMatrix( #Shift Matrix is decrypted
                ciphertext_matrix = encryptedShiftMatrix,
                secret_key        = dataowner.sk,
            )
            descrypy_st = time.time() - decrypt_start_time
            logger.info({
                "event":"DECRYPT",
                "algorithm":algorithm,
                "plaintext_matrix_id":plaintext_matrix_id,
                "m":int(m),
                "encrypted_shift_matrix_shape":str(encryptedShiftMatrix_shape),
                "encrypted_shift_matrix_dtype":str(encryptedShiftMatrix_dtype),
                "service_time":descrypy_st
            })

            del encryptedShiftMatrix
            
            shift_matrix= cipher_schema_res.matrix

            encrypted_start_time = time.time()
            fdhope_encrypted_shift_matrix = Fdhope.encryptMatrix( #Re-encrypt shift matrix with the FDHOPE scheme
                plaintext_matrix = shift_matrix, 
                messagespace     = dataowner.messageIntervals,
                cipherspace      = dataowner.cypherIntervals
            )
            
            del shift_matrix

            shift_matrix_ope = fdhope_encrypted_shift_matrix.matrix
            shift_matrix_ope_shape = shift_matrix_ope.shape
            shift_matrix_ope_dtype = shift_matrix_ope.dtype
            logger.info({
                "event":"ENCRYPT.FDHOPE",
                "algorithm":algorithm,
                "plaintext_matrix_id":plaintext_matrix_id,
                "m":int(m),
                "shift_matrix_shape":str(shift_matrix_ope_shape),
                "shift_matrix_dtype":str(shift_matrix_ope_dtype),
                "service_time":time.time() - encrypted_start_time
            })
            shift_matrix_id     = "{}shiftmatrix".format(plaintext_matrix_id) # The id of the Shift matrix is formed
            shift_matrix_ope_id = "{}shiftmatrixope".format(plaintext_matrix_id) # The id of the Shift matrix is formed
            
            put_matrix_start_time = time.time()
          
            maybe_shift_matrix_chunks = Chunks.from_ndarray(
                ndarray      = shift_matrix_ope,
                group_id     = shift_matrix_ope_id,
                chunk_prefix = Some(shift_matrix_ope_id),
                num_chunks   = num_chunks,
            )

            if maybe_shift_matrix_chunks.is_none:
                raise "something went wrong creating the chunks: Encrypted shift matrix"

            logger.info({
                "event":"CHUNKS.FROM.NDARRAY",
                "algorithm":algorithm,
                "plaintext_matrix_id":plaintext_matrix_id,
                "key":shift_matrix_ope_id,
                "bucket_id":BUCKET_ID,
                "shift_matrix_shape":str(shift_matrix_ope_shape),
                "shift_matrix_dtype":str(shift_matrix_ope_dtype)
            })

            
            t_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
                client = STORAGE_CLIENT,
                bucket_id      = BUCKET_ID,
                key            = shift_matrix_ope_id,
                chunks         = maybe_shift_matrix_chunks.unwrap(),
                tags = {
                    "full_shape": str(shift_matrix_ope_shape),
                    "full_dtype": str(shift_matrix_ope_dtype)
                },
                timeout=MICTLANX_TIMEOUT
            )
            del maybe_shift_matrix_chunks
            del shift_matrix_ope
                
    
            status = Constants.ClusteringStatus.WORK_IN_PROGRESS #Status is updated
            logger.info({
                "event":"DELETE.AND.PUT.CHUNKED",
                "algorithm":algorithm,
                "plaintext_matrix_id":plaintext_matrix_id,
                "key":shift_matrix_ope_id,
                "bucket_id":BUCKET_ID,
                "shift_matrix_shape":str(shift_matrix_ope_shape),
                "shift_matrix_dtype":str(shift_matrix_ope_dtype),
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
                    "Encrypted-Udm-Shape"    : str(initial_encrypted_udm_shape),
                    "Num-Chunks"             : str(num_chunks),
                    "Iterations"             : str(iterations),
                    "K"                      : str(k),
                    "M"                      : str(m), 
                    "Experiment-Iteration"   : str(experiment_iteration), 
                    "Max-Iterations"         : str(MAX_ITERATIONS) 
            }
            
            run2_start_time = time.time()
            worker_run2_response = worker.run(
                timeout = WORKER_TIMEOUT,
                headers = run2_headers
            ) #Start run 2

            worker_run2_response.raise_for_status()
            # str_run2_response  = worker_run2_response.content.decode("utf-8") #Response from worker
            run2_json          = worker_run2_response.json()
            # json.loads(str_run2_response) #pass to json
            initial_encrypted_udm_shape  = eval(run2_json["encrypted_udm_shape"])
            run2_service_time  = run2_json["service_time"]
            run2_response_time = time.time() - run2_start_time
            
            logger.info({
                "event":"WORKER.RUN.2",
                "step_index":"2",
                "clustering_status":str(status),
                "algorithm":algorithm,
                "plaintext_matrix_id":plaintext_matrix_id,
                "encrypted_matrix_id":encrypted_matrix_id,
                "shift_matrix_id":shift_matrix_id,
                "shift_matrix_ope_id":shift_matrix_ope_id,
                "encrypted_matrix_shape":"({},{},{})".format(r,a,m),
                "encrypted_matrix_dtype":"float64", 
                "encrypted_udm_shape":str(initial_encrypted_udm_shape),
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
                "algorithm":algorithm,
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

        worker_end_time       = time.time()
        worker_response_time  = worker_end_time - worker_start_time

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
            "service_time_encrypted_matrix":encrypt_segment_service_time,
            "service_time_dm_generation":udm_st,
            "service_time_dm_encrypted":segment_encrypt_fdhope_st,
            "service_time_manager":get_worker_service_time,
            "service_time_worker":worker_response_time,
            "service_time_clustering":response_time
        })

        return Response(
            response = json.dumps({
                "label_vector" : label_vector,
                "iterations":iterations,
                "algorithm":algorithm,
                "worker_id":worker_id,
                "service_time_manager":get_worker_service_time,
                "service_time_worker":worker_response_time,
                "service_time_client":service_time_client,
                "response_time_clustering":response_time,
            }),
            status   = 200,
            headers  = {}
        )
    except Exception as e:
        logger.error({
            "msg":str(e)
        })
        return Response(response= str(e) , status= 500)
    
#DBSNNC
@clustering.route("/dbsnnc", methods      = ["POST"])
async def dbsnnc():
    try:
        local_start_time             = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        TESTING                      = current_app.config.get("TESTING",True)
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        dataowner:DataOwner          = current_app.config.get("dataowner")
        STORAGE_CLIENT:V4Client      = current_app.config.get("ASYNC_STORAGE_CLIENT")
        _num_chunks                  = current_app.config.get("NUM_CHUNKS",4)
        max_workers                  = current_app.config.get("MAX_WORKERS",2)
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        np_random                    = current_app.config.get("np_random")
        securitylevel                = current_app.config.get("LIU_SECURITY_LEVEL",128)
        if executor                  == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm                 = Constants.ClusteringAlgorithms.DBSNNC
        s                         = Session()
        request_headers           = request.headers #Headers for the request
        num_chunks                = int(request_headers.get("Num-Chunks",_num_chunks))
        plaintext_matrix_id       = request_headers.get("Plaintext-Matrix-Id","matrix0")
        encrypted_matrix_id       = "encrypted{}".format(plaintext_matrix_id) # The id of the encrypted matrix is built
        dm_id                     = "{}dm".format(plaintext_matrix_id)
        encrypted_dm_id           = "{}encrypteddm".format(plaintext_matrix_id) # The iudm id is built
        plaintext_matrix_filename = request_headers.get("Plaintext-Matrix-Filename","matrix-0")
        extension                 = request_headers.get("Extension","csv")
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
            "security_level":securitylevel,
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
        local_read_dataset_start_time = time.time()
        plaintext_matrix_result  = await RoryCommon.read_numpy_from(
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
        
        cores       = os.cpu_count()
        max_workers = num_chunks if max_workers > num_chunks else max_workers
        max_workers = cores if max_workers > cores else max_workers
        encryption_start_time = time.time()

        n = r*a*m

        segment_encrypt_start_time = time.time()
        encrypted_matrix_chunks = RoryCommon.segment_and_encrypt_liu_with_executor( #Encrypt 
            executor         = executor,
            key              = encrypted_matrix_id,
            dataowner        = dataowner,
            plaintext_matrix = plaintext_matrix,
            n                = n,
            np_random        = np_random,
            num_chunks=num_chunks
        )
        segment_encrypt_st = time.time() - segment_encrypt_start_time
        logger.info({
            "event":"SEGMENT.ENCRYPT.LIU",
            "max_workers": max_workers,
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "encrypted_matrix_id":encrypted_matrix_id,
            "num_chunks":num_chunks,
            "n":n,
            "service_time":segment_encrypt_st
        })
        
        put_chunks_start_time = time.time()
        
     

        put_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
            client = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = encrypted_matrix_id,
            chunks         = encrypted_matrix_chunks,
            tags = {
                "full_shape": str((r,a,m)),
                "full_dtype":"float64"
            }
        )

        put_chunks_st = time.time() - put_chunks_start_time

        logger.info({
            "event":"DELETE.AND.PUT.CHUNKED",
            "key":encrypted_matrix_id,
            "num_chunks":num_chunks,
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "service_time":put_chunks_st
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
        
        
        n = r*r

        segment_encrypt_fdhope_start_time = time.time()

        encrypted_matrix_DM_chunks = RoryCommon.segment_and_encrypt_fdhope_with_executor( #Encrypt 
            executor         = executor,
            algorithm        = algorithm,
            key              = encrypted_dm_id,
            dataowner        = dataowner,
            matrix           = dm,
            n                = n,
            num_chunks       = num_chunks,
            sens             = sens,
        )
        segment_encrypt_fdhope_st = time.time() - segment_encrypt_fdhope_start_time
        logger.info({
            "event":"SEGMENT.ENCRYPT.FDHOPE",
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "encrypted_dm_id":encrypted_dm_id,
            "n":n,
            "num_chunks":num_chunks,
            "max_wokers":max_workers,
            "threshold":threshold,
            "dm_shape":str(dm.shape),
            "dm_dtype":str(dm.dtype),
            "service_time":segment_encrypt_fdhope_st
        })
        
    
        put_chunks_start_time = time.time()

     
        put_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
            client         = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = encrypted_dm_id,
            chunks         = encrypted_matrix_DM_chunks,
            tags = {
                "full_shape": str((r,r)),
                "full_dtype":"float64"
            }
        )

        put_chunks_st = time.time() - put_chunks_start_time
        segment_encrypt_fdhope_st = time.time() - segment_encrypt_start_time

        logger.info({
            "event":"DELETE.AND.PUT.CHUNKED",
            "key":encrypted_matrix_id,
            "num_chunks":num_chunks,
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "service_time":segment_encrypt_fdhope_st
        })
        
   
        encrypted_threshold  = Fdhope.encrypt( #Threshold is encrypted
				plaintext    = threshold,
				messagespace = dataowner.messageIntervals, 
				cipherspace  = dataowner.cypherIntervals,
                sens         = sens,
			)
 
        service_time_client         = time.time() - local_start_time
        get_worker_start_time       = time.time()
        managerResponse:RoryManager = current_app.config.get("manager") # Communicates with the manager
        
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
            "plaintext_matrix_id":plaintext_matrix_id,
            "worker_id":_worker_id,
            "port":port,
            "m":m,
            "service_time":get_worker_service_time
        })

        worker_start_time = time.time()
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


        run1_response = worker.run(
            timeout = WORKER_TIMEOUT, 
            headers = run_headers
        )
        run1_response.raise_for_status()
        
        # stringWorkerResponse = run1_response.content.decode("utf-8") #Response from worker
        jsonWorkerResponse   = run1_response.json()
        # json.loads(stringWorkerResponse) #pass to json
        endTime              = time.time() # Get the time when it ends
        worker_service_time  = jsonWorkerResponse["service_time"]
        label_vector         = jsonWorkerResponse["label_vector"]
        response_time        = endTime - local_start_time # Get the service time
        worker_end_time       = time.time()
        worker_response_time  = worker_end_time - worker_start_time
        logger.info({
            "event":"DBSNNC.COMPLETED",
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "encrypted_matrix_id":encrypted_matrix_id,
            "encrypted_distance_matrix_id":encrypted_dm_id,
            "num_chunks":num_chunks,
            "m":m,
            "threshold":threshold,
            "worker_id":_worker_id,
            "service_time_encrypted_matrix":segment_encrypt_st,
            "service_time_dm_generation":generate_dm_st,
            "service_time_manager":get_worker_service_time,
            "service_time_worker":worker_response_time,
            "service_time_clustering":response_time
        })
        return Response(
            response = json.dumps({
                "label_vector" :label_vector,
                "algorithm":algorithm,
                "worker_id":worker_id,
                "service_time_manager":get_worker_service_time,
                "service_time_worker":worker_response_time,
                "service_time_client":service_time_client,
                "response_time_clustering":response_time,
                
            }),
            status   = 200,
            headers  = {}
        )
    except Exception as e:
        logger.error({
            "msg":str(e)
        })
        return Response(response =None, status= 500, headers={"Error-Message":str(e)})
    
#NNC
@clustering.route("/nnc", methods = ["POST"])
async def nnc():
    try:
        local_start_time             = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        TESTING                      = current_app.config.get("TESTING",True)
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        dataowner:DataOwner          = current_app.config.get("dataowner")
        STORAGE_CLIENT:V4Client      = current_app.config.get("ASYNC_STORAGE_CLIENT")
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
        request_id                = "request{}".format(plaintext_matrix_id)
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

  

        local_read_dataset_start_time = time.time()
        plaintext_matrix_result = await RoryCommon.read_numpy_from( 
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
        

        put_ptm_start_time = time.time()

        plaintext_matrix_chunks = Chunks.from_ndarray(
            ndarray      = plaintext_matrix,
            group_id     = plaintext_matrix_id,
            chunk_prefix = Some(plaintext_matrix_id),
            num_chunks   = num_chunks,
        )

        if plaintext_matrix_chunks.is_none:
            raise "something went wrong creating the chunks"
        
        # logger.info({
        #     "event":"CHUNKS.FROM.NDARRAY",
        #     "algorithm":algorithm,
        #     "plaintext_matrix_id":plaintext_matrix_id,
        #     "key":plaintext_matrix_id,
        #     "bucket_id":BUCKET_ID,
        #     "plaintext_matrix_shape":str(plaintext_matrix.shape),
        #     "plaintext_matrix_dtype":str(plaintext_matrix.dtype),
        # })


        # chunks_bytes = Utils.chunks_to_bytes_gen(
        #     chs = 
        # )

        t_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
            client         = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = plaintext_matrix_id,
            chunks         = plaintext_matrix_chunks.unwrap(),
            tags           = {
                "full_shape": str(plaintext_matrix.shape),
                "full_dtype": str(plaintext_matrix.dtype)
            }
        )

        put_ptm_st = time.time() - put_ptm_start_time

        logger.info({
            "event":"DELETE.AND.PUT.CHUNKED",
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "key":plaintext_matrix_id,
            "bucket_id":BUCKET_ID,
            "plaintext_matrix_shape":str(plaintext_matrix.shape),
            "plaintext_matrix_dtype":str(plaintext_matrix.dtype),
            "service_time":put_ptm_st
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

        put_ptm_start_time = time.time()
        maybe_dm_chunks = Chunks.from_ndarray(
            ndarray      = dm,
            group_id     = dm_id,
            chunk_prefix = Some(dm_id),
            num_chunks   = num_chunks
        )

        if maybe_dm_chunks.is_none:
            raise "something went wrong creating the chunks"
        
        put_ptm_st = time.time() - put_ptm_start_time
        logger.info({
            "event":"CHUNKS.FROM.NDARRAY",
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "dm_id":dm_id,
            "bucket_id":BUCKET_ID,
            "dm_shape":str(dm.shape),
            "dm_dtype":str(dm.dtype),
            "service_time":put_ptm_st
        })


        put_dm_start_time = time.time()


        
        put_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
            client         = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = dm_id,
            chunks         = maybe_dm_chunks.unwrap(),
            tags = {
                "full_shape":str(dm.shape),
                "full_dtype":str(dm.dtype)
            }
        )

        put_dm_st = time.time() - put_dm_start_time
        service_time_client = time.time() - local_start_time
        logger.info({
            "event":"DELETE.AND.PUT.CHUNKED",
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "key":dm_id,
            "bucket_id":BUCKET_ID,
            "shape":str(dm.shape),
            "dtype":str(dm.dtype),
            "service_time":put_dm_st
        })

        get_worker_start_time = time.time()
        managerResponse:RoryManager = current_app.config.get("manager") # Communicates with the manager
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
            "plaintext_matrix_id":plaintext_matrix_id,
            "worker_id":_worker_id,
            "port":port,
            "service_time":get_worker_service_time,
        })

        worker_start_time = time.time()
        worker         = RoryWorker( #Allows to establish the connection with the worker
            workerId   = worker_id,
            port       = port,
            session    = s,
            algorithm  = algorithm
        )
        pm_shape = (r,a)
        dm_shape = (r,r)
        run_headers = {
            "Plaintext-Matrix-Id"    : plaintext_matrix_id,
            "Request-Id"             : request_id,
            "Num-Chunks"             : str(num_chunks),
            "Threshold"              : str(threshold),
            "Plaintext-Matrix-Shape" : str(pm_shape),
            "Plaintext-Matrix-Dtype" : str(plaintext_matrix.dtype),
            "Dm-Shape"               : str(dm_shape),
            "Dm-Dtype"               : str(dm.dtype),
        }

        worker_response  = worker.run(
            timeout = WORKER_TIMEOUT, 
            headers = run_headers
        )
        
        worker_response.raise_for_status()
        # stringWorkerResponse = worker_response.content.decode("utf-8") #Response from worker
        jsonWorkerResponse   = worker_response.json()
        end_time             = time.time() # Get the time when it ends
        worker_service_time  = jsonWorkerResponse["service_time"]
        label_vector         = jsonWorkerResponse["label_vector"]
        response_time        = end_time - local_start_time # Get the service time
        worker_end_time       = time.time()
        worker_response_time  = worker_end_time - worker_start_time
        logger.info({
            "event":"NNC.COMPLETED",
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "worker_id":_worker_id,
            "service_time_manager":get_worker_service_time,
            "service_time_worker":worker_response_time,
            "service_time_client":service_time_client,
            "response_time_clustering":response_time,
        })

        return Response(
            response = json.dumps({
                "label_vector" : label_vector,
                "algorithm":algorithm,
                "worker_id":worker_id,
                "service_time_manager":get_worker_service_time,
                "service_time_worker":worker_response_time,
                "service_time_client":service_time_client,
                "response_time_clustering":response_time,
            }),
            status   = 200,
            headers  = {}
        )
    except Exception as e:
        logger.error({
            "msg":str(e)
        })
        return Response(response =None, status= 500, headers={"Error-Message":str(e)})
 
#PCQ-SKMEANS
@clustering.route("/pqc/skmeans",methods = ["POST"])
async def pqc_skmeans():
    try:
        arrivalTime                  = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        TESTING                      = current_app.config.get("TESTING",True)
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        STORAGE_CLIENT:V4Client      = current_app.config.get("ASYNC_STORAGE_CLIENT")
        _num_chunks                  = current_app.config.get("NUM_CHUNKS",4)
        max_workers                  = current_app.config.get("MAX_WORKERS",2)
        securitylevel                = current_app.config.get("LIU_SECURITY_LEVEL",128)
        np_random                    = current_app.config.get("np_random")
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        
        if executor == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm                 = Constants.ClusteringAlgorithms.SKMEANS_PQC
        s                         = Session()
        request_headers           = request.headers #Headers for the request
        num_chunks                = int(request_headers.get("Num-Chunks",_num_chunks))
        plaintext_matrix_id       = request_headers.get("Plaintext-Matrix-Id","matrix0")
        encrypted_matrix_id       = "encrypted{}".format(plaintext_matrix_id) # The id of the encrypted matrix is built
        udm_id                    = "{}udm".format(plaintext_matrix_id) # The iudm id is built
        plaintext_matrix_filename = request_headers.get("Plaintext-Matrix-Filename","matrix0")
        extension                 = request_headers.get("Extension","csv")
        k                         = int(request_headers.get("K"))
        experiment_iteration      = request_headers.get("Experiment-Iteration","0")
        MAX_ITERATIONS            = int(request_headers.get("Max-Iterations",current_app.config.get("MAX_ITERATIONS",10)))
        WORKER_TIMEOUT            = int(current_app.config.get("WORKER_TIMEOUT",300))
        requestId                 = "request-{}".format(plaintext_matrix_id)
        plaintext_matrix_path     = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_filename, extension)

        cent_i_id = "{}centi".format(plaintext_matrix_id) #Build the id of Cent_i
        cent_j_id = "{}centj".format(plaintext_matrix_id) #Build the id of Cent_j

        _round   = False
        decimals = 2
        path               = os.environ.get("KEYS_PATH","/rory/keys")
        ctx_filename       = os.environ.get("CTX_FILENAME","ctx")
        pubkey_filename    = os.environ.get("PUBKEY_FILENAME","pubkey")
        secretkey_filename = os.environ.get("SECRET_KEY_FILENAME","secretkey")
        
        delay = 2 
        backoff_factor = .5
        max_retries = 10
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

        logger.debug({
            "event":"SKMEANS.STARTED",
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "encrypted_matrix_id":encrypted_matrix_id,
            "udm_id":udm_id,
            "plaintext_matrix_filename":plaintext_matrix_filename,
            "plaintext_matrix_path":plaintext_matrix_path,
            "security_level":securitylevel,
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
        
        local_read_dataset_start_time = time.time()
        plaintext_matrix_result  = await RoryCommon.read_numpy_from(
            path      = plaintext_matrix_path,
            extension = extension,
        )
        if plaintext_matrix_result.is_err:
            return Response(status=500, response="Failed to local read plain text matrix.")
        plaintext_matrix = plaintext_matrix_result.unwrap()
        
        plaintext_matrix = plaintext_matrix.astype(np.float64)

        r = plaintext_matrix.shape[0]
        a = plaintext_matrix.shape[1]

        max_workers = Utils.get_workers(num_chunks=num_chunks)
       

        encryption_start_time = time.time()
        n = a*r
        print(ckks.he_object)
        encrypted_matrix_chunks = RoryCommon.segment_and_encrypt_ckks_with_executor( #Encrypt 
            executor           = executor,
            key                = encrypted_matrix_id,
            plaintext_matrix   = plaintext_matrix,
            n                  = n,
            _round             = _round,
            decimals           = decimals,
            path               = path,
            ctx_filename       = ctx_filename,
            pubkey_filename    = pubkey_filename,
            secretkey_filename = secretkey_filename,
            num_chunks         = num_chunks,
        )
        segment_encrypt_service_time = time.time() - encryption_start_time
        logger.info({
            "event":"SEGMENT.ENCRYPT.CKKS",
            "key":encrypted_matrix_id,
            "plaintext_matrix_shape":str(plaintext_matrix.shape),
            "plaintext_matrix_dtype":str(plaintext_matrix.dtype),
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "n":n,
            "num_chunks":num_chunks,
            "max_workers":max_workers,
            "service_time":segment_encrypt_service_time
        })
  
        put_chunks_start_time = time.time()

        put_encrypted_matrix_result = await RoryCommon.delete_and_put_chunks(
            client = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = encrypted_matrix_id,
            chunks         = encrypted_matrix_chunks,
            tags = {
                "full_shape": str((r,a)),
                "full_dtype":"float64"
            }
        )
        # raise Exception("BOOM!")
        put_chunks_st = time.time() - put_chunks_start_time
        logger.info({
            "event":"DELETE.AND.PUT.CHUNKED",
            "key":encrypted_matrix_id,
            "num_chunks":num_chunks,
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "ok":put_encrypted_matrix_result.is_ok,
            "service_time":put_chunks_st
        })

        udm_start_time = time.time()
        udm            = dataowner.get_U(
            plaintext_matrix = plaintext_matrix,
            algorithm        = algorithm
        )
        
        udm_gen_st = time.time()- udm_start_time
        logger.info({
            "event":"UDM.GENERATION",
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "shape":str(udm.shape),
            "udm_id":udm_id,
            "service_time":udm_gen_st,
        })
        
        udm_put_start_time = time.time()
        
        maybe_udm_matrix_chunks = Chunks.from_ndarray(
            ndarray      = udm,
            group_id     = udm_id,
            chunk_prefix = Some(udm_id),
            num_chunks   = num_chunks,
        )

        if maybe_udm_matrix_chunks.is_none:
            error = "Something went wrong creating the UDM chunks"
            logger.error(error)
            return Response(status=500,response=error)
        
        logger.info({
            "event":"CHUNKS.FROM.NDARRAY",
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "key":udm_id,
            "bucket_id":BUCKET_ID,
            "udm_shape":str(udm.shape),
            "udm_dtype":str(udm.dtype),
        })



        udm_put_result = await RoryCommon.delete_and_put_chunks(
            client         = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = udm_id,
            chunks         = maybe_udm_matrix_chunks.unwrap(),
            tags = {
                "full_shape": str(udm.shape),
                "full_dtype": str(udm.dtype)
            }
        )

        if udm_put_result.is_err:
            error = udm_put_result.unwrap_err()
            e = f"Failed to put the udm: {error}"
            return Response(status= 500, response=e)
        udm_put_st = time.time() - udm_put_start_time

        service_time_client = time.time() - arrivalTime
        logger.info({            
            "event":"DELETE.AND.PUT.CHUNKED",
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "key":udm_id,
            "bucket_id":BUCKET_ID,
            "udm_shape":str(udm.shape),
            "udm_dtype":str(udm.dtype),
            "ok":udm_put_result.is_ok,
            "service_time":udm_put_st
        })
        
        init_sm_id = "{}initsm".format(plaintext_matrix_id)
        
        zero_shiftmatrix = np.zeros((k, a))
        n2 = a*k
        encrypt_ckks_start_time = time.time()
        encrypted_zero_shiftmatrix_chunks = RoryCommon.segment_and_encrypt_ckks_with_executor( #Encrypt 
            executor           = executor,
            key                = init_sm_id,
            plaintext_matrix   = zero_shiftmatrix,
            n                  = n2,
            num_chunks         = num_chunks,
            _round             = _round,
            decimals           = decimals,
            path               = path,
            ctx_filename       = ctx_filename,
            pubkey_filename    = pubkey_filename,
            secretkey_filename = secretkey_filename
        )

        logger.info({
            "event":"SEGMENT.ENCRYPT.INITSHIFTMATRIX",
            "algorithm":algorithm,
            "key":init_sm_id,
            "plaintext_matrix_id":plaintext_matrix_id,
            "n":n,
            "num_chunks":num_chunks,
            "max_workers":max_workers,
            "service_time": time.time() - encrypt_ckks_start_time
        })

        put_chunks_start_time = time.time()
        put_encrypted_matrix_result = await RoryCommon.delete_and_put_chunks(
            client         = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = init_sm_id,
            chunks         = encrypted_zero_shiftmatrix_chunks,
            tags = {
                "full_shape": str((k,a)),
                "full_dtype":"float64"
            }
        )
        if put_encrypted_matrix_result.is_err:
            e =f"Failed put chunks: {put_encrypted_matrix_result.unwrap_err()}" 
            logger.error(e)
            return Response(status=500, response=e)
        put_chunks_st = time.time() - put_chunks_start_time
        logger.info({
            "event":"DELETE.AND.PUT.CHUNKED",
            "bucket_id":BUCKET_ID,
            "key":init_sm_id,
            "algorithm":algorithm,
            "ok":put_encrypted_matrix_result.is_ok,
            "service_time":put_chunks_st,
        })
        
        get_worker_start_time       = time.time()
        managerResponse:RoryManager = current_app.config.get("manager") # Communicates with the manager
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
            "event":"MANAGER.GET.WORKER1",
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "worker_id":worker_id,
            "port":port,
            "k":k,
            "service_time":get_worker_service_time,
        })
        
        worker_start_time = time.time()
        worker = RoryWorker( #Allows to establish the connection with the worker
            workerId  = worker_id,
            port      = port,
            session   = s,
            algorithm = algorithm,
        )
        status               = Constants.ClusteringStatus.START #Set the status to start
        worker_run1_response = None

        interaction_arrival_time = time.time()
        iterations               = 0
        label_vector             = None
        # raise Exception("Boom!")
        while (status != Constants.ClusteringStatus.COMPLETED): #While the status is not completed
            
            inner_interaction_arrival_time = time.time()
            run1_headers  = {
                "Step-Index"             : "1",
                "Clustering-Status"      : str(status),
                "Plaintext-Matrix-Id"    : plaintext_matrix_id,
                "Request-Id"             : requestId,
                "Encrypted-Matrix-Id"    : encrypted_matrix_id,
                "Encrypted-Matrix-Shape" : "({},{})".format(r,a),
                "Encrypted-Matrix-Dtype" : "float64",
                "Encrypted-Udm-Dtype"    : "float64",
                "Num-Chunks"             : str(num_chunks),
                "Iterations"             : str(iterations),
                "K"                      : str(k),
                "Experiment-Iteration"   : str(experiment_iteration), 
                "Max-Iterations"         : str(MAX_ITERATIONS) 
            }  
            
            logger.debug({
                "event":"WORKER.RUN1.BEFORE",
                "algorithm":algorithm,
                "plaintext_matrix_id":plaintext_matrix_id,
                "step_index":"1",
                "clustering_status":str(status),
                "request_id":requestId,
                "encrypted_matrix_id":encrypted_matrix_id,
                "encrypted_matrix_dtype":"float64", 
                "num_chunks":num_chunks,
                "iterations":iterations,
                "k":k, 
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
            # stringWorkerResponse      = worker_run1_response.content.decode("utf-8") #Response from worker
            jsonWorkerResponse        = worker_run1_response.json()
            encrypted_shift_matrix_id = jsonWorkerResponse["encrypted_shift_matrix_id"]
            run1_service_time         = jsonWorkerResponse["service_time"]
            run1_n_iterations         = jsonWorkerResponse["n_iterations"]
            label_vector              = jsonWorkerResponse["label_vector"]

            logger.info({
                "event":"SKMEANS.RUN1",
                "algorithm":algorithm,
                "plaintext_matrix_id":plaintext_matrix_id,
                "step_index":"1",
                "clustering_status":str(status),
                "request_id":requestId,
                "encrypted_matrix_id":encrypted_matrix_id,
                "encrypted_matrix_shape":"({},{})".format(r,a),
                "encrypted_matrix_dtype":"float64", 
                "num_chunks":num_chunks,
                "iterations":iterations,
                "k":k, 
                "experiment_iteration":experiment_iteration,
                "max_iterations":MAX_ITERATIONS,
                "status":worker_run1_status,
                "worker_service_time": run1_service_time,
                "n_iterations":run1_n_iterations,
                "response_time":time.time() - inner_interaction_arrival_time
            })
            encrypted_shift_matrix_start_time = time.time()
            
            encrypted_shift_matrix = await RoryCommon.get_pyctxt(
                client    = STORAGE_CLIENT,
                bucket_id = BUCKET_ID,
                key       = encrypted_shift_matrix_id,
                ckks      = ckks,
                force     = True,
                backoff_factor=backoff_factor,
                delay=delay,
                max_retries=max_retries
            )
            # response               = encrypted_shift_matrix_result.unwrap().value
            # encrypted_shift_matrix = Utils.bytes_to_pyctxt_list_v2(ckks = ckks, data=response)
            
            logger.debug({
                "event":"CKKS.DECRYPT.BEFORE",
                "algorithm":algorithm,
                "plaintext_matrix_id":plaintext_matrix_id,
            })

            decrypt_start_time = time.time()
            shift_matrix = ckks.decryptMatrix( #Shift Matrix is decrypted
                ciphertext_matrix = encrypted_shift_matrix,
                shape = [k,a]
            )
            shift_matrix_id = "{}shiftmatrix".format(plaintext_matrix_id) # The id of the Shift matrix is formed
            put_shift_matrix_start_time = time.time()

            shift_matrix_chunks = Chunks.from_ndarray(
                ndarray      = shift_matrix,
                group_id     = shift_matrix_id,
                chunk_prefix = Some(shift_matrix_id),
                num_chunks   = num_chunks,
                )
            if shift_matrix_chunks.is_none:
                return Response (status=500, response= "something went wrong creating the chunks")
            
            # chunks_bytes = Utils.chunks_to_bytes_gen(
            #     chs = shift_matrix_chunks.unwrap()
            # )
            
            put_shift_matrix_result = await RoryCommon.delete_and_put_chunks(
                client = STORAGE_CLIENT,
                bucket_id      = BUCKET_ID,
                key            = shift_matrix_id,
                chunks         = shift_matrix_chunks.unwrap(),
                tags = {
                    "full_shape": str(shift_matrix.shape),
                    "full_dtype": str(shift_matrix.dtype)
                }
            )
            if put_shift_matrix_result.is_err:
                return Response ( status = 500, response = "Failed to put shiftmatrix")

            Cent_i= await RoryCommon.get_pyctxt(
                client = STORAGE_CLIENT,
                bucket_id = BUCKET_ID, 
                key       = cent_i_id, 
                ckks= ckks
            )
            Cent_j = await RoryCommon.get_pyctxt(
                client = STORAGE_CLIENT,
                bucket_id = BUCKET_ID, 
                key       = cent_j_id,
                ckks=ckks
            )
            decrypted_cent_i = ckks.decryptMatrix(
                ciphertext_matrix = Cent_i, 
                shape             = [1,k],
            )
            
            decrypted_cent_j = ckks.decryptMatrix(
                ciphertext_matrix = Cent_j, 
                shape             = [1,k],
            )
            min_error = 0.15
            isZero = Utils.verify_mean_error(
                old_matrix = decrypted_cent_i, 
                new_matrix = decrypted_cent_j, 
                min_error  = min_error
            )
            print("IS+_ZERO", isZero)

            status = Constants.ClusteringStatus.WORK_IN_PROGRESS #Status is updated

            run2_headers = {
                "Step-Index"             : "2",
                "Clustering-Status"      : str(status),
                "Shift-Matrix-Id"        : shift_matrix_id,
                "Plaintext-Matrix-Id"    : plaintext_matrix_id,
                "Encrypted-Matrix-Id"    : encrypted_matrix_id,
                "Encrypted-Matrix-Shape" : "({},{})".format(r,a),
                "Encrypted-Matrix-Dtype" : "float64",
                "Num-Chunks"             : str(num_chunks),
                "Iterations"             : str(iterations),
                "K"                      : str(k),
                "Experiment-Iteration"   : str(experiment_iteration), 
                "Max-Iterations"         : str(MAX_ITERATIONS),
                "Is-Zero"                : str(int(isZero))
            }

            worker_run2_response = worker.run(
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
            inner_interaction_service_time = endTime - inner_interaction_arrival_time

            logger.info({
                "event":"SKMEANS.ITERATION.COMPLETED",
                "algorithm":algorithm,
                "plaintext_matrix_id":plaintext_matrix_id,
                "worker_id":worker_id,
                "k":k,
                "iterations":iterations,
                "service_time":inner_interaction_service_time,
            })

        interaction_end_time     = time.time()
        interaction_service_time = interaction_end_time - interaction_arrival_time 
        worker_response_time     = endTime - worker_start_time
        response_time            = endTime - arrivalTime 

        logger.info({
            "event":"SKMEANS.COMPLETED",
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "encrypted_matrix_id":encrypted_matrix_id,
            "num_chunks":num_chunks,
            "worker_id":worker_id,
            "k":k,
            "n_iterations":iterations, 
            "max_iterations":MAX_ITERATIONS,
            "service_time_encrypted_matrix":segment_encrypt_service_time,
            "service_time_dm_generation":udm_gen_st,
            "service_time_manager":get_worker_service_time,
            "service_time_worker":worker_response_time,
            "service_time_client":service_time_client,
            "response_time_clustering":response_time,
            "iterations_service_time":interaction_service_time
        })
    
        return Response(
            response = json.dumps({
                "label_vector" : label_vector,
                "iterations":iterations,
                "algorithm":algorithm,
                "worker_id":worker_id,
                "service_time_manager":get_worker_service_time,
                "service_time_worker":worker_response_time,
                "service_time_client":service_time_client,
                "response_time_clustering":response_time,
            }),
            status   = 200,
            headers  = {}
        )

    except Exception as e:
        logger.error({
            "msg":str(e)
        })
        return Response(response= str(e) , status= 500)
    
       
#PCQ-DBSKMEANS
@clustering.route("/pqc/dbskmeans",methods = ["POST"])
async def pqc_dbskmeans():
    try:
        arrivalTime                  = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        TESTING                      = current_app.config.get("TESTING",True)
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        STORAGE_CLIENT:V4Client      = current_app.config.get("ASYNC_STORAGE_CLIENT")
        max_workers                  = int(current_app.config.get("MAX_WORKERS",2))
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        _num_chunks                  = current_app.config.get("NUM_CHUNKS",4)
        np_random                    = current_app.config.get("np_random")
        do_fdhope:DataOwner          = current_app.config.get("dataowner")

        if executor               == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm                 = Constants.ClusteringAlgorithms.DBSKMEANS_PQC
        algorithm_fdhope          = Constants.ClusteringAlgorithms.DBSKMEANS
        s                         = Session()
        request_headers           = request.headers #Headers for the request
        num_chunks                = int(request_headers.get("Num-Chunks",_num_chunks ) )
        plaintext_matrix_id       = request_headers.get("Plaintext-Matrix-Id","matrix0")
        encrypted_matrix_id       = "encrypted{}".format(plaintext_matrix_id) # The id of the encrypted matrix is built
        encrypted_udm_id          = "{}encryptedudm".format(plaintext_matrix_id) # The iudm id is built
        plaintext_matrix_filename = request_headers.get("Plaintext-Matrix-Filename","matrix0")
        extension                 = request_headers.get("Extension","csv")
        k                         = int(request_headers.get("K"))
        sens                      = float(request_headers.get("Sens","0.00000001"))
        experiment_iteration      = request_headers.get("Experiment-Iteration","0")
        MAX_ITERATIONS            = int(request_headers.get("Max-Iterations",current_app.config.get("MAX_ITERATIONS",10)))
        WORKER_TIMEOUT            = int(current_app.config.get("WORKER_TIMEOUT",3600))
        MICTLANX_TIMEOUT          = int(current_app.config.get("MICTLANX_TIMEOUT",3600))

        request_id                = "request{}".format(plaintext_matrix_id)
        plaintext_matrix_path     = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_filename, extension)
        
        init_sm_id = "{}initsm".format(plaintext_matrix_id)
        cent_i_id  = "{}centi".format(plaintext_matrix_id) #Build the id of Cent_i
        cent_j_id  = "{}centj".format(plaintext_matrix_id) #Build the id of Cent_j

        _round   = False
        decimals = 2

        path               = os.environ.get("KEYS_PATH","/rory/keys")
        ctx_filename       = os.environ.get("CTX_FILENAME","ctx")
        pubkey_filename    = os.environ.get("PUBKEY_FILENAME","pubkey")
        secretkey_filename = os.environ.get("SECRET_KEY_FILENAME","secretkey")
        backoff_factor = 0.5
        delay =2 
        max_retries = 10
        
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
        dataowner = DataOwnerPQC(scheme = ckks, sens=sens)

        logger.debug({
            "event":"DBSKMEANS.STARTED",
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "encrypted_matrix_id":encrypted_matrix_id,
            "plaintext_matrix_filename":plaintext_matrix_filename,
            "plaintext_matrix_path":plaintext_matrix_path,
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
        
        local_read_dataset_start_time = time.time()
        plaintext_matrix_result  = await RoryCommon.read_numpy_from(
            path      = plaintext_matrix_path,
            extension = extension,
        )
        if plaintext_matrix_result.is_ok:
            plaintext_matrix = plaintext_matrix_result.unwrap()
        else:
            raise plaintext_matrix_result.unwrap_err()

        local_read_dataset_st = time.time() - local_read_dataset_start_time
        
        plaintext_matrix = plaintext_matrix.astype(np.float64)

        r = plaintext_matrix.shape[0]
        a = plaintext_matrix.shape[1]

        # cores       = os.cpu_count()
        # max_workers = num_chunks if max_workers > num_chunks else max_workers
        # max_workers = cores if max_workers > cores else max_workers
        max_workers = Utils.get_workers(num_chunks=num_chunks)

        encryption_start_time = time.time()
        n                     = a*r
        encrypted_matrix_chunks =  RoryCommon.segment_and_encrypt_ckks_with_executor( #Encrypt 
            executor           = executor,
            key                = encrypted_matrix_id,
            plaintext_matrix   = plaintext_matrix,
            n                  = n,
            num_chunks         = num_chunks,
            _round             = _round,
            decimals           = decimals,
            path               = path,
            ctx_filename       = ctx_filename,
            pubkey_filename    = pubkey_filename,
            secretkey_filename = secretkey_filename
        )

        segment_encrypt_service_time = time.time() - encryption_start_time
        logger.info({
            "event":"SEGMENT.ENCRYPT.CKKS",
            "key":encrypted_matrix_id,
            "plaintext_matrix_shape":str(plaintext_matrix.shape),
            "plaintext_matrix_dtype":str(plaintext_matrix.dtype),
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "n":n,
            "num_chunks":num_chunks,
            "max_workers":max_workers,
            "service_time":segment_encrypt_service_time
        })
  
        put_chunks_start_time = time.time()

        put_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
            client = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = encrypted_matrix_id,
            chunks         = encrypted_matrix_chunks,
            tags = {
                "full_shape": str((r,a)),
                "full_dtype":"float64"
            }
        )
        if put_chunks_generator_results.is_err:
            return Response(status=500, response="Failed to put encrypted matrix")
        put_chunks_st = time.time() - put_chunks_start_time
        logger.info({
            "bucket_id":BUCKET_ID, 
            "key":encrypted_matrix_id,
            "response_time":put_chunks_st
        })

        logger.debug({
            "event":"UDM.GENERATION.BEFORE",
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "plaintext_matrix_shape":str(plaintext_matrix.shape),
            "plaintext_matrix_dtype":str(plaintext_matrix.dtype),
        })
        udm_start_time = time.time()
        udm            = do_fdhope.get_U(
            plaintext_matrix = plaintext_matrix,
            algorithm        = algorithm_fdhope
        )

        udm_shape = udm.shape
        udm_dtype = udm.dtype

        del plaintext_matrix
        
        udm_gen_st = time.time() - udm_start_time
        logger.info({
            "event":"UDM.GENERATION",
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "shape":str(udm.shape),
            "encrypted_udm_id":encrypted_udm_id,
            "service_time":udm_gen_st,
        })
        
        n         = r*r*a
        threshold = 0.0
        segment_encrypt_fdhope_start_time = time.time()
        logger.debug({
            "event":"SEGMENT.ENCRYPT.FDHOPE.BEFORE",
            "key":encrypted_udm_id,
            "udm_shape":str(udm_shape),
            "udm_dtype":str(udm_dtype),
            "n":n,
            "num_chunks":num_chunks,
            "max_workers":max_workers,
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "threshold":threshold
        })

        encrypted_matrix_UDM_chunks = RoryCommon.segment_and_encrypt_fdhope_with_executor( #Encrypt 
            executor   = executor,
            algorithm  = algorithm_fdhope,
            key        = encrypted_udm_id,
            dataowner  = do_fdhope,
            matrix     = udm,
            n          = n,
            num_chunks = num_chunks,
            sens       = sens,
        )
        
        segment_encrypt_fdhope_st = time.time() - segment_encrypt_fdhope_start_time

        logger.info({
            "event":"SEGMENT.ENCRYPT.FDHOPE",
            "key":encrypted_udm_id,
            "udm_shape":str(udm_shape),
            "udm_dtype":str(udm_dtype),
            "n":n,
            "num_chunks":num_chunks,
            "max_workers":max_workers,
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "threshold":threshold,
            "service_time":segment_encrypt_fdhope_st
        })
        

        put_chunks_start_time = time.time()
        
    
        put_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
            client = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = encrypted_udm_id,
            chunks         = encrypted_matrix_UDM_chunks,
            tags = {
                # "shape": str(udm_shape),
                "full_shape": str((r,r,a)), 
                "full_dtype":"float64"
            },
            timeout=MICTLANX_TIMEOUT
        )

        if put_chunks_generator_results.is_err:
            return Response(status=500, response="Failed to put encrypted udm matrix")
        logger.info({
            "event":"PUT",
            "bucket_id":BUCKET_ID,
            "key":encrypted_udm_id,
            "response_time":time.time() - put_chunks_start_time
        })

        service_time_client = time.time() - arrivalTime
        
        del udm 
        del encrypted_matrix_UDM_chunks

        zero_shiftmatrix = np.zeros((k, a))
        n2 = a*k
        
        init_shiftmatrix_start_time = time.time()
        
        encrypted_shiftmatrix_chunks = RoryCommon.segment_and_encrypt_ckks_with_executor( #Encrypt 
            executor           = executor,
            key                = init_sm_id,
            plaintext_matrix   = zero_shiftmatrix,
            n                  = n2,
            num_chunks         = num_chunks,
            _round             = _round,
            decimals           = decimals,
            path               = path,
            ctx_filename       = ctx_filename,
            pubkey_filename    = pubkey_filename,
            secretkey_filename = secretkey_filename
        )
        
        logger.info({
            "event":"SEGMENT.ENCRYPT.INITSHIFTMATRIX",
            "key":init_sm_id,
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "n":n,
            "num_chunks":num_chunks,
            "max_workers":max_workers,
            "service_time":time.time() - init_shiftmatrix_start_time
        })

   
        put_chunks_start_time = time.time()

        put_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
            client = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = init_sm_id,
            chunks         = encrypted_shiftmatrix_chunks,
            tags = {
                "shape": str((k,a)),
                "dtype":"float64"
            }
        )
        if put_chunks_generator_results.is_err:
            return Response(status=500, response="Failed to put encrypted init shift matrix")
        put_chunks_st = time.time() - put_chunks_start_time
        logger.info({
            "event":"DELETE.AND.PUT.CHUNKED",
            "bucket_id":BUCKET_ID,
            "key":init_sm_id,
            "response_time":put_chunks_st,
        })

        get_worker_start_time       = time.time()
        managerResponse:RoryManager = current_app.config.get("manager") # Communicates with the manager
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
            "event":"MANAGER.GET.WORKER1",
            "worker_id":worker_id,
            "port":port,
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "service_time":get_worker_service_time,
            "k":k,
        })
        
        worker_start_time = time.time()
        worker = RoryWorker( #Allows to establish the connection with the worker
            workerId  = worker_id,
            port      = port,
            session   = s,
            algorithm = algorithm,
        )
        status                      = Constants.ClusteringStatus.START #Set the status to start
        worker_run1_response        = None
        initial_encrypted_udm_shape = (r,r,a)
        interaction_arrival_time    = time.time()
        iterations                  = 0
        label_vector                = None

        while (status != Constants.ClusteringStatus.COMPLETED): #While the status is not completed
            
            inner_interaction_arrival_time = time.time()
            run1_headers  = {
                "Step-Index"             : "1",
                "Clustering-Status"      : str(status),
                "Plaintext-Matrix-Id"    : plaintext_matrix_id,
                "Request-Id"             : request_id,
                "Encrypted-Matrix-Id"    : encrypted_matrix_id,
                "Encrypted-Matrix-Shape" : "({},{})".format(r,a),
                "Encrypted-Matrix-Dtype" : "float64",
                "Encrypted-Udm-Dtype"    : "float64",
                "Encrypted-Udm-Shape"    : str(initial_encrypted_udm_shape),
                "Num-Chunks"             : str(num_chunks),
                "Iterations"             : str(iterations),
                "K"                      : str(k),
                "Experiment-Iteration"   : str(experiment_iteration), 
                "Max-Iterations"         : str(MAX_ITERATIONS) 
            }  
            
            worker_run1_response = worker.run(
                timeout = WORKER_TIMEOUT, 
                headers = run1_headers
            ) #Run 1 starts
            
            logger.info("worker response {}".format(worker_run1_response))
            worker_run1_status = worker_run1_response.status_code

            if worker_run1_status !=200:
                return Response(response="Worker error: {}".format(worker_run1_response.content),status=500)
            
            worker_run1_response.raise_for_status()
            # stringWorkerResponse      = worker_run1_response.content.decode("utf-8") #Response from worker
            jsonWorkerResponse        = worker_run1_response.json()
            # json.loads(stringWorkerResponse) #pass to json
            encrypted_shift_matrix_id = jsonWorkerResponse["encrypted_shift_matrix_id"]
            run1_service_time         = jsonWorkerResponse["service_time"]
            run1_n_iterations         = jsonWorkerResponse["n_iterations"]
            label_vector              = jsonWorkerResponse["label_vector"]

            logger.info({
                "event":"DBSKMEANS.RUN1",
                "algorithm":algorithm,
                "plaintext_matrix_id":plaintext_matrix_id,
                "step_index":"1",
                "clustering_status":str(status),
                "request_id":request_id,
                "encrypted_matrix_id":encrypted_matrix_id,
                "encrypted_matrix_shape":"({},{})".format(r,a),
                "encrypted_matrix_dtype":"float64", 
                "num_chunks":num_chunks,
                "iterations":iterations,
                "k":k, 
                "experiment_iteration":experiment_iteration,
                "max_iterations":MAX_ITERATIONS,
                "status":worker_run1_status,
                "worker_service_time": run1_service_time,
                "n_iterations":run1_n_iterations,
                "response_time":time.time() - inner_interaction_arrival_time
            })
            
            encrypted_shift_matrix_start_time = time.time()
            
            encrypted_shift_matrix = await RoryCommon.get_pyctxt(
                client    = STORAGE_CLIENT,
                bucket_id = BUCKET_ID, 
                key       = encrypted_shift_matrix_id,
                ckks = ckks,
                delay=delay,
                max_retries=max_retries,
                backoff_factor=backoff_factor,
                force=True
            )
            
            decrypt_start_time = time.time()
            shift_matrix = ckks.decryptMatrix( #Shift Matrix is decrypted
                ciphertext_matrix = encrypted_shift_matrix,
                shape = [k,a]
            )
            # raise Exception("BOOM!")

            encrypted_start_time = time.time()
            shift_matrix_ope_res = Fdhope.encryptMatrix( #Re-encrypt shift matrix with the FDHOPE scheme
                plaintext_matrix = shift_matrix,
                messagespace     = do_fdhope.messageIntervals,
                cipherspace      = do_fdhope.cypherIntervals,
                sens             = sens
            )        

            shift_matrix_ope = shift_matrix_ope_res.matrix
            shift_matrix_ope_shape = shift_matrix_ope.shape
            shift_matrix_ope_dtype = shift_matrix_ope.dtype
            logger.info({
                "event":"ENCRYPT.FDHOPE",
                "algorithm":algorithm,
                "plaintext_matrix_id":plaintext_matrix_id,
                "shift_matrix_shape":str(shift_matrix_ope_shape),
                "shift_matrix_dtype":str(shift_matrix_ope_dtype),
                "service_time":time.time() - encrypted_start_time
            })
            shift_matrix_id     = "{}shiftmatrix".format(plaintext_matrix_id) # The id of the Shift matrix is formed
            shift_matrix_ope_id = "{}shiftmatrixope".format(plaintext_matrix_id) # The id of the Shift matrix is formed
            
            put_matrix_start_time = time.time()
          
            maybe_shift_matrix_chunks = Chunks.from_ndarray(
                ndarray      = shift_matrix_ope,
                group_id     = shift_matrix_ope_id,
                chunk_prefix = Some(shift_matrix_ope_id),
                num_chunks   = num_chunks,
            )

            if maybe_shift_matrix_chunks.is_none:
                return Response(status= 500, response="something went wrong creating the chunks")


            
            encrypted_sm_ope_result = await RoryCommon.delete_and_put_chunks(
                client = STORAGE_CLIENT,
                bucket_id      = BUCKET_ID,
                key            = shift_matrix_ope_id,
                chunks         = maybe_shift_matrix_chunks.unwrap(),
                tags = {
                    "full_shape": str(shift_matrix_ope_shape),
                    "full_dtype": str(shift_matrix_ope_dtype)
                },
                timeout=MICTLANX_TIMEOUT
            )
            del maybe_shift_matrix_chunks
            del shift_matrix_ope
            if encrypted_sm_ope_result.is_err:
                return Response(status = 500, response="Failed to put encrypted shiftmatrix ope")
            
            Cent_i = await RoryCommon.get_pyctxt(
                client         = STORAGE_CLIENT,
                bucket_id      = BUCKET_ID,
                key            = cent_i_id,
                delay          = delay,
                backoff_factor = backoff_factor,
                force          = True,
                max_retries    = max_retries,
                ckks           = ckks,
            )
            # raise Exception("BOOM!")
     
            Cent_j = await RoryCommon.get_pyctxt(
                client = STORAGE_CLIENT,
                ckks = ckks,
                bucket_id = BUCKET_ID, 
                key       = cent_j_id,
                delay          = delay,
                backoff_factor = backoff_factor,
                force          = True,
                max_retries    = max_retries,
            )

            decrypted_cent_i = ckks.decryptMatrix(
                ciphertext_matrix = Cent_i, 
                shape             = [1,k],
            )
            
            decrypted_cent_j = ckks.decryptMatrix(
                ciphertext_matrix = Cent_j, 
                shape             = [1,k],
            )

            min_error = 0.15
            
            isZero = Utils.verify_mean_error(
                old_matrix = decrypted_cent_i, 
                new_matrix = decrypted_cent_j, 
                min_error  = min_error
            )

            status = Constants.ClusteringStatus.WORK_IN_PROGRESS #Status is updated

            run2_headers = {
                "Step-Index"             : "2",
                "Clustering-Status"      : str(status),
                "Shift-Matrix-Id"        : shift_matrix_id,
                "Plaintext-Matrix-Id"    : plaintext_matrix_id,
                "Encrypted-Matrix-Id"    : encrypted_matrix_id,
                "Encrypted-Matrix-Shape" : "({},{})".format(r,a),
                "Encrypted-Matrix-Dtype" : "float64",
                "Encrypted-Udm-Dtype"    : "float64",
                "Encrypted-Udm-Shape"    : str(initial_encrypted_udm_shape),
                "Num-Chunks"             : str(num_chunks),
                "Iterations"             : str(iterations),
                "K"                      : str(k),
                "Experiment-Iteration"   : str(experiment_iteration), 
                "Max-Iterations"         : str(MAX_ITERATIONS),
                "Is-Zero"                : str(int(isZero))
            }

            worker_run2_response = worker.run(
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
            inner_interaction_service_time = endTime - inner_interaction_arrival_time

            logger.info({
                "event":"SKMEANS.ITERATION.COMPLETED",
                "algorithm":algorithm,
                "plaintext_matrix_id":plaintext_matrix_id,
                "worker_id":worker_id,
                "k":k,
                "iterations":iterations,
                "service_time":inner_interaction_service_time,
            })

        interaction_end_time     = time.time()
        interaction_service_time = interaction_end_time - interaction_arrival_time 
        worker_response_time     = endTime - worker_start_time
        response_time            = endTime - arrivalTime 

        logger.info({
            "event":"DBSKMEANS.COMPLETED",
            "algorithm":algorithm,
            "plaintext_matrix_id":plaintext_matrix_id,
            "encrypted_matrix_id":encrypted_matrix_id,
            "num_chunks":num_chunks,
            "worker_id":worker_id,
            "k":k,
            "n_iterations":iterations, 
            "max_iterations":MAX_ITERATIONS,
            "service_time_encrypted_matrix":segment_encrypt_service_time,
            "service_time_dm_generation":udm_gen_st,
            "service_time_manager":get_worker_service_time,
            "service_time_worker":worker_response_time,
            "service_time_client":service_time_client,
            "response_time_clustering":response_time,
            "iterations_service_time":interaction_service_time
        })
    
        return Response(
            response = json.dumps({
                "label_vector" : label_vector,
                "iterations":iterations,
                "algorithm":algorithm,
                "worker_id":worker_id,
                "service_time_manager":get_worker_service_time,
                "service_time_worker":worker_response_time,
                "service_time_client":service_time_client,
                "response_time_clustering":response_time,
            }),
            status   = 200,
            headers  = {}
        )

    except Exception as e:
        logger.error({
            "msg":str(e)
        })
        return Response(response= str(e) , status= 500)