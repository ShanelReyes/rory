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
    """Health check and component identification endpoint.

    This method serves as a simple diagnostic tool to verify the availability 
    of the Client component and confirm its role within the Rory platform 
    architecture. It is used during deployment and orchestration to ensure 
    proper network connectivity between nodes.

    Returns:
        Response: A Flask Response object containing a JSON payload:
            component_type (str): Identifies this node as "client".
            
        Headers:
            Component-Type: "client"
            
        Status Code:
            200: If the service is running and reachable.
    """
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
    """
    This method handles the lifecycle of a clustering task by reading a local plaintext dataset, 
    externalizing it to the Cloud Storage System (CSS), requesting an available execution 
    node (Worker) from the Manager, and finally triggering the privacy-preserving 
    mining process.
    The method also tracks and logs performance metrics (service times) for the Client, 
    Manager, and Worker interactions to facilitate experimental auditing.

    Note:
    **Protocol Initiation**: All execution parameters for this algorithm are passed exclusively 
    via **HTTP Headers**. The request body must remain empty.
    
    Attributes:
        Plaintext-Matrix-Id (str): Unique identifier for the matrix in CSS. Defaults to "matrix-0".
        Plaintext-Matrix-Filename (str): Name of the local file (without extension). Defaults to "matrix-0".
        Extension (str): File extension of the local dataset (e.g., "csv", "npy"). Defaults to "csv".
        K (int): The number of clusters to form. Defaults to "3".
        Experiment-Id (str): A unique identifier for the execution trace. Defaults to a hex UUID.

    Returns:
        label_vector (list): The cluster assignment for each dataset point.
        iterations (int): Total iterations performed by the algorithm.
        algorithm (str): The name of the algorithm executed (kmeans).
        worker_id (str): Identifier of the worker node that processed the task.
        service_time_manager (float): Time spent coordinating with the Manager.
        service_time_worker (float): Time spent during Worker execution.
        service_time_client (float): Time spent in local data preparation/reading.
        response_time_clustering (float): Total end-to-end execution time.

    Raises:
        Exception: Captures and logs any failure during local I/O, CSS communication, 
            or Manager/Worker interaction, returning a 500 status code with the 
            error details in the headers.
    """
    try:
        arrivalTime                = time.time()
        logger                     = current_app.config["logger"]
        TESTING                    = current_app.config.get("TESTING",True)
        SOURCE_PATH                = current_app.config["SOURCE_PATH"]
        STORAGE_CLIENT:AsyncClient = current_app.config.get("ASYNC_STORAGE_CLIENT")
        BUCKET_ID:str              = current_app.config.get("BUCKET_ID","rory")
        WORKER_TIMEOUT             = int(current_app.config.get("WORKER_TIMEOUT",300))
        num_chunks                 = current_app.config.get("NUM_CHUNKS",4)
        algorithm                  = Constants.ClusteringAlgorithms.KMEANS
        s                          = Session()
        request_headers            = request.headers #Headers for the request
        plaintext_matrix_id        = request_headers.get("Plaintext-Matrix-Id","matrix-0")
        plaintext_matrix_filename  = request_headers.get("Plaintext-Matrix-Filename","matrix-0")
        extension                  = request_headers.get("Extension","csv")
        k                          = request_headers.get("K","3")
        experiment_id              = request_headers.get("Experiment-Id",uuid4().hex[:10])
        plaintext_matrix_path      = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_filename, extension)
        read_dataset_start_time    = time.time()
        plaintext_matrix_result    = await RoryCommon.read_numpy_from(
            path      = plaintext_matrix_path,
            extension = extension
        )

        if plaintext_matrix_result.is_ok:
            plaintextMatrix = plaintext_matrix_result.unwrap()
        else:
            raise plaintext_matrix_result.unwrap_err()

        local_read_entry = ExperimentLogEntry(
            event         = "LOCAL.READ", 
            experiment_id = experiment_id,
            algorithm     = algorithm,
            start_time    = read_dataset_start_time,
            end_time      = time.time(),
            id            = plaintext_matrix_id,
            num_chunks    = num_chunks,
            k             = k,
        )
        logger.info(local_read_entry.model_dump())

        
        put_pm_start_time = time.time()
        put_ptm_result = await RoryCommon.put_ndarray(
            client     = STORAGE_CLIENT, 
            key        = plaintext_matrix_id, 
            matrix     = plaintextMatrix,
            tags       = {}, 
            bucket_id  = BUCKET_ID
        )
        if put_ptm_result.is_err:
            error = put_ptm_result.unwrap_err()
            logger.error({
                "msg":str(error)
            })
            return Response(response=str(error), status=500)

        put_ptm_entry = ExperimentLogEntry(
            event         = "PUT",
            experiment_id = experiment_id,
            algorithm     = algorithm,
            start_time    = put_pm_start_time,
            end_time      = time.time(),
            id            = plaintext_matrix_id,
            worker_id     = "",
            num_chunks    = num_chunks,
            k             = k,
            workers       = 0,
        )
        logger.info(put_ptm_entry.model_dump())
        
        service_time_client   = time.time() - arrivalTime
        get_worker_start_time = time.time()
        manager:RoryManager   = current_app.config.get("manager") # Communicates with the manager
        get_worker_result     = manager.getWorker( #Gets the worker from the manager
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
        
        get_worker_end_time = time.time()
        worker_id           = "localhost" if TESTING else worker_id

        get_worker_entry = ExperimentLogEntry(
            event         = "GET.WORKER",
            experiment_id = experiment_id,
            algorithm     = algorithm,
            start_time    = get_worker_start_time,
            end_time      = get_worker_end_time,
            id            = plaintext_matrix_id,
            worker_id     = worker_id,
            num_chunks    = num_chunks,
            k             = k,
            workers       = 0
        )
        logger.info(get_worker_entry.model_dump())
        
        worker_run_1_start_time = time.time()
        manager_service_time = worker_run_1_start_time - get_worker_start_time
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
            event         = "WORKER.RUN.1",
            experiment_id = experiment_id,
            algorithm     = algorithm,
            start_time    = worker_run_1_start_time,
            end_time      = time.time(),
            id            = plaintext_matrix_id,
            worker_id     = worker_id,
            num_chunks    = num_chunks,
            k             = k,
            workers       = 0
        )
        logger.info(worker_run_1_entry.model_dump())

        jsonWorkerResponse   = workerResponse.json()
        iterations           = int(jsonWorkerResponse["iterations"]) # Extract the current number of iterations
        endTime              = time.time() # Get the time when it ends
        worker_response_time = endTime - worker_run_1_start_time
        response_time        = endTime - arrivalTime # Get the service time

        kmeans_completed_entry = ExperimentLogEntry(
            event         = "COMPLETED",
            experiment_id = experiment_id,
            algorithm     = algorithm,
            start_time    = arrivalTime,
            end_time      = time.time(),
            id            = plaintext_matrix_id,
            worker_id     = worker_id,
            num_chunks    = num_chunks,
            k             = k,
            iterations    = iterations,
            worker_time   = worker_response_time,
            client_time   = service_time_client,
            manager_time  = manager_service_time
        )
        logger.info(kmeans_completed_entry.model_dump())

        return Response(
            response = json.dumps({
                "label_vector" : jsonWorkerResponse.get("label_vector",[]),
                "iterations":iterations,
                "algorithm":algorithm,
                "worker_id":worker_id,
                "service_time_manager":manager_service_time,
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
    """
    This method implements an interactive, privacy-preserving K-Means clustering protocol 
    powered by Liu's homomorphic encryption scheme. The workflow is designed for 
    Privacy-Preserving Data Mining as a Service (PPDMaaS), where the Client (Data Owner) 
    remains the only entity capable of decrypting intermediate computations.

    Note:
    **Interactive Protocol**: This endpoint initiates the secure clustering flow. All parameters, 
    including cryptographic metadata, must be passed via **HTTP Headers**.

    Attributes:
        Plaintext-Matrix-Id (str): Unique ID for the matrix. Defaults to "matrix0".
        Plaintext-Matrix-Filename (str): Local filename for data reading. Defaults to "matrix0".
        Extension (str): File extension of the dataset. Defaults to "csv".
        K (int): Number of clusters to identify. **Required**.
        Max-Iterations (int): Maximum number of protocol rounds. Defaults to 10.
        Experiment-Id (str): Tracking ID for performance auditing.
        Experiment-Iteration (str): Current loop index of the experiment.

    Returns:
        label_vector (list): Final cluster assignments for the dataset.
        iterations (int): Actual number of iterations performed.
        algorithm (str): "skmeans".
        worker_id (str): ID of the node that performed the secure computations.
        service_time_manager (float): Time spent in Worker allocation.
        service_time_worker (float): Cumulative time of remote computation.
        service_time_client (float): Total local time (Encryption/Decryption/IO).
        response_time_clustering (float): End-to-end execution time.

    Raises:
        Exception: Returns a 500 status code if the process executor is missing, 
            or if failures occur during encryption, CSS I/O, or Worker interaction.
    """
    try:
        arrivalTime                  = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        TESTING                      = current_app.config.get("TESTING",True)
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        liu:Liu                      = current_app.config.get("liu")
        dataowner:DataOwner          = current_app.config.get("dataowner")
        STORAGE_CLIENT:AsyncClient   = current_app.config.get("ASYNC_STORAGE_CLIENT")
        max_workers                  = current_app.config.get("MAX_WORKERS",2)
        num_chunks                   = current_app.config.get("NUM_CHUNKS",4)
        security_level               = current_app.config.get("LIU_SECURITY_LEVEL",128)
        np_random:bool               = current_app.config.get("np_random")
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        
        if executor == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm                 = Constants.ClusteringAlgorithms.SKMEANS
        s                         = Session()
        request_headers           = request.headers #Headers for the request
        # num_chunks                = int(request_headers.get("Num-Chunks",_num_chunks))
        plaintext_matrix_id       = request_headers.get("Plaintext-Matrix-Id","matrix0")
        encrypted_matrix_id       = "encrypted{}".format(plaintext_matrix_id) # The id of the encrypted matrix is built
        udm_id                    = "{}udm".format(plaintext_matrix_id) # The iudm id is built
        plaintext_matrix_filename = request_headers.get("Plaintext-Matrix-Filename","matrix0")
        extension                 = request_headers.get("Extension","csv")
        experiment_id             = request_headers.get("Experiment-Id",uuid4().hex[:10])
        k                         = int(request_headers.get("K"))
        experiment_iteration      = request_headers.get("Experiment-Iteration","0")
        
        requestId                 = "request-{}".format(plaintext_matrix_id)
        m                         = dataowner.m
        plaintext_matrix_path     = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_filename, extension)
        cores                     = os.cpu_count()
        # max_workers               = Utils.get_workers(num_chunks=num_chunks)

        MAX_ITERATIONS            = int(request_headers.get("Max-Iterations",current_app.config.get("MAX_ITERATIONS",10)))
        WORKER_TIMEOUT            = int(current_app.config.get("WORKER_TIMEOUT",300))
        MICTLANX_TIMEOUT          = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
        MICTLANX_DELAY            = int(current_app.config.get("MICTLANX_DELAY","2"))
        MICTLANX_BACKOFF_FACTOR   = float(current_app.config.get("MICTLANX_BACKOFF_FACTOR","0.5"))
        MICTLANX_MAX_RETRIES      = int(current_app.config.get("MICTLANX_MAX_RETRIES","10")) 
        
        local_read_dataset_start_time = time.time()
        plaintext_matrix_result  = await RoryCommon.read_numpy_from(
            path      = plaintext_matrix_path,
            extension = extension,
        )
        if plaintext_matrix_result.is_ok:
            plaintext_matrix = plaintext_matrix_result.unwrap()
        else:
            raise plaintext_matrix_result.unwrap_err()
        
        r = plaintext_matrix.shape[0]
        a = plaintext_matrix.shape[1]

        local_read_entry = ExperimentLogEntry(
            event          = "LOCAL.READ",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = local_read_dataset_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = "",
            num_chunks     = num_chunks,
            k              = k,
            workers        = max_workers, 
            security_level = security_level,
            m              = m
        )
        logger.info(local_read_entry.model_dump())
        
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
            event          = "SEGMENT.ENCRYPT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = encryption_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = "",
            num_chunks     = num_chunks,
            k              = k,
            workers        = max_workers, 
            security_level = security_level,
            m              = m
        )
        logger.info(segment_encrypt_entry.model_dump())
        
        put_ptm_start_time = time.time()
        put_ptm_chunks_results = await RoryCommon.put_chunks(
            client    = STORAGE_CLIENT,
            bucket_id = BUCKET_ID,
            key       = encrypted_matrix_id,
            chunks    = encrypted_ptm_chunks,
            tags      = {
                "full_shape": str((r,a,m)),
                "full_dtype":"float32"
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
            client    = STORAGE_CLIENT,
            bucket_id = BUCKET_ID,
            key       = udm_id,
            chunks    = maybe_udm_matrix_chunks.unwrap(),
            tags      = {
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

        get_worker_start_time = time.time()
        manager:RoryManager   = current_app.config.get("manager") # Communicates with the manager
        get_worker_result     = manager.getWorker( #Gets the worker from the manager
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
        
        worker_start_time = time.time()
        worker = RoryWorker( #Allows to establish the connection with the worker
            workerId  = worker_id,
            port      = port,
            session   = s,
            algorithm = algorithm,
        )
        status               = Constants.ClusteringStatus.START #Set the status to start
        worker_run1_response = None
        iterations   = 0
        label_vector = None
        endTime      = 0 
        while (status != Constants.ClusteringStatus.COMPLETED): #While the status is not completed
            
            inner_interaction_arrival_time = time.time()
            run1_headers  = {
                "Step-Index"             : "1",
                "Clustering-Status"      : str(status),
                "Plaintext-Matrix-Id"    : plaintext_matrix_id,
                "Request-Id"             : requestId,
                "Encrypted-Matrix-Id"    : encrypted_matrix_id,
                "Encrypted-Matrix-Shape" : "({},{},{})".format(r,a,m),
                "Encrypted-Matrix-Dtype" : "float32",
                "Encrypted-Udm-Dtype"    : "float32",
                "Num-Chunks"             : str(num_chunks),
                "Iterations"             : str(iterations),
                "K"                      : str(k),
                "M"                      : str(m), 
                "Experiment-Iteration"   : str(experiment_iteration), 
                "Max-Iterations"         : str(MAX_ITERATIONS),
                "Experiment-Id"          : experiment_id
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
                event          = "RUN1",
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
                client         = STORAGE_CLIENT, 
                key            = encrypted_shift_matrix_id,
                bucket_id      = BUCKET_ID,
                backoff_factor = MICTLANX_BACKOFF_FACTOR,
                max_retries    = MICTLANX_MAX_RETRIES,
                delay          = MICTLANX_DELAY,
                timeout        = MICTLANX_TIMEOUT
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
                iterations     = run1_n_iterations
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
                iterations     = run1_n_iterations
            )
            logger.info(decrypt_entry.model_dump())

            shift_matrix    = shiftMatrix_chipher_schema_res.matrix
            shift_matrix_id = "{}shiftmatrix".format(plaintext_matrix_id) # The id of the Shift matrix is formed
            put_shift_matrix_start_time = time.time()

            maybe_shift_matrix_chunks = Chunks.from_ndarray(
                ndarray      = shift_matrix,
                group_id     = shift_matrix_id,
                chunk_prefix = Some(shift_matrix_id),
                num_chunks   = num_chunks,
                )

            if maybe_shift_matrix_chunks.is_none:
                raise "something went wrong creating the chunks"
            
            put_shift_matrix_result = await RoryCommon.delete_and_put_chunks(
                client    = STORAGE_CLIENT,
                bucket_id = BUCKET_ID,
                key       = shift_matrix_id,
                chunks    = maybe_shift_matrix_chunks.unwrap(),
                timeout   = MICTLANX_TIMEOUT,
                max_tries = MICTLANX_MAX_RETRIES,
                tags      = {
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
                iterations     = run1_n_iterations
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
                "Encrypted-Matrix-Dtype" : "float32",
                "Num-Chunks"             : str(num_chunks),
                "Iterations"             : str(iterations),
                "K"                      : str(k),
                "M"                      : str(m), 
                "Experiment-Iteration"   : str(experiment_iteration), 
                "Max-Iterations"         : str(MAX_ITERATIONS),
                "Experiment-Id"          : experiment_id
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
                iterations     = run1_n_iterations
            )
            logger.info(skmeans_iteration_completed_entry.model_dump())

        worker_response_time = endTime - worker_start_time
        response_time        = endTime - arrivalTime 

        clustering_completed_entry = ExperimentLogEntry(
            event          = "COMPLETED",
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
            iterations     = iterations,
            client_time    = service_time_client,
            manager_time   = get_worker_service_time,
            worker_time    = worker_response_time,
        )
        logger.info(clustering_completed_entry.model_dump())

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
    """
    This method implements a privacy-preserving protocol that ensures 
    both the Worker and the Manager remain "blind" to the underlying data. It 
    leverages a hybrid encryption approach, using Liu's homomorphic scheme for 
    initial data protection and the FDHOPE scheme for secure operations on distance 
    metrics.

    Note:
    **Multi-Party Security**: Parameters for the double-blind execution are handled via **HTTP Headers**. 
    Ensure the correct 'Experiment-Id' is provided for session tracking.

    Attributes:
        Plaintext-Matrix-Id (str): Unique ID for the matrix. Defaults to "matrix0".
        Plaintext-Matrix-Filename (str): Local file to be processed. Defaults to "matrix0".
        K (int): Number of clusters. Defaults to "3".
        Sens (float): Sensitivity parameter for the FDHOPE scheme. Defaults to 0.00000001.
        Max-Iterations (int): Maximum protocol rounds. Defaults to 10.
        Experiment-Id (str): Tracking ID for performance auditing.

    Returns:
        label_vector (list): Final cluster assignments.
        iterations (int): Total rounds performed.
        algorithm (str): "dbskmeans".
        worker_id (str): ID of the node that performed the secure computations.
        service_time_manager (float): Time spent in Worker allocation.
        service_time_worker (float): Cumulative time of remote computation.
        service_time_client (float): Total local time (Encryption/Decryption/IO).
        response_time_clustering (float): End-to-end execution time.

    Raises:
        Exception: Returns a 500 status code if the process executor is unavailable, 
            or if failures occur during the hybrid encryption/decryption chain 
            or CSS communication.
    """
    try:
        local_start_time             = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        TESTING                      = current_app.config.get("TESTING",True)
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        liu:Liu                      = current_app.config.get("liu")
        dataowner:DataOwner          = current_app.config.get("dataowner")
        STORAGE_CLIENT:AsyncClient   = current_app.config.get("ASYNC_STORAGE_CLIENT")
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        max_workers                  = current_app.config.get("MAX_WORKERS",2)
        num_chunks                   = current_app.config.get("NUM_CHUNKS",4)
        np_random                    = current_app.config.get("np_random")
        security_level               = current_app.config.get("LIU_SECURITY_LEVEL",128)
        if executor               == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm                 = Constants.ClusteringAlgorithms.DBSKMEANS
        s                         = Session()
        request_headers           = request.headers #Headers for the request
        plaintext_matrix_id       = request_headers.get("Plaintext-Matrix-Id","matrix0")
        encrypted_matrix_id       = "encrypted{}".format(plaintext_matrix_id) # The id of the encrypted matrix is built
        encrypted_udm_id          = "{}encryptedudm".format(plaintext_matrix_id) # The iudm id is built
        plaintext_matrix_filename = request_headers.get("Plaintext-Matrix-Filename","matrix0")
        extension                 = request_headers.get("Extension","csv")
        experiment_id             = request_headers.get("Experiment-Id",uuid4().hex[:10])
        m                         = dataowner.m
        k                         = request_headers.get("K","3")
        sens                      = float(request_headers.get("Sens","0.00000001"))
        experiment_iteration      = request_headers.get("Experiment-Iteration","0") 
        cores                     = os.cpu_count()
        # max_workers               = Utils.get_workers(num_chunks=num_chunks)

        MAX_ITERATIONS            = int(request_headers.get("Max-Iterations",current_app.config.get("MAX_ITERATIONS",10)))
        WORKER_TIMEOUT            = int(current_app.config.get("WORKER_TIMEOUT",3600))
        MICTLANX_TIMEOUT          = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
        MICTLANX_DELAY            = int(current_app.config.get("MICTLANX_DELAY","2"))
        MICTLANX_BACKOFF_FACTOR   = float(current_app.config.get("MICTLANX_BACKOFF_FACTOR","0.5"))
        MICTLANX_MAX_RETRIES      = int(current_app.config.get("MICTLANX_MAX_RETRIES","10")) 
        
        request_id                = "request{}".format(plaintext_matrix_id)
        plaintext_matrix_path     = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_filename, extension)

        local_read_dataset_start_time = time.time()
        plaintext_matrix_result       = await RoryCommon.read_numpy_from(
            path      = plaintext_matrix_path,
            extension = extension,
        )
        if plaintext_matrix_result.is_ok:
            plaintext_matrix = plaintext_matrix_result.unwrap()
        else:
            raise plaintext_matrix_result.unwrap_err()
        
        r = plaintext_matrix.shape[0]
        a = plaintext_matrix.shape[1]
        plaintext_matrix_dtype = plaintext_matrix.dtype
        plaintext_matrix_shape = plaintext_matrix.shape

        local_read_entry = ExperimentLogEntry(
            event          = "LOCAL.READ",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = local_read_dataset_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = "",
            num_chunks     = num_chunks,
            k              = k,
            workers        = max_workers, 
            security_level = security_level,
            m              = m
        )
        logger.info(local_read_entry.model_dump())

        n = a*r*int(m)

        encrypt_segment_start_time = time.time()
        encrypted_matrix_chunks = RoryCommon.segment_and_encrypt_liu_with_executor( #Encrypt 
            executor         = executor,
            key              = encrypted_matrix_id,
            dataowner        = dataowner,
            plaintext_matrix = plaintext_matrix,
            n                = n,
            np_random        = np_random,
            num_chunks       = num_chunks,
        )

        segment_encrypt_entry = ExperimentLogEntry(
            event          = "SEGMENT.ENCRYPT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = encrypt_segment_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = "",
            num_chunks     = num_chunks,
            k              = k,
            workers        = max_workers, 
            security_level = security_level,
            m              = m
        )
        logger.info(segment_encrypt_entry.model_dump())
        
        put_chunks_start_time = time.time()
        put_encrypted_matrix_result = await RoryCommon.delete_and_put_chunks(
            client    = STORAGE_CLIENT,
            bucket_id = BUCKET_ID,
            key       = encrypted_matrix_id,
            chunks    = encrypted_matrix_chunks,
            timeout   = MICTLANX_TIMEOUT,
            max_tries = MICTLANX_MAX_RETRIES,
            tags      = {
                "full_shape": str((r,a,m)),
                "full_dtype":"float32"
            },
        )

        put_encrypted_ptm_entry = ExperimentLogEntry(
            event          = "PUT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = put_chunks_start_time,
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
        udm_shape = udm.shape
        udm_dtype = udm.dtype

        # Plaintext matrix is useless from here to bottom. Free some memory:
        del plaintext_matrix

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

        n = r*r*a
        segment_encrypt_fdhope_start_time = time.time()

        encrypted_matrix_UDM_chunks = RoryCommon.segment_and_encrypt_fdhope_with_executor( #Encrypt 
            executor   = executor,
            algorithm  = algorithm,
            key        = encrypted_udm_id,
            dataowner  = dataowner,
            matrix     = udm,
            n          = n,
            num_chunks = num_chunks,
            sens       = sens
        )
        
        put_encrypted_ptm_entry = ExperimentLogEntry(
            event          = "SEGMENT.ENCRYPT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = segment_encrypt_fdhope_start_time,
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

        put_chunks_start_time = time.time()
    
        put_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
            client    = STORAGE_CLIENT,
            bucket_id = BUCKET_ID,
            key       = encrypted_udm_id,
            chunks    = encrypted_matrix_UDM_chunks,
            timeout   = MICTLANX_TIMEOUT,
            max_tries = MICTLANX_MAX_RETRIES,
            tags      = {
                "full_shape": str((r,r,a)),
                "full_dtype":"float32"
            },
        )

        if put_chunks_generator_results.is_err:
            return Response("Put chunks failed: UDM",status=500)

        put_encrypted_ptm_entry = ExperimentLogEntry(
            event          = "PUT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = put_chunks_start_time,
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

        service_time_client = time.time() - local_start_time
        del udm 
        del encrypted_matrix_UDM_chunks
        
        # Manager
        get_worker_start_time = time.time()
        managerResponse:RoryManager = current_app.config.get("manager") # Communicates with the manager
        
        get_worker_result = managerResponse.getWorker( #Gets the worker from the manager
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

        worker_start_time = time.time()
        worker         = RoryWorker( #Allows to establish the connection with the worker
            workerId   = worker_id,
            port       = port,
            session    = s,
            algorithm  = algorithm
        )
        status = Constants.ClusteringStatus.START #Set the status to start
        worker_run2_response = None
        initial_encrypted_udm_shape = (r,r,a)
        global_start_time = time.time()
        label_vector = []
        iterations   = 0

        while (status != Constants.ClusteringStatus.COMPLETED): #While the status is not completed
            inner_interaction_arrival_time = time.time()
            
            run1_headers = {
                "Start-Time":str(global_start_time),
                "Step-Index"             : "1",
                "Clustering-Status"      : str(status),
                "Plaintext-Matrix-Id"    : plaintext_matrix_id,
                "Encrypted-Matrix-Id"    : encrypted_matrix_id,
                "Encrypted-Matrix-Shape" : "({},{},{})".format(r,a,m),
                "Encrypted-Matrix-Dtype" : "float32",
                "Encrypted-Udm-Shape"    : str(initial_encrypted_udm_shape),
                "Encrypted-Udm-Dtype"    : "float32",
                "Num-Chunks"             : str(num_chunks),
                "Iterations"             : str(iterations),
                "K"                      : str(k),
                "M"                      : str(m), 
                "Experiment-Iteration"   : str(experiment_iteration), 
                "Max-Iterations"         : str(MAX_ITERATIONS) 
            }
            workerResponse1 = worker.run(timeout = WORKER_TIMEOUT,headers =run1_headers) #Run 1 starts
            workerResponse1.raise_for_status()
            
            jsonWorkerResponse        = workerResponse1.json()
            encrypted_shift_matrix_id = jsonWorkerResponse["encrypted_shift_matrix_id"]
            run1_service_time         = jsonWorkerResponse['service_time']
            label_vector              = jsonWorkerResponse["label_vector"]
            run1_n_iterations         = jsonWorkerResponse["n_iterations"]
            
            run1_worker_entry = ExperimentLogEntry(
                event          = "RUN1",
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

            get_matrix_start_time = time.time()
            encryptedShiftMatrix = await RoryCommon.get_and_merge(
                bucket_id      = BUCKET_ID,
                key            = encrypted_shift_matrix_id,
                client         = STORAGE_CLIENT, 
                timeout        = MICTLANX_TIMEOUT,
                backoff_factor = MICTLANX_BACKOFF_FACTOR,
                delay          = MICTLANX_DELAY,
                max_retries    = MICTLANX_MAX_RETRIES,
            )

            encryptedShiftMatrix_shape = encryptedShiftMatrix.shape
            encryptedShiftMatrix_dtype = encryptedShiftMatrix.dtype

            get_encrypted_sm_entry = ExperimentLogEntry(
                event          = "GET",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = get_matrix_start_time,
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
            logger.info(get_encrypted_sm_entry.model_dump())


            decrypt_start_time = time.time()
            cipher_schema_res  = liu.decryptMatrix( #Shift Matrix is decrypted
                ciphertext_matrix = encryptedShiftMatrix,
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
                iterations     = run1_n_iterations
            )
            logger.info(decrypt_entry.model_dump())

            del encryptedShiftMatrix
            
            shift_matrix         = cipher_schema_res.matrix
            encrypted_start_time = time.time()
            fdhope_encrypted_shift_matrix = Fdhope.encryptMatrix( #Re-encrypt shift matrix with the FDHOPE scheme
                plaintext_matrix = shift_matrix, 
                messagespace     = dataowner.messageIntervals,
                cipherspace      = dataowner.cypherIntervals
            )
            
            del shift_matrix
            shift_matrix_ope       = fdhope_encrypted_shift_matrix.matrix
            shift_matrix_ope_shape = shift_matrix_ope.shape
            shift_matrix_ope_dtype = shift_matrix_ope.dtype
            
            encrypt_entry = ExperimentLogEntry(
                event          = "ENCRYPT",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = encrypted_start_time,
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
            logger.info(encrypt_entry.model_dump())

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
            
            t_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
                client    = STORAGE_CLIENT,
                bucket_id = BUCKET_ID,
                key       = shift_matrix_ope_id,
                chunks    = maybe_shift_matrix_chunks.unwrap(),
                timeout   = MICTLANX_TIMEOUT,
                max_tries = MICTLANX_MAX_RETRIES,
                tags      = {
                    "full_shape": str(shift_matrix_ope_shape),
                    "full_dtype": str(shift_matrix_ope_dtype)
                },
            )
            del maybe_shift_matrix_chunks
            del shift_matrix_ope
            
            put_sm_entry = ExperimentLogEntry(
                event          = "PUT",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = put_matrix_start_time,
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
            logger.info(put_sm_entry.model_dump())
    
            status = Constants.ClusteringStatus.WORK_IN_PROGRESS #Status is updated
            run2_headers = {
                "Start-Time":str(global_start_time),
                "Step-Index"             : "2",
                "Clustering-Status"      : str(status),
                "Shift-Matrix-Id"        : shift_matrix_id,
                "Shift-Matrix-Ope-Id"    : shift_matrix_ope_id,
                "Plaintext-Matrix-Id"    : plaintext_matrix_id,
                "Encrypted-Matrix-Id"    : encrypted_matrix_id,
                "Encrypted-Matrix-Shape" : "({},{},{})".format(r,a,m),
                "Encrypted-Matrix-Dtype" : "float32",
                "Encrypted-Udm-Dtype"    : "float32",
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
            run2_json          = worker_run2_response.json()
            initial_encrypted_udm_shape  = eval(run2_json["encrypted_udm_shape"])
            run2_service_time  = run2_json["service_time"]
            
            iterations+=1
            if (iterations >= MAX_ITERATIONS): #If the number of iterations is equal to the maximum
                status = Constants.ClusteringStatus.COMPLETED #Change the status to complete
            else: 
                status = int(worker_run2_response.headers.get("Clustering-Status",Constants.ClusteringStatus.WORK_IN_PROGRESS)) #Status is maintained
            end_time = time.time() # Get the time when it ends
            
            dbskmeans_iteration_completed_entry = ExperimentLogEntry(
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
                iterations     = run1_n_iterations
            )
            logger.info(dbskmeans_iteration_completed_entry.model_dump())


        interaction_end_time = time.time()
        service_time         = interaction_end_time - global_start_time
        worker_end_time      = time.time()
        worker_response_time = worker_end_time - worker_start_time
        response_time        = end_time - local_start_time 

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
            workers        = max_workers,
            security_level = security_level,
            m              = m,
            iterations     = iterations,
            client_time    = service_time_client,
            manager_time   = get_worker_service_time,
            worker_time    = worker_response_time,
        )
        logger.info(clustering_completed_entry.model_dump())

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
    """
    This method implements a non-iterative, privacy-preserving clustering protocol 
    based on nearest neighbors. It utilizes a Double-Blind Secure (DBS) approach 
    where sensitive data and distance metrics are protected using a combination of 
    Liu's homomorphic encryption and the FDHOPE scheme.

    Note:
    All identifiers for the input matrices and distance metrics are extracted from **HTTP Headers**.

    Attributes:
        Plaintext-Matrix-Id (str): Unique ID for the matrix. Defaults to "matrix0".
        Plaintext-Matrix-Filename (str): Local file to be processed. Defaults to "matrix-0".
        Sens (float): Sensitivity parameter for FDHOPE encryption. Defaults to 0.00000001.
        Threshold (float): Distance threshold for clustering. If -1, it is calculated 
            automatically from the dataset.
        Experiment-Id (str): Tracking ID for performance auditing.

    Returns:
        label_vector (list): The resulting cluster assignments.
        algorithm (str): "dbsnnc".
        worker_id (str): ID of the node that performed the secure computations.
        service_time_manager (float): Time spent in Worker allocation.
        service_time_worker (float): Cumulative time of remote computation.
        service_time_client (float): Total local time (Encryption/Decryption/IO).
        response_time_clustering (float): End-to-end execution time.
    Raises:
        Exception: Returns a 500 status code if the process executor is missing, 
            or if errors occur during encryption, CSS communication, or Worker execution.
    """
    try:
        local_start_time             = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        TESTING                      = current_app.config.get("TESTING",True)
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        dataowner:DataOwner          = current_app.config.get("dataowner")
        STORAGE_CLIENT:AsyncClient   = current_app.config.get("ASYNC_STORAGE_CLIENT")
        max_workers                  = current_app.config.get("MAX_WORKERS",2)
        num_chunks                   = current_app.config.get("NUM_CHUNKS",4)
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        np_random                    = current_app.config.get("np_random")
        securitylevel                = current_app.config.get("LIU_SECURITY_LEVEL",128)
        if executor                  == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm                 = Constants.ClusteringAlgorithms.DBSNNC
        s                         = Session()
        request_headers           = request.headers #Headers for the request
        # num_chunks                = int(request_headers.get("Num-Chunks",_num_chunks))
        plaintext_matrix_id       = request_headers.get("Plaintext-Matrix-Id","matrix0")
        encrypted_matrix_id       = "encrypted{}".format(plaintext_matrix_id) # The id of the encrypted matrix is built
        dm_id                     = "{}dm".format(plaintext_matrix_id)
        encrypted_dm_id           = "{}encrypteddm".format(plaintext_matrix_id) # The iudm id is built
        plaintext_matrix_filename = request_headers.get("Plaintext-Matrix-Filename","matrix-0")
        extension                 = request_headers.get("Extension","csv")
        experiment_id             = request_headers.get("Experiment-Id",uuid4().hex[:10])
        m                         = dataowner.m
        sens                      = float(request_headers.get("Sens","0.00000001"))
        threshold                 = float(request_headers.get("Threshold",-1))
        request_id                = "request{}".format(plaintext_matrix_id)
        plaintext_matrix_path     = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_filename, extension)
        experiment_iteration      = request_headers.get("Experiment-Iteration","0")
        cores                     = os.cpu_count()
        # max_workers               = Utils.get_workers(num_chunks=num_chunks)

        WORKER_TIMEOUT          = int(current_app.config.get("WORKER_TIMEOUT",300))
        MICTLANX_TIMEOUT        = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
        MICTLANX_MAX_RETRIES    = int(current_app.config.get("MICTLANX_MAX_RETRIES","10"))

        local_read_dataset_start_time = time.time()
        plaintext_matrix_result  = await RoryCommon.read_numpy_from(
            path      = plaintext_matrix_path,
            extension = extension,
        )
        if plaintext_matrix_result.is_ok:
            plaintext_matrix = plaintext_matrix_result.unwrap()
        else:
            raise plaintext_matrix_result.unwrap_err()

        r = plaintext_matrix.shape[0]
        a = plaintext_matrix.shape[1]
        
        encryption_start_time = time.time()

        local_read_entry = ExperimentLogEntry(
            event          = "LOCAL.READ",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = local_read_dataset_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = "",
            num_chunks     = num_chunks,
            workers        = max_workers,
            m              = m,
            security_level = securitylevel
        )
        logger.info(local_read_entry.model_dump())

        n = r*a*m

        segment_encrypt_start_time = time.time()
        encrypted_matrix_chunks = RoryCommon.segment_and_encrypt_liu_with_executor( #Encrypt 
            executor         = executor,
            key              = encrypted_matrix_id,
            dataowner        = dataowner,
            plaintext_matrix = plaintext_matrix,
            n                = n,
            np_random        = np_random,
            num_chunks       = num_chunks
        )
        
        segment_encrypt_entry = ExperimentLogEntry(
            event          = "SEGMENT.ENCRYPT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = segment_encrypt_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = "",
            num_chunks     = num_chunks,
            workers        = max_workers, 
            m              = m,
            security_level = securitylevel
        )
        logger.info(segment_encrypt_entry.model_dump())
        
        put_chunks_start_time = time.time()
        put_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
            client    = STORAGE_CLIENT,
            bucket_id = BUCKET_ID,
            key       = encrypted_matrix_id,
            chunks    = encrypted_matrix_chunks,
            timeout   = MICTLANX_TIMEOUT,
            max_tries = MICTLANX_MAX_RETRIES,
            tags      = {
                "full_shape": str((r,a,m)),
                "full_dtype":"float32"
            }
        )

        put_encrypted_ptm_entry = ExperimentLogEntry(
            event          = "PUT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = put_chunks_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = "",
            num_chunks     = num_chunks,
            workers        = max_workers,
            m              = m,
            security_level = securitylevel
        )
        logger.info(put_encrypted_ptm_entry.model_dump())

        dm_start_time = time.time()
        dm            = dataowner.get_U(
            plaintext_matrix = plaintext_matrix,
            algorithm        = algorithm
        )

        udm_gen_entry = ExperimentLogEntry(
            event          = "UDM.GENERATION",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = dm_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = "",
            num_chunks     = num_chunks,
            workers        = max_workers,
            m              = m,
            security_level = securitylevel
        )
        logger.info(udm_gen_entry.model_dump())

        if threshold==-1:
            threshold = RoryUtils.get_threshold(
                distance_matrix = dm
        )
        n = r*r

        segment_encrypt_fdhope_start_time = time.time()
        encrypted_matrix_DM_chunks = RoryCommon.segment_and_encrypt_fdhope_with_executor( #Encrypt 
            executor   = executor,
            algorithm  = algorithm,
            key        = encrypted_dm_id,
            dataowner  = dataowner,
            matrix     = dm,
            n          = n,
            num_chunks = num_chunks,
            sens       = sens,
        )

        segment_encrypt_entry = ExperimentLogEntry(
            event          = "SEGMENT.ENCRYPT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = segment_encrypt_fdhope_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = "",
            num_chunks     = num_chunks,
            workers        = max_workers, 
            m              = m,
            security_level = securitylevel
        )
        logger.info(segment_encrypt_entry.model_dump())
        put_chunks_start_time = time.time()

        put_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
            client    = STORAGE_CLIENT,
            bucket_id = BUCKET_ID,
            key       = encrypted_dm_id,
            chunks    = encrypted_matrix_DM_chunks,
            timeout   = MICTLANX_TIMEOUT,
            max_tries = MICTLANX_MAX_RETRIES,
            tags      = {
                "full_shape": str((r,r)),
                "full_dtype":"float32"
            }
        )

        udm_put_entry = ExperimentLogEntry(
            event          = "PUT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = put_chunks_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = "",
            num_chunks     = num_chunks,
            workers        = max_workers,
            m              = m,
            security_level = securitylevel
        )
        logger.info(udm_put_entry.model_dump())
        
        encrypted_threshold = Fdhope.encrypt( #Threshold is encrypted
            plaintext    = threshold,
            messagespace = dataowner.messageIntervals, 
            cipherspace  = dataowner.cypherIntervals,
            sens         = sens,
        )
 
        service_time_client         = time.time() - local_start_time
        get_worker_start_time       = time.time()
        managerResponse:RoryManager = current_app.config.get("manager") # Communicates with the manager
        
        get_worker_result = managerResponse.getWorker( #Gets the worker from the manager
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

        get_worker_entry = ExperimentLogEntry(
            event          = "GET.WORKER",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_worker_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            workers        = max_workers,
            m              = m,
            security_level = securitylevel
        )
        logger.info(get_worker_entry.model_dump())

        worker_start_time = time.time()
        worker = RoryWorker( #Allows to establish the connection with the worker
            workerId  = worker_id,
            port      = port,
            session   = s,
            algorithm = algorithm
        )
        dm_shape = (r,r)

        encrypted_matrix_shape = (r,a,m)
        encrypted_matrix_dtype = "float32"
        run_headers = {
            "Plaintext-Matrix-Id"    : plaintext_matrix_id,
            "Request-Id"             : request_id,
            "Encrypted-Matrix-Id"    : encrypted_matrix_id,
            "Encrypted-Matrix-Shape" : str(encrypted_matrix_shape),
            "Encrypted-Matrix-Dtype" : encrypted_matrix_dtype,
            "Encrypted-Dm-Id"        : encrypted_dm_id,
            "Encrypted-Dm-Shape"     : str(dm_shape),
            "Encrypted-Dm-Dtype"     : "float32",
            "Num-Chunks"             : str(num_chunks),
            "M"                      : str(m),
            "Encrypted-Threshold"    : str(encrypted_threshold),
            "Dm-Shape"               : str(dm_shape),
            "Dm-Dtype"               : "float32",
        }

        run1_response = worker.run(
            timeout = WORKER_TIMEOUT, 
            headers = run_headers
        )
        run1_response.raise_for_status()
        
        jsonWorkerResponse   = run1_response.json()
        endTime              = time.time() # Get the time when it ends
        worker_service_time  = jsonWorkerResponse["service_time"]
        label_vector         = jsonWorkerResponse["label_vector"]
        response_time        = endTime - local_start_time # Get the service time
        worker_end_time      = time.time()
        worker_response_time = worker_end_time - worker_start_time
        
        clustering_completed_entry = ExperimentLogEntry(
            event          = "COMPLETED",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = local_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            workers        = max_workers,
            m              = m,
            security_level = securitylevel,
            client_time    = service_time_client,
            manager_time   = get_worker_service_time,
            worker_time    = worker_response_time
        )
        logger.info(clustering_completed_entry.model_dump())

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
    """
    This method implements a distributed version of the Nearest Neighbor Clustering 
    algorithm. Unlike its secure counterpart (DBSNNC), this version operates on 
    plaintext data externalized to the Cloud Storage System (CSS), focusing on 
    performance and orchestration within the Rory platform architecture.

    Note:
    All identifiers for the input matrices and distance metrics are extracted from **HTTP Headers**.

    Attributes:
        Plaintext-Matrix-Id (str): Unique identifier for the matrix in CSS. 
            Defaults to "matrix0".
        Plaintext-Matrix-Filename (str): Local filename (without extension). 
            Defaults to "matrix-0".
        Extension (str): Dataset file extension (e.g., "csv"). Defaults to "csv".
        Threshold (float): Distance limit for clustering. If -1, it is calculated 
            dynamically using platform utilities.
        Experiment-Id (str): Unique ID for performance tracking and logging.

    Returns:
        label_vector (list): Final cluster assignments for each data point.
        algorithm (str): "nnc".
        worker_id (str): ID of the worker node that processed the task.
        service_time_manager (float): Time spent coordinating with the Manager.
        service_time_worker (float): Time spent during Worker execution.
        service_time_client (float): Time spent in local data preparation and IO.
        response_time_clustering (float): Total end-to-end execution time.

    Raises:
        Exception: Returns a 500 status code if the process executor is missing, 
            or if failures occur during local I/O, CSS communication, or 
            Worker interaction.
    """
    try:
        local_start_time             = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        TESTING                      = current_app.config.get("TESTING",True)
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        dataowner:DataOwner          = current_app.config.get("dataowner")
        STORAGE_CLIENT:AsyncClient   = current_app.config.get("ASYNC_STORAGE_CLIENT")
        max_workers                  = current_app.config.get("MAX_WORKERS",2)
        num_chunks                   = current_app.config.get("NUM_CHUNKS",4)
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        if executor == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm                 = Constants.ClusteringAlgorithms.NNC
        s                         = Session()
        request_headers           = request.headers #Headers for the request
        # num_chunks                = int(request_headers.get("Num-Chunks",1))
        plaintext_matrix_id       = request_headers.get("Plaintext-Matrix-Id","matrix0")
        dm_id                     = "{}dm".format(plaintext_matrix_id)
        plaintext_matrix_filename = request_headers.get("Plaintext-Matrix-Filename","matrix-0")
        extension                 = request_headers.get("Extension","csv")
        request_id                = "request{}".format(plaintext_matrix_id)
        threshold                 = float(request_headers.get("Threshold",-1))
        plaintext_matrix_path     = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_filename, extension)
        experiment_id             = request_headers.get("Experiment-Id",uuid4().hex[:10])
        WORKER_TIMEOUT            = int(current_app.config.get("WORKER_TIMEOUT",300))
        MICTLANX_TIMEOUT          = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
        MICTLANX_MAX_RETRIES      = int(current_app.config.get("MICTLANX_MAX_RETRIES","10"))
        # max_workers               = Utils.get_workers(num_chunks=num_chunks)

        local_read_dataset_start_time = time.time()
        plaintext_matrix_result = await RoryCommon.read_numpy_from( 
            path      = plaintext_matrix_path,
            extension = extension,
        )

        if plaintext_matrix_result.is_ok:
            plaintext_matrix = plaintext_matrix_result.unwrap()
        else:
            raise plaintext_matrix_result.unwrap_err()
        
        r = plaintext_matrix.shape[0]
        a = plaintext_matrix.shape[1]
        
        local_read_entry = ExperimentLogEntry(
            event         = "LOCAL.READ",
            experiment_id = experiment_id,
            algorithm     = algorithm,
            start_time    = local_read_dataset_start_time,
            end_time      = time.time(),
            id            = plaintext_matrix_id,
            worker_id     = "",
            num_chunks    = num_chunks,
        )
        logger.info(local_read_entry.model_dump())

        put_ptm_start_time = time.time()

        plaintext_matrix_chunks = Chunks.from_ndarray(
            ndarray      = plaintext_matrix,
            group_id     = plaintext_matrix_id,
            chunk_prefix = Some(plaintext_matrix_id),
            num_chunks   = num_chunks,
        )

        if plaintext_matrix_chunks.is_none:
            raise "something went wrong creating the chunks"

        t_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
            client    = STORAGE_CLIENT,
            bucket_id = BUCKET_ID,
            key       = plaintext_matrix_id,
            chunks    = plaintext_matrix_chunks.unwrap(),
            timeout   = MICTLANX_TIMEOUT,
            max_tries = MICTLANX_MAX_RETRIES,
            tags      = {
                "full_shape": str(plaintext_matrix.shape),
                "full_dtype": str(plaintext_matrix.dtype)
            }
        )

        put_encrypted_ptm_entry = ExperimentLogEntry(
            event          = "PUT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = put_ptm_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = "",
            num_chunks     = num_chunks
        )
        logger.info(put_encrypted_ptm_entry.model_dump())
     
        dm_start_time = time.time()
        dm            = dataowner.get_U(
            plaintext_matrix = plaintext_matrix,
            algorithm        = algorithm
        )
        
        dm_gen_entry = ExperimentLogEntry(
            event         = "DM.GENERATION",
            experiment_id = experiment_id,
            algorithm     = algorithm,
            start_time    = dm_start_time,
            end_time      = time.time(),
            id            = plaintext_matrix_id,
            worker_id     = "",
            num_chunks    = num_chunks,
        )
        logger.info(dm_gen_entry.model_dump())

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
        
        put_dm_start_time = time.time()        
        put_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
            client    = STORAGE_CLIENT,
            bucket_id = BUCKET_ID,
            key       = dm_id,
            chunks    = maybe_dm_chunks.unwrap(),
            timeout   = MICTLANX_TIMEOUT,
            max_tries = MICTLANX_MAX_RETRIES,
            tags      = {
                "full_shape":str(dm.shape),
                "full_dtype":str(dm.dtype)
            }
        )

        service_time_client = time.time() - local_start_time
        
        dm_put_entry = ExperimentLogEntry(
            event          = "PUT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = put_dm_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = "",
            num_chunks     = num_chunks,
        )
        logger.info(dm_put_entry.model_dump())

        get_worker_start_time = time.time()
        managerResponse:RoryManager = current_app.config.get("manager") # Communicates with the manager
        get_worker_result = managerResponse.getWorker( #Gets the worker from the manager
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

        get_worker_entry = ExperimentLogEntry(
            event          = "GET.WORKER",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = get_worker_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks
        )
        logger.info(get_worker_entry.model_dump())

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
        jsonWorkerResponse   = worker_response.json()
        end_time             = time.time() # Get the time when it ends
        worker_service_time  = jsonWorkerResponse["service_time"]
        label_vector         = jsonWorkerResponse["label_vector"]
        response_time        = end_time - local_start_time # Get the service time
        worker_end_time      = time.time()
        worker_response_time = worker_end_time - worker_start_time
        
        clustering_completed_entry = ExperimentLogEntry(
            event          = "COMPLETED",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = local_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = worker_id,
            num_chunks     = num_chunks,
            workers        = max_workers,
            client_time    = service_time_client,
            manager_time   = get_worker_service_time,
            worker_time    = worker_response_time,
        )
        logger.info(clustering_completed_entry.model_dump())

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
    """
    This method implements a clustering protocol using the 
    CKKS homomorphic encryption scheme. It is specifically designed 
    for Post-Quantum Privacy-Preserving Data Mining as a Service (PPDMaaS), allowing 
    complex floating-point computations on encrypted data while the Client 
    retains the secret key.

    Note:
    **Post-Quantum Parameters**: Security levels and CKKS-specific metadata are passed via **HTTP Headers**. 
    Body content will be ignored.

    Attributes:
        Plaintext-Matrix-Id (str): Unique ID for the matrix. Defaults to "matrix0".
        Plaintext-Matrix-Filename (str): Local file to be processed. Defaults to "matrix0".
        K (int): Number of clusters. **Required**.
        Max-Iterations (int): Maximum protocol rounds. Defaults to 10.
        Experiment-Id (str): Tracking ID for performance auditing.

    Returns:
        label_vector (list): Final cluster assignments for the dataset.
        iterations (int): Actual number of iterations performed.
        algorithm (str): "skmeans pqc".
        worker_id (str): ID of the node that performed the secure computations.
        service_time_manager (float): Time spent in Worker allocation.
        service_time_worker (float): Cumulative time of remote computation.
        service_time_client (float): Total local time (Encryption/Decryption/IO).
        response_time_clustering (float): End-to-end execution time.
        

    Raises:
        Exception: Returns a 500 status code if the process executor is missing, 
            CKKS context fails, or communication errors occur.
    """
    try:
        arrivalTime                  = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        TESTING                      = current_app.config.get("TESTING",True)
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        STORAGE_CLIENT:AsyncClient   = current_app.config.get("ASYNC_STORAGE_CLIENT")
        max_workers                  = current_app.config.get("MAX_WORKERS",2)
        num_chunks                   = current_app.config.get("NUM_CHUNKS",4)
        np_random                    = current_app.config.get("np_random")
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        security_level               = current_app.config.get("LIU_SECURITY_LEVEL",128)
        
        if executor == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm                 = Constants.ClusteringAlgorithms.SKMEANS_PQC
        s                         = Session()
        request_headers           = request.headers #Headers for the request
        # num_chunks                = int(request_headers.get("Num-Chunks",_num_chunks))
        plaintext_matrix_id       = request_headers.get("Plaintext-Matrix-Id","matrix0")
        encrypted_matrix_id       = "encrypted{}".format(plaintext_matrix_id) # The id of the encrypted matrix is built
        udm_id                    = "{}udm".format(plaintext_matrix_id) # The iudm id is built
        plaintext_matrix_filename = request_headers.get("Plaintext-Matrix-Filename","matrix0")
        extension                 = request_headers.get("Extension","csv")
        k                         = int(request_headers.get("K"))
        experiment_iteration      = request_headers.get("Experiment-Iteration","0")
        experiment_id             = request_headers.get("Experiment-Id",uuid4().hex[:10])
        requestId                 = "request-{}".format(plaintext_matrix_id)
        plaintext_matrix_path     = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_filename, extension)

        cent_i_id = "{}centi".format(plaintext_matrix_id) #Build the id of Cent_i
        cent_j_id = "{}centj".format(plaintext_matrix_id) #Build the id of Cent_j

        _round             = bool(int(current_app.config.get("_round","0"))) #False
        decimals           = int(current_app.config.get("DECIMALS","2"))
        path               = current_app.config.get("KEYS_PATH","/rory/keys")
        ctx_filename       = current_app.config.get("CTX_FILENAME","ctx")
        pubkey_filename    = current_app.config.get("PUBKEY_FILENAME","pubkey")
        secretkey_filename = current_app.config.get("SECRET_KEY_FILENAME","secretkey")
        relinkey_filename  = current_app.config.get("RELINKEY_FILENAME","relinkey")
        # max_workers        = Utils.get_workers(num_chunks=num_chunks)

        MAX_ITERATIONS          = int(request_headers.get("Max-Iterations",current_app.config.get("MAX_ITERATIONS",10)))
        WORKER_TIMEOUT          = int(current_app.config.get("WORKER_TIMEOUT",300))
        MICTLANX_TIMEOUT        = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
        MICTLANX_DELAY          = int(current_app.config.get("MICTLANX_DELAY","2"))
        MICTLANX_BACKOFF_FACTOR = float(current_app.config.get("MICTLANX_BACKOFF_FACTOR","0.5"))
        MICTLANX_MAX_RETRIES    = int(current_app.config.get("MICTLANX_MAX_RETRIES","10"))
        
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
        dataowner = DataOwnerPQC(scheme = ckks) 
        
        local_read_dataset_start_time = time.time()
        plaintext_matrix_result  = await RoryCommon.read_numpy_from(
            path      = plaintext_matrix_path,
            extension = extension,
        )
        if plaintext_matrix_result.is_err:
            return Response(status=500, response="Failed to local read plain text matrix.")
        plaintext_matrix = plaintext_matrix_result.unwrap()
        
        plaintext_matrix = plaintext_matrix.astype(np.float32)

        r = plaintext_matrix.shape[0]
        a = plaintext_matrix.shape[1]

        local_read_entry = ExperimentLogEntry(
            event          = "LOCAL.READ",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = local_read_dataset_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = "",
            num_chunks     = num_chunks,
            k              = k,
            workers        = max_workers,
            security_level = security_level,
        )
        logger.info(local_read_entry.model_dump())

        max_workers = Utils.get_workers(num_chunks=num_chunks)
       
        encryption_start_time = time.time()
        n = a*r
        
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
            relinkey_filename  = relinkey_filename,
        )
        
        segment_encrypt_entry = ExperimentLogEntry(
            event          = "SEGMENT.ENCRYPT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = encryption_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = "",
            num_chunks     = num_chunks,
            k              = k,
            workers        = max_workers,
            security_level = security_level,
        )
        logger.info(segment_encrypt_entry.model_dump())
  
        put_chunks_start_time = time.time()
        put_encrypted_matrix_result = await RoryCommon.delete_and_put_chunks(
            client    = STORAGE_CLIENT,
            bucket_id = BUCKET_ID,
            key       = encrypted_matrix_id,
            chunks    = encrypted_matrix_chunks,
            timeout   = MICTLANX_TIMEOUT,
            max_tries = MICTLANX_MAX_RETRIES,
            tags      = {
                "full_shape": str((r,a)),
                "full_dtype":"float32"
            }
        )
        
        put_encrypted_ptm_entry = ExperimentLogEntry(
            event          = "PUT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = put_chunks_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = "",
            num_chunks     = num_chunks,
            k              = k,
            workers        = max_workers,
            security_level = security_level,
        )
        logger.info(put_encrypted_ptm_entry.model_dump())

        udm_start_time = time.time()
        udm            = dataowner.get_U(
            plaintext_matrix = plaintext_matrix,
            algorithm        = algorithm
        )
        
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
        )
        logger.info(udm_gen_entry.model_dump())

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
        
        udm_put_result = await RoryCommon.delete_and_put_chunks(
            client    = STORAGE_CLIENT,
            bucket_id = BUCKET_ID,
            key       = udm_id,
            chunks    = maybe_udm_matrix_chunks.unwrap(),
            timeout   = MICTLANX_TIMEOUT,
            max_tries = MICTLANX_MAX_RETRIES,
            tags      = {
                "full_shape": str(udm.shape),
                "full_dtype": str(udm.dtype)
            }
        )
        if udm_put_result.is_err:
            error = udm_put_result.unwrap_err()
            e = f"Failed to put the udm: {error}"
            return Response(status= 500, response=e)
        
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
        )
        logger.info(udm_put_entry.model_dump())
        
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
            secretkey_filename = secretkey_filename,
            relinkey_filename  = relinkey_filename
        )

        encrypt_entry = ExperimentLogEntry(
            event          = "ENCRYPT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = encrypt_ckks_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = "",
            num_chunks     = num_chunks,
            k              = k,
            workers        = max_workers,
            security_level = security_level,
        )
        logger.info(encrypt_entry.model_dump())

        put_chunks_start_time = time.time()
        put_encrypted_matrix_result = await RoryCommon.delete_and_put_chunks(
            client    = STORAGE_CLIENT,
            bucket_id = BUCKET_ID,
            key       = init_sm_id,
            chunks    = encrypted_zero_shiftmatrix_chunks,
            timeout   = MICTLANX_TIMEOUT,
            max_tries = MICTLANX_MAX_RETRIES,
            tags      = {
                "full_shape": str((k,a)),
                "full_dtype":"float32"
            }
        )
        if put_encrypted_matrix_result.is_err:
            e =f"Failed put chunks: {put_encrypted_matrix_result.unwrap_err()}" 
            logger.error(e)
            return Response(status=500, response=e)
        
        udm_put_entry = ExperimentLogEntry(
            event          = "PUT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = put_chunks_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = "",
            num_chunks     = num_chunks,
            k              = k,
            workers        = max_workers,
            security_level = security_level,
        )
        logger.info(udm_put_entry.model_dump())
        
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
        )
        logger.info(get_worker_entry.model_dump())
        
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
        
        while (status != Constants.ClusteringStatus.COMPLETED): #While the status is not completed
            
            inner_interaction_arrival_time = time.time()
            run1_headers  = {
                "Step-Index"             : "1",
                "Clustering-Status"      : str(status),
                "Plaintext-Matrix-Id"    : plaintext_matrix_id,
                "Request-Id"             : requestId,
                "Encrypted-Matrix-Id"    : encrypted_matrix_id,
                "Encrypted-Matrix-Shape" : "({},{})".format(r,a),
                "Encrypted-Matrix-Dtype" : "float32",
                "Encrypted-Udm-Dtype"    : "float32",
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
                event          = "RUN1",
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
                iterations     = run1_n_iterations
            )
            logger.info(run1_worker_entry.model_dump())

            encrypted_shift_matrix_start_time = time.time()
            encrypted_shift_matrix = await RoryCommon.get_pyctxt(
                client         = STORAGE_CLIENT,
                bucket_id      = BUCKET_ID,
                key            = encrypted_shift_matrix_id,
                ckks           = ckks,
                force          = True,
                backoff_factor = MICTLANX_BACKOFF_FACTOR,
                delay          = MICTLANX_DELAY,
                max_retries    = MICTLANX_MAX_RETRIES
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
                iterations     = run1_n_iterations
            )
            logger.info(get_encrypted_sm_entry.model_dump())

            decrypt_start_time = time.time()
            shift_matrix = ckks.decryptMatrix( #Shift Matrix is decrypted
                ciphertext_matrix = encrypted_shift_matrix,
                shape             = [k,a]
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
                iterations     = run1_n_iterations
            )
            logger.info(decrypt_entry.model_dump())

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
            
            put_shift_matrix_result = await RoryCommon.delete_and_put_chunks(
                client    = STORAGE_CLIENT,
                bucket_id = BUCKET_ID,
                key       = shift_matrix_id,
                chunks    = shift_matrix_chunks.unwrap(),
                timeout   = MICTLANX_TIMEOUT,
                max_tries = MICTLANX_MAX_RETRIES,
                tags      = {
                    "full_shape": str(shift_matrix.shape),
                    "full_dtype": str(shift_matrix.dtype)
                }
            )
            if put_shift_matrix_result.is_err:
                return Response ( status = 500, response = "Failed to put shiftmatrix")
            
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
                iterations     = run1_n_iterations
            )
            logger.info(put_sm_entry.model_dump())

            Cent_i= await RoryCommon.get_pyctxt(
                client    = STORAGE_CLIENT,
                bucket_id = BUCKET_ID, 
                key       = cent_i_id, 
                ckks      = ckks
            )
            Cent_j = await RoryCommon.get_pyctxt(
                client    = STORAGE_CLIENT,
                bucket_id = BUCKET_ID, 
                key       = cent_j_id,
                ckks      = ckks
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

            status       = Constants.ClusteringStatus.WORK_IN_PROGRESS #Status is updated
            run2_headers = {
                "Step-Index"             : "2",
                "Clustering-Status"      : str(status),
                "Shift-Matrix-Id"        : shift_matrix_id,
                "Plaintext-Matrix-Id"    : plaintext_matrix_id,
                "Encrypted-Matrix-Id"    : encrypted_matrix_id,
                "Encrypted-Matrix-Shape" : "({},{})".format(r,a),
                "Encrypted-Matrix-Dtype" : "float32",
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
                iterations     = run1_n_iterations
            )
            logger.info(skmeans_iteration_completed_entry.model_dump())

        interaction_end_time     = time.time()
        interaction_service_time = interaction_end_time - interaction_arrival_time 
        worker_response_time     = endTime - worker_start_time
        response_time            = endTime - arrivalTime 

        clustering_completed_entry = ExperimentLogEntry(
            event          = "COMPLETED",
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
            iterations     = iterations,
            client_time    = service_time_client,
            manager_time   = get_worker_service_time,
            worker_time    = worker_response_time,
        )
        logger.info(clustering_completed_entry.model_dump())
    
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
    """
    This method achieves a "Double-Blind" state by combining CKKS for data protection and FDHOPE for secure 
    distance matrix updates.

    Note:
    **Hybrid Secure Protocol**: Combines post-quantum security with double-blind logic.
     Mandatory parameters are required in the **HTTP Headers**.

    Attributes:
        Plaintext-Matrix-Id (str): Unique ID for CSS storage. Defaults to "matrix0".
        Plaintext-Matrix-Filename (str): Local dataset name. Defaults to "matrix0".
        K (int): Number of clusters. **Required**.
        Sens (float): Sensitivity for FDHOPE. Defaults to 0.00000001.
        Max-Iterations (int): Maximum protocol rounds. Defaults to 10.
        Experiment-Id (str): Tracking ID for performance auditing.

    Returns:
        label_vector (list): Final cluster assignments for the dataset.
        iterations (int): Actual number of iterations performed.
        algorithm (str): "dbskmeans pqc".
        worker_id (str): ID of the node that performed the secure computations.
        service_time_manager (float): Time spent in Worker allocation.
        service_time_worker (float): Cumulative time of remote computation.
        service_time_client (float): Total local time (Encryption/Decryption/IO).
        response_time_clustering (float): End-to-end execution time.

    Raises:
        Exception: Returns a 500 status code if failures occur in the hybrid 
            encryption chain (CKKS/FDHOPE), CSS I/O, or Worker orchestration.
    """
    try:
        arrivalTime                  = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        TESTING                      = current_app.config.get("TESTING",True)
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        STORAGE_CLIENT:AsyncClient   = current_app.config.get("ASYNC_STORAGE_CLIENT")
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        max_workers                  = current_app.config.get("MAX_WORKERS",2)
        num_chunks                   = current_app.config.get("NUM_CHUNKS",4)
        np_random                    = current_app.config.get("np_random")
        do_fdhope:DataOwner          = current_app.config.get("dataowner")
        if executor               == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm                 = Constants.ClusteringAlgorithms.DBSKMEANS_PQC
        algorithm_fdhope          = Constants.ClusteringAlgorithms.DBSKMEANS
        s                         = Session()
        security_level            = current_app.config.get("LIU_SECURITY_LEVEL",128)
        request_headers           = request.headers #Headers for the request
        # num_chunks                = int(request_headers.get("Num-Chunks",_num_chunks ) )
        plaintext_matrix_id       = request_headers.get("Plaintext-Matrix-Id","matrix0")
        encrypted_matrix_id       = "encrypted{}".format(plaintext_matrix_id) # The id of the encrypted matrix is built
        encrypted_udm_id          = "{}encryptedudm".format(plaintext_matrix_id) # The iudm id is built
        plaintext_matrix_filename = request_headers.get("Plaintext-Matrix-Filename","matrix0")
        extension                 = request_headers.get("Extension","csv")
        experiment_id             = request_headers.get("Experiment-Id",uuid4().hex[:10])
        k                         = int(request_headers.get("K"))
        sens                      = float(request_headers.get("Sens","0.00000001"))
        experiment_iteration      = request_headers.get("Experiment-Iteration","0")
        request_id                = "request{}".format(plaintext_matrix_id)
        plaintext_matrix_path     = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_filename, extension)
        
        init_sm_id = "{}initsm".format(plaintext_matrix_id)
        cent_i_id  = "{}centi".format(plaintext_matrix_id) #Build the id of Cent_i
        cent_j_id  = "{}centj".format(plaintext_matrix_id) #Build the id of Cent_j

        # max_workers        = Utils.get_workers(num_chunks=num_chunks)
        _round             = bool(int(current_app.config.get("_round","0"))) #False
        decimals           = int(current_app.config.get("DECIMALS","2"))
        path               = current_app.config.get("KEYS_PATH","/rory/keys")
        ctx_filename       = current_app.config.get("CTX_FILENAME","ctx")
        pubkey_filename    = current_app.config.get("PUBKEY_FILENAME","pubkey")
        secretkey_filename = current_app.config.get("SECRET_KEY_FILENAME","secretkey")
        relinkey_filename  = current_app.config.get("RELINKEY_FILENAME","relinkey")

        MAX_ITERATIONS          = int(request_headers.get("Max-Iterations",current_app.config.get("MAX_ITERATIONS",10)))
        WORKER_TIMEOUT          = int(current_app.config.get("WORKER_TIMEOUT",3600))
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
        dataowner = DataOwnerPQC(scheme = ckks, sens=sens)
        
        local_read_dataset_start_time = time.time()
        plaintext_matrix_result  = await RoryCommon.read_numpy_from(
            path      = plaintext_matrix_path,
            extension = extension,
        )
        if plaintext_matrix_result.is_ok:
            plaintext_matrix = plaintext_matrix_result.unwrap()
        else:
            raise plaintext_matrix_result.unwrap_err()
        
        plaintext_matrix = plaintext_matrix.astype(np.float32)

        r = plaintext_matrix.shape[0]
        a = plaintext_matrix.shape[1]
        max_workers = Utils.get_workers(num_chunks=num_chunks)

        local_read_entry = ExperimentLogEntry(
            event          = "LOCAL.READ",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = local_read_dataset_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = "",
            num_chunks     = num_chunks,
            k              = k,
            workers        = max_workers, 
            security_level = security_level,
        )
        logger.info(local_read_entry.model_dump())

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
            secretkey_filename = secretkey_filename,
            relinkey_filename  = relinkey_filename
        )
        
        segment_encrypt_entry = ExperimentLogEntry(
            event          = "SEGMENT.ENCRYPT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = encryption_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = "",
            num_chunks     = num_chunks,
            k              = k,
            workers        = max_workers, 
            security_level = security_level
        )
        logger.info(segment_encrypt_entry.model_dump())
  
        put_chunks_start_time = time.time()

        put_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
            client    = STORAGE_CLIENT,
            bucket_id = BUCKET_ID,
            key       = encrypted_matrix_id,
            chunks    = encrypted_matrix_chunks,
            timeout   = MICTLANX_TIMEOUT,
            max_tries = MICTLANX_MAX_RETRIES,
            tags      = {
                "full_shape": str((r,a)),
                "full_dtype":"float32"
            }
        )
        if put_chunks_generator_results.is_err:
            return Response(status=500, response="Failed to put encrypted matrix")
        
        put_encrypted_ptm_entry = ExperimentLogEntry(
            event          = "PUT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = put_chunks_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = "",
            num_chunks     = num_chunks,
            k              = k,
            workers        = max_workers,
            security_level = security_level
        )
        logger.info(put_encrypted_ptm_entry.model_dump())

        udm_start_time = time.time()
        udm            = do_fdhope.get_U(
            plaintext_matrix = plaintext_matrix,
            algorithm        = algorithm_fdhope
        )

        udm_shape = udm.shape
        udm_dtype = udm.dtype
        del plaintext_matrix
        
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
        )
        logger.info(udm_gen_entry.model_dump())
                
        n = r*r*a
        segment_encrypt_fdhope_start_time = time.time()
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

        put_encrypted_ptm_entry = ExperimentLogEntry(
            event          = "SEGMENT.ENCRYPT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = segment_encrypt_fdhope_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = "",
            num_chunks     = num_chunks,
            k              = k,
            workers        = max_workers,
            security_level = security_level
        )
        logger.info(put_encrypted_ptm_entry.model_dump())

        put_chunks_start_time = time.time()
        put_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
            client    = STORAGE_CLIENT,
            bucket_id = BUCKET_ID,
            key       = encrypted_udm_id,
            chunks    = encrypted_matrix_UDM_chunks,
            timeout   = MICTLANX_TIMEOUT,
            max_tries = MICTLANX_MAX_RETRIES,
            tags      = {
                "full_shape": str((r,r,a)), 
                "full_dtype":"float32"
            },
        )

        if put_chunks_generator_results.is_err:
            return Response(status=500, response="Failed to put encrypted udm matrix")
        
        put_encrypted_ptm_entry = ExperimentLogEntry(
            event          = "PUT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = put_chunks_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = "",
            num_chunks     = num_chunks,
            k              = k,
            workers        = max_workers,
            security_level = security_level,
        )
        logger.info(put_encrypted_ptm_entry.model_dump())

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
            secretkey_filename = secretkey_filename,
            relinkey_filename  = relinkey_filename
        )
        
        encrypt_sm_entry = ExperimentLogEntry(
            event          = "ENCRYPT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = init_shiftmatrix_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = "",
            num_chunks     = num_chunks,
            k              = k,
            workers        = max_workers,
            security_level = security_level
        )
        logger.info(encrypt_sm_entry.model_dump())

   
        put_chunks_start_time = time.time()

        put_chunks_generator_results = await RoryCommon.delete_and_put_chunks(
            client    = STORAGE_CLIENT,
            bucket_id = BUCKET_ID,
            key       = init_sm_id,
            chunks    = encrypted_shiftmatrix_chunks,
            timeout   = MICTLANX_TIMEOUT,
            max_tries = MICTLANX_MAX_RETRIES,
            tags = {
                "shape": str((k,a)),
                "dtype":"float32"
            }
        )
        if put_chunks_generator_results.is_err:
            return Response(status=500, response="Failed to put encrypted init shift matrix")
        
        put_encrypted_ptm_entry = ExperimentLogEntry(
            event          = "PUT",
            experiment_id  = experiment_id,
            algorithm      = algorithm,
            start_time     = put_chunks_start_time,
            end_time       = time.time(),
            id             = plaintext_matrix_id,
            worker_id      = "",
            num_chunks     = num_chunks,
            k              = k,
            workers        = max_workers,
            security_level = security_level,
        )
        logger.info(put_encrypted_ptm_entry.model_dump())

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
        )
        logger.info(get_worker_entry.model_dump())
        
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
                "Encrypted-Matrix-Dtype" : "float32",
                "Encrypted-Udm-Dtype"    : "float32",
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
            jsonWorkerResponse        = worker_run1_response.json()
            encrypted_shift_matrix_id = jsonWorkerResponse["encrypted_shift_matrix_id"]
            run1_service_time         = jsonWorkerResponse["service_time"]
            run1_n_iterations         = jsonWorkerResponse["n_iterations"]
            label_vector              = jsonWorkerResponse["label_vector"]

            run1_worker_entry = ExperimentLogEntry(
                event          = "RUN1",
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
                iterations     = run1_n_iterations
            )
            logger.info(run1_worker_entry.model_dump())

            encrypted_shift_matrix = await RoryCommon.get_pyctxt(
                client         = STORAGE_CLIENT,
                bucket_id      = BUCKET_ID, 
                key            = encrypted_shift_matrix_id,
                ckks           = ckks,
                delay          = MICTLANX_DELAY,
                max_retries    = MICTLANX_MAX_RETRIES,
                backoff_factor = MICTLANX_BACKOFF_FACTOR,
                force          = True
            )
            
            decrypt_start_time = time.time()
            shift_matrix = ckks.decryptMatrix( #Shift Matrix is decrypted
                ciphertext_matrix = encrypted_shift_matrix,
                shape = [k,a]
            )

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
            
            get_encrypted_sm_entry = ExperimentLogEntry(
                event          = "ENCRYPT",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = encrypted_start_time,
                end_time       = time.time(),
                id             = plaintext_matrix_id,
                worker_id      = worker_id,
                num_chunks     = num_chunks,
                k              = k,
                workers        = max_workers,
                security_level = security_level,
                iterations     = run1_n_iterations
            )
            logger.info(get_encrypted_sm_entry.model_dump())

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
                client    = STORAGE_CLIENT,
                bucket_id = BUCKET_ID,
                key       = shift_matrix_ope_id,
                chunks    = maybe_shift_matrix_chunks.unwrap(),
                timeout   = MICTLANX_TIMEOUT,
                max_tries = MICTLANX_MAX_RETRIES,
                tags      = {
                    "full_shape": str(shift_matrix_ope_shape),
                    "full_dtype": str(shift_matrix_ope_dtype)
                },
            )

            del maybe_shift_matrix_chunks
            del shift_matrix_ope
            if encrypted_sm_ope_result.is_err:
                return Response(status = 500, response="Failed to put encrypted shiftmatrix ope")
            
            put_sm_entry = ExperimentLogEntry(
                event          = "PUT",
                experiment_id  = experiment_id,
                algorithm      = algorithm,
                start_time     = put_matrix_start_time,
                end_time       = time.time(),
                id             = plaintext_matrix_id,
                worker_id      = worker_id,
                num_chunks     = num_chunks,
                k              = k,
                workers        = max_workers,
                security_level = security_level,
                iterations     = run1_n_iterations
            )
            logger.info(put_sm_entry.model_dump())

            Cent_i = await RoryCommon.get_pyctxt(
                client         = STORAGE_CLIENT,
                bucket_id      = BUCKET_ID,
                key            = cent_i_id,
                delay          = MICTLANX_DELAY,
                backoff_factor = MICTLANX_BACKOFF_FACTOR,
                force          = True,
                max_retries    = MICTLANX_MAX_RETRIES,
                ckks           = ckks,
            )
     
            Cent_j = await RoryCommon.get_pyctxt(
                client         = STORAGE_CLIENT,
                ckks           = ckks,
                bucket_id      = BUCKET_ID, 
                key            = cent_j_id,
                delay          = MICTLANX_DELAY,
                backoff_factor = MICTLANX_BACKOFF_FACTOR,
                force          = True,
                max_retries    = MICTLANX_MAX_RETRIES,
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
                "Encrypted-Matrix-Dtype" : "float32",
                "Encrypted-Udm-Dtype"    : "float32",
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

            dbskmeans_iteration_completed_entry = ExperimentLogEntry(
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
                iterations     = run1_n_iterations
            )
            logger.info(dbskmeans_iteration_completed_entry.model_dump())

        worker_response_time     = endTime - worker_start_time
        response_time            = endTime - arrivalTime 

        clustering_completed_entry = ExperimentLogEntry(
            event          = "COMPLETED",
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
            iterations     = iterations,
            client_time    = service_time_client,
            manager_time   = get_worker_service_time,
            worker_time    = worker_response_time,
        )
        logger.info(clustering_completed_entry.model_dump())
    
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