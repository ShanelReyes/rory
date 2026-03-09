import time, json
import numpy as np
import numpy.typing as npt
from typing import List,Tuple
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
from mictlanx import AsyncClient
from option import Result, Some
from mictlanx.utils.segmentation import Chunks
from option import Option,Some,NONE
from rorycommon import Common as RoryCommon
from Pyfhel import PyCtxt,Pyfhel
from models import ExperimentLogEntry
clustering = Blueprint("clustering",__name__,url_prefix = "/clustering")

@clustering.route("/test",methods=["GET","POST"])
def test():
    """Health check and component identification endpoint for the Worker node.
    This method serves as a heartbeat signal for the Rory Manager, allowing the orchestrator to confirm 
    the node's availability and its specific role within the PPDMaaS ecosystem. It returns the component 
    type both in the JSON payload and the HTTP response headers to facilitate automated discovery and 
    load balancing.

    Note:
        **Infrastructure Check**: This endpoint does not require cryptographic parameters or session identifiers, making it the primary tool for connectivity troubleshooting.

    Returns:
        component_type (str): The identification string "worker".
        
        Headers:
            Component-Type (str): Metadata indicating the node's functional role.
    """
    return Response(
        response = json.dumps({
            "component_type":"worker"
        }),
        status   = 200,
        headers  = {
            "Component-Type":"worker"
        }
    )

async def skmeans_1(requestHeaders) -> Response:
    """
    First interactive phase of the Secure K-Means protocol (Liu scheme) within the Worker node. 
    This method orchestrates the retrieval of encrypted datasets and UDM matrices from the CSS, 
    performs the initial privacy-preserving distance calculations (Run 1), and persists the resulting 
    intermediate shift matrices and centroids back to storage. 
    The process intentionally pauses at this stage, awaiting Client interaction to resolve secure updates 
    before proceeding to subsequent rounds of the iterative clustering lifecycle.

    Note:
        **State Management**: This step handles both the initial 'START' status and subsequent 'WORK_IN_PROGRESS' states, requiring specific cryptographic metadata via **HTTP Headers**.

    Attributes:
        Plaintext-Matrix-Id (str): Root identifier used to derive storage keys for centroids and shift matrices.
        Encrypted-Matrix-Id (str): Storage key for the primary encrypted dataset in the CSS.
        Encrypted-Matrix-Shape (str): Tuple string representing the dimensions of the encrypted matrix.
        Encrypted-Matrix-Dtype (str): Data type of the encrypted matrix elements.
        K (int): The number of clusters to form. Defaults to "3".
        M (int): Encryption scheme multiplier parameter. Defaults to "3".
        Num-Chunks (int): Number of storage fragments the matrices are divided into.
        Clustering-Status (int): Current state of the algorithm (Start=0, Progress=1).
        Experiment-Id (str): Unique identifier for execution tracing and auditing.
        Iterations (int): Current count of completed protocol cycles.

    Returns:
        label_vector (list): Intermediate cluster assignments for the dataset points.
        service_time (float): Total execution time for the Step 1 operations.
        n_iterations (int): Incremented count of total protocol iterations.
        encrypted_shift_matrix_id (str): Identifier for the generated matrix S stored in the CSS.

    Raises:
        Exception: Occurs if mandatory headers (Shape/Dtype) are missing, CSS retrieval fails, or errors arise during the persistence of intermediate chunks.
    """
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


async def skmeans_2(requestHeaders):
    """Second interactive phase of the Secure K-Means protocol (Liu scheme). 
    This method processes the decrypted shift matrix returned by the Client to determine protocol convergence. 
    If the error is within the permissible threshold, the clustering is marked as completed; otherwise, 
    it triggers 'Run 2' to update the encrypted UDM matrices and persists them to the CSS, 
    preparing the system for the next iterative cycle of the privacy-preserving mining process.

    Note:
        **Iterative Control**: This endpoint manages the transition between 'WORK_IN_PROGRESS' and 'COMPLETED' 
        states based on the convergence criteria evaluated from **HTTP Headers**.

    Attributes:
        Plaintext-Matrix-Id (str): Root identifier for deriving storage keys (UDM, centroids).
        Encrypted-Matrix-Id (str): Identifier for the encrypted dataset in the CSS.
        Shift-Matrix-Id (str): Storage key for the decrypted shift matrix provided by the client.
        Encrypted-Matrix-Shape (str): Tuple string representing the dimensions of the encrypted data.
        K (int): The number of clusters to form. Defaults to "3".
        M (int): Encryption scheme multiplier parameter. Defaults to "3".
        Num-Chunks (int): Number of storage fragments for matrix persistence.
        Iterations (int): The current count of protocol cycles completed.
        Experiment-Id (str): Unique identifier for execution tracing and auditing.

    Returns:
        Clustering-Status (int): The updated state (1 for In Progress, 2 for Completed).
        Service-Time (float): Execution time for this specific step.
        Total-Service-Time (float): Cumulative time if the algorithm has reached completion.

    Raises:
        Exception: If mandatory identifiers are missing, storage retrieval fails, or errors occur during the persistence of updated UDM chunks.
        """
    
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
    """ 
    This method evaluates the protocol's state via a step index to coordinate the privacy-preserving mining 
    process, ensuring that intermediate computations are handled by the appropriate sub-routine 
    (Step 1 or Step 2) to maintain data confidentiality throughout the clustering lifecycle.

    Attributes:
        Step-Index (int): The current round of the interactive protocol. Use "1" for distance calculation and "2" for centroid and label updates. Defaults to "1".
        Experiment-Id (str): A unique identifier for the execution trace and performance auditing.
        Plaintext-Matrix-Id (str): Identifier for the encrypted matrix stored in the CSS.

    Returns:
        An object forwarded from the corresponding sub-step (skmeans_1 or skmeans_2), 
        typically containing encrypted intermediate results or final labels.

    Raises:
        Exception: If the Step-Index is outside the valid range (1-2) or if the sub-routines encounter communication failures with the storage system.
    """
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
    """
    This method retrieves a plaintext matrix from the Cloud Storage System (CSS) using asynchronous clients, 
    performs the clustering logic locally, and logs detailed performance metrics—including retrieval and 
    execution times—to support experimental auditing and service time analysis within the distributed platform.

    Note:
        **Execution Parameters**: All attributes listed below must be passed exclusively via **HTTP Headers**.

    Attributes:
        Plaintext-Matrix-Id (str): Unique identifier for the matrix stored in the CSS/Bucket.
        K (int): The number of clusters to form. Defaults to "3".
        Experiment-Id (str): A unique identifier for the execution trace and logging.

    Returns:
        label_vector (list): The final cluster assignment for each data point.
        iterations (int): Total number of iterations performed until convergence.
        service_time (float): Total time elapsed during the worker's execution flow.

    Raises:
        Exception: Captures and logs failures during CSS matrix retrieval, MictlanX communication timeouts, or clustering computation errors.
    """
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


async def dbskmeans_1(requestHeaders) -> Response:
    """
    First interactive phase of the Double-Blind Secure K-Means (DBS K-Means) protocol within the Worker node. 
    This method orchestrates the retrieval and merging of encrypted data matrices and encrypted UDM matrices 
    from the CSS. It performs the initial privacy-preserving distance calculations ('Run 1') using the 
    double-blind scheme, generates the intermediate encrypted shift matrix, and persists the updated
    centroids and state back to storage. The execution pauses after this step, awaiting the Client to perform 
    the necessary partial decryption of the shift matrix.

    Note:
        **Double-Blind Execution**: This protocol requires both the data and the UDM to be in an encrypted state. All cryptographic metadata and identifiers must be provided via **HTTP Headers**.

    Attributes:
        Plaintext-Matrix-Id (str): Root identifier for deriving storage keys for centroids and UDM.
        Encrypted-Matrix-Id (str): Storage key for the primary encrypted dataset.
        Encrypted-Matrix-Shape (str): Tuple string representing the dimensions of the dataset.
        Encrypted-Matrix-Dtype (str): Data type of the encrypted dataset elements.
        Encrypted-Udm-Shape (str): Tuple string representing the dimensions of the encrypted UDM.
        Encrypted-Udm-Dtype (str): Data type of the encrypted UDM elements.
        K (int): The number of clusters to form. Defaults to "3".
        M (int): Encryption scheme multiplier parameter. Defaults to "3".
        Num-Chunks (int): Number of storage fragments for matrix persistence.
        Clustering-Status (int): Current state of the algorithm (Start=0, Progress=1).
        Iterations (int): Current count of completed protocol cycles.
        Experiment-Id (str): Unique identifier for execution tracing and auditing.

    Returns:
        label_vector (list): Intermediate cluster assignments.
        encrypted_shift_matrix_id (str): Identifier for the generated matrix S1 stored in the CSS.
        n_iterations (int): The current iteration count.
        service_time (float): Total time elapsed during this worker phase.

    Raises:
        Exception: If mandatory headers (Shape/Dtype) are missing, CSS retrieval fails, or errors arise during the persistence of intermediate encrypted chunks.
        """
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


async def dbskmeans_2(requestHeaders):
    """Second interactive phase of the Double-Blind Secure K-Means (DBS K-Means) protocol in the Worker node. 
    This method processes the decrypted shift matrix provided by the Client to evaluate the algorithm's 
    convergence based on a mean error threshold. If the error is within limits, the process terminates and 
    returns cumulative performance metrics. Otherwise, it executes the 'Run 2' logic to update the encrypted 
    UDM matrices, segments them into chunks, and persists them back to the CSS to enable the next iteration 
    of the privacy-preserving mining cycle.

    Note:
        **Iterative Control**: The protocol's state transition (Progress vs. Completion) and all cryptographic metadata are managed exclusively via **HTTP Headers**.

    Attributes:
        Plaintext-Matrix-Id (str): Root identifier for locating centroids and encrypted UDM fragments.
        Encrypted-Matrix-Id (str): Storage key for the primary encrypted dataset.
        Shift-Matrix-Ope-Id (str): Identifier for the decrypted shift matrix returned by the client.
        Encrypted-Matrix-Shape (str): Tuple string representing the dimensions of the encrypted data.
        Encrypted-Udm-Shape (str): Tuple string representing the dimensions of the encrypted UDM.
        K (int): The number of clusters to form. Defaults to "3".
        M (int): Encryption scheme multiplier parameter. Defaults to "3".
        Num-Chunks (int): Number of storage fragments for matrix persistence. Defaults to "4".
        Start-Time (str): Global start timestamp used to calculate total response time upon completion.
        Iterations (int): The current count of completed protocol cycles.
        Experiment-Id (str): Unique identifier for execution tracing and auditing.

    Returns:
        response_time (str): Cumulative time since the global start (only on completion).
        service_time (float): Execution time for this specific worker phase.
        encrypted_udm_shape (str): The shape of the updated UDM matrix.
        encrypted_udm_dtype (str): The data type of the updated UDM matrix.

    Raises:
        Exception: If mandatory identifiers are missing, CSS retrieval fails, or errors occur during the segmentation and persistence of updated encrypted UDM chunks.
        """
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


@clustering.route("/dbskmeans", methods = ["POST"])
async def dbskmeans():
    """Main routing endpoint for the Double-Blind Secure K-Means (DBS K-Means) interactive protocol 
    within the Worker node. 
    This method manages the multi-party privacy-preserving lifecycle by evaluating the protocol's state 
    through a step index. It ensures that encrypted computations are correctly delegated to either the 
    initial distance calculation phase (Step 1) or the final convergence and centroid update phase (Step 2), 
    maintaining the double-blind security guarantees throughout the distributed execution.

    Note:
        **Interactive Protocol Control**: The execution flow and state transition are managed exclusively 
        via the 'Step-Index' attribute passed in the **HTTP Headers**.

    Attributes:
        Step-Index (int): The current round of the interactive protocol. Use "1" for the initial distance and shift matrix calculation, and "2" for convergence verification and label assignment. Defaults to "1".
        Experiment-Id (str): A unique identifier used for performance auditing and execution tracing within the Rory platform.
        Plaintext-Matrix-Id (str): The root identifier used to locate the encrypted dataset and its associated cryptographic metadata in the CSS.

    Returns:
        An object forwarded from the corresponding sub-routine (dbskmeans_1 or dbskmeans_2), 
        containing either intermediate ciphertexts or final clustering results.

    Raises:
        Exception: If the Step-Index is invalid or if the internal sub-routines encounter failures during CSS retrieval or cryptographic processing.
    """
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
    """
    Asynchronous endpoint for the Worker node to execute the Double-Blind Secure Nearest Neighbor Clustering 
    (DBSNNC) algorithm.
    This method performs a privacy-preserving clustering operation by retrieving both the encrypted data 
    matrix and the encrypted distance matrix from the CSS. It executes the DBSNNC logic directly on the 
    ciphertext using a secure threshold comparison, effectively grouping data points based on their proximity 
    without exposing the underlying plaintext or the actual distances to the cloud environment.

    Note:
        **Single-Round Execution**: Unlike SK-Means, this protocol is non-interactive at the worker level. All cryptographic identifiers and threshold values must be provided via **HTTP Headers**.

    Attributes:
        Plaintext-Matrix-Id (str): Root identifier for deriving keys and session logging.
        Encrypted-Matrix-Id (str): Storage key for the primary encrypted dataset.
        Encrypted-Dm-Id (str): Storage key for the pre-calculated encrypted distance matrix.
        Encrypted-Threshold (float): The secure distance limit used to determine neighborhood connectivity.
        Encrypted-Matrix-Shape (str): Tuple string representing the dimensions of the data matrix.
        Encrypted-Dm-Shape (str): Tuple string representing the dimensions of the distance matrix.
        M (int): Encryption scheme multiplier parameter. Defaults to "3".
        Num-Chunks (int): Number of storage fragments for the input matrices.
        Experiment-Id (str): Unique identifier for performance auditing and tracing.

    Returns:
        label_vector (list): The final cluster assignment for each data point.
        service_time (float): Total time elapsed during the worker's execution flow.

    Raises:
        Exception: If mandatory headers (Shape/Dtype) are missing, CSS retrieval fails, or errors arise during the secure clustering computation.
        """
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
    """
    Asynchronous endpoint for the Worker node to execute the standard Nearest Neighbor Clustering (NNC) 
    algorithm on plaintext data.
    This method retrieves the plaintext data matrix and its corresponding distance matrix from the CSS 
    using asynchronous operations. It then executes the NNC logic to group data points based on a provided 
    distance threshold, returning the resulting labels and logging performance metrics for auditing and 
    benchmarking against secure variants.

    Note:
        **Plaintext Execution**: All required parameters and matrix identifiers must be provided exclusively via **HTTP Headers**. The request body is not utilized.

    Attributes:
        Plaintext-Matrix-Id (str): Root identifier used to locate the data matrix and the pre-calculated distance matrix.
        Threshold (float): The distance limit used to establish connectivity between neighboring data points.
        Plaintext-Matrix-Shape (str): Tuple string representing the dimensions of the data matrix.
        Plaintext-Matrix-Dtype (str): Data type of the data matrix elements.
        Dm-Shape (str): Tuple string representing the dimensions of the distance matrix.
        Dm-Dtype (str): Data type of the distance matrix elements.
        Num-Chunks (int): Number of storage fragments for the input matrices.
        Experiment-Id (str): Unique identifier for performance auditing and execution tracing.

    Returns:
        label_vector (list): The final cluster assignment for each data point.
        service_time (float): Total execution time for the worker's processing flow.

    Raises:
        Exception: If mandatory headers (Shape/Dtype) are missing, CSS retrieval fails, or errors occur during the clustering computation.
        """
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
    """
    First interactive phase of the Post-Quantum Secure K-Means (PQC SK-Means) protocol within the Worker node.
    This method implements the CKKS homomorphic encryption scheme to process encrypted datasets, retrieving 
    ciphertexts and UDM matrices from the CSS. It performs the initial privacy-preserving 
    distance calculations (Run 1) in a post-quantum secure domain, persisting intermediate shift matrices 
    and centroids as encrypted fragments. The process halts after this phase, requiring the Client to perform 
    secure refreshes or partial decryptions of the CKKS ciphertexts before the next iteration.

    Note:
        **Post-Quantum Execution**: All execution parameters, including CKKS metadata and matrix identifiers, must be provided exclusively via **HTTP Headers**.

    Attributes:
        Plaintext-Matrix-Id (str): Root identifier used to derive storage keys for CKKS-encrypted centroids and shift matrices.
        Encrypted-Matrix-Id (str): Storage key for the primary CKKS-encrypted dataset in the CSS.
        Encrypted-Matrix-Shape (str): Tuple string representing the dimensions of the encrypted matrix.
        Encrypted-Matrix-Dtype (str): Data type of the encrypted matrix elements.
        K (int): The number of clusters to form. Defaults to "3".
        Num-Chunks (int): Number of storage fragments the ciphertexts are divided into.
        Clustering-Status (int): Current state of the algorithm (Start=0, Progress=1).
        Experiment-Id (str): Unique identifier for execution tracing and auditing.
        Iterations (int): Current count of completed PQC protocol cycles.

    Returns:
        label_vector (list): Intermediate cluster assignments derived from the PQC computation.
        service_time (float): Total execution time for the Step 1 PQC operations.
        n_iterations (int): Incremented count of total PQC protocol iterations.
        encrypted_shift_matrix_id (str): Identifier for the generated CKKS shift matrix stored in the CSS.

    Raises:
        Exception: Occurs if CKKS key initialization fails, mandatory headers are missing, 
        or errors arise during the retrieval or persistence of ciphertexts.
        """
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
        print("MAUBE_CENT_I", maybe_cent_i_chunks)
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
        print("X",x)
        logger.debug(str(x))

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
        print(maybe_cent_j_chunks)
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
        print(maybe_encrypted_shift_matrix_chunks)
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
    """
    Second interactive phase of the Post-Quantum Secure K-Means (PQC SK-Means) protocol using the CKKS scheme.
    This method evaluates the convergence signal ('Is-Zero') provided by the Client. 
    If the algorithm has converged, it finalizes the process and logs cumulative performance metrics. 
    If convergence is not reached, it retrieves the refreshed/decrypted shift matrix from the Client and the 
    initial parameters to execute 'Run 2', updating the encrypted UDM matrices and persisting 
    them back to the CSS to enable the next iteration of the post-quantum mining lifecycle.

    Note:
        **Quantum-Resistant Iteration**: The protocol state and convergence flow are managed exclusively via **HTTP Headers**, ensuring that the Worker operates in a zero-knowledge state regarding the underlying plaintext.

    Attributes:
        Plaintext-Matrix-Id (str): Root identifier used to derive storage keys for CKKS-encrypted UDM and centroids.
        Encrypted-Matrix-Id (str): Storage key for the primary CKKS-encrypted dataset in the CSS.
        Shift-Matrix-Id (str): Identifier for the decrypted or refreshed shift matrix provided by the Client.
        Encrypted-Matrix-Shape (str): Tuple string representing the dimensions of the dataset.
        Is-Zero (int): Convergence flag provided by the Client (1 if converged, 0 otherwise).
        K (int): The number of clusters to form. Defaults to "3".
        Num-Chunks (int): Number of storage fragments for matrix persistence.
        Iterations (int): The current count of completed PQC protocol cycles.
        Experiment-Id (str): Unique identifier for execution tracing and auditing within Rory.

    Returns:
        Clustering-Status (int): The updated state (1 for In Progress, 2 for Completed).
        Service-Time (float): Execution time for this specific PQC worker phase.
        Total-Service-Time (float): Cumulative time if the algorithm has reached completion.

    Raises:
        Exception: If mandatory identifiers are missing, CKKS key initialization fails, 
        or errors occur during the retrieval or persistence of ciphertext chunks.
        """
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
    """
    Main routing endpoint for the Post-Quantum Secure K-Means (PQC SK-Means) interactive protocol 
    within the Worker node. 
    This method manages the lifecycle of clustering operations resilient to quantum computing threats 
    by utilizing the CKKS homomorphic encryption scheme. It acts as an orchestrator that evaluates 
    the protocol's state via a step index, delegating tasks to either the initial distance calculation 
    phase (Step 1) or the final centroid update and convergence phase (Step 2), ensuring a secure 
    multi-round communication flow with the Client.

    Note:
        **Post-Quantum Protocol Control**: The execution flow and state transitions are managed 
        exclusively via the 'Step-Index' attribute passed in the **HTTP Headers**. The request 
        body is not utilized.

    Attributes:
        Step-Index (int): The current round of the interactive PQC protocol. Use "1" for initial encrypted distance calculations and "2" for processing decrypted updates from the Client. Defaults to "1".
        Experiment-Id (str): A unique identifier for performance auditing and execution tracing within the Rory platform.
        Plaintext-Matrix-Id (str): The root identifier used to locate the encrypted dataset and associated CKKS cryptographic metadata in the CSS.

    Returns:
        An object forwarded from the corresponding sub-routine (pqc_skmeans_1 or pqc_skmeans_2), 
        containing intermediate PQC ciphertexts or final clustering results.

    Raises:
        Exception: If the Step-Index is invalid or if the internal sub-routines encounter failures during CSS retrieval or CKKS-based processing.
    """
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
    """
    First interactive phase of the Post-Quantum Double-Blind Secure K-Means (PQC DBS K-Means) protocol 
    within the Worker node.
    This method implements a hybrid privacy-preserving approach by combining  CKKS scheme 
    with double-blind logic. It retrieves PQC-encrypted data matrices and encrypted UDM fragments from the CSS, 
    performing the initial distance calculations (Run 1) directly on the ciphertexts. The resulting encrypted 
    shift matrices and intermediate centroids are persisted back to storage, pausing the execution to allow 
    the Client to perform secure refreshes or partial decryptions of the ciphertexts before 
    the next iteration.

    Note:
        **Hybrid Secure Execution**: This protocol requires both CKKS parameters and double-blind metadata. 
        All identifiers and cryptographic configurations must be provided exclusively via **HTTP Headers**.

    Attributes:
        Plaintext-Matrix-Id (str): Root identifier used to derive storage keys for CKKS-encrypted centroids and shift matrices.
        Encrypted-Matrix-Id (str): Storage key for the primary CKKS-encrypted dataset in the CSS.
        Encrypted-Matrix-Shape (str): Tuple string representing the dimensions of the encrypted matrix.
        Encrypted-Udm-Shape (str): Tuple string representing the dimensions of the encrypted UDM.
        K (int): The number of clusters to form. Defaults to "3".
        Num-Chunks (int): Number of storage fragments the ciphertexts are divided into.
        Clustering-Status (int): Current state of the algorithm (Start=0, Progress=1).
        Experiment-Id (str): Unique identifier for execution tracing and auditing.
        Iterations (int): Current count of completed PQC double-blind protocol cycles.

    Returns:
        label_vector (list): Intermediate cluster assignments derived from the PQC double-blind computation.
        service_time (float): Total execution time for the Step 1 hybrid operations.
        n_iterations (int): Incremented count of total protocol iterations.
        encrypted_shift_matrix_id (str): Identifier for the generated CKKS shift matrix stored in the CSS.

    Raises:
        Exception: Occurs if CKKS initialization fails, mandatory headers are missing, 
        or errors arise during the retrieval or persistence of ciphertext chunks.
        """
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
            __Cent_j = await RoryCommon.get_pyctxt(
            # __Cent_j = RoryCommon.get_pyctxt_with_retry(
                client = STORAGE_CLIENT, 
                bucket_id      = BUCKET_ID, 
                key            = cent_i_id, 
                # num_chunks     = num_chunks,
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
    """
    Second interactive phase of the Post-Quantum Double-Blind Secure K-Means (PQC DBS K-Means) protocol.
    This method evaluates the convergence signal ('Is-Zero') provided by the Client after the partial decryption 
    of the shift matrix. If convergence is reached, it marks the experiment as completed and logs final metrics. 
    If not, it retrieves the refreshed shift matrix and the previous encrypted UDM from the CSS to execute 
    'Run 2' within the CKKS domain. The resulting updated UDM matrix is then segmented into chunks and persisted 
    back to the storage system to enable the next iterative cycle of the hybrid secure mining process.

    Note:
        **Hybrid Secure Convergence**: This step manages the transition between protocol states using 
        cryptographic parameters. All control signals and identifiers must be provided via **HTTP Headers**.

    Attributes:
        Plaintext-Matrix-Id (str): Root identifier used to locate CKKS-encrypted UDM, centroids, and initial shift matrices.
        Encrypted-Matrix-Id (str): Storage key for the primary CKKS-encrypted dataset in the CSS.
        Shift-Matrix-Ope-Id (str): Identifier for the decrypted/refreshed shift matrix returned by the Client.
        Encrypted-Matrix-Shape (str): Tuple string representing the dimensions of the encrypted dataset.
        Encrypted-Udm-Shape (str): Tuple string representing the dimensions of the encrypted UDM.
        Is-Zero (int): Convergence flag provided by the Client (1 if converged, 0 otherwise).
        K (int): The number of clusters to form. Defaults to "3".
        Num-Chunks (int): Number of storage fragments for matrix persistence. Defaults to "-1".
        Iterations (int): The current count of completed protocol cycles.
        Experiment-Id (str): Unique identifier for execution tracing and auditing.

    Returns:
        Clustering-Status (int): Updated state (1 for In Progress, 2 for Completed).
        Service-Time (float): Execution time for this specific hybrid worker phase.
        Total-Service-Time (float): Cumulative time if the algorithm has reached completion.

    Raises:
        Exception: If mandatory identifiers are missing, CKKS key initialization fails, 
        or errors arise during the retrieval or persistence of updated ciphertext chunks.
        """
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
    """
    Main routing endpoint for the Post-Quantum Double-Blind Secure K-Means (PQC DBS K-Means) interactive 
    protocol within the Worker node. 
    This method orchestrates a multi-party privacy-preserving clustering flow that is resilient to quantum 
    computing attacks by utilizing CKKS scheme. It manages the protocol state via a 
    step index to coordinate the complex interaction between encrypted data matrices and encrypted UDM 
    matrices, ensuring that the cloud environment operates in a "zero-knowledge" state while delegating 
    specific cryptographic refreshes to the Client.

    Note:
        **Hybrid Secure Protocol**: This execution combines post-quantum security with double-blind logic. 
        All control parameters and cryptographic identifiers must be passed exclusively via **HTTP Headers**.

    Attributes:
        Step-Index (int): The current round of the interactive PQC double-blind protocol. Use "1" for the initial distance and shift matrix calculation, and "2" for convergence verification and UDM updates. Defaults to "1".
        Experiment-Id (str): A unique identifier used for performance auditing and execution tracing within the Rory platform.
        Plaintext-Matrix-Id (str): The root identifier used to locate the CKKS-encrypted dataset and associated double-blind metadata in the CSS.

    Returns:
        An object forwarded from the corresponding sub-routine (pqc_dbskmeans_1 or pqc_dbskmeans_2), 
        containing either intermediate PQC-encrypted shift matrices or final clustering results.

    Raises:
        Exception: If the Step-Index is invalid or if the internal sub-routines encounter failures during ciphertext retrieval or processing.
        """
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