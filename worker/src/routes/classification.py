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
    """Health check and role verification endpoint for the Worker's Classification module.
    This method acts as a diagnostic heartbeat, allowing the Rory Manager to verify that the Classification blueprint is correctly registered and the node is ready to process KNN or PQC-based inference tasks. It returns the component type in both the JSON payload and the HTTP response headers for automated service discovery and configuration auditing.

    Note:
        **Service Availability**: This endpoint provides a zero-overhead way to test the connectivity between the Manager and the Worker node without requiring cryptographic keys or session identifiers.

    Returns:
        Response: A Flask Response object with a 200 status containing a JSON payload with:
            component_type (str): The identification string "worker".
        
        Headers:
            Component-Type (str): Functional metadata indicating the node's classification role.
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

async def sknn_pedict_1(requestHeaders):
    """
    First interactive phase of the Secure K-Nearest Neighbors (SKNN) prediction protocol within the Worker node.
    This method orchestrates the retrieval of encrypted models and test records from the CSS. It performs the 
    privacy-preserving distance calculation between the target records and the model's training set using the 
    specified metric (e.g., Manhattan or Euclidean). The resulting secure distance matrix is segmented and 
    persisted back to storage, awaiting the Client's interaction to resolve the k-nearest labels in the 
    subsequent protocol round.

    Note:
        **Secure Inference Phase 1**: This step operates exclusively on ciphertext to maintain data confidentiality. All identifiers, shapes, and cryptographic metadata must be provided via **HTTP Headers**.

    Attributes:
        Model-Id (str): Root identifier used to locate the encrypted training model.
        Records-Test-Id (str): Storage key for the encrypted records to be classified.
        Encrypted-Model-Shape (str): Tuple string representing the dimensions of the encrypted model.
        Encrypted-Model-Dtype (str): Data type of the encrypted model elements.
        Encrypted-Records-Shape (str): Tuple string representing the dimensions of the encrypted test records.
        Encrypted-Records-Dtype (str): Data type of the encrypted record elements.
        Num-Chunks (int): Number of storage fragments for matrix retrieval and persistence.
        Experiment-Id (str): Unique identifier for execution tracing and auditing within Rory.

    Returns:
        distances_id (str): Storage key for the calculated secure distance matrix.
        distances_shape (str): Dimensions of the resulting distance matrix.
        distances_dtype (str): Data type of the distance elements.
        service_time (float): Total time elapsed during this worker phase.

    Raises:
        Exception: If mandatory headers are missing, CSS retrieval fails, or errors occur during the secure distance computation or chunk persistence.
    """
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
    MICTLANX_DELAY          = int(current_app.config.get("MICTLANX_DELAY","2"))
    MICTLANX_BACKOFF_FACTOR = float(current_app.config.get("MICTLANX_BACKOFF_FACTOR","0.5"))
    MICTLANX_MAX_RETRIES    = int(current_app.config.get("MICTLANX_MAX_RETRIES","10"))  
    
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
                "full_dtype":"float32"
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
    """
    Final interactive phase of the Secure K-Nearest Neighbors (SKNN) prediction protocol within the Worker node.
    This method completes the secure classification process by retrieving the resolved nearest-neighbor 
    indices from the CSS (previously computed and uploaded by the Client). It then maps these indices to 
    their corresponding model labels to generate the final classification vector. The process concludes 
    by logging cumulative performance metrics and the total service time for the interactive inference session.

    Note:
        **Secure Inference Phase 2**: This step acts as the final label resolution. All identifiers and shape metadata for the labels and indices must be provided via **HTTP Headers**.

    Attributes:
        Model-Id (str): Root identifier used to locate the associated model labels.
        Model-Labels-Shape (str): Tuple string representing the dimensions of the model labels (Mandatory).
        Records-Test-Id (str): Storage key used to derive the identifier for the resolved distance indices.
        Experiment-Id (str): Unique identifier for execution tracing and auditing within the Rory platform.

    Returns:
        label_vector (list): The final predicted class labels for the test records.
        service_time (float): Total execution time for this specific worker phase.

    Raises:
        Exception: If the 'Model-Labels-Shape' header is missing, CSS retrieval of labels or indices fails, or errors occur during the label mapping logic.
    """
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
    MICTLANX_DELAY          = int(current_app.config.get("MICTLANX_DELAY","2"))
    MICTLANX_BACKOFF_FACTOR = float(current_app.config.get("MICTLANX_BACKOFF_FACTOR","0.5"))
    MICTLANX_MAX_RETRIES    = int(current_app.config.get("MICTLANX_MAX_RETRIES","10"))

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
    """
    Main routing endpoint for the Secure K-Nearest Neighbors (SKNN) interactive prediction protocol within 
    the Worker node. 
    This method manages the multi-round secure inference process by evaluating the protocol's state via a 
    step index. It coordinates the transition between the encrypted distance calculation and the final 
    label assignment, ensuring that the Worker performs heavy computations on ciphertext (using the FDHOPE scheme) 
    while delegating specific decryption tasks to the Client to maintain end-to-end privacy.

    Note:
        **Interactive Inference Control**: The execution flow and state transitions are managed exclusively via the 'Step-Index' attribute passed in the **HTTP Headers**. The request body is not utilized.

    Attributes:
        Step-Index (int): The current round of the interactive SKNN protocol. Use "1" for the initial secure distance calculation and "2" for processing client-aided label resolution. Defaults to "1".
        Model-Id (str): Identifier for the encrypted model stored in the CSS to be used for prediction.
        Experiment-Id (str): A unique identifier for execution tracing and performance auditing within the Rory platform.

    Returns:
        An object forwarded from the corresponding sub-routine (sknn_predict_1 or sknn_predict_2), 
        containing either intermediate secure scores or the final classification results.

    Raises:
        Exception: If the Step-Index is invalid or if the sub-routines encounter failures during CSS retrieval or cryptographic processing.
        """
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
    """
    Asynchronous endpoint for the Worker node to execute K-Nearest Neighbors (KNN) classification on 
    plaintext data.
    This method retrieves a pre-trained model, its associated labels, and the target test records from the CSS. 
    It performs the classification logic using the distance metric configured in the global system settings 
    (e.g., Euclidean or Manhattan) and returns the resulting label vector. Detailed performance metrics for 
    each retrieval and computation step are logged to support experimental analysis within the PPDMaaS platform.

    Note:
        **Standard Classification**: All required identifiers for models and datasets must be passed exclusively via **HTTP Headers**. The request body is not utilized.

    Attributes:
        Model-Id (str): Unique identifier for the trained KNN model stored in the CSS. Defaults to "model0".
        Model-Labels-Shape (str): Tuple string representing the dimensions of the model labels (Mandatory).
        Records-Test-Id (str): Storage key for the dataset records to be classified. Defaults to "matrix0".
        Experiment-Id (str): Unique identifier for execution tracing and auditing.

    Returns:
        label_vector (list): The predicted class assignments for the test records.
        service_time (float): Total time elapsed during the worker's processing flow.

    Raises:
        Exception: If the 'Model-Labels-Shape' header is missing, CSS retrieval fails, or errors occur during the KNN prediction logic.
    """

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
    MICTLANX_DELAY          = int(current_app.config.get("MICTLANX_DELAY","2"))
    MICTLANX_BACKOFF_FACTOR = float(current_app.config.get("MICTLANX_BACKOFF_FACTOR","0.5"))
    MICTLANX_MAX_RETRIES    = int(current_app.config.get("MICTLANX_MAX_RETRIES","10"))
  
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
    """
    First interactive phase of the Post-Quantum Secure K-Nearest Neighbors (PQC SKNN) prediction protocol 
    within the Worker node.
    This method utilizes the CKKS homomorphic encryption scheme to process ciphertexts. 
    It orchestrates the retrieval and merging of the encrypted model and test records from the CSS, performs 
    the privacy-preserving distance calculation directly on the PQC ciphertexts, and persists the resulting 
    secure distance matrix back to storage. The process pauses here, awaiting the Client's refresh or partial 
    decryption of the CKKS distances to identify the k-nearest neighbors in the next round.

    Note:
        **Post-Quantum Inference Phase 1**: This step operates exclusively on CKKS ciphertexts. 
        All identifiers, parameters, and matrix shapes must be provided via **HTTP Headers**.

    Attributes:
        Model-Id (str): Root identifier used to locate the CKKS-encrypted training model.
        Records-Test-Id (str): Storage key for the CKKS-encrypted records to be classified.
        Encrypted-Model-Shape (str): Tuple string representing the dimensions of the PQC-encrypted model.
        Encrypted-Model-Dtype (str): Data type of the encrypted model elements.
        Encrypted-Records-Shape (str): Tuple string representing the dimensions of the encrypted test records.
        Encrypted-Records-Dtype (str): Data type of the encrypted record elements.
        Num-Chunks (int): Number of storage fragments for matrix retrieval and persistence.
        Experiment-Id (str): Unique identifier for execution tracing and auditing within the Rory platform.

    Returns:
        distances_id (str): Storage key for the calculated CKKS-encrypted distance matrix.
        distances_shape (str): Dimensions of the resulting PQC distance matrix.
        distances_dtype (str): Data type of the PQC distance elements.
        service_time (float): Total time elapsed during this worker phase.

    Raises:
        Exception: If mandatory headers are missing, CKKS key initialization fails, or errors arise during the retrieval or persistence of lattice-based ciphertext chunks.
    """
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
    _round                   = bool(int(current_app.config.get("_round","0"))) #False
    decimals                 = int(current_app.config.get("DECIMALS","2"))
    experiment_id            = requestHeaders.get("Experiment-Id","")
    MICTLANX_TIMEOUT         = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
    MICTLANX_DELAY           = int(current_app.config.get("MICTLANX_DELAY","2"))
    MICTLANX_BACKOFF_FACTOR  = float(current_app.config.get("MICTLANX_BACKOFF_FACTOR","0.5"))
    MICTLANX_MAX_RETRIES     = int(current_app.config.get("MICTLANX_MAX_RETRIES","10")) 

    MICTLANX_CHUNK_SIZE        = current_app.config.get("MICTLANX_CHUNK_SIZE","256kb")
    MICTLANX_MAX_PARALELL_GETS = int(current_app.config.get("MICTLANX_MAX_PARALELL_GETS","2"))
    
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

    path               = current_app.config.get("KEYS_PATH","/rory/keys")
    ctx_filename       = current_app.config.get("CTX_FILENAME","ctx")
    pubkey_filename    = current_app.config.get("PUBKEY_FILENAME","pubkey")
    secretkey_filename = current_app.config.get("SECRET_KEY_FILENAME","secretkey")
    relinkey_filename  = current_app.config.get("RELINKEY_FILENAME","relinkey")

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
    """
    Final interactive phase of the Post-Quantum Secure K-Nearest Neighbors (PQC SKNN) prediction protocol 
    within the Worker node.
    This method completes the secure classification lifecycle by retrieving the model labels and the resolved 
    nearest-neighbor indices (previously refreshed or decrypted by the Client) from the CSS. It maps these 
    indices to the corresponding class labels to produce the final classification vector. 

    Note:
        **Post-Quantum Inference Phase 2**: This step acts as the final label mapping. 
        All required identifiers for models, labels, and indices must be provided via **HTTP Headers**.

    Attributes:
        Model-Id (str): Root identifier used to locate the model labels and trained parameters.
        Records-Test-Id (str): Storage key used to identify the resolved distance indices.
        Experiment-Id (str): Unique identifier for execution tracing and auditing within the Rory platform.

    Returns:
        label_vector (list): The final predicted class assignments for the test dataset.
        service_time (float): Total execution time for this specific worker phase.

    Raises:
        Exception: If CSS retrieval for labels or indices fails, or if errors arise during the final label vector construction.
        """
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
    MICTLANX_DELAY           = int(current_app.config.get("MICTLANX_DELAY","2"))
    MICTLANX_BACKOFF_FACTOR  = float(current_app.config.get("MICTLANX_BACKOFF_FACTOR","0.5"))
    MICTLANX_MAX_RETRIES     = int(current_app.config.get("MICTLANX_MAX_RETRIES","10")) 

    MICTLANX_CHUNK_SIZE        = current_app.config.get("MICTLANX_CHUNK_SIZE","256kb")
    MICTLANX_MAX_PARALELL_GETS = int(current_app.config.get("MICTLANX_MAX_PARALELL_GETS","2"))

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
    """
    Main routing endpoint for the Post-Quantum Secure K-Nearest Neighbors (PQC SKNN) interactive prediction 
    protocol within the Worker node. 
    This method manages the secure inference lifecycle using CKKS scheme. It acts as 
    an orchestrator that evaluates the protocol's state via a step index, delegating tasks to either the 
    initial PQC-encrypted distance calculation (Step 1) or the final label resolution based on client-aided 
    results (Step 2). This ensures that the classification of sensitive records remains confidential against 
    both classical and quantum-era threats.

    Note:
        **Post-Quantum Protocol Control**: The execution flow and state transitions are managed exclusively via the 'Step-Index' attribute passed in the **HTTP Headers**. The request body is not utilized.

    Attributes:
        Step-Index (int): The current round of the interactive PQC SKNN protocol. Use "1" for the initial CKKS-encrypted distance calculations and "2" for final label assignment. Defaults to "1".
        Model-Id (str): Identifier for the CKKS-encrypted model stored in the CSS.
        Experiment-Id (str): A unique identifier for performance auditing and execution tracing within the Rory platform.

    Returns:
        An object forwarded from the corresponding sub-routine (sknn_pqc_predict_1 or sknn_pqc_predict_2), 
        containing either intermediate PQC secure scores or the final classification results.

    Raises:
        Exception: If the Step-Index is invalid or if the sub-routines encounter failures during CKKS key initialization or CSS retrieval.
        """
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
    