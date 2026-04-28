import os
import time, json
import numpy as np
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
from rory.core.utils.utils import Utils
from mictlanx import AsyncClient
from mictlanx.utils.segmentation import Chunks
from concurrent.futures import ProcessPoolExecutor
from option import Some
from utils.utils import Utils
from models import ExperimentLogEntry
from rory.core.security.cryptosystem.pqc.ckks import Ckks, CkksModes

machinelearning = Blueprint("machinelearning",__name__,url_prefix = "/machine-learning")

@machinelearning.route("/test",methods=["GET","POST"])
def test():
    """Diagnostic and health check endpoint for the logisticregression component.

    This method provides a simple mechanism to verify that the 
    logisticregression routes are active and reachable. It is primarily used 
    by the Rory platform's orchestration layer to identify the node type 
    and ensure proper network synchronization before initiating machine 
    learning workflows.

    Returns:
        Response: A Flask Response object containing a JSON payload:
            component_type (str): "client".
            
        Headers:
            Component-Type: "client"
            
        Status Code:
            200: If the logisticregression service is operational.
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

@machinelearning.route("/logistic-regression/train",methods = ["POST"])
async def logistic_regression_train():
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
        
        if executor == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm                       = Constants.MachineLearningAlgorithms.LOGISTIC_REGRESSION_TRAIN
        s                               = Session()
        request_headers                 = request.headers #Headers for the request
        experiment_id                   = request_headers.get("Experiment-Id",uuid4().hex[:10])
        experiment_iteration            = request_headers.get("Experiment-Iteration","0")
        plaintext_matrix_train_id       = request_headers.get("Plaintext-Matrix-Train-Id","train_x")
        # plaintext_matrix_test_id        = request_headers.get("Plaintext-Matrix-Test-Id","test_x")
        plaintext_matrix_train_label_id = request_headers.get("Plaintext-Matrix-Train-Label-Id","train_y")
        plaintext_matrix_train_filename = request_headers.get("Plaintext-Matrix-Train-Filename","train_x")
        # plaintext_matrix_test_filename  = request_headers.get("Plaintext-Matrix-Test-Filename","test_x")
        plaintext_matrix_train_label_filename = request_headers.get("Plaintext-Matrix-Train-Label-Filename","train_y")
        extension                       = request_headers.get("Extension","csv")
        epochs                          = int(request_headers.get("Epochs", "1"))
        learning_rate                   = float(request_headers.get("Learning-Rate", "0.01"))
        plaintext_matrix_train_path     = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_train_filename, extension)    
        # plaintext_matrix_test_path      = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_test_filename, extension) 
        plaintext_matrix_train_label_path     = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_train_label_filename, extension)    
        MAX_ITERATIONS          = int(request_headers.get("Max-Iterations",current_app.config.get("MAX_ITERATIONS",10)))
        WORKER_TIMEOUT          = int(current_app.config.get("WORKER_TIMEOUT",300))
        MICTLANX_TIMEOUT        = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
        MICTLANX_DELAY          = int(current_app.config.get("MICTLANX_DELAY","2"))
        MICTLANX_BACKOFF_FACTOR = float(current_app.config.get("MICTLANX_BACKOFF_FACTOR","0.5"))
        MICTLANX_MAX_RETRIES    = int(current_app.config.get("MICTLANX_MAX_RETRIES","10"))

        logger.debug({
            "algorithm" : algorithm,
            "plaintext_matrix_train_id": plaintext_matrix_train_id,
            # "plaintext_matrix_test_id": plaintext_matrix_test_id,
            "plaintext_matrix_train_path": plaintext_matrix_train_path,
            # "plaintext_matrix_test_path": plaintext_matrix_test_path,   
            "plaintext_matrix_train_filename": plaintext_matrix_train_filename,
            # "plaintext_matrix_test_filename": plaintext_matrix_test_filename,
            "plaintext_matrix_train_label_id": plaintext_matrix_train_label_id,
            "plaintext_matrix_train_label_path": plaintext_matrix_train_label_path,
            "plaintext_matrix_train_label_filename": plaintext_matrix_train_label_filename,
            "extension" : extension,
            "epoch": epochs, 
            "learning_rate": learning_rate, 
            "max_iterations": MAX_ITERATIONS,
        })
        plaintext_matrix_train_result = await RoryCommon.read_numpy_from(
            path = plaintext_matrix_train_path, 
            extension = extension
        )

        if plaintext_matrix_train_result.is_err:
            return Response(status=500, response="Failed to read plaintext matrix train")

        plaintext_matrix_train = plaintext_matrix_train_result.unwrap()
        
        logger.debug({
            "msg": "Training dataset read successfully"
        })

        plaintext_matrix_train_chunks = Chunks.from_ndarray(
            ndarray      = plaintext_matrix_train, 
            group_id     = plaintext_matrix_train_id,
            num_chunks   = num_chunks,
            chunk_prefix = Some(plaintext_matrix_train_id)
            )
    
        logger.debug({
            "msg": "Training dataset split into chunks"
        })

        plaintext_train_put_chunk = await RoryCommon.delete_and_put_chunks(
            client    = STORAGE_CLIENT,
            bucket_id = BUCKET_ID,
            key       = plaintext_matrix_train_id,
            chunks    = plaintext_matrix_train_chunks.unwrap(),
            timeout   = MICTLANX_TIMEOUT,
            max_tries = MICTLANX_MAX_RETRIES,
            tags = {
                "shape": str(plaintext_matrix_train.shape),
                "dtype": str(plaintext_matrix_train.dtype)
            }
        )

        logger.debug({
            "msg": "Training dataset in cloud storage"
        })
        #__________________________

        # plaintext_matrix_test_result = await RoryCommon.read_numpy_from(
        #     path = plaintext_matrix_test_path, 
        #     extension = extension
        # )

        # if plaintext_matrix_test_result.is_err:
        #     return Response(status=500, response="Failed to read plaintext matrix test")

        # plaintext_matrix_test = plaintext_matrix_test_result.unwrap()
        
        # logger.debug({
        #     "msg": "Test dataset read successfully"
        # })

        # plaintext_matrix_test_chunks = Chunks.from_ndarray(
        #         ndarray      = plaintext_matrix_test, 
        #         group_id     = plaintext_matrix_test_id,
        #         num_chunks   = num_chunks,
        #         chunk_prefix = Some(plaintext_matrix_test_id)
        #         )
        
        # logger.debug({
        #     "msg": "Test dataset split into chunks"
        # })

        # plaintext_test_put_chunk = await RoryCommon.delete_and_put_chunks(
        #     client    = STORAGE_CLIENT,
        #     bucket_id = BUCKET_ID,
        #     key       = plaintext_matrix_test_id,
        #     chunks    = plaintext_matrix_test_chunks.unwrap(),
        #     timeout   = MICTLANX_TIMEOUT,
        #     max_tries = MICTLANX_MAX_RETRIES,
        #     tags = {
        #         "shape": str(plaintext_matrix_test.shape),
        #         "dtype": str(plaintext_matrix_test.dtype)
        #     }
        # )

        # logger.debug({
        #     "msg": "Test dataset in cloud storage"
        # })
        #________________________________

        plaintext_matrix_train_label_result = await RoryCommon.read_numpy_from(
            path = plaintext_matrix_train_label_path, 
            extension = extension
        )
        if plaintext_matrix_train_label_result.is_err:
            return Response(status=500, response="Failed to read training label dataset")

        plaintext_matrix_train_label = plaintext_matrix_train_label_result.unwrap()
        
        logger.debug({
            "msg": "Training label vector dataset read successfully"
        })

        plaintext_matrix_train_label_chunks = Chunks.from_ndarray(
                ndarray      = plaintext_matrix_train_label, 
                group_id     = plaintext_matrix_train_label_id,
                num_chunks   = num_chunks,
                chunk_prefix = Some(plaintext_matrix_train_label_id)
                )
        
        logger.debug({
            "msg": "Training label vector dataset split into chunks"
        })

        plaintext_train_label_put_chunk = await RoryCommon.delete_and_put_chunks(
            client    = STORAGE_CLIENT,
            bucket_id = BUCKET_ID,
            key       = plaintext_matrix_train_label_id,
            chunks    = plaintext_matrix_train_label_chunks.unwrap(),
            timeout   = MICTLANX_TIMEOUT,
            max_tries = MICTLANX_MAX_RETRIES,
            tags = {
                "shape": str(plaintext_matrix_train_label.shape),
                "dtype": str(plaintext_matrix_train_label.dtype)
            }
        )

        logger.debug({
            "msg": "Training label vector dataset in cloud storage"
        })


        logger.debug({
            "msg": "Begin the comunication"
        })
        # Comunicarse con el manager y con el worker
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
        logger.debug({
            "msg": "Complete comunication",
            "worker id": worker_id
        })

        worker = RoryWorker( #Allows to establish the connection with the worker
            workerId  = worker_id,
            port      = port,
            session   = s,
            algorithm = algorithm,
        )
        
        status = Constants.ClusteringStatus.START #Set the status to start
        iteration = 0

        worker_headers = {
            "Clustering-Status"         : str(status),
            "Experiment-Id"             : experiment_id,
            "Iterations"                : str(iteration),
            "Plaintext-Matrix-Train-Id" : plaintext_matrix_train_id,
            # "Plaintext-Matrix-Test-Id"  : plaintext_matrix_test_id,
            "Plaintext-Matrix-Train-Label-Id" : plaintext_matrix_train_label_id,
            "Epochs"                 : str(epochs),
            "Learning-Rate"          : str(learning_rate),
        }

        logger.debug({
            "msg": "Connection with the worker"
        })
        # enviarle headers al worker 
        worker_response = worker.run(
                timeout = WORKER_TIMEOUT, 
                headers = worker_headers
            ) #Run 1 starts
        worker_status = worker_response.status_code

        logger.debug({
            "worker_status": str(worker_response),
            "worker_id": worker_id,
            "worker_port" : port,
        })

        if worker_status !=200:
            return Response("Worker error: {}".format(worker_response.content),status=500)

        logger.debug({
            "msg": "Worker response"
        })
        worker_response.raise_for_status()

        jsonWorkerResponse        = worker_response.json()
        run1_out_predictions_id   = jsonWorkerResponse["out_predictions_id"]

        logger.debug({
            "run1_out_predictions_id": run1_out_predictions_id, 
        })

        return Response(
            response = json.dumps({
                "x": "This endpoint is under development. Please check back later."                
            }),
            status   = 200,
            headers  = {}
            )
    except Exception as e:
        logger.error("CLIENT_ERROR "+str(e))
        return Response(
            response = None, 
            status = 500, 
            headers={"Error-Message":str(e)})

@machinelearning.route("/logistic-regression/predict",methods = ["POST"])
async def logistic_regression_predict():
    pass

@machinelearning.route("/pplr/train",methods = ["POST"])
async def pplr_train():
    try:
        arrivalTime                  = time.time()
        logger                       = current_app.config["logger"]
        BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
        TESTING                      = current_app.config.get("TESTING",True)
        SOURCE_PATH                  = current_app.config["SOURCE_PATH"]
        STORAGE_CLIENT:AsyncClient   = current_app.config.get("ASYNC_STORAGE_CLIENT")
        max_workers                  = current_app.config.get("MAX_WORKERS",2)
        num_chunks                   = current_app.config.get("NUM_CHUNKS",2)
        np_random                    = current_app.config.get("np_random")
        executor:ProcessPoolExecutor = current_app.config.get("executor")
        security_level               = current_app.config.get("LIU_SECURITY_LEVEL",128)
        if executor == None:
            raise Response(None, status=500, headers={"Error-Message":"No process pool executor available"})
        algorithm                       = Constants.MachineLearningAlgorithms.PPLR_TRAIN
        MODE                            = CkksModes.ML
        s                               = Session()
        request_headers                 = request.headers #Headers for the request
        experiment_id                   = request_headers.get("Experiment-Id",uuid4().hex[:10])
        experiment_iteration            = request_headers.get("Experiment-Iteration","0")
        plaintext_matrix_train_id       = request_headers.get("Plaintext-Matrix-Train-Id","train_x")
        encrypted_matrix_train_id       = "encrypted{}".format(plaintext_matrix_train_id) # The id of the encrypted matrix is built
        plaintext_label_vector_train_id = request_headers.get("Plaintext-Label-Vector-Train-Id","train_y")
        encrypted_label_vector_train_id = "encrypted{}".format(plaintext_label_vector_train_id)
        plaintext_matrix_train_filename = request_headers.get("Plaintext-Matrix-Train-Filename","train_x")
        plaintext_label_vector_train_filename = request_headers.get("Plaintext-Label-Vector-Train-Filename","train_y")
        extension                         = request_headers.get("Extension","csv")
        plaintext_matrix_train_path       = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_train_filename, extension)
        plaintext_label_vector_train_path = "{}/{}.{}".format(SOURCE_PATH, plaintext_label_vector_train_filename, extension)
        
        epochs             = int(request_headers.get("Epochs", "1"))
        learning_rate      = float(request_headers.get("Learning-Rate", "0.01"))
        accuracy_threshold = float(request_headers.get("Accuracy-Threshold", "0.80"))

        plaintext_weight_id = request_headers.get("Plaintext-Weight-Id","weight")
        encrypted_weight_id = "encrypted{}".format(plaintext_weight_id)
        plaintext_bias_id   = request_headers.get("Plaintext-Bias-Id","bias")
        encrypted_bias_id   = "encrypted{}".format(plaintext_bias_id)

        _round             = bool(int(current_app.config.get("_round","0")))            #False
        decimals           = int(current_app.config.get("DECIMALS","4"))
        keys_path          = current_app.config.get("KEYS_PATH","/rory/keys/keys128")
        ctx_filename       = current_app.config.get("CTX_FILENAME","ctx")
        pubkey_filename    = current_app.config.get("PUBKEY_FILENAME","pubkey")
        secretkey_filename = current_app.config.get("SECRET_KEY_FILENAME","secretkey")
        relinkey_filename  = current_app.config.get("RELINKEY_FILENAME","relinkey")
        rotatekey_filename = current_app.config.get("ROTATEKEY_FILENAME","rotatekey")

        MAX_ITERATIONS          = int(request_headers.get("Max-Iterations",current_app.config.get("MAX_ITERATIONS",10)))
        WORKER_TIMEOUT          = int(current_app.config.get("WORKER_TIMEOUT",300))
        MICTLANX_TIMEOUT        = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
        MICTLANX_DELAY          = int(current_app.config.get("MICTLANX_DELAY","2"))
        MICTLANX_BACKOFF_FACTOR = float(current_app.config.get("MICTLANX_BACKOFF_FACTOR","0.5"))
        MICTLANX_MAX_RETRIES    = int(current_app.config.get("MICTLANX_MAX_RETRIES","10"))

        logger.debug({
            "algorithm" : algorithm,
            "plaintext_matrix_train_id": plaintext_matrix_train_id,
            "encrypted_matrix_train_id": encrypted_matrix_train_id,
            "plaintext_matrix_train_path": plaintext_matrix_train_path,
            "plaintext_matrix_train_filename": plaintext_matrix_train_filename,
            # "plaintext_matrix_test_filename": plaintext_matrix_test_filename,
            "plaintext_matrix_train_label_id": plaintext_label_vector_train_id,
            "encrypted_matrix_train_label_id": encrypted_label_vector_train_id,
            "plaintext_matrix_train_label_path": plaintext_label_vector_train_path, 
            "plaintext_matrix_train_label_filename": plaintext_matrix_train_filename,
            "extension" : extension,
            "plaintext_weight_matrix_id": plaintext_weight_id,
            "encrypted_weight_matrix_id": encrypted_weight_id,
            "plaintext_bias_vector_id": plaintext_bias_id,
            "encrypted_bias_vector_id": encrypted_bias_id,
            "epoch": epochs, 
            "learning_rate": learning_rate, 
            "accuracy_threshold": accuracy_threshold,
            "max_iterations": MAX_ITERATIONS,
        })
        
        ckks = Ckks.from_pyfhel(
            _round             = _round,
            decimals           = decimals,
            path               = keys_path,
            ctx_filename       = ctx_filename,
            pubkey_filename    = pubkey_filename,
            secretkey_filename = secretkey_filename,
            relinkey_filename  = relinkey_filename,
            rotatekey_filename = rotatekey_filename
        )

        max_workers      = Utils.get_workers(num_chunks=num_chunks)
        #_______________________________

        plaintext_matrix_train_result = await RoryCommon.from_matrix_on_disk_to_cloud_storage_ckks(
            path               = plaintext_matrix_train_path,
            extension          = extension,
            client             = STORAGE_CLIENT,
            bucket_id          = BUCKET_ID,
            ball_id            = encrypted_matrix_train_id,
            keys_path          = keys_path,
            ctx_filename       = ctx_filename,
            pubkey_filename    = pubkey_filename,
            secretkey_filename = secretkey_filename,
            relinkey_filename  = relinkey_filename,
            rotatekey_filename = rotatekey_filename,
            decimals           = decimals,
            num_chunks         = num_chunks,
            tags               = { },
            timeout      = MICTLANX_TIMEOUT,
            max_attempts = MICTLANX_MAX_RETRIES,
        )
        logger.debug({
            "msg": "Read, segment, encrypt and put in storage dataset train"
        })
        #_________________
        plaintext_label_vector_train = await RoryCommon.from_matrix_on_disk_to_cloud_storage_ckks(
            path               = plaintext_label_vector_train_path,
            extension          = extension,
            client             = STORAGE_CLIENT,
            bucket_id          = BUCKET_ID,
            ball_id            = encrypted_label_vector_train_id,
            keys_path          = keys_path,
            ctx_filename       = ctx_filename,
            pubkey_filename    = pubkey_filename,
            secretkey_filename = secretkey_filename,
            relinkey_filename  = relinkey_filename,
            rotatekey_filename = rotatekey_filename,
            decimals           = decimals,
            num_chunks         = num_chunks,
            tags               = { },
            timeout      = MICTLANX_TIMEOUT,
            max_attempts = MICTLANX_MAX_RETRIES,

        )
        #_________________
        scale = ckks.SECURITY_LEVELS[MODE.value][security_level]["scale"]
        logger.debug({
            "scale": scale
        })

        n_features=8
        n_samples=2
        plaintext_weight = np.zeros((1,n_features), dtype=np.float32)

        logger.debug({
            "weights matrix": plaintext_weight.shape,
            "weights matrix": str(plaintext_weight.dtype)
        })

        encrypted_weight_result = await RoryCommon.from_vector_to_cloud_storage_ckks(
            vector             = plaintext_weight,
            client             = STORAGE_CLIENT,
            bucket_id          = BUCKET_ID,
            ball_id            = encrypted_weight_id,
            keys_path          = keys_path,
            ctx_filename       = ctx_filename,
            pubkey_filename    = pubkey_filename,
            secretkey_filename = secretkey_filename,
            relinkey_filename  = relinkey_filename,
            rotatekey_filename = rotatekey_filename,
            decimals           = decimals,
            num_chunks         = num_chunks,
            tags               = { },
            timeout            = MICTLANX_TIMEOUT,
            max_attempts       = MICTLANX_MAX_RETRIES,
            _round             = False
        )

        logger.debug({
            "msg": "Segment, encrypt and put in storage plaintext weights"
        })
        #_________________

        plaintext_bias = np.array([0.0], dtype=np.float32)

        logger.debug({
            "bias vector shape": plaintext_bias.shape
        })
        
        encrypted_bias_result = await RoryCommon.from_vector_to_cloud_storage_ckks(
            vector             = plaintext_bias,
            client             = STORAGE_CLIENT,
            bucket_id          = BUCKET_ID,
            ball_id            = encrypted_bias_id,
            keys_path          = keys_path,
            ctx_filename       = ctx_filename,
            pubkey_filename    = pubkey_filename,
            secretkey_filename = secretkey_filename,
            relinkey_filename  = relinkey_filename,
            rotatekey_filename = rotatekey_filename,
            decimals           = decimals,
            num_chunks         = num_chunks,
            tags               = { },
            timeout            = MICTLANX_TIMEOUT,
            max_attempts       = MICTLANX_MAX_RETRIES,
            _round             = False
        )
        #__________________________
        
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
        logger.debug({
            "msg": "Complete comunication",
            "worker id": worker_id
        })
        
        worker = RoryWorker( #Allows to establish the connection with the worker
            workerId  = worker_id,
            port      = port,
            session   = s,
            algorithm = algorithm,
        )

        status = Constants.ClusteringStatus.START #Set the status to start
        iteration = 0
        

        worker_headers = {
            "Clustering-Status"      : str(status),
            "Experiment-Id"          : experiment_id,
            "Epochs"                 : str(epochs),
            "Learning-Rate"          : str(learning_rate),
            "Accuracy-Threshold"     : str(accuracy_threshold),
            "Iterations"             : str(iteration),
            "Encrypted-Matrix-Train-Id": encrypted_matrix_train_id,
            "Encrypted-Matrix-Train-Label-Id": encrypted_matrix_train_id,
            "Encrypted-Weights-Id": encrypted_weight_id,
            "Encrypted-Bias-Id": encrypted_bias_id, 
            "Scale": str(scale),
            "N-Features": str(n_features),
            "N-Samples": str(n_samples),
        }

        logger.debug({
            "msg": "Connection with the worker"
        })
        # # enviarle headers al worker 
        worker_response = worker.run(
                timeout = WORKER_TIMEOUT, 
                headers = worker_headers
            ) #Run 1 starts
        worker_status = worker_response.status_code

        logger.debug({
            "worker_status": str(worker_response),
            "worker_id"    : worker_id,
            "worker_port"  : port,
        })

        if worker_status !=200:
            return Response("Worker error: {}".format(worker_response.content),status=500)
        
        worker_response.raise_for_status()
        jsonWorkerResponse                = worker_response.json()
        run1_encrypted_out_weights_id     = jsonWorkerResponse["encrypted_out_weights_id"]
        run1_encrypted_out_bias_id        = jsonWorkerResponse["encrypted_out_bias_id"]
        run1_encrypted_out_predictions_id = jsonWorkerResponse["encrypted_out_predictions_id"]

        logger.debug({
            "run1_encrypted_out_weights_id"     : run1_encrypted_out_weights_id,
            "run1_encrypted_out_bias_id"        : run1_encrypted_out_bias_id,
            "run1_encrypted_out_predictions_id" : run1_encrypted_out_predictions_id, 
        })

        return Response(
            response = json.dumps({
                "x": "This endpoint is under development. Please check back later."
            }),
            status   = 200,
            headers  = {}
        )

    except Exception as e:
        logger.error({
            "msg":str(e)
        })
        return Response(
            response = None, 
            status = 500, 
            headers={"Error-Message":str(e)}
        )


@machinelearning.route("/pplr/predict",methods = ["POST"])
async def pplr_predict():   
    # Identificar headers para la prediccion.
    # Definir headers base, los que se extraen con current_app.config
    # Definir headers para predict, por ejemplo, vector de pesos, dataset_test, dataset_test_id, n_test, etc.
    # Agregar logger debug con cada cada header.
    # Verifica que los valores que estan en el logger debug sean los correctos.

    # Inicializar ckks con Ckks.from_pyfhel
    # Leer cada dataset que se tenga que usar en esta etapa. Por ejemplo dataset_test con read_numpy_from
    # Colocar logger.debu con informacion importante para el siguiente paso
    # Cifrar cada dataset con segment_and_encrypt_ckks_with_executor
    # Colocar logger.debug con informacion importante para el siguiente paso
    # Subir cada dataset cifrado al storage con delete_and_put_chunks
    # Colocar logger.debug 

    # Agrega el return Response con un mensaje. Incluye response, status=200 y headers={}.
    pass

        