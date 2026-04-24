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
from models import ExperimentLogEntry
from rory.core.security.cryptosystem.pqc.ckks import Ckks, CkksModes

machinelearning = Blueprint("machinelearning",__name__,url_prefix = "/machinelearning")

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

@machinelearning.route("/logisticregression",methods = ["POST"])
async def logisticregression():
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
        algorithm                       = Constants.MachineLearningAlgorithms.LOGISTIC_REGRESSION
        s                               = Session()
        request_headers                 = request.headers #Headers for the request
        experiment_id                   = request_headers.get("Experiment-Id",uuid4().hex[:10])
        plaintext_matrix_train_id       = request_headers.get("Plaintext-Matrix-Train-Id","matrix0")
        plaintext_matrix_test_id        = request_headers.get("Plaintext-Matrix-Test-Id","matrix1")
        plaintext_matrix_train_filename = request_headers.get("Plaintext-Matrix-Train-Filename","matrix0")
        plaintext_matrix_test_filename  = request_headers.get("Plaintext-Matrix-Test-Filename","matrix1")
        extension                       = request_headers.get("Extension","csv")
        epochs                          = int(request_headers.get("Epochs", "1"))
        learning_rate                   = float(request_headers.get("Learning-Rate", "0.01"))
        experiment_id                   = request_headers.get("Experiment-Id",uuid4().hex[:10])
        plaintext_matrix_train_path     = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_train_filename, extension)    
        plaintext_matrix_test_path     = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_test_filename, extension) 

        MAX_ITERATIONS          = int(request_headers.get("Max-Iterations",current_app.config.get("MAX_ITERATIONS",10)))
        WORKER_TIMEOUT          = int(current_app.config.get("WORKER_TIMEOUT",300))
        MICTLANX_TIMEOUT        = int(current_app.config.get("MICTLANX_TIMEOUT",3600))
        MICTLANX_DELAY          = int(current_app.config.get("MICTLANX_DELAY","2"))
        MICTLANX_BACKOFF_FACTOR = float(current_app.config.get("MICTLANX_BACKOFF_FACTOR","0.5"))
        MICTLANX_MAX_RETRIES    = int(current_app.config.get("MICTLANX_MAX_RETRIES","10"))

        logger.debug({
            "algorithm" : algorithm,
            "plaintext_matrix_train_id": plaintext_matrix_train_id,
            "plaintext_matrix_test_id": plaintext_matrix_test_id,
            "plaintext_matrix_train_path": plaintext_matrix_train_path,
            "plaintext_matrix_test_path": plaintext_matrix_test_path,   
            "plaintext_matrix_train_filename": plaintext_matrix_train_filename,
            "plaintext_matrix_test_filename": plaintext_matrix_test_filename,
            "extension" : extension,
            "epoch": epochs, 
            "learning_rate": learning_rate, 
            "max_iterations": MAX_ITERATIONS,
        })

        #_____________________________
        # Leer el dataset de entrenamiento desde source_path
        # plaintext_matrix_train = RoryCommon.read_numpy_from(path, extension)

        # Colocar un logger.debug con un mensaje que indique que se leyo correctamente

        # Partir en chunks el dataset de entrenamiento
        # Chunks.from_ndarray()

        # Colocar un logger.debug

        # Escribir dataset de entrenamiento cifrado en el SAD
        # RoryCommon.delete_and_put_chunks()

        #Colocar un logger.debug
        #_____________________________

        # Leer el dataset de prueba desde source_path
        # plaintext_matrix_test = RoryCommon.read_numpy_from(path, extension)

        # Colocar un logger.debug
        # Partir en chunks el dataset de prueba
        # Chunks.from_ndarray() 

        # Colocar un logger.debug

        # Escribir dataset de prueba cifrado en el SAD
        # RoryCommon.delete_and_put_chunks()
        # Colocar un logger.debug
        # _____________________________

        # Leer el label_vector de entrenamiento desde source_path
        # RoryCommon.read_numpy_from(source_path, ".npy")
        # Colocar un logger.debug

        # Partir en chunks el label_vector de entrenamiento
        # Chunks.from_ndarray()
        # Colocar un logger.debug

        # Escribir label_vector de entrenamiento cifrado en el SAD
        # RoryCommon.delete_and_put_chunks()
        # Colocar un logger.debug
        # _____________________________


        return Response(
            response = json.dumps({
                "x": "This endpoint is under development. Please check back later."                
            }),
            status   = 200,
            headers  = {}
            )
    except Exception as e:
        logger.error("CLIENT_ERROR "+str(e))
        return Response(response = None, status = 500, headers={"Error-Message":str(e)})


@machinelearning.route("/pplr",methods = ["POST"])
async def pplr():
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
        algorithm                       = Constants.MachineLearningAlgorithms.PPLR
        MODE                            = CkksModes.ML
        s                               = Session()
        request_headers                 = request.headers #Headers for the request
        experiment_id                   = request_headers.get("Experiment-Id",uuid4().hex[:10])
        experiment_iteration            = request_headers.get("Experiment-Iteration","0")

        plaintext_matrix_train_id       = request_headers.get("Plaintext-Matrix-Train-Id","matrix-train")
        encrypted_matrix_train_id       = "encrypted{}".format(plaintext_matrix_train_id) # The id of the encrypted matrix is built
        plaintext_matrix_test_id        = request_headers.get("Plaintext-Matrix-Test-Id","matrix-test")
        encrypted_matrix_test_id        = "encrypted{}".format(plaintext_matrix_test_id)
        plaintext_matrix_train_filename = request_headers.get("Plaintext-Matrix-Train-Filename","matrix-train")
        plaintext_matrix_test_filename  = request_headers.get("Plaintext-Matrix-Test-Filename","matrix-test")
        extension                       = request_headers.get("Extension","csv")
        plaintext_matrix_train_path     = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_train_filename, extension)    
        plaintext_matrix_test_path      = "{}/{}.{}".format(SOURCE_PATH, plaintext_matrix_test_filename, extension) 

        epochs                          = int(request_headers.get("Epochs", "1"))
        learning_rate                   = float(request_headers.get("Learning-Rate", "0.01"))
        accuracy_threshold              = float(request_headers.get("Accuracy-Threshold", "0.80"))

        _round             = bool(int(current_app.config.get("_round","0"))) #False
        decimals           = int(current_app.config.get("DECIMALS","4"))
        path               = current_app.config.get("KEYS_PATH","/rory/keys")
        ctx_filename       = current_app.config.get("CTX_FILENAME","ctx")
        pubkey_filename    = current_app.config.get("PUBKEY_FILENAME","pubkey")
        secretkey_filename = current_app.config.get("SECRET_KEY_FILENAME","secretkey")
        relinkey_filename  = current_app.config.get("RELINKEY_FILENAME","relinkey")
        rotatekey_filename  = current_app.config.get("ROTATEKEY_FILENAME","rotatekey")

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
            "plaintext_matrix_test_id": plaintext_matrix_test_id,
            "encrypted_matrix_test_id": encrypted_matrix_test_id,
            "plaintext_matrix_train_path": plaintext_matrix_train_path,
            "plaintext_matrix_test_path": plaintext_matrix_test_path,   
            "plaintext_matrix_train_filename": plaintext_matrix_train_filename,
            "plaintext_matrix_test_filename": plaintext_matrix_test_filename,
            "extension" : extension,
            "epoch": epochs, 
            "learning_rate": learning_rate, 
            "accuracy_threshold": accuracy_threshold,
            "max_iterations": MAX_ITERATIONS,
        })

        ckks = Ckks.from_pyfhel(
            _round             = _round,
            decimals           = decimals,
            path               = path,
            ctx_filename       = ctx_filename,
            pubkey_filename    = pubkey_filename,
            secretkey_filename = secretkey_filename,
            relinkey_filename  = relinkey_filename,
            rotatekey_filename = rotatekey_filename
        )

        #_____________________________
        # Leer del dataset de entrenamiento desde source_path 
        # utilizando el plaintext_matrix_train_filename y extension
        # plaintext_matrix_train = RoryCommon.read_numpy_from(path, extension)

        # Colocar un logger.debug con un mensaje que indique que se leyo 
        # el dataset de entrenamiento
        # logger.debug({
        #     "msg": "Dataset de entrenamiento leído exitosamente")

        # generar variables r que se refiere a los registros a partir del
        # dataset de entrenamiento y a que se refiere a los atributos o caracteristicas 
        #n_samples_train = plaintext_matrix_train.shape[0]
        #n_features_train = plaintext_matrix_train.shape[1]
        
        #generamos la variable n que se refiere al numero de elementos en la matriz
        #n = n_features_train * n_samples_train 
        
        # Colocar un logger.debug con las variables r, a y n generadas
        
        # Cifrar dataset de entrenamiento utilizando CKKS
        # RoryCommon.segment_and_encrypt_with_executor()
        # donde key es el id de la matriz que vas a cifrar
        # n es la variable que acabamos de generar
        
        # Colocar un logger.debug

        # Escribir dataset de entrenamiento cifrado en el SAD
        # RoryCommon.delete_and_put_chunks()

        #Colocar un logger.debug
        #_____________________________


        # Leer del dataset de prueba desde source_path 
        # plaintext_matrix_test = RoryCommon.read_numpy_from(path, extension)

        # Colocar un logger.debug

        # generar variables r y a a partir del dataset de prueba
        #n_samples_test = plaintext_matrix_test.shape[0]
        #n_features_test = plaintext_matrix_test.shape[1]

        #generamos la variable n
        #n = n_features_test * n_samples_test 

        # Colocar un logger.debug
        
        # Cifrar dataset de prueba utilizando CKKS
        # RoryCommon.segment_and_encrypt()
        
        # Colocar un logger.debug

        # Escribir dataset de prueba cifrado en el SAD
        # RoryCommon.delete_and_put_chunks()

        # Colocar un logger.debug
        # _____________________________

        # Obtener número de scale (Escala de CKKS)
        # colocar un logger.debug con el numero de scale obtenido

        # _______________________________
        # generar matriz de pesos vacios de tamano 1 x n_features
        # np.zeros()
        
        # Colocar un logger.debug con la matriz de pesos generada

        # Cifrar matriz de pesos utilizando CKKS
        # RoryCommon.segment_and_encrypt_ckks_with_executor()

        # Colocar un logger.debug con la matriz de pesos cifrada

        # Colocar matriz de pesos cifrada en el SAD
        # RoryCommon.delete_and_put_chunks()

        # Colocar un logger.debug
        # _______________________________

        # Generar vector de sesgo
        # np.zeros() de tamano 1 x 1

        # Colocar logger.debug con el vector de sesgo generado

        # Cifrar vector de sesgo utilizando CKKS
        # RoryCommon.segment_and_encrypt_ckks_with_executor()

        # Colocar logger.debug 
        
        # Colocar vector de sesgo cifrado en el SAD
        # RoryCommon.delete_and_put_chunks()

        # Colocar logger.debug

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


        


        