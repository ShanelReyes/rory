import sys
import os
import requests 
import pandas as pd
import logging
import time
import numpy as np
from rory.core.interfaces.client_response import ClientResponse
from rory.core.logger.Logger import create_logger
from option import Some
import numpy.typing as npt
from dotenv import load_dotenv
from retry.api import retry_call
from typing import Dict
from concurrent.futures import ThreadPoolExecutor,wait,ALL_COMPLETED,as_completed
from typing import Tuple,List 
import pandas._typing as pdt
import requests as R
from option import Result,Ok,Err
import json
from mictlanx.logger.log import Log
import string
from nanoid import generate as nanogenerate

load_dotenv()

NODE_ID                   = os.environ.get("NODE_ID","rory-dataowner-0") 
CLIENT_ID                 = os.environ.get("CLIENT_ID","rory-client-0")
CLIENT_INDEX              = os.environ.get("CLIENT_INDEX","0")
CLIENT_PORT               = os.environ.get("CLIENT_PORT",3000)
CLIENT_IP_ADDR            = os.environ.get("CLIENT_IP_ADDR","localhost")
TASK_ID                   = os.environ.get("TASK_ID","CLUSTERING")
ALGORITHM                 = os.environ.get("ALGORITHM","KMEANS")
BATCH_ID                  = os.environ.get("BATCH_ID",0)
TRACE_ID                  = os.environ.get("TRACE_ID",ALGORITHM)
MAX_EXPERIMENT_ITERATIONS = int(os.environ.get("EXPERIMENT_ITERATION",31))
BATCH_INDEX               = "batch_{}".format(BATCH_ID)
LOGGER_NAME               = NODE_ID
MAX_RETRIES               = int(os.environ.get("MAX_RETRIES","10"))
EXPERIMENT_ID             = "{}_C{}".format(TASK_ID,CLIENT_INDEX)
MAX_THREADS               = int(os.environ.get("MAX_THREADS",1))
TRACE_EXTENSION           = os.environ.get("TRACE_EXTENSION","csv")
DATASET_EXTENSION         = os.environ.get("DATASET_EXTENSION","csv")
SOURCE_PATH               = os.environ.get("SOURCE_PATH","/rory/source")
SINK_PATH                 = os.environ.get("SINK_PATH","/rory/sink")
LOG_PATH                  = os.environ.get("LOG_PATH","/rory/log")
TRACE_PATH                = os.environ.get("TRACE_PATH","{}/{}.{}".format(SOURCE_PATH,TRACE_ID,TRACE_EXTENSION))
CLIENT_TIMEOUT            = int(os.environ.get("CLIENT_TIMEOUT",300))

try:
    os.makedirs(SOURCE_PATH,exist_ok = True)
    os.makedirs(SINK_PATH,  exist_ok = True)
    os.makedirs(LOG_PATH,   exist_ok = True)
except Exception as e:
    print("MAKE_FOLDER_ERROR",e)

LOGGER = Log(
    name                   = LOGGER_NAME,
    path                   = LOG_PATH,
    console_handler_filter = lambda record: record.levelno == logging.DEBUG or record.levelno == logging.INFO or record.levelno == logging.ERROR,
    interval               = 24,
    when                   = "h"
)

def write_to_file(filename:str, lv:npt.NDArray):
    path = "{}/{}.npy".format(SINK_PATH,filename)
    try:
        with open(path, "wb") as f:
            np.save(f,lv)
        
    except Exception as e:
        LOGGER.error(str(e))

def client_request(row:pd.Series,url:str,headers:Dict[str,str], timeout:int = 300):
    try:
        response = requests.post(
            url     = url,
            headers = headers,
            timeout = timeout
        )
        response.raise_for_status()
        return response
    except Exception as e:
        print(e)
        LOGGER.error("Error to process {} ".format(row["DATASET_ID"]))
        raise e

def clustering_experiment(row:pd.Series,current_experiment_iteration:int):
    try:
        arrival_time      = time.time()
        id_nanoid         = nanogenerate(size=4,alphabet=string.ascii_lowercase)
        plainTextMatrixId = str(row["DATASET_ID"])
        sens              = row.get("SENS","0.00001")
        max_iterations    = row.get("MAX_ITERATIONS","10")
        k                 = row.get("K","2")
        threshold         = row.get("THRESHOLD",-1)

        LOGGER.debug({
            "event":"CLUSTERING.STARTED",
            "algorithm":ALGORITHM,
            "matrix_id":plainTextMatrixId,
            "clustering_max_iterations": max_iterations,
            "k": k,
            "sens":sens,
            "current_iterations":current_experiment_iteration,
            "experiment_max_iterations": MAX_EXPERIMENT_ITERATIONS,
        })
        
        headers = {
            "Plaintext-Matrix-Id": "{}{}".format(plainTextMatrixId,current_experiment_iteration),
            "Plaintext-Matrix-Filename":plainTextMatrixId,
            "K": str(k),
            "Sens": str(sens),
            "Extension": DATASET_EXTENSION,
            "Client-Id": CLIENT_ID,
            "Max-Iterations": str(max_iterations),
            "Experiment-Iteration": str(current_experiment_iteration),
            "Threshold":str(threshold)
        }

        url = "http://{}:{}/clustering/{}".format(CLIENT_IP_ADDR,CLIENT_PORT,ALGORITHM.lower())

        _response   = client_request(
            row     = row,
            url     = url,
            headers = headers, 
            timeout = CLIENT_TIMEOUT
        )
        _response.raise_for_status()
        label_vector_id = "{}{}{}".format(plainTextMatrixId,ALGORITHM,current_experiment_iteration)

        stringClientResponse  = _response.content.decode("utf-8") #Response from worker
        jsonClientResponse    = json.loads(stringClientResponse) #pass to json
        labelVector             = jsonClientResponse["label_vector"]
        service_time_manager    = jsonClientResponse["service_time_manager"]
        service_time_worker     = jsonClientResponse["service_time_worker"]
        service_time_client     = jsonClientResponse["service_time_client"]
        service_time_clustering = jsonClientResponse["response_time_clustering"]


        write_to_file(label_vector_id,labelVector)
        end_time       = time.time() # Get the time when it ends
        response_time = end_time - arrival_time 

        LOGGER.info({
            "event":"CLUSTERING.COMPLETED",
            "matrix_id":plainTextMatrixId,
            "algorithm":ALGORITHM,
            "arrival_time":arrival_time,
            "end_time":end_time,
            "response_time":response_time,
            "current_iteration":current_experiment_iteration,
            "max_iterations":max_iterations,
            "service_time_manager":service_time_manager,
            "service_time_worker":service_time_worker,
            "service_time_client":service_time_client,
            "response_time_client":service_time_clustering
        })
        
        print("_"*40)
        return Ok((row,_response,current_experiment_iteration))
    except Exception as e:
        LOGGER.error("DATAOWNER_ERROR "+str(e))
        return Err((row,Exception,current_experiment_iteration))

def classification_experiment(row:pd.Series,current_experiment_iteration:int)->Result[Tuple[pd.Series,R.Response,int],Tuple[pd.Series,Exception, int]]:
    try:
        arrivalTime         = time.time()
        id_nanoid           = nanogenerate(size=4,alphabet=string.ascii_lowercase)
        matrixId            = str(row["DATASET_ID"]) #fertility
        modelId             = "{}{}{}".format(matrixId,current_experiment_iteration,id_nanoid) #fertility-0
        modelFilename       = "{}model".format(matrixId) #fertility_model
        modelLabelsFilename = "{}labels".format(modelFilename) #fertility_model_labels
        recordsTestId       = "{}{}data{}".format(matrixId,current_experiment_iteration,id_nanoid) #fertility-0_data
        recordsTestFilename = "{}data".format(matrixId) #fertility_data
        
        LOGGER.debug({
            "event":"CLASSIFICATION.TRAIN.STARTED",
            "algorithm":ALGORITHM,
            "matrix_id":matrixId,
            "model_id":modelId,
            "model_filename":modelFilename,
            "model-labels_filename":modelLabelsFilename,
            "records_test_id":recordsTestId,
            "records_test_filename":recordsTestFilename,
            "current_experiment_iteration":current_experiment_iteration,
            "experiment_max_iterations": MAX_EXPERIMENT_ITERATIONS,
        })
            
        headers = {
            "Model-Id": modelId,
            "Model-Filename":modelFilename, #fertility_model
            "Model-Labels-Filename":modelLabelsFilename,
            "Extension": DATASET_EXTENSION,
            "Client-Id": CLIENT_ID,
            "Experiment-Iteration": str(current_experiment_iteration)
        }

        url_train = "http://{}:{}/classification/{}/train".format(CLIENT_IP_ADDR,CLIENT_PORT,ALGORITHM.lower())

        _response   = client_request(
            row     = row,
            url     = url_train,
            headers = headers, 
            timeout = CLIENT_TIMEOUT
        )
        _response.raise_for_status()
        stringClientResponse  = _response.content.decode("utf-8") #Response from worker
        jsonClientResponse    = json.loads(stringClientResponse) #pass to json
        encrypted_model_shape = jsonClientResponse.get("encrypted_model_shape",0)
        encrypted_model_Dtype = jsonClientResponse.get("encrypted_model_dtype",0)
        service_time_train    = jsonClientResponse.get("response_time",0)

        LOGGER.debug({
            "event":"CLASSIFICATION.PREDICT.STARTED",
            "algorithm":ALGORITHM,
            "matrix_id":matrixId,
            "model_id":modelId,
            "model_filename":modelFilename,
            "model-labels_filename":modelLabelsFilename,
            "records_test_id":recordsTestId,
            "records_test_filename":recordsTestFilename,
            "encrypted_model_shape":str(encrypted_model_shape),
            "encrypted_model_dtype":str(encrypted_model_Dtype),
            "current_experiment_iteration":current_experiment_iteration,
            "experiment_max_iterations": MAX_EXPERIMENT_ITERATIONS,
            "train_service_time":service_time_train
        })
        headers_pred = {
            "Model-Id": modelId,
            "Model-Filename":modelFilename,
            "Model-Labels-Filename":modelLabelsFilename,
            "Records-Test-Id": recordsTestId,
            "Records-Test-Filename":recordsTestFilename,
            "Extension": DATASET_EXTENSION,
            "Client-Id": CLIENT_ID,
            "Experiment-Iteration": str(current_experiment_iteration),
            "Encrypted-Model-Shape": str(encrypted_model_shape),
            "Encrypted-Model-Dtype": str(encrypted_model_Dtype)

        }
        url_predict = "http://{}:{}/classification/{}/predict".format(CLIENT_IP_ADDR,CLIENT_PORT,ALGORITHM.lower())
        _response   = client_request(
            row     = row,
            url     = url_predict,
            headers = headers_pred, 
            timeout = CLIENT_TIMEOUT
        )

        stringClient2Response = _response.content.decode("utf-8") #Response from worker
        jsonClient2Response   = json.loads(stringClient2Response) #pass to json
        labelVectorId         = "{}{}{}".format(matrixId,ALGORITHM,current_experiment_iteration)
        labelVector           = jsonClient2Response["label_vector"]
        service_time_manager  = jsonClient2Response["service_time_manager"]
        service_time_worker   = jsonClient2Response["service_time_worker"]
        service_time_client   = jsonClientResponse["service_time_client"]
        service_time_predict  = jsonClient2Response["service_time_predict"]
        
        write_to_file(labelVectorId,labelVector)
        endTime       = time.time() # Get the time when it ends
        response_time = endTime - arrivalTime 
        
        LOGGER.info({
            "event":"CLASSIFICATION.COMPLETED",
            "algorithm":ALGORITHM,
            "matrix_id":matrixId,
            "model_id":modelId,
            "record_test_id":recordsTestId,
            "experiment_iteration": current_experiment_iteration,
            "arrival_time":arrivalTime,
            "end_time":endTime,
            "service_time_manager":service_time_manager,
            "service_time_worker":service_time_worker,
            "service_time_train":service_time_train,
            "service_time_client_predict":service_time_client,
            "service_time_predict":service_time_predict,
            "response_time":response_time
        })
        print("_"*40)
        return Ok((row, R.Response(),current_experiment_iteration))
    except Exception as e:
        LOGGER.error("DATAOWNER_ERROR "+str(e))
        return Err((row,Exception,current_experiment_iteration))

def run_experiment(row:pd.Series,current_experiment_iteration:int)->Result[Tuple[pd.Series,R.Response,int],Tuple[pd.Series,R.Response, int]]:
    algorithm = row["ALGORITHM"]
    if algorithm == "KNN" or  algorithm =="SKNN":
        return classification_experiment(row=row, current_experiment_iteration=current_experiment_iteration)
    else:
        return clustering_experiment(row=row, current_experiment_iteration=current_experiment_iteration)


def main(trace_df:pd.DataFrame,max_experiment_iterations:int= 31)->Result[int, pd.DataFrame]:
    failed_operations:List[pd.Series]  = [] # Lista de operaciones fallidas
    try:
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor: # Configuracion de la THREADPOOL. 
            start_time = time.time() # Tiempo de inicio de la experimentacion.
            for index,row in trace_df.iterrows(): # Iteracion de los registros de la traza.
                futures    = [] # Lista de futuros (operaciones asincronas/no bloqueantes)
                for experiment_iteration in range(max_experiment_iterations): # Cada registro se repetira EXPERIMENT_ITERATION veces
                    fut = executor.submit(run_experiment,row,experiment_iteration) # Lanzar la operacion a un thread utilizando la Thread Pool. 
                    futures.append(fut) # Añadir a la lista de futuros el nuevo futuro.
                    time.sleep(row["INTERARRIVAL_TIME"]) # Duerme el thread para esperar INTERARRIVAL_TIME
                
                for fut in as_completed(futures): # Espera para completar todos los EXPERIMENT_ITERATIONS 
                    result:Result[Tuple[pd.Series,R.Response, int], Tuple[pd.Series, Exception, int]] = fut.result() # Saca el resultado del futuro
                    if result.is_err: # Si falla                         
                        (failed_row, error_response, experiment_iteration) = result.unwrap_err() # Sacamos la parte derecha con el método unwrap_err() del Result[T,Error] <- extraemos la Error.
                        datasetId = failed_row["DATASET_ID"] # Sacamos el DatasetID
                        LOGGER.error({
                            "dataset_id":datasetId,
                            "msg":str(error_response),
                            "experiment_iteration":experiment_iteration,
                            "failed_operations":len(failed_operations)+1
                        })
                        print("_"*40)
                        failed_operations.append(failed_row) # Añadimos la fila a la lista de operaciones fallidas
                    else:
                        (_row, _, experiment_iteration) = result.unwrap() # Extrae la parte buena con unwrap()
                        datasetId = _row["DATASET_ID"] # Mostramos informacion de la operacion exitosa.
                        LOGGER.debug({
                            "event":"DATASET.COMPLETED",
                            "dataset_id": datasetId,
                            "current_iteration":experiment_iteration,
                        })    
                        print("_"*40)
            end_time = time.time()
            total_time = end_time - start_time
            failed_operations_len = len(failed_operations)
            LOGGER.info({
                "event":"EXPERIMENT.COMPLETED",
                "failed_operations":failed_operations_len,
                "total_time":total_time 
            })
            if failed_operations_len == 0:
                return Ok(0)
            else:
                return Err(pd.DataFrame(failed_operations))
    except Exception as e:
        LOGGER.error({
            "msg":str(e),
            "failed_operations":len(failed_operations),
        })
        return Err(pd.DataFrame(failed_operations))

if __name__ =="__main__":
    # 1. Traza.
    trace_df      = pd.read_csv(TRACE_PATH)
    n_trace_records = trace_df.shape[0]
    total_estimated_operations = n_trace_records * MAX_EXPERIMENT_ITERATIONS
    LOGGER.debug({
        "event":"DATAOWNER.STARTED",
        "client_id":NODE_ID,
        "algorithm":ALGORITHM,
        "max_threads":MAX_THREADS,
        "max_iterations":MAX_EXPERIMENT_ITERATIONS,
        "trace_path":TRACE_PATH,
        "n_trace_records": n_trace_records,
        "total_estimated_operations": total_estimated_operations,
        "sink_path":SINK_PATH,
        "source_path":SOURCE_PATH,
        "log_path":LOG_PATH
    })
    # 2. Experimentos (programa principal).
    result        = main(trace_df=trace_df,max_experiment_iterations=MAX_EXPERIMENT_ITERATIONS)
    # 3. Reintento de experimentos fallidos. 
    current_tries = 0
    start_time = time.time()
    while result.is_err and current_tries < MAX_RETRIES:
        failed_rows       = result.unwrap_err()
        sucess_percentage = ((trace_df.shape[0] - failed_rows.shape[0]) / trace_df.shape[0])*100
        error_percentage  = 100 - sucess_percentage
        LOGGER.error("{} completed with {} failed operations".format(EXPERIMENT_ID, failed_rows.shape[0]))
        LOGGER.debug({
            "event":"COMPLETED.WITH.ERRORS",
            "completed": EXPERIMENT_ID,
            "failed": failed_rows.shape[0],
            "sucess_percentage":sucess_percentage,
            "failed_percentage":error_percentage
        })
        current_tries += 1
        result =  main(
            trace_df             = failed_rows,
            max_experiment_iterations = 1
        )
    if result.is_ok:
        end_time = time.time()
        total_time = end_time - start_time
        LOGGER.debug({
            "event":"SUCCESSFULLY.COMPLETED",
            "response_time":total_time
        })
        sys.exit(0)
    else:
        LOGGER.error("MAX_RETRIES_REACHED")
        sys.exit(1)