import sys
import os
# import requests 
import pandas as pd
import logging
import time
import numpy as np
from rory.core.interfaces.client_response import ClientResponse
# from rory.core.logger.Logger import create_logger
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
from roryclient.client import RoryClient
import time as T

RORY_CLIENT_HOSTNAME = os.environ.get("RORY_CLIENT_HOSTNAME","localhost")
RORY_CLIENT_PORT = int(os.environ.get("RORY_CLIENT_PORT","3001"))

client = RoryClient(hostname=RORY_CLIENT_HOSTNAME,port=RORY_CLIENT_PORT)

ENV_FILE_PATH = os.environ.get("ENV_FILE_PATH","/home/sreyes/rory/dataowner/envs/.env-kmeans") 
if os.path.exists(ENV_FILE_PATH):
    load_dotenv(ENV_FILE_PATH)

NODE_ID                   = os.environ.get("NODE_ID","rory-dataowner-0") 
# CLIENT_INDEX              = os.environ.get("CLIENT_INDEX","0")
# CLIENT_PORT               = os.environ.get("CLIENT_PORT",3000)
# CLIENT_IP_ADDR            = os.environ.get("CLIENT_IP_ADDR","localhost")
TASK_ID                   = os.environ.get("TASK_ID","CLUSTERING")
# ALGORITHM                 = os.environ.get("ALGORITHM","KMEANS")
# BATCH_ID                  = os.environ.get("BATCH_ID",0)
TRACE_ID                  = os.environ.get("TRACE_ID","KMEANS")
MAX_EXPERIMENT_ITERATIONS = int(os.environ.get("EXPERIMENT_ITERATION",31))
# BATCH_INDEX               = "batch_{}".format(BATCH_ID)
LOGGER_NAME               = NODE_ID
MAX_RETRIES               = int(os.environ.get("MAX_RETRIES","10"))
# EXPERIMENT_ID             = "{}_C{}".format(TASK_ID,CLIENT_INDEX)
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

def run_experiment(row:pd.Series,current_experiment_iteration:int)->Result[Tuple[pd.Series,R.Response,int],Tuple[pd.Series,R.Response, int]]:

    algorithm           = row["ALGORITHM"]
    TASK_ID             = os.environ.get("TASK_ID","CLUSTERING")
    plaintext_matrix_id = row.get("DATASET_ID","")
    model_id            = row.get("MODEL_ID","")
    label_vector_id     = "{}{}{}".format( plaintext_matrix_id if TASK_ID == "CLUSTERING" else model_id ,algorithm,current_experiment_iteration)

    if algorithm == "KMEANS":
        result = client.kmeans(
            plaintext_matrix_id       = plaintext_matrix_id,
            plaintext_matrix_filename = row["DATASET_FILENAME"],
            extension                 = row["EXTENSION"],
            k                         = row["K"],
            num_chunks                = row["NUM_CHUNKS"],   
        )
    elif algorithm == "SKMEANS":
        result = client.skmeans(
            plaintext_matrix_id       = plaintext_matrix_id,
            plaintext_matrix_filename = row["DATASET_FILENAME"],
            extension                 = row["EXTENSION"],
            k                         = row["K"],
            num_chunks                = row["NUM_CHUNKS"],
            max_iterations            = row["MAX_ITERATIONS"],
            experiment_iteration      = current_experiment_iteration,            
        )
    elif algorithm == "DBSKMEANS":
        result = client.dbskmeans(
            plaintext_matrix_id       = plaintext_matrix_id,
            plaintext_matrix_filename = row["DATASET_FILENAME"],
            extension                 = row["EXTENSION"],
            k                         = row["K"],
            num_chunks                = row["NUM_CHUNKS"],
            max_iterations            = row["MAX_ITERATIONS"],
            sens                      = row["SENS"],
            experiment_iteration      = current_experiment_iteration,            
        )
    elif algorithm == "SKMEANSPQC":
        result = client.skmeans_pqc(
            plaintext_matrix_id       = plaintext_matrix_id,
            plaintext_matrix_filename = row["DATASET_FILENAME"],
            extension                 = row["EXTENSION"],
            k                         = row["K"],
            num_chunks                = row["NUM_CHUNKS"],
            max_iterations            = row["MAX_ITERATIONS"],
            experiment_iteration      = current_experiment_iteration,            
        )
    elif algorithm == "DBSKMEANSPQC":
        result = client.dbskmeans_pqc(
            plaintext_matrix_id       = plaintext_matrix_id,
            plaintext_matrix_filename = row["DATASET_FILENAME"],
            extension                 = row["EXTENSION"],
            k                         = row["K"],
            num_chunks                = row["NUM_CHUNKS"],
            max_iterations            = row["MAX_ITERATIONS"],
            sens                      = row["SENS"],
            experiment_iteration      = current_experiment_iteration,            
        )
    elif algorithm == "NNC":
        result = client.nnc(
            plaintext_matrix_id       = plaintext_matrix_id,
            plaintext_matrix_filename = row["DATASET_FILENAME"],
            extension                 = row["EXTENSION"],
            threshold                 = row["THRESHOLD"]          
        )
    elif algorithm == "DBSNNC":
        result = client.dbsnnc(
            plaintext_matrix_id       = plaintext_matrix_id,
            plaintext_matrix_filename = row["DATASET_FILENAME"],
            extension                 = row["EXTENSION"],
            threshold                 = row["THRESHOLD"],
            num_chunks                = row["NUM_CHUNKS"],
            sens                      = row["SENS"],        
        )
    elif algorithm == "KNN":
        result = client.knn(
            model_id              = model_id,
            model_filename        = row["MODEL_FILENAME"],
            model_labels_filename = row["MODEL_LABELS_FILENAME"],
            record_test_id        = row["RECORD_TEST_ID"],
            record_test_filename  = row["RECORD_TEST_FILENAME"],
            extension             = row["EXTENSION"],
        )
    elif algorithm == "SKNN":
        result = client.sknn(
            model_id              = model_id,
            model_filename        = row["MODEL_FILENAME"],
            model_labels_filename = row["MODEL_LABELS_FILENAME"],
            record_test_id        = row["RECORD_TEST_ID"],
            record_test_filename  = row["RECORD_TEST_FILENAME"],
            num_chunks            = row["NUM_CHUNKS"],
            extension             = row["EXTENSION"],
        )
    elif algorithm == "SKNNPQC":
        result = client.sknn_pqc(
            model_id              = model_id,
            model_filename        = row["MODEL_FILENAME"],
            model_labels_filename = row["MODEL_LABELS_FILENAME"],
            record_test_id        = row["RECORD_TEST_ID"],
            record_test_filename  = row["RECORD_TEST_FILENAME"],
            num_chunks            = row["NUM_CHUNKS"],
            extension             = row["EXTENSION"],
        )
    else:
        print("UNKNOWN ALGORITM", algorithm)
        return Err((row, Exception("Unknown algorithm"), current_experiment_iteration))
    
    if result.is_err:
        return Err((row, result.unwrap_err(), current_experiment_iteration))
    response    = result.unwrap()
    labelVector = response.label_vector
    write_to_file(label_vector_id,labelVector)    
    return Ok((row, current_experiment_iteration))

def main(trace_df:pd.DataFrame,max_experiment_iterations:int= 31)->Result[int, pd.DataFrame]:
    failed_operations:List[pd.Series]  = [] # Lista de operaciones fallidas
    try:
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor: # Configuracion de la THREADPOOL. 
            start_time = time.time() # Tiempo de inicio de la experimentacion.
            for index,row in trace_df.iterrows(): # Iteracion de los registros de la traza.
                futures = [] # Lista de futuros (operaciones asincronas/no bloqueantes)
                
                for experiment_iteration in range(max_experiment_iterations): # Cada registro se repetira EXPERIMENT_ITERATION veces
                    fut = executor.submit(run_experiment,row,experiment_iteration) # Lanzar la operacion a un thread utilizando la Thread Pool. 
                    futures.append(fut) # Añadir a la lista de futuros el nuevo futuro.
                    time.sleep(float(row["INTERARRIVAL_TIME"])) # Duerme el thread para esperar INTERARRIVAL_TIME
                
                for fut in as_completed(futures): # Espera para completar todos los EXPERIMENT_ITERATIONS 
                    result:Result[Tuple[pd.Series, int], Tuple[pd.Series, Exception, int]] = fut.result() # Saca el resultado del futuro
                    
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
                        (_row, experiment_iteration) = result.unwrap() # Extrae la parte buena con unwrap()
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
        "max_threads":MAX_THREADS,
        "max_iterations":MAX_EXPERIMENT_ITERATIONS,
        "trace_path":TRACE_PATH,
        "n_trace_records": n_trace_records,
        "total_estimated_operations": total_estimated_operations,
        "sink_path":SINK_PATH,
        "source_path":SOURCE_PATH,
        "log_path":LOG_PATH,
        "client_timeout":CLIENT_TIMEOUT,
    })
    # 2. Experimentos (programa principal).
    result        = main(trace_df=trace_df,max_experiment_iterations=MAX_EXPERIMENT_ITERATIONS)
    # 3. Reintento de experimentos fallidos. 
    current_tries = 0
    start_time = time.time()
    while result.is_err and current_tries < MAX_RETRIES:
        failed_rows       = result.unwrap_err()
        failed_rows_n = failed_rows.shape[0]
        sucess_percentage = ((n_trace_records - failed_rows_n ) / n_trace_records )*100
        error_percentage  = 100 - sucess_percentage
        event_name = "COMPLETED.WITH.ERRORS"if failed_rows_n >0 else "COMPLETED.SUCCESSFULLY" 
        LOGGER.debug({
            "event":event_name,
            "completed": TASK_ID,
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