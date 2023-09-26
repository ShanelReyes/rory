import sys
import os
import requests 
import pandas as pd
import logging
import time
import numpy as np
from rory.core.interfaces.client_response import ClientResponse
from rory.core.interfaces.logger_metrics import LoggerMetrics
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

load_dotenv()

NODE_ID              = os.environ.get("NODE_ID","rory-dataowner-0") 
CLIENT_ID            = os.environ.get("CLIENT_ID","rory-client-0")
CLIENT_INDEX         = os.environ.get("CLIENT_INDEX","0")
CLIENT_PORT          = os.environ.get("CLIENT_PORT",3000)
CLIENT_IP_ADDR       = os.environ.get("CLIENT_IP_ADDR","localhost")
TASK_ID              = os.environ.get("TASK_ID","CLUSTERING")
ALGORITHM            = os.environ.get("ALGORITHM","KMEANS")
BATCH_ID             = os.environ.get("BATCH_ID",0)
TRACE_ID             = os.environ.get("TRACE_ID",ALGORITHM)
EXPERIMENT_ITERATION = int(os.environ.get("EXPERIMENT_ITERATION",31))
BATCH_INDEX          = "batch_{}".format(BATCH_ID)
LOGGER_NAME          = NODE_ID
SOURCE_PATH          = os.environ.get("SOURCE_PATH","/rory/source")
SINK_PATH            = os.environ.get("SINK_PATH","/rory/sink")
LOG_PATH             = os.environ.get("LOG_PATH","/rory/log")
MAX_RETRIES          = int(os.environ.get("MAX_RETRIES","10"))
EXPERIMENT_ID        = "{}_C{}".format(TASK_ID,CLIENT_INDEX)
MAX_THREADS          = int(os.environ.get("MAX_THREADS",1))
DATASET_EXTENSION    = "csv"
TRACE_PATH           = os.environ.get("TRACE_PATH","{}/{}.{}".format(SOURCE_PATH,TRACE_ID,DATASET_EXTENSION))

try:
    os.makedirs(SOURCE_PATH,exist_ok = True)
    os.makedirs(SINK_PATH,  exist_ok = True)
    os.makedirs(LOG_PATH,   exist_ok = True)
except Exception as e:
    print("MAKE_FOLDER_ERROR",e)

# MICTLANX
MICTLANX_TIMEOUT                 = int(os.environ.get("MICTLANX_TIMEOUT",120))
MICTLANX_APP_ID                  = os.environ.get("MICTLANX_APP_ID")
MICTLANX_CLIENT_ID               = os.environ.get("MICTLANX_CLIENT_ID")
MICTLANX_SECRET                  = os.environ.get("MICTLANX_SECRET")
MICTLANX_PROXY_IP_ADDR           = os.environ.get("MICTLANX_PROXY_IP_ADDR","localhost")
MICTLANX_PROXY_PORT              = int(os.environ.get("MICTLANX_PROXY_PORT","8080"))
MICTLANX_XOLO_IP_ADDR            = os.environ.get("MICTLANX_XOLO_IP_ADDR","localhost")
MICTLANX_XOLO_PORT               = int(os.environ.get("MICTLANX_XOLO_PORT","10000"))
MICTLANX_REPLICA_MANAGER_IP_ADDR = os.environ.get("MICTLANX_REPLICA_MANAGER_IP_ADDR","localhost")
MICTLANX_REPLICA_MANAGER_PORT    = int(os.environ.get("MICTLANX_REPLICA_MANAGER_PORT","20001"))
MICTLANX_API_VERSION             = int(os.environ.get("MICTLANX_API_VERSION","3"))
MICTLANX_EXPIRES_IN              = os.environ.get("MICTLANX_EXPIRES_IN","15d")

LOGGER = create_logger(
    name                   = LOGGER_NAME,
    LOG_FILENAME           = NODE_ID,
    LOG_PATH               = LOG_PATH,
    console_handler_filter = lambda record: record.levelno == logging.DEBUG or record.levelno == logging.INFO or record.levelno == logging.ERROR,
    file_handler_filter    = lambda record:  record.levelno == logging.INFO,
)

def write_to_file(filename:str, lv:npt.NDArray):
    path = "{}/{}.npy".format(SINK_PATH,filename)
    try:
        with open(path, "wb") as f:
            np.save(f,lv)
        
    except Exception as e:
        LOGGER.error(str(e))

def client_request(row:pd.Series,url:str,headers:Dict[str,str]):
    try:
        return requests.post(
            url     = url,
            headers = headers,
            timeout = 300
        )
    except Exception as e:
        print(e)
        LOGGER.error("Error to process {} ".format(row["DATASET_ID"]))
        raise e

def run_experiment(row:pd.Series,experiment_iteration:int)->Result[Tuple[pd.Series,R.Response,int],Tuple[pd.Series,R.Response, int]]:
        try:
            #return Err((row, R.Response(), -1))
            arrivalTime       = time.time()
            plainTextMatrixId = str(row["DATASET_ID"])
            LOGGER.debug("INIT_EXPERIMENT {} {}".format(plainTextMatrixId,experiment_iteration))
            headers = {
                "Plaintext-Matrix-Id": "{}-{}".format(plainTextMatrixId,experiment_iteration),
                "Plaintext-Matrix-Filename":plainTextMatrixId,
                "K": str(row["K"]),
                "M": str(row["M"]),
                "Threshold": str(row["THRESHOLD"]),
                "Extension": DATASET_EXTENSION,
                "Client-Id": CLIENT_ID,
                "Max-Iterations": str(row["MAX_ITERATIONS"]),
                "Experiment-Iteration": str(experiment_iteration)
            }

            url = "http://{}:{}/clustering/{}".format(CLIENT_IP_ADDR,CLIENT_PORT,ALGORITHM.lower())

            _response     = client_request(row=row,url=url,headers=headers)
            _response.raise_for_status()
            response      = ClientResponse.fromResponse(_response)
            labelVectorId = "{}_{}_{}".format(plainTextMatrixId,ALGORITHM,experiment_iteration)

            write_to_file(labelVectorId,response.labelVector)
            endTime       = time.time() # Get the time when it ends
            response_time = endTime - arrivalTime 

            logger_metrics = LoggerMetrics(
                operation_type = ALGORITHM,
                matrix_id      = plainTextMatrixId,
                algorithm      = ALGORITHM,
                arrival_time   = arrivalTime, 
                end_time       = endTime, 
                service_time   = response_time,
                n_iterations   = experiment_iteration,
            )
            LOGGER.info(str(logger_metrics))
            print("_"*40)
            return Ok((row,_response,experiment_iteration))
        except Exception as e:
            LOGGER.error("DATAOWNER_ERROR "+str(e))
            return Err((row,R.Response(),experiment_iteration))



def main(trace_df:pd.DataFrame,EXPERIMENT_ITERATION:int= 31)->Result[int, pd.DataFrame]:
    failed_operations:List[pd.Series]  = []
    try:
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            start_time = time.time()
            for index,row in trace_df.iterrows():
                futures    = []

                for experiment_iteration in range(EXPERIMENT_ITERATION):
                    
                    fut = executor.submit(run_experiment,row,experiment_iteration)

                    futures.append(fut)
                    time.sleep(row["INTERARRIVAL_TIME"])

                for fut in as_completed(futures):
                    result:Result[Tuple[pd.Series,R.Response, int], Tuple[pd.Series, R.Response, int]] = fut.result()
                    if result.is_err:
                        (failed_row, error_response, experiment_iteration) = result.unwrap_err()
                        datasetId = failed_row["DATASET_ID"]
                        LOGGER.error(str(error_response))
                        LOGGER.error("dataset_id={} iteration={} failed".format(datasetId,experiment_iteration))
                        print("_"*40)
                        failed_operations.append(failed_row)
                    else:
                        (_row, response, experiment_iteration) = result.unwrap()
                        datasetId = _row["DATASET_ID"]
                        LOGGER.debug("dataset_id={} iteration={} completed successfully".format(datasetId,experiment_iteration))
                        print("_"*40)
            end_time = time.time()
            total_time = end_time - start_time
            LOGGER.debug("Total time {}".format(total_time))
            if len(failed_operations)  == 0:
                return Ok(0)
            else:
                return Err(pd.DataFrame(failed_operations))
    except Exception as e:
        print("DATAOWNER_ERROR",e)
        return Err(pd.DataFrame(failed_operations))

if __name__ =="__main__":
    trace_df      = pd.read_csv(TRACE_PATH)
    result        = main(trace_df=trace_df,EXPERIMENT_ITERATION=EXPERIMENT_ITERATION)
    current_tries = 0
    while result.is_err and current_tries < MAX_RETRIES:
        failed_rows       = result.unwrap_err()
        print(trace_df.shape[0])
        sucess_percentage = ((trace_df.shape[0] - failed_rows.shape[0]) / trace_df.shape[0])*100
        error_percentage  = 100 - sucess_percentage
        LOGGER.error("{} completed with {} failed operations".format(EXPERIMENT_ID, failed_rows.shape[0]))
        LOGGER.debug("SUCESS_PERCENTAGE={} ERROR_PERCENTAGE={}".format(sucess_percentage,error_percentage))
        current_tries += 1
        result        =  main(trace_df=failed_rows,EXPERIMENT_ITERATION=1)
    if result.is_ok:
        LOGGER.debug("{} completed successfully".format(EXPERIMENT_ID))
        sys.exit(0)
    else:
        LOGGER.error("MAX_RETRIES_REACHED")
        sys.exit(1)

    # success_rate = r