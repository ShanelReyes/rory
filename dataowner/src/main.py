import os
import requests 
import pandas as pd
import logging
import time
import numpy as np
from rory.core.interfaces.client_response import ClientResponse
from rory.core.interfaces.logger_metrics import LoggerMetrics
from mictlanx.v3.client import Client as StorageClient
from mictlanx.v3.services.xolo import Xolo
from mictlanx.v3.services.proxy import Proxy
from mictlanx.v3.services.replica_manger import ReplicaManager
from rory.core.logger.Logger import create_logger
from option import Some
import numpy.typing as npt
from dotenv import load_dotenv
from retry.api import retry_call
from typing import Dict
from concurrent.futures import ThreadPoolExecutor,wait,ALL_COMPLETED

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
try:
    os.makedirs(SOURCE_PATH,exist_ok = True)
    os.makedirs(SINK_PATH,  exist_ok = True)
    os.makedirs(LOG_PATH,   exist_ok = True)
except Exception as e:
    print("MAKE_FOLDER_ERROR",e)

# MICTLANX
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
replica_manager = ReplicaManager(
    ip_addr     = MICTLANX_REPLICA_MANAGER_IP_ADDR, 
    port        = MICTLANX_REPLICA_MANAGER_PORT, 
    api_version = Some(MICTLANX_API_VERSION)
)
xolo            = Xolo(
    ip_addr     = MICTLANX_XOLO_IP_ADDR, 
    port        = MICTLANX_XOLO_PORT, 
    api_version = Some(MICTLANX_API_VERSION)
)
proxy           = Proxy(
    ip_addr     = MICTLANX_PROXY_IP_ADDR, 
    port        = MICTLANX_PROXY_PORT, 
    api_version = Some(MICTLANX_API_VERSION)
)
STORAGE_CLIENT  = StorageClient(
    app_id          = MICTLANX_APP_ID,
    client_id       = Some(MICTLANX_CLIENT_ID),
    secret          = MICTLANX_SECRET,
    replica_manager = replica_manager, 
    xolo            = xolo, 
    proxies         = [proxy], 
    expires_in      = Some(MICTLANX_EXPIRES_IN)
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
            url=url,
            headers=headers
        )
    except Exception as e:
        LOGGER.error("Error to process {} ".format(row["DATASET_ID"]))
        raise e

def run_experiment(row,experiment_iteration):
        arrivalTime       = time.time()
        plainTextMatrixId = str(row["DATASET_ID"])
        LOGGER.debug("INIT_EXPERIMENT {} {}".format(plainTextMatrixId,experiment_iteration))
        headers = {
            "Plaintext-Matrix-Id": plainTextMatrixId,
            "K": str(row["K"]),
            "M": str(row["M"]),
            "Threshold": str(row["THRESHOLD"]),
            "Extension": DATASET_EXTENSION,
            "Client-Id": CLIENT_ID,
            "Max-Iterations": str(row["MAX_ITERATIONS"])
        }
        url = "http://{}:{}/clustering/{}".format(CLIENT_IP_ADDR,CLIENT_PORT,ALGORITHM.lower())

        _response = retry_call(
            client_request,
            fkwargs={
                "row":row,
                "url":url,
                "headers": headers
            },
            tries=100,
            delay=1,
            max_delay=3,
            jitter=0.1
        )

        response = ClientResponse.fromResponse(_response)
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


try:
    EXPERIMENT_ID     = "{}_C{}".format(TASK_ID,CLIENT_INDEX)
    DATASET_EXTENSION = "csv"
    TRACE_PATH        = os.environ.get("TRACE_PATH","{}/{}.{}".format(SOURCE_PATH,TRACE_ID,DATASET_EXTENSION))
    trace_df          = pd.read_csv(TRACE_PATH)
    MAX_THREADS       = int(os.environ.get("MAX_THREADS",1))

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        for index,row in trace_df.iterrows():
            start_time = time.time()
            futures =[]
            for experiment_iteration in range(EXPERIMENT_ITERATION):
                fut = executor.submit(run_experiment,row,experiment_iteration)
                futures.append(fut)
                time.sleep(row["INTERARRIVAL_TIME"])
            
            wait(futures,None,ALL_COMPLETED )
            experiment_time = time.time() - start_time
            LOGGER.debug("{},{},{}".format(row["DATASET_ID"], EXPERIMENT_ITERATION,experiment_time))
            print("_"*50)

except Exception as e:
    print("ERROR",e)
finally:
    STORAGE_CLIENT.logout()