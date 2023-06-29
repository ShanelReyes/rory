import os
import requests 
import pandas as pd
import logging
import time
from rory.core.interfaces.client_response import ClientResponse
from rory.core.interfaces.logger_metrics import LoggerMetrics
from mictlanx.v3.client import Client as StorageClient
from mictlanx.v3.services.xolo import Xolo
from mictlanx.v3.services.proxy import Proxy
from mictlanx.v3.services.replica_manger import ReplicaManager
from rory.core.logger.Logger import create_logger
from option import Some
from dotenv import load_dotenv
load_dotenv()

NODE_ID        = os.environ.get("NODE_ID","rory-dataowner-0") 
CLIENT_ID      = os.environ.get("CLIENT_ID","rory-client-0")
CLIENT_INDEX   = os.environ.get("CLIENT_INDEX","0")
CLIENT_PORT    = os.environ.get("CLIENT_PORT",3000)
CLIENT_IP_ADDR = os.environ.get("CLIENT_IP_ADDR","localhost")

TASK_ID      = os.environ.get("TASK_ID","CLUSTERING")
ALGORITHM    = os.environ.get("ALGORITHM","KMEANS")
BATCH_ID     = os.environ.get("BATCH_ID",0)
TRACE_ID     = os.environ.get("TRACE_ID",ALGORITHM)
BATCH_INDEX  = "batch_{}".format(BATCH_ID)
LOGGER_NAME  = NODE_ID

SOURCE_PATH      = os.environ.get("SOURCE_PATH","/rory/source")
SINK_PATH        = os.environ.get("SINK_PATH","/rory/sink")
LOG_PATH         = os.environ.get("LOG_PATH","/rory/log")
try:
    os.makedirs(SOURCE_PATH,exist_ok = True)
    os.makedirs(SINK_PATH,  exist_ok = True)
    os.makedirs(LOG_PATH,   exist_ok = True)
except Exception as e:
    print("MAKE_FOLDER_ERROR",e)

# SINK_FOLDER   = "/rory/{}/sink".format(NODE_ID)
# SOURCE_FOLDER = "/rory/{}/source".format(NODE_ID)
# LOG_FOLDER    = "/rory/{}/log".format(NODE_ID)
# os.makedirs(SINK_FOLDER,  exist_ok = True)
# os.makedirs(SOURCE_FOLDER,exist_ok = True)
# os.makedirs(LOG_FOLDER,   exist_ok = True)

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

try:
    EXPERIMENT_ID     = "{}_C{}".format(TASK_ID,CLIENT_INDEX)
    DATASET_EXTENSION = "csv"
    TRACE_PATH        = os.environ.get("TRACE_PATH","{}/{}.{}".format(SOURCE_PATH,TRACE_ID,DATASET_EXTENSION))
    trace_df          = pd.read_csv(TRACE_PATH)
    for index,row in trace_df.iterrows():
        arrivalTime       = time.time()
        plainTextMatrixId = str(row["DATASET_ID"])
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
        
        _response = requests.post(
            url    = url,
            data   = None, 
            headers= headers
        )
        #print("RESPONSE",_response)
        response = ClientResponse.fromResponse(_response)
        labelVectorId = "{}_{}".format(plainTextMatrixId,ALGORITHM)
        _ = STORAGE_CLIENT.put_ndarray(
                key     = labelVectorId,
                ndarray = response.labelVector,
                update  = True
        ).unwrap()
        #print(response.labelVector)
        endTime       = time.time() # Get the time when it ends
        # service_time  = response.headers.get("Service-Time",0) 
        response_time = endTime - arrivalTime 

        logger_metrics = LoggerMetrics(
            operation_type = ALGORITHM,
            matrix_id      = plainTextMatrixId,
            algorithm      = ALGORITHM,
            arrival_time   = arrivalTime, 
            end_time       = endTime, 
            service_time   = response_time
        )
        LOGGER.info(str(logger_metrics))
except Exception as e:
    print("ERROR",e)
finally:
    STORAGE_CLIENT.logout()