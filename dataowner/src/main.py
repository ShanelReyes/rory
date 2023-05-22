import os
import requests 
import pandas as pd
import logging
import time, json
from logger.Logger import create_logger
from mictlanx.client import Client as StorageClient
from interfaces.dataowner_response import DataownerResponse


NODE_ID             = os.environ.get("NODE_ID","client-0") 
LOG_PATH            = os.environ.get("LOG_PATH","/log")
CLIENT_ID           = "client-0"
SOURCE_PATH         = "/source/"
DATA_OWNER_BASE_URL = "localhost"
DATA_OWNER_PORT     = 3000
DATASET_EXTENSION   = "csv"
STORAGE_HOSTNAME    = os.environ.get("STORAGE_SYSTEM_HOSTNAME","localhost")
STORAGE_PORT        = int(os.environ.get("STORAGE_SYSTEM_PORT",6001))
STORAGE_CLIENT      = StorageClient(hostname = STORAGE_HOSTNAME,port = STORAGE_PORT, LOG_PATH = LOG_PATH) 
LOGGER              = create_logger(
    name                  = NODE_ID,
    LOG_FILENAME           = NODE_ID,
    LOG_PATH               = LOG_PATH,
    console_handler_filter = lambda record: record.levelno == logging.INFO or record.levelno == logging.ERROR or record.levelno == logging.DEBUG,
    file_handler_filter    = lambda record: record.levelno == logging.DEBUG or record.levelno == logging.INFO,
)

trace_id     = "trace_threshold"
#trace_id = "trace_mictlanx"
#trace_id = "trace_kmeans"
trace_path   = "{}{}.{}".format(SOURCE_PATH,trace_id,DATASET_EXTENSION)
trace        = pd.read_csv(trace_path)
generate_url = lambda algorithm: "http://{}:{}/clustering/{}".format(DATA_OWNER_BASE_URL,DATA_OWNER_PORT,algorithm)
algorithms   = ["KMEANS","SKMEANS","DBSKMEANS","DBSNNC"]
identity     = lambda x:x

def dataownerClustering():
    for row_index,row in trace.iterrows():
        arrivalTime = time.time()
        isKmeans    = row["KMEANS"]    == 1
        isSkmeans   = row["SKMEANS"]   == 1
        isDbskmeans = row["DBSKMEANS"] == 1
        isDbsnnc    = row["DBSNNC"]    == 1
        unfiltered_xs = [isKmeans, isSkmeans, isDbskmeans, isDbsnnc]
        unfiltered_xs = list(zip(list(range(0,len(algorithms))),unfiltered_xs))
        
        xs = list(filter(lambda x:x[1], unfiltered_xs)) #if x:
        for index,x in xs:
            algorithm:str     = algorithms[index]
            url               = generate_url(algorithm.lower())
            plaintextMatrixId = row["PLAINTEXT_MATRIX_ID"]
            k                 = row["K"]
            headers = {
                "Plaintext-Matrix-Id":plaintextMatrixId,
                "K":str(k),
                "M":str(row["M"]),
                "Threshold":str(row["THRESHOLD"]),
                "Extension":DATASET_EXTENSION,
                "Client-Id":CLIENT_ID,
                "Max-Iterations": str(row["MAX_ITERATIONS"])
            }

            response = requests.post(
                url    = url,
                data   = None, 
                headers= headers
            )

            response      = DataownerResponse.fromResponse(response)
            labelVectorId = "{}_{}_k{}".format(plaintextMatrixId,algorithm,k)
            _             = STORAGE_CLIENT.put_ndarray(id = labelVectorId, matrix = response.labelVector)
            endTime       = time.time() # Get the time when it ends
            service_time  = response.headers.get("Service-Time",0) 
            response_time = endTime - arrivalTime 

            LOGGER.info("{} {} {} {} {}".format(#Show the final result in a logger
                algorithm.upper(),
                plaintextMatrixId,
                k,
                service_time,
                response_time
            ))

def dataOwnerMetrics():
    for row_index,row in trace.iterrows():
        arrivalTime = time.time()
        algorithm = "metrics"
        #algorithm = "dbsnnc"
        url               = generate_url(algorithm)
        plaintextMatrixId = row["PLAINTEXT_MATRIX_ID"]
        k                 = row["K"]
        headers = {
            "Plaintext-Matrix-Id":plaintextMatrixId,
            "K":str(k),
            "Extension":DATASET_EXTENSION,
            "Algorithm": "DBSNNC"
        }
        response = requests.get(
            url    = url,
            data   = None, 
            headers= headers
        )
        LOGGER.debug("RESPONSE_STATUS {}".format(response.status_code))
        endTime       = time.time() # Get the time when it ends
        service_time  = response.headers.get("Service-Time",0) 
        response_time = endTime - arrivalTime 

        LOGGER.info("{} {} {} {} {}".format(#Show the final result in a logger
            algorithm.upper(),
            plaintextMatrixId,
            k,
            service_time,
            response_time
        ))


if __name__ == "__main__":
    #dataownerClustering()
    dataOwnerMetrics()
