import pandas as pd
import numpy as np
import time
import requests
import json
from concurrent.futures import ThreadPoolExecutor
from rory.security.cryptosystem.dataowner import DataOwner
from rory.utils.Utils import Utils
from rory.utils.constants import Constants
from rory.interfaces.clusteringrequest import ClusteringRequestClient
from rory.interfaces.secureclusteringworker import SecureClusteringWorker,DumbClusteringWorker
from storage.client import Client as StorageClient
import hashlib as H

"""
Description: 
    Function that allows to maintain a session between the client and the worker. 
    Includes dataset encryption process, UDM calculation, decryption of matrix S'
"""
def processSkMeans(*args,**kwargs):
    NODE_ID                = kwargs.get("NODE_ID")
    SINK_PATH              = kwargs.get("SINK_PATH","/sink")
    SOURCE_PATH            = kwargs.get("SOURCE_PATH","/source")
    DATASET_EXTENSION      = kwargs.get("DATASET_EXTENSION","csv")
    DATASET_FOLDER         = kwargs.get("DATASET_FOLDER")
    MAX_ITERATIONS         = kwargs.get("MAX_ITERATIONS",10)
    ciphertextMatrixId     = kwargs.get("ciphertextMatrixId")
    encryptedShiftMatrixId = "{}-EncryptedShiftMatrix".format(ciphertextMatrixId)
    record_index           = kwargs.get("record_index")
    worker                 = kwargs.get("worker")
    logger                 = kwargs.get("logger")
    session                = kwargs.get("session")
    record                 = kwargs.get("record")
    liu                    = kwargs.get("liu")
    STORAGE_CLIENT         = kwargs.get("STORAGE_CLIENT")
    try:
        startTime               = time.time()
        workerId                = worker.workerId
        m                       = record["M"]
        k                       = record["K"]
        datasetId               = record["DATASET_ID"]
        algorithm               = record["ALGORITHM"]
        datasetPath             = "{}/{}/{}.{}".format(SOURCE_PATH,DATASET_FOLDER,datasetId,DATASET_EXTENSION)
        UDMId                   = "{}-UDM".format(ciphertextMatrixId)
        df                      = pd.read_csv(datasetPath,header=None)
        plaintext_matrix_values = df.values
        plaintext_matrix        = plaintext_matrix_values.tolist()
        dow                     = DataOwner(
            m          = m,
            liu_scheme = liu,
        )
        secret_key            = dow.sk
        outsourced_data_stats = dow.outsourcedData(
            plaintext_matrix  = np.array(plaintext_matrix),
            secret_key        = dow.sk,
            m                 = m,
            algorithm         = algorithm
        )
        ciphertext_matrix = outsourced_data_stats.encrypted_matrix
        UDM               = np.array(outsourced_data_stats.UDM)
        logger.info("ENCRYPTION_TIME,{},{},{},{},{},{}".format(
            NODE_ID,datasetId,workerId,"SKMEANS",0,outsourced_data_stats.encrypted_matrix_time)
        )
        logger.info("UDM_TIME,{},{},{},{},{},{}".format(
            NODE_ID,datasetId,workerId,"SKMEANS",0,outsourced_data_stats.udm_time)
        )
        _ = STORAGE_CLIENT.put_matrix(id=ciphertextMatrixId,matrix =ciphertext_matrix)
        _ = STORAGE_CLIENT.put_matrix(id=UDMId,matrix =UDM)
        iterations_counter = 0 
        isCompleted        = False
        requestId          = "request-{}".format(record_index)
        while not isCompleted:
            ClusteringStatus = Constants.ClusteringStatus.START if(iterations_counter == 0 ) else Constants.ClusteringStatus.WORK_IN_PROGRESS
            response         = worker.SKMeans_1(  
                K                = str(k),
                M                = str(m),
                CipherMatrixId   = ciphertextMatrixId,
                UDMId            = UDMId,
                ClusteringStatus = str(ClusteringStatus),
                RequestId        = requestId,
                ClientId         = NODE_ID,
                DatasetId        = datasetId
            )
            responseHeaders       = response.headers
            encrypted_shift_matrix_metadata,encryptedShifMatrix =  STORAGE_CLIENT.get_matrix(id = encryptedShiftMatrixId,sink_path = SINK_PATH)
            shiftMatrix           = liu.decryptMatrix(
                ciphertext_matrix = encryptedShifMatrix.tolist(),
                secret_key        = secret_key,
                m                 = m
            )
            shiftMatrixId         = "{}-ShiftMatrix".format(ciphertextMatrixId)
            shift_matrix_storage_response = STORAGE_CLIENT.put_matrix(id=shiftMatrixId,matrix =np.array(shiftMatrix))
            responseRun2   = worker.SKMeans_2(
                ClusteringStatus = str(Constants.ClusteringStatus.WORK_IN_PROGRESS ),
                ShiftMatrixId    = shiftMatrixId,
            )
            print(responseRun2)
            responseRun2Headers = responseRun2.headers
            status              = int(responseRun2Headers.get("ClusteringStatus",Constants.ClusteringStatus.WORK_IN_PROGRESS ))
            isCompleted         = (status == Constants.ClusteringStatus.COMPLETED) or (iterations_counter == MAX_ITERATIONS)
            iterations_counter += 1 
        session.close() # Terminar la session entre cliente y trabajador.
        responseTime        = time.time() - startTime
        return {"Iterations": iterations_counter}
    except Exception as e:
        logger.error("{} {} {}".format(datasetId,workerId,e))
        raise e


def processDbskmeans(*args,**kwargs):
    NODE_ID                = kwargs.get("NODE_ID")
    SINK_PATH              = kwargs.get("SINK_PATH","/sink")
    SOURCE_PATH            = kwargs.get("SOURCE_PATH","/source")
    DATASET_EXTENSION      = kwargs.get("DATASET_EXTENSION","csv")
    DATASET_FOLDER         = kwargs.get("DATASET_FOLDER")
    MAX_ITERATIONS         = kwargs.get("MAX_ITERATIONS",10)
    ciphertextMatrixId     = kwargs.get("ciphertextMatrixId")
    encryptedShiftMatrixId = "{}-EncryptedShiftMatrix".format(ciphertextMatrixId)
    record_index           = kwargs.get("record_index")
    worker                 = kwargs.get("worker")
    logger                 = kwargs.get("logger")
    session                = kwargs.get("session")
    record                 = kwargs.get("record")
    liu                    = kwargs.get("liu")
    STORAGE_CLIENT         = kwargs.get("STORAGE_CLIENT")
    try:
        startTime               = time.time()
        workerId                = worker.workerId
        m                       = record["M"]
        k                       = record["K"]
        datasetId               = record["DATASET_ID"]
        algorithm               = record["ALGORITHM"]
        datasetPath             = "{}/{}/{}.{}".format(SOURCE_PATH,DATASET_FOLDER,datasetId,DATASET_EXTENSION)
        UDMId                   = "{}-UDM".format(ciphertextMatrixId)
        df                      = pd.read_csv(datasetPath,header=None)
        plaintext_matrix_values = df.values
        plaintext_matrix        = plaintext_matrix_values.tolist()
        dow                     = DataOwner(
            m          = m,
            liu_scheme = liu,
        )
        secret_key            = dow.sk
        outsourced_data_stats = dow.outsourcedData(
            plaintext_matrix  = np.array(plaintext_matrix),
            secret_key        = dow.sk,
            m                 = m,
            algorithm         = algorithm
        )
        ciphertext_matrix = outsourced_data_stats.encrypted_matrix
        UDM               = np.array(outsourced_data_stats.UDM)
        logger.info("ENCRYPTION_TIME,{},{},{},{},{},{}".format(
            NODE_ID,datasetId,workerId,"SKMEANS",0,outsourced_data_stats.encrypted_matrix_time)
        )
        logger.info("UDM_TIME,{},{},{},{},{},{}".format(
            NODE_ID,datasetId,workerId,"SKMEANS",0,outsourced_data_stats.udm_time)
        )
        _ = STORAGE_CLIENT.put_matrix(id=ciphertextMatrixId,matrix =ciphertext_matrix)
        _ = STORAGE_CLIENT.put_matrix(id=UDMId,matrix =UDM)
        iterations_counter = 0
        isCompleted        = False
        requestId          = "request-{}".format(record_index)
        while not isCompleted:
            ClusteringStatus = Constants.ClusteringStatus.START if(iterations_counter == 0 ) else Constants.ClusteringStatus.WORK_IN_PROGRESS
            response         = worker.Dbskmeans_1(  
                K                = str(k),
                M                = str(m),
                CipherMatrixId   = ciphertextMatrixId,
                UDMId            = UDMId,
                ClusteringStatus = str(ClusteringStatus),
                RequestId        = requestId,
                ClientId         = NODE_ID,
                DatasetId        = datasetId
            )
            responseHeaders       = response.headers
            encrypted_shift_matrix_metadata,encryptedShifMatrix =  STORAGE_CLIENT.get_matrix(id = encryptedShiftMatrixId,sink_path = SINK_PATH)
            shiftMatrix           = liu.decryptMatrix(
                ciphertext_matrix = encryptedShifMatrix.tolist(),
                secret_key        = secret_key,
                m                 = m
            )
            shiftMatrixId         = "{}-ShiftMatrix".format(ciphertextMatrixId)
            shift_matrix_storage_response = STORAGE_CLIENT.put_matrix(id=shiftMatrixId,matrix =np.array(shiftMatrix))
            responseRun2   = worker.Dbskmeans_1(
                ClusteringStatus = str(Constants.ClusteringStatus.WORK_IN_PROGRESS ),
                ShiftMatrixId    = shiftMatrixId,
            )
            print(responseRun2)
            responseRun2Headers = responseRun2.headers
            status              = int(responseRun2Headers.get("ClusteringStatus",Constants.ClusteringStatus.WORK_IN_PROGRESS ))
            isCompleted         = (status == Constants.ClusteringStatus.COMPLETED) or (iterations_counter == MAX_ITERATIONS)
            iterations_counter += 1 
        session.close() # Terminar la session entre cliente y trabajador.
        responseTime        = time.time() - startTime
        return {"Iterations": iterations_counter}
    except Exception as e:
        logger.error("{} {} {}".format(datasetId,workerId,e))
        raise e

def processDbsnnc(*args,**kwargs):
    NODE_ID                = kwargs.get("NODE_ID")
    SINK_PATH              = kwargs.get("SINK_PATH","/sink")
    SOURCE_PATH            = kwargs.get("SOURCE_PATH","/source")
    DATASET_EXTENSION      = kwargs.get("DATASET_EXTENSION","csv")
    DATASET_FOLDER         = kwargs.get("DATASET_FOLDER")
    MAX_ITERATIONS         = kwargs.get("MAX_ITERATIONS",10)
    ciphertextMatrixId     = kwargs.get("ciphertextMatrixId")
    encryptedShiftMatrixId = "{}-EncryptedShiftMatrix".format(ciphertextMatrixId)
    record_index           = kwargs.get("record_index")
    worker                 = kwargs.get("worker")
    logger                 = kwargs.get("logger")
    session                = kwargs.get("session")
    record                 = kwargs.get("record")
    liu                    = kwargs.get("liu")
    STORAGE_CLIENT         = kwargs.get("STORAGE_CLIENT")
    try:
        startTime               = time.time()
        workerId                = worker.workerId
        m                       = record["M"]
        k                       = record["K"]
        datasetId               = record["DATASET_ID"]
        algorithm               = record["ALGORITHM"]
        datasetPath             = "{}/{}/{}.{}".format(SOURCE_PATH,DATASET_FOLDER,datasetId,DATASET_EXTENSION)
        UDMId                   = "{}-UDM".format(ciphertextMatrixId)
        df                      = pd.read_csv(datasetPath,header=None)
        plaintext_matrix_values = df.values
        plaintext_matrix        = plaintext_matrix_values.tolist()
        dow                     = DataOwner(
            m          = m,
            liu_scheme = liu,
        )
        secret_key            = dow.sk
        outsourced_data_stats = dow.outsourcedData(
            plaintext_matrix  = np.array(plaintext_matrix),
            secret_key        = dow.sk,
            m                 = m,
            algorithm         = algorithm
        )
        ciphertext_matrix = outsourced_data_stats.encrypted_matrix
        UDM               = np.array(outsourced_data_stats.UDM)
        logger.info("ENCRYPTION_TIME,{},{},{},{},{},{}".format(
            NODE_ID,datasetId,workerId,"SKMEANS",0,outsourced_data_stats.encrypted_matrix_time)
        )
        logger.info("UDM_TIME,{},{},{},{},{},{}".format(
            NODE_ID,datasetId,workerId,"SKMEANS",0,outsourced_data_stats.udm_time)
        )
        _ = STORAGE_CLIENT.put_matrix(id=ciphertextMatrixId,matrix =ciphertext_matrix)
        _ = STORAGE_CLIENT.put_matrix(id=UDMId,matrix =UDM)
        iterations_counter = 0
        isCompleted        = False
        requestId          = "request-{}".format(record_index)
        while not isCompleted:
            ClusteringStatus = Constants.ClusteringStatus.START if(iterations_counter == 0 ) else Constants.ClusteringStatus.WORK_IN_PROGRESS
            response         = worker.Dbskmeans_1(  
                K                = str(k),
                M                = str(m),
                CipherMatrixId   = ciphertextMatrixId,
                UDMId            = UDMId,
                ClusteringStatus = str(ClusteringStatus),
                RequestId        = requestId,
                ClientId         = NODE_ID,
                DatasetId        = datasetId
            )
            responseHeaders       = response.headers
            encrypted_shift_matrix_metadata,encryptedShifMatrix =  STORAGE_CLIENT.get_matrix(id = encryptedShiftMatrixId,sink_path = SINK_PATH)
            shiftMatrix           = liu.decryptMatrix(
                ciphertext_matrix = encryptedShifMatrix.tolist(),
                secret_key        = secret_key,
                m                 = m
            )
            shiftMatrixId         = "{}-ShiftMatrix".format(ciphertextMatrixId)
            shift_matrix_storage_response = STORAGE_CLIENT.put_matrix(id=shiftMatrixId,matrix =np.array(shiftMatrix))
            responseRun2   = worker.Dbskmeans_1(
                ClusteringStatus = str(Constants.ClusteringStatus.WORK_IN_PROGRESS ),
                ShiftMatrixId    = shiftMatrixId,
            )
            print(responseRun2)
            responseRun2Headers = responseRun2.headers
            status              = int(responseRun2Headers.get("ClusteringStatus",Constants.ClusteringStatus.WORK_IN_PROGRESS ))
            isCompleted         = (status == Constants.ClusteringStatus.COMPLETED) or (iterations_counter == MAX_ITERATIONS)
            iterations_counter += 1 
        session.close() # Terminar la session entre cliente y trabajador.
        responseTime        = time.time() - startTime
        return {"Iterations": iterations_counter}
    except Exception as e:
        logger.error("{} {} {}".format(datasetId,workerId,e))
        raise e


"""
Description:
    Function that reads the received trace
"""
def process_trace(*args):
    TRACE_PATH              = args[0]
    MAX_WORKERS             = args[1]
    MANAGER                 = args[2]
    logger                  = args[3]
    liu                     = args[4]
    SOURCE_PATH             = args[5]
    SINK_PATH               = args[6]
    DATASET_EXTENSION       = args[7]
    NODE_ID                 = args[8]
    NODE_PORT               = args[9]
    MAX_ITERATIONS          = args[10]
    DATASET_FOLDER          = args[11]
    STORAGE_SYSTEM_HOSTNAME = args[12]
    STORAGE_SYSTEM_PORT     = args[13]
    traces                  = pd.read_csv(TRACE_PATH)
    
    """
    Description:
        Allows to establish the session with the worker
    """
    def make_request(*args):
        try:
            record_index,record = args[0]
            STORAGE_CLIENT      = StorageClient(
                hostname = STORAGE_SYSTEM_HOSTNAME, 
                port     = STORAGE_SYSTEM_PORT
            )            
            # GET M, K and Algorithm type from record
            dataset_id = record.get("DATASET_ID","cm-{}".format(record_index))
            dataset_df = pd.read_csv("{}/{}/{}.csv".format(SOURCE_PATH,DATASET_FOLDER,dataset_id))
            STORAGE_CLIENT.put_matrix(id = dataset_id,matrix = dataset_df.values)
            target_id  = "{}_target".format(dataset_id) # UPLOAD TARGET AND COUNTER 
            target_df  = pd.read_csv("{}/{}/{}.csv".format(SOURCE_PATH,DATASET_FOLDER,target_id))
            STORAGE_CLIENT.put_matrix(id = target_id,matrix = target_df.values,)
            counter_id = "{}_counter".format(dataset_id)
            counter_df = pd.read_csv("{}/{}/{}.csv".format(SOURCE_PATH,DATASET_FOLDER,counter_id))
            STORAGE_CLIENT.put_matrix(id = counter_id,matrix = counter_df.values,)
            # SAVE DATASET IN THE STORAGE SYSTEM
            m          = record["M"]
            k          = record["K"]
            datasetId  = record["DATASET_ID"]
            batch_id   = record["BATCH"]
            algorithm  = record["ALGORITHM"]
            logger.debug("ALGORITHM {} {}".format(datasetId,algorithm) )
            data       = ClusteringRequestClient( 
                encryptedDatasetId = dataset_id,
                m                  = m,
                k                  = k,
                algorithm          = algorithm
            )
            managerStartTime    = time.time() # Manager response a workerId
            response            = MANAGER.sendSecureClusteringRequest(data = data)
            session             = requests.Session() # Generate a session to maintain the state between client and worker
            responseJson        = json.loads(response.text) # Deserialize the manager's response
            workerId            = responseJson["workerId"] # Get the workerId from the data sent by the manager
            managerResponseTime = time.time() - managerStartTime # LOAD BALANCING TIME.
            worker              = SecureClusteringWorker(workerId = workerId, session = session,algorithm = algorithm) # Generate an object with the data sent by the manager and the session
            startTime           = time.time()
            response            = None
            if(Constants.ClusteringAlgorithms.SKMEANS == algorithm):
                response = processSkMeans(
                    NODE_ID            = NODE_ID,
                    NODE_PORT          = NODE_PORT,
                    SINK_PATH          = SINK_PATH,
                    SOURCE_PATH        = SOURCE_PATH,
                    DATASET_EXTENSION  = DATASET_EXTENSION,
                    ciphertextMatrixId = dataset_id+"_cipher",
                    record_index       = record_index,
                    worker             = worker,
                    logger             = logger,
                    session            = session,
                    record             = record,
                    liu                = liu,
                    MAX_ITERATIONS     = MAX_ITERATIONS,
                    DATASET_FOLDER     = DATASET_FOLDER ,
                    STORAGE_CLIENT     = STORAGE_CLIENT,
                )
            elif(Constants.ClusteringAlgorithms.KMEANS == algorithm) :
                platintextMatrixPath = "{}/{}/{}.{}".format(SOURCE_PATH,DATASET_FOLDER,record["DATASET_ID"],DATASET_EXTENSION)
                df               = pd.read_csv(platintextMatrixPath) # SAVE DATASET IN STORAGE SYSTEM
                storage_response = STORAGE_CLIENT.put_matrix(id=record["DATASET_ID"],matrix = df.values)
                logger.debug("{}".format(storage_response)) 
                k                = record["K"]
                response         = worker.kmeans(
                    K         = str(k),
                    DatasetId = datasetId
                )
                print(response)
                response = response.headers
            elif(Constants.ClusteringAlgorithms.DBSKMEANS == algorithm) :
                response = processDbskmeans(
                    NODE_ID            = NODE_ID,
                    NODE_PORT          = NODE_PORT,
                    SINK_PATH          = SINK_PATH,
                    SOURCE_PATH        = SOURCE_PATH,
                    DATASET_EXTENSION  = DATASET_EXTENSION,
                    ciphertextMatrixId = dataset_id+"_cipher",
                    record_index       = record_index,
                    worker             = worker,
                    logger             = logger,
                    session            = session,
                    record             = record,
                    liu                = liu,
                    MAX_ITERATIONS     = MAX_ITERATIONS,
                    DATASET_FOLDER     = DATASET_FOLDER ,
                    STORAGE_CLIENT     = STORAGE_CLIENT,
                )
            elif(Constants.ClusteringAlgorithms.DBSNNC == algorithm):
                response = processDbsnnc(
                    NODE_ID            = NODE_ID,
                    NODE_PORT          = NODE_PORT,
                    SINK_PATH          = SINK_PATH,
                    SOURCE_PATH        = SOURCE_PATH,
                    DATASET_EXTENSION  = DATASET_EXTENSION,
                    ciphertextMatrixId = dataset_id+"_cipher",
                    record_index       = record_index,
                    worker             = worker,
                    logger             = logger,
                    session            = session,
                    record             = record,
                    liu                = liu,
                    MAX_ITERATIONS     = MAX_ITERATIONS,
                    DATASET_FOLDER     = DATASET_FOLDER ,
                    STORAGE_CLIENT     = STORAGE_CLIENT,
                )
            else: 
                logger.error("NO_IMPLEMENTED_ALGORITHMS")
            responseTime = time.time() - startTime
            iterations   = response.get("Iterations",0)
            logger.info("CLUSTERING,{},{},{},{},{},{}".format(NODE_ID,datasetId,workerId,algorithm,iterations,responseTime))
        except Exception as e:
            print(e)
            logger.error(str(e))
    rows = list(traces.iterrows())
    print("ROWS = {}".format(len(rows)))
    with ThreadPoolExecutor(max_workers = MAX_WORKERS ) as executor:
            executor.map(make_request,rows)