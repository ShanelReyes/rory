import os
import time
from option import Option,NONE,Some
from mictlanx.v3.interfaces.payloads import SummonContainerPayload,ExposedPort,AuthTokenPayload
from mictlanx.v3.services.xolo import Xolo
from mictlanx.v3.services.summoner import Summoner

def deploy_nodes(
        summoner:Summoner,
        NODE_ID:str,
        PORT:str,
        DOCKER_IMAGE:str, 
        DOCKER_NETWORK_ID:str,
        MICTLANX_APP_ID:str ,
        MICTLANX_CLIENT_ID:str,
        MICTLANX_SECRET:str,
        MICTLANX_PROXY_IP_ADDR:str,
        MICTLANX_PROXY_PORT:int,
        MICTLANX_XOLO_IP_ADDR:str,
        MICTLANX_XOLO_PORT:int,
        MICTLANX_REPLICA_MANAGER_IP_ADDR:str,
        MICTLANX_REPLICA_MANAGER_PORT:int,
        xolo:Xolo,
        init_workers:int = 1,
        MICTLANX_API_VERSION:Option[int] = Some(3),
        MICTLANX_EXPIRES_IN:Option[str] = Some("15d"),
        NODE_PREFIX:str="scw-",
        init_port:int = 3000,
):
    auth_result = xolo.auth(payload= AuthTokenPayload(app_id=MICTLANX_APP_ID, client_id= MICTLANX_CLIENT_ID, secret= MICTLANX_SECRET,expires_in= Some(MICTLANX_EXPIRES_IN)))
    if(auth_result.is_err):
        raise auth_result.unwrap_err()
    auth_response = auth_result.unwrap()
    for i in range(init_workers): 
        container_id = "{}{}".format(NODE_PREFIX,i)
        payload = SummonContainerPayload(
            image=DOCKER_IMAGE, 
            container_id= container_id,
            hostname=container_id,
            exposed_ports=[
                ExposedPort(ip_addr=NONE,host_port=init_port+i,container_port=init_port,protocolo=NONE)
            ],
            envs={
                "NODE_IP_ADDR":container_id,
                # "10.0.0.{}".format(30+i),
                "NODE_PORT":str(init_port),
                "SECURE_CLUSTERING_MANAGER_IP_ADDR":NODE_ID,
                "SECURE_CLUSTERING_MANAGER_PORT":str(PORT),
                "DEBUG":"0",
                "RELOAD":"0",
                "LIU_ROUND":"2",
                "SINK_PATH":"/sink", 
                "SOURCE_PATH":"/source",
                "MAX_ITERATIONS":"10",
                "LOG_PATH":"/log",
                "TESTING":"0",
                "M":"3",
                "MICTLANX_APP_ID":MICTLANX_APP_ID, 
                "MICTLANX_CLIENT_ID":MICTLANX_CLIENT_ID,
                "MICTLANX_SECRET":MICTLANX_SECRET,
                "MICTLANX_PROXY_IP_ADDR": MICTLANX_PROXY_IP_ADDR,
                "MICTLANX_PROXY_IP_PORT": MICTLANX_PROXY_PORT,
                
                "MICTLANX_XOLO_IP_ADDR":str(MICTLANX_XOLO_IP_ADDR),
                "MICTLANX_XOLO_PORT":str(MICTLANX_XOLO_PORT),
                
                "MICTLANX_REPLICA_MANAGER_IP_ADDR":MICTLANX_REPLICA_MANAGER_IP_ADDR,
                "MICTLANX_REPLICA_MANAGER_PORT":str(MICTLANX_REPLICA_MANAGER_PORT),

                "MICTLANX_API_VERSION":str(MICTLANX_API_VERSION),
                "MICTLANX_EXPIRES_IN":MICTLANX_EXPIRES_IN
            }, 
            memory=1000000000,
            cpu_count=1, 
            mounts= {
                "/log":"/log",
                "/sink":"/sink",
                "/source":"/source"
            },
            network_id= DOCKER_NETWORK_ID,
            # ip_addr= Some(container_id)

        )
        response = summoner.summon(payload=payload, client_id=Some(MICTLANX_CLIENT_ID), app_id=Some(MICTLANX_APP_ID), authorization=Some(auth_response.token), secret=Some(MICTLANX_SECRET))
        print("Summon[{}] {}".format(i,response))
# from flask import current_app
# from concurrent.futures import ThreadPoolExecutor
# from rory.core.interfaces.createsecureclusteringworker import CreateSecureClusteringWorker

# """
# Description:
#     Deploy workers
# """
# def deploy_init_workers(*args):

#     INIT_WORKERS                       = args[0]
#     SECURE_CLUSTERING_MANAGER_HOSTNAME = args[1]
#     SECURE_CLUSTERING_MANAGER_PORT     = args[2]
#     DOCKER_IMAGE                       = args[3]
#     REPLICATOR                         = args[4]
#     LOGGER                             = args[5]
#     NODE_PREFIX                        = args[6]
#     MAX_WORKERS                        = args[7]
#     app                                = args[8]
#     TESTING                            = args[9]
#     STORAGE_SYSTEM_HOSTNAME            = args[10]
#     STORAGE_SYSTEM_PORT                = args[11]
#     MAX_RETRIES                        = args[12]
    
#     def __deploy_worker(*argss):
#         workerIndex = argss[0]
#         success     = False
#         try_counter = 0
#         while not success:
#             try:
#                 nodeId     = "{}{}".format(NODE_PREFIX,workerIndex)
#                 cr         = CreateSecureClusteringWorker(
#                     nodeId = nodeId,
#                     ports  = {
#                         "host": 9000+workerIndex, 
#                         "docker": 9000
#                     },
#                     nodeIndex = workerIndex,
#                     SECURE_CLUSTERING_MANAGER_HOSTNAME = SECURE_CLUSTERING_MANAGER_HOSTNAME,
#                     SECURE_CLUSTERING_MANAGER_PORT = SECURE_CLUSTERING_MANAGER_PORT,
#                     image = DOCKER_IMAGE,
#                     envs= {
#                         "TESTING":str(int(TESTING)),
#                         "STORAGE_SYSTEM_HOSTNAME":STORAGE_SYSTEM_HOSTNAME,
#                         "STORAGE_SYSTEM_PORT":str(STORAGE_SYSTEM_PORT)
#                     },
#                     HOST_LOG_PATH  = os.environ.get("HOST_LOG_PATH","/test/logs"),
#                     HOST_SINK_PATH = os.environ.get("HOST_SINK_PATH","/test/sink")  
#                 )
#                 LOGGER.debug(str(cr))
#                 startTime = time.time()
#                 with app.app_context():
#                     current_app.config["DEPLOY_START_TIMES"][nodeId] = startTime 
#                 response = REPLICATOR.deploy(cr)
#                 LOGGER.debug("DEPLOYMENT {}".format(nodeId))
#                 success=True
#             except Exception as e:
#                 try_counter += 1
#                 if(try_counter>=MAX_RETRIES):
#                     break
#                 LOGGER.error("ERROR_DEPLOY_WORKER tries={} worker_id={}".format(try_counter,workerIndex))
#                 success = False

#     with ThreadPoolExecutor(max_workers= MAX_WORKERS) as executor:
#         executor.map(__deploy_worker, range(INIT_WORKERS))