import os, sys, time
from option import NONE,Some,Result,Ok,Err
from mictlanx.logger.log import Log
from mictlanx.v4.summoner.summoner import Summoner,SummonContainerPayload,ExposedPort
from mictlanx.interfaces.payloads import MountX
from typing import List

def deploy_nodes(
        log:Log,
        summoner:Summoner,
        NODE_ID:str,
        PORT:str,
        WORKER_MAX_THREADS:int,
        DOCKER_IMAGE:str, 
        DOCKER_NETWORK_ID:str,
        MAX_RETRIES:int,
        SERVER_IP_ADDR:str,
        DISTANCE:str,
        MIN_ERROR:str,
        CKKS_ROUND:int,
        CKKS_DECIMALS:str,
        CTX_FILENAME:str,
        PUBKEY_FILENAME:str,
        SECRET_KEY_FILENAME:str,
        RELINKEY_FILENAME:str,
        MICTLANX_CLIENT_ID:str,
        MICTLANX_DEBUG:bool,
        MICTLANX_TIMEOUT:int,
        MICTLANX_API_VERSION:int,
        MICTLANX_PROTOCOL:str,
        MICTLANX_LOG_PATH:str,
        MICTLANX_LOG_INTERVAL:str,
        MICTLANX_LOG_WHEN:str,
        MICTLANX_BUCKET_ID:str,
        MICTLANX_DELAY:int,
        MICTLANX_BACKOFF_FACTOR:float,
        MICTLANX_MAX_RETRIES:int,
        MICTLANX_CHUNK_SIZE:str,
        MICTLANX_MAX_PARALELL_GETS:int,
        LIU_ROUND:int               = 2, 
        MICTLANX_SUMMONER_MODE:str  = "docker",
        init_workers:int            = 1,
        NODE_PREFIX:str             = "rory-worker-",
        FOLDER_KEYS:str             = "keys128",
        init_port:int               = 3000,
        WORKER_MEMORY:str           = "1000000000",
        WORKER_CPU:int              = 2,
        WORKER_MICTLANX_ROUTERS:str = "mictlanx-peer-0:localhost:7000",
        MICTLANX_MAX_WORKERS:int    = 12,
        swarm_nodes:List[str]       = ["2","3","4","8"]
)->Result[bool, Exception]:

    try:
        WORKER_BASE_PATH = os.environ.get("WORKER_BASE_PATH","/rory")
        WORKER_KEYS_BASE_PATH = os.environ.get("WORKER_KEYS_BASE_PATH","/rory")
        N = len(swarm_nodes)
        for i in range(init_workers): 
            container_id     = "{}{}".format(NODE_PREFIX,i)
            # HOST_BASE_PATH   = "{}/{}".format(WORKER_BASE_PATH, container_id)
            # HOST_SOURCE_PATH = "{}/source".format(HOST_BASE_PATH)
            # HOST_SINK_PATH   = "{}/sink".format(HOST_BASE_PATH)
            # HOST_LOG_PATH    = "{}/log".format(HOST_BASE_PATH)
            # HOST_MICTLANX_CLIENT_PATH = "{}/mictlanx".format(HOST_BASE_PATH)

            CONTAINER_SOURCE_PATH = "{}/source".format(WORKER_BASE_PATH)
            CONTAINER_SINK_PATH   = "{}/sink".format(WORKER_BASE_PATH)
            CONTAINER_LOG_PATH    = "{}/log".format(WORKER_BASE_PATH)
            CONTAINER_KEYS_PATH   = "{}/keys".format(WORKER_BASE_PATH)
            CONTAINER_MICTLANX_CLIENT_PATH = "{}/mictlanx".format(WORKER_BASE_PATH)
            
            mounts = [
                MountX(
                    source = "{}-source".format(container_id),
                    target = CONTAINER_SOURCE_PATH,
                    mount_type=1
                ),
                MountX(
                    source = "{}-sink".format(container_id),
                    target = CONTAINER_SINK_PATH,
                    mount_type=1
                ),
                MountX(
                    source = "{}-log".format(container_id),
                    target = CONTAINER_LOG_PATH,
                    mount_type=1
                ),
                MountX(
                    source = "{}-mictlanx".format(container_id),
                    target = CONTAINER_MICTLANX_CLIENT_PATH,
                    mount_type=1
                ),
                MountX(
                    #/rory/rory-worker-0/keys/keys128
                    source = "{}/{}/keys/{}".format(WORKER_KEYS_BASE_PATH,container_id,FOLDER_KEYS),
                    target = CONTAINER_KEYS_PATH,
                    mount_type=0
                )
            ]
            selected_node = swarm_nodes[i % N]
            payload       = SummonContainerPayload(
                image         = DOCKER_IMAGE, 
                container_id  = container_id,
                hostname      = container_id,
                exposed_ports = [
                    ExposedPort(
                        ip_addr        = NONE,
                        host_port      = init_port+i,
                        container_port = init_port,
                        protocolo      = NONE
                    )
                ],
                envs={
                    "NODE_INDEX":str(i),
                    "NODE_ID":NODE_ID,
                    "NODE_IP_ADDR":container_id,
                    "NODE_PORT":str(init_port),
                    "RORY_MANAGER_IP_ADDR":NODE_ID,
                    "RORY_MANAGER_PORT":str(PORT),
                    "SERVER_IP_ADDR":str(SERVER_IP_ADDR),
                    "DEBUG":"0",
                    "RELOAD":"0",
                    "LIU_ROUND":str(LIU_ROUND),
                    "DISTANCE":str(DISTANCE),
                    "MIN_ERROR":str(MIN_ERROR),
                    "CKKS_ROUND":str(CKKS_ROUND),
                    "CKKS_DECIMALS":str(CKKS_DECIMALS),
                    "CTX_FILENAME":str(CTX_FILENAME),
                    "PUBKEY_FILENAME":str(PUBKEY_FILENAME),
                    "SECRET_KEY_FILENAME":str(SECRET_KEY_FILENAME),
                    "RELINKEY_FILENAME":str(RELINKEY_FILENAME),
                    "SOURCE_PATH":CONTAINER_SOURCE_PATH,
                    "SINK_PATH":CONTAINER_SINK_PATH, 
                    "LOG_PATH":CONTAINER_LOG_PATH,
                    "KEYS_PATH":CONTAINER_KEYS_PATH,
                    "TESTING":"0",
                    "MAX_RETRIES":str(MAX_RETRIES),
                    "MAX_THREADS":str(WORKER_MAX_THREADS),
                    "MICTLANX_ROUTERS":WORKER_MICTLANX_ROUTERS,
                    "MICTLANX_DEBUG":str(int(MICTLANX_DEBUG)),
                    "MICTLANX_MAX_WORKERS":str(MICTLANX_MAX_WORKERS),
                    "MICTLANX_TIMEOUT":str(MICTLANX_TIMEOUT),
                    "MICTLANX_API_VERSION":str(MICTLANX_API_VERSION),
                    "MICTLANX_PROTOCOL":str(MICTLANX_PROTOCOL),
                    "MICTLANX_LOG_PATH":str(MICTLANX_LOG_PATH),
                    "MICTLANX_LOG_INTERVAL":str(MICTLANX_LOG_INTERVAL),
                    "MICTLANX_LOG_WHEN":str(MICTLANX_LOG_WHEN),
                    "MICTLANX_BUCKET_ID":str(MICTLANX_BUCKET_ID),
                    "MICTLANX_DELAY":str(MICTLANX_DELAY),
                    "MICTLANX_BACKOFF_FACTOR":str(MICTLANX_BACKOFF_FACTOR),
                    "MICTLANX_MAX_RETRIES":str(MICTLANX_MAX_RETRIES),
                    "MICTLANX_CHUNK_SIZE":str(MICTLANX_CHUNK_SIZE),
                    "MICTLANX_MAX_PARALELL_GETS":str(MICTLANX_MAX_PARALELL_GETS),
                }, 
                labels={"target":"rory"},
                memory=int(WORKER_MEMORY),
                cpu_count=int(WORKER_CPU),
                mounts= mounts,
                network_id= DOCKER_NETWORK_ID,
                selected_node=Some(selected_node),
                force=Some(True)
            )
            response = summoner.summon(
                payload=payload,
                mode=MICTLANX_SUMMONER_MODE,
                client_id=Some(MICTLANX_CLIENT_ID),
                app_id=NONE,
                authorization=NONE,
                secret= NONE
            )
            if response.is_err:
                log.error({
                    "msg":str(response.unwrap_err())
                })
        return Ok(True)

    except Exception as e:
        return Err(e)