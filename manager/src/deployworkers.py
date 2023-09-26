import os, sys, time
from option import Option,NONE,Some
from mictlanx.v3.interfaces.payloads import SummonContainerPayload,ExposedPort,AuthTokenPayload, LogoutPayload
from mictlanx.v3.services.xolo import Xolo
from mictlanx.v3.services.summoner import Summoner

def deploy_nodes(
        summoner:Summoner,
        NODE_ID:str,
        PORT:str,
        WORKER_MAX_THREADS:int,
        DOCKER_IMAGE:str, 
        DOCKER_NETWORK_ID:str,
        MICTLANX_APP_ID:str ,
        MICTLANX_CLIENT_ID:str,
        MICTLANX_SECRET:str,
        xolo:Xolo,
        MICTLANX_SUMMONER_MODE:str="docker",
        init_workers:int = 1,
        MICTLANX_EXPIRES_IN:Option[str] = Some("15d"),
        NODE_PREFIX:str="rory-worker-",
        init_port:int = 3000,
        XOLO_ENABLE:bool = False,
        WORKER_MEMORY:str = "1000000000",
        WORKER_CPU:int = 2,
        WORKER_MICTLANX_PEERS:str = "mictlanx-peer-0:localhost:7000"
):
    
    swarm_nodes = ["2","3","4","8"]

    if XOLO_ENABLE:
        auth_result = xolo.auth(
            payload = AuthTokenPayload(
                app_id     = MICTLANX_APP_ID, 
                client_id  = MICTLANX_CLIENT_ID, 
                secret     = MICTLANX_SECRET,
                expires_in = Some(MICTLANX_EXPIRES_IN)
            )
        )
        if(auth_result.is_err):
            raise auth_result.unwrap_err()
        auth_response = auth_result.unwrap()
        authorization = Some(auth_response.token)
    else:
        authorization = NONE


    try:
        WORKER_BASE_PATH = os.environ.get("WORKER_BASE_PATH","/rory")
        N = len(swarm_nodes)
        for i in range(init_workers): 
            container_id     = "{}{}".format(NODE_PREFIX,i)
            HOST_BASE_PATH   = "{}/{}".format(WORKER_BASE_PATH, container_id)
            HOST_SOURCE_PATH = "{}/source".format(HOST_BASE_PATH)
            HOST_SINK_PATH   = "{}/sink".format(HOST_BASE_PATH)
            HOST_LOG_PATH    = "{}/log".format(HOST_BASE_PATH)

            CONTAINER_SOURCE_PATH = "{}/source".format(WORKER_BASE_PATH)
            CONTAINER_SINK_PATH   = "{}/sink".format(WORKER_BASE_PATH)
            CONTAINER_LOG_PATH    = "{}/log".format(WORKER_BASE_PATH)

            mounts = {
                    HOST_SOURCE_PATH:CONTAINER_SOURCE_PATH,
                    HOST_SINK_PATH:CONTAINER_SINK_PATH,
                    HOST_LOG_PATH:CONTAINER_LOG_PATH,

            }
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
                    "NODE_IP_ADDR":container_id,
                    "NODE_PORT":str(init_port),
                    "RORY_MANAGER_IP_ADDR":NODE_ID,
                    "RORY_MANAGER_PORT":str(PORT),
                    "DEBUG":"0",
                    "RELOAD":"0",
                    "LIU_ROUND":"2",
                    "SOURCE_PATH":CONTAINER_SOURCE_PATH,
                    "SINK_PATH":CONTAINER_SINK_PATH, 
                    "LOG_PATH":CONTAINER_LOG_PATH,
                    "MAX_ITERATIONS":"10",
                    "TESTING":"0",
                    "M":"3",
                    "MAX_THREADS":str(WORKER_MAX_THREADS),
                    "MICTLANX_PEERS":WORKER_MICTLANX_PEERS
                }, 
                labels={"target":"rory"},
                memory=int(WORKER_MEMORY),
                cpu_count=int(WORKER_CPU),
                mounts= mounts,
                network_id= DOCKER_NETWORK_ID,
                selected_node=Some(selected_node)
            )
            response = summoner.summon(
                payload=payload,
                mode=MICTLANX_SUMMONER_MODE,
                client_id=Some(MICTLANX_CLIENT_ID),
                app_id=Some(MICTLANX_APP_ID),
                authorization= authorization,
                secret=Some(MICTLANX_SECRET)
            )
    except Exception as e:
        print(e)
        sys.exit(1)
    finally:
        if XOLO_ENABLE:
            payload = LogoutPayload(
                app_id    = MICTLANX_APP_ID, 
                client_id = MICTLANX_CLIENT_ID,
                secret    = MICTLANX_SECRET, 
                token     = auth_response.token
            )
            xolo.logout(payload=payload)