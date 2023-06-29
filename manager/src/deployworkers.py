import os, sys, time
from option import Option,NONE,Some
from mictlanx.v3.interfaces.payloads import SummonContainerPayload,ExposedPort,AuthTokenPayload, LogoutPayload
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
        # logger:
        MICTLANX_SUMMONER_MODE:str="docker",
        init_workers:int = 1,
        MICTLANX_API_VERSION:Option[int] = Some(3),
        MICTLANX_EXPIRES_IN:Option[str] = Some("15d"),
        NODE_PREFIX:str="rory-worker-",
        init_port:int = 3000,
):
    
    swarm_nodes = list(map(str, range(3,10)))

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

    try:
        WORKER_BASE_PATH = os.environ.get("WORKER_BASE_PATH","/rory")
        # SWARM_CLUSTER_NODES = int(os.environ.get("SWARM_CLUSTER_NODES",10))
        N = len(swarm_nodes)
        for i in range(init_workers): 
            container_id  = "{}{}".format(NODE_PREFIX,i)
            HOST_BASE_PATH = "{}/{}".format(WORKER_BASE_PATH, container_id)
            # 
            HOST_SOURCE_PATH = "{}/source".format(HOST_BASE_PATH)
            HOST_SINK_PATH = "{}/sink".format(HOST_BASE_PATH)
            HOST_LOG_PATH = "{}/log".format(HOST_BASE_PATH)

            CONTAINER_SOURCE_PATH = "{}/source".format(WORKER_BASE_PATH)
            CONTAINER_SINK_PATH = "{}/sink".format(WORKER_BASE_PATH)
            CONTAINER_LOG_PATH = "{}/log".format(WORKER_BASE_PATH)
            # CONTAINER_BASE_PATH = "{}/{}".format(CONTAINER_BASE_PATH,container_id) # /rory/<container_id>/
            # LOG_PATH      = "{}/log".format(WORKER_PATH)
            # SINK_PATH     = "{}/sink".format(WORKER_PATH)
            # SOURCE_PATH   = "{}/source".format(WORKER_PATH)
            selected_node = swarm_nodes[i % N]
            payload     = SummonContainerPayload(
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
                    "MICTLANX_APP_ID":MICTLANX_APP_ID, 
                    "MICTLANX_CLIENT_ID":container_id,
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
                    HOST_SOURCE_PATH:CONTAINER_SOURCE_PATH,
                    HOST_SINK_PATH: CONTAINER_SINK_PATH,
                    HOST_LOG_PATH:CONTAINER_LOG_PATH,
                },
                network_id= DOCKER_NETWORK_ID,
                selected_node=Some(selected_node)

            )
            response = summoner.summon(
                payload=payload,
                mode=MICTLANX_SUMMONER_MODE,
                client_id=Some(MICTLANX_CLIENT_ID),
                app_id=Some(MICTLANX_APP_ID),
                authorization=Some(auth_response.token),
                secret=Some(MICTLANX_SECRET)
            )
            if response.is_ok:
                print("WORKER_ID={}".format(container_id))
                print("HOST_WORKER_PORT={}".format(init_port+i))
                print("CONTAINER_WORKER_PORT={}".format(init_port))

                print("HOST_SOURCE_PATH",HOST_SOURCE_PATH)
                print("CONTAINER_SOURCE_PATH",CONTAINER_SOURCE_PATH)

                print("HOST_SINK_PATH",HOST_SINK_PATH)
                print("CONTAINER_SINK_PATH",CONTAINER_SINK_PATH)

                print("HOST_LOG_PATH",HOST_LOG_PATH)
                print("CONTAINER_LOG_PATH",CONTAINER_LOG_PATH)
            else:
                error = response.unwrap_err()
                print("Summoner error {}".format(error))
            
            # print("Summon[{}] {}".format(i,response))
    except Exception as e:
        print(e)
        sys.exit(1)
    finally:
        payload = LogoutPayload(app_id=MICTLANX_APP_ID, client_id=MICTLANX_CLIENT_ID,secret=MICTLANX_SECRET, token=auth_response.token)
        xolo.logout(payload=payload)