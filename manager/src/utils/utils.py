from mictlanx.services.summoner.summoner import Summoner,SummonContainerPayload,ExposedPort
# from mictlanx.v3.interfaces.payloads import SummonContainerPayload,ExposedPort
# from mictlanx.v3.interfaces.errors import ApiError 
from option import NONE,Result,Some

class Utils:
    @staticmethod
    def deploy_worker(
        replicator: Summoner,
        node_index:int,
        container_id:str,
        container_port:int,
        manager_ip_addr:str,
        manager_port:int,
        debug:str,
        _reload:str,
        liu_round:str,
        source_path:str,
        sink_path:str,
        log_path:str,
        max_iterations:int,
        testing:str,
        m:str,
        worker_max_threads:str,
        worker_mictlanx_peers:str,
        mictlanx_client_lb_algorithm:str,
        mictlanx_debug:str, 
        mictlanx_daemon:str,
        mictlanx_show_metrics:str,
        mictlanx_max_workers:str,
        mictlanx_disabled_log:str,
        docker_image:str,
        host_port:str,
        worker_memory:int,
        worker_cpu:int,
        docker_network_id:str

    )->Result[SummonContainerPayload,Exception]:
        envs =     {
                "NODE_INDEX"           : str(node_index),
                "NODE_IP_ADDR"         : container_id,
                "NODE_PORT"            : container_port,
                "RORY_MANAGER_IP_ADDR" : manager_ip_addr,
                "RORY_MANAGER_PORT"    : str(manager_port),
                "DEBUG"                : debug,
                "RELOAD"               : _reload,
                "LIU_ROUND"            : liu_round,
                "SOURCE_PATH"          : source_path,
                "SINK_PATH"            : sink_path, 
                "LOG_PATH"             : log_path,
                "max_iterations"       : max_iterations,
                "TESTING"              : testing,
                "M"                    : m,
                "MAX_THREADS":worker_max_threads,
                "MICTLANX_PEERS":worker_mictlanx_peers,
                "MICTLANX_CLIENT_LB_ALGORITHM":mictlanx_client_lb_algorithm,
                "MICTLANX_DEBUG":mictlanx_debug,
                "MICTLANX_DAEMON":mictlanx_daemon,
                "MICTLANX_SHOW_METRICS":mictlanx_show_metrics,
                "MICTLANX_MAX_WORKERS":mictlanx_max_workers,
                "MICTLANX_DISABLED_LOG":mictlanx_disabled_log
        }

        payload = SummonContainerPayload(
            image         = docker_image, 
            container_id  = container_id,
            hostname      = container_id,
            exposed_ports = [
                ExposedPort (
                    ip_addr        = NONE,
                    host_port      = int(host_port),
                    container_port = int(container_port),
                    protocolo      = NONE 
                )
            ],
            envs = envs,
            labels={"target":"rory"},
            memory=int(worker_memory),
            cpu_count=int(worker_cpu), 
            mounts= {
                "/rory/{}/source".format(container_id):"/rory/source",
                "/rory/{}/sink".format(container_id):"/rory/sink",
                "/rory/{}/logs".format(container_id):"/rory/log",
                "/rory/{}/mictlanx".format(container_id):"/rory/mictlanx",
            },
            network_id= docker_network_id,
            force=Some(True)
        )

        response = replicator.summon(
            payload       = payload, 
            client_id     = NONE,
            app_id        = NONE,
            authorization = NONE, 
            secret        = NONE
        )
        return response