import time
import json
from option import Result
from threading import Semaphore
from flask import Blueprint,current_app,request,Response
from mictlanx.v4.summoner.summoner import Summoner,SummonContainerPayload
from utils.utils import Utils
from rory.core.interfaces.worker import Worker

clustering = Blueprint("clustering",__name__,  url_prefix = "/clustering")
sem        = Semaphore(1)


@clustering.route("/test",methods=["GET","POST"])
def test():
    return Response(
        response = json.dumps({
            "component_type":"manager"
        }),
        status   = 200,
        headers  = {
            "Component-Type":"manager"
        }
    )

# GET clustering/secure
@clustering.route("/secure", methods = ["POST","GET"])
def test_secure():
    global sem
    try:
        sem.acquire()
        arrival_time                 = time.time()        
        logger                       = current_app.config["logger"]
        lb                           = current_app.config["lb"] # Get the load balancing 
        workers                      = current_app.config.get("workers",{}) # Get the current bins (skmeans / dbskmeans nodes)
        workers                      = list( filter( lambda x: x[1].isStarted, workers.items()))
        replicator:Summoner          = current_app.config["replicator"]
        NODE_ID                      = current_app.config["NODE_ID"]
        PORT                         = current_app.config["NODE_PORT"]
        DOCKER_IMAGE                 = current_app.config["DOCKER_IMAGE"]
        DOCKER_NETWORK_ID            = current_app.config["DOCKER_NETWORK_ID"]
        INIT_PORT                    = int(current_app.config["INIT_WORKER_PORT"])
        logger                       = current_app.config["logger"]
        current_workers              = current_app.config["workers"]
        n_workers                    = len(current_workers)
        worker_port                  = str(n_workers+INIT_PORT)
        headers                      = request.headers
        algorithm                    = headers.get("Algorithm")
        startRequestTime             = headers.get("Start-Request-Time",0)
        getWorkerStartTime           = headers.get("Get-Worker-Start-Time",0)
        latency                      = arrival_time - getWorkerStartTime
        HOST_PORT                    = headers.get("Host-Port",worker_port)
        container_id                 = headers.get("Container-Id","worker-{}".format(n_workers))
        CONTAINER_PORT               = headers.get("Container-Port", worker_port)
        WORKER_MEMORY                = headers.get("Worker-Memory","1000000000")
        WORKER_CPU                   = headers.get("Worker-Cpu","1")
        DEBUG                        = headers.get("Debug","0")
        RELOAD                       = headers.get("Reload","0")
        LIU_ROUND                    = headers.get("Liu-Round","1")
        SINK_PATH                    = headers.get("Sink-Path","/rory/sink")
        SOURCE_PATH                  = headers.get("Source-Path","/rory/source")
        LOG_PATH                     = headers.get("Log-Path","/rory/log")
        TESTING                      = headers.get("Testing","0")
        MAX_ITERATIONS               = headers.get("Max-Iterations","10")
        M                            = headers.get("M","3")
        WORKER_MAX_THREADS           = headers.get("Max-Threads","4")
        WORKER_MICTLANX_PEERS        = headers.get("Mictlanx-Peers","mictlanx-router-0:localhost:60666")
        MICTLANX_CLIENT_LB_ALGORITHM = headers.get("Mictlanx-Lb-Algorithm","2CHOICES_UF")
        MICTLANX_DEBUG               = headers.get("Mictlanx-Debug","0")
        MICTLANX_DAEMON              = headers.get("Mictlanx-Daemon","0")
        MICTLANX_SHOW_METRICS        = headers.get("Mictlanx-Show-Metrics","0")
        MICTLANX_MAX_WORKERS         = headers.get("Mictlanx-Max-Workers","4")
        MICTLANX_DISABLED_LOG        = headers.get("Mictlanx-Disabled-Log","1")
        matrix_id                    = headers.get("Matrix-id","matrix0")

        OPERATION_NAME = "BALANCING"
        if(request.method == "GET"):
            if(len(workers) == 0):
                logger.debug({
                    "event":"NO.WORKER",
                    "algorithm":algorithm,
                    "docker_image":DOCKER_IMAGE,
                    "docker_network_id":DOCKER_NETWORK_ID,
                    "init_port":INIT_PORT,
                    "host_port":HOST_PORT,
                    "container_id":container_id,
                    "container_port":CONTAINER_PORT,
                    "worker_memory":WORKER_MEMORY,
                    "worker_cpu":WORKER_CPU,
                    "debug":DEBUG,
                    "reload":RELOAD,
                    "liu_round":LIU_ROUND,
                    "sink_path":SINK_PATH,
                    "log_path":LOG_PATH,
                    "testing":TESTING,
                    "max_iterations":MAX_ITERATIONS,
                    "m":M,
                    "worker_max_threads":WORKER_MAX_THREADS,
                    "worker_mictlanx_peers":WORKER_MICTLANX_PEERS,
                    "mictlanx_client_lb_algorithm":MICTLANX_CLIENT_LB_ALGORITHM,
                    "mictlanx_debug":MICTLANX_DEBUG,
                    "mictlanx_daemon":MICTLANX_DAEMON,
                    "mictlanx_show_metrics":MICTLANX_SHOW_METRICS,
                    "mictlanx_max_workers":MICTLANX_MAX_WORKERS,
                    "mictlanx_disabled_log":MICTLANX_DISABLED_LOG

                })
                result:Result[SummonContainerPayload,Exception] = Utils.deploy_worker(
                    replicator=replicator,
                    node_index=n_workers,
                    container_id=container_id,
                    container_port=CONTAINER_PORT,
                    manager_ip_addr=NODE_ID,
                    manager_port= PORT,
                    debug=DEBUG,
                    _reload=RELOAD,
                    liu_round=LIU_ROUND,
                    source_path=SOURCE_PATH,
                    sink_path=SINK_PATH,
                    log_path=LOG_PATH,
                    max_iterations=MAX_ITERATIONS,
                    testing=TESTING,
                    m=M,
                    worker_max_threads=WORKER_MAX_THREADS,
                    worker_mictlanx_peers=WORKER_MICTLANX_PEERS,
                    mictlanx_client_lb_algorithm= MICTLANX_CLIENT_LB_ALGORITHM,
                    mictlanx_debug=MICTLANX_DEBUG,
                    mictlanx_daemon=MICTLANX_DAEMON,
                    mictlanx_show_metrics=MICTLANX_SHOW_METRICS,
                    mictlanx_max_workers=MICTLANX_MAX_WORKERS,
                    mictlanx_disabled_log=MICTLANX_DISABLED_LOG,
                    docker_image=DOCKER_IMAGE,
                    host_port=HOST_PORT,
                    worker_memory=WORKER_MEMORY,
                    worker_cpu=WORKER_CPU,
                    docker_network_id=DOCKER_NETWORK_ID
                )
                if result.is_err:
                    error = result.unwrap_err()
                    logger.error(str(error))
                    return Response(str(error), status=500)
                response = result.unwrap()
                _worker = Worker(
                    workerId  = container_id,
                    port      = CONTAINER_PORT,
                    isStarted = True
                )
                workers = current_app.config["workers"]
                current_app.config["workers"] = {**workers, **{container_id:_worker} }

                service_time = time.time() - arrival_time
                sem.release()
                logger.info({
                    "event":OPERATION_NAME,
                    "service_time":service_time,
                    "algorithm":algorithm,
                    "worker_id":worker_id
                })
                return Response(
                    response = json.dumps({
                        "worker_id":response.container_id,
                        "worker_port": worker_port,
                        "service_time":service_time
                    }),
                    status   = 200,
                    headers  = {
                        "Service-Time": str(service_time),
                    }
                )
            else:
                headers      = request.headers
                worker_id    = lb.balance()
                workers      = current_app.config["workers"]
                worker       = workers[worker_id]
                worker_port  = worker.port
                end_time     = time.time()
                service_time = end_time - arrival_time
                
                logger.info({
                    "event":OPERATION_NAME,
                    "service_time":service_time,
                    "matrix_id":matrix_id,
                    "algorithm":algorithm,
                    "worker_id":worker_id
                })

                sem.release()
                return Response(
                    response = json.dumps({
                        "worker_id":worker_id,
                        "worker_port": worker_port,
                        "service_time":service_time
                    }),
                    status   = 200,
                    headers  = {
                        "Service-Time": str(service_time),
                    }
                )
        else:
            sem.release()
            return Response(
                response = None,
                status   = 403
            )
    except Exception as e:
        sem.release()
        logger.error(str(e))
        return ("SERVER_ERROR",500)