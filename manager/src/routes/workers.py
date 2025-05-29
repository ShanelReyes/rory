import time
import json
from flask import Blueprint,current_app,request,abort,Response
from threading import Lock
from rory.core.interfaces.worker import Worker
from rory.core.interfaces.logger_metrics import LoggerMetrics
from mictlanx.v4.summoner.summoner import Summoner,SummonContainerPayload
from utils.utils import Utils
from option import Result

lock = Lock()
workers =   Blueprint("workers",__name__,url_prefix = "/workers")
@workers.route("/started", methods = ["POST","GET"])
def started():
    lock.acquire()
    logger      = current_app.config["logger"]
    arrivalTime = time.time()
    headers     = request.headers
    workerId    = headers["Worker-Id"]
    port        = int(headers["Worker-Port"])
    _worker     = Worker(
        workerId  = workerId,
        port      = port,
        isStarted = True
    )
    workers      = current_app.config["workers"]
    current_app.config["workers"] = {**workers, **{workerId:_worker} }
    end_time     = time.time() 
    service_time = end_time - arrivalTime 

    lock.release()
    if(request.method == "POST"):
        logger.info({
            "event":"WORKER.STARTED",
            "worker_id":workerId,
            "num_workers":len(current_app.config["workers"]),
            "service_time":service_time,
        })
        return ('',204)
    else: abort(404)

#GET /workers/
@workers.route("", methods = ["GET"])
def getAll():
    arrivalTime    = time.time()
    logger         = current_app.config["logger"]
    workers        = current_app.config["workers"]
    logger.debug(str(workers))
    workers        = dict([(k,v.__dict__) for k,v in workers.items()]) 
    end_time       = time.time()
    service_time   = end_time - arrivalTime
    logger_metrics = LoggerMetrics(
            operation_type = "GET_ALL_WORKERS",
            arrival_time   = arrivalTime, 
            end_time       = end_time,
            service_time   = service_time 
        )
    logger.info(str(logger_metrics))
    return json.dumps(workers)

@workers.route("/deploy", methods = ["POST"])
def deploy_worker():
    arrival_time                 = time.time()
    headers                      = request.headers
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
    HOST_PORT                    = headers.get("Host-Port",worker_port)
    container_id                 = headers.get("Container-Id","worker-{}".format(n_workers))
    CONTAINER_PORT               = headers.get("Container-Port", worker_port)
    WORKER_MEMORY                = headers.get("Worker-Memory","1000000000")
    WORKER_CPU                   = headers.get("Worker-Cpu","1")
    DEBUG                        = headers.get("Debug","0")
    RELOAD                       = headers.get("Reload","0")
    LIU_ROUND                    = headers.get("Liu-Round","1")
    SINK_PATH                    = headers.get("Sink-Path","/sink")
    SOURCE_PATH                  = headers.get("Source-Path","/source")
    LOG_PATH                     = headers.get("Log-Path","/log")
    TESTING                      = headers.get("Testing","0")
    MAX_ITERATIONS               = headers.get("Max-Iterations","10")
    M                            = headers.get("M","3")
    WORKER_MAX_THREADS           = headers.get("Max-Threads","4")
    WORKER_MICTLANX_PEERS        = headers.get("Mictlanx-Peers","mictlanx-peer-0:mictlanx-peer-0:7000")
    MICTLANX_CLIENT_LB_ALGORITHM = headers.get("Mictlanx-Lb-Algorithm","2CHOICES_UF")
    MICTLANX_DEBUG               = headers.get("Mictlanx-Debug","0")
    MICTLANX_DAEMON              = headers.get("Mictlanx-Daemon","0")
    MICTLANX_SHOW_METRICS        = headers.get("Mictlanx-Show-Metrics","0")
    MICTLANX_MAX_WORKERS         = headers.get("Mictlanx-Max-Workers","4")
    MICTLANX_DISABLED_LOG        = headers.get("Mictlanx-Disabled-Log","1")

    envs =     {
            "NODE_INDEX"           : str(n_workers),
            "NODE_IP_ADDR"         : container_id,
            "NODE_PORT"            : CONTAINER_PORT,
            "RORY_MANAGER_IP_ADDR" : NODE_ID,
            "RORY_MANAGER_PORT"    : str(PORT),
            "DEBUG"                : DEBUG,
            "RELOAD"               : RELOAD,
            "LIU_ROUND"            : LIU_ROUND,
            "SOURCE_PATH"          : SOURCE_PATH,
            "SINK_PATH"            : SINK_PATH, 
            "LOG_PATH"             : LOG_PATH,
            "MAX_ITERATIONS"       : MAX_ITERATIONS,
            "TESTING"              : TESTING,
            "M"                    : M,
            "MAX_THREADS":WORKER_MAX_THREADS,
            "MICTLANX_PEERS":WORKER_MICTLANX_PEERS,
            "MICTLANX_CLIENT_LB_ALGORITHM":MICTLANX_CLIENT_LB_ALGORITHM,
            "MICTLANX_DEBUG":MICTLANX_DEBUG,
            "MICTLANX_DAEMON":MICTLANX_DAEMON,
            "MICTLANX_SHOW_METRICS":MICTLANX_SHOW_METRICS,
            "MICTLANX_MAX_WORKERS":MICTLANX_MAX_WORKERS,
            "MICTLANX_DISABLED_LOG":MICTLANX_DISABLED_LOG
    }

    logger.debug({
        "event":"WORKER.DEPLOY.ENVS",
        **envs
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
        m = M,
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
        return Response(str(result.unwrap_err()), status=500)
    

    response = result.unwrap()
    logger.info({
        "event":"WORKER.DEPLOY",
        "container_id":response.container_id,
        "cpu_count":response.cpu_count,
        "memory":response.memory,
    })
    response = json.dumps({
      "container_id":container_id,
      "port":worker_port
    })
    return Response(response, status=200)