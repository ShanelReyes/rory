import time
import json
from option import Some,NONE,Option
from uuid import uuid4
from flask import Blueprint,current_app,request,abort
from threading import Lock
from rory.core.interfaces.worker import Worker
from rory.core.interfaces.createroryworker import CreateRoryWorker
from rory.core.interfaces.logger_metrics import LoggerMetrics
from mictlanx.v3.services.summoner import Summoner
from mictlanx.v3.interfaces.payloads import SummonContainerPayload,ExposedPort
from option import NONE
from mictlanx.v3.client import Client

workers =   Blueprint("workers",__name__,url_prefix = "/workers")
@workers.route("/started", methods = ["POST","GET"])
def started():
    lock = Lock()
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
        logger_metrics = LoggerMetrics(
            operation_type = "STARTED_NODE",
            matrix_id      = workerId, 
            arrival_time   = arrivalTime, 
            end_time       = end_time,
            service_time   = service_time 
        )
        logger.info(str(logger_metrics))
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

@workers.route("/create", methods = ["POST"])
def create():
    arrival_time                     = time.time()
    container_id                     = "worker-0"
    replicator:Summoner              = current_app.config["replicator"]
    STORAGE_CLIENT:Client            = current_app.config["STORAGE_CLIENT"]
    NODE_ID                          = current_app.config["NODE_ID"]
    PORT                             = current_app.config["NODE_PORT"]
    DOCKER_IMAGE                     = current_app.config["DOCKER_IMAGE"]
    MICTLANX_APP_ID                  = current_app.config["MICTLANX_APP_ID"]
    MICTLANX_SECRET                  = current_app.config["MICTLANX_SECRET"]
    MICTLANX_PROXY_IP_ADDR           = current_app.config["MICTLANX_PROXY_IP_ADDR"]
    MICTLANX_PROXY_PORT              = current_app.config["MICTLANX_PROXY_PORT"]
    MICTLANX_XOLO_IP_ADDR            = current_app.config["MICTLANX_XOLO_IP_ADDR"]
    MICTLANX_XOLO_PORT               = current_app.config["MICTLANX_XOLO_PORT"]
    MICTLANX_REPLICA_MANAGER_IP_ADDR = current_app.config["MICTLANX_REPLICA_MANAGER_IP_ADDR"]
    MICTLANX_REPLICA_MANAGER_PORT    = current_app.config["MICTLANX_REPLICA_MANAGER_PORT"]
    MICTLANX_API_VERSION             = current_app.config["MICTLANX_API_VERSION"]
    MICTLANX_EXPIRES_IN              = current_app.config["MICTLANX_EXPIRES_IN"]
    MICTLANX_CLIENT_ID               = current_app.config["MICTLANX_CLIENT_ID"]
    HOST_PORT                        = 7000
    DOCKER_NETWORK_ID                = current_app.config["DOCKER_NETWORK_ID"]
    CONTAINER_PORT                   = 7000
    authorization                    = STORAGE_CLIENT.credentials.authorization
    payload                          = SummonContainerPayload(
        image         = DOCKER_IMAGE, 
        container_id  = container_id,
        hostname      = container_id,
        exposed_ports = [
            ExposedPort (
                ip_addr        = NONE,
                host_port      = HOST_PORT,
                container_port = CONTAINER_PORT,
                protocolo      = NONE 
            )
        ],
        envs = {
            "NODE_IP_ADDR"                    : container_id,
            "NODE_PORT"                       : CONTAINER_PORT,
            "RORY_MANAGER_IP_ADDR"            : NODE_ID,
            "RORY_MANAGER_PORT"               : str(PORT),
            "DEBUG"                           : "0",
            "RELOAD"                          : "0",
            "LIU_ROUND"                       : "2",
            "SINK_PATH"                       : "/sink", 
            "SOURCE_PATH"                     : "/source",
            "MAX_ITERATIONS"                  : "10",
            "LOG_PATH"                        : "/log",
            "TESTING"                         : "0",
            "M"                               : "3",
            "MICTLANX_APP_ID"                 : MICTLANX_APP_ID, 
            "MICTLANX_CLIENT_ID"              : container_id,
            "MICTLANX_SECRET"                 : MICTLANX_SECRET,
            "MICTLANX_PROXY_IP_ADDR"          : MICTLANX_PROXY_IP_ADDR,
            "MICTLANX_PROXY_IP_PORT"          : MICTLANX_PROXY_PORT,
            "MICTLANX_XOLO_IP_ADDR"           : str(MICTLANX_XOLO_IP_ADDR),
            "MICTLANX_XOLO_PORT"              : str(MICTLANX_XOLO_PORT),
            "MICTLANX_REPLICA_MANAGER_IP_ADDR": MICTLANX_REPLICA_MANAGER_IP_ADDR,
            "MICTLANX_REPLICA_MANAGER_PORT"   : str(MICTLANX_REPLICA_MANAGER_PORT),
            "MICTLANX_API_VERSION"            : str(MICTLANX_API_VERSION),
            "MICTLANX_EXPIRES_IN"             : MICTLANX_EXPIRES_IN
        }, 
        memory=1000000000,
        cpu_count=1, 
        mounts= {
            "/rory/{}/data".format(container_id):"/log",
            "/rory/{}/logs".format(container_id):"/log",
        },
        network_id= DOCKER_NETWORK_ID,
        # ip_addr= Some(container_id)
    )

    response = replicator.summon(
        payload       = payload, 
        client_id     = Some(MICTLANX_CLIENT_ID), 
        app_id        = Some(MICTLANX_APP_ID), 
        authorization = authorization, 
        secret        = Some(MICTLANX_SECRET))