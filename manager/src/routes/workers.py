import time
import json
from uuid import uuid4
from flask import Blueprint,current_app,request,abort
from threading import Lock
from rory.core.interfaces.worker import Worker
from rory.core.interfaces.createsecureclusteringworker import CreateSecureClusteringWorker
from rory.core.interfaces.logger_metrics import LoggerMetrics
from mictlanx.v3.services.summoner import Summoner
from mictlanx.v3.interfaces.payloads import SummonContainerPayload,ExposedPort
from option import NONE

workers =   Blueprint("workers",__name__,url_prefix = "/workers")
@workers.route("/started", methods = ["POST","GET"])
def started():
    lock = Lock()
    lock.acquire()
    logger           = current_app.config["logger"]
    # deployStartTime  = current_app.config.get("DEPLOY_START_TIMES",{})
    # latency          = 
    arrivalTime      = time.time()
    headers          = request.headers
    workerId         = headers["Worker-Id"]
    # deployStartTime  = deployStartTime.get(workerId,0)
    port             = int(headers["Worker-Port"])
    _worker          = Worker(
        workerId  = workerId,
        port      = port,
        isStarted = True
    )
    workers                       = current_app.config["workers"]
    current_app.config["workers"] = {**workers, **{workerId:_worker} }
    #serviceTime  = _worker.createdAt - arrivalTime
    # responseTime = time.time() - deployStartTime
    end_time = time.time() 
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
        # logger.info("STARTED_NODE,{},{}".format(serviceTime,responseTime))
        return ('',204)
    else: abort(404)

#GET /workers/
@workers.route("", methods = ["GET"])
def getAll():
    arrivalTime = time.time()
    logger      = current_app.config["logger"]
    workers     = current_app.config["workers"]
    logger.debug(str(workers))
    workers     = dict([(k,v.__dict__) for k,v in workers.items()]) 
    end_time     = time.time()
    service_time = end_time - arrivalTime
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
    arrival_time = time.time()
    container_id = "worker-0"
    payload = SummonContainerPayload(
        container_id=container_id,
        image="sreyes/rory:worker",
        hostname= container_id,
        exposed_ports=[
            ExposedPort(
                ip_addr=NONE,
                container_port= 6666,
                host_port= 6666
            )
        ],
        envs= {},
        memory=1000000000,
        cpu_count=1,
        mounts= {}
    )
    # summoner.summon(payload)