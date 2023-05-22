import time
import json
from uuid import uuid4
from flask import Blueprint,current_app,request,abort
from threading import Lock
from rory.core.interfaces.worker import Worker
from rory.core.interfaces.createsecureclusteringworker import CreateSecureClusteringWorker

workers =   Blueprint("workers",__name__,url_prefix = "/workers")
@workers.route("/started", methods = ["POST","GET"])
def started():
    lock = Lock()
    lock.acquire()
    logger           = current_app.config["logger"]
    deployStartTimes = current_app.config.get("DEPLOY_START_TIMES",{})
    arrivalTime      = time.time()
    headers          = request.headers
    workerId         = headers["Worker-Id"]
    deployStartTime  = deployStartTimes.get(workerId,0)
    port             = int(headers["Worker-Port"])
    _worker          = Worker(
        workerId  = workerId,
        port      = port,
        isStarted = True
    )
    workers = current_app.config["workers"]
    current_app.config["workers"] = {**workers, **{workerId:_worker} }
    serviceTime  = _worker.createdAt - arrivalTime
    responseTime = time.time() - deployStartTime
    lock.release()
    if(request.method == "POST"):
        logger.info("STARTED_NODE,{},{}".format(serviceTime,responseTime))
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
    serviceTime = time.time() - arrivalTime
    logger.info("{},{}".format("GET_ALL_WORKERS",serviceTime))
    return json.dumps(workers)

@workers.route("/create", methods = ["POST"])
def create():
    arrivalTime = time.time()
    if(request.method == "POST"):
        data          = request.json
        workerId      = data.get("workerId",None) #Get Bin ID 
        workers       = current_app.config["workers"] #Get from config all workers
        replicator    = current_app.config["replicator"]
        if workerId in workers:
            return "UPDATED"
        else:
            _worker =  Worker(**data)
            current_app.config["workers"][workerId] = _worker
            createBin = CreateSecureClusteringWorker(
                **data,
                ports = {
                    "host":_worker.port + len(workers), 
                    "docker":6000
                }, 
                envs = {
                  "SECURE_CLUSTERING_MANAGER_HOSTNAME": current_app.config["HOSTNAME"],
                  "SECURE_CLUSTERING_MANAGER_PORT": current_app.config["PORT"],
                }
            )
            nodeReplicatorResponse = replicator.deploy(createBin = createBin)
            print(nodeReplicatorResponse)
            exitTime    = time.time()
            serviceTime = exitTime - arrivalTime
            return {
                "requestId":str(uuid4()),
                "serviceTime":serviceTime,
                "timeUnit":"sec"
            }