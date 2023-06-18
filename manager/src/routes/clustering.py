import time
import json
from threading import Lock,Semaphore
from flask import Blueprint,current_app,request,abort,Response
from rory.core.interfaces.logger_metrics import LoggerMetrics

clustering = Blueprint("clustering",__name__,  url_prefix = "/clustering")
sem        = Semaphore(1)

# GET clustering/secure
@clustering.route("/secure", methods = ["POST","GET"])
def test():
    try:
        sem.acquire()
        arrivalTime        = time.time()        
        logger             = current_app.config["logger"]
        lb                 = current_app.config["lb"] # Get the load balancing 
        workers            = current_app.config.get("workers",{}) # Get the current bins (skmeans / dbskmeans nodes)
        workers            = list( filter( lambda x: x[1].isStarted, workers.items()))
        headers            = request.headers
        algorithm          = headers.get("Algorithm")
        startRequestTime   = headers.get("Start-Request-Time",0)
        getWorkerStartTime = headers.get("Get-Worker-Start-Time",0)
        latency            = arrivalTime - getWorkerStartTime

        if(request.method == "GET"):
            if(len(workers) == 0):
                sem.release()
                # Deploy a new Skmeans or dbskmeans
                abort(403)
            else:
                # ClusteringRequestClient\
                headers        = request.headers
                workerId       = lb.balance()
                workers        = current_app.config["workers"]
                worker         = workers[workerId]
                workerPort     = worker.port
                endTime        = time.time()
                serviceTime    = endTime - arrivalTime
                OPERATION_NAME = "BALANCING"
                LATENCY        = arrivalTime - float(startRequestTime)
                
                logger_metrics = LoggerMetrics(
                    operation_type = OPERATION_NAME,
                    matrix_id      = workerId,
                    algorithm      = algorithm,
                    arrival_time   = arrivalTime, 
                    end_time       = endTime, 
                    service_time   = serviceTime
                )
                logger.info(str(logger_metrics)+",{}".format(LATENCY))

                sem.release()
                return Response(
                    response = json.dumps({"workerId":workerId, "workerPort": workerPort}),
                    status   = 200,
                    headers  = {
                        "Service-Time": str(serviceTime),
                        "Latency": str(LATENCY),
                    }
                )
        else:
            sem.release()
            return Response(
                response = None,
                status   = 403
            )
    except Exception as e:
        logger.error(str(e))
        sem.release()
        return ("SERVER_ERROR",500)