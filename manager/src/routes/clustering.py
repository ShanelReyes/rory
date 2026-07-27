import time
from threading import Semaphore
from fastapi import APIRouter, Depends, HTTPException, Request
from mictlanx.services.summoner.summoner import Summoner, SummonContainerPayload
from utils.utils import Utils
from rory.core.interfaces.worker import Worker
from option import Result

from models.requests import SecureWorkerRequest
from models.responses import HealthCheckResponse, SecureWorkerResponse

from dependencies import get_logger, get_replicator, get_lb, get_workers, get_settings

router = APIRouter(prefix="/clustering", tags=["Clustering"])
sem = Semaphore(1)


@router.api_route(
    "/test",
    methods=["GET", "POST"],
    response_model=HealthCheckResponse,
    summary="Health check and component identification",
    description="Verify the availability of the Manager component and confirm its role within the Rory platform architecture.",
)
def test():
    return {"component_type": "manager"}


@router.api_route(
    "/secure",
    methods=["GET", "POST"],
    response_model=SecureWorkerResponse,
    summary="Load-balanced worker allocation",
    description="""Securely allocate a worker node for distributed computation.

If no workers are available, a new Docker container is automatically deployed via the Summoner.
Otherwise, the configured load-balancing algorithm selects an existing worker.

**GET**: Returns an available worker (auto-deploys if none exist).  
**POST**: Rejected with HTTP 403 (only GET is supported for secure operations).""",
)
def secure(
    request: Request,
    body: SecureWorkerRequest = Depends(),
    logger=Depends(get_logger),
    lb=Depends(get_lb),
    replicator: Summoner = Depends(get_replicator),
    settings=Depends(get_settings),
):
    global sem
    try:
        sem.acquire()
        arrival_time = time.time()
        workers_dict = get_workers()
        active_workers = list(filter(lambda x: x[1].isStarted, workers_dict.items()))
        n_workers = len(workers_dict)
        worker_port = str(n_workers + settings.init_port)

        host_port = body.host_port or worker_port
        container_id = body.container_id or f"worker-{n_workers}"
        container_port = body.container_port or worker_port

        if request.method == "GET":
            if len(active_workers) == 0:
                logger.debug({
                    "event": "NO.WORKER",
                    "algorithm": body.algorithm,
                    "docker_image": settings.docker_image,
                    "docker_network_id": settings.docker_network_id,
                    "init_port": settings.init_port,
                    "host_port": host_port,
                    "container_id": container_id,
                    "container_port": container_port,
                    "worker_memory": body.worker_memory,
                    "worker_cpu": body.worker_cpu,
                    "debug": body.debug,
                    "reload": body.reload,
                    "liu_round": body.liu_round,
                    "sink_path": body.sink_path,
                    "log_path": body.log_path,
                    "testing": body.testing,
                    "max_iterations": body.max_iterations,
                    "m": body.m,
                    "worker_max_threads": body.max_threads,
                    "worker_mictlanx_peers": body.mictlanx_peers,
                    "mictlanx_client_lb_algorithm": body.mictlanx_lb_algorithm,
                    "mictlanx_debug": body.mictlanx_debug,
                    "mictlanx_daemon": body.mictlanx_daemon,
                    "mictlanx_show_metrics": body.mictlanx_show_metrics,
                    "mictlanx_max_workers": body.mictlanx_max_workers,
                    "mictlanx_disabled_log": body.mictlanx_disabled_log,
                })

                result: Result[SummonContainerPayload, Exception] = Utils.deploy_worker(
                    replicator=replicator,
                    node_index=n_workers,
                    container_id=container_id,
                    container_port=container_port,
                    manager_ip_addr=settings.node_id,
                    manager_port=settings.node_port,
                    debug=body.debug,
                    _reload=body.reload,
                    liu_round=body.liu_round,
                    source_path=body.source_path,
                    sink_path=body.sink_path,
                    log_path=body.log_path,
                    max_iterations=body.max_iterations,
                    testing=body.testing,
                    m=body.m,
                    worker_max_threads=body.max_threads,
                    worker_mictlanx_peers=body.mictlanx_peers,
                    mictlanx_client_lb_algorithm=body.mictlanx_lb_algorithm,
                    mictlanx_debug=body.mictlanx_debug,
                    mictlanx_daemon=body.mictlanx_daemon,
                    mictlanx_show_metrics=body.mictlanx_show_metrics,
                    mictlanx_max_workers=body.mictlanx_max_workers,
                    mictlanx_disabled_log=body.mictlanx_disabled_log,
                    docker_image=settings.docker_image,
                    host_port=host_port,
                    worker_memory=body.worker_memory,
                    worker_cpu=body.worker_cpu,
                    docker_network_id=settings.docker_network_id,
                )

                if result.is_err:
                    error = result.unwrap_err()
                    logger.error(str(error))
                    sem.release()
                    raise HTTPException(status_code=500, detail=str(error))

                response = result.unwrap()
                _worker = Worker(
                    workerId=container_id,
                    port=container_port,
                    isStarted=True,
                )
                workers_dict[container_id] = _worker

                service_time = time.time() - arrival_time
                worker_id = response.container_id
                sem.release()

                logger.info({
                    "event": "BALANCING",
                    "service_time": service_time,
                    "algorithm": body.algorithm,
                    "worker_id": worker_id,
                })

                return {
                    "worker_id": worker_id,
                    "worker_port": worker_port,
                    "service_time": service_time,
                }
            else:
                worker_id = lb.balance()
                worker = workers_dict[worker_id]
                worker_port = str(worker.port)
                end_time = time.time()
                service_time = end_time - arrival_time

                logger.info({
                    "event": "BALANCING",
                    "service_time": service_time,
                    "matrix_id": body.matrix_id,
                    "algorithm": body.algorithm,
                    "worker_id": worker_id,
                })

                sem.release()
                return {
                    "worker_id": worker_id,
                    "worker_port": worker_port,
                    "service_time": service_time,
                }
        else:
            sem.release()
            raise HTTPException(status_code=403, detail="Only GET method is supported for secure operations")
    except HTTPException:
        sem.release()
        raise
    except Exception as e:
        sem.release()
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))
