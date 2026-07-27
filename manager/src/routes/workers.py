import time
from threading import Lock
from fastapi import APIRouter, Depends, HTTPException, Request
from rory.core.interfaces.worker import Worker
from rory.core.interfaces.logger_metrics import LoggerMetrics
from mictlanx.services.summoner.summoner import Summoner, SummonContainerPayload
from utils.utils import Utils
from option import Result

from models.requests import WorkerStartedRequest, DeployWorkerRequest
from models.responses import DeployWorkerResponse, WorkerInfo

from dependencies import get_logger, get_replicator, get_workers, get_settings

lock = Lock()
router = APIRouter(prefix="/workers", tags=["Workers"])


@router.api_route(
    "/started",
    methods=["GET", "POST"],
    status_code=204,
    summary="Register a new worker node",
    description="""Thread-safe registration of a new Worker node in the Manager's global registry.

The worker calls this endpoint on startup to register itself.
**POST**: Registers the worker, returns 204 No Content.  
**GET**: Returns 404 Not Found.""",
)
async def started(
    request: Request,
    logger=Depends(get_logger),
):
    if request.method != "POST":
        raise HTTPException(status_code=404)

    body = await request.json()
    from pydantic import ValidationError
    try:
        req = WorkerStartedRequest.model_validate(body)
    except ValidationError:
        raise HTTPException(status_code=422, detail="Invalid request body")

    lock.acquire()
    try:
        arrivalTime = time.time()
        _worker = Worker(
            workerId=req.worker_id,
            port=req.worker_port,
            isStarted=True,
        )
        workers_dict = get_workers()
        workers_dict[req.worker_id] = _worker

        end_time = time.time()
        service_time = end_time - arrivalTime

        logger.info({
            "event": "WORKER.STARTED",
            "worker_id": req.worker_id,
            "num_workers": len(workers_dict),
            "service_time": service_time,
        })
    finally:
        lock.release()

    return None


@router.get(
    "",
    summary="List all registered workers",
    description="""Retrieve a complete snapshot of all Worker nodes currently registered and active.

Returns metadata for the entire distributed worker pool including worker IDs, ports, and operational status.""",
)
def get_all(
    logger=Depends(get_logger),
):
    arrivalTime = time.time()
    workers_dict = get_workers()
    logger.debug(str(workers_dict))
    result = {k: WorkerInfo(workerId=v.workerId, port=v.port, isStarted=v.isStarted).model_dump(by_alias=True) for k, v in workers_dict.items()}

    end_time = time.time()
    service_time = end_time - arrivalTime
    logger_metrics = LoggerMetrics(
        operation_type="GET_ALL_WORKERS",
        arrival_time=arrivalTime,
        end_time=end_time,
        service_time=service_time,
    )
    logger.info(str(logger_metrics))
    return result


@router.post(
    "/deploy",
    response_model=DeployWorkerResponse,
    summary="Manual on-demand worker deployment",
    description="""Orchestrates the dynamic deployment of a new Worker container.

Facilitates on-demand scaling for privacy-preserving computations by spawning a containerized 
node with configured network identity, resource limits, and environment variables.""",
)
def deploy_worker(
    body: DeployWorkerRequest = Depends(),
    replicator: Summoner = Depends(get_replicator),
    logger=Depends(get_logger),
    settings=Depends(get_settings),
):
    workers_dict = get_workers()
    n_workers = len(workers_dict)
    worker_port = str(n_workers + settings.init_port)

    host_port = body.host_port or worker_port
    container_id = body.container_id or f"worker-{n_workers}"
    container_port = body.container_port or worker_port

    envs = {
        "NODE_INDEX": str(n_workers),
        "NODE_IP_ADDR": container_id,
        "NODE_PORT": container_port,
        "RORY_MANAGER_IP_ADDR": settings.node_id,
        "RORY_MANAGER_PORT": str(settings.node_port),
        "DEBUG": body.debug,
        "RELOAD": body.reload,
        "LIU_ROUND": body.liu_round,
        "SOURCE_PATH": body.source_path,
        "SINK_PATH": body.sink_path,
        "LOG_PATH": body.log_path,
        "MAX_ITERATIONS": body.max_iterations,
        "TESTING": body.testing,
        "M": body.m,
        "MAX_THREADS": body.max_threads,
        "MICTLANX_PEERS": body.mictlanx_peers,
        "MICTLANX_CLIENT_LB_ALGORITHM": body.mictlanx_lb_algorithm,
        "MICTLANX_DEBUG": body.mictlanx_debug,
        "MICTLANX_DAEMON": body.mictlanx_daemon,
        "MICTLANX_SHOW_METRICS": body.mictlanx_show_metrics,
        "MICTLANX_MAX_WORKERS": body.mictlanx_max_workers,
        "MICTLANX_DISABLED_LOG": body.mictlanx_disabled_log,
    }

    logger.debug({
        "event": "WORKER.DEPLOY.ENVS",
        **envs,
    })

    try:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if result.is_err:
        raise HTTPException(status_code=500, detail=str(result.unwrap_err()))

    response = result.unwrap()
    logger.info({
        "event": "WORKER.DEPLOY",
        "container_id": response.container_id,
        "cpu_count": response.cpu_count,
        "memory": response.memory,
    })

    return {
        "container_id": container_id,
        "port": worker_port,
    }
