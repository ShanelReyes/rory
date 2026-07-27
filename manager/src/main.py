import sys
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI

from config import Settings
from dependencies import (
    get_settings,
    LOGGER,
    REPLICATOR,
    LB,
    WORKERS,
    _settings,
)
from routes.clustering import router as clustering_router
from routes.workers import router as workers_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        LOGGER.debug({
            "event": "MANAGER_STARTED",
            "load_balancing_algorithm": _settings.load_balancing,
            "node_id": _settings.node_id,
            "port": _settings.node_port,
            "node_prefix": _settings.node_prefix,
            "debug": _settings.debug,
            "docker_image_name": _settings.docker_image_name,
            "docker_image_tag": _settings.docker_image_tag,
            "docker_image": _settings.docker_image,
            "docker_network_id": _settings.docker_network_id,
            "worker_timeout": _settings.worker_timeout,
            "worker_memory": _settings.worker_memory,
            "worker_cpu": _settings.worker_cpu,
            "worker_max_threads": _settings.worker_max_threads,
            "mictlanx_max_workers": _settings.mictlanx_max_workers,
            "mictlanx_timeout": _settings.mictlanx_timeout,
        })

        if _settings.init_workers > 0:
            deploy_workers_start_time = time.time()
            from deployworkers import deploy_nodes
            LOGGER.debug({
                "event": "DEPLOY_NODES",
                "node_id": _settings.node_id,
                "port": _settings.node_port,
                "init_workers": _settings.init_workers,
                "worker_memory": _settings.worker_memory,
                "worker_cpu": _settings.worker_cpu,
                "init_port": _settings.init_port,
                "docker_image": _settings.docker_image,
                "mictlanx_uri": _settings.mictlanx_uri,
                "swarm_nodes": ",".join(_settings.swarm_nodes),
            })
            deploy_nodes_result = deploy_nodes(
                log=LOGGER,
                summoner=REPLICATOR,
                NODE_ID=_settings.node_id,
                PORT=str(_settings.node_port),
                WORKER_MAX_THREADS=_settings.worker_max_threads,
                DOCKER_IMAGE=_settings.docker_image,
                DOCKER_NETWORK_ID=_settings.docker_network_id,
                MICTLANX_CLIENT_ID=_settings.mictlanx_client_id,
                MICTLANX_SUMMONER_MODE=_settings.mictlanx_summoner_mode,
                init_workers=_settings.init_workers,
                NODE_PREFIX=_settings.node_prefix,
                FOLDER_KEYS=_settings.folder_keys,
                init_port=_settings.init_port,
                WORKER_MEMORY=_settings.worker_memory,
                WORKER_CPU=_settings.worker_cpu,
                WORKER_MICTLANX_URI=_settings.mictlanx_uri,
                MICTLANX_DEBUG=_settings.mictlanx_debug,
                MICTLANX_MAX_WORKERS=_settings.mictlanx_max_workers,
                swarm_nodes=_settings.swarm_nodes,
                SERVER_IP_ADDR=_settings.server_ip_addr,
                MAX_RETRIES=_settings.max_retries,
                DISTANCE=_settings.distance,
                MIN_ERROR=_settings.min_error,
                CKKS_ROUND=_settings.ckks_round,
                CKKS_DECIMALS=_settings.ckks_decimals,
                CTX_FILENAME=_settings.ctx_filename,
                PUBKEY_FILENAME=_settings.pubkey_filename,
                SECRET_KEY_FILENAME=_settings.secret_key_filename,
                RELINKEY_FILENAME=_settings.relinkey_filename,
                MICTLANX_TIMEOUT=_settings.mictlanx_timeout,
                MICTLANX_API_VERSION=_settings.mictlanx_api_version,
                MICTLANX_PROTOCOL=_settings.mictlanx_protocol,
                MICTLANX_LOG_PATH=_settings.mictlanx_log_path,
                MICTLANX_LOG_INTERVAL=_settings.mictlanx_log_interval,
                MICTLANX_LOG_WHEN=_settings.mictlanx_log_when,
                MICTLANX_BUCKET_ID=_settings.mictlanx_bucket_id,
                MICTLANX_DELAY=_settings.mictlanx_delay,
                MICTLANX_BACKOFF_FACTOR=_settings.mictlanx_backoff_factor,
                MICTLANX_MAX_RETRIES=_settings.mictlanx_max_retries,
                MICTLANX_CHUNK_SIZE=_settings.mictlanx_chunk_size,
                MICTLANX_MAX_PARALELL_GETS=_settings.mictlanx_max_parallel_gets,
                LIU_ROUND=_settings.liu_round,
            )
            if deploy_nodes_result.is_err:
                LOGGER.error({"msg": str(deploy_nodes_result.unwrap_err())})
                sys.exit(1)
            else:
                LOGGER.info({
                    "event": "DEPLOY_NODES",
                    "node_id": _settings.node_id,
                    "port": _settings.node_port,
                    "init_workers": _settings.init_workers,
                    "worker_memory": _settings.worker_memory,
                    "worker_cpu": _settings.worker_cpu,
                    "folder_keys": _settings.folder_keys,
                    "init_port": _settings.init_port,
                    "docker_image": _settings.docker_image,
                    "peers": _settings.mictlanx_uri,
                    "swarm_nodes": ",".join(_settings.swarm_nodes),
                    "service_time": time.time() - deploy_workers_start_time,
                })

    except Exception as e:
        LOGGER.error({"event": "MANAGER_STARTUP_ERROR", "msg": str(e)})
        print(f"MANAGER STARTUP ERROR: {e}", flush=True)
        raise

    yield
    LOGGER.info({"event": "MANAGER_SHUTDOWN"})


app = FastAPI(
    title="Rory Manager API",
    description="""Orchestration and worker management for the Rory privacy-preserving platform.

## Components

- **Clustering**: Secure worker allocation with load balancing and auto-deployment
- **Workers**: Worker registration, manual deployment, and worker pool inspection

The Manager is responsible for distributing computation tasks across worker nodes
using configurable load-balancing strategies (RoundRobin, TwoChoices, Random).
""",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Clustering", "description": "Secure worker allocation and load balancing endpoints"},
        {"name": "Workers", "description": "Worker registration, deployment, and pool management"},
    ],
)

app.include_router(clustering_router)
app.include_router(workers_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=_settings.server_ip_addr,
        port=_settings.node_port,
        reload=_settings.debug,
        log_level="debug" if _settings.debug else "info",
        timeout_keep_alive=3600,
        workers=1,
    )
