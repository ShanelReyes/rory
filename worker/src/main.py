from contextlib import asynccontextmanager
from threading import Thread
import requests
from retry.api import retry_call
from fastapi import FastAPI

from dependencies import (
    get_settings,
    LOGGER,
    ASYNC_STORAGE_CLIENT,
    CKKS,
    _settings,
)
from routes.clustering import router as clustering_router
from routes.classification import router as classification_router
from routes.machinelearning import router as machinelearning_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        LOGGER.debug({
        "event": "WORKER_STARTED",
        "node_id": _settings.node_id,
        "port": _settings.node_port,
        "debug": _settings.debug,
        "mictlanx_max_workers": _settings.mictlanx_max_workers,
        "mictlanx_timeout": _settings.mictlanx_timeout,
        "log_disabled": _settings.rory_worker_log_disabled,
        "mictlanx_log_disable": _settings.mictlanx_log_disable,
    })

    except Exception as e:
        LOGGER.error({"event": "WORKER_STARTUP_ERROR", "msg": str(e)})
        print(f"WORKER STARTUP ERROR: {e}", flush=True)
        raise

    t1 = Thread(target=started_completed, daemon=True, args=())
    t1.start()

    yield
    LOGGER.info({"event": "WORKER_SHUTDOWN"})


app = FastAPI(
    title="Rory Worker API",
    description="""Privacy-preserving computation worker for the Rory platform.

Executes clustering, classification, and machine learning algorithms on encrypted data
using homomorphic encryption schemes (Liu, CKKS) with distributed storage via MictlanX.

## Components

- **Clustering**: K-Means, Secure K-Means, Double-Blind K-Means, NNC, DBSNNC (plaintext and PQC variants)
- **Classification**: KNN, Secure KNN, PQC KNN (predict only)
- **Machine Learning**: Logistic Regression, Privacy-Preserving Logistic Regression (train and predict)
""",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Clustering", "description": "Clustering algorithm execution (K-Means, SK-Means, NNC, DBSNNC)"},
        {"name": "Classification", "description": "Classification algorithm execution (KNN, SKNN, PQC-SKNN)"},
        {"name": "Machine Learning", "description": "ML algorithm execution (Logistic Regression, PPLR)"},
    ],
)

app.include_router(clustering_router)
app.include_router(classification_router)
app.include_router(machinelearning_router)


def started_completed():
    def __inner():
        url = f"http://{_settings.rory_manager_ip_addr}:{_settings.rory_manager_port}/workers/started"
        LOGGER.debug({
            "event": "MANAGER.STARTED_STARTED",
            "manager_ip_addr": _settings.rory_manager_ip_addr,
            "manager_port": _settings.rory_manager_port,
            "node_id": _settings.node_id,
            "port": _settings.node_port,
        })
        response = requests.post(
            url,
            json={"worker_id": _settings.node_id, "worker_port": _settings.node_port},
            timeout=300,
        )
        response.raise_for_status()
        LOGGER.debug({
            "event": "MANAGER.STARTED_COMPLETED",
            "manager_ip_addr": _settings.rory_manager_ip_addr,
            "manager_port": _settings.rory_manager_port,
            "node_id": _settings.node_id,
            "port": _settings.node_port,
        })
        return response

    retry_call(__inner, tries=_settings.max_retries, delay=1, backoff=1)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=_settings.server_ip_addr,
        port=_settings.node_port,
        reload=_settings.reload_flag,
        log_level="debug" if _settings.debug else "info",
        timeout_keep_alive=3600,
        workers=1,
    )
