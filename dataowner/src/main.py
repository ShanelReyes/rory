from contextlib import asynccontextmanager
from fastapi import FastAPI

from dependencies import (
    get_settings,
    get_logger,
    get_manager,
    get_liu,
    get_dataowner,
    get_executor,
    get_ckks,
    get_storage_client,
    ASYNC_STORAGE_CLIENT,
    EXECUTOR,
    CKKS,
    DATAOWNER,
    LIU,
    LOGGER,
    MANAGER,
    _settings,
)
from routes.clustering import router as clustering_router
from routes.classification import router as classification_router
from routes.machinelearning import router as machinelearning_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        LOGGER.debug({
        "event": "DATAOWNER_STARTED",
        "source_path": _settings.source_path,
        "sink_path": _settings.sink_path,
        "node_id": _settings.node_id,
        "log_path": _settings.log_path,
        "debug": _settings.debug,
        "max_iterations": _settings.max_iterations,
        "testing": _settings.testing,
        "num_chunks": _settings.num_chunks,
        "mictlanx_timeout": _settings.mictlanx_timeout,
        "worker_timeout": _settings.worker_timeout,
        "log_disabled": _settings.rory_dataowner_log_disabled,
        "mictlanx_log_disable": _settings.mictlanx_log_disable,
        "liu_params": {
            "security_level": _settings.liu_security_level,
            "secure_random": _settings.liu_secure_random,
            "seed": _settings.liu_seed,
            "use_np_random": _settings.liu_use_np_random,
            "round": _settings.liu_round,
            "decimals": _settings.liu_decimals,
        },
        })
    except Exception as e:
        LOGGER.error({"event": "DATAOWNER_STARTUP_ERROR", "msg": str(e)})
        print(f"DATAOWNER STARTUP ERROR: {e}", flush=True)
        raise

    yield
    EXECUTOR.shutdown()
    LOGGER.info({"event": "DATAOWNER_SHUTDOWN"})


app = FastAPI(
    title="Rory Dataowner API",
    description="""Privacy-preserving clustering, classification, and machine learning API.

Part of the Rory platform architecture for secure distributed computation using homomorphic encryption (Liu, CKKS) and Cloud Storage System (MictlanX).

## Components

- **Clustering**: K-Means, Secure K-Means, Double-Blind K-Means, NNC, DBSNNC (plaintext and PQC variants)
- **Classification**: KNN, Secure KNN, PQC KNN (train and predict)
- **Machine Learning**: Logistic Regression, Privacy-Preserving Logistic Regression (train and predict)
""",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Clustering", "description": "Privacy-preserving clustering algorithms (K-Means, SK-Means, NNC, DBSNNC)"},
        {"name": "Classification", "description": "Privacy-preserving classification algorithms (KNN, SKNN, PQC-SKNN)"},
        {"name": "Machine Learning", "description": "Privacy-preserving machine learning algorithms (Logistic Regression, PPLR)"},
    ],
)

app.include_router(clustering_router)
app.include_router(classification_router)
app.include_router(machinelearning_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=_settings.server_ip_addr,
        port=_settings.node_port,
        reload=_settings.reload,
        log_level="debug" if _settings.debug else "info",
        timeout_keep_alive=3600,
        workers=1,
    )
