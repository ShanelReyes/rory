from pydantic import BaseModel, Field
from typing import Optional


class WorkerDeploymentConfig(BaseModel):
    """Shared deployment configuration for secure worker allocation and manual deploy endpoints."""

    host_port: Optional[str] = Field(
        default=None,
        description="Host port for the worker container. Auto-computed from n_workers + INIT_PORT if not provided.",
    )
    container_id: Optional[str] = Field(
        default=None,
        description="Docker container identifier. Auto-generated as 'worker-{n_workers}' if not provided.",
    )
    container_port: Optional[str] = Field(
        default=None,
        description="Internal container port. Auto-computed from n_workers + INIT_PORT if not provided.",
    )
    worker_memory: str = Field(
        default="1000000000",
        description="Memory resource limit for the container in bytes",
    )
    worker_cpu: str = Field(
        default="1",
        description="CPU resource quota assigned to the worker",
    )
    debug: str = Field(default="0", description="Debug mode flag for the worker")
    reload: str = Field(default="0", description="Reload flag for the worker service")
    liu_round: str = Field(default="1", description="Liu encryption rounding parameter")
    sink_path: str = Field(default="/rory/sink", description="Sink path for worker output data")
    source_path: str = Field(default="/rory/source", description="Source path for worker input data")
    log_path: str = Field(default="/rory/log", description="Log path for worker logs")
    testing: str = Field(default="0", description="Testing mode flag")
    max_iterations: str = Field(default="10", description="Maximum iterations for iterative algorithms")
    m: str = Field(default="3", description="Security parameter m for homomorphic encryption")
    max_threads: str = Field(default="4", description="Maximum concurrent threads for worker processing")
    mictlanx_peers: str = Field(
        default="mictlanx-router-0:localhost:60666",
        description="MictlanX peer addresses for distributed storage connectivity",
    )
    mictlanx_lb_algorithm: str = Field(default="2CHOICES_UF", description="MictlanX client load balancing algorithm")
    mictlanx_debug: str = Field(default="0", description="MictlanX debug flag")
    mictlanx_daemon: str = Field(default="0", description="MictlanX daemon flag")
    mictlanx_show_metrics: str = Field(default="0", description="MictlanX metrics display flag")
    mictlanx_max_workers: str = Field(default="4", description="MictlanX maximum workers")
    mictlanx_disabled_log: str = Field(default="1", description="MictlanX log disabling flag")


class SecureWorkerRequest(WorkerDeploymentConfig):
    """Parameters for the secure worker load-balancing endpoint (GET /clustering/secure)."""

    algorithm: Optional[str] = Field(
        default=None,
        description="Algorithm name for the worker (used for logging and routing)",
    )
    start_request_time: str = Field(
        default="0",
        description="Timestamp when the original request started",
    )
    get_worker_start_time: str = Field(
        default="0",
        description="Timestamp when the getWorker call was initiated",
    )
    matrix_id: str = Field(
        default="matrix0",
        description="Matrix identifier associated with the request",
    )


class DeployWorkerRequest(WorkerDeploymentConfig):
    """Parameters for manual worker deployment (POST /workers/deploy). Overrides some defaults."""

    sink_path: str = Field(default="/sink", description="Sink path for worker output data")
    source_path: str = Field(default="/source", description="Source path for worker input data")
    log_path: str = Field(default="/log", description="Log path for worker logs")
    mictlanx_peers: str = Field(
        default="mictlanx-peer-0:mictlanx-peer-0:7000",
        description="MictlanX peer addresses for distributed storage connectivity",
    )


class WorkerStartedRequest(BaseModel):
    """Parameters for worker registration (POST /workers/started). Both fields are mandatory."""

    worker_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the registering worker node",
    )
    worker_port: int = Field(
        ...,
        gt=0,
        description="The network port where the worker service is listening for instructions",
    )
