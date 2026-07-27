from pydantic import BaseModel, Field
from typing import Optional


class HealthCheckResponse(BaseModel):
    component_type: str = Field("manager", description="Identifies this node as 'manager'")


class SecureWorkerResponse(BaseModel):
    worker_id: str = Field(description="Identifier of the assigned worker node")
    worker_port: str = Field(description="Network port of the assigned worker")
    service_time: float = Field(description="Time spent processing the request (seconds)")


class DeployWorkerResponse(BaseModel):
    container_id: str = Field(description="The unique identifier of the deployed container")
    port: str = Field(description="The assigned host port for the new service")


class WorkerInfo(BaseModel):
    worker_id: str = Field(alias="workerId", description="Unique identifier of the node")
    port: int = Field(description="Network port assigned to the worker service")
    is_started: bool = Field(alias="isStarted", description="Current operational status of the worker")
