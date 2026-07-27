from pydantic import BaseModel, Field
from typing import List, Optional


class HealthCheckResponse(BaseModel):
    component_type: str = Field("worker", description="Identifies this node as 'worker'")


class WorkerRun1Response(BaseModel):
    label_vector: List[int] = Field(default_factory=list)
    service_time: float = Field(0.0)
    n_iterations: int = Field(0)
    encrypted_shift_matrix_id: str = Field("")


class WorkerRun2Response(BaseModel):
    pass


class WorkerDbsnncResponse(BaseModel):
    label_vector: List[int] = Field(default_factory=list)
    service_time: float = Field(0.0)


class WorkerNncResponse(BaseModel):
    label_vector: List[int] = Field(default_factory=list)
    service_time: float = Field(0.0)
