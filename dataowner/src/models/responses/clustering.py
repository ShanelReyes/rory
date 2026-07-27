from pydantic import BaseModel, Field
from typing import Optional, List
from uuid import uuid4


class HealthCheckResponse(BaseModel):
    component_type: str = Field("dataowner", description="Identifies this node as 'dataowner'")


class BaseTimingResponse(BaseModel):
    service_time_manager: float = Field(0.0, description="Time spent coordinating with the Manager (seconds)")
    service_time_worker: float = Field(0.0, description="Time spent during Worker execution (seconds)")
    service_time_dataowner: float = Field(0.0, description="Time spent in local data preparation (seconds)")


class ClusteringResponse(BaseTimingResponse):
    label_vector: List[int] = Field(default_factory=list, description="Cluster assignment for each dataset point")
    algorithm: str = Field(default="", description="Algorithm executed")
    worker_id: str = Field(default="", description="ID of the worker node that processed the task")
    response_time_clustering: float = Field(default=0.0, description="Total end-to-end execution time (seconds)")


class KmeansResponse(ClusteringResponse):
    iterations: int = Field(0, description="Total iterations performed by the algorithm")


class SkmeansResponse(ClusteringResponse):
    iterations: int = Field(0, description="Total iterations performed by the algorithm")


class DbskmeansResponse(ClusteringResponse):
    iterations: int = Field(0, description="Total iterations performed by the algorithm")


class NncResponse(ClusteringResponse):
    pass


class DbsnncResponse(ClusteringResponse):
    pass


class PqcSkmeansResponse(ClusteringResponse):
    iterations: int = Field(0, description="Total iterations performed by the algorithm")


class PqcDbskmeansResponse(ClusteringResponse):
    iterations: int = Field(0, description="Total iterations performed by the algorithm")
