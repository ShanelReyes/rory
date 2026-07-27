from pydantic import BaseModel, Field
from typing import List, Optional


class MLTimingResponse(BaseModel):
    service_time_manager: float = Field(0.0, description="Time spent in Manager interaction (seconds)")
    service_time_worker: float = Field(0.0, description="Time spent in remote computation (seconds)")
    service_time_dataowner: float = Field(0.0, description="Time spent in local I/O and data preparation (seconds)")


class LRTrainResponse(MLTimingResponse):
    worker_id: str = Field("", description="ID of the worker node that processed the task")
    service_time_train: float = Field(0.0, description="Total end-to-end training time (seconds)")
    algorithm: str = Field(description="Algorithm constant (logistic_regression_train)")


class LRPredictResponse(MLTimingResponse):
    label_vector: List[int] = Field(default_factory=list, description="The predicted class for each record")
    algorithm: str = Field(description="Algorithm constant (logistic_regression_predict)")
    worker_id: str = Field("", description="ID of the worker node that processed the task")
    service_time_predict: float = Field(0.0, description="Total end-to-end prediction time (seconds)")


class PPLRTrainResponse(MLTimingResponse):
    algorithm: str = Field(description="Algorithm constant (pplr_train)")
    worker_id: str = Field("", description="ID of the worker node that processed the task")
    epochs: int = Field(0, description="Total epochs executed")
    service_time_train: float = Field(0.0, description="Total end-to-end training time (seconds)")


class PPLRPredictResponse(MLTimingResponse):
    label_vector: List[int] = Field(default_factory=list, description="The predicted class for each record (0 or 1)")
    algorithm: str = Field(description="Algorithm constant (pplr_predict)")
    worker_id: str = Field("", description="ID of the worker node that processed the task")
    service_time_predict: float = Field(0.0, description="Total end-to-end prediction time (seconds)")
