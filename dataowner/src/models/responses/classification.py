from pydantic import BaseModel, Field
from typing import List, Optional


class TrainResponse(BaseModel):
    response_time: float = Field(0.0, description="Total execution time for the preparation phase (seconds)")


class KnnTrainResponse(TrainResponse):
    algorithm: str = Field(description="The specific algorithm constant (knn_train)")
    model_labels_shape: List[int] = Field(default_factory=list, description="Dimensions of the uploaded labels matrix")


class SknnTrainResponse(TrainResponse):
    encrypted_model_shape: str = Field("", description="The 3D shape of the Liu-encrypted matrix (r, a, m)")
    encrypted_model_dtype: str = Field("float32", description="Data type of the encrypted matrix")
    algorithm: str = Field(description="The specific constant for sknn_train")
    model_labels_shape: List[int] = Field(default_factory=list, description="Dimensions of the uploaded labels matrix")


class PqcSknnTrainResponse(TrainResponse):
    encrypted_model_shape: str = Field("", description="The dimensions of the CKKS-encrypted matrix")
    encrypted_model_dtype: str = Field("float32", description="Data type (float32)")
    algorithm: str = Field(description="The SKNN_PQC_TRAIN constant")
    model_labels_shape: List[int] = Field(default_factory=list, description="Dimensions of the uploaded labels matrix")


class PredictResponse(BaseModel):
    label_vector: List[int] = Field(default_factory=list, description="The predicted class for each record")
    algorithm: str = Field(description="The specific algorithm constant")
    worker_id: str = Field("", description="ID of the worker node that processed the task")
    service_time_manager: float = Field(0.0, description="Time spent in Manager interaction (seconds)")
    service_time_worker: float = Field(0.0, description="Time spent in remote computation (seconds)")
    service_time_dataowner: float = Field(0.0, description="Time spent in local I/O and data preparation (seconds)")
    service_time_predict: float = Field(0.0, description="Total end-to-end prediction time (seconds)")
