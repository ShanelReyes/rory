from typing import List,Optional,Dict,Any
import uuid
from pydantic import BaseModel, Field
class EncryptDTO(BaseModel):
    bucket_id: str
    ball_id:str
    chunk_indexes:List[int]
    algorithm:str
    security_level: Optional[int] = 128
class KMeansDTO(BaseModel):
    bucket_id: str
    ball_id: str
    chunk_indexes: List[int]
    n_clusters: int
    random_state: Optional[int] = None
  
class SegementResponseDTO(BaseModel):
    task_id:str
    message: Any
    ball_id:str
    bucket_id:str
class SegmentDTO(BaseModel):
    bucket_id: str
    ball_id: str
    filename: str
    max_attempts:Optional[int] = 10
    max_backoff: Optional[int] = 5
    row_chunk_size: Optional[int] = 100
    tags: Optional[Dict[str,str]] = {}
    timeout:Optional[int]  = 120

class EncryptionPolicy(BaseModel):
    algorithm: str = Field(default="LIU")
    security_level: int = Field(default=128)


class ExecutionPolicy(BaseModel):
    mode: str = Field(default="DISTRIBUTED")
    workers: int = Field(default=1)
    cpu:int = Field(default=1)
    memory:int = Field(default=1000000000)
    base_port:int = Field(default=45000)
    # chunk_size: str = Field(default="256kb")


class DataHandlingPolicy(BaseModel):
    storage_strategy: str = Field(default="DISTRIBUTED")
    chunk_indexes: List[int]
    compression: bool = Field(default=False)


class ResiliencePolicy(BaseModel):
    fault_tolerance: bool = Field(default=True)
    max_attempts: int = Field(default=10)


class AuditPolicy(BaseModel):
    level: str = Field(default="DEBUG")
    metrics_collection: bool = Field(default=True)


class Policy(BaseModel):
    policy_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    encryption: EncryptionPolicy
    execution: ExecutionPolicy
    data_handling: DataHandlingPolicy
    resilience: ResiliencePolicy
    audit: AuditPolicy

    filename: str
    bucket_id: str
    ball_id: str

class KnnTrainResponse(BaseModel):
    response_time:float
    algorithm:str
    model_labels_shape:List[int]

class KnnPredictResponse(BaseModel):
    label_vector:List[int]
    algorithm:str
    worker_id:str
    service_time_manager:float
    service_time_worker:float
    service_time_client:float
    service_time_predict:float

class KnnResponse(BaseModel):
    algorithm: str
    label_vector:List[int]
    worker_id:str
    service_time_manager:float
    service_time_worker:float
    service_time_client:float
    service_time_predict:float
    service_time_train: float


class SknnTrainResponse(BaseModel):
    response_time:float
    encrypted_model_shape:str
    encrypted_model_dtype:str
    algorithm:str
    model_labels_shape:List[int]


class KmeansResponse(BaseModel):
    label_vector:List[int]
    iterations:int
    algorithm:str
    worker_id:str
    service_time_manager:float
    service_time_worker:float
    service_time_client:float
    response_time_clustering:float


class NncResponse(BaseModel):
    label_vector:List[int]
    algorithm:str
    worker_id:str
    service_time_manager:float
    service_time_worker:float
    service_time_client:float
    response_time_clustering:float


class LogisticRegressionTrainResponse(BaseModel):
    algorithm: str
    worker_id: str
    service_time_manager: float
    service_time_worker: float
    service_time_client: float
    service_time_train: float


class LogisticRegressionPredictResponse(BaseModel):
    label_vector: List[int]
    algorithm: str
    worker_id: str
    service_time_manager: float
    service_time_worker: float
    service_time_client: float
    service_time_predict: float


class LogisticRegressionResponse(BaseModel):
    algorithm: str
    label_vector: List[int]
    worker_id: str
    service_time_manager: float
    service_time_worker: float
    service_time_client: float
    service_time_predict: float
    service_time_train: float


class PPLRTrainResponse(BaseModel):
    algorithm: str
    epochs: int
    worker_id: str
    service_time_manager: float
    service_time_worker: float
    service_time_client: float
    service_time_train: float


class PPLRPredictResponse(BaseModel):
    label_vector: List[int]
    algorithm: str
    worker_id: str
    service_time_manager: float
    service_time_worker: float
    service_time_client: float
    service_time_predict: float


class PPLRResponse(BaseModel):
    algorithm: str
    label_vector: List[int]
    worker_id: str
    epochs: int
    service_time_manager: float
    service_time_worker: float
    service_time_client: float
    service_time_predict: float
    service_time_train: float