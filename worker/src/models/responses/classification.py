from pydantic import BaseModel, Field
from typing import List


class KnnPredictResponse(BaseModel):
    label_vector: List[int] = Field(default_factory=list)
    service_time: float = Field(0.0)


class SknnPredictStep1Response(BaseModel):
    distances_id: str = Field("")
    distances_shape: str = Field("")
    distances_dtype: str = Field("")
    service_time: float = Field(0.0)


class SknnPredictStep2Response(BaseModel):
    label_vector: List[int] = Field(default_factory=list)
    service_time: float = Field(0.0)
