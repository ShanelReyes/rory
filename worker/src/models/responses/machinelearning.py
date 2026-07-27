from pydantic import BaseModel, Field
from typing import Optional


class LRTrainResponse(BaseModel):
    service_time: float = Field(0.0)
    train_time: float = Field(0.0)
    algorithm: str = Field("")


class LRPredictResponse(BaseModel):
    predictions_id: str = Field("")
    predict_time: float = Field(0.0)
    service_time: float = Field(0.0)


class PPLRTrainResponse(BaseModel):
    service_time: float = Field(0.0)
    train_time: float = Field(0.0)
    algorithm: str = Field("")


class PPLRPredictResponse(BaseModel):
    encrypted_predictions_id: str = Field("")
    predict_time: float = Field(0.0)
    service_time: float = Field(0.0)
    algorithm: str = Field("")
