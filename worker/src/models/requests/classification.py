from pydantic import BaseModel, Field
from typing import Optional

from .clustering import BaseRequest


class KnnPredictWorkerRequest(BaseRequest):
    model_id: str = Field(default="model0", description="Model identifier in CSS")
    records_test_id: str = Field(default="matrix0", description="Test records identifier")
    model_labels_shape: Optional[str] = Field(default=None, description="Shape of the model labels (required)")
    experiment_id: Optional[str] = Field(default=None, description="Experiment ID")


class SknnPredictWorkerRequest(BaseRequest):
    step_index: int = Field(default=1, ge=1, le=2, description="Protocol step")
    model_id: str = Field(default="model0", description="Model identifier")
    records_test_id: str = Field(default="matrix0", description="Test records identifier")
    encrypted_model_shape: Optional[str] = Field(default=None, description="Encrypted model shape (required step 1)")
    encrypted_model_dtype: Optional[str] = Field(default=None, description="Encrypted model data type (required step 1)")
    encrypted_records_shape: Optional[str] = Field(default=None, description="Encrypted records shape (required step 1)")
    encrypted_records_dtype: Optional[str] = Field(default=None, description="Encrypted records data type (required step 1)")
    num_chunks: Optional[str] = Field(default=None, description="Number of chunks (required)")
    model_labels_shape: Optional[str] = Field(default=None, description="Model labels shape (required step 2)")
    min_distances_index_id: Optional[str] = Field(default=None, description="Min distances index ID (step 2)")


class PqcSknnPredictWorkerRequest(SknnPredictWorkerRequest):
    pass
