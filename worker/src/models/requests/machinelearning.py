from pydantic import BaseModel, Field
from typing import Optional

from .clustering import BaseRequest


class LRTrainWorkerRequest(BaseRequest):
    plaintext_matrix_train_id: str = Field(default="train_x", description="Training matrix identifier")
    plaintext_label_vector_train_id: str = Field(default="train_y", description="Label vector identifier")
    weights_id: Optional[str] = Field(default=None, description="Weights identifier (required)")
    bias_id: Optional[str] = Field(default=None, description="Bias identifier (required)")
    epochs: Optional[str] = Field(default="1", description="Number of epochs")
    learning_rate: Optional[str] = Field(default="0.01", description="Learning rate")


class LRPredictWorkerRequest(BaseRequest):
    plaintext_matrix_train_id: str = Field(default="train_x", description="Training matrix identifier")
    plaintext_matrix_test_id: str = Field(default="test_x", description="Test matrix identifier")
    weights_id: Optional[str] = Field(default=None, description="Weights identifier (required)")
    bias_id: Optional[str] = Field(default=None, description="Bias identifier (required)")


class PPLRTrainWorkerRequest(BaseRequest):
    learning_rate: str = Field(default="0.01", description="Learning rate")
    encrypted_matrix_train_id: Optional[str] = Field(default=None, description="Encrypted train matrix ID (required)")
    encrypted_label_vector_train_id: Optional[str] = Field(default=None, description="Encrypted labels ID (required)")
    encrypted_weights_id: Optional[str] = Field(default=None, description="Encrypted weights ID (required)")
    encrypted_bias_id: Optional[str] = Field(default=None, description="Encrypted bias ID (required)")
    scale: str = Field(default="40", description="CKKS scale parameter")
    n_features: str = Field(default="0", description="Number of features")
    n_samples: str = Field(default="0", description="Number of samples")
    num_chunks: str = Field(default="1", description="Number of chunks")


class PPLRPredictWorkerRequest(BaseRequest):
    encrypted_matrix_test_id: Optional[str] = Field(default=None, description="Encrypted test matrix ID (required)")
    encrypted_weights_id: Optional[str] = Field(default=None, description="Encrypted weights ID (required)")
    encrypted_bias_id: Optional[str] = Field(default=None, description="Encrypted bias ID (required)")
    scale: str = Field(default="40", description="CKKS scale parameter")
    n_features: str = Field(default="0", description="Number of features")
    num_chunks: str = Field(default="1", description="Number of chunks")
