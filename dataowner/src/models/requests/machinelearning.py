from pydantic import BaseModel, Field
from typing import Optional

from .clustering import BaseRequest


class LRTrainRequest(BaseRequest):
    """Parameters for plaintext Logistic Regression training."""
    plaintext_matrix_train_id: str = Field(
        default="train_x",
        description="Unique identifier for the training matrix in Cloud Storage System"
    )
    plaintext_label_vector_train_id: str = Field(
        default="train_y",
        description="Unique identifier for the training label vector in Cloud Storage System"
    )
    plaintext_matrix_train_filename: Optional[str] = Field(
        default=None,
        description="Local filename for the training matrix (without extension). Defaults to matrix_train_id"
    )
    plaintext_label_vector_train_filename: Optional[str] = Field(
        default=None,
        description="Local filename for the label vector (without extension). Defaults to label_vector_train_id"
    )
    extension: str = Field(
        default="csv",
        description="Dataset file extension (e.g., csv, npy)"
    )
    epochs: int = Field(
        default=1,
        ge=1,
        le=1000,
        description="Number of training epochs"
    )
    learning_rate: float = Field(
        default=0.01,
        gt=0,
        le=1.0,
        description="Learning rate for gradient descent"
    )

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if self.plaintext_matrix_train_filename is None:
            self.plaintext_matrix_train_filename = self.plaintext_matrix_train_id
        if self.plaintext_label_vector_train_filename is None:
            self.plaintext_label_vector_train_filename = self.plaintext_label_vector_train_id


class LRPredictRequest(BaseRequest):
    """Parameters for plaintext Logistic Regression prediction."""
    plaintext_matrix_train_id: str = Field(
        default="train_x",
        description="ID of the training matrix used during training"
    )
    plaintext_matrix_test_id: str = Field(
        default="test_x",
        description="Unique ID for the test records to be stored in Cloud Storage System"
    )
    plaintext_matrix_test_filename: Optional[str] = Field(
        default=None,
        description="Local filename for the test dataset. Defaults to matrix_test_id"
    )
    extension: str = Field(
        default="csv",
        description="Dataset file extension (e.g., csv, npy)"
    )

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if self.plaintext_matrix_test_filename is None:
            self.plaintext_matrix_test_filename = self.plaintext_matrix_test_id


class PPLRTrainRequest(BaseRequest):
    """Parameters for Privacy-Preserving Logistic Regression training (CKKS encryption)."""
    plaintext_matrix_train_id: str = Field(
        default="train_x",
        description="Unique identifier for the training matrix"
    )
    plaintext_label_vector_train_id: str = Field(
        default="train_y",
        description="Unique identifier for the label vector"
    )
    plaintext_matrix_train_filename: Optional[str] = Field(
        default=None,
        description="Local filename for the training matrix. Defaults to matrix_train_id"
    )
    plaintext_label_vector_train_filename: Optional[str] = Field(
        default=None,
        description="Local filename for the label vector. Defaults to label_vector_train_id"
    )
    extension: str = Field(
        default="csv",
        description="Dataset file extension"
    )
    epochs: int = Field(
        default=1,
        ge=1,
        le=1000,
        description="Number of training epochs"
    )
    learning_rate: float = Field(
        default=0.01,
        gt=0,
        le=1.0,
        description="Learning rate for gradient descent"
    )

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if self.plaintext_matrix_train_filename is None:
            self.plaintext_matrix_train_filename = self.plaintext_matrix_train_id
        if self.plaintext_label_vector_train_filename is None:
            self.plaintext_label_vector_train_filename = self.plaintext_label_vector_train_id


class PPLRPredictRequest(BaseRequest):
    """Parameters for Privacy-Preserving Logistic Regression prediction (CKKS)."""
    plaintext_matrix_test_id: str = Field(
        default="test_x",
        description="Unique ID for the test records"
    )
    plaintext_matrix_test_filename: Optional[str] = Field(
        default=None,
        description="Local filename for the test dataset. Defaults to matrix_test_id"
    )
    plaintext_matrix_train_id: str = Field(
        default="train_x",
        description="ID of the training matrix used during training"
    )
    extension: str = Field(
        default="csv",
        description="Dataset file extension"
    )
    experiment_iteration: str = Field(
        default="0",
        description="Current loop index of the experiment"
    )

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if self.plaintext_matrix_test_filename is None:
            self.plaintext_matrix_test_filename = self.plaintext_matrix_test_id
