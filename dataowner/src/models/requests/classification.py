from pydantic import BaseModel, Field
from typing import Optional
from uuid import uuid4

from .clustering import BaseRequest


class KnnTrainRequest(BaseRequest):
    """Parameters for KNN training phase (plaintext reference dataset upload)."""
    model_id: str = Field(
        default="matrix0model",
        description="Unique identifier for the model in Cloud Storage System"
    )
    model_filename: Optional[str] = Field(
        default=None,
        description="Local filename for the feature matrix (without extension). Defaults to model_id"
    )
    model_labels_filename: Optional[str] = Field(
        default=None,
        description="Local filename for the labels vector (without extension). Defaults to '<model_id>labels'"
    )
    extension: str = Field(
        default="npy",
        description="File extension of the source data"
    )
    num_chunks: int = Field(
        default=4,
        ge=1,
        le=100,
        description="Number of chunks for Cloud Storage segmentation"
    )

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if self.model_filename is None:
            self.model_filename = self.model_id
        if self.model_labels_filename is None:
            self.model_labels_filename = f"{self.model_id}labels"


class KnnPredictRequest(BaseRequest):
    """Parameters for KNN prediction phase (plaintext nearest neighbor search)."""
    model_id: str = Field(
        default="model-0",
        description="ID of the pre-trained model stored in Cloud Storage System"
    )
    records_test_id: str = Field(
        default="matrix0data",
        description="Unique ID for the test records to be stored in CSS"
    )
    records_test_filename: Optional[str] = Field(
        default=None,
        description="Local filename for the test dataset. Defaults to records_test_id"
    )
    model_labels_shape: str = Field(
        ...,
        description="The shape of the model's labels (required for distributed distance calculations)"
    )
    extension: str = Field(
        default="npy",
        description="File extension of the source data"
    )

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if self.records_test_filename is None:
            self.records_test_filename = self.records_test_id


class SknnTrainRequest(BaseRequest):
    """Parameters for Secure KNN training phase (Liu homomorphic encryption)."""
    model_id: str = Field(
        default="matrix-0_model",
        description="Unique identifier for the model and its labels"
    )
    model_filename: Optional[str] = Field(
        default=None,
        description="Local filename for the feature matrix. Defaults to model_id"
    )
    model_labels_filename: Optional[str] = Field(
        default=None,
        description="Local filename for the labels vector. Defaults to '<model_id>labels'"
    )
    extension: str = Field(
        default="npy",
        description="Source data file extension"
    )

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if self.model_filename is None:
            self.model_filename = self.model_id
        if self.model_labels_filename is None:
            self.model_labels_filename = f"{self.model_id}labels"


class SknnPredictRequest(BaseRequest):
    """Parameters for Secure KNN prediction phase (Liu homomorphic, interactive 2-round)."""
    model_id: str = Field(
        default="model0",
        description="ID of the encrypted model in Cloud Storage System"
    )
    records_test_id: str = Field(
        default="matrix0data",
        description="ID for the encrypted test records"
    )
    records_test_filename: Optional[str] = Field(
        default=None,
        description="Local filename for the test records. Defaults to records_test_id"
    )
    encrypted_model_shape: str = Field(
        ...,
        description="The 3D shape of the model (r, a, m) from training phase"
    )
    encrypted_model_dtype: str = Field(
        ...,
        description="Data type of the encrypted model (e.g., 'float32')"
    )
    model_labels_shape: str = Field(
        ...,
        description="The shape of the labels vector from training phase"
    )
    extension: str = Field(
        default="npy",
        description="Source data file extension"
    )

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if self.records_test_filename is None:
            self.records_test_filename = self.records_test_id


class PqcSknnTrainRequest(BaseRequest):
    """Parameters for PQC Secure KNN training phase (CKKS post-quantum encryption)."""
    model_id: str = Field(
        default="matrix-0_model",
        description="Unique identifier for the model"
    )
    model_filename: Optional[str] = Field(
        default=None,
        description="Local filename for the feature matrix. Defaults to model_id"
    )
    model_labels_filename: Optional[str] = Field(
        default=None,
        description="Local filename for the label vector. Defaults to '<model_id>labels'"
    )
    extension: str = Field(
        default="npy",
        description="Source data file extension"
    )

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if self.model_filename is None:
            self.model_filename = self.model_id
        if self.model_labels_filename is None:
            self.model_labels_filename = f"{self.model_id}labels"


class PqcSknnPredictRequest(BaseRequest):
    """Parameters for PQC Secure KNN prediction phase (CKKS, interactive 2-round)."""
    model_id: str = Field(
        default="model0",
        description="ID of the pre-trained PQC model in Cloud Storage System"
    )
    records_test_id: str = Field(
        default="matrix0data",
        description="Unique ID for the test records"
    )
    records_test_filename: Optional[str] = Field(
        default=None,
        description="Local filename for the test records. Defaults to records_test_id"
    )
    encrypted_model_shape: str = Field(
        ...,
        description="Dimensions of the CKKS model matrix from training phase"
    )
    encrypted_model_dtype: str = Field(
        ...,
        description="Data type of the encrypted model (e.g., 'float32')"
    )
    records_test_extension: str = Field(
        default="npy",
        description="File extension of the test records (e.g., 'npy')"
    )
    extension: str = Field(
        default="npy",
        description="Source data file extension"
    )

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if self.records_test_filename is None:
            self.records_test_filename = self.records_test_id
