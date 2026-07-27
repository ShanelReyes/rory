from pydantic import BaseModel, Field
from typing import Optional, List
from uuid import uuid4


def _default_experiment_id() -> str:
    return uuid4().hex[:10]


class BaseRequest(BaseModel):
    experiment_id: Optional[str] = Field(
        default=None,
        description="Unique tracking ID for performance auditing. Auto-generated if not provided.",
    )

    def model_post_init(self, __context):
        if self.experiment_id is None:
            self.experiment_id = uuid4().hex[:10]


class KmeansWorkerRequest(BaseRequest):
    plaintext_matrix_id: str = Field(
        default="matrix-0",
        description="Unique identifier for the matrix in Cloud Storage System",
    )
    k: int = Field(
        default=3,
        ge=2,
        le=100,
        description="Number of clusters to form",
    )


class SkmeansWorkerRequest(BaseRequest):
    step_index: int = Field(
        default=1,
        ge=1,
        le=2,
        description="Protocol step (1=initial clustering, 2=convergence update)",
    )
    clustering_status: str = Field(default="0", description="Current clustering status")
    k: str = Field(default="3", description="Number of clusters")
    m: str = Field(default="3", description="Security parameter M")
    plaintext_matrix_id: str = Field(default="matrix0", description="Matrix identifier")
    encrypted_matrix_id: str = Field(default="", description="Encrypted matrix identifier in CSS")
    encrypted_matrix_shape: Optional[str] = Field(default=None, description="Shape of the encrypted matrix (required for step 1)")
    encrypted_matrix_dtype: Optional[str] = Field(default=None, description="Data type of the encrypted matrix (required for step 1)")
    iterations: str = Field(default="0", description="Current iteration count")
    num_chunks: Optional[str] = Field(default=None, description="Number of chunks (required)")
    is_zero: Optional[str] = Field(default=None, description="Convergence flag for step 2")
    shift_matrix_id: Optional[str] = Field(default=None, description="Shift matrix ID for step 2")


class DbskmeansWorkerRequest(SkmeansWorkerRequest):
    encrypted_udm_shape: Optional[str] = Field(default=None, description="Shape of the encrypted UDM (step 1)")
    encrypted_udm_dtype: Optional[str] = Field(default=None, description="Data type of the encrypted UDM (step 1)")


class DbsnncWorkerRequest(BaseRequest):
    plaintext_matrix_id: str = Field(default="matrix0", description="Matrix identifier")
    encrypted_matrix_id: str = Field(default="", description="Encrypted matrix identifier")
    encrypted_dm_id: Optional[str] = Field(default=None, description="Encrypted distance matrix identifier")
    encrypted_threshold: Optional[str] = Field(default=None, description="Encrypted threshold value")
    encrypted_matrix_shape: Optional[str] = Field(default=None, description="Shape (required)")
    encrypted_matrix_dtype: Optional[str] = Field(default=None, description="Data type (required)")
    encrypted_dm_shape: Optional[str] = Field(default=None, description="DM shape (required)")
    encrypted_dm_dtype: Optional[str] = Field(default=None, description="DM data type (required)")
    m: str = Field(default="3", description="Security parameter M")
    num_chunks: Optional[str] = Field(default=None, description="Number of chunks (required)")


class NncWorkerRequest(BaseRequest):
    plaintext_matrix_id: str = Field(default="matrix0", description="Matrix identifier")
    threshold: Optional[str] = Field(default=None, description="Distance threshold")
    plaintext_matrix_shape: Optional[str] = Field(default=None, description="Shape (required)")
    plaintext_matrix_dtype: Optional[str] = Field(default=None, description="Data type (required)")
    dm_shape: Optional[str] = Field(default=None, description="DM shape (required)")
    dm_dtype: Optional[str] = Field(default=None, description="DM data type (required)")
    num_chunks: Optional[str] = Field(default=None, description="Number of chunks (required)")


class PqcSkmeansWorkerRequest(BaseRequest):
    step_index: int = Field(default=1, ge=1, le=2, description="Protocol step")
    clustering_status: str = Field(default="0", description="Current clustering status")
    k: str = Field(default="3", description="Number of clusters")
    plaintext_matrix_id: str = Field(default="matrix0", description="Matrix identifier")
    encrypted_matrix_id: str = Field(default="", description="Encrypted matrix identifier")
    encrypted_matrix_shape: Optional[str] = Field(default=None, description="Shape (required for step 1)")
    encrypted_matrix_dtype: Optional[str] = Field(default=None, description="Data type (required for step 1)")
    iterations: str = Field(default="0", description="Current iteration count")
    num_chunks: Optional[str] = Field(default=None, description="Number of chunks (required)")
    is_zero: Optional[str] = Field(default=None, description="Convergence flag for step 2")
    shift_matrix_id: Optional[str] = Field(default=None, description="Shift matrix ID for step 2")


class PqcDbskmeansWorkerRequest(PqcSkmeansWorkerRequest):
    encrypted_udm_shape: Optional[str] = Field(default=None, description="Shape of encrypted UDM (step 1)")
    encrypted_udm_dtype: Optional[str] = Field(default=None, description="Data type of encrypted UDM (step 1)")
    shift_matrix_ope_id: Optional[str] = Field(default=None, description="Shift matrix OPE ID for step 2")
