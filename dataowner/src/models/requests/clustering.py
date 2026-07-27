from pydantic import BaseModel, Field
from typing import Optional
from uuid import uuid4


def _default_experiment_id() -> str:
    return uuid4().hex[:10]


class BaseRequest(BaseModel):
    experiment_id: Optional[str] = Field(
        default=None,
        description="Unique tracking ID for performance auditing. Auto-generated if not provided."
    )

    def model_post_init(self, __context):
        if self.experiment_id is None:
            self.experiment_id = uuid4().hex[:10]


class MatrixRequest(BaseRequest):
    """Shared fields for clustering algorithms that read a local plaintext matrix."""
    plaintext_matrix_id: str = Field(
        default="matrix-0",
        description="Unique identifier for the matrix in Cloud Storage System"
    )
    plaintext_matrix_filename: str = Field(
        default="matrix-0",
        description="Local filename for data reading (without extension)"
    )
    extension: str = Field(
        default="csv",
        description="File extension of the dataset (e.g., csv, npy)"
    )


class KmeansRequest(MatrixRequest):
    """Parameters for plaintext K-Means clustering."""
    k: int = Field(
        default=3,
        ge=2,
        le=100,
        description="Number of clusters to form"
    )


class NncRequest(MatrixRequest):
    """Parameters for plaintext Nearest Neighbor Clustering."""
    threshold: float = Field(
        default=-1.0,
        description="Distance limit for clustering. If -1, calculated automatically from dataset"
    )


class EncryptedClusteringRequest(MatrixRequest):
    """Shared fields for encrypted interactive clustering algorithms."""
    k: int = Field(description="Number of clusters to identify", ge=2, le=100)
    experiment_iteration: str = Field(
        default="0",
        description="Current loop index of the experiment"
    )
    max_iterations: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Maximum number of protocol rounds"
    )


class SkmeansRequest(EncryptedClusteringRequest):
    """Parameters for Liu homomorphic encryption K-Means clustering."""
    convergence_threshold: float = Field(
        default=0.000001,
        gt=0,
        description="Tolerance for centroid shift convergence"
    )


class DbskmeansRequest(EncryptedClusteringRequest):
    """Parameters for Double-Blind Secure K-Means clustering."""
    sens: float = Field(
        default=0.00000001,
        gt=0,
        description="Sensitivity parameter for the FDHOPE encryption scheme"
    )
    convergence_threshold: float = Field(
        default=0.000001,
        gt=0,
        description="Tolerance for centroid shift convergence"
    )


class DbsnncRequest(MatrixRequest):
    """Parameters for Double-Blind Secure Nearest Neighbor Clustering."""
    sens: float = Field(
        default=0.00000001,
        gt=0,
        description="Sensitivity parameter for FDHOPE encryption"
    )
    threshold: float = Field(
        default=-1.0,
        description="Distance threshold for clustering. If -1, auto-calculated from dataset"
    )


class PqcSkmeansRequest(EncryptedClusteringRequest):
    """Parameters for CKKS post-quantum secure K-Means clustering."""


class PqcDbskmeansRequest(PqcSkmeansRequest):
    """Parameters for CKKS post-quantum double-blind secure K-Means clustering."""
    sens: float = Field(
        default=0.00000001,
        gt=0,
        description="Sensitivity parameter for FDHOPE encryption"
    )
