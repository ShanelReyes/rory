from dataclasses import dataclass, field
from typing import Tuple,Dict,Optional
import numpy.typing as npt

@dataclass
class PutPlaintextResult:
    """Model for the result of a plaintext put operation, including metadata and timing information."""
    extension: str
    """File extension indicating the format of the stored plaintext (e.g., '.txt', '.csv')."""
    bucket_id: str
    """Identifier for the storage bucket where the plaintext was stored."""
    ball_id: str
    """Identifier for the specific storage object (ball) where the plaintext was stored."""
    tags: Dict[str, str]
    """Dictionary of tags associated with the stored plaintext, which can be used for metadata or categorization."""
    shape: Tuple[int,int]
    """Shape of the stored plaintext data, if applicable (e.g., for arrays or matrices)."""
    dtype: npt.DTypeLike
    """Data type of the stored plaintext data (e.g., 'float32', 'int64')."""
    read_time: float
    """Time taken to read the plaintext data from the source, in seconds."""
    segment_time: float
    """Time taken to segment the plaintext data into appropriate chunks for storage, in seconds."""
    upload_time: float
    """Time taken to upload the plaintext data to the storage backend, in seconds."""
    path: Optional[str] = field(default=None)
    """Optional path where the plaintext was stored. This may be None if the storage backend does not provide a path."""
    def __str__(self):
        return (f"PutPlaintextResult(path={self.path}, extension={self.extension}, bucket_id={self.bucket_id}, "
                f"ball_id={self.ball_id}, tags={self.tags}, shape={self.shape}, dtype={self.dtype}, "
                f"read_time={self.read_time:.4f}s, segment_time={self.segment_time:.4f}s, upload_time={self.upload_time:.4f}s)")

from enum import Enum
class SourceType(Enum):
    """Enumeration of possible source types for retrieved data."""
    FILE = "file"
    """Data was retrieved from a file storage system (e.g., local filesystem, network file share)."""
    URL = "url"
    """Data was retrieved from a URL (e.g., via HTTP/HTTPS)."""
    DATABASE = "database"
    """Data was retrieved from a database (e.g., SQL, NoSQL)."""
    IN_MEMORY = "in_memory"
    """Data was retrieved from an in-memory data structure (e.g., cache, variable)."""
    CLOUD = "cloud"
    """Data was retrieved from a cloud storage service (e.g., AWS S3, Google Cloud Storage)."""
    OTHER = "other"
    """Data was retrieved from an unspecified or unknown source type."""

from typing import Generic,TypeVar,List
from Pyfhel import PyCtxt
TList = TypeVar('TList', List[bytes], List[PyCtxt], List[float], List[int], npt.NDArray)
@dataclass
class GetResult(Generic[TList]):
    """Model for the result of a get operation, including metadata and timing information."""
    source: Optional[SourceType] = field(default=None)
    """Optional source type indicating where the data was retrieved from (e.g., FILE, URL, DATABASE, IN_MEMORY, CLOUD, OTHER)."""
    raw_value: Optional[TList] = field(default=None)
    """Optional raw value retrieved from the storage backend."""
    read_time: Optional[float] = field(default=None)
    """Optional time taken to read the data from the source, in seconds."""

    def to_nd_array(self) -> Optional[npt.NDArray]:
        if self.raw_value is not None and isinstance(self.raw_value, npt.NDArray):
            return self.raw_value
        return None
    def to_list(self) -> Optional[List]:
        if self.raw_value is not None and isinstance(self.raw_value, list):
            return self.raw_value
        return None
    def to_pyctxt_list(self) -> Optional[List[PyCtxt]]:
        if self.raw_value is not None and isinstance(self.raw_value, list) and all(isinstance(item, PyCtxt) for item in self.raw_value):
            return self.raw_value
        return None
    def to_bytes_list(self) -> Optional[List[bytes]]:
        if self.raw_value is not None and isinstance(self.raw_value, list) and all(isinstance(item, bytes) for item in self.raw_value):
            return self.raw_value
        return None
    def to_float_list(self) -> Optional[List[float]]:
        if self.raw_value is not None and isinstance(self.raw_value, list) and all(isinstance(item, float) for item in self.raw_value):
            return self.raw_value
        return None
    def to_int_list(self) -> Optional[List[int]]:
        if self.raw_value is not None and isinstance(self.raw_value, list) and all(isinstance(item, int) for item in self.raw_value):
            return self.raw_value
        return None



@dataclass
class PutCiphertextResult:
    """Model for the result of a ciphertext put operation, including metadata and timing information."""
    path: Optional[str]
    """Optional path where the ciphertext was stored. This may be None if the storage backend does not provide a path."""
    extension: str
    """File extension indicating the format of the stored ciphertext (e.g., '.ctxt', '.bin')."""
    bucket_id: str
    """Identifier for the storage bucket where the ciphertext was stored."""
    ball_id: str
    """Identifier for the specific storage object (ball) where the ciphertext was stored."""
    tags: Dict[str, str]
    """Dictionary of tags associated with the stored ciphertext, which can be used for metadata or categorization."""
    shape: Tuple[int,int]
    """Shape of the stored ciphertext data, if applicable (e.g., for arrays or matrices)."""
    dtype: npt.DTypeLike
    """Data type of the stored ciphertext data (e.g., 'float32', 'int64')."""
    read_time: float
    """Time taken to read the plaintext data from the source, in seconds."""
    segment_time: float
    """Time taken to segment the plaintext data into appropriate chunks for encryption, in seconds."""
    encrypt_time: float
    """Time taken to encrypt the plaintext data, in seconds."""
    upload_time: float
    """Time taken to upload the ciphertext data to the storage backend, in seconds."""
    def __str__(self):        return (f"PutCiphertextResult(path={self.path}, extension={self.extension}, bucket_id={self.bucket_id}, "
                f"ball_id={self.ball_id}, tags={self.tags}, shape={self.shape}, dtype={self.dtype}, "
                f"read_time={self.read_time:.4f}s, segment_time={self.segment_time:.4f}s, encrypt_time={self.encrypt_time:.4f}s, upload_time={self.upload_time:.4f}s)") 
    