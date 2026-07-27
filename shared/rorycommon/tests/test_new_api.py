import pytest
import numpy as np
from uuid import uuid4
from rorycommon import Common as RoryCommon

# -----------------------------------------------------------------------------
# 1. Real Fixtures
# -----------------------------------------------------------------------------

    
@pytest.fixture
def real_matrix():
    """Provides a real numpy array for testing."""
    return np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64)

@pytest.fixture
def temp_matrix_file(tmp_path, real_matrix):
    """Saves the real matrix to a temporary disk location and returns the path."""
    file_path = tmp_path / "test_matrix.npy"
    np.save(file_path, real_matrix)
    # TargetClass expects path without extension and extension separately
    return file_path, "npy"


@pytest.fixture
def test_env_args(client):
    """Base arguments needed for the cloud storage methods."""
    iid  = uuid4().hex[:5]
    return {
        "client": client,
        "bucket_id": f"bucket_id_{iid}",  # UPDATE THIS
        "ball_id": f"ball_id_{iid}",
        "tags": {"env": "pytest"},
        "timeout": 120,
        "max_attempts": 3
    }

@pytest.fixture
def ckks_keys_args(keys_path):
    """
    Paths and filenames for your pre-generated Pyfhel keys.
    These keys MUST exist on your disk for the tests to pass.
    """
    return {
        "keys_path": keys_path, # UPDATE THIS to your key folder
        "ctx_filename": "ctx",
        "pubkey_filename": "pubkey",
        "secretkey_filename": "secretkey",
        "relinkey_filename": "relinkey",
        "rotatekey_filename": "rotatekey",
        "decimals": 5,
        "num_chunks": 2,
    }


# -----------------------------------------------------------------------------
# 2. Plain Text Integration Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_integration_plaintext_disk_to_cloud(temp_matrix_file, test_env_args):
    path, ext = temp_matrix_file
    
    result = await RoryCommon.from_matrix_on_disk_to_cloud_storage(
        path=path,
        extension=ext,
        **test_env_args
    )
    
    assert result.is_ok

@pytest.mark.asyncio
async def test_integration_plaintext_matrix_to_cloud_and_back(real_matrix, test_env_args):
    # 1. Upload the matrix
    upload_result = await RoryCommon.from_matrix_to_cloud_storage(
        plaintext_matrix=real_matrix,
        **test_env_args
    )
    assert upload_result.is_ok
    
    # 2. Download it back to verify the round trip
    download_result = await RoryCommon.from_cloud_storage_to_matrix(
        client=test_env_args["client"],
        bucket_id=test_env_args["bucket_id"],
        ball_id=test_env_args["ball_id"],
        chunk_size="256kb"
    )
    
    assert download_result.is_ok
    downloaded_matrix = download_result.unwrap()
    assert np.allclose(real_matrix, downloaded_matrix.raw_value)


# -----------------------------------------------------------------------------
# 3. CKKS Integration Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_integration_ckks_disk_to_cloud(temp_matrix_file, test_env_args, ckks_keys_args):
    path, ext = temp_matrix_file
    
    result = await RoryCommon.from_matrix_on_disk_to_cloud_storage_ckks(
        path=path,
        extension=ext,
        **test_env_args,
        **ckks_keys_args
    )
    
    assert result.is_ok

@pytest.mark.asyncio
async def test_integration_ckks_vector_disk_to_cloud(tmp_path, test_env_args, ckks_keys_args):
    # Setup a 1D vector instead of a matrix
    vector = np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float64)
    file_path = tmp_path / "test_vector.npy"
    np.save(file_path, vector)
    
    result = await RoryCommon.from_vector_on_disk_to_cloud_storage_ckks(
        path      = str(file_path),
        extension = "npy",
        _round    = False,
        **test_env_args,
        **ckks_keys_args
    )
    
    assert result.is_ok

@pytest.mark.asyncio
async def test_integration_ckks_matrix_to_cloud(real_matrix, test_env_args, ckks_keys_args):
    result = await RoryCommon.from_matrix_to_cloud_storage_ckks(
        plaintext_matrix=real_matrix,
        _round=False,
        **test_env_args,
        **ckks_keys_args
    )
    
    assert result.is_ok

@pytest.mark.asyncio
async def test_integration_ckks_vector_to_cloud(test_env_args, ckks_keys_args,ckks):
    vector = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    
    result = await RoryCommon.from_vector_to_cloud_storage_ckks(
        vector=vector,
        _round=False,
        **test_env_args,
        **ckks_keys_args
    )
    
    assert result.is_ok

    result = await RoryCommon.get_pyctxt(
        client=test_env_args["client"],
        bucket_id=test_env_args["bucket_id"],
        key=test_env_args["ball_id"],
        ckks= ckks
    )
    assert len(result) == len(vector)
    # print("VECTOR LEN:", len(vector))
    # print("ENCRYPTED VECTOR:", len(result))
    # print("Decrypted vector:", result)
    # print("Decrypted vector:", result.unwrap())
    # assert result.is_ok