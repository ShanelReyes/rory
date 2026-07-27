import pytest
import numpy as np
import pandas as pd
from uuid import uuid4
from Pyfhel import PyCtxt
from rorycommon import StorageBuilder, StorageBackend, StorageParams, Scheme, CkksParams, LiuParams, FdhopeParams
from rory.core.security.pqc.dataowner import DataOwner as DataOwnerPQC

import os
RORY_KEYS_PATH             = os.environ.get("RORY_KEYS_PATH", "/rory/keys/test2")
RORY_COMMON_CTX_FILENAME   = os.environ.get("RORY_COMMON_CTX_FILENAME", "ctx")
RORY_COMMON_PUBKEY_FILENAME = os.environ.get("RORY_COMMON_PUBKEY_FILENAME", "pubkey")
RORY_COMMON_SECRETKEY_FILENAME = os.environ.get("RORY_COMMON_SECRETKEY_FILENAME", "secretkey")
RORY_COMMON_RELINKEY_FILENAME  = os.environ.get("RORY_COMMON_RELINKEY_FILENAME", "relinkey")
RORY_COMMON_ROTATEKEY_FILENAME = os.environ.get("RORY_COMMON_ROTATEKEY_FILENAME", "rotatekey")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_matrix():
    return np.random.random((4, 4)).astype(np.float64)


@pytest.fixture
def small_vector():
    return np.random.random((16,)).astype(np.float64)


@pytest.fixture
def tmp_npy_vector_file(tmp_path, small_vector):
    path = tmp_path / "vector.npy"
    np.save(path, small_vector)
    return str(path), "npy"


@pytest.fixture
def storage_ids():
    iid = uuid4().hex[:6]
    return {"bucket_id": f"test{iid}", "ball_id": f"ball{iid}"}


# @pytest.fixture
# def paillier_do():
#     do = DataOwnerPHE(securitylevel=128)
#     do.generate_keys()
#     return do


@pytest.fixture
def tmp_npy_file(tmp_path, small_matrix):
    path = tmp_path / "matrix.npy"
    np.save(path, small_matrix)
    return str(path), "npy"


@pytest.fixture
def tmp_csv_file(tmp_path, small_matrix):
    path = tmp_path / "matrix.csv"
    pd.DataFrame(small_matrix).to_csv(path, header=False, index=False)
    return str(path), "csv"


def ckks_builder(client, ckks, ckks_params):
    """StorageBuilder wired for CKKS with full key config."""
    return StorageBuilder(
        storage_client = client,
        scheme      = Scheme.CKKS,
        ckks           = ckks,
        ckks_params    = ckks_params,
    ).build()


def liu_builder(client, liu_params):
    return StorageBuilder(storage_client=client, scheme=Scheme.LIU, liu_params=liu_params).build()


def fdhope_builder(client, fdhope_params):
    return StorageBuilder(storage_client=client, scheme=Scheme.FDHOPE, fdhope_params=fdhope_params).build()


# ---------------------------------------------------------------------------
# StorageParams — unit tests (no network)
# ---------------------------------------------------------------------------

def test_storage_params_defaults():
    p = StorageParams()
    assert p.backoff_factor == 0.5
    assert p.num_chunks == 2
    assert p.chunk_index == 0
    assert p.chunk_size == "256kb"
    assert p.delay == 1
    assert p.force is True
    assert p.headers == {}
    assert p.http2 is False
    assert p.max_attempts == 5
    assert p.max_parallel_gets == 10
    assert p.timeout == 300


def test_storage_params_custom():
    p = StorageParams(num_chunks=4, timeout=60, chunk_size="128kb")
    assert p.num_chunks == 4
    assert p.timeout == 60
    assert p.chunk_size == "128kb"


# ---------------------------------------------------------------------------
# StorageBuilder — unit tests (no network)
# ---------------------------------------------------------------------------

async def test_builder_with_scheme(client, ckks):
    backend = (
        StorageBuilder(storage_client=client, scheme=Scheme.CKKS, ckks=ckks)
        .with_scheme(Scheme.LIU)
        .build()
    )
    assert backend.scheme == Scheme.LIU


async def test_builder_with_ckks(client, ckks):
    backend = (
        StorageBuilder(storage_client=client, scheme=Scheme.CKKS)
        .with_ckks(ckks)
        .build()
    )
    assert backend.ckks is ckks


async def test_storage_params_applied(client, ckks):
    params = StorageParams(backoff_factor=1.5, num_chunks=4, timeout=60)
    backend = (
        StorageBuilder(storage_client=client, scheme=Scheme.CKKS, ckks=ckks)
        .with_storage_params(params)
        .build()
    )
    assert backend.params.backoff_factor == 1.5
    assert backend.params.num_chunks == 4
    assert backend.params.timeout == 60


async def test_builder_defaults_to_storage_params(client, ckks):
    backend = StorageBuilder(storage_client=client, scheme=Scheme.CKKS, ckks=ckks).build()
    assert isinstance(backend.params, StorageParams)


async def test_builder_ckks_key_config(client, ckks, ckks_params):
    backend = ckks_builder(client, ckks, ckks_params)
    assert backend.ckks_params.keys_path == RORY_KEYS_PATH
    assert backend.ckks_params.ctx_filename == RORY_COMMON_CTX_FILENAME


async def test_builder_with_fdhope_params(client, fdhope_params):
    backend = (
        StorageBuilder(storage_client=client, scheme=Scheme.FDHOPE)
        .with_fdhope_params(fdhope_params)
        .build()
    )
    assert backend.fdhope_params is fdhope_params


async def test_builder_fdhope_config_round_trip(client, fdhope_params):
    backend = fdhope_builder(client, fdhope_params)
    cloned = backend.as_builder().build()
    assert cloned.scheme == Scheme.FDHOPE
    assert cloned.fdhope_params == fdhope_params


# ---------------------------------------------------------------------------
# Default put/get (no segment, no encrypt) — single blob
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_put_default_ckks(client, ckks, ckks_params, small_matrix, storage_ids):
    backend = ckks_builder(client, ckks, ckks_params)
    result = await backend.put(**storage_ids, data=small_matrix)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().shape == small_matrix.shape


@pytest.mark.asyncio
async def test_put_get_default_ckks(client, ckks, ckks_params, small_matrix, storage_ids):
    backend = ckks_builder(client, ckks, ckks_params)
    await backend.put(**storage_ids, data=small_matrix)
    result = await backend.get(**storage_ids)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().raw_value is not None
    assert isinstance(result.unwrap().raw_value, np.ndarray)


@pytest.mark.asyncio
async def test_put_default_liu(client, liu_params, small_matrix, storage_ids):
    backend = liu_builder(client, liu_params)
    result = await backend.put(**storage_ids, data=small_matrix)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().shape == small_matrix.shape


@pytest.mark.asyncio
async def test_put_get_default_liu(client, liu_params, small_matrix, storage_ids):
    backend = liu_builder(client, liu_params)
    await backend.put(**storage_ids, data=small_matrix)
    result = await backend.get(**storage_ids)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().raw_value is not None
    assert isinstance(result.unwrap().raw_value, np.ndarray)


# ---------------------------------------------------------------------------
# Segment only (segment=True, encrypt=False) — chunked, unencrypted
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_put_segment_only_ckks(client, ckks, ckks_params, small_matrix, storage_ids):
    backend = ckks_builder(client, ckks, ckks_params)
    result = await backend.put(**storage_ids, data=small_matrix, segment=True)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().shape == small_matrix.shape


@pytest.mark.asyncio
async def test_put_get_segment_ckks(client, ckks, ckks_params, small_matrix, storage_ids):
    backend: StorageBackend = ckks_builder(client, ckks, ckks_params)
    bucket_id = storage_ids["bucket_id"]
    ball_id = storage_ids["ball_id"]
    res = await backend.put(bucket_id=bucket_id, ball_id=ball_id, data=small_matrix, segment=True)
    assert res.is_ok, res.unwrap_err()
    result = await backend.get(bucket_id=bucket_id, ball_id=ball_id, segment=True)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().raw_value is not None


@pytest.mark.asyncio
async def test_put_segment_only_liu(client, liu_params, small_matrix, storage_ids):
    backend = liu_builder(client, liu_params)
    result = await backend.put(**storage_ids, data=small_matrix, segment=True)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().shape == small_matrix.shape


@pytest.mark.asyncio
async def test_put_get_segment_liu(client, liu_params, small_matrix, storage_ids):
    backend = liu_builder(client, liu_params)
    await backend.put(**storage_ids, data=small_matrix, segment=True)
    result = await backend.get(**storage_ids, segment=True)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().raw_value is not None


# ---------------------------------------------------------------------------
# Segment + encrypt — CKKS and LIU
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_put_encrypt_ckks(client, ckks, ckks_params, small_matrix, storage_ids):
    backend = ckks_builder(client, ckks, ckks_params)
    result = await backend.put(**storage_ids, data=small_matrix, encrypt=True)
    assert result.is_ok, result.unwrap_err()


@pytest.mark.asyncio
async def test_put_get_encrypt_ckks(client, ckks, ckks_params, small_matrix, storage_ids):
    backend = ckks_builder(client, ckks, ckks_params)
    put = await backend.put(**storage_ids, data=small_matrix, encrypt=True)
    assert put.is_ok, put.unwrap_err()
    result = await backend.get(**storage_ids, encrypt=True)
    assert result.is_ok, result.unwrap_err()
    value = result.unwrap()
    assert value.raw_value is not None
    assert isinstance(value.raw_value, list)
    assert all(isinstance(x, PyCtxt) for x in value.raw_value)


@pytest.mark.asyncio
async def test_put_encrypt_liu(client, liu_params, small_matrix, storage_ids):
    backend = liu_builder(client, liu_params)
    result = await backend.put(**storage_ids, data=small_matrix, encrypt=True)
    assert result.is_ok, result.unwrap_err()


@pytest.mark.asyncio
async def test_put_get_encrypt_liu(client, liu_params, small_matrix, storage_ids):
    backend = liu_builder(client, liu_params)
    put = await backend.put(**storage_ids, data=small_matrix, encrypt=True)
    assert put.is_ok, put.unwrap_err()
    result = await backend.get(**storage_ids, encrypt=True)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().raw_value is not None
    assert isinstance(result.unwrap().raw_value, np.ndarray)


@pytest.mark.asyncio
async def test_put_encrypt_fdhope(client, dataowner, fdhope_params, small_matrix, storage_ids):
    udm = dataowner.get_U(
        algorithm        = "DBSKMEANS",
        plaintext_matrix = small_matrix,
    )
    backend = fdhope_builder(client, fdhope_params)
    result = await backend.put(**storage_ids, data=udm, encrypt=True)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().shape == udm.shape


@pytest.mark.asyncio
async def test_put_get_encrypt_fdhope(client, dataowner, fdhope_params, small_matrix, storage_ids):
    udm = dataowner.get_U(
        algorithm        = "DBSKMEANS",
        plaintext_matrix = small_matrix,
    )
    backend = fdhope_builder(client, fdhope_params)
    put = await backend.put(**storage_ids, data=udm, encrypt=True)
    assert put.is_ok, put.unwrap_err()
    result = await backend.get(**storage_ids, encrypt=True)
    assert result.is_ok, result.unwrap_err()
    value = result.unwrap()
    assert value.raw_value is not None
    assert isinstance(value.raw_value, np.ndarray)
    assert value.raw_value.shape == udm.shape


@pytest.mark.asyncio
async def test_get_encrypt_fdhope_uses_get_and_merge(client, fdhope_params, monkeypatch):
    backend = fdhope_builder(client, fdhope_params)
    expected = np.arange(4, dtype=np.float64).reshape(2, 2)
    calls = {"get_and_merge": 0, "get_pyctxt": 0}

    async def fake_get_and_merge(**kwargs):
        calls["get_and_merge"] += 1
        assert kwargs["bucket_id"] == "bucket"
        assert kwargs["key"] == "ball"
        return expected

    async def fake_get_pyctxt(**kwargs):
        calls["get_pyctxt"] += 1
        raise AssertionError("FDHOPE get should not use CKKS retrieval")

    monkeypatch.setattr("rorycommon.Common.get_and_merge", fake_get_and_merge)
    monkeypatch.setattr("rorycommon.Common.get_pyctxt", fake_get_pyctxt)

    result = await backend.get(bucket_id="bucket", ball_id="ball", encrypt=True)

    assert result.is_ok, result.unwrap_err()
    assert calls == {"get_and_merge": 1, "get_pyctxt": 0}
    assert np.array_equal(result.unwrap().raw_value, expected)


# ---------------------------------------------------------------------------
# Pre-processed TList — CKKS List[PyCtxt]
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_put_tlist_ckks(client, ckks, ckks_params, small_matrix, storage_ids):
    dataowner_pqc = DataOwnerPQC(scheme=ckks)
    ciphertexts = dataowner_pqc.ckks_encrypt_matrix_chunk(small_matrix)
    backend = ckks_builder(client, ckks, ckks_params)
    result = await backend.put(**storage_ids, data=ciphertexts)
    assert result.is_ok, result.unwrap_err()


# ---------------------------------------------------------------------------
# put_from_file
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_put_from_file_default_ckks(client, ckks, ckks_params, storage_ids, tmp_npy_file):
    path, ext = tmp_npy_file
    backend = ckks_builder(client, ckks, ckks_params)
    result = await backend.put_from_file(**storage_ids, path=path, extension=ext)
    assert result.is_ok, result.unwrap_err()


@pytest.mark.asyncio
async def test_put_from_file_encrypt_ckks(client, ckks, ckks_params, storage_ids, tmp_npy_file):
    path, ext = tmp_npy_file
    backend = ckks_builder(client, ckks, ckks_params)
    result = await backend.put_from_file(**storage_ids, path=path, extension=ext, encrypt=True)
    assert result.is_ok, result.unwrap_err()


@pytest.mark.asyncio
async def test_put_from_file_default_liu(client, liu_params, storage_ids, tmp_npy_file):
    path, ext = tmp_npy_file
    backend = liu_builder(client, liu_params)
    result = await backend.put_from_file(**storage_ids, path=path, extension=ext)
    assert result.is_ok, result.unwrap_err()


@pytest.mark.asyncio
async def test_put_from_file_encrypt_liu(client, liu_params, storage_ids, tmp_npy_file):
    path, ext = tmp_npy_file
    backend = liu_builder(client, liu_params)
    result = await backend.put_from_file(**storage_ids, path=path, extension=ext, encrypt=True)
    assert result.is_ok, result.unwrap_err()


@pytest.mark.asyncio
async def test_put_from_file_csv_default_ckks(client, ckks, ckks_params, storage_ids, tmp_csv_file):
    path, ext = tmp_csv_file
    backend = ckks_builder(client, ckks, ckks_params)
    result = await backend.put_from_file(**storage_ids, path=path, extension=ext)
    assert result.is_ok, result.unwrap_err()


# ---------------------------------------------------------------------------
# Vector (1-D ndarray) — CKKS encrypt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_put_encrypt_ckks_vector(client, ckks, ckks_params, small_vector, storage_ids):
    backend = ckks_builder(client, ckks, ckks_params)
    result = await backend.put(**storage_ids, data=small_vector, encrypt=True)
    assert result.is_ok, result.unwrap_err()


@pytest.mark.asyncio
async def test_put_get_encrypt_ckks_vector(client, ckks, ckks_params, small_vector, storage_ids):
    backend = ckks_builder(client, ckks, ckks_params)
    put = await backend.put(**storage_ids, data=small_vector, encrypt=True)
    assert put.is_ok, put.unwrap_err()
    result = await backend.get(**storage_ids, encrypt=True)
    assert result.is_ok, result.unwrap_err()
    value = result.unwrap()
    assert value.raw_value is not None
    assert isinstance(value.raw_value, list)
    assert all(isinstance(x, PyCtxt) for x in value.raw_value)


@pytest.mark.asyncio
async def test_put_from_file_encrypt_ckks_vector(client, ckks, ckks_params, storage_ids, tmp_npy_vector_file):
    path, ext = tmp_npy_vector_file
    backend = ckks_builder(client, ckks, ckks_params)
    result = await backend.put_from_file(**storage_ids, path=path, extension=ext, encrypt=True)
    assert result.is_ok, result.unwrap_err()


# ---------------------------------------------------------------------------
# delete flag
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_put_delete_flag_ckks(client, ckks, ckks_params, small_matrix, storage_ids):
    backend = ckks_builder(client, ckks, ckks_params)
    # First put to create the object, then overwrite with delete=True
    first = await backend.put(**storage_ids, data=small_matrix)
    assert first.is_ok, first.unwrap_err()
    result = await backend.put(**storage_ids, data=small_matrix, delete=True)
    assert result.is_ok, result.unwrap_err()


@pytest.mark.asyncio
async def test_put_delete_flag_on_nonexistent_key(client, ckks, ckks_params, small_matrix, storage_ids):
    backend = ckks_builder(client, ckks, ckks_params)
    # delete=True on a key that doesn't exist yet must not error
    result = await backend.put(**storage_ids, data=small_matrix, delete=True)
    assert result.is_ok, result.unwrap_err()


@pytest.mark.asyncio
async def test_put_from_file_delete_flag(client, ckks, ckks_params, storage_ids, tmp_npy_file):
    path, ext = tmp_npy_file
    backend = ckks_builder(client, ckks, ckks_params)
    first = await backend.put_from_file(**storage_ids, path=path, extension=ext)
    assert first.is_ok, first.unwrap_err()
    result = await backend.put_from_file(**storage_ids, path=path, extension=ext, delete=True)
    assert result.is_ok, result.unwrap_err()

@pytest.mark.asyncio
async def test_01(client,ckks,ckks_params,storage_ids,small_vector):
    backend = ckks_builder(client, ckks, ckks_params)
    # Delete + Segment + Encrypt + Put
    result = await backend.put(
        bucket_id = storage_ids["bucket_id"],
        ball_id   = storage_ids["ball_id"],
        data      = small_vector,
        encrypt   = True,
        segment   = True,
        delete    = True
    )
    assert result.is_ok, result.unwrap_err()
    # Get the encrypted data
    result = await backend.get(
        bucket_id = storage_ids["bucket_id"],
        ball_id   = storage_ids["ball_id"],
        encrypt   = True
    )
    assert result.is_ok, result.unwrap_err()    
    x = result.unwrap()
    raw_value = x.raw_value 
    assert len(raw_value) == len(small_vector)
    # from rorycommon import Common
    import time as T
    dx = ckks.decrypt_list(raw_value,take=1)
  
    # Delete + Segment + Put
    result = await backend.put(
        bucket_id = storage_ids["bucket_id"],
        ball_id   = storage_ids["ball_id"],
        data      = raw_value,
        delete    = True,
        segment   = True,
        encrypt   = False
    )
    assert result.is_ok, result.unwrap_err()
    # Get the encrypted, segmented data
    result = await backend.get(
        bucket_id = storage_ids["bucket_id"],
        ball_id   = storage_ids["ball_id"],
        encrypt   = True,
        segment   = True
    )
    assert result.is_ok, result.unwrap_err()
    raw_value2 = result.unwrap().raw_value
    dx1 = ckks.decrypt_list(raw_value2,take=1)
    assert len(raw_value2) == len(small_vector) 
    assert all(isinstance(x, PyCtxt) for x in raw_value2)
    assert len(raw_value) == len(raw_value2)
    assert all(abs(x[0] - x[0]) < 1e-5 for x in zip(dx, dx1))
    # print(dx)
    # print(dx1)

# ---------------------------------------------------------------------------
# put with string path — auto-delegates to put_from_file
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_put_string_path_default_ckks(client, ckks, ckks_params, storage_ids, tmp_npy_file):
    path, _ = tmp_npy_file
    backend = ckks_builder(client, ckks, ckks_params)
    result = await backend.put(**storage_ids, data=path)
    assert result.is_ok, result.unwrap_err()


@pytest.mark.asyncio
async def test_put_string_path_encrypt_ckks(client, ckks, ckks_params, storage_ids, tmp_npy_file):
    path, _ = tmp_npy_file
    backend = ckks_builder(client, ckks, ckks_params)
    result = await backend.put(**storage_ids, data=path, encrypt=True)
    assert result.is_ok, result.unwrap_err()


@pytest.mark.asyncio
async def test_put_string_path_default_liu(client, liu_params, storage_ids, tmp_npy_file):
    path, _ = tmp_npy_file
    backend = liu_builder(client, liu_params)
    result = await backend.put(**storage_ids, data=path)
    assert result.is_ok, result.unwrap_err()


@pytest.mark.asyncio
async def test_put_string_path_delete_flag(client, ckks, ckks_params, storage_ids, tmp_npy_file):
    path, _ = tmp_npy_file
    backend = ckks_builder(client, ckks, ckks_params)
    first = await backend.put(**storage_ids, data=path)
    assert first.is_ok, first.unwrap_err()
    result = await backend.put(**storage_ids, data=path, delete=True)
    assert result.is_ok, result.unwrap_err()


@pytest.mark.asyncio
async def test_put_a_list_with_one_element_ckks(client, ckks, ckks_params, storage_ids):
    backend = StorageBuilder(storage_client=client, scheme=Scheme.CKKS, ckks=ckks, ckks_params=ckks_params)\
    .with_storage_params(params=StorageParams(num_chunks=2))\
    .build()
    # ckks_builder(client, ckks, ckks_params)
    data = np.array([1])
    result = await backend.put(**storage_ids, data=data, delete=True, segment=True, encrypt=True)
    assert result.is_ok, result.unwrap_err()
    get_result = await backend.get(**storage_ids, segment=True, encrypt=True)
    assert get_result.is_ok, get_result.unwrap_err()
    value = get_result.unwrap()
    # print(value.raw_value)
    result2 = await backend.put(**storage_ids, data=value.raw_value, delete=True, segment=True, encrypt=False)
    assert result2.is_ok, result2.unwrap_err()


# ---------------------------------------------------------------------------
# List[int] / List[float] auto-conversion
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_put_list_int_plaintext(client, liu_params, storage_ids):
    """List[int] is accepted and stored as a float64 plaintext blob."""
    backend = StorageBuilder(storage_client=client, scheme=Scheme.LIU, liu_params=liu_params).build()
    data = [1, 2, 3, 4, 5]
    result = await backend.put(**storage_ids, data=data)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().dtype == np.float64


@pytest.mark.asyncio
async def test_put_list_float_plaintext(client, liu_params, storage_ids):
    """List[float] is accepted and stored as a float64 plaintext blob."""
    backend = StorageBuilder(storage_client=client, scheme=Scheme.LIU, liu_params=liu_params).build()
    data = [0.1, 0.2, 0.3, 0.4]
    result = await backend.put(**storage_ids, data=data)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().dtype == np.float64


@pytest.mark.asyncio
async def test_put_list_mixed_int_float_plaintext(client, liu_params, storage_ids):
    """A mixed List[int | float] is cast to float64."""
    backend = StorageBuilder(storage_client=client, scheme=Scheme.LIU, liu_params=liu_params).build()
    data = [1, 2.5, 3, 4.0]
    result = await backend.put(**storage_ids, data=data)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().dtype == np.float64


@pytest.mark.asyncio
async def test_put_list_empty_returns_err(client, liu_params, storage_ids):
    """An empty list returns Err, not an exception."""
    backend = StorageBuilder(storage_client=client, scheme=Scheme.LIU, liu_params=liu_params).build()
    result = await backend.put(**storage_ids, data=[])
    assert result.is_err


@pytest.mark.asyncio
async def test_put_get_list_float_round_trip(client, liu_params, storage_ids):
    """Round-trip: List[float] put then get returns the same values."""
    backend = StorageBuilder(storage_client=client, scheme=Scheme.LIU, liu_params=liu_params).build()
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = await backend.put(**storage_ids, data=data)
    assert result.is_ok, result.unwrap_err()
    get_result = await backend.get(**storage_ids)
    assert get_result.is_ok, get_result.unwrap_err()
    retrieved = get_result.unwrap().raw_value
    np.testing.assert_array_almost_equal(retrieved.flatten(), np.array(data, dtype=np.float64))


@pytest.mark.asyncio
async def test_put_list_int_segment(client, liu_params, storage_ids):
    """List[int] with segment=True splits into plaintext chunks correctly."""
    backend = StorageBuilder(
        storage_client = client,
        scheme         = Scheme.LIU,
        liu_params     = liu_params,
    ).with_storage_params(StorageParams(num_chunks=2)).build()
    data = [10, 20, 30, 40, 50, 60, 70, 80]
    result = await backend.put(**storage_ids, data=data, segment=True)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().dtype == np.float64


@pytest.mark.asyncio
async def test_put_list_int_encrypt_ckks(client, ckks, ckks_params, storage_ids):
    """List[int] with encrypt=True on a CKKS backend uses the vector CKKS path."""
    backend = ckks_builder(client, ckks, ckks_params)
    data = [1, 2, 3, 4]
    result = await backend.put(**storage_ids, data=data, encrypt=True)
    assert result.is_ok, result.unwrap_err()
    get_result = await backend.get(**storage_ids, encrypt=True)
    assert get_result.is_ok, get_result.unwrap_err()
    assert len(get_result.unwrap().raw_value) > 0


# ---------------------------------------------------------------------------
# scheme=None (plaintext-only backend) — unit tests (no network)
# ---------------------------------------------------------------------------

def test_builder_no_scheme(client):
    """StorageBuilder with no scheme builds successfully; backend.scheme is None."""
    backend = StorageBuilder(storage_client=client).build()
    assert backend.scheme is None


def test_builder_no_scheme_round_trip(client):
    """as_builder() preserves scheme=None through a round-trip."""
    backend = StorageBuilder(storage_client=client).build()
    cloned = backend.as_builder().build()
    assert cloned.scheme is None


# ---------------------------------------------------------------------------
# scheme=None — integration tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_put_encrypt_no_scheme_returns_err(client, small_matrix, storage_ids):
    """encrypt=True with scheme=None returns Err instead of silently storing plaintext."""
    backend = StorageBuilder(storage_client=client).build()
    result = await backend.put(**storage_ids, data=small_matrix, encrypt=True)
    assert result.is_err
    assert "scheme" in str(result.unwrap_err()).lower()


@pytest.mark.asyncio
async def test_put_no_scheme_plaintext(client, small_matrix, storage_ids):
    """scheme=None stores a plaintext ndarray without errors."""
    backend = StorageBuilder(storage_client=client).build()
    result = await backend.put(**storage_ids, data=small_matrix)
    assert result.is_ok, result.unwrap_err()
    assert result.unwrap().shape == small_matrix.shape


@pytest.mark.asyncio
async def test_put_get_no_scheme_plaintext(client, small_matrix, storage_ids):
    """scheme=None round-trip: put then get returns the original matrix."""
    backend = StorageBuilder(storage_client=client).build()
    await backend.put(**storage_ids, data=small_matrix)
    get_result = await backend.get(**storage_ids)
    assert get_result.is_ok, get_result.unwrap_err()
    np.testing.assert_array_almost_equal(get_result.unwrap().raw_value, small_matrix)


@pytest.mark.asyncio
async def test_put_get_no_scheme_segment(client, small_matrix, storage_ids):
    """scheme=None with segment=True stores and retrieves chunked plaintext."""
    backend = (
        StorageBuilder(storage_client=client)
        .with_storage_params(StorageParams(num_chunks=2))
        .build()
    )
    result = await backend.put(**storage_ids, data=small_matrix, segment=True)
    assert result.is_ok, result.unwrap_err()
    get_result = await backend.get(**storage_ids, segment=True)
    assert get_result.is_ok, get_result.unwrap_err()
    assert isinstance(get_result.unwrap().raw_value, np.ndarray)