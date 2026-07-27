import pytest
import warnings
import numpy as np
import numpy.typing as npt
from mictlanx import AsyncClient
from mictlanx.utils.segmentation import Chunks
from rorycommon import Common as RoryCommon, LiuParams
from concurrent.futures import ProcessPoolExecutor
from rory.core.security.dataowner import DataOwner


@pytest.mark.asyncio
async def test_liu(
    dataowner:DataOwner,
    key:str,
    generated_matrix:npt.NDArray[np.float64],
    executor:ProcessPoolExecutor,
    get_context:dict
):
    RORY_MAX_WORKERS = get_context["max_workers"]

    n   = generated_matrix.shape[0]*generated_matrix.shape[1]*dataowner.m
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        emt = RoryCommon.segment_and_encrypt_liu_with_executor(
            executor         = executor,
            dataowner        = dataowner,
            key              = key,
            n                = n,
            np_random        = True,
            num_chunks       = RORY_MAX_WORKERS,
            plaintext_matrix = generated_matrix
        )
    assert len(emt) == RORY_MAX_WORKERS


def test_liu_with_executor_timed_returns_tuple(
    dataowner: DataOwner,
    key: str,
    generated_matrix: npt.NDArray[np.float64],
    executor: ProcessPoolExecutor,
    get_context: dict,
):
    RORY_MAX_WORKERS = get_context["max_workers"]
    n = generated_matrix.shape[0] * generated_matrix.shape[1] * dataowner.m

    result = RoryCommon.segment_and_encrypt_liu_with_executor_timed(
        executor         = executor,
        dataowner        = dataowner,
        key              = key,
        n                = n,
        np_random        = True,
        num_chunks       = RORY_MAX_WORKERS,
        plaintext_matrix = generated_matrix,
    )

    chunks, segment_time, encrypt_time = result
    assert isinstance(chunks, Chunks)
    assert len(chunks) == RORY_MAX_WORKERS
    assert segment_time >= 0.0
    assert encrypt_time >= 0.0


def test_liu_timed_returns_tuple(
    dataowner: DataOwner,
    key: str,
    generated_matrix: npt.NDArray[np.float64],
    get_context: dict,
):
    RORY_MAX_WORKERS = get_context["max_workers"]
    n = generated_matrix.shape[0] * generated_matrix.shape[1] * dataowner.m

    result = RoryCommon.segment_and_encrypt_liu_timed(
        key              = key,
        dataowner        = dataowner,
        plaintext_matrix = generated_matrix,
        n                = n,
        np_random        = True,
        num_chunks       = RORY_MAX_WORKERS,
        max_workers      = RORY_MAX_WORKERS,
    )

    chunks, segment_time, encrypt_time = result
    assert isinstance(chunks, Chunks)
    assert len(chunks) == RORY_MAX_WORKERS
    assert segment_time >= 0.0
    assert encrypt_time >= 0.0


def test_liu_deprecated_warns(
    dataowner: DataOwner,
    key: str,
    generated_matrix: npt.NDArray[np.float64],
    executor: ProcessPoolExecutor,
    get_context: dict,
):
    RORY_MAX_WORKERS = get_context["max_workers"]
    n = generated_matrix.shape[0] * generated_matrix.shape[1] * dataowner.m

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        RoryCommon.segment_and_encrypt_liu_with_executor(
            executor         = executor,
            dataowner        = dataowner,
            key              = key,
            n                = n,
            np_random        = True,
            num_chunks       = RORY_MAX_WORKERS,
            plaintext_matrix = generated_matrix,
        )
    assert len(w) == 1
    assert issubclass(w[0].category, DeprecationWarning)
    assert "1.0.0" in str(w[0].message)


def test_liu_no_executor_deprecated_warns(
    dataowner: DataOwner,
    key: str,
    generated_matrix: npt.NDArray[np.float64],
    get_context: dict,
):
    RORY_MAX_WORKERS = get_context["max_workers"]
    n = generated_matrix.shape[0] * generated_matrix.shape[1] * dataowner.m

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        RoryCommon.segment_and_encrypt_liu(
            key              = key,
            dataowner        = dataowner,
            plaintext_matrix = generated_matrix,
            n                = n,
            np_random        = True,
            num_chunks       = RORY_MAX_WORKERS,
            max_workers      = RORY_MAX_WORKERS,
        )
    assert len(w) == 1
    assert issubclass(w[0].category, DeprecationWarning)
    assert "1.0.0" in str(w[0].message)




def test_liu_with_initialized_executor_timed_returns_tuple(
    key: str,
    generated_matrix: npt.NDArray[np.float64],
    liu_params: LiuParams,
    get_context: dict,
):
    RORY_MAX_WORKERS = get_context["max_workers"]
    n = generated_matrix.size

    result = RoryCommon.segment_and_encrypt_liu_with_initialized_executor_timed(
        key              = key,
        plaintext_matrix = generated_matrix,
        n                = n,
        np_random        = True,
        liu_params       = liu_params,
        num_chunks       = RORY_MAX_WORKERS,
    )

    chunks, segment_time, encrypt_time = result
    assert isinstance(chunks, Chunks)
    assert len(chunks) == RORY_MAX_WORKERS
    assert segment_time >= 0.0
    assert encrypt_time >= 0.0


def test_liu_initialized_executor_chunk_count_matches_num_chunks(
    key: str,
    generated_matrix: npt.NDArray[np.float64],
    liu_params: LiuParams,
):
    for num_chunks in [1, 2]:
        chunks, _, _ = RoryCommon.segment_and_encrypt_liu_with_initialized_executor_timed(
            key              = key,
            plaintext_matrix = generated_matrix,
            n                = generated_matrix.size,
            np_random        = True,
            liu_params       = liu_params,
            num_chunks       = num_chunks,
        )
        assert len(chunks) == num_chunks


def test_liu_initialized_executor_encrypt_chunk_raises_without_init():
    """encrypt_chunk_liu_with_initialized_executor raises if the global is not set."""
    import rorycommon as rc
    original = rc.Common.liu_dataowner
    try:
        rc.Common.liu_dataowner = None
        from mictlanx.utils.segmentation import Chunk
        from option import Some
        dummy = Chunk.from_ndarray(
            group_id = "test",
            index    = 0,
            ndarray  = np.zeros((2, 2)),
            chunk_id = Some("test_0"),
        )
        with pytest.raises(Exception, match="not initialized"):
            RoryCommon.encrypt_chunk_liu_with_initialized_executor("test", dummy, True)
    finally:
        rc.Common.liu_dataowner = original


@pytest.mark.asyncio
async def test_fdhope(
    generated_matrix:np.ndarray,
    dataowner:DataOwner,
    key:str,
    executor:ProcessPoolExecutor,
    get_context:dict
):
    RORY_MAX_WORKERS = get_context["max_workers"]

    n          = generated_matrix.shape[1]*generated_matrix.shape[1]*generated_matrix.shape[0]
    sens       = 0.2
    scheme  = "DBSKMEANS"
    udm        = dataowner.get_U(
        algorithm     = scheme,
        plaintext_matrix = generated_matrix
    )
    # print(udm)
    emt = RoryCommon.segment_and_encrypt_fdhope_with_executor(
        executor         = executor,
        scheme        = scheme,
        key              = key,
        dataowner        = dataowner,
        matrix           = udm,
        n                = n,
        num_chunks       = RORY_MAX_WORKERS,
        sens             = sens
    )
    for c in emt:
        print(c)


@pytest.mark.asyncio
async def test_ckks(generated_matrix:npt.NDArray[np.float64],key:str, initialized_executor:ProcessPoolExecutor, get_context:dict):
    RORY_MAX_WORKERS               = get_context["max_workers"]
    RORY_COMMON_CTX_FILENAME       = get_context["ctx"]
    RORY_COMMON_PUBKEY_FILENAME    = get_context["pubkey"]
    RORY_COMMON_SECRETKEY_FILENAME = get_context["secretkey"]
    RORY_KEYS_PATH                 = get_context["keys_path"]
    pmt                            = generated_matrix
    n                              = pmt.shape[1]*pmt.shape[1]
    (emt,_,_) = RoryCommon.segment_and_encrypt_ckks_with_initialized_executor(
        key                = key,
        plaintext_matrix   = pmt,
        n                  = n,
        num_chunks         = RORY_MAX_WORKERS,
        _round             = False,
        ctx_filename       = RORY_COMMON_CTX_FILENAME,
        pubkey_filename    = RORY_COMMON_PUBKEY_FILENAME,
        secretkey_filename = RORY_COMMON_SECRETKEY_FILENAME,
        path               = RORY_KEYS_PATH,
        decimals           = 2
    )
    for c in emt:
        print(c)


@pytest.mark.asyncio
async def test_segement_ckks_encrypt_with_initialized_executor(generated_matrix:npt.NDArray[np.float64],key:str, executor:ProcessPoolExecutor, get_context:dict):
    RORY_MAX_WORKERS               = get_context["max_workers"]
    RORY_COMMON_CTX_FILENAME       = get_context["ctx"]
    RORY_COMMON_PUBKEY_FILENAME    = get_context["pubkey"]
    RORY_COMMON_SECRETKEY_FILENAME = get_context["secretkey"]
    RORY_KEYS_PATH                 = get_context["keys_path"]

    n          = generated_matrix.shape[1]*generated_matrix.shape[1]
    (emt,_,_) = RoryCommon.segment_and_encrypt_ckks_with_initialized_executor(
        key                = key,
        plaintext_matrix   = generated_matrix,
        n                  = n,
        num_chunks         = RORY_MAX_WORKERS,
        _round             = False,
        ctx_filename       = RORY_COMMON_CTX_FILENAME,
        pubkey_filename    = RORY_COMMON_PUBKEY_FILENAME,
        secretkey_filename = RORY_COMMON_SECRETKEY_FILENAME,
        path               = RORY_KEYS_PATH,
        decimals           = 2
    )
    for c in emt:
        print(c)

@pytest.mark.asyncio
async def test_segement_ckks_encrypt_put_chunks_with_initialized_executor(
    ball_id:str,
    bucket_id:str,
    key:str,
    generated_matrix:npt.NDArray[np.float64],
    client:AsyncClient,
    get_context:dict
):
    RORY_MAX_WORKERS               = get_context["max_workers"]
    RORY_COMMON_CTX_FILENAME       = get_context["ctx"]
    RORY_COMMON_PUBKEY_FILENAME    = get_context["pubkey"]
    RORY_COMMON_SECRETKEY_FILENAME = get_context["secretkey"]
    RORY_COMMON_RELINKEY_FILENAME  = get_context["relinkey"]
    RORY_COMMON_ROTATEKEY_FILENAME = get_context["rotatekey"]
    RORY_KEYS_PATH                 = get_context["keys_path"]
    max_attempts = 3
    tags = {"ball_id": ball_id, "bucket_id": bucket_id}
    decimals = 2

    n          = generated_matrix.shape[1]*generated_matrix.shape[1]
    (emt,_,_,_) = await RoryCommon.segement_and_encrypt_ckks_with_initialized_executor_put_chunks(
        client             = client,
        bucket_id          = bucket_id,
        ball_id            = ball_id,
        key                = key,
        plaintext_matrix   = generated_matrix,
        n                  = n,
        num_chunks         = RORY_MAX_WORKERS,
        _round             = False,
        keys_path          = RORY_KEYS_PATH,
        ctx_filename       = RORY_COMMON_CTX_FILENAME,
        pubkey_filename    = RORY_COMMON_PUBKEY_FILENAME,
        secretkey_filename = RORY_COMMON_SECRETKEY_FILENAME,
        max_retries        = max_attempts,
        relinkey_filename  = RORY_COMMON_RELINKEY_FILENAME,
        rotatekey_filename = RORY_COMMON_ROTATEKEY_FILENAME,
        tags               = tags,
        decimals           = decimals
    )
    assert emt.is_ok
