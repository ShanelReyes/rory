import pytest
from rorycommon import Common as RoryCommon
import numpy as np
from option import Some
from mictlanx import AsyncClient
from mictlanx.utils.segmentation import Chunks
from concurrent.futures import ProcessPoolExecutor
from rory.core.security.dataowner import DataOwner
import numpy.typing as npt
from rory.core.utils.constants import Constants
from rory.core.clustering.secure.pqc.skmeans import Skmeans as SkmeansPQC
from uuid import uuid4


@pytest.mark.asyncio
async def test_ckks_encrypt_put_and_get_chunk(dataowner_pqc, client,generated_matrix,get_context):
    # bucket_id = "rory"
    MICTLANX_BUCKET_ID = get_context["bucket_id"]
    RORY_COMMON_RECORDS = get_context["records"]
    RORY_COMMON_ATTRIBUTES = get_context["attributes"]
    RORY_MAX_WORKERS = get_context["max_workers"]
    ball_id   = uuid4().hex.replace("-","")
    res = await RoryCommon.encrypt_ckks_and_put_chunk(
        client       = client,
        ball_id      = ball_id,
        bucket_id    = MICTLANX_BUCKET_ID,
        dataowner    = dataowner_pqc,
        index        = 0,
        full_shape   = (RORY_COMMON_RECORDS, RORY_COMMON_ATTRIBUTES),
        max_attempts = 5,
        ndarray      = generated_matrix,
        num_chunks   = RORY_MAX_WORKERS,
        max_backoff  = 5,
        timeout      = 120


    )
    assert res.is_ok, "Failed to encrypt and put chunk: {}".format(res.unwrap_err())

    res = await RoryCommon.get_pyctxt_chunk(
        client    = client,
        ckks      = dataowner_pqc.scheme,
        ball_id   = ball_id,
        bucket_id = MICTLANX_BUCKET_ID,
        index     = 0
    )
    assert res.is_ok, "Failed to get chunk: {}".format(res.unwrap_err())




@pytest.mark.asyncio
async def test_liu_segment_encrypt_and_put_chunks(dataowner, client,generated_matrix,get_context):
    key                = uuid4().hex.replace("-","")
    n                  = generated_matrix.shape[0]*generated_matrix.shape[1]*dataowner.m
    RORY_MAX_WORKERS   = get_context["max_workers"]
    MICTLANX_BUCKET_ID = get_context["bucket_id"]
    emt = RoryCommon.segment_and_encrypt_liu_with_executor(
        executor         = ProcessPoolExecutor(max_workers=RORY_MAX_WORKERS),
        dataowner        = dataowner,
        key              = key,
        n                = n,
        np_random        = True,
        num_chunks       = RORY_MAX_WORKERS,
        plaintext_matrix = generated_matrix
    )
    # emt.sort()
    for c in emt:
        print(c.chunk_id,c.checksum,c.to_ndarray())
    # checksum,_ = XoloUtils.sha256_stream(emt.to_generator())
    # checksum1= XoloUtils.sha256(emt.to_bytes())


    put_res = await RoryCommon.put_chunks(
        client    = client,
        key       = key,
        bucket_id = MICTLANX_BUCKET_ID,
        chunks    = emt,
        tags      = {
            "full_shape": str((generated_matrix.shape[0],generated_matrix.shape[1],dataowner.m)),
            "full_dtype": str(generated_matrix.dtype)
        }
    ) 
    assert put_res.is_ok, "Failed to put chunks: {}".format(put_res.unwrap_err())

    get_res = await RoryCommon.get_and_merge_safe(
        client = client,
        key = key,
        bucket_id = MICTLANX_BUCKET_ID
    )
    assert get_res.is_ok, "Failed to get and merge chunks: {}".format(get_res.unwrap_err())
    merged_array = get_res.unwrap()
    assert merged_array.shape == (generated_matrix.shape[0],generated_matrix.shape[1],dataowner.m), "Merged array shape mismatch: expected {}, got {}".format((generated_matrix.shape[0],generated_matrix.shape[1],dataowner.m), merged_array.shape)


@pytest.mark.asyncio
async def test_liu_shortcut_segment_encrypt_and_put_chunks_with_executor(
    executor:ProcessPoolExecutor,
    dataowner:DataOwner,
    client:AsyncClient,
    generated_matrix:npt.NDArray[np.float64],
    get_context:dict
):
    key                = uuid4().hex.replace("-","")
    n                  = generated_matrix.shape[0]*generated_matrix.shape[1]*dataowner.m
    RORY_MAX_WORKERS   = get_context["max_workers"]
    MICTLANX_BUCKET_ID = get_context["bucket_id"]
    
    pack = (await RoryCommon.segment_and_encrypt_liu_and_put_chunks(
        executor   = executor,
        dataowner  = dataowner,
        n          = n,
        key        = key,
        num_chunks = RORY_MAX_WORKERS,
        bucket_id  = MICTLANX_BUCKET_ID,
        client     = client,
        matrix     = generated_matrix,
        np_random  = True,
        tags       = {}
    ))

    (emt,_,_) = pack

    assert emt.is_ok, "Failed to segment, encrypt and put chunks: {}".format(emt.unwrap_err())




# @pytest.mark.skip("")
@pytest.mark.asyncio
async def test_ckks_segment_encrypt_delete_put_get_chunks(key,ckks, client,generated_matrix,get_context):
    # key                            = uuid4().hex.replace("-","")
    n                              = generated_matrix.shape[1]*generated_matrix.shape[1]
    num_chunks                     = get_context["max_workers"]
    RORY_COMMON_CTX_FILENAME       = get_context["ctx"]
    RORY_COMMON_PUBKEY_FILENAME    = get_context["pubkey"]
    RORY_COMMON_SECRETKEY_FILENAME = get_context["secretkey"]
    RORY_KEYS_PATH                 = get_context["keys_path"]
    MICTLANX_BUCKET_ID             = get_context["bucket_id"]

    (emt,_,_) = RoryCommon.segment_and_encrypt_ckks_with_initialized_executor(
        # executor           = ProcessPoolExecutor(max_workers=num_chunks),
        key                = key,
        plaintext_matrix   = generated_matrix,
        n                  = n,
        num_chunks         = num_chunks,
        _round             = False,
        ctx_filename       = RORY_COMMON_CTX_FILENAME,
        pubkey_filename    = RORY_COMMON_PUBKEY_FILENAME,
        secretkey_filename = RORY_COMMON_SECRETKEY_FILENAME,
        path               = RORY_KEYS_PATH,
        decimals           = 2
    ) 
    assert len(emt) == num_chunks

    res = await RoryCommon.delete_and_put_chunks(
        client    = client,
        bucket_id = MICTLANX_BUCKET_ID,
        key       = key,
        chunks    = emt,
        tags      = {
            "x": "Something_test"
        }
    )
    assert res.is_ok, "Failed to delete and put chunks: {}".format(res.unwrap_err())
    
    x = await RoryCommon.get_pyctxt(
        client    = client,
        key       = key,
        bucket_id = MICTLANX_BUCKET_ID,
        ckks      = ckks
    )
    assert x is not None, "Failed to get pyctxt: {}".format(x)






# @pytest.mark.skip("")
@pytest.mark.asyncio
async def test_full_skmeans_pqc(executor,dataowner_pqc, client,generated_matrix,ckks,get_context):
    # executor            = ProcessPoolExecutor(max_workers=2)
    RORY_MAX_WORKERS               = get_context["max_workers"]
    MICTLANX_BUCKET_ID             = get_context["bucket_id"]
    RORY_COMMON_CTX_FILENAME       = get_context["ctx"]
    RORY_COMMON_PUBKEY_FILENAME    = get_context["pubkey"]
    RORY_COMMON_SECRETKEY_FILENAME = get_context["secretkey"]
    RORY_KEYS_PATH                 = get_context["keys_path"]
    init_sm_id                     = f"initsmid{uuid4().hex.replace('-','')}"
    # num_chunks                     = 2
    _round                         = False
    decimals                       = 2
    dataowner                      = dataowner_pqc
    plaintext_matrix               = generated_matrix
    encrypted_matrix_id            = uuid4().hex.replace("-","")
    r                              = plaintext_matrix.shape[0]
    a                              = plaintext_matrix.shape[1]
    n                              = a*r
    k                              = 2
    udm_id              = f"udmid{uuid4().hex.replace('-','')}"
    encrypted_shift_matrix_id = f"encryptedshiftmatrixid{uuid4().hex.replace('-','')}"

    (encrypted_matrix_chunks,_,_) = RoryCommon.segment_and_encrypt_ckks_with_initialized_executor( #Encrypt 
        # executor           = executor,
        key                = encrypted_matrix_id,
        plaintext_matrix   = plaintext_matrix,
        n                  = n,
        _round             = _round,
        decimals           = decimals,
        path               = RORY_KEYS_PATH,
        ctx_filename       = RORY_COMMON_CTX_FILENAME,
        pubkey_filename    = RORY_COMMON_PUBKEY_FILENAME,
        secretkey_filename = RORY_COMMON_SECRETKEY_FILENAME,
        num_chunks         = RORY_MAX_WORKERS,
    )
    put_encrypted_matrix_result = await RoryCommon.delete_and_put_chunks(
        client = client,
        bucket_id      = MICTLANX_BUCKET_ID,
        key            = encrypted_matrix_id,
        chunks         = encrypted_matrix_chunks,
        tags = {
            "full_shape": str((r,a)),
            "full_dtype":"float64"
        }
    )
    assert put_encrypted_matrix_result.is_ok, "Failed to put encrypted matrix chunks: {}".format(put_encrypted_matrix_result.unwrap_err())

    # ===================================================
    zero_shiftmatrix = np.zeros((k, a))
    n2 = a*k
    (encrypted_zero_shiftmatrix_chunks,_,_) = RoryCommon.segment_and_encrypt_ckks_with_initialized_executor( #Encrypt 
        # executor           = executor,
        key                = init_sm_id,
        plaintext_matrix   = zero_shiftmatrix,
        n                  = n2,
        num_chunks         = RORY_MAX_WORKERS,
        _round             = _round,
        decimals           = decimals,
        path               = RORY_KEYS_PATH,
        ctx_filename       = RORY_COMMON_CTX_FILENAME,
        pubkey_filename    = RORY_COMMON_PUBKEY_FILENAME,
        secretkey_filename = RORY_COMMON_SECRETKEY_FILENAME
    )
    assert len(encrypted_zero_shiftmatrix_chunks) == RORY_MAX_WORKERS, "Number of encrypted zero shift matrix chunks mismatch: expected {}, got {}".format(RORY_MAX_WORKERS, len(encrypted_zero_shiftmatrix_chunks))

 
    put_encrypted_zero_shiftmatrix_result = await RoryCommon.delete_and_put_chunks(
        client         = client,
        bucket_id      = MICTLANX_BUCKET_ID,
        key            = init_sm_id,
        chunks         = encrypted_zero_shiftmatrix_chunks,
        tags = {
            "full_shape": str((k,a)),
            "full_dtype":"float64"
        }
    )
    assert put_encrypted_zero_shiftmatrix_result.is_ok, "Failed to put encrypted zero shift matrix chunks: {}".format(put_encrypted_zero_shiftmatrix_result.unwrap_err())
    # # ===================================================
    udm            = dataowner.get_U(
        plaintext_matrix = plaintext_matrix,
        algorithm        = Constants.ClusteringAlgorithms.SKMEANS_PQC
    )

    maybe_udm_matrix_chunks = Chunks.from_ndarray(
        ndarray      = udm,
        group_id     = udm_id,
        chunk_prefix = Some(udm_id),
        num_chunks   = RORY_MAX_WORKERS,
    )


    udm_put_result = await RoryCommon.delete_and_put_chunks(
        client         = client,
        bucket_id      = MICTLANX_BUCKET_ID,
        key            = udm_id,
        chunks         = maybe_udm_matrix_chunks.unwrap(),
        tags = {
            "full_shape": str(udm.shape),
            "full_dtype": str(udm.dtype)
        }
    )
    assert udm_put_result.is_ok, "Failed to put UDM matrix chunks: {}".format(udm_put_result.unwrap_err())
    # # ===================================================
    init_shiftmatrix = await RoryCommon.get_pyctxt(
            client    = client,
            bucket_id = MICTLANX_BUCKET_ID,
            key       = init_sm_id,
            ckks      = ckks,
    )
    

    skmeans = SkmeansPQC(he_object=ckks.he_object, init_shiftmatrix=init_shiftmatrix)
    _encryptedMatrix = await RoryCommon.get_pyctxt(
        client    = client,
        bucket_id = MICTLANX_BUCKET_ID,
        key       = encrypted_matrix_id,
        ckks      = ckks
    )
    _udm = await RoryCommon.get_and_merge(
        client    = client,
        bucket_id = MICTLANX_BUCKET_ID,
        key       = udm_id,
    )



    S1,_Cent_i,_Cent_j, label_vector  = skmeans.run1(
        status= Constants.ClusteringStatus.WORK_IN_PROGRESS, 
        k = k, 
        encryptedMatrix= _encryptedMatrix, 
        Cent_j=init_shiftmatrix,
        num_attributes=plaintext_matrix.shape[1],
        UDM=_udm,
    ).unwrap()

    encrypted_shift_matrix_chunks = RoryCommon.from_pyctxts_to_chunks(key=encrypted_shift_matrix_id, xs = S1,num_chunks=RORY_MAX_WORKERS).unwrap()
    assert len(encrypted_shift_matrix_chunks) == RORY_MAX_WORKERS, "Number of encrypted shift matrix chunks mismatch: expected {}, got {}".format(RORY_MAX_WORKERS, len(encrypted_shift_matrix_chunks))
    z = await RoryCommon.delete_and_put_chunks(
        client = client,
        bucket_id      = MICTLANX_BUCKET_ID,
        key            = encrypted_shift_matrix_id,
        chunks         = encrypted_shift_matrix_chunks
    )
    assert z.is_ok, "Failed to put encrypted shift matrix chunks: {}".format(z.unwrap_err())
    # await asyncio.sleep(5)
    zz = await RoryCommon.get_pyctxt(
        client=client, 
        bucket_id=MICTLANX_BUCKET_ID,
        key=encrypted_shift_matrix_id,
        ckks=ckks,
        delay=2,
        max_retries=20
    )
    assert zz is not None, "Failed to get encrypted shift matrix: {}".format(zz)
    # assert zz.is_ok, "Failed to get encrypted shift matrix: {}".format(zz.unwrap_err())
