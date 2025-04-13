import pytest
from rorycommon import Common as RoryCommon
import os 
import numpy as np
from mictlanx import AsyncClient
from mictlanx.utils import Utils
from concurrent.futures import ProcessPoolExecutor
from rory.core.security.dataowner import DataOwner
from rory.core.security.cryptosystem.liu import Liu
from xolo.utils.utils import Utils as XoloUtils
from rory.core.security.cryptosystem.pqc.ckks import Ckks
dataowner = DataOwner(
    liu_scheme= Liu(
        _round         = True,
        decimals       = 2,
        secure_random  = False,
        seed           = 1,
        use_np_random  = True,
        security_level = 128
    ),
)

key = "encryptedskmeansy"
bucket_id = "rory"

@pytest.mark.skip("")
@pytest.mark.asyncio
async def test_liu():
    pmt = await RoryCommon.read_numpy_from(
        path="/rory/source/clusteringc0r10a5k20.npy",
        extension="npy"
    )
    pmt = pmt.unwrap()
    # print(pmt.dtype)
    # np.random.seed(10)

    n = pmt.shape[0]*pmt.shape[1]*dataowner.m
    num_chunks = 2
    emt = RoryCommon.segment_and_encrypt_liu_with_executor(
        executor= ProcessPoolExecutor(max_workers=num_chunks),
        dataowner=dataowner,
        key=key,
        n=n,
        np_random=True,
        num_chunks=num_chunks,
        plaintext_matrix=pmt
    )

    



@pytest.mark.skip("")
@pytest.mark.asyncio
async def test_fdhope():
    pmt_result = await RoryCommon.read_numpy_from(
        path="/rory/source/clusteringc0r10a5k20.npy",
        extension="npy"
    )
    pmt = pmt_result.unwrap()
    # print(pmt.dtype)
    # np.random.seed(10)

    n          = pmt.shape[1]*pmt.shape[1]*pmt.shape[0]
    num_chunks = 2
    sens       = 0.2
    algorithm  = "DBSKMEANS"
    udm        = dataowner.get_U(
        algorithm        = algorithm,
        plaintext_matrix = pmt
    )
    # print(udm)
    emt = RoryCommon.segment_and_encrypt_fdhope_with_executor(
        executor         = ProcessPoolExecutor(max_workers=num_chunks),
        algorithm        = algorithm,
        key              = key,
        dataowner        = dataowner,
        matrix           = udm,
        n                = n,
        num_chunks       = num_chunks,
        sens             = sens
    )
    for c in emt:
        print(c)

@pytest.mark.skip("")
@pytest.mark.asyncio
async def test_generate_ckks_keys():
    x = Ckks.create_client(
        scheme="CKKS",
        n=8192,
        scale=2**30,
        qi_sizes=[60,40,40,60],
        decimals=2,
        save=True,
        output_path="/rory/keys"

    )
    x.he_object.res
    print(x)
@pytest.mark.skip("")
@pytest.mark.asyncio
async def test_ckks():
    pmt_result = await RoryCommon.read_numpy_from(
        path="/rory/source/clusteringc0r10a5k20.npy",
        extension="npy"
    )
    assert pmt_result.is_ok
    pmt = pmt_result.unwrap()

    n          = pmt.shape[1]*pmt.shape[1]
    num_chunks = 2
    emt = RoryCommon.segment_and_encrypt_ckks_with_executor(
        executor         = ProcessPoolExecutor(max_workers=num_chunks),
        key              = key,
        plaintext_matrix = pmt,
        n                = n,
        num_chunks       = num_chunks,
        _round = False,
        ctx_filename="ctx", 
        pubkey_filename="pubkey",
        secretkey_filename="secretkey",
        path="/rory/keys",
        decimals=2
    )
    for c in emt:
        print(c)

