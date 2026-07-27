import pytest
from rorycommon import Common as RoryCommon
import os 
import numpy as np
from mictlanx import AsyncClient
from concurrent.futures import ProcessPoolExecutor
from dotenv import load_dotenv
from uuid import uuid4
from rory.core.security.cryptosystem.pqc.ckks import Ckks

RORY_COMMON_ENV_FILE_PATH = os.environ.get("RORY_COMMON_ENV_FILE_PATH","./.env.test")
if os.path.exists(RORY_COMMON_ENV_FILE_PATH):
    load_dotenv(dotenv_path=RORY_COMMON_ENV_FILE_PATH)


MICTLANX_CLIENT_ID             = os.environ.get("MICTLANX_CLIENT_ID","{}_mictlanx".format("rory-common"))
MICTLANX_TIMEOUT               = int(os.environ.get("MICTLANX_TIMEOUT",3600))
MICTLANX_API_VERSION           = int(os.environ.get("MICTLANX_API_VERSION","4"))
MICTLANX_MAX_WORKERS           = int(os.environ.get("MICTLANX_MAX_WORKERS","12"))
MICTLANX_BUCKET_ID             = os.environ.get("MICTLANX_BUCKET_ID","rory")
MICTLANX_OUTPUT_PATH           = os.environ.get("MICTLANX_OUTPUT_PATH","/rory/mictlanx")
MICTLANX_PROTOCOL              = os.environ.get("MICTLANX_PROTOCOL","http")
MICTLANX_URI                   = os.environ.get("MICTLANX_URI",f"mictlanx://mictlanx-router-0@localhost:63666?api_version={MICTLANX_API_VERSION}&protocol={MICTLANX_PROTOCOL}")
MICTLANX_DEBUG                 = bool(int(os.environ.get("MICTLANX_DEBUG",0)))
RORY_KEYS_PATH                 = os.environ.get("RORY_KEYS_PATH","/rory/keys")
RORY_COMMON_CTX_FILENAME       = os.environ.get("RORY_COMMON_CTX_FILENAME","ctx")
RORY_COMMON_PUBKEY_FILENAME    = os.environ.get("RORY_COMMON_PUBKEY_FILENAME","pubkey")
RORY_COMMON_SECRETKEY_FILENAME = os.environ.get("RORY_COMMON_SECRETKEY_FILENAME","secretkey")
RORY_COMMON_RELINKEY_FILENAME  = os.environ.get("RORY_COMMON_RELINKEY_FILENAME","relinkey")
RORY_COMMON_ROTATEKEY_FILENAME = os.environ.get("RORY_COMMON_ROTATEKEY_FILENAME","")
RORY_COMMON_SECURITY_LEVEL     = int(os.environ.get("RORY_COMMON_SECURITY_LEVEL",128))
RORY_COMMON_RECORDS            = int(os.environ.get("RORY_COMMON_RECORDS","10"))
RORY_COMMON_ATTRIBUTES         = int(os.environ.get("RORY_COMMON_ATTRIBUTES","10"))

@pytest.fixture(scope="module")
def ckks():
    _ = Ckks.create_client(
        scheme             = "CKKS",
        decimals           = 2,
        enable_relinearize = True,
        security_level     = 128,
        save               = True,
        output_path        = RORY_KEYS_PATH
    )
    ckks = Ckks.from_pyfhel(
        path=RORY_KEYS_PATH,
    )
    return ckks




@pytest.mark.asyncio
async def test_encrypt_vector(client:AsyncClient,ckks: Ckks):
    # Plain bias
    # bias = np.array([0.],dtype=np.float32)
    bias = np.zeros(shape=10,dtype=np.float32)

    # ppe = ProcessPoolExecutor(max_workers=1)

    key            = f"test_key_{uuid4().hex[:4]}"

    (encrypted_bias,_,_) = RoryCommon.encrypt_vector_ckks_with_initialized_executor(
        # executor           = ppe,
        key                = key,
        vector             = bias,
        _round             = True,
        decimals           = 2,
        path               = RORY_KEYS_PATH,
        ctx_filename       = RORY_COMMON_CTX_FILENAME,
        pubkey_filename    = RORY_COMMON_PUBKEY_FILENAME,
        secretkey_filename = RORY_COMMON_SECRETKEY_FILENAME
    )
    
    print(encrypted_bias)
    put_result = await RoryCommon.delete_and_put_chunks(
        client    = client,
        bucket_id = MICTLANX_BUCKET_ID,
        key       = key,
        chunks    = encrypted_bias,
        timeout   = MICTLANX_TIMEOUT
    )
    assert put_result.is_ok

    get_result = await RoryCommon.get_pyctxt(
        client    = client,
        bucket_id = MICTLANX_BUCKET_ID,
        key       = key,
        timeout   = MICTLANX_TIMEOUT,
        ckks      = ckks
    )
    assert len(get_result) == len(bias)
    num_chunks = 2
    maybe_chunks = RoryCommon.from_pyctxts_to_chunks(key=key,xs=get_result,num_chunks=num_chunks)
    assert maybe_chunks.is_some
    chunks = maybe_chunks.unwrap()
    result = await RoryCommon.put_chunks(
        client    = client,
        bucket_id = MICTLANX_BUCKET_ID,
        key       = key,
        chunks    = chunks,
    )
    assert result.is_ok

    get_result2 = await RoryCommon.get_pyctxt(
        client    = client,
        bucket_id = MICTLANX_BUCKET_ID,
        key       = key,
        timeout   = MICTLANX_TIMEOUT,
        ckks      = ckks
    )

    assert len(get_result2) == len(bias) and len(get_result2) == len(get_result)
 