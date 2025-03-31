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
MICTLANX_CLIENT_ID           = os.environ.get("MICTLANX_CLIENT_ID","{}_mictlanx".format("rory-common"))
MICTLANX_TIMEOUT             = int(os.environ.get("MICTLANX_TIMEOUT",3600))
MICTLANX_API_VERSION         = int(os.environ.get("MICTLANX_API_VERSION","3"))
MICTLANX_ROUTERS             = os.environ.get("MICTLANX_ROUTERS", "mictlanx-router-0:localhost:60666") #mictlanx-peer-2:localhost:7002")
MICTLANX_DEBUG               = bool(int(os.environ.get("MICTLANX_DEBUG",0)))
MICTLANX_DAEMON              = bool(int(os.environ.get("MICTLANX_DAEMON",1)))
MICTLANX_SHOW_METRICS        = bool(int(os.environ.get("MICTLANX_SHOW_METRICS",0)))
MICTLANX_DISABLED_LOG        = bool(int(os.environ.get("MICTLANX_DISABLED_LOG",0)))
MICTLANX_MAX_WORKERS         = int(os.environ.get("MICTLANX_MAX_WORKERS","12"))
MICTLANX_CLIENT_LB_ALGORITHM = os.environ.get("MICTLANX_CLIENT_LB_ALGORITHM","2CHOICES_UF")
MICTLANX_BUCKET_ID           = os.environ.get("MICTLANX_BUCKET_ID","rory") 
MICTLANX_OUTPUT_PATH         = os.environ.get("MICTLANX_OUTPUT_PATH","/rory/mictlanx")

client = AsyncClient(
    client_id=MICTLANX_CLIENT_ID,
    capacity_storage="200mb",
    debug=False,
    eviction_policy="LRU",
    max_workers= MICTLANX_MAX_WORKERS,
    routers=list(Utils.routers_from_str(routers_str=MICTLANX_ROUTERS,protocol="https")),
    verify=False
)
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
async def test_put_chunks():
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
    # emt.sort()
    for c in emt:
        print(c.chunk_id,c.checksum,c.to_ndarray())
    checksum,_ = XoloUtils.sha256_stream(emt.to_generator())
    checksum1= XoloUtils.sha256(emt.to_bytes())
    print("FULL_H1",checksum)
    print("FULL_H2",checksum1)
    put_res = await RoryCommon.put_chunks(
        client=client,
        key=key,
        bucket_id=bucket_id,
        chunks=emt,
        tags={
            "full_shape":str((pmt.shape[0],pmt.shape[1],dataowner.m)),
            "full_dtype":str(pmt.dtype)
        }
    ) 
    print(put_res)

    

@pytest.mark.skip("")
@pytest.mark.asyncio
async def test_get_and_merge():
    # key = "encryptedskmeansy"
    # bucket_id = "rory"
    x = await RoryCommon.get_and_merge(
        client = client, 
        key = key, 
        bucket_id = bucket_id
    )
    # xx = dataowner.liu_scheme.decryptMatrix(x)
    # print(xx)
    
    # print(x)
    