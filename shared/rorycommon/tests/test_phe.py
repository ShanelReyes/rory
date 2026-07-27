import pytest
from rory.core.security.dataowner_paillier import DataOwner
from rorycommon import Common as RoryCommon
import numpy as np
from concurrent.futures import ProcessPoolExecutor 
import time as T
from phe import generate_paillier_keypair,EncryptedNumber
import os





@pytest.mark.skip(reason="This test is for development purposes only and should not be run in CI.")
@pytest.mark.asyncio
async def test_put_chunk(client):
    ball_id = "x"
    do = DataOwner(securitylevel=128)
    do.generate_keys()
    t1           = T.time()
    dataset_size = (100,10)
    xs           = np.random.random(size=dataset_size )
    res = await RoryCommon.encrypt_paillier_and_put_chunk(
        client=client,
        bucket_id="rory",
        ball_id="phex",
        dataowner=do,
        full_shape=(1000,10),
        index=1,
        max_attempts=10,
        max_backoff=5,
        ndarray=xs,
        num_chunks=10,
        timeout=120,
    )
    print(res)
    # max_workers  = 2
    # ppe          = ProcessPoolExecutor(max_workers=max_workers)
    # res          = RoryCommon.segment_and_encrypt_paillier_with_executor(executor=ppe, key="x",dataowner=do,num_chunks=max_workers,plaintext_matrix=xs,n=len(xs))
    # for c in res:
        # res = RoryCommon.paill
        # print(c)
    # res = await RoryCommon.get_paillier_chunk_or_error(client=client, bucket_id="rory",ball_id=ball_id,index=1)
    # print(res)

@pytest.mark.skip("")
@pytest.mark.asyncio
async def test_get_chunk(client):
    ball_id = "phex"
    res = await RoryCommon.get_paillier_chunk_or_error(client=client, bucket_id="rory",ball_id=ball_id,index=1)
    print(res)


@pytest.mark.skip("")
@pytest.mark.asyncio
async def test_get(client):
    ball_id = "phex"
    res = await RoryCommon.get_paillier_matrix(client=client, bucket_id="rory",ball_id=ball_id)
    print(res)
@pytest.mark.skip("")
@pytest.mark.asyncio
async def test_segment_and_encrypt(client):
    do = DataOwner(securitylevel=128)
    do.generate_keys()
    t1           = T.time()
    dataset_size = (1000,10)
    xs           = np.random.random(size=dataset_size )
    max_workers  = 2
    ppe          = ProcessPoolExecutor(max_workers=max_workers)
    res          = RoryCommon.segment_and_encrypt_paillier_with_executor(executor=ppe, key="x",dataowner=do,num_chunks=max_workers,plaintext_matrix=xs,n=len(xs))
    result = await RoryCommon.put_chunks(client=client,key ="phe", bucket_id="rory",chunks=res,tags={
        "full_shape": str(xs.shape), 
        "dtype":"object"
    })
    print(result)
    # print(res)
    # for c in res:
        # print(RoryCommon.deserialize_matrix_with_pickle(c.data))

@pytest.mark.skip("")
def test_phe_serialize():
    pk,sk = generate_paillier_keypair(n_length=128)
    x = 10.5
    res = pk.encrypt(value=x)
    y= res.ciphertext(be_secure=False)
    res_d = sk.decrypt(encrypted_number=EncryptedNumber(public_key=pk, ciphertext=y, exponent=res.exponent))
    print(res_d,res.exponent)


@pytest.mark.skip("test phe parallel")
def test_paralell_phe():
    t1  = T.time()
    dataset_size = (10000,10)
    xs = np.random.random(size=dataset_size )
    max_workers = 2
    ppe = ProcessPoolExecutor(max_workers=max_workers)
    sls = [128,192,256]
    print(f"Benchmark Config - SECURITY LEVELS: {sls} DATASET SIZE: {dataset_size} MAX WORKERS: {max_workers}")
    # security_level = 128
    for security_level in sls:
    # do = DataOwner(securitylevel=security_level)
    # do.generate_keys(output_path="/rory/schemes_experiments/keys",filename=f"rory-phe-{security_level}",save=True)
        do = DataOwner.from_keys(path="/rory/schemes_experiments/keys",filename=f"rory-phe-{security_level}")
        res = RoryCommon.segment_and_encrypt_paillier_with_executor(dataowner=do,executor=ppe,key="x",plaintext_matrix=xs,n=len(xs))
        print(f"Security Level: {security_level} - Dataset Size: {dataset_size} - ST = {T.time()-t1}" )

@pytest.mark.skip("test phe parallel")
def test_generate_keys():
    sl = 256
    do = DataOwner(securitylevel=sl)
    do.generate_keys(output_path="/rory/keys",filename=f"rory-phe-{sl}",save=True)