import os
# from httpx import AsyncClient
import pytest
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from rory.core.security.dataowner import DataOwner
from rory.core.security.pqc.dataowner import DataOwner as DataOwnerPQC
from rory.core.security.cryptosystem.liu import Liu
from rory.core.security.cryptosystem.pqc.ckks import Ckks,CkksModes
from uuid import uuid4
from dotenv import load_dotenv

RORY_COMMON_ENV_FILE_PATH = os.environ.get("RORY_COMMON_ENV_FILE_PATH","./.env.test")
if os.path.exists(RORY_COMMON_ENV_FILE_PATH):
    load_dotenv(dotenv_path=RORY_COMMON_ENV_FILE_PATH)

from mictlanx import AsyncClient
from rorycommon import Common as RoryCommon, CkksParams, LiuParams, FdhopeParams



RORY_MAX_WORKERS                            = int(os.environ.get("RORY_MAX_WORKERS","2"))
RORY_KEYS_PATH                              = os.environ.get("RORY_KEYS_PATH", "/rory/keys/test2")
RORY_COMMON_CTX_FILENAME                    = os.environ.get("RORY_COMMON_CTX_FILENAME","ctx")
RORY_COMMON_PUBKEY_FILENAME                 = os.environ.get("RORY_COMMON_PUBKEY_FILENAME","pubkey")
RORY_COMMON_SECRETKEY_FILENAME              = os.environ.get("RORY_COMMON_SECRETKEY_FILENAME","secretkey")
RORY_COMMON_RELINKEY_FILENAME               = os.environ.get("RORY_COMMON_RELINKEY_FILENAME","relinkey")
RORY_COMMON_ROTATEKEY_FILENAME              = os.environ.get("RORY_COMMON_ROTATEKEY_FILENAME","rotatekey")
RORY_COMMON_SECURITY_LEVEL                  = int(os.environ.get("RORY_COMMON_SECURITY_LEVEL",128))
RORY_ENABLE_ROTATE_KEY_GENERATION           = bool(int(os.environ.get("RORY_ENABLE_ROTATE_KEY_GENERATION",1)))
RORY_ENABLE_REALINEARIZATION_KEY_GENERATION = bool(int(os.environ.get("RORY_ENABLE_REALINEARIZATION_KEY_GENERATION",1)))
RORY_SOURCE_PATH                            = os.environ.get("RORY_SOURCE_PATH", "/rory/source")
MICTLANX_CLIENT_ID                          = os.environ.get("MICTLANX_CLIENT_ID","{}_mictlanx".format("rory-common"))
MICTLANX_TIMEOUT                            = int(os.environ.get("MICTLANX_TIMEOUT",3600))
x = os.environ.get("MICTLANX_API_VERSION","4")
MICTLANX_API_VERSION                        = int(x)
MICTLANX_MAX_WORKERS                        = int(os.environ.get("MICTLANX_MAX_WORKERS","12"))
MICTLANX_BUCKET_ID                          = os.environ.get("MICTLANX_BUCKET_ID","rory")
MICTLANX_OUTPUT_PATH                        = os.environ.get("MICTLANX_OUTPUT_PATH","/rory/mictlanx")
MICTLANX_PROTOCOL                           = os.environ.get("MICTLANX_PROTOCOL","http")
MICTLANX_URI                                = os.environ.get("MICTLANX_URI",f"mictlanx://mictlanx-router-0@localhost:63666?api_version={MICTLANX_API_VERSION}&protocol={MICTLANX_PROTOCOL}")
MICTLANX_DEBUG                              = bool(int(os.environ.get("MICTLANX_DEBUG",0)))
MICTLANX_LOG_DISABLED                     = bool(int(os.environ.get("MICTLANX_LOG_DISABLED",1)))
RORY_COMMON_RECORDS                         = int(os.environ.get("RORY_COMMON_RECORDS","10"))
RORY_COMMON_ATTRIBUTES                      = int(os.environ.get("RORY_COMMON_ATTRIBUTES","10"))
RORY_CKKS_MODE                              = CkksModes(os.environ.get("RORY_CKKS_MODE","ml"))
RORY_COMMON_CKKS_DECIMALS                   = int(os.environ.get("RORY_COMMON_CKKS_DECIMALS","2"))
RORY_COMMON_CKKS_SECURITY_LEVEL             = int(os.environ.get("RORY_COMMON_CKKS_SECURITY_LEVEL","128"))



@pytest.fixture
def max_workers():
    return RORY_MAX_WORKERS
@pytest.fixture
def keys_path():
    return RORY_KEYS_PATH
@pytest.fixture
def get_context():
    return {
        "ctx"       : RORY_COMMON_CTX_FILENAME,
        "pubkey"    : RORY_COMMON_PUBKEY_FILENAME,
        "secretkey" : RORY_COMMON_SECRETKEY_FILENAME,
        "relinkey"  : RORY_COMMON_RELINKEY_FILENAME,
        "rotatekey" : RORY_COMMON_ROTATEKEY_FILENAME,
        "security_level" : RORY_COMMON_SECURITY_LEVEL,
        "enable_rotate_key_generation" : RORY_ENABLE_ROTATE_KEY_GENERATION,
        "keys_path" : RORY_KEYS_PATH,
        "max_workers" : RORY_MAX_WORKERS,
        "bucket_id" : MICTLANX_BUCKET_ID,
        "records"   : RORY_COMMON_RECORDS,
        "attributes": RORY_COMMON_ATTRIBUTES
    }



@pytest.fixture(scope="module")
async def client():
    client = AsyncClient(
        uri              = MICTLANX_URI,
        client_id        = MICTLANX_CLIENT_ID,
        capacity_storage = "200mb",
        enable_logging   = not MICTLANX_LOG_DISABLED,
        eviction_policy  = "LRU",
        max_workers      = MICTLANX_MAX_WORKERS,
        verify           = False
    )
    yield client

@pytest.fixture
def ckks():
    if not os.path.exists(RORY_KEYS_PATH):
        os.makedirs(RORY_KEYS_PATH,exist_ok=True)

    _ = Ckks.create_client(
        scheme             = "CKKS",
        decimals           = RORY_COMMON_CKKS_DECIMALS,
        security_level     = RORY_COMMON_SECURITY_LEVEL,
        save               = True,
        output_path        = RORY_KEYS_PATH,
        enable_rotate      = RORY_ENABLE_ROTATE_KEY_GENERATION,
        enable_relinearize = RORY_ENABLE_REALINEARIZATION_KEY_GENERATION,
        mode               = RORY_CKKS_MODE  
    )

    ckks = Ckks.from_pyfhel(
        _round   = True,
        decimals = RORY_COMMON_CKKS_DECIMALS,
        path     = RORY_KEYS_PATH,
    ) 
    return ckks
@pytest.fixture
def dataowner_pqc(ckks):
    dataowner_pqc = DataOwnerPQC(
        scheme        = ckks,
        securitylevel = RORY_COMMON_CKKS_SECURITY_LEVEL
    )
    return dataowner_pqc

@pytest.fixture
def dataowner():
    dataowner = DataOwner(
        liu_scheme= Liu(
            _round         = True,
            decimals       = RORY_COMMON_CKKS_DECIMALS,
            secure_random  = False,
            seed           = 1,
            use_np_random  = True,
            security_level = 128
        ),
    )
    return dataowner

@pytest.fixture
def ckks_params():
    return CkksParams(
        keys_path          = RORY_KEYS_PATH,
        ctx_filename       = RORY_COMMON_CTX_FILENAME,
        pubkey_filename    = RORY_COMMON_PUBKEY_FILENAME,
        secretkey_filename = RORY_COMMON_SECRETKEY_FILENAME,
        relinkey_filename  = RORY_COMMON_RELINKEY_FILENAME,
        rotatekey_filename = RORY_COMMON_ROTATEKEY_FILENAME,
        decimals           = RORY_COMMON_CKKS_DECIMALS,
        _round             = True,

    )

@pytest.fixture
def liu_params():
    return LiuParams(
        _round         = True,
        decimals       = RORY_COMMON_CKKS_DECIMALS,
        secure_random  = False,
        seed           = 1,
        use_np_random  = True,
        security_level = 128,
    )


@pytest.fixture
def fdhope_params():
    return FdhopeParams(
        scheme         = "DBSKMEANS",
        sens           = 0.2,
        _round         = True,
        decimals       = RORY_COMMON_CKKS_DECIMALS,
        secure_random  = False,
        seed           = 1,
        use_np_random  = True,
        security_level = 128,
    )




@pytest.fixture
def key():
    return uuid4().hex.replace("-","")[:16]

@pytest.fixture
def ball_id():
    return uuid4().hex.replace("-","")[:16]
@pytest.fixture
def bucket_id():
    return f"testbucket{uuid4().hex[:4]}"

@pytest.fixture
def generated_matrix():
    return np.random.random(size=(10,10))

@pytest.fixture
def executor():
    with ProcessPoolExecutor(max_workers=RORY_MAX_WORKERS) as executor:
        yield executor
@pytest.fixture
def initialized_executor():
    executor   = ProcessPoolExecutor(
        max_workers = RORY_MAX_WORKERS,
        initializer = RoryCommon.init_ckks_worker_context,
        initargs= (RORY_KEYS_PATH, RORY_COMMON_CTX_FILENAME, RORY_COMMON_PUBKEY_FILENAME, RORY_COMMON_SECRETKEY_FILENAME)
    )
    yield executor
