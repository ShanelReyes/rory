import os
from mictlanx.utils.index import Utils as MUtils
from mictlanx.v4.interfaces.responses import GetNDArrayResponse,GetBytesResponse,Metadata,PutResponse,PutChunkedResponse
from mictlanx.utils.segmentation import Chunks,Chunk
from option import Option,NONE,Result,Ok,Err
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor,wait
import unittest
from mictlanx.v4.client import Client
from rory.core.security.cryptosystem.pqc.ckks import Ckks
from retry import retry
from Pyfhel import PyCtxt
from typing import List

plaintext  = int(123125234)
plaintext_vector = [1,2,3,4,5]
plaintext_matrix = [
    [0.73,8.84],
    [49.93,34.44],
    [0.57,65.04],
    [62.15,32.29],
    [59.47,36.04]
]

MAX_RETRIES = int(os.environ.get("MAX_RETRIES","10"))
MAX_DELAY   = int(os.environ.get("MAX_DELAY","1"))
JITTER      = eval(os.environ.get("JITTER","0"))


mictlanx_client = Client(
    client_id       = "rory-worker-0",
    # debug           = True,
    # lb_algorithm    = "ROUND_ROBIN",
    routers         = MUtils.routers_from_str("mictlanx-router-0:localhost:60666"),
    log_output_path = "/mictlanx/client"
)
# _____________________________________ 
import os

def empty_clusters(**kwargs):
    k = kwargs.get("k",1)
    xs = np.zeros((k,1))
    xs[:] = np.nan
    return xs 

class TestApp(unittest.TestCase):
    
    @staticmethod
    def bytes_to_pyctxt_list(ckks:Ckks,serialized_ctxt_bytes:bytes)->List[PyCtxt]:
        scheme  = ckks.he_object
        xx      = list(map(lambda x: PyCtxt(None,scheme,None,x,'FRACTIONAL'), serialized_ctxt_bytes))
        return xx

    @staticmethod
    @retry(tries=MAX_RETRIES,delay=MAX_DELAY,jitter=JITTER)
    def get_pyctxt_or_error(client:Client,ckks:Ckks,key:str, bucket_id:str)->List[PyCtxt]:
        x:Result[GetBytesResponse, Exception] = client.get_with_retry(key = key, bucket_id=bucket_id)
        if x.is_err:
            e = x.unwrap_err()
            raise e
        serialized_ctxt_bytes = x.unwrap().value
        return TestApp.bytes_to_pyctxt_list(ckks=ckks ,serialized_ctxt_bytes=serialized_ctxt_bytes)

    
    def test_xx(self):
        ckks = Ckks.from_pyfhel(
            _round   = False,
            decimals = 2,
            path               = os.environ.get("KEYS_PATH","/rory/keys"),
            ctx_filename       = os.environ.get("CTX_FILENAME","ctx"),
            pubkey_filename    = os.environ.get("PUBKEY_FILENAME","pubkey"),
            relinkey_filename  = os.environ.get("RELINKEY_FILENAME","relinkey"),
            rotatekey_filename = os.environ.get("ROTATE_KEY_FILENAME","rotatekey"),
            secretkey_filename = os.environ.get("SECRET_KEY_FILENAME","secretkey"),
        )
        scheme  = ckks.he_object
        # x = PyCtxt(None, scheme, None,127 ,"FRACTIONAL")
        # print("X",x)


    @unittest.skip("")
    def test_get_pyctxt(self):
        ckks = Ckks.from_pyfhel(
            _round   = False,
            decimals = 2,
            path               = os.environ.get("KEYS_PATH","/rory/keys"),
            ctx_filename       = os.environ.get("CTX_FILENAME","ctx"),
            pubkey_filename    = os.environ.get("PUBKEY_FILENAME","pubkey"),
            relinkey_filename  = os.environ.get("RELINKEY_FILENAME","relinkey"),
            rotatekey_filename = os.environ.get("ROTATE_KEY_FILENAME","rotatekey"),
            secretkey_filename = os.environ.get("SECRET_KEY_FILENAME","secretkey"),
        )

        key = "encryptedskmeans123xxx888"
        bucket_id = "rory"
        
        result = TestApp.get_pyctxt_or_error(
            ckks = ckks,
            client= mictlanx_client,
            bucket_id = bucket_id,
            key=key
        )
        print("RESULT", result)

   

if __name__ == '__main__':
    unittest.main()