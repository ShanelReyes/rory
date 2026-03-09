import time as T
import asyncio
from mictlanx.v4.client import Client as V4Client
from mictlanx import AsyncClient
from mictlanx.utils.index import Utils as MictlanXUtils
from mictlanx.v4.interfaces.responses import GetNDArrayResponse,GetBytesResponse,Metadata,PutChunkedResponse,PutResponse
from option import Option, NONE,Result,Ok,Err,Some
from functools import reduce
import numpy as np
import numpy.typing as npt
import pandas as pd
import os
import operator
from typing import Tuple, Generator,Dict
from retry import retry
from mictlanx.utils.segmentation import Chunks,Chunk
from rory.core.security.dataowner import DataOwner
from rory.core.security.pqc.dataowner import DataOwner as DataOwnerPQC
from typing import List,Awaitable
from concurrent.futures import ProcessPoolExecutor
from Pyfhel import PyCtxt
import pickle
from rory.core.security.cryptosystem.pqc.ckks import Ckks
# from option import Result,Some

MAX_RETRIES = int(os.environ.get("MAX_RETRIES","10"))
MAX_DELAY   = int(os.environ.get("MAX_DELAY","2"))
JITTER      = eval(os.environ.get("JITTER","(.1,.5)"))

class Utils:
    

    @staticmethod 
    def get_workers(num_chunks:int = 2):
        """
            Determine the optimal number of worker threads/processes based on CPU cores and desired chunk parallelism.

            This method calculates and returns the appropriate number of workers to efficiently process tasks in parallel,
            taking into account the available CPU cores and a user-specified maximum number of parallel chunks (`num_chunks`).

            The returned worker count ensures that system resources are efficiently utilized without oversubscription,
            maintaining optimal performance.

            Args:
                num_chunks (int, optional): The preferred maximum number of parallel chunks/tasks to execute concurrently.
                                            Defaults to 2.

            Returns:
                int: Optimal number of worker threads/processes calculated by considering:
                    - The number of CPU cores available on the host system.
                    - The specified maximum number of parallel chunks (`num_chunks`).

                    The final worker count returned will not exceed either:
                    - The number of available CPU cores.
                    - The specified `num_chunks`.

            Raises:
                ValueError: If `num_chunks` provided is less than 1.

            Examples:
                >>> Utils.get_workers(num_chunks=4)
                4  # assuming at least 4 cores available

                >>> Utils.get_workers(num_chunks=32)
                8  # assuming 8 CPU cores available, returns cores count as limit

            Notes:
                - Ensure this method is used when distributing tasks that benefit from parallel processing.
                - Excessive parallelism beyond CPU cores may degrade performance due to overhead.

        """
        cores = os.cpu_count()
        return cores if num_chunks > cores else num_chunks

    # Serializer
    @staticmethod
    def pyctxt_list_to_bytes(ciphertext:List[PyCtxt]):
        serialized_ciphertexts = [ctxt.to_bytes() for ctxt in ciphertext]
        return pickle.dumps(serialized_ciphertexts)
    
    def pyctxt_matrix_to_bytes(ciphertext:List[PyCtxt]):
        serialized_ciphertexts = []
        for ctxts in ciphertext:
            inner_sctxts = []
            for ctxt in ctxts:
                inner_sctxts.append(ctxt.to_bytes())
            serialized_ciphertexts.append(inner_sctxts)
        return pickle.dumps(serialized_ciphertexts)

    @staticmethod
    def bytes_to_pyctxt_list_v1(ckks:Ckks,serialized_ctxt_bytes:bytes)->List[PyCtxt]:
        scheme  = ckks.he_object
        xx      = list(map(lambda x: PyCtxt(None,scheme,None,x,'FRACTIONAL'), serialized_ctxt_bytes))
        return xx

    @staticmethod
    def bytes_to_pyctxt_list(ckks:Ckks,serialized_ctxt_bytes:List[bytes], logger= None)->List[PyCtxt]:
        scheme  = ckks.he_object
        xx = []
        for x in serialized_ctxt_bytes:
            y = PyCtxt(None, scheme, None,x, "FRACTIONAL")
            xx.append(y)
        return xx
    @staticmethod
    def bytes_to_pyctxt_list_v2(ckks:Ckks, data:bytes):
        xs = pickle.loads(data)
        scheme = ckks.he_object
        xx = [PyCtxt(None, scheme, None, x, "FRACTIONAL") for x in xs ]
        return xx

    #  Segmentation

    @staticmethod
    def segment_and_encrypt_ckks_with_executor(
        executor:ProcessPoolExecutor,
        key:str,
        plaintext_matrix:npt.NDArray,
        n:int,
        _round:bool, decimals:int, path:str, ctx_filename:str, 
        pubkey_filename:str, 
        secretkey_filename:str,
        num_chunks:int=2, 
    ):
        plaintext_matrix_chunks                 = Chunks.from_ndarray(ndarray= plaintext_matrix, group_id = key, num_chunks= num_chunks).unwrap()
        awaitable_chunks:List[Awaitable[Chunk]] = []
        for plaintext_matrix_chunk in plaintext_matrix_chunks.iter():
            future = executor.submit(
                Utils.encrypt_chunk_ckks,
                key       = key,
                chunk     = plaintext_matrix_chunk,
                _round    = _round,
                decimals  = decimals,
                path               = path,
                ctx_filename       = ctx_filename,
                pubkey_filename    = pubkey_filename,
                secretkey_filename = secretkey_filename,
            )
            awaitable_chunks.append(future)
        return Chunks(chs= Utils.to_chunks_generator(awaitable_chunks=awaitable_chunks),n =n)
    
    @staticmethod
    def encrypt_chunk_ckks(key:str, chunk:Chunk, _round:bool, decimals:int, path:str, ctx_filename:str, 
                           pubkey_filename:str, secretkey_filename:str)-> Chunk:
        try:
            dataowner = DataOwnerPQC(
                scheme= Ckks.from_pyfhel(
                    _round   = _round,
                    decimals = decimals,
                    path               = path,
                    ctx_filename       = ctx_filename,
                    pubkey_filename    = pubkey_filename,
                    secretkey_filename = secretkey_filename,
                ) 
            )
            plaintext_matrix = chunk.to_ndarray().unwrap().copy()
            encyrpted_chunk:List[PyCtxt] = dataowner.ckks_encrypt_matrix_chunk(plaintext_matrix = plaintext_matrix)
            data = Utils.pyctxt_list_to_bytes(ciphertext=encyrpted_chunk)
            return Chunk(
                group_id=key,
                index= chunk.index,
                data=data,
                chunk_id = Some("{}_{}".format(key,chunk.index))
            )
        except Exception as e:
            print("ENCRYPT_CHUNK_ERROR",e)
    
    @staticmethod
    def segment_and_encrypt_ckks_with_executor_v2(
        executor:ProcessPoolExecutor,
        key:str,
        plaintext_matrix:npt.NDArray,
        n:int,
        _round:bool, decimals:int, path:str, ctx_filename:str, 
        pubkey_filename:str, 
        secretkey_filename:str,
        num_chunks:int=2, 
    ):
        plaintext_matrix_chunks = Chunks.from_ndarray(ndarray= plaintext_matrix, group_id = key, num_chunks= num_chunks).unwrap()
        awaitable_chunks:List[Awaitable[Chunk]] = []
        for plaintext_matrix_chunk in plaintext_matrix_chunks.iter():
            future = executor.submit(
                Utils.encrypt_chunk_ckks_v2,
                key       = key,
                chunk     = plaintext_matrix_chunk,
                _round    = _round,
                decimals  = decimals,
                path               = path,
                ctx_filename       = ctx_filename,
                pubkey_filename    = pubkey_filename,
                secretkey_filename = secretkey_filename,

            )
            awaitable_chunks.append(future)
        return Chunks(chs= Utils.to_chunks_generator(awaitable_chunks=awaitable_chunks),n =n)
    
    @staticmethod
    def encrypt_chunk_ckks_v2(key:str, chunk:Chunk, _round:bool, decimals:int, path:str, ctx_filename:str, 
                           pubkey_filename:str, secretkey_filename:str)-> Chunk:
        try:
            dataowner = DataOwnerPQC(
                scheme= Ckks.from_pyfhel(
                    _round   = _round,
                    decimals = decimals,
                    path               = path,
                    ctx_filename       = ctx_filename,
                    pubkey_filename    = pubkey_filename,
                    secretkey_filename = secretkey_filename,
                ) 
            )
            plaintext_matrix = chunk.to_ndarray().unwrap().copy()
            encyrpted_chunk:List[PyCtxt] = dataowner.ckks_encrypt_matrix_list_chunk(plaintext_chunk = plaintext_matrix)
            data = Utils.pyctxt_matrix_to_bytes(ciphertext=encyrpted_chunk)
            return Chunk(
                group_id=key,
                index= chunk.index,
                data=data,
                chunk_id = Some("{}_{}".format(key,chunk.index))
            )
        except Exception as e:
            print("ENCRYPT_CHUNK_ERROR",e)

    @staticmethod
    def get_pyctxt_with_retry(
        STORAGE_CLIENT:V4Client,
        bucket_id:str, 
        num_chunks:int,
        key:str,
        ckks:Ckks,
        )-> List[PyCtxt]:
        x = STORAGE_CLIENT.get_with_retry(key = key, bucket_id=bucket_id,headers={"Accept-Encoding":"identity"})
        if x.is_err:
            e = x.unwrap_err()
            raise e
        serialized_ctxt_bytes = x.unwrap().value 
        chs = Chunks.from_bytes(data= serialized_ctxt_bytes, group_id="", num_chunks = num_chunks).unwrap()
        encryptedMatrix = Utils.chunks_to_pyctxt_list(ckks= ckks, chunks= chs)
        return encryptedMatrix
    
    @staticmethod
    def chunks_to_pyctxt_list(chunks:Chunks, ckks:Ckks)->List[List[PyCtxt]]:
        xs = []
        for ch in chunks.iter():
            x  = pickle.loads(ch.data)
            xx = Utils.bytes_to_pyctxt_list(ckks=ckks,serialized_ctxt_bytes=x)
            xs.append(xx)
        return xs
    
    @staticmethod
    def verify_mean_error(old_matrix:npt.NDArray, new_matrix:npt.NDArray, min_error:float=0.15)->bool:
        mean_error = np.mean(np.abs((old_matrix - new_matrix) / old_matrix))
        return mean_error <= min_error

    @staticmethod
    def get_pyctxt_matrix_with_retry(
            STORAGE_CLIENT:V4Client,
            bucket_id:str, 
            num_chunks:int,
            key:str,
            ckks:Ckks,
            chunk_size:str = "5MB"
        )-> List[PyCtxt]:
        x = STORAGE_CLIENT.get_with_retry(key = key, bucket_id=bucket_id, chunk_size=chunk_size, headers={"Accept-Encoding":"identity"})
        if x.is_err:
            e = x.unwrap_err()
            raise e
        serialized_ctxt_bytes = x.unwrap().value
        chs = Chunks.from_bytes(data= serialized_ctxt_bytes, group_id="", num_chunks = num_chunks).unwrap()
        encryptedMatrix = Utils.chunks_to_pyctxt_matrix(ckks= ckks, chunks= chs)
        return encryptedMatrix
    

    @staticmethod
    def chunks_to_pyctxt_matrix(chunks:Chunks, ckks:Ckks)->List[PyCtxt]:
        xs = []
        for ch in chunks.iter():
            x = ch.data
            x  = pickle.loads(x)
            xx = Utils.bytes_to_pyctxt_matrix(ckks=ckks,serialized_ctxt_bytes=x)
            xs.extend(xx)
        return xs
    
    @staticmethod
    def bytes_to_pyctxt_matrix(ckks:Ckks,serialized_ctxt_bytes:List[bytes], logger= None)->List[PyCtxt]:
        scheme  = ckks.he_object
        matrix = []
        for xs in serialized_ctxt_bytes:
            tmp_row = []
            for x in xs:
                element = PyCtxt(None, scheme, None,x, "FRACTIONAL")
                tmp_row.append(element)
            matrix.append(tmp_row)
        return matrix
    

