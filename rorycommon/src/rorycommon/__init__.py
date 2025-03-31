import time as T
import asyncio
from mictlanx import AsyncClient
# from mictlanx.v4.interfaces.responses import PutChunkedResponse,AsyncGetResponse
import mictlanx.interfaces as InterfaceX
from option import Option, NONE,Result,Ok,Err,Some
from functools import reduce
import numpy as np
import numpy.typing as npt
import pandas as pd
import os
from typing import Tuple, Generator,Dict
from mictlanx.utils.segmentation import Chunks,Chunk
from rory.core.security.dataowner import DataOwner
from rory.core.security.pqc.dataowner import DataOwner as DataOwnerPQC
from typing import List,Awaitable
from concurrent.futures import ProcessPoolExecutor
from Pyfhel import PyCtxt
import pickle
from rory.core.security.cryptosystem.pqc.ckks import Ckks
import hashlib as H
# from xolo.utils.utils import Utils as XoloUtils
# from option import Result,Some

# MAX_RETRIES = int(os.environ.get("MAX_RETRIES","10"))
# MAX_DELAY   = int(os.environ.get("MAX_DELAY","2"))
# JITTER      = eval(os.environ.get("JITTER","(.1,.5)"))

class Common:
    
    @staticmethod
    async def read_numpy_from(path:str="",extension:str="")->Result[npt.NDArray,Exception]:
        try:
            if extension == "csv":
                plaintextMatrix = pd.read_csv(
                    path, 
                    header=None
                ).values
                return Ok(plaintextMatrix)
            elif extension == "npy":
                with open(path, "rb") as f:
                    plaintextMatrix = np.load(f)
                    return Ok(plaintextMatrix.astype(np.float64))
            else:
                return Err(Exception("Either path or extension  was not provided"))
        except Exception as e:
            return Err(e)
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


    @staticmethod
    def encrypt_chunk_liu(key:str,dataowner:DataOwner,chunk:Chunk, np_random:bool)-> Chunk:
        ptm = chunk.to_ndarray().unwrap()
        encyrpted_chunk:npt.NDArray = dataowner.liu_encrypt_matrix_chunk(plaintext_matrix = ptm, np_random=np_random)
        return Chunk.from_ndarray(group_id=key, index= chunk.index, ndarray= encyrpted_chunk, chunk_id=Some("{}_{}".format(key,chunk.index)))

    @staticmethod
    def to_chunks_generator(awaitable_chunks:List[Awaitable[Chunk]]):
        xs = list(map(lambda fut: fut.result(), awaitable_chunks))
        return xs

    #  Segmentation
    @staticmethod
    def segment_and_encrypt_liu(key:str,dataowner:DataOwner,plaintext_matrix:npt.NDArray, n:int, np_random:bool, num_chunks:int=2,max_workers:int = int(os.cpu_count()/2)):
        plaintext_matrix_chunks = Chunks.from_ndarray(ndarray= plaintext_matrix, group_id = key, num_chunks= num_chunks).unwrap()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            awaitable_chunks:List[Awaitable[Chunk]] = []
            for plaintext_matrix_chunk in plaintext_matrix_chunks.iter():
                future = executor.submit(Common.encrypt_chunk_liu,key = key, dataowner = dataowner,chunk = plaintext_matrix_chunk, np_random = np_random)
                awaitable_chunks.append(future)
            return Chunks(chs= Common.to_chunks_generator(awaitable_chunks=awaitable_chunks),n =n  )

    @staticmethod
    def segment_and_encrypt_liu_with_executor(executor:ProcessPoolExecutor,key:str,dataowner:DataOwner,plaintext_matrix:npt.NDArray, n:int, np_random:bool, num_chunks:int=2 ):
        plaintext_matrix_chunks:Chunks = Chunks.from_ndarray(ndarray= plaintext_matrix, group_id = key, num_chunks= num_chunks).unwrap()
        awaitable_chunks:List[Awaitable[Chunk]] = []
        for plaintext_matrix_chunk in plaintext_matrix_chunks.iter():
            # print("PLAIN_CHUNK",plaintext_matrix_chunk.data)
            future = executor.submit(Common.encrypt_chunk_liu,key = key, dataowner = dataowner,chunk = plaintext_matrix_chunk, np_random = np_random)
            awaitable_chunks.append(future)
            # print("="*30)
        return Chunks(chs= Common.to_chunks_generator(awaitable_chunks=awaitable_chunks),n =n  )


    
    @staticmethod
    def segment_and_encrypt_fdhope(algorithm:str, key:str,dataowner:DataOwner,plaintext_matrix:npt.NDArray, n:int ,num_chunks:int=2, threshold:float = 0.0, max_workers:int = int(os.cpu_count()/2) ):
        plaintext_matrix_chunks = Chunks.from_ndarray(ndarray= plaintext_matrix, group_id = key, num_chunks= num_chunks).unwrap()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            awaitable_chunks:List[Awaitable[Chunk]] = []
            for plaintext_matrix_chunk in plaintext_matrix_chunks.iter():
                future = executor.submit(Common.encrypt_chunk_fdhope,
                        key       = key, 
                        dataowner = dataowner,
                        chunk     = plaintext_matrix_chunk,
                        algorithm = algorithm,
                        threshold = threshold
                        )
                awaitable_chunks.append(future)
            return Chunks(chs= Common.to_chunks_generator(awaitable_chunks=awaitable_chunks),n =n  )

    @staticmethod
    def segment_and_encrypt_fdhope_with_executor(executor:ProcessPoolExecutor,algorithm:str, key:str,dataowner:DataOwner,matrix:npt.NDArray, n:int ,num_chunks:int=2, sens:float = 0.00001 ):
        plaintext_matrix_chunks = Chunks.from_ndarray(ndarray= matrix, group_id = key, num_chunks= num_chunks).unwrap()
        awaitable_chunks:List[Awaitable[Chunk]] = []
        for plaintext_matrix_chunk in plaintext_matrix_chunks.iter():
            future = executor.submit(Common.encrypt_chunk_fdhope,
                    key       = key, 
                    dataowner = dataowner,
                    chunk     = plaintext_matrix_chunk,
                    algorithm = algorithm,
                    sens      = sens 
                    )
            awaitable_chunks.append(future)
        return Chunks(chs= Common.to_chunks_generator(awaitable_chunks=awaitable_chunks),n =n  )

    @staticmethod
    def encrypt_chunk_fdhope(key:str,dataowner:DataOwner,chunk:Chunk,algorithm:str,sens:float=0.00001)-> Chunk:
        try:
            encyrpted_chunk = dataowner.encrypt_udm_chunks(
                plaintext_matrix = chunk.to_ndarray().unwrap(),
                algorithm=algorithm,
                sens = sens
                )
            return Chunk.from_ndarray(group_id=key, index= chunk.index, ndarray= encyrpted_chunk.matrix, chunk_id=Some("{}_{}".format(key,chunk.index)))
        except Exception as e:
            print("ERROR", e)
            raise e
    
    @staticmethod
    def chunks_to_bytes_gen(chs:Chunks) -> Generator[bytes,None,None]:
        for chunk in chs.iter():
            yield chunk.data

    
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
                Common.encrypt_chunk_ckks,
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
        return Chunks(chs= Common.to_chunks_generator(awaitable_chunks=awaitable_chunks),n =n)
    
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
            data = Common.pyctxt_list_to_bytes(ciphertext=encyrpted_chunk)
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
                Common.encrypt_chunk_ckks_v2,
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
        return Chunks(chs= Common.to_chunks_generator(awaitable_chunks=awaitable_chunks),n =n)
    
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
            data = Common.pyctxt_matrix_to_bytes(ciphertext=encyrpted_chunk)
            return Chunk(
                group_id=key,
                index= chunk.index,
                data=data,
                chunk_id = Some("{}_{}".format(key,chunk.index))
            )
        except Exception as e:
            print("ENCRYPT_CHUNK_ERROR",e)


    
    @staticmethod
    def chunks_to_pyctxt_list(chunks:Chunks, ckks:Ckks)->List[List[PyCtxt]]:
        xs = []
        for ch in chunks.iter():
            x  = pickle.loads(ch.data)
            xx = Common.bytes_to_pyctxt_list(ckks=ckks,serialized_ctxt_bytes=x)
            xs.append(xx)
        return xs
    
    def verify_mean_error(old_matrix:npt.NDArray, new_matrix:npt.NDArray, min_error:float=0.15):
        mean_error = np.mean(np.abs((old_matrix - new_matrix) / old_matrix))
        return mean_error <= min_error


    @staticmethod
    def chunks_to_pyctxt_matrix(chunks:Chunks, ckks:Ckks)->List[PyCtxt]:
        xs = []
        for ch in chunks.iter():
            x = ch.data
            x  = pickle.loads(x)
            xx = Common.bytes_to_pyctxt_matrix(ckks=ckks,serialized_ctxt_bytes=x)
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
    


    @staticmethod
    async def while_not_delete_ball_id(STORAGE_CLIENT:AsyncClient ,bucket_id:str, key:str,timeout:int = 3600,max_tries:int = 5): 
        n_deletes = -1
        i = 0
        while not ( n_deletes == 0 or i >= max_tries):
            _delete_result = await STORAGE_CLIENT.delete(bucket_id=bucket_id,ball_id=key,timeout=timeout,force = True)
            if _delete_result.is_ok:
                del_response = _delete_result.unwrap()
                n_deletes = del_response.n_deletes
            i+=1
        return n_deletes
    
    @staticmethod
    async def delete_and_put_bytes(
        client:AsyncClient,
        bucket_id:str,
        key:str,
        data:bytes, 
        chunk_size:str="128kb",
        tags:Dict[str,str]={},
        timeout:int = 3600,
        max_tries:int =5
    )->Result[InterfaceX.PutChunkedResponse,Exception]:
        condition = True
        put_res = None
        i = 0
        while condition or i >= max_tries: 
            _delete_result = await Common.while_not_delete_ball_id(
                STORAGE_CLIENT=client, 
                bucket_id=bucket_id, 
                key=key
            )
            put_res = await client.put( # Saving Cent_i to storage
                bucket_id  = bucket_id,
                key        = key, 
                value      = data,
                chunk_size = chunk_size,
                tags       = tags,
                timeout    = timeout,
            )
            if put_res.is_ok:
                return put_res
            condition = put_res.is_err and not (_delete_result == 0)
            if condition:
                print(f"Put failed reytring in 1 second... Attemp {i}/{max_tries}")
                await asyncio.sleep(1)
                i+=1
        return put_res
    
    @staticmethod
    async def delete_and_put_chunks(
        client:AsyncClient,
        bucket_id:str,
        key:str,
        chunks:Chunks, 
        tags:Dict[str,str]={},
        timeout:int = 3600,
        max_tries:int =5
    )->Result[InterfaceX.PutChunkedResponse,Exception]:
        condition = True
        put_res = None
        i = 0
        while condition and i < max_tries: 
            _delete_result = await Common.while_not_delete_ball_id(
                STORAGE_CLIENT=client, 
                bucket_id=bucket_id, 
                key=key
            )
            put_res = await client.put_chunks( # Saving Cent_i to storage
                bucket_id  = bucket_id,
                key        = key, 
                chunks     = chunks,
                tags       = tags,
                timeout    = timeout,
            )
            if put_res.is_ok:
                return put_res
            condition = put_res.is_err and not (_delete_result == 0)
            if condition:
                print(f"Put failed reytring in 1 second... Attemp {i}/{max_tries}")
                await asyncio.sleep(1)
                i+=1
        return put_res

    @staticmethod
    async def put_ndarray(client:AsyncClient,key:str,matrix:npt.NDArray,num_chunks:int =2,tags:Dict[str,str]={},bucket_id:str= "rory"):

        put_chunks_generator_results = await Common.delete_and_put_bytes(
            client = client,
            bucket_id      = bucket_id,
            key            = key,
            data         = matrix.tobytes(),
            tags = {
                "shape": str(matrix.shape),
                "dtype": str(matrix.dtype),
                **tags
            }
        )
        return put_chunks_generator_results
    @staticmethod
    async def segment_encrypt_and_put_chunks(
        executor:ProcessPoolExecutor,
        dataowner:DataOwner,
        n:int,
        np_random:bool,
        client:AsyncClient,
        key:str,matrix:npt.NDArray,
        num_chunks:int=2,
        tags:Dict[str,str]={},
        bucket_id:str= "rory"
    )->Tuple[Result[InterfaceX.PutChunkedResponse,Exception],float,float]:
        t1 = T.time()
        chunks = Common.segment_and_encrypt_liu_with_executor( #Encrypt 
            executor         = executor,
            key              = key,
            plaintext_matrix = matrix,
            dataowner        = dataowner,
            n                = n,
            num_chunks       = num_chunks,
            np_random        = np_random
        )
        seg_encrypt_rt = T.time() -t1
        put_chunks_generator_results = await Common.put_chunks(
            client = client,
            bucket_id      = bucket_id,
            key            = key,
            chunks         = chunks,
            tags= tags

        )
        return put_chunks_generator_results,seg_encrypt_rt,T.time()-t1
    @staticmethod
    async def put_chunks(client:AsyncClient,key:str,chunks:Chunks,tags:Dict[str,str]={},bucket_id:str= "rory"):
        chunks.sort()
        put_chunks_generator_results = await Common.delete_and_put_chunks(
            client = client,
            bucket_id      = bucket_id,
            key            = key,
            chunks         = chunks,
            tags=tags
        )
        return put_chunks_generator_results
    
    @staticmethod
    async def get_matrix_or_error(
        client:AsyncClient,
        key:str, 
        bucket_id:str,
        max_retries:int =5,
        delay:int = 1
    )->npt.NDArray:
        i =0
        while i <= max_retries :
            x = await client.get(bucket_id=bucket_id,key=key)
            # x:Result[GetNDArrayResponse, Exception] = client.get_ndarray( key = key, bucket_id=bucket_id,headers={"Accept-Encoding":"identity"}).result()
            if x.is_err:
                e = x.unwrap_err()
                print(f"Retrying in {delay} seconds... (Attemp {i}/{max_retries})")
                i+=i
                await asyncio.sleep(delay)
                continue
                
                # print("GET_ERROR",str(e))
                
            get_response = x.unwrap()

            dtype = get_response.metadatas[0].tags.get("dtype","float32")
            raw_ndarray = np.frombuffer(buffer=get_response.data.tobytes(),dtype=dtype)
            shape = eval(get_response.metadatas[0].tags.get("shape",str(raw_ndarray.shape)))
            return raw_ndarray.reshape(shape)
        raise Exception(f"Get {bucket_id}@{key} failed: Max tries reached")
    @staticmethod
    async def get_and_merge(
        client:AsyncClient,
        key:str,
        bucket_id:str="rory",
        max_retries:int=10,
        timeout:int = 120, 
        delay: float = 1,
        backoff_factor: float = 0.5
    ):
        try:
            i= 0
            while i < max_retries:
                x_result = client.get_chunks(
                    bucket_id=bucket_id,
                    key=key,
                    timeout=timeout,
                    max_retries=max_retries,
                    delay=delay,
                    backoff_factor=backoff_factor

                )
                ms:List[InterfaceX.Metadata] = []
                xs:List[Tuple[int, npt.NDArray,bytes]] = []
                h = H.sha256()
                async for (m,data) in x_result:
                    data_bytes = data.tobytes()
                    # h.update(data_bytes)
                    ms.append(m)
                    shape = eval(m.tags.get("shape"))
                    dtype = m.tags.get("dtype","float64")
                    x = np.frombuffer(data_bytes,dtype= dtype ).reshape(shape)
                    print(x)
                    # print("X",dtype)

                    index = int(m.tags.get("index","-1"))
                    xs.append((index,x,data_bytes))
                if len(ms) >0:
                    m = ms[0]
                    num_chunks = int(m.tags.get("num_chunks","-1"))
                    if num_chunks == -1 or num_chunks != len(ms):
                        raise Exception("Faile to get the chunks")
                    else:
                        full_shape  = m.tags.get("full_shape")
                        full_dtype  = m.tags.get("full_dtype","float64")
                        xs_sorted = sorted(xs, key=lambda t: t[0])  # Sort by index value
                        # ordered_chunks = [x[1] for x in xs_sorted]  # Extract sorted arrays
                        ordered_chunks = []
                        # [x[1] for x in xs_sorted]  # Extract sorted arrays
                        for (i,nd, data) in xs_sorted:
                            h2 = H.sha256()
                            h.update(data)
                            h2.update(data)
                            ordered_chunks.append(nd)
                        checksum = h.hexdigest()
                        return np.concatenate(ordered_chunks, axis=0)
            
        except Exception as e:
            raise e
