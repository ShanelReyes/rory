from mictlanx.v4.client import Client as V4Client
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
        # decrypt = ckks.decryptMatrix(xx, shape=[6,2])

    @staticmethod
    def bytes_to_pyctxt_list(ckks:Ckks,serialized_ctxt_bytes:List[bytes], logger= None)->List[PyCtxt]:
        scheme  = ckks.he_object
        xx = []
        for x in serialized_ctxt_bytes:
          
            y = PyCtxt(None, scheme, None,x, "FRACTIONAL")
            xx.append(y)
            # xx      = list(map(lambda x: PyCtxt(None,scheme,None,x,'FRACTIONAL'), serialized_ctxt_bytes))
        return xx
    @staticmethod
    def bytes_to_pyctxt_list_v2(ckks:Ckks, data:bytes):
        xs = pickle.loads(data)
        scheme = ckks.he_object
        xx = [PyCtxt(None, scheme, None, x, "FRACTIONAL") for x in xs ]
        return xx


    @staticmethod
    @retry(tries=MAX_RETRIES,delay=MAX_DELAY,jitter=JITTER)
    def read_numpy_from(client:V4Client,path:str="",extension:str="",plaintext_matrix_id:Option[str]=NONE,bucket_id:Option[str]=NONE)->Result[npt.NDArray,Exception]:
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
                    return Ok(plaintextMatrix.astype(np.float32))
            else:
                if plaintext_matrix_id.is_some and bucket_id.is_some:
                    key = plaintext_matrix_id.unwrap()
                    fut = client.get_ndarray(key=key,bucket_id=bucket_id.unwrap())
                    result:Result[GetNDArrayResponse,Exception]   = fut.result()
                    if result.is_ok:
                        response = result.unwrap()
                        return Ok(response.value.astype(np.float32))
                    else:
                        return result
                else:
                    return Err(Exception("Path, extension and plaintext_matrix_id was not provided"))

        except Exception as e:
            return Err(e)


    @staticmethod
    def encrypt_chunk_liu(key:str,dataowner:DataOwner,chunk:Chunk, np_random:bool)-> Chunk:
        encyrpted_chunk:npt.NDArray = dataowner.liu_encrypt_matrix_chunk(plaintext_matrix = chunk.to_ndarray().unwrap(), np_random=np_random)
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
                future = executor.submit(Utils.encrypt_chunk_liu,key = key, dataowner = dataowner,chunk = plaintext_matrix_chunk, np_random = np_random)
                awaitable_chunks.append(future)
            return Chunks(chs= Utils.to_chunks_generator(awaitable_chunks=awaitable_chunks),n =n  )

    @staticmethod
    def segment_and_encrypt_liu_with_executor(executor:ProcessPoolExecutor,key:str,dataowner:DataOwner,plaintext_matrix:npt.NDArray, n:int, np_random:bool, num_chunks:int=2, max_workers:int = int(os.cpu_count()/2) ):
        plaintext_matrix_chunks = Chunks.from_ndarray(ndarray= plaintext_matrix, group_id = key, num_chunks= num_chunks).unwrap()
        awaitable_chunks:List[Awaitable[Chunk]] = []
        for plaintext_matrix_chunk in plaintext_matrix_chunks.iter():
            future = executor.submit(Utils.encrypt_chunk_liu,key = key, dataowner = dataowner,chunk = plaintext_matrix_chunk, np_random = np_random)
            awaitable_chunks.append(future)
        return Chunks(chs= Utils.to_chunks_generator(awaitable_chunks=awaitable_chunks),n =n  )

    @staticmethod
    @retry(tries=MAX_RETRIES,delay=MAX_DELAY,jitter=JITTER)
    def get_matrix_or_error(client:V4Client,key:str, bucket_id:str,timeout:int = 3600)->GetNDArrayResponse:
        x:Result[GetNDArrayResponse, Exception] = client.get_ndarray( key = key, bucket_id=bucket_id,timeout=timeout).result()
        if x.is_err:
            e = x.unwrap_err()
            # print("GET_ERROR",e)
            raise e
        return x.unwrap()

    @retry(tries=MAX_RETRIES,delay=MAX_DELAY,jitter=JITTER)
    def get_and_merge_ndarray(STORAGE_CLIENT:V4Client,bucket_id:str, key:str,num_chunks:int, shape:tuple,dtype:str)->Tuple[npt.NDArray,Metadata]:
        encryptedMatrix_result:Result[GetBytesResponse,Exception] = STORAGE_CLIENT.get_and_merge_with_num_chunks(bucket_id=bucket_id,key=key,num_chunks=num_chunks).result()
        if encryptedMatrix_result.is_err:
            raise Exception("{} not found".format(key))
        
        encryptedMatrix_response = encryptedMatrix_result.unwrap()
        _encryptedMatrix         = np.frombuffer(encryptedMatrix_response.value,dtype=dtype)
        expected_shape           = reduce(operator.mul,shape)
        if not _encryptedMatrix.size == expected_shape:
            raise Exception("Matrix sizes are not equal: calculated: {} != expected: {}".format(_encryptedMatrix.size, expected_shape ))
        
        encryptedMatrix = _encryptedMatrix.reshape(shape)
        return (encryptedMatrix,encryptedMatrix_response.metadata)
    
    @staticmethod
    def segment_and_encrypt_fdhope(algorithm:str, key:str,dataowner:DataOwner,plaintext_matrix:npt.NDArray, n:int ,num_chunks:int=2, threshold:float = 0.0, max_workers:int = int(os.cpu_count()/2) ):
        plaintext_matrix_chunks = Chunks.from_ndarray(ndarray= plaintext_matrix, group_id = key, num_chunks= num_chunks).unwrap()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            awaitable_chunks:List[Awaitable[Chunk]] = []
            for plaintext_matrix_chunk in plaintext_matrix_chunks.iter():
                future = executor.submit(Utils.encrypt_chunk_fdhope,
                        key       = key, 
                        dataowner = dataowner,
                        chunk     = plaintext_matrix_chunk,
                        algorithm = algorithm,
                        threshold = threshold
                        )
                awaitable_chunks.append(future)
            return Chunks(chs= Utils.to_chunks_generator(awaitable_chunks=awaitable_chunks),n =n  )

    @staticmethod
    def segment_and_encrypt_fdhope_with_executor(executor:ProcessPoolExecutor,algorithm:str, key:str,dataowner:DataOwner,plaintext_matrix:npt.NDArray, n:int ,num_chunks:int=2, sens:float = 0.00001 ):
        plaintext_matrix_chunks = Chunks.from_ndarray(ndarray= plaintext_matrix, group_id = key, num_chunks= num_chunks).unwrap()
        awaitable_chunks:List[Awaitable[Chunk]] = []
        
        for plaintext_matrix_chunk in plaintext_matrix_chunks.iter():
            future = executor.submit(Utils.encrypt_chunk_fdhope,
                    key       = key, 
                    dataowner = dataowner,
                    chunk     = plaintext_matrix_chunk,
                    algorithm = algorithm,
                    sens      = sens 
                    )
            awaitable_chunks.append(future)
        # print("FUTURES", awaitable_chunks)
        # raise Exception("INSIDE_BOOM")
        return Chunks(chs= Utils.to_chunks_generator(awaitable_chunks=awaitable_chunks),n =n  )

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
    def while_not_delete(STORAGE_CLIENT:V4Client ,bucket_id:str, key:str): 
        n_deletes = -1

        while not n_deletes == 0:
            _delete_result = STORAGE_CLIENT.delete(bucket_id=bucket_id,key=key)
            if _delete_result.is_ok:
                del_response = _delete_result.unwrap()
                n_deletes = del_response.n_deletes
        return n_deletes
    
    @staticmethod
    def while_not_delete_ball_id(STORAGE_CLIENT:V4Client ,bucket_id:str, ball_id:str,timeout:int = 3600,max_tries:int = 5): 
        n_deletes = -1
        i = 0
        # print("AQUI")
        while not ( n_deletes == 0 or i >= max_tries):
            _delete_result = STORAGE_CLIENT.delete_by_ball_id(bucket_id=bucket_id,ball_id=ball_id,timeout=timeout)
            # print("DELETE_RESULT",_delete_result)
            if _delete_result.is_ok:
                del_response = _delete_result.unwrap()
                n_deletes = del_response.n_deletes
            i+=1
        return n_deletes
    
    @staticmethod
    def delete_and_put_ndarray_by_ball_id(STORAGE_CLIENT:V4Client,bucket_id:str,ball_id:str,ndarray:npt.NDArray,tags:Dict[str,str]={})->Result[PutResponse,Exception]:
        condition = True
        put_res = None
        while condition: 
            _delete_result = Utils.while_not_delete_ball_id(STORAGE_CLIENT=STORAGE_CLIENT, bucket_id=bucket_id, ball_id=ball_id)
            put_res:Result[PutResponse,Exception] = STORAGE_CLIENT.put_ndarray( # Saving Cent_i to storage
            key       = ball_id, 
            ndarray   = ndarray,
            tags      = tags,
            bucket_id = bucket_id
            )
            if put_res.is_ok:
                return put_res
            
            condition = put_res.is_err and not (_delete_result == 0)
        return put_res

    @staticmethod
    def delete_and_put_chunked_by_ball_id(STORAGE_CLIENT:V4Client,bucket_id:str,ball_id:str,chunks:Generator[bytes,None,None], tags:Dict[str,str]={})->Result[PutChunkedResponse,Exception]:
        condition = True
        put_res = None
        while condition: 
            _delete_result = Utils.while_not_delete_ball_id(STORAGE_CLIENT=STORAGE_CLIENT, bucket_id=bucket_id, ball_id=ball_id)
            put_res = STORAGE_CLIENT.put_chunked( # Saving Cent_i to storage
            key       = ball_id, 
            chunks= chunks,
            tags      = tags,
            bucket_id = bucket_id
            )
            if put_res.is_ok:
                return put_res
            condition = put_res.is_err and not (_delete_result == 0)
        return put_res
    

    @staticmethod
    def delete_and_put_ndarray_by_key(STORAGE_CLIENT:V4Client,bucket_id:str,key:str,ndarray:npt.NDArray,tags:Dict[str,str]={})->Result[PutResponse,Exception]:
        condition = True
        put_res = None
        while condition: 
            _delete_result = Utils.while_not_delete(STORAGE_CLIENT=STORAGE_CLIENT, bucket_id=bucket_id, key=key)
            
            put_res:Result[PutResponse,Exception] = STORAGE_CLIENT.put_ndarray( # Saving Cent_i to storage
            key       = key, 
            ndarray   = ndarray,
            tags      = tags,
            bucket_id = bucket_id
            )

            if put_res.is_ok:
                return put_res
            
            condition = put_res.is_err and not (_delete_result == 0)
        return put_res

    @staticmethod
    def delete_and_put_chunked_by_key(STORAGE_CLIENT:V4Client,bucket_id:str,key:str,chunks:Generator[bytes,None,None], tags:Dict[str,str]={})->Result[PutChunkedResponse,Exception]:
        condition = True
        put_res = None
        while condition: 
            _delete_result = Utils.while_not_delete(STORAGE_CLIENT=STORAGE_CLIENT, bucket_id=bucket_id, key=key)
            put_res = STORAGE_CLIENT.put_chunked( # Saving Cent_i to storage
            key       = key, 
            chunks= chunks,
            tags      = tags,
            bucket_id = bucket_id
            )
            if put_res.is_ok:
                return put_res
            condition = put_res.is_err and not (_delete_result == 0)
        return put_res
    
    @staticmethod
    def delete_and_put_ndarray(STORAGE_CLIENT:V4Client,bucket_id:str,ball_id:str,key:str,ndarray:npt.NDArray, tags:Dict[str,str]={})->Result[PutChunkedResponse,Exception]:
        condition = True
        put_res = None
        while condition: 
            _delete_result = Utils.while_not_delete_ball_id(STORAGE_CLIENT=STORAGE_CLIENT, bucket_id=bucket_id, ball_id=ball_id)
            put_res:Result[PutResponse,Exception] = STORAGE_CLIENT.put_ndarray( # Saving Cent_i to storage
            key       = key, 
            ndarray   = ndarray,
            tags      = tags,
            bucket_id = bucket_id
            )
            # print("_DEL_RESULT", _delete_result)
            # print("PUT_RES", put_res)
            if put_res.is_ok:
                return put_res
            condition = put_res.is_err and not (_delete_result == 0)
        return put_res
    @staticmethod
    def delete_and_put_chunked(
        STORAGE_CLIENT:V4Client,
        bucket_id:str,
        ball_id:str,
        key:str,chunks:Generator[bytes,None,None], tags:Dict[str,str]={},
        timeout:int = 3600
    )->Result[PutChunkedResponse,Exception]:
        condition = True
        put_res = None
        while condition: 
            _delete_result = Utils.while_not_delete_ball_id(
                STORAGE_CLIENT=STORAGE_CLIENT, 
                bucket_id=bucket_id, 
                ball_id=ball_id
            )
            # print("DEL_RES", _delete_result)
            put_res = STORAGE_CLIENT.put_chunked( # Saving Cent_i to storage
                key       = key, 
                chunks    = chunks,
                tags      = tags,
                bucket_id = bucket_id,
                timeout=timeout
            )
            # print("PUT_RES", put_res)
            if put_res.is_ok:
                return put_res
            condition = put_res.is_err and not (_delete_result == 0)
        return put_res
    
    @staticmethod
    def segment_and_encrypt_ckks_with_executor(
        executor:ProcessPoolExecutor,
        key:str,
        plaintext_matrix:npt.NDArray,
        n:int,
        _round:bool, decimals:int, path:str, ctx_filename:str, 
        pubkey_filename:str, 
        # relinkey_filename:str, rotatekey_filename:str,
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
                # relinkey_filename  = relinkey_filename,
                # rotatekey_filename = rotatekey_filename,
                secretkey_filename = secretkey_filename,

            )
            awaitable_chunks.append(future)
        return Chunks(chs= Utils.to_chunks_generator(awaitable_chunks=awaitable_chunks),n =n)
    
    @staticmethod
    def encrypt_chunk_ckks(key:str, chunk:Chunk, _round:bool, decimals:int, path:str, ctx_filename:str, 
                           pubkey_filename:str, secretkey_filename:str)-> Chunk:
        try:
            # print("ENCRYPTED_CHUNK_CKSS")
            dataowner = DataOwnerPQC(
                scheme= Ckks.from_pyfhel(
                    _round   = _round,
                    decimals = decimals,
                    path               = path,
                    ctx_filename       = ctx_filename,
                    pubkey_filename    = pubkey_filename,
                    # relinkey_filename  = relinkey_filename,
                    # rotatekey_filename = rotatekey_filename,
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
        # return Chunk(grp)
    
    @staticmethod
    def segment_and_encrypt_ckks_with_executor_v2(
        executor:ProcessPoolExecutor,
        key:str,
        plaintext_matrix:npt.NDArray,
        n:int,
        _round:bool, decimals:int, path:str, ctx_filename:str, 
        pubkey_filename:str, 
        # relinkey_filename:str, rotatekey_filename:str,
        secretkey_filename:str,
        num_chunks:int=2, 
    ):
        
        plaintext_matrix_chunks                 = Chunks.from_ndarray(ndarray= plaintext_matrix, group_id = key, num_chunks= num_chunks).unwrap()

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
                # relinkey_filename  = relinkey_filename,
                # rotatekey_filename = rotatekey_filename,
                secretkey_filename = secretkey_filename,

            )
            awaitable_chunks.append(future)
        return Chunks(chs= Utils.to_chunks_generator(awaitable_chunks=awaitable_chunks),n =n)
    
    @staticmethod
    def encrypt_chunk_ckks_v2(key:str, chunk:Chunk, _round:bool, decimals:int, path:str, ctx_filename:str, 
                           pubkey_filename:str, secretkey_filename:str)-> Chunk:
        try:
            # print("ENCRYPTED_CHUNK_CKSS")
            dataowner = DataOwnerPQC(
                scheme= Ckks.from_pyfhel(
                    _round   = _round,
                    decimals = decimals,
                    path               = path,
                    ctx_filename       = ctx_filename,
                    pubkey_filename    = pubkey_filename,
                    # relinkey_filename  = relinkey_filename,
                    # rotatekey_filename = rotatekey_filename,
                    secretkey_filename = secretkey_filename,
                ) 
            )
            plaintext_matrix = chunk.to_ndarray().unwrap().copy()
            # print("Plaintext",plaintext_matrix)
            encyrpted_chunk:List[PyCtxt] = dataowner.ckks_encrypt_matrix_list_chunk(plaintext_chunk = plaintext_matrix)
            # print("ENCRYPTED_CHUNK", encyrpted_chunk)

            data = Utils.pyctxt_matrix_to_bytes(ciphertext=encyrpted_chunk)
            return Chunk(
                group_id=key,
                index= chunk.index,
                data=data,
                chunk_id = Some("{}_{}".format(key,chunk.index))
            )
        except Exception as e:
            print("ENCRYPT_CHUNK_ERROR",e)
        # return Chunk(grp)

    @staticmethod
    def get_pyctxt_with_retry(
        STORAGE_CLIENT,
        bucket_id:str, 
        num_chunks:int,
        key:str,
        ckks:Ckks,
        )-> List[PyCtxt]:
        x = STORAGE_CLIENT.get_with_retry(key = key, bucket_id=bucket_id)
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
            # print("CHUNK",ch)
            x  = pickle.loads(ch.data)
            xx = Utils.bytes_to_pyctxt_list(ckks=ckks,serialized_ctxt_bytes=x)
            xs.append(xx)
        return xs
    
    def verify_mean_error(old_matrix:npt.NDArray, new_matrix:npt.NDArray, min_error:float=0.15):
        mean_error = np.mean(np.abs((old_matrix - new_matrix) / old_matrix))
        return mean_error <= min_error

    @staticmethod
    def get_pyctxt_matrix_with_retry(
            STORAGE_CLIENT:V4Client,
            bucket_id:str, 
            num_chunks:int,
            key:str,
            ckks:Ckks,
            # pickle_chunks:bool = True,
            chunk_size:str = "5MB"
    )-> List[PyCtxt]:
        
        x = STORAGE_CLIENT.get_with_retry(key = key, bucket_id=bucket_id, chunk_size=chunk_size)
        if x.is_err:
            e = x.unwrap_err()
            raise e
        # print("X",x)
        serialized_ctxt_bytes = x.unwrap().value 
        # print("SERIALIZED_CTX_BYTEs", serialized_ctxt_bytes)
        chs = Chunks.from_bytes(data= serialized_ctxt_bytes, group_id="", num_chunks = num_chunks).unwrap()
        # print("CHS",chs)
        encryptedMatrix = Utils.chunks_to_pyctxt_matrix(ckks= ckks, chunks= chs)
        return encryptedMatrix
    


    @staticmethod
    def chunks_to_pyctxt_matrix(chunks:Chunks, ckks:Ckks)->List[PyCtxt]:
        xs = []
        for ch in chunks.iter():
            # print("CHUNK",ch)
            x = ch.data
          
            # if pickle_chunks:
            x  = pickle.loads(x)

            xx = Utils.bytes_to_pyctxt_matrix(ckks=ckks,serialized_ctxt_bytes=x)

            xs.extend(xx)
        return xs
    
    @staticmethod
    def bytes_to_pyctxt_matrix(ckks:Ckks,serialized_ctxt_bytes:List[bytes], logger= None)->List[PyCtxt]:
        scheme  = ckks.he_object
        # xx = []
        matrix = []
        for xs in serialized_ctxt_bytes:
            tmp_row = []
            # print("XS",xs)
            for x in xs:
                element = PyCtxt(None, scheme, None,x, "FRACTIONAL")
                tmp_row.append(element)
            matrix.append(tmp_row)
        return matrix
    

if __name__ =="__main__":
    MICTLANX_TIMEOUT      = 120
    MICTLANX_CLIENT_ID    = "rory-test"
    MICTLANX_EXPIRES_IN   = "15d"
    MICTLANX_PEERS        = "mictlanx-peer-0:localhost:7000 mictlanx-peer-1:localhost:7001"
    MICTLANX_DEBUG        = True
    MICTLANX_DAEMON       = True
    MICTLANX_SHOW_METRICS = False
    MICTLANX_DISABLED_LOG = False
    MICTLANX_MAX_WORKERS = 4
    MICTLANX_CLIENT_LB_ALGORITHM = "2CHOICES_UF"
    STORAGE_CLIENT = V4Client(
        client_id    = "rory-test",
        peers        = list(MictlanXUtils.peers_from_str(MICTLANX_PEERS)),
        daemon       = MICTLANX_DEBUG,
        debug        = MICTLANX_DAEMON,
        show_metrics = MICTLANX_SHOW_METRICS,
        max_workers  = MICTLANX_MAX_WORKERS,
        lb_algorithm = MICTLANX_CLIENT_LB_ALGORITHM,
        bucket_id    = "rory",
        disable_log  = MICTLANX_DISABLED_LOG,
        output_path  = "/rory/mictlanx"
    )
    res= Utils.read_numpy_from(
        client=STORAGE_CLIENT,
        path="/source/datasets/CLUSTERING_C1_50_R10_A10_K3_24.csv",
        extension="csv",
    )
    print(res)