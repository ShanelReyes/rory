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
from typing import List,Awaitable
from concurrent.futures import ProcessPoolExecutor
# from option import Result,Some

MAX_RETRIES = int(os.environ.get("MAX_RETRIES","10"))
MAX_DELAY   = int(os.environ.get("MAX_DELAY","2"))
JITTER      = eval(os.environ.get("JITTER","(.1,.5)"))

class Utils:
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
        return Chunks(chs= Utils.to_chunks_generator(awaitable_chunks=awaitable_chunks),n =n  )

    @staticmethod
    def encrypt_chunk_fdhope(key:str,dataowner:DataOwner,chunk:Chunk,algorithm:str,sens:float=0.00001)-> Chunk:
        #print("plaintext_matrix encrypt_chunk", type(chunk.to_ndarray().unwrap()))
        encyrpted_chunk = dataowner.encrypt_udm_chunks(
            plaintext_matrix = chunk.to_ndarray().unwrap(),
            algorithm=algorithm,
            sens = sens
            )
        return Chunk.from_ndarray(group_id=key, index= chunk.index, ndarray= encyrpted_chunk.matrix, chunk_id=Some("{}_{}".format(key,chunk.index)))
    
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
    def while_not_delete_ball_id(STORAGE_CLIENT:V4Client ,bucket_id:str, ball_id:str,timeout:int = 3600): 
        n_deletes = -1
        
        while not n_deletes == 0:
            _delete_result = STORAGE_CLIENT.delete_by_ball_id(bucket_id=bucket_id,ball_id=ball_id,timeout=timeout)
            if _delete_result.is_ok:
                del_response = _delete_result.unwrap()
                # print("DEL_RESPONSE_BY_BID", del_response)
                n_deletes = del_response.n_deletes
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
            ).result()
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
            ).result()
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
            ).result()

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
            put_res = STORAGE_CLIENT.put_chunked( # Saving Cent_i to storage
                key       = key, 
                chunks    = chunks,
                tags      = tags,
                bucket_id = bucket_id,
                timeout=timeout
            )
            if put_res.is_ok:
                return put_res
            condition = put_res.is_err and not (_delete_result == 0)
        return put_res

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