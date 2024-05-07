from mictlanx.v4.client import Client as V4Client
from mictlanx.v4.interfaces.responses import GetNDArrayResponse,GetBytesResponse,Metadata,PutResponse,PutChunkedResponse
from mictlanx.utils.segmentation import Chunks,Chunk
from option import Option,NONE,Result,Ok,Err
from functools import reduce
from typing import Tuple, Generator,Dict
import operator
import pandas as pd
import numpy as np
import numpy.typing as npt
from retry import retry
import os

MAX_RETRIES = int(os.environ.get("MAX_RETRIES","10"))
MAX_DELAY   = int(os.environ.get("MAX_DELAY","1"))
JITTER      = eval(os.environ.get("JITTER","0"))

class Utils:
    @staticmethod
    @retry(tries=MAX_RETRIES,delay=MAX_DELAY,jitter=JITTER)
    def get_matrix_or_error(client:V4Client,key:str, bucket_id:str)->GetNDArrayResponse:
        x:Result[GetNDArrayResponse, Exception] = client.get_ndarray( key = key, bucket_id=bucket_id).result()
        if x.is_err:
            e = x.unwrap_err()
            # print("GET_ERROR",str(e))
            raise e
        return x.unwrap()
    
    @staticmethod
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
                    return Ok(plaintextMatrix)
            else:
                if plaintext_matrix_id.is_some and bucket_id.is_some:
                    key = plaintext_matrix_id.unwrap()
                    fut = client.get_ndarray(key=key,bucket_id=bucket_id.unwrap())
                    result:Result[GetNDArrayResponse,Exception]   = fut.result()
                    if result.is_ok:
                        response = result.unwrap()
                        return Ok(response.value )
                    else:
                        return result
                else:
                    return Err(Exception("Path, extension and plaintext_matrix_id was not provided"))
        except Exception as e:
            return Err(e)
        
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
    def while_not_delete_ball_id(STORAGE_CLIENT:V4Client ,bucket_id:str, ball_id:str): 
        n_deletes = -1   
        while not n_deletes == 0:
            _delete_result = STORAGE_CLIENT.delete_by_ball_id(bucket_id=bucket_id,ball_id=ball_id)
            if _delete_result.is_ok:
                del_response = _delete_result.unwrap()
                n_deletes = del_response.n_deletes
        return n_deletes
    
    @staticmethod
    def delete_and_put_ndarray(STORAGE_CLIENT:V4Client,bucket_id:str,ball_id:str,key:str,ndarray:npt.NDArray,tags:Dict[str,str]={})->Result[PutResponse,Exception]:
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
    def delete_and_put_chunked(STORAGE_CLIENT:V4Client,bucket_id:str,ball_id:str,key:str,chunks:Generator[bytes,None,None], tags:Dict[str,str]={})->Result[PutChunkedResponse,Exception]:
        condition = True
        put_res = None
        while condition: 
            _delete_result = Utils.while_not_delete_ball_id(STORAGE_CLIENT=STORAGE_CLIENT, bucket_id=bucket_id, ball_id=ball_id)
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