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
from typing import List,Awaitable
from rory.core.security.cryptosystem.pqc.ckks import Ckks
import os
from Pyfhel import PyCtxt
import pickle

MAX_RETRIES = int(os.environ.get("MAX_RETRIES","10"))
MAX_DELAY   = int(os.environ.get("MAX_DELAY","1"))
JITTER      = eval(os.environ.get("JITTER","0"))

class Utils:

    # Serializer
    @staticmethod
    def pyctxt_list_to_bytes(ciphertext:List[PyCtxt]) -> bytes:
        serialized_ciphertexts = [ctxt.to_bytes() for ctxt in ciphertext]
        return pickle.dumps(serialized_ciphertexts)
    
    @staticmethod
    def pyctxt_list_to_gen_bytes(ciphertext:List[PyCtxt]) -> Generator[bytes, None, None]:
        try: 
            # print("PYCTXT_LIST_GNE", ciphertext)
            serialized_ciphertexts = [ctxt.to_bytes() for ctxt in ciphertext]
            # print("SERIALIZED", len(serialized_ciphertexts))
            xs= pickle.dumps(serialized_ciphertexts)
            # print("XS")
            # print("*"*20)
            for x in xs:
                yield x
        except Exception as e:
            print("EXCEPTION",e)

    @staticmethod
    def bytes_to_pyctxt_list_v2(ckks:Ckks, data:bytes):
        xs = pickle.loads(data)
        scheme = ckks.he_object
        xx = [PyCtxt(None, scheme, None, x, "FRACTIONAL") for x in xs ]
        return xx

    
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
        # print("X",x)
        serialized_ctxt_bytes = x.unwrap().value 
        # print("SERIALIZED_CTX_BYTEs", serialized_ctxt_bytes)
        chs = Chunks.from_bytes(data= serialized_ctxt_bytes, group_id="", num_chunks = num_chunks).unwrap()
        encryptedMatrix = Utils.chunks_to_pyctxt_list(ckks= ckks, chunks= chs)
        return encryptedMatrix
    
    @staticmethod
    def chunks_to_pyctxt_list(chunks:Chunks, ckks:Ckks)->List[PyCtxt]:
        xs = []
        for ch in chunks.iter():
            # print("CHUNK",ch)
            x  = pickle.loads(ch.data)
            xx = Utils.bytes_to_pyctxt_list(ckks=ckks,serialized_ctxt_bytes=x)
            xs.extend(xx)
        return xs
    
    @staticmethod
    def chunks_to_pyctxt_list_v1(
        ckks:Ckks,
        serialized_ctxt_bytes:bytes, 
    )->List[PyCtxt]:
        
        x  = pickle.loads(serialized_ctxt_bytes)
        xx = Utils.bytes_to_pyctxt_list(ckks=ckks, serialized_ctxt_bytes=x)
        return xx
    
    @staticmethod
    def get_pyctxt_matrix_with_retry(
            STORAGE_CLIENT:V4Client,
            bucket_id:str, 
            num_chunks:int,
            key:str,
            ckks:Ckks,
            chunk_size:str = "5MB"
            )-> List[PyCtxt]:
        x = STORAGE_CLIENT.get_with_retry(key = key, bucket_id=bucket_id,chunk_size=chunk_size,headers={"Accept-Encoding":"identity"})
        if x.is_err:
            e = x.unwrap_err()
            raise e
        # print("X",x)
        serialized_ctxt_bytes = x.unwrap().value 
        # print("SERIALIZED_CTX_BYTEs", serialized_ctxt_bytes)
        chs = Chunks.from_bytes(data= serialized_ctxt_bytes, group_id="", num_chunks = num_chunks).unwrap()
        encryptedMatrix = Utils.chunks_to_pyctxt_matrix(ckks= ckks, chunks= chs)
        return encryptedMatrix
    
    @staticmethod
    def chunks_to_pyctxt_matrix(chunks:Chunks, ckks:Ckks)->List[PyCtxt]:
        xs = []
        for ch in chunks.iter():
            # print("CHUNK",ch)
            x  = pickle.loads(ch.data)
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
            for x in xs:
                element = PyCtxt(None, scheme, None,x, "FRACTIONAL")
                tmp_row.append(element)
            matrix.append(tmp_row)
        return matrix
    
    @staticmethod
    @retry(tries=MAX_RETRIES,delay=MAX_DELAY,jitter=JITTER)
    def get_pyctxt_or_error(
        client:V4Client,
        ckks:Ckks,
        key:str,
        bucket_id:str
    )->List[PyCtxt]:
        
        x:Result[GetBytesResponse, Exception] = client.get_with_retry(key = key, bucket_id=bucket_id,headers={"Accept-Encoding":"identity"})
        if x.is_err:
            e = x.unwrap_err()
            raise e
        serialized_ctxt_bytes = x.unwrap().value
        return Utils.bytes_to_pyctxt_list(ckks=ckks ,serialized_ctxt_bytes=serialized_ctxt_bytes)


    @staticmethod
    @retry(tries=MAX_RETRIES,delay=MAX_DELAY,jitter=JITTER)
    def get_matrix_or_error(client:V4Client,key:str, bucket_id:str)->GetNDArrayResponse:
        x:Result[GetNDArrayResponse, Exception] = client.get_ndarray( key = key, bucket_id=bucket_id,headers={"Accept-Encoding":"identity"}).result()
        if x.is_err:
            e = x.unwrap_err()
            # print("GET_ERROR",str(e))
            raise e
        return x.unwrap()
    
    @staticmethod
    @retry(tries=MAX_RETRIES,delay=MAX_DELAY,jitter=JITTER)
    def get_and_merge_ndarray(STORAGE_CLIENT:V4Client,bucket_id:str, key:str,num_chunks:int, shape:tuple,dtype:str)->Tuple[npt.NDArray,Metadata]:
        encryptedMatrix_result:Result[GetBytesResponse,Exception] = STORAGE_CLIENT.get_and_merge_with_num_chunks(
            bucket_id=bucket_id,key=key,num_chunks=num_chunks,headers={"Accept-Encoding":"identity"}
            ).result()
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
                    fut = client.get_ndarray(key=key,bucket_id=bucket_id.unwrap(), headers={"Accept-Encoding":"identity"})
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
    def while_not_delete_ball_id(STORAGE_CLIENT:V4Client ,bucket_id:str, ball_id:str,max_tries:int=5): 
        n_deletes = -1   
        i = 0
        while not ( n_deletes == 0 or i >= max_tries):
            _delete_result = STORAGE_CLIENT.delete_by_ball_id(bucket_id=bucket_id,ball_id=ball_id)
            # print("_DETELTE", _delete_result)
            if _delete_result.is_ok:
                del_response = _delete_result.unwrap()
                n_deletes = del_response.n_deletes
            i+=1
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
                bucket_id = bucket_id,
                headers={"Accept-Encoding":"identity"}
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
                bucket_id = bucket_id,
                # headers={"Accept-Encoding":"identity"}
            )
            # print("PUT_CHUNK_RES", put_res)
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
            bucket_id = bucket_id,
            headers={"Accept-Encoding":"identity"}
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
            bucket_id = bucket_id,
            headers={"Accept-Encoding":"identity"}
            )
            if put_res.is_ok:
                return put_res
            condition = put_res.is_err and not (_delete_result == 0)
        return put_res
    
    def pyctxt_matrix_to_bytes(ciphertext:List[PyCtxt]):
        serialized_ciphertexts = []
        for ctxts in ciphertext:
            inner_sctxts = []
            for ctxt in ctxts:
                inner_sctxts.append(ctxt.to_bytes())
            serialized_ciphertexts.append(inner_sctxts)
        return pickle.dumps(serialized_ciphertexts)
    