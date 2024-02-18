# import magic as M
import time as T
import requests as R 
import hashlib as H
import logging as L
import concurrent.futures
from concurrent.futures import Future
import numpy as np
from option import Result,Ok,Err,Some,NONE,Option
from lfu_cache import LFUCache
# from option import Option,NONE,Some
from nanoid import generate as nanoid_
from typing import Dict,List
from retry.api import retry_call
import numpy.typing as npt
from concurrent.futures import ThreadPoolExecutor,as_completed,ALL_COMPLETED,wait
import multiprocessing
# Mictlanx
from mictlanx.v3.services.xolo import Xolo
from mictlanx.v3.services.replica_manger import ReplicaManager
from mictlanx.v3.services.proxy import Proxy
from mictlanx.v3.interfaces.storage_node import  PutResponse as SNPutResponse
from mictlanx.v3.interfaces.payloads import AuthTokenPayload,SignUpPayload,PutMetadataPayload,LogoutPayload
from mictlanx.v3.interfaces.responses import GetBytesResponse,GetNDArrayResponse,PutMetadataResponse,PutDataResponse,GetMetadataResponse,GetChunkedResponse
from mictlanx.v3.interfaces.core import Metadata
from mictlanx.v3.interfaces.xolo import XoloCredentials
from mictlanx.v3.interfaces.errors import ApiError ,CannotConvertChunksToNDArray
from mictlanx.utils.segmentation import Chunks,Chunk
from mictlanx.logger.log import Log


class Client(object):
    def __init__(self,app_id:str = None,client_id:Option[str]=NONE,metadata:Option[Dict[str,str]]=NONE,replica_manager:ReplicaManager=None, xolo:Xolo = None, proxies:List[Proxy] = [],secret:str = None,expires_in:Option[str] = NONE,cache_limit:int=10,log_name:str ="mictlanx-client"):
        self.client_id                 = client_id.unwrap_or(nanoid_())
        self.rm_service:ReplicaManager = replica_manager
        self.xolo:Xolo                 = xolo
        self.max_workers               = multiprocessing.cpu_count()
        self.proxies:List[Proxy]       = proxies
        self.logger         = Log(
            name = log_name,
            console_handler_filter = lambda record: record.levelno == L.INFO or record.levelno == L.ERROR,
        )
        # _____________________________________________________________
        self.operation_counter = 0
        
        payload = AuthTokenPayload(app_id=app_id,client_id=self.client_id,secret = secret,expires_in  =expires_in )

        signup_payload = SignUpPayload(app_id=app_id,client_id=self.client_id,secret=secret,metadata=metadata.unwrap_or({}),expires_in=expires_in)
        
        auth_result = self.xolo.authenticate_or_signup(payload,Some(signup_payload))
        if(auth_result.is_ok):
            auth_response = auth_result.unwrap()
            self.token = auth_response.token
        else:
            error = auth_result.unwrap_err()
            raise error
        
        self.credentials = XoloCredentials(application_id= app_id, client_id= self.client_id, secret = secret,authorization=self.token)
        self.cache              = LFUCache(cache_limit)

    def __get_current_proxy(self)->Proxy:
        selected_index = self.operation_counter % len(self.proxies)
        selected_proxy = self.proxies[selected_index]
        self.operation_counter+=1
        return selected_proxy
        

    def put_ndarray(self, key:str, ndarray:npt.NDArray, tags:Dict[str,str] = {},group_id:str = nanoid_(),cache:bool = False, update:bool = False)->Result[SNPutResponse,ApiError]:
        value         = ndarray.tobytes() 
        tags["shape"] = str(ndarray.shape)
        tags["dtype"] = str(ndarray.dtype)
        response      = self.put(key=key,value=value, tags=tags,group_id=group_id,update=update,cache=cache)
        return response

    
    
    def put_ndarray_chunks(self,group_id:str,ndarray:npt.NDArray,num_chunks:int=1, tags:Dict[str,str] = {},cache:bool = False, update:bool = False)->Result[List[SNPutResponse],ApiError]:
        maybe_chunks = Chunks.from_ndarray(ndarray=ndarray,group_id=group_id,chunk_prefix=Some(group_id),num_chunks=num_chunks,chunk_size=NONE)

        if maybe_chunks.is_some:
            chunks      = maybe_chunks.unwrap()
            chunks_len  = chunks.len()
            max_workers = chunks_len if chunks_len <= self.max_workers else self.max_workers
            headers     = self.credentials.to_headers()
            def __put(*args,**kwargs):
                res = self.put(
                    key                = kwargs.get("key"),
                    value              = kwargs.get("value"),
                    tags               = kwargs.get("tags",{}),
                    group_id           = kwargs.get("group_id"),
                    storage_node_id    = Some(kwargs.get("storage_node_id")).filter(lambda x: not x == None),
                    replica_manager_id = Some(kwargs.get("replica_manager_id")).filter(lambda x: not x == None),
                    cache              = kwargs.get("cache",False),
                    update             = kwargs.get("update",True), 
                    content_type       = Some("application/octet-stream")
                )
                return res
            
            results:List[SNPutResponse] = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for chunk in chunks.chunks:
                    chunk.metadata = {**chunk.metadata,**tags }
                    chunk.metadata["producer_id"]  = self.client_id
                    chunk.metadata["content_type"] = "application/octet-stream"
                    # M.from_buffer(chunk.data,mime=True)
                    chunk.metadata["created_at"]   = str(int(T.time_ns()))
                    chunk.metadata["group_id"]     = group_id
                    fut                            = executor.submit(__put,
                        key = chunk.chunk_id,
                        value = chunk.data,
                        tags = chunk.metadata,
                        group_id = chunk.group_id,
                        update = update,
                        cache = cache,
                        headers = self.credentials.to_headers()
                    )
                    x = fut.result()
                    if x.is_ok:
                        results.append(x.unwrap())
                    else:
                        return x
            return Ok(results)




    def put_with_checksum_as_key(self,
        value:bytes = bytearray(),
        tags:Dict[str,str]={},
        storage_node_id:Option[str]=NONE,
        replica_manager_id:Option[str] = NONE,
        cache:bool= False,
        update:bool = False,
    )->Result[SNPutResponse,ApiError]:
        return self.put(key="",value=value,tags=tags,storage_node_id=storage_node_id, replica_manager_id=replica_manager_id,cache=cache,update=update)

    def put(self,
            key:str,
            value:bytes = bytearray(),
            tags:Dict[str,str]={},
            group_id:str = nanoid_(),
            storage_node_id:Option[str]=NONE,
            replica_manager_id:Option[str] = NONE,
            content_type:Option[str] =NONE,
            cache:bool= False,
            update:bool = False,
            headers:Dict[str,str] = {},
    )->Result[SNPutResponse,ApiError]:
        start_time = T.time()
        _headers = {**self.credentials.to_headers(),**headers}
        proxy   = self.__get_current_proxy()
        
        _content_type        = "application/octet-stream"
        # content_type.unwrap() if content_type.is_some else M.from_buffer(value,mime=True)
        tags["content_type"] = _content_type
        tags["created_at"]   = str(int(T.time_ns()))

        try:
            start_time            = T.time()
            ball_size             = len(value)
            hasher                =  H.sha256()
            hasher.update(value)
            checksum              =  hasher.hexdigest()
            proxy                 = self.__get_current_proxy()
            put_metadata_payload  = PutMetadataPayload(key=key,size=ball_size,checksum=checksum,producer_id=Some(self.client_id),group_id=group_id,node_id=storage_node_id, replica_manager_id=replica_manager_id, tags=tags)
            _key                  = put_metadata_payload.ball_id
            # _______________________________________
            def __inner():
                if(update):
                    self.delete(key=_key,headers=headers,proxy=Some(proxy)) 
                put_metadata_response = proxy.put_metadata(payload=put_metadata_payload, headers= _headers)
                if put_metadata_response.is_err:
                    error = put_metadata_response.unwrap_err()
                    wt = T.time() - start_time
                    self.logger.info("WAIT {} {} {}".format(key,ball_size,wt))
                    self.logger.error("{}".format(error))
                    raise error
                

                return put_metadata_response
            # ____________________________________
            put_metadata_response =  retry_call(__inner,fargs=[],tries=10,delay=1, jitter=1, max_delay=2 )
            if put_metadata_response.is_err:
                return put_metadata_response
            
            _put_metadata_response = put_metadata_response.unwrap()
            operation_id = _put_metadata_response.operation_id

            

            put_data_response:Result[PutDataResponse,R.RequestException]     = proxy.put_data(operation_id=operation_id,data=value,headers=_headers) 
            service_time = T.time() - start_time
            response = put_data_response.map(lambda x: SNPutResponse(key= _key,size=ball_size,service_time=service_time))
            if cache and response.is_ok:
                metadata = Metadata(id= _key, size=len(value),checksum=checksum,group_id=group_id,tags=tags)
                self.cache.put(_key,(metadata,value))
            self.logger.info("PUT {} {} {}".format(key,ball_size,service_time))
            return response
        except Exception as e:
            return Err(e)
    
    
    def put_chunks_bytes(
            self, 
            key:str,
            chunks:List[bytes],
            tags:Dict[str,str]={},
            group_id:str = nanoid_(),
            storage_node_id:Option[str]=NONE,
            replica_manager_id:Option[str] = NONE,
            content_type:Option[str] =NONE,
            cache:bool= False,
            update:bool = False,
            headers:Dict[str,str] = {},
    ):

        start_time  = T.time()
        # _headers    = {**self.credentials.to_headers(),**headers}
        # proxy       = self.__get_current_proxy()
        chunks_len  = len(chunks)
        max_workers = chunks_len if chunks_len <= self.max_workers else self.max_workers

        def __put(*args,**kwargs):
            res = self.put(
                key                = kwargs.get("key"),
                value              = kwargs.get("value"),
                tags               = kwargs.get("tags",{}),
                group_id           = kwargs.get("group_id"),
                storage_node_id    = Some(kwargs.get("storage_node_id")).filter(lambda x: not x == None),
                replica_manager_id = Some(kwargs.get("replica_manager_id")).filter(lambda x: not x == None),
                cache              = kwargs.get("cache",False),
                update             = kwargs.get("update",True), 
                content_type       = Some("application/octet-stream")
            )
            return res
        
        results:List[SNPutResponse] = []
        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            total_size = 0
            for i,chunk in enumerate(chunks):
                # chunk.metadata = {**chunk.metadata,**tags }
                # chunk.metadata["producer_id"]  = self.client_id
                # chunk.metadata["content_type"] = M.from_buffer(chunk.data,mime=True)
                # chunk.metadata["created_at"]   = str(int(T.time_ns()))
                # chunk.metadata["group_id"]     = group_id
                total_size += len(chunk)

                metadata = {"index":str(i),**tags}
                fut                            = executor.submit(__put,
                    key = "{}_{}".format(key,i),
                    value = chunk,
                    tags = metadata,
                    group_id = group_id,
                    update = update,
                    cache = cache,
                    headers = self.credentials.to_headers()
                )
                futures.append(fut)
            response_time =T.time() -  start_time 
            res = wait(futures,timeout=None,return_when=ALL_COMPLETED)
            self.logger.info("PUT_CHUNKED {} {} {}".format(key,total_size,response_time))
            print("RESULT",res)
        

        


    def put_metadata(self,
                     payload:PutMetadataPayload,
                     update:bool=False)->Result[PutMetadataResponse, ApiError]:

        try:
            start_time = T.time()
            headers = self.credentials.to_headers()
            proxy = self.__get_current_proxy()
            
            def inner():
                if update:
                    self.delete(key= payload.ball_id, headers=headers,proxy=Some(proxy))
                response = proxy.put_metadata(payload= payload, headers=headers)
                if response.is_err:
                    error = response.unwrap_err()
                    waiting_time = T.time() - start_time
                    self.logger.info("WAIT {} {}".format(payload.ball_id,waiting_time))
                    # self.logger.info("MictlanX is processing your operation. Please wait, You've been waiting for = {}s ERROR={}".format(waiting_time,error.strerror))
                    raise error
                return response
            
            response = retry_call(inner, fargs=[],tries=30, delay=2,jitter=.5)
            response_time = T.time() - start_time
            self.logger.info("PUT_METADATA {} {} {}".format(payload.ball_id, payload.ball_size, response_time))
            return response
        except Exception as e:
            return Err(e)
    


    def put_data(self,
                     operation_id:str,
                     data:bytes,
                     )->Result[PutDataResponse, ApiError]:

        start_time    = T.time()
        headers       = self.credentials.to_headers()
        proxy         = self.__get_current_proxy()
    
        response      = proxy.put_data(operation_id=operation_id,data=data, headers=headers)
        if response.is_err:
            return response
        response_time = T.time() - start_time
        self.logger.info("PUT_DATA {} {} {}".format(operation_id,len(data), response_time))
        return response

    def delete(self,key:str = None,headers:Dict[str,str]={},proxy:Option[Proxy] = NONE)->Result[str,ApiError]:
        start_time    = T.time()
        _proxy        = proxy.unwrap_or(self.__get_current_proxy())
        # try:
        result        = _proxy.delete(ball_id=key,headers=headers)
        response_time = T.time() - start_time
        self.logger.info("DEL {} {} {}".format(key, 0, response_time))
        return result
        # except Exception as e:
            # return Err(e)
    
    

    def get_metadata(self,key:str):
        start_time = T.time()
        proxy      = self.__get_current_proxy()
        headers    = self.credentials.to_headers()
        def __inner():
            result    = proxy.get_metadata(key=key, headers=headers)
            if result.is_err:
                wt    = T.time() - start_time
                self.logger.error("WAIT {} {} {}".format(key,0,wt))
                error = result.unwrap_err()
                error_response:R.Response = error.response
                error_headers = error_response.headers
                error_code = int(error_headers.get("Error-Code","-1"))
                if error_code == 4 or error_code == 5:
                    raise error
                return result
            return result
        return retry_call(__inner,fargs=[],tries=50, delay=1,jitter=1,max_delay=2)


        # if force :
            
            
    def get_chunked(self, key:str, cache:bool = False, force:bool = True)-> Result[GetChunkedResponse, R.RequestException]:
        start_time = T.time()
        proxy            = self.__get_current_proxy()
        chunks_metadata_result = proxy.get_group_metadata(group_id=key)
        # hasher = H.sha256()
        if chunks_metadata_result.is_ok:
            chunks_metadata = chunks_metadata_result.unwrap()
            total_chunks = len(chunks_metadata)
            expected_total_size = sum(list(map(lambda x: x.size, chunks_metadata)))

            max_workers = total_chunks if total_chunks <= self.max_workers else self.max_workers
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                chunk_futures_results:List[Future[Result[GetBytesResponse,ApiError]]] = []
                for chunk_metadata in chunks_metadata:
                    chunk_future_result = executor.submit(self.get, key = chunk_metadata.id, cache= cache, force=force)
                    chunk_futures_results.append(chunk_future_result)
            responses_times = []
            def __inner():
                total_size = 0
                for chunk_future_result in concurrent.futures.as_completed(chunk_futures_results):
                    # print("AAA")
                    get_bytes_response_result = chunk_future_result.result()
                    if get_bytes_response_result.is_ok:
                        get_bytes_response = get_bytes_response_result.unwrap()
                        # get_bytes_responses.append(get_bytes_response)
                        chunk_size  = len(get_bytes_response.value)
                        total_size  += chunk_size
                        # print(get_bytes_response.metadata.tags)
                        chunk       = Chunk(
                            group_id=key,
                            index=int(get_bytes_response.metadata.tags.get("index","0")),
                            metadata=get_bytes_response.metadata.tags,
                            data=get_bytes_response.value
                        )
                        responses_times.append(get_bytes_response.response_time)
                        yield chunk
                    else:
                        print("FAIL")
                        return get_bytes_response_result
            xs = __inner()
            # print("XS",xs)
            chunks = Chunks(chs=xs , n=total_chunks)
            # get_bytes_responses.sort(key=lambda o: int(o.metadata.tags.get("index","0")))
            response_time = np.array(responses_times).max
            res           = GetChunkedResponse(chunks=chunks,response_time=response_time)
            return Ok(res)
        else:
            return chunks_metadata_result
            

    def get_chunked_ndarray(self, key:str, cache:bool = False, force:bool = True)->Result[GetNDArrayResponse,ApiError]:
        start_time = T.time()
        result     = self.get_chunked(key=key,cache=cache,force=force)
        if result.is_ok:
            response       = result.unwrap()
            maybe_ndarray  = response.chunks.to_ndarray()
            if maybe_ndarray.is_none:
                return Err(CannotConvertChunksToNDArray())
            
            ndarray,metadata = maybe_ndarray.unwrap()
            response_time = T.time() - start_time
            self.logger.info("GET_CHUNKED_NDARRAY {} {} {}".format(key,metadata.size,response_time))
            return Ok(GetNDArrayResponse(value = ndarray, metadata = metadata, response_time = response_time))
        else:
            return result


    def get(self,key:str=None, cache:bool=False, force:bool=True)->Result[GetBytesResponse,ApiError]:
        start_time = T.time()
        proxy = self.__get_current_proxy()
        headers = self.credentials.to_headers()
        def __inner():
            result = proxy.get( key,  headers= headers )
            if result.is_err:
                wt = T.time() - start_time
                error = result.unwrap_err()
                self.logger.error(str(error))
                
                self.logger.info("WAIT {} {} {}".format(key,0, wt))
                raise error
            return result


        if(force):
            result =retry_call(__inner,fargs=[],tries=50, delay=1, jitter=1, max_delay=2)
            # proxy.get( key,  headers= headers )
            # return result
        else:
            if(cache and key in self.cache.cache ):
                element = self.cache.get(key)
                metadata,value = element
                st = T.time() - start_time
                self.logger.info("HIT {} {} {}".format(key, len(value), st))
                result = Ok(GetBytesResponse(value =value,metadata =metadata,response_time=st ))
                # return result
            else:
                st = T.time() - start_time
                self.logger.info("MISS {} {} 0".format(key, 0,st ))
                result = retry_call(__inner,fargs=[],tries=50, delay=1, jitter=1, max_delay=2)

                # proxy.get(key,headers=headers)
        response_time = T.time() - start_time
        if(result.is_ok):
            response = result.unwrap()
            if cache and force:
                self.cache.put(key=key,value=(response.metadata,response.value))
            self.logger.info("GET {} {} {}".format(key,len(response.value),response_time))
            return result
        else:
            error = result.unwrap_err()
            self.logger.error(str(error))
        return result
    
    
    def get_ndarray(self,key:str=None, cache:bool=False, force:bool=True)->Result[GetNDArrayResponse,ApiError]:
        start_time = T.time()
        result = self.get(key=key,cache=cache,force=force)
        if(result.is_ok):
            response      = result.unwrap()
            metadata      = response.metadata
            dtype         = metadata.tags.get("dtype",np.float64)
            ndarray       = np.frombuffer(response.value,dtype=dtype)
            _shape = metadata.tags.get("shape")
            shape         = eval(_shape)
                # # print("METADATA",metadata)
                # print("SHAPE",_shape,shape)
            ndarray       = ndarray.reshape(shape)
            response_time = T.time() - start_time
            self.logger.info("GET_NDARRAY {} {} {}".format(key,len(response.value),response_time))
            return Ok(GetNDArrayResponse(value = ndarray, metadata = metadata, response_time = response_time))
        else:
            error = result.unwrap_err()
            self.logger.error(str(error))
            return result

    def logout(self):
        payload = LogoutPayload(app_id=self.credentials.application_id, client_id=self.credentials.client_id,token = self.credentials.authorization, secret=self.credentials.secret)
        self.xolo.logout(payload=payload)