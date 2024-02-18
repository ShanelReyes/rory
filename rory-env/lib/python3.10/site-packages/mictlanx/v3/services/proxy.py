import requests as R
from mictlanx.v3.interfaces.service import Service
from mictlanx.v3.interfaces.payloads import PutPayload,GetPayload,PutMetadataPayload
from mictlanx.v3.interfaces.responses import GetResponse,GetBytesResponse,PutMetadataResponse,PutDataResponse,GetMetadataResponse
from mictlanx.v3.interfaces.storage_node import PutPayload,PutResponse
from mictlanx.v3.interfaces.core import Metadata
# from mictlanx.v3.interfaces.errors import ApiError,Unauthorized,ServerInternalError
from option import Result,Ok,Err,Option,NONE
import time as T
import json as J 
from typing  import Dict,List,Any
import hashlib as H
from retry.api import retry_call
from nanoid import generate as nanoid_



class Proxy(Service):
    def __init__(self,ip_addr:str, port:int, api_version: Option[int] = NONE):
        super(Proxy,self).__init__(ip_addr=ip_addr, port= port, api_version=api_version, )
        self.put_url = '{}'.format(self.base_url)
        self.put_metadata_url = '{}/{}'.format(self.base_url,"metadata")
        self.put_data_url = lambda x : '{}/{}/{}'.format(self.base_url,"data",x)
        self.get_url = lambda x: '{}/{}'.format(self.base_url,x)
        self.del_url = lambda x: '{}/{}'.format(self.base_url,x)
        self.get_metadata_url = lambda x: '{}/metadata/{}'.format(self.base_url,x)
        self.get_group_metadata_url = lambda x: '{}/metadata/group/{}'.format(self.base_url,x)
        # http://localhost:8080/api/v3/metadata/group/grouped_matrix
    
    def delete(self,ball_id:str,headers:dict = {})->Result[str,R.RequestException]:
        try:
            url = self.del_url(ball_id)
            response = R.delete(
                url,
                headers=headers,
                timeout=1800
            )
            response.raise_for_status()
            # get_response = GetBytesResponse(value = response.content,metadata =metadata,response_time = T.time()- start_time)
            return Ok(ball_id)
        except Exception as e:
            return Err(e)
            # if(type(e) is R.RequestException):
                # response:R.Response = e.response
                # return Err(ServerInternalError(message = response.headers.get("Error-Message"), metadata = response.headers  ))
            # else:
                # return Err(e)


    def get(self,key:str,headers:Dict[str,str])->Result[GetBytesResponse,R.RequestException]:
        try:
            start_time = T.time()
            url = self.get_url(key)
            response = R.get(
                url,
                headers=headers,
                timeout=1800
            )
            metadata_response = R.get(
                self.get_metadata_url(key),
                headers=headers
            )
            response.raise_for_status()
            metadata_response.raise_for_status()
            _metadata = {**{"producer_id":"MictlanX"},**metadata_response.json()}
            metadata = Metadata(**_metadata)
            get_response = GetBytesResponse(value = response.content,metadata =metadata,response_time = T.time()- start_time)
            return Ok(get_response)
        except Exception as e:
            return Err(e)

    def get_metadata(self,key:str,headers:Dict[str,str])->Result[GetMetadataResponse,R.RequestException]:
        try:
            # start_time = T.time()
            url = self.get_metadata_url(key)
            response = R.get(
                url,
                headers=headers,
                timeout=1800
            )
            # metadata_response = R.get(
            #     self.get_metadata_url(key),
            #     headers=headers
            # )
            
            response.raise_for_status()
            metadata = GetMetadataResponse(**response.json())
            return Ok(metadata)
        except Exception as e:
            return Err(e)
    def get_group_metadata(self,group_id:str,headers:Dict[str,str] = {})->Result[List[GetMetadataResponse],R.RequestException]:
        try:
            start_time        = T.time()
            url               = self.get_group_metadata_url(group_id)
            response = R.get(
                url,
                headers=headers,
                timeout=1800


            )
            response.raise_for_status()
            metadatas = response.json()
            xs       = list(map(lambda m: GetMetadataResponse(**m),metadatas))
            # print("METADATA",m)
            return Ok(xs)
        except Exception as e:
            return Err(e)
            # metadata_response.raise_for_status()
            # metadata = metadata_response.json()
            # get_response = GetBytesResponse(value = response.content,metadata =metadata,response_time = T.time()- start_time)
            # return Ok(get_response)
    



    def put(self,
            value:bytes,
            group_id:str = nanoid_(),
            key:Option[str] = NONE,
            metadata: Dict[str,str] = {},
            storage_node_id:Option[str]=NONE,
            replica_manager_id:Option[str] = NONE,
            headers:Dict[str,str] = {} 
    )->Result[PutResponse,R.RequestException]:

        try:
            start_time            = T.time()
            ball_size             = len(value)
            hasher                =  H.sha256()
            hasher.update(value)
            checksum              =  hasher.hexdigest()
            _key                  = key.unwrap_or(checksum)
            put_metadata_payload  = PutMetadataPayload(key=_key,size=ball_size,checksum=checksum,group_id=group_id,node_id=storage_node_id, replica_manager_id=replica_manager_id, tags=metadata)
            put_metadata_response = self.put_metadata(payload=put_metadata_payload, headers= headers).unwrap()
            #     if result.is_err:
            #         raise result.unwrap_err()
            #     return result.unwrap()
                # pass
            # print("1")
            # put_metadata_response =  retry_call(retry_cb_put_metadata,fargs=[put_metadata_payload],tries=100,delay=1, jitter=1 )
            put_data_response:Result[PutDataResponse,R.RequestException]     = self.put_data(operation_id=put_metadata_response.operation_id,data=value,headers=headers) 
            service_time = T.time() - start_time
            response = put_data_response.map(lambda x: PutResponse(key= _key,size=ball_size,service_time=service_time))
            return response
        except Exception as e:
            # print("ERROR",e)
            return Err(e)
    

    # def put_ndarray(self, value:npt.ND)
   
    def put_data(self,operation_id:str,data:bytes, headers=Dict[str,str])->Result[PutDataResponse, R.RequestException]: 
        try:
            response = R.post(
                self.put_data_url(operation_id),
                files= {
                    "upload":(operation_id,data,"application/octet-stream")
                },
                headers= headers,
                timeout= 1800
            )
            response.raise_for_status()
            data = PutDataResponse(**response.json())
            return Ok(data)
        except Exception as e:
            return Err(e)
        
    # def put_metadata_chunks(self, payload:List[PutMetadataPayload], headers:Dict[str,str]={})-> Result[List[PutMetadataResponse], R.RequestException] :


    def put_metadata(self, payload:PutMetadataPayload, headers:Dict[str,str] = {})->Result[PutMetadataResponse,R.RequestException]:
        try:
            payload_dict = payload.to_dict()
            response = R.post(
                self.put_metadata_url,
                json= payload_dict,
                headers= headers,
                timeout= 1800
            )
            response.raise_for_status()
            response = PutMetadataResponse(**response.json())
            return Ok(response)
        except Exception as e:
            return Err(e)



# if __name__ ()