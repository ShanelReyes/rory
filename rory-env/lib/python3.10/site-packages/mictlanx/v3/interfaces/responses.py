
from mictlanx.v3.interfaces.core import Metadata
from mictlanx.utils.segmentation import Chunks,Chunk
import numpy.typing as npt
from typing import Generic,TypeVar,Dict,List
T = TypeVar("T")




class GetMetadataResponse(object):
    def __init__(self,id:str, size:int, checksum:str, group_id:str,producer_id:str="MictlanX",tags:Dict[str,str]={}):
        self.id          = id
        self.size        = size
        self.checksum    = checksum
        self.producer_id = producer_id
        self.group_id    = group_id
        self.tags        = tags
    def __str__(self):
        return "GetMetadataResponse(key={}, checksum={}, size={})".format(self.key,self.checksum,self.size)
class SummonContainerResponse(object):
    def __init__(self,container_id:str,ip_addr:str, port:int, service_time:int):
        self.container_id = container_id
        self.service_time = service_time
        self.ip_addr= ip_addr
        self.port =port
        
        

class SignUpResponse(object):
    def __init__(self,token:str=None,service_time:int=None):
        self.token:str = token
        # kwargs.get("token")
        self.service_time = service_time
        # kwargs.get("service_time",0)
    def __str__(self):
        return "SignUpResponse(token={}, service_time={})".format(self.token,self.service_time)
class RefreshTokenResponse(object):
    def __init__(self,**kwargs):
        self.token = kwargs.get("token")
        self.service_time = kwargs.get("service_time",0)

class LogoutResponse(object):
    def __init__(self,**kwargs):
        self.logout = kwargs.get("logout",False)
        self.service_time = kwargs.get("service_time",0)

class VerifyTokenResponse(object):
    def __init__(self,**kwargs):
        self.verified = kwargs.get("verified",False)
        self.service_time = kwargs.get("service_time",0)


class AuthResponse(object):
    def __init__(self,client_id=None, token=None, jti=None, metadata={},service_time=None):
        self.client_id    = client_id
        self.token        = token
        self.jti          = jti
        self.metadata     = metadata
        self.service_time = service_time
    def __str__(self):
        return "AuthResponse(client_id={})".format(self.client_id)


class GetInMemoryResponse(object):
    def __init__(self,**kwargs):
        self.node_id       = kwargs.get("key")
        self.response_time = kwargs.get("response_time")
        self.bytes         = kwargs.get("bytes")



class GetResponse(Generic[T]):
    def __init__(self,value:T, metadata:Metadata,response_time:int=-1):
        self.value:T           = value
        self.metadata:Metadata = metadata
        self.response_time     = response_time
class GetChunkedResponse(object):
    def __init__(self,chunks:Chunks,response_time:int=-1):
        self.chunks = chunks
        self.response_time           = response_time
        

class GetBytesResponse(GetResponse[bytes]):
    def __init__(self,value:bytes, metadata:Metadata,response_time:int=-1):
        super(GetBytesResponse,self).__init__(value=value,metadata=metadata,response_time=response_time)
    def __str__(self):
        return "GetResponse(response_time={}, size={})".format(self.response_time,len(self.value))
class GetNDArrayResponse(GetResponse[npt.NDArray]):
    def __init__(self,value:npt.NDArray, metadata:Metadata,response_time:int=-1):
        super(GetNDArrayResponse,self).__init__(value=value,metadata=metadata,response_time=response_time)
    def __str__(self):
        return "GetResponse(response_time={}, shape={})".format(self.response_time,self.value.shape)


class PutDataResponse(object):
    def __init__(self,ball_id:str,service_time:int,waiting_time:int):
        self.ball_id      = ball_id
        self.service_time = service_time
        self.waiting_time = waiting_time

class PutMetadataResponse(object):
    def __init__(self, arrival_time:int, operation_id:str, key:str, service_time: int,is_replica:bool):
        self.arrival_time  = arrival_time 
        self.operation_id  = operation_id
        self.key           = key
        self.service_time  = service_time
        self.is_replica = is_replica
        