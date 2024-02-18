from abc import ABC
from nanoid import generate as nanoid_
import numpy as np
import numpy.typing as npt
import hashlib as H
from typing import List,Dict
from option import Option,Some,NONE
from durations import Duration

class Payload(ABC):
    def __init__(self,*args,**kwargs):
        pass
    def to_dict(self):
        return self.__dict__
class LogoutPayload(Payload):
    def __init__(self,app_id:str=None, client_id:str=None, token:str=None, secret:str=None):
        super(LogoutPayload,self).__init__()
        self.app_id:str    = app_id
        self.client_id:str = client_id
        self.token:str     = token
        self.secret:str    =  secret


class VerifyTokenPayload(Payload):
    def __init__(self,app_id:str=None,client_id:str=None,token:str=None,secret:str=None):
        super(VerifyTokenPayload,self).__init__()
        self.app_id:str    = app_id
        self.client_id:str = client_id
        self.token:str     = token
        self.secret:str    = secret

class RefreshTokenPayload(Payload):
    def __init__(self,app_id:str=None, client_id:str=None, token:str=None, secret:str=None):
        super(RefreshTokenPayload,self).__init__()
        self.app_id:str    = app_id
        self.client_id:str = client_id
        self.token:str     =  token
        self.secret:str    = secret

class SignUpPayload(Payload):
    def __init__(self,app_id:str=None, client_id:str=None, secret:str=None, metadata:Dict[str,str]={},claims:Dict[str,str]={},expires_in:Option[str]=NONE):
        super(SignUpPayload,self).__init__()
        self.app_id:str             = app_id
        self.client_id:str          = client_id
        self.secret:str             = secret
        self.metadata:Dict[str,str] = metadata
        self.claims:Dict[str,str]   = claims
        self.expires_in = int(Duration(expires_in.unwrap_or("1d")).to_seconds())

class AuthTokenPayload(Payload):
    def __init__(self,app_id:str = None, client_id:str=None,secret:str=None, expires_in:Option[str]=NONE):
        super(AuthTokenPayload,self).__init__()
        self.app_id     = app_id
        self.client_id  = client_id
        self.secret     = secret
        self.expires_in = int(Duration(expires_in.unwrap_or("1d")).to_seconds())

class GetPayload(Payload):
    def __init__(self,key:str=None):
        super(GetPayload,self).__init__()
        self.key:str = key


class PutMetadataPayload(Payload):
    def __init__(self,key:str,size:int,checksum:str,producer_id:Option[str]=NONE, group_id:str=nanoid_(),node_id:Option[str]=NONE,replica_manager_id:Option[str]=NONE,tags:Dict[str,str]={}):
        super(PutMetadataPayload,self).__init__()
        self.ball_size              = size
        self.node_id                = node_id
        self.checksum = checksum 
        self.producer_id            =producer_id.unwrap_or("MictlanX")
        self.ball_id:str            = self.checksum if key == "" or key == None else key
        self.group_id = group_id
        self.replica_manager_id     = replica_manager_id 
        self.tags:Dict[str,str] = tags
    
    def to_dict(self):
        x = {
            "ball_id":self.ball_id,
            "ball_size":self.ball_size,
            "group_id":self.group_id,
            "checksum": self.checksum,
            "producer_id":self.producer_id,
            "tags":self.tags
        }
        if self.node_id.is_some:
            x["node_id"] = self.node_id.unwrap()
        if self.replica_manager_id.is_some:
            x["replica_manager_id"] = self.replica_manager_id.unwrap()
        return x
    def __str__(self):
        return "PutPayload(key={})".format(self.key)
    

class PutPayload(Payload):
    def __init__(self,key:str=nanoid_(),data:bytes=None, metadata:Dict[str,str]=None):
        super(PutPayload,self).__init__()
        self.key:str                = key
        self.bytes:bytes            = data
        self.metadata:Dict[str,str] = metadata
    
    def __str__(self):
        return "PutPayload(key={})".format(self.key)
    
    def get_hash(self)->str:
        sha256 = H.sha256()
        sha256.update(self.bytes)
        return sha256.hexdigest()


class PutNDArrayPayload(Payload):
    def __init__(self,key:str=nanoid_(),ndarray:npt.NDArray=None,tags:Dict[str,str]={}):
        super(PutNDArrayPayload,self).__init__()
        self.key   = key
        self.ndarray:npt.NDArray = ndarray
        self.tags = tags
        # kwargs.get("metadata",{})
    def into(self)->PutPayload:
        return PutPayload(
            data = self.ndarray.tobytes(), 
            key = self.key, 
            metadata = {**self.tags, "shape": str(self.ndarray.shape),"dtype":str(self.ndarray.dtype)
                        }

        )
    


class ExposedPort(object):
    def __init__(self,ip_addr:Option[str], host_port:int, container_port:int, protocolo:Option[str]):
        self.ip_addr = ip_addr
        self.host_port = host_port
        self.container_port = container_port 
        self.protocolo = protocolo
    def to_dict(self): 
        x = {}
        if(self.ip_addr.is_some):
            x["ip_addr"] = self.ip_addr.unwrap()
        x["host_port"] = self.host_port
        x["container_port"] = self.container_port 
        if(self.protocolo.is_some):
            x["protocolo"] = self.protocolo.unwrap()
        return x
            
    
class SummonContainerPayload(Payload):
    def __init__(self,
                 container_id:str,
                 image:str,
                #  image_tag:str,
                 hostname:str,
                 exposed_ports:List[ExposedPort],
                 envs:Dict[str,str],
                 memory:int,
                 cpu_count:int,
                 mounts:Dict[str,str],
                 network_id:str,
                 selected_node:Option[str] = NONE,
                 labels:Dict[str,str ]= {},
                 force:Option[bool]=NONE,
                 ip_addr:Option[str]=NONE 
                ):
        self.container_id  = container_id
        self.image         = image
        # self.image_tag     = image_tag 
        self.hostname      = hostname
        self.exposed_ports = exposed_ports
        self.envs          = envs
        self.memory        = memory
        self.cpu_count     = cpu_count
        self.mounts        = mounts
        self.network_id    = network_id
        self.selected_node = selected_node
        # .unwrap_or("0")
        self.labels        = labels
        self.force         = force.unwrap_or(True)
        self.ip_addr       = ip_addr.unwrap_or(self.container_id)
    def to_dict(self):
        current_dict = {
            "container_id":self.container_id,
            "image":self.image,
            "hostname":self.hostname,
            "exposed_ports":list(map(lambda x: x.to_dict(),self.exposed_ports)),
            "envs":self.envs,
            "memory":self.memory,
            "cpu_count": self.cpu_count, 
            "mounts":self.mounts,
            "network_id":self.network_id,
            "labels":self.labels,
            "force":self.force
        }
        if self.selected_node.is_some:
            current_dict["selected_node"] = self.selected_node.unwrap()

        return current_dict
    
if __name__ == "__main__":
    sc = SummonContainerPayload(
        container_id="scm-0",
        image="secure-clustering:manager",
        hostname = "scm-0",
        exposed_ports=[ExposedPort(NONE,6000,6000,NONE)],
        envs= {
            "NODE_PORT":"6000",
            "NODE_HOST":"0.0.0.0",
            "NODE_PREFIX":"scw-",
            "MAX_WORKERS":"5",
            "DOCKER_IMAGE_NAME":"secure-clustering",
            "DOCKER_IMAGE_TAG":"worker",
            "MICTLANX_SUMMONER_IP_ADDR":"10.0.0.100",
            "MICTLANX_SUMMONER_PORT":"15000",
            "MICTLANX_API_VERSION":"3",
            "DEBUG":"0",
            "RELOAD":"0",
            "LOG_PATH":"/log",
            "SINK_PATH":"/sink", 
            "SOURCE_PATH":"/source",
            "TESTING":"0",
            "MAX_RETRIES":"10",
            "LOAD_BALANCING":"0"
        },
        memory=1000000000,
        cpu_count=1,
        mounts={
            "/log":"/log",
            "/sink":"/sink",
            "/source":"/source"   
        }
    )
    print(sc.to_dict())
