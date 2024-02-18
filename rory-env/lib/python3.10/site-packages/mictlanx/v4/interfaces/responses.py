from typing import Dict ,Any,TypeVar,Generic,List
import numpy.typing as npt
from dataclasses import dataclass
# from typing import Generic,TypeVar,Dict,List
T = TypeVar("T")

@dataclass
class PutChunkedResponse:
    node_id:str
    combined_key:str
    bucket_id:str 
    key:str
    size:int
    throughput:float
    service_time:float
    
class Metadata(object):
    def __init__(self,
                 key:str, # Unique identifier 
                 size:int, # Size in bytes of the data
                 checksum:str, # Sha256 checksum
                 tags:Dict[str,str], # User-defined metadata
                 content_type:str, # Define the type of the content
                 producer_id:str, # Unique identifier of the user that allocate the data. 
                 ball_id:str, # Unique identifier used for segmentation purposes ball_id -> [chunk1, chunk2,...,chunkN]
                 bucket_id:str = "", # Unique identifier used for MictlanX Sync
                 is_disabled:bool=False,
                 **kwargs
    ):
        self.size = size
        self.checksum = checksum
        self.producer_id = producer_id
        self.tags=tags
        self.content_type= content_type
        self.key = key
        self.ball_id = ball_id
        self.is_disabled = is_disabled
        self.bucket_id = bucket_id
    def __str__(self):
        return "Metadata(key={}, ball_id={})".format(self.key,self.ball_id)
    
class GetBucketMetadataResponse(object):
    def __init__(self,peer_id:str,balls:List[Dict[str,Any]]):
        self.balls = list(map(lambda ball: Metadata(**ball),balls))
        self.peer_id:str = peer_id
    def __str__(self):
        return "Bucket(n={})".format(len(self.balls))

class GetMetadataResponse(object):
    def __init__(self,service_time:int,node_id:str,metadata:Dict[str,Any]):
        self.service_time = service_time
        self.node_id = node_id
        self.metadata = Metadata(**metadata)

class GetUFSResponse(object):
    def __init__(self,total_disk:int, used_disk:int, disk_uf:float):
        self.total_disk = total_disk
        self.used_disk  = used_disk
        self.disk_uf    = disk_uf
    def __str__(self):
        return "GetUFSResponse(total_disk={}, used_disk={}, disk_uf={})".format(self.total_disk, self.used_disk, self.disk_uf)

class PutMetadataResponse(object):
    def __init__(self, key:str,node_id:str, service_time:int, task_id:str):
        self.key = key 
        self.service_time = service_time 
        self.node_id = node_id
        self.task_id = task_id 
class PutDataResponse(object):
    def __init__(self,service_time:int,throughput:float):
        self.service_time = service_time 
        self.throughput = throughput
# class GetResponse(object):
#     def __init__(self,)

class GetResponse(Generic[T]):
    def __init__(self,value:T, metadata:Metadata,response_time:int=-1):
        self.value:T           = value
        self.metadata:Metadata = metadata
        self.response_time     = response_time

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
    
class PutResponse(object):
    def __init__(self,response_time:int,throughput:float,node_id:str,key:str=""):
        self.key           = key
        self.response_time = response_time
        self.node_id       = node_id
        self.throughput    = throughput