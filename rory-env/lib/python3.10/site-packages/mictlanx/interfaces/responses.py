from typing import Generic ,TypeVar,Dict,Any
# import numpy as np
import numpy.typing as npt




class PutResponse(object):
    def __init__(self,**kwargs):
        self.id:str            = kwargs.get("id","PUT_ID")
        self.metadata:Metadata = Metadata(**kwargs.get("metadata",{}))
        self.status:int        = kwargs.get("status",0)
        self.size:int          = kwargs.get("size",0)
        self.throughput:float  = kwargs.get("throughput",0.0)
        self.service_time:int  = kwargs.get("service_time",0)
    # ______________________________________________
    def new(**kwargs):
        return PutResponse(**kwargs)
    def empty():
        return PutResponse()
    def __str__(self):
        return "PutResponse(id={}, metadata.id={}, metadata.size={}, service_time={}, throughput={}, status={})".format(self.id,self.metadata.id,self.metadata.size, self.service_time, self.throughput,self.status)
    

class Metadata(object):
    def __init__(self,**kwargs):
        self.id:str             = kwargs.get("id","BALL_ID")
        self.size:int           = kwargs.get("size",0)
        self.checksum:str       = kwargs.get("checksum","CHECKSUM")
        self.tags:Dict[str,str] = kwargs.get("tags",{})
        # self.reads:int          = kwargs.get("reads",0)
        self.created_at:int     = kwargs.get("created_at",0)
        # self.last_read:int      = kwargs.get("last_read",0)
    
    def empty(**kwargs):
        return Metadata(**kwargs)

    def __str__(self)->str:
        return "Metadata(id={}, size={}, checksum={}, created_at={})".format(self.id,self.size,self.checksum,self.created_at)


T = TypeVar("T")

class GetResponse(Generic[T]):
    def __init__(self,**kwargs):
        self.id:str                   = kwargs.get("id","GET_ID")
        self.status:int               = kwargs.get("status",0)
        self.metadata:Metadata        = Metadata(**kwargs.get("metadata",{}))
        self.throughput:float         = kwargs.get("throughput",0.0)
        self.service_time:int         = kwargs.get("service_time",1)
        self.value:T                  = kwargs.get("value",None)
        self.preserved_integrity:bool = kwargs.get("preserved_integrity",True)
    # ______________________________________________
    def new(**kwargs):
        return GetResponse(**kwargs)
    def empty():
        return GetResponse(status = -2)
    def __str__(self)->str:
        return "GetResponse(id={}, metadata.id={}, metadata.size={}, service_time={}, throughput={}, status={})".format(self.id,self.metadata.id,self.metadata.size, self.service_time, self.throughput,self.status)


class GetNDArrayResponse(GetResponse[npt.NDArray]):
    def __init__(self,**kwargs):
        super.__init__(**kwargs)
        # self.value

class GetBytesResponse(GetResponse[bytes]):
    def __init__(self,**kwargs):
        super.__init__(**kwargs)

