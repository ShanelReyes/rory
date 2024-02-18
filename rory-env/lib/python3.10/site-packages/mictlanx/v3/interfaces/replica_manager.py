from nanoid import generate as nanoid_
from mictlanx.v3.interfaces.payloads import Payload
import hashlib as H
class GetPayload(Payload):
    def __init__(self,**kwargs):
        self.ball_id = kwargs.get("key")
    def __str__(self):
        return "GetPayload(ball_id={}, ball_size={})".format(self.ball_id)

class CompleteOperationPayload(Payload):
    def __init__(self,*args,**kwargs):
        super(CompleteOperationPayload,self).__init__(*args,**kwargs)
        self.node_id        = kwargs.get("node_id")
        self.operation_id   = kwargs.get("operation_id")
        self.operation_type = kwargs.get("operation_type")
        
class PutPayload(Payload):
    def __init__(self,**kwargs):
        self.ball_id = kwargs.get("ball_id")
        self.ball_size = kwargs.get("size")
        self.metadata  = kwargs.get("metadata")
    def new(**kwargs):
        return PutPayload(ball_id = nanoid_(), **kwargs)

    def __str__(self):
        return "PutPayload(ball_id={}, ball_size={})".format(self.ball_id,self.ball_size)



class GetResponse(object):
    def __init__(self,**kwargs):
        self.node_id      = kwargs.get("node_id")        
        self.ip_addr      = kwargs.get("ip_addr")
        self.port         = kwargs.get("port")
        self.service_time = kwargs.get("service_time")
        self.metadata      = kwargs.get("metadata",{})
        

class PutResponse(object):
    def __init__(self,**kwargs):
        self.operation_id = kwargs.get("operation_id")
        self.node_id = kwargs.get("node_id")
        self.ip_addr = kwargs.get("ip_addr")
        self.port    = kwargs.get("port")
        self.service_time = kwargs.get("service_time")
    def __str__(self):
        return "PutResponse(operation_id={}, node_id={}, ip_addr={}, port={})".format(self.operation_id,self.node_id,self.ip_addr,self.port)