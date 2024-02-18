from mictlanx.v3.interfaces.payloads import Payload
class PutPayload(Payload):
    def __init__(self,*args,**kwargs):
        super(PutPayload,self).__init__(*args,**kwargs)
        self.bytes = kwargs.get("bytes")

class PutResponse(object):
    def __init__(self,key:str, size:int, service_time:int):
        self.key          = key
        self.size         = size
        self.service_time = service_time
    def __str__(self):
        return "PutPayload(key={}, service_time={})".format(self.key,self.service_time)