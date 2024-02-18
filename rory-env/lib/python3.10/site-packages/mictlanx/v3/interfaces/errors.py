from abc import ABC
from requests import Response
class ApiError(ABC):
    def __init__(self,**kwargs):
        self.message   = kwargs.get("message","ERROR")
        self.code      = kwargs.get("code",-1)
        self.metadata  = kwargs.get("metadata",{})
    def __str__(self):
        return "ApiError(message={}, code={})".format(self.message,self.code)

class NotAvailableNodes(ApiError):
    def __init__(self,*args,**kwargs):
        super(NotAvailableNodes,self).__init__(*args,**{**kwargs,"code":0})
class Unauthorized(ApiError):
    def __init__(self,*args,**kwargs):
        super(Unauthorized,self).__init__(*args,**{**kwargs,"code":1})
class NotFound(ApiError):
    def __init__(self,*args,**kwargs):
        super(NotFound,self).__init__(*args,**{**kwargs,"code":2})
class ServerInternalError(ApiError):
    def __init__(self,*args,**kwargs):
        super(ServerInternalError,self).__init__(*args,**{**kwargs,"code":3})

class CannotConvertChunksToNDArray(ApiError):
    def __init__(self,*args,**kwargs):
        super(CannotConvertChunksToNDArray,self).__init__(*args,**{**kwargs,"message":"Cannot convert chunks to ndarray.","code":4})
        