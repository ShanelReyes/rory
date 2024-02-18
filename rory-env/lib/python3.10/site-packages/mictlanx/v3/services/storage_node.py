import requests as R
from mictlanx.v3.interfaces.service import Service
from mictlanx.v3.interfaces.storage_node import PutPayload,PutResponse
from mictlanx.v3.interfaces.errors import ApiError,Unauthorized,ServerInternalError
from option import Ok,Result,Err
from nanoid import generate as nanoid_
from option import Option,NONE,Some

class StorageNode(Service):
    def __init__(self, ip_addr:str, port:int, api_version:Option[int]=Some(3)):
        super(StorageNode,self).__init__(ip_addr=ip_addr, port= port, api_version=api_version)
        self.put_url = "{}".format(self.base_url)
        self.get_url = lambda x: "{}/{}".format(self.base_url,x)
    # ________________________________________________________
    def put(self,bytes,**kwargs)->Result[PutResponse,ApiError]:
        try:
            ball_id  = kwargs.get("key",nanoid_())
            response = R.post(self.put_url,files ={ball_id: (ball_id,bytes)} )
            # print("SN_RESPONSE",response)
            response.raise_for_status()
            response_json = PutResponse(**response.json())
            return Ok(response_json)
        except R.RequestException as e:
            response:R.Response = e.response
            return Err(ServerInternalError(message = response.headers.get("Error-Message"), metadata = response.headers  ))
        
    def get(self,**kwargs)->Result[bytes, ApiError]:
        key = kwargs.get("key")
        try:
            response:R.Response = R.get(self.get_url(key))
            response.raise_for_status()
            return Ok(response.content)
        except R.RequestException as e:
            response:R.Response = e.response
            return Err(ServerInternalError(message = response.headers.get("Error-Message"), metadata = response.headers  ))