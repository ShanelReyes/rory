import requests as R
from requests import HTTPError,RequestException
from mictlanx.v3.interfaces.service import Service
from mictlanx.v3.interfaces.replica_manager import PutPayload,PutResponse,GetPayload,GetResponse,CompleteOperationPayload
# from mictlanx.v3.interfaces.replica_manager import PutResponse
from mictlanx.v3.interfaces.errors import ApiError,NotAvailableNodes,Unauthorized,ServerInternalError
# from typing import
from option import Result,Ok,Err,Option,NONE

class ReplicaManager(Service):
    def __init__(self, ip_addr:str, port:int, api_version:Option[int]=NONE):
        super(ReplicaManager,self).__init__(ip_addr=ip_addr,port=port,api_version=api_version)
        self.put_url                = "{}/{}".format(self.base_url,"put")
        self.get_url                = lambda x: "{}/{}/{}".format(self.base_url,"get",x)
        self.complete_operation_url = lambda node_id, operation_type, operation_id: "{}/{}/{}/{}/{}".format(self.base_url,"operations",node_id,operation_type,operation_id)


    
    def complete_operation(self,payload:CompleteOperationPayload,**kwargs)->Result[None,ApiError]:
        try:
            response = R.post(self.complete_operation_url(payload.node_id,payload.operation_type,payload.operation_id),headers={
                "Client-Id":kwargs.get("client_id"),
                "Authorization":kwargs.get("authorization"),
                "Secret":kwargs.get("password")
            })
            response.raise_for_status()
            return Ok(None)
        except RequestException as e:
            response:R.Response = e.response
            return Err(ServerInternalError(message = response.headers.get("Error-Message"), metadata = response.headers))

        # return 

    def put(self,payload:PutPayload,**kwargs)->Result[PutResponse,ApiError]:
        # print("PUT_URL",self.put_url)
        try:
            response = R.post(self.put_url,json = payload.to_dict(),headers={
                "Client-Id":kwargs.get("client_id"),
                "Authorization":kwargs.get("authorization"),
                "Password-Hash":kwargs.get("password")
            })
            response.raise_for_status()
            response_json = PutResponse(**response.json())
            return Ok(response_json)
        except RequestException as e:
            response:R.Response = e.response
            if(response.status_code == 404):
                return Err(NotAvailableNodes(message = response.headers.get("Error-Message")))
            elif(response.status_code == 401):
                return Err(Unauthorized(message = response.headers.get("Error-Message")))
            else:
                return Err(ServerInternalError(message = response.headers.get("Error-Message")))
    
    def get(self,payload:GetPayload,**kwargs)->Result[GetResponse,ApiError]:
        try:
            response = R.get(self.get_url(payload.ball_id),headers={
                "Client-Id":kwargs.get("client_id"),
                "Authorization":kwargs.get("authorization"),
                "Password-Hash":kwargs.get("password")
            })
            print("RM_RESPONSE",response)
            response.raise_for_status()
            response_json = response.json()
            print("RM_GET",response_json)
            return Ok(GetResponse(**response_json))
        except RequestException as e:
            response:R.Response = e.response
            return Err(ServerInternalError(message = response.headers.get("Error-Message"), metadata = response.headers))