import requests as R
import json as J
from typing import Dict,List
from mictlanx.v4.xolo.api.dto.auth import AuthenticatedDTO
from option import Result,Ok,Err,Option,NONE,Some

class XoloAPI(object):
    def __init__(self,hostname:str, protocol:str ="http", port:Option[int]=NONE,version:int=4):
        self.protocol = protocol
        self.hostname  = hostname
        self.port     = port
        self.version  = version 
    
    def base_url(self):
        if self.port.is_none:
            return "{}://{}".format(self.protocol,self.hostname)
        else:
            port = self.port.unwrap()
            return "{}://{}:{}".format(self.protocol,self.hostname,port)
    def create_user(self,
                    username:str,
                    first_name:str,
                    last_name:str,
                    email:str,
                    password:str,
                    profile_photo:str=""
    )->Result[R.Response,Exception]:
        try:
            url  = "{}/api/v{}/users".format(self.base_url(),self.version)
            data = J.dumps({
                "username":username,
                "first_name":first_name,
                "last_name":last_name,
                "email":email,
                "password":password,
                "profile_photo":profile_photo
            })
            response = R.post(url=url,data=data)
            response.raise_for_status()
            return Ok(response)
        except Exception as e:
            return Err(e)
    
    def verify(self,access_token:str,username:str,secret:str="")->bool:
        try:
            url  = "{}/api/v{}/users/verify".format(self.base_url(),self.version)
            data = J.dumps({
                "access_token":access_token,
                "username":username,
                "secret":secret
            })
            response = R.post(url=url,data=data)
            response.raise_for_status()
            return True
        except Exception as e:
            return False

    def auth(self,
             username:str,
             password:str
    )->Result[AuthenticatedDTO,Exception]:
        try:
            url = "{}/api/v{}/users/auth".format(self.base_url(),self.version)
            data = J.dumps({
                "username":username,
                "password":password
            })
            response = R.post(url=url,data=data)
            response.raise_for_status()
            data_json = response.json()
            return Ok(
                AuthenticatedDTO(**data_json)
            )
        except Exception as e:
            return Err(e)
        
    # ACL
    def grants(self,grants:Dict[str,Dict[str,List[str]]])->Result[bool, Exception]:
        try:
            url = "{}/api/v{}/users/grants".format(self.base_url(),self.version)
            data = J.dumps({
                "grants":grants
            })
            response = R.post(url=url, data=data)
            response.raise_for_status()
            return Ok(True)
        except Exception as e:
            return Err(e)
    def check(self,role:str,resource:str, permission:str)->bool:
        try:
            url = "{}/api/v{}/users/check".format(self.base_url(),self.version)
            data = J.dumps({
                "role":role,
                "resource":resource,
                "permission":permission
            })
            response = R.post(url=url, data=data)
            response.raise_for_status()
            response_data = response.json()
            return response_data.get("result",False)
        except Exception as e:
            return False


if __name__ == "__main__":
    xolo_api = XoloAPI(hostname="localhost",protocol="http",port=Some(10001),version=4)
    x = xolo_api.auth(username="test",password="password123")
    if x.is_err:
        print(x.unwrap_err())
    else:
        response = x.unwrap()
        is_verified = xolo_api.verify(access_token=response.access_token,username=response.username, secret=response.temporal_secret)
        print("IS_VERIFIED",is_verified)
        y = xolo_api.grants(grants={
            "user":{
                "client2":["write","read"]
            }
        })
        print(y)
        check_result = xolo_api.check(role='user',resource="client2",permission="write")
        print(check_result)
        check_result = xolo_api.check(role='user',resource="client2",permission="delete")
        print(check_result)

    # x = 
    # xolo_api.create_user(username="test",first_name="test",last_name="test",email="email@emil.com",password="password123")
    # print(x)
