from typing import Dict
class XoloCredentials(object):
    def __init__(self, application_id: str, client_id:str, secret:str,authorization:str):
        self.application_id = application_id
        self.client_id = client_id 
        self.secret = secret 
        self.authorization = authorization 
    def to_headers(self)->Dict[str,str]:
        return {
            "Authorization":self.authorization,
            "Application-Id":self.application_id,
            "Client-Id": self.client_id ,
            "Secret": self.secret 
        }