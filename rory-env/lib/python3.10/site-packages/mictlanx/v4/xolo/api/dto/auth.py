from typing import Dict
class AuthenticatedDTO(object):
    def __init__(self,username:str,first_name:str, last_name:str,email:str, profile_photo:str, access_token:str, temporal_secret:str,role:str,metadata:Dict[str,str]={}):
        self.username        = username
        self.first_name      = first_name
        self.last_name       = last_name
        self.email           = email
        self.profile_photo   = profile_photo
        self.access_token    = access_token
        self.metadata        = metadata
        self.temporal_secret = temporal_secret
        self.role            = role