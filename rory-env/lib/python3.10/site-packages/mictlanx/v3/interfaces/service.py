from abc import ABC
from option import Option,NONE

class Service(ABC):
    def __init__(self,ip_addr:str, port:int, api_version:Option[int]=NONE,protocol:str="http"):
        self.ip_addr:str     = ip_addr
        self.port:int        = port

        self.api_version:int = api_version.unwrap_or(3)
        self.protocol = protocol
        self.base_url =    "{}://{}:{}/api/v{}".format(
            protocol,
            self.ip_addr,
            self.port,
            self.api_version
        ) if port>1024 else "{}://{}/api/v{}".format(protocol, self.ip_addr,self.api_version)