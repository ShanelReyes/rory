
from typing import Dict
class Metadata(object):
    def __init__(self,id:str, size:int, checksum:str, group_id:str,producer_id:str="MictlanX", tags:Dict[str,str ]= {}):
        self.id       = id
        self.size     = size
        self.checksum = checksum
        self.producer_id = producer_id
        self.group_id = group_id
        self.tags     = tags
    