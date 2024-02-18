import json

class PutParameters(object):
    def __init__(self,**kwargs):
        self.id        = kwargs.get("id","ball")
        self.size      = kwargs.get("size",0)
        self.client_id = kwargs.get("client_id","")
        self.checksum  = kwargs.get("checksum","CHECKSUM")
        self.force     = kwargs.get("force",True)
        self.tags      = kwargs.get("tags",{})
    def to_json(self):
        return json.dumps(self.__dict__)
        

class GetParameters(object):
    def __init__(self,**kwargs):
        self.id    = kwargs.get("id")
        self._from = kwargs.get("from",None)
    
    def empty():
        return GetParameters(id="")
    def to_json(self):
        return json.dumps({
            "id" :self.id,
            "from": self._from
        })