from uuid import uuid4 
import time
# 
class Worker(object):
    def __init__(self,**kws):
        self.workerId     = kws.get("workerId",str(uuid4))
        self.port      = kws.get("port")
        self.balls     = kws.get("balls", [])
        self.isStarted = kws.get("isStarted",False)
        self.createdAt = kws.get("createdAt",time.time())