
from ipcqueue import posixmq
from threading import Thread
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.resource_tracker import unregister
from option import Some,NONE,Option
from mictlanx.logger.log import Log
from typing import Dict,Any,List,Tuple
from uuid import uuid4
import logging
import json as J
import time as T
from uuid import uuid4
from mictlanx.v4.client import Client
from mictlanx.v4.interfaces.index import Ball
from mictlanx.v4.xolo.utils import Utils as XoloUtils
from dataclasses import dataclass,field
from humanfriendly import parse_size


# class InMemoryBall(object):
@dataclass
class Task:
    task_id:str
    start_time:float
    key:str
    status:int=0
    response_time:float = -1.0
    data:Option[memoryview]=NONE
    event:Dict[str,Any] = field(default_factory=dict)


class Client(object):
    def __init__(self,client_id:str= uuid4().hex):
        if client_id =="mictlanx":
            raise Exception("mictlanx is a special queue name. Please select another queue name")
        self.client_id                                              = client_id
        self.__q                                                    = posixmq.Queue('/{}'.format(client_id))
        self.__shared_memory                                        = SharedMemory(name="/mictlanx-shm",create=False,size=parse_size("512MB"))
        self.__pending_tasks:Dict[str, Task]                        = {}
        self.__completed_tasks:Dict[str,Any]                        = {}
        # self.__memory_distribution_schema:Dict[str, Tuple[int,int]] = {}
        # self.__balls:Dict[str, Ball]                                = {}
        self.__log                                                  = Log(
            name = client_id,
            path = "/mictlanx/client",
            console_handler_filter= lambda record: record.levelno == logging.DEBUG or record.levelno == logging.INFO or record.levelno == logging.ERROR
        )

        self.__mictlanx_q    = posixmq.Queue(name="/mictlanx")
        self.__is_running    = True
        self.__daemon_thread = Thread(target=self.__run,name="mictlanx",daemon=True)
        self.__daemon_thread.start()

    def put(self, path:str,key:str="")->str:
        try:
            start_time = T.time()
            task_id = "task-{}".format(uuid4().hex)
            event = {
                "client_id":self.client_id,
                "task_id":task_id,
                "key":key, 
                "path":path,
                "operation_type":"PUT",
            }
            self.__mictlanx_q.put_nowait(item=event, priority=100)
            service_time = T.time() - start_time
            self.__log.info({
                "client_id":self.client_id,
                "task_id":task_id,
                "key":key,
                "path":path,
                "operation_type":"PUT",
                "service_time":service_time
            })
            self.__pending_tasks[task_id] = Task(task_id=task_id,start_time=start_time,key=key)
            return task_id
            # self.__pending_tasks.append((task_id,T.time()))
        except Exception as e:
            print("PUT_ERROR",e)
    
    def get(self,key:str="")->str:
        try:
            start_time = T.time()
            task_id = "task-{}".format(uuid4().hex)
            payload = {
                "task_id":task_id,
                "client_id":self.client_id,
                "key":key,
                "operation_type":"GET",
            }
            self.__mictlanx_q.put_nowait(item=payload, priority=1)
            
            service_time = T.time() - start_time
            self.__log.debug({
                "task_id":task_id,
                "operation_type":"GET",
                "key":key,
                "service_time":service_time
            })
            self.__pending_tasks[task_id] = Task(task_id=task_id,start_time=start_time,key=key)
            return task_id
        except Exception as e:
            print("GET_ERROR",e)
    
    def get_task_result(self,task_id:str,timeout:int=10,interval:float=0.5)->Option[Task]:
        start_time = T.time() 
        elapsed    = T.time() - start_time
        while elapsed < timeout:
            if task_id in self.__completed_tasks:
                return Some(self.__completed_tasks[task_id])
            T.sleep(interval)
            elapsed = T.time() - start_time

    def shutdown(self):
        try:
            self.__is_running = False
            self.__daemon_thread.join()
            self.__q.close()
            self.__q.unlink()
            self.__pending_tasks.clear()
            self.__completed_tasks.clear()
            # ______________________________________
            self.__shared_memory.close()
            unregister(self.__shared_memory._name,"shared_memory")
        except Exception as e:
            print("SHUTDOWN_ERROR {}".format(e))
            
            # print("Error closing and unlinking the queue: {}".format(e))
        

    def __run(self):
        while self.__is_running:
            T.sleep(2)
            qsize = self.__q.qsize()
            if qsize == 0:
                continue
            response:Dict[str,Any] = self.__q.get_nowait()
            # self.__log.debug()
            status         = int(response.get("status","-1"))
            if status < 0 :
                self.__log.error(response)
                continue
            task_id        = response.get("task_id","")
            if task_id == "":
                self.__log.error(response)
                continue
            if task_id in self.__pending_tasks:
                pending_task = self.__pending_tasks[task_id]
                client_id      = response.get("client_id","")
                key            = response.get("key","")
                operation_type = response.get("operation_type","")
                checksum       = response.get("checksum","")
                offset         = int(response.get("offset","0"))
                size           = int(response.get("size","-1"))
                service_time   = int(response.get("service_time","-1"))
                if operation_type == "GET":
                    task               = self.__pending_tasks.pop(task_id)
                    task.response_time = T.time() - task.start_time
                    task.event         = response
                    data               = self.__shared_memory.buf[offset:offset+size]
                    task.data=Some(data)
                    # local_checksum     = XoloUtils.sha256(data)
                    # print("LOCAL_CHECKSUM",local_checksum)
                    # print("CHECKSUM",checksum)
                    self.__completed_tasks[task_id] = task
                    self.__log.info(response)
                elif operation_type =="PUT":
                    task               = self.__pending_tasks.pop(task_id)
                    task.response_time = T.time() - task.start_time
                    task.event         = response
                    self.__completed_tasks[task_id] = task
                    self.__log.info(response)
                else:
                    pass
            print("_"*50)
    

# if __name__ == "__main__":
#     c = Client(client_id="mictlanx")
#     x = input("Type something to shutdown the client")
#     c.shutdown()
    