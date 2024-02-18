import sys
import os
import json as J
import time as T
from ipcqueue import posixmq
from threading import Thread,Event
from typing import List,Dict,Tuple,Awaitable
from multiprocessing.shared_memory import SharedMemory
from mictlanx.v4.interfaces.index import Ball
from mictlanx.logger.log import Log
from mictlanx.v4.xolo.utils import Utils as XoloUtils
from mictlanx.utils.index import Utils
from mictlanx.v4.interfaces.index import Peer
from mictlanx.v4.client import Client
from mictlanx.v4.interfaces.responses import GetMetadataResponse,PutResponse
from option import Result
# import magic as M
import hashlib as H
from humanfriendly import parse_size,format_size,parse_timespan
import logging
from dataclasses import dataclass
from concurrent.futures import as_completed


@dataclass
class DistributionSchemaItem:
    offset:int
    size:int
    checksum:str

class Server(object):
    def __init__(
            self,
            bucket_id:str,          
            max_memory:str         = "512MB",
            chunk_size:str         = "1MB",
            to_disk_interval:str   = "10s",
            to_cloud_interval:str  ="10m",
            heartbeat_interval:str = "1s",
            base_path:str          = "/mictlanx/.server",
            block:bool            = False,
            # Client
            peers:List[Peer]       = [],
            debug:bool             = False,
            show_metrics:bool      = False,
            client_daemon:bool     = True,
            client_heartbeat_interval:str = "5s",
            max_workers:int        = 6,
            lb_algorithm:str       = "2CHOICES_UF",
            output_path:str        = "/mictlanx/.client",
            max_put_timeout:str    = "2m"
        ):
        self.__max_workers         = max_workers
        self.__base_path           = base_path
        self.__done_event          = Event()
        self.__queue_id            = "mictlanx"
        self.__q                   = posixmq.Queue(
            name= '/{}'.format(self.__queue_id),
            maxsize=10, 
            maxmsgsize= 1024
        )
        self.__client_queues:Dict[str, posixmq.Queue] = {}
        self.__total_memory                           = parse_size(max_memory)
        self.__shared_memory                          = SharedMemory(
            name   = "/mictlanx-shm",
            create = True, 
            size   = self.__total_memory 
        )
        self.__distribution_schema:Dict[str,DistributionSchemaItem] = {}
        self.__chunk_size      = parse_size(chunk_size)
        self.__offset          = 0 
        self.__log             = Log(
            name = "mictlanx-server-0",
            path = "/mictlanx/.server",
            console_handler_filter= lambda record: record.levelno == logging.DEBUG or record.levelno == logging.INFO or record.levelno == logging.ERROR
        )
        self.__data_path = "{}/data".format(self.__base_path)
        if not os.path.exists(self.__data_path):
            os.makedirs(self.__data_path)

        self.balls:List[Ball] = []
        self.__heartbeat_interval = parse_timespan(heartbeat_interval)
        # ________________________
        self.__used_memory       = 0
        self.__is_running        = True
        self.__to_disk_interval  = int(parse_timespan(to_disk_interval)/self.__heartbeat_interval)
        self.__to_cloud_interval = int(parse_timespan(to_cloud_interval)/self.__heartbeat_interval)
        self.__bucket_id         = bucket_id
        self.__max_put_timeout   = parse_timespan(max_put_timeout)
        
        self.client =  Client(
            client_id           = "mictlanx-client",
            peers               = peers,
            debug               = debug,
            show_metrics        = show_metrics,
            daemon              = client_daemon,
            max_workers         = max_workers,
            lb_algorithm        = lb_algorithm,
            output_path         = output_path,
            heartbeat_interval  = client_heartbeat_interval,
            metrics_buffer_size = 100 
        )
        # This block 
        self.__daemon_thread  = Thread(target=self.__run,name="mictlanx",daemon=block)
    
    
    def start(self):
        self.__log.debug({
            "event":"STARTING",
            "id":self.__queue_id,
            "chunk_size":format_size(self.__chunk_size),
            "total_memory":format_size(self.__total_memory)
        })
        self.__daemon_thread.start()
        self.__done_event.wait()

    
    def __to_disk(
            self,
            path:str,
            data:memoryview,
            local_checksum:str,
            content_type:str,
            ds:DistributionSchemaItem
    ):
        start_time = T.time()
        with open(path,"wb") as f:
            f.write(data)
        service_time = T.time() - start_time
        self.__log.info({
            "arrival_time":start_time,
            "path": path,
            "operation_type":"FLUSH",
            "checksum":local_checksum,
            "offset":ds.offset,
            "size":ds.size,
            "status":0,
            "content_type":content_type,
            "service_time":service_time
        })
    def __check_to_disk(self,heartbeats:int):
        start_time = T.time()
        if heartbeats % self.__to_disk_interval == 0:
            for key, ds in self.__distribution_schema.items():
                path           = "{}/{}".format(self.__data_path, key)
                data           = self.__shared_memory.buf[ds.offset:ds.offset+ds.size]
                local_checksum = XoloUtils.sha256(value=data)
                content_type = "application/octet-stream"
                # content_type   = str(M.from_buffer(buffer=bytes(data[:1024]),mime=True))
                # print(local_checksum,)
                if os.path.exists(path):
                    file_checksum,size = XoloUtils.sha256_file(path=path)
                    if local_checksum == file_checksum:
                        service_time = T.time() - start_time
                        self.__log.debug({
                            "arrival_time":start_time,
                            "path": path,
                            "operation_type":"NO_FLUSH",
                            "checksum":local_checksum,
                            "offset":ds.offset,
                            "size":ds.size,
                            "status":0,
                            "content_type":content_type,
                            "service_time":service_time,
                            "description":"No changes detected in data."
                        })
                    else:
                        os.remove(path=path)
                        self.__to_disk(path=path, data=data,local_checksum=local_checksum, content_type=content_type, ds=ds)
                else:
                    self.__to_disk(path=path, data=data,local_checksum=local_checksum, content_type=content_type, ds=ds)
                
    def __to_cloud(self,key:str, data:memoryview, path:str, ds:DistributionSchemaItem, content_type: str, file_exists:bool,start_time:int = 0):
        put_result:Result[PutResponse,Exception] = self.client.put(value= bytes(data),key=key,bucket_id=self.__bucket_id,timeout=self.__max_put_timeout).result()
        if put_result.is_ok:
            put_response:PutResponse = put_result.unwrap()
            service_time = T.time() - start_time    
            self.__log.info({
                "key":key,
                "arrival_time":start_time,
                "path": path,
                "operation_type":"PUSH",
                "offset":ds.offset,
                "size":ds.size,
                "status":0,
                "content_type":content_type,
                # "step":1,
                "service_time":service_time,
                "peer_id": put_response.node_id,
                "ondisk":file_exists,
            })
    def __check_to_cloud(self,heartbeats:int=0):
        if heartbeats % self.__to_cloud_interval == 0:
            task_counter = 0
            max_workers_buffer = int(self.__max_workers / 2 )
            metadatas_buffer:List[Awaitable[Result[GetMetadataResponse,Exception]]] = []
            n = len(self.__distribution_schema.keys())
            for  key,ds in self.__distribution_schema.items():
                start_time = T.time()
                fut = self.client.get_metadata(key=key)
                metadatas_buffer.append(fut)
                
                if task_counter % max_workers_buffer == 0 or (max_workers_buffer >= n  and (task_counter+1) >= n   ):
                    for metadata_fut in as_completed(metadatas_buffer):
                        result:Result[GetMetadataResponse,Exception] = metadata_fut.result()
                        path           = "{}/{}".format(self.__data_path, key)
                        file_exists    = os.path.exists(path=path)
                        data           = self.__shared_memory.buf[ds.offset:ds.offset+ds.size]
                        local_checksum = XoloUtils.sha256(value=data)
                        content_type   = "application/octet-stream"
                
                        # str(M.from_buffer(buffer=bytes(data[:1024]),mime=True))
                        if result.is_ok:
                            response:GetMetadataResponse = result.unwrap()
                            if response.metadata.checksum == local_checksum :
                                service_time = T.time() - start_time
                                self.__log.debug({
                                    "arrival_time":start_time,
                                    "path": path,
                                    "operation_type":"NO_PUSH",
                                    "checksum":local_checksum,
                                    "offset":ds.offset,
                                    "size":ds.size,
                                    "status":0,
                                    "content_type":content_type,
                                    "service_time":service_time,
                                    "description":"No changes detected in data."
                                })
                            else:
                                self.__to_cloud(
                                    key=key,
                                    start_time=start_time,
                                    data=data,
                                    path=path,
                                    ds=ds,
                                    content_type=content_type,
                                    file_exists=file_exists,
                                )
                        else:
                            self.__to_cloud(
                                key=key,
                                start_time=start_time,
                                data=data,
                                path=path,
                                ds=ds,
                                content_type=content_type,
                                file_exists=file_exists,
                            )
                    metadatas_buffer.clear()
                task_counter+=1
                

            # put_futures:List[Awaitable[Result[PutResponse,Exception]]] = []
            # for key,(fut, ds) in futures.items():
            #     # result:Result[GetMetadataResponse,Exception] = fut.result()
            #     # _________________________________________________________
            #     if result.is_ok:
            #         response = result.unwrap()
            #         print(key,"EXISTS_IN_CLOUD",response.metadata)
            #     else:
            #         put_fut = self.client.put(value= bytes(data),key=key,bucket_id=self.__bucket_id,timeout=self.__max_put_timeout)
            #         put_futures.append(put_fut)
                    
            #         self.__log.info({
            #             "key":key,
            #             "arrival_time":start_time,
            #             "path": path,
            #             "operation_type":"PUSH",
            #             "offset":ds.offset,
            #             "size":ds.size,
            #             "status":0,
            #             "content_type":content_type,
            #             "step":0,
            #             "ondisk":file_exists,
            #         })
            # for put_fut in as_completed(put_futures):
            #     put_result:Result[PutResponse,Exception] = put_fut.result()
            #     print("PUT_RESULT>>>>>>>>>>>",put_result)

    def __run(self):
        try:
            heartbeats = 0
            while self.__is_running:
                T.sleep(self.__heartbeat_interval)

                qsize = self.__q.qsize()
                start_time = T.time()
                # ______________________________________________________________________
                self.__check_to_disk(heartbeats=heartbeats)
                self.__check_to_cloud(heartbeats=heartbeats)
                # if (heartbeats % self.__to_cloud_interval == 0):
                #     print("SEND TO CLOUD")
                # ______________________________________________________________________
                if qsize == 0:
                    heartbeats+=1
                    self.__log.debug({
                        "arrival_time":start_time,
                        "event":"HEALTH",
                        "used_memory": self.__used_memory,
                        "available_memory": self.__total_memory - self.__used_memory,
                        "total_memory": self.__total_memory,
                        "memory_uf":Utils.calculate_disk_uf(total=self.__total_memory,used= self.__used_memory, size=0),
                        "offset":self.__offset
                    })
                    continue
                request         = self.__q.get_nowait()
                client_id       = request.get("client_id","")
                key             = request.get("key","")
                path            = request.get("path","")
                operation_type  = str(request.get("operation_type","")).upper()
                task_id         = request.get("task_id","")
                
                # Write in shared memory
                if  operation_type =="" or task_id =="" or client_id == "":
                    service_time = T.time() - start_time
                    event = {
                        "client_id":client_id,
                        "task_id":task_id,
                        "key":key,
                        "path": path,
                        "operation_type":operation_type,
                        "status":-1,
                        "msg": "Bad params: ".format(path),
                        "service_time":service_time
                    }
                    self.__log.error(event)
                    heartbeats+=1
                    continue
                else:
                    # Check if the queue is already in the local queues
                    if not client_id in self.__client_queues:
                        self.__client_queues[client_id] = posixmq.Queue(name="/{}".format(client_id))
                        client_queue = self.__client_queues[client_id]
                    else:
                        client_queue = self.__client_queues[client_id]


                    if operation_type == "PUT":
                        if not os.path.exists(path=path):
                            service_time = T.time() - start_time
                            event = {
                                "arrival_time":start_time,
                                "client_id":client_id,
                                "task_id":task_id,
                                "key":key,
                                "path": path,
                                "operation_type":operation_type,
                                "status":-1,
                                "msg": "{} does not exists in the host filesystem".format(path),
                                "service_time":service_time
                            }
                            self.__log.error(event)
                            client_queue.put_nowait(event)
                            heartbeats+=1
                            continue
                        checksum,size   = XoloUtils.sha256_file(path=path)
                        # Check if exists
                        if checksum in self.__distribution_schema or key in self.__distribution_schema:
                            service_time = T.time() - start_time
                            event = {
                                "arrival_time":start_time,
                                "client_id":client_id,
                                "task_id":task_id,
                                "key":key,
                                "path": path,
                                "operation_type":operation_type,
                                "status":-1,
                                "msg": "{}/{} already exists in memory".format(key,checksum),
                                "service_time":service_time
                            }
                            self.__log.error(event)
                            client_queue.put_nowait(event)
                            heartbeats+=1
                            continue
                        _key = checksum if key=="" else key
                        if self.__offset + size  <=self.__total_memory:
                            hasher = H.sha256()
                            with open(path, "rb") as f:
                                current_read_bytes = 0
                                left_bytes         = size - current_read_bytes
                                i                  = 0 

                                # self.__offset      += len(chunk_size)
                                
                                
                                while current_read_bytes < size:
                                    chunk_start_time = T.time()
                                    if left_bytes < self.__chunk_size:
                                        self.__log.debug({"msg":"{} is too small to create a chunk of size {}".format(left_bytes,format_size(self.__chunk_size))})
                                        chunk          = f.read()
                                    else:
                                        chunk = f.read(self.__chunk_size)
                                        
                                    hasher.update(chunk)
                                    chunk_size     = len(chunk)
                                    chunk_checksum = XoloUtils.sha256(value=chunk)
                                    current_read_bytes += chunk_size
                                    # self.__shared_memory.buf[(self.__offset + chunk_size*i):self.__offset + current_read_bytes] = chunk
                                    self.__shared_memory.buf[self.__offset: self.__offset + chunk_size] = chunk
                                    service_time = T.time()- start_time
                                    self.__log.debug({
                                        "arrival_time":start_time,
                                        "operation_type":"WRITE_CHUNK",
                                        "start_offset":self.__offset,
                                        "end_offset": self.__offset+chunk_size,
                                        "checksum":chunk_checksum,
                                        "chunk_size":chunk_size,
                                        "current_read_byte":current_read_bytes,
                                        "service_time":service_time
                                    })
                                    
                                    self.__offset      += chunk_size
                                    self.__used_memory += chunk_size
                                    i+=1
                                    # T.sleep(10)
                            local_checksum = hasher.hexdigest()
                            if not local_checksum == checksum:
                                service_time = T.time() - start_time
                                event = {
                                    "arrival_time":start_time,
                                    "client_id":client_id,
                                    "task_id":task_id,
                                    "key":key,
                                    "path": path,
                                    "operation_type":operation_type,
                                    "status":-1,
                                    "checksum":checksum,
                                    "_checkcsum":local_checksum,
                                    "msg":"Integrity issue: {} != {}".format(checksum,local_checksum),
                                    "service_time":service_time
                                }
                                self.__log.error(event)
                                client_queue.put_nowait(event)
                                heartbeats+=1
                                continue
                            

                            self.__distribution_schema[_key] = DistributionSchemaItem(offset=self.__offset-size, size=size, checksum=checksum)
                            # COMPLETED
                            
                            service_time = T.time() - start_time
                            event = {
                                "arrival_time":start_time,
                                "client_id":client_id,
                                "task_id":task_id,
                                "key":_key,
                                "path": path,
                                "operation_type":operation_type,
                                "checksum":checksum,
                                "offset":self.__offset-size,
                                "size":size,
                                "status":0,
                                "service_time":service_time
                            }
                            self.__log.info(event)
                            client_queue.put_nowait(event)
                            heartbeats+=1
                        else:
                            service_time = T.time() - start_time
                            event = {
                                "arrival_time":start_time,
                                "client_id":client_id,
                                "task_id":task_id,
                                "key":_key,
                                "path": path,
                                "operation_type":operation_type,
                                "status":-1,
                                "msg": "Run out of memory".format(path),
                                "service_time":service_time
                            }
                            self.__log.error(event)
                            client_queue.put_nowait(event)
                            heartbeats+=1
                            continue
                    elif operation_type =="GET":
                        
                        if key in self.__distribution_schema:
                            dsi = self.__distribution_schema[key]
                            print(dsi)
                            service_time = T.time() - start_time
                            event = {
                                "arrival_time":start_time,
                                "client_id":client_id,
                                "task_id":task_id,
                                "key": key,
                                "path": path,
                                "operation_type":operation_type,
                                "service_time":service_time,
                                "offset":dsi.offset,
                                "size":dsi.size,
                                "status":0,
                                "checksum":dsi.checksum
                            }
                            self.__log.info(event)
                            client_queue.put_nowait(event)
                        else:
                            event = {
                                "arrival_time":start_time,
                                "client_id":client_id,
                                "task_id":task_id,
                                "key":_key,
                                "path": path,
                                "operation_type":operation_type,
                                "status":-1,
                                "msg":"{} not found".format(key)
                            }
                            self.__log.error(event)
                        heartbeats+=1
                    else:
                        pass
            # heartbeats+=1
        except Exception as e:
            print(e)
        finally:
            self.__done_event.set()
            # self.__done_event.set()

    def shutdown(self):
        try:
            self.__log.debug({
                "event":"SHUTDOWN",
                "id":self.__queue_id,
                "msg":"Cleaning all resources",
            })
            self.__is_running = False
            self.__daemon_thread.join()
            self.__q.unlink()
            self.__q.close()
            self.__shared_memory.close()
            self.__shared_memory.unlink()
        except Exception as e:
            print("Error closing or unlinking the queue / shared memory: {}".format(self.__queue_id))
