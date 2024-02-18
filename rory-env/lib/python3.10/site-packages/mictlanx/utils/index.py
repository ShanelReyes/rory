from typing import List
from mictlanx.v4.interfaces.index import Peer
from typing import Iterator,Tuple,Dict
import hashlib as H
import humanfriendly as HF
from concurrent.futures import ThreadPoolExecutor,as_completed
from typing import Generator
import os
from mictlanx.v4.xolo.utils import Utils as XoloUtils
from collections import namedtuple
from pathlib import Path

# 
FileInfoBase = namedtuple("FileInfo","path checksum size")
class FileInfo(FileInfoBase):
    def upadate_path_relative_to(self,relative_to:str):
        _relative_to = Path(relative_to)
        path = Path(self.path).relative_to(_relative_to)
        return FileInfoBase(str(path),self.checksum,self.size)

class Utils(object):
    # @staticmethod
    # def extract_checksum_from_tags(tags:Dict[str,str])
    # 
    # def __sha
    @staticmethod
    def get_checksums_and_sizes(path:str,max_workers:int = 2)->Generator[FileInfo,None,None]:
        futures = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as tp:
            for (root,_, fullnames) in os.walk(path):
                for fullname in fullnames:
                    file_path = "{}/{}".format(root,fullname)
                    fut = tp.submit(XoloUtils.extract_path_sha256_size, path = file_path )
                    futures.append(fut)
            for future in as_completed(futures):
                result =FileInfo(*future.result())
                yield result



    @staticmethod
    def file_to_chunks_gen(path:str, chunk_size:str="1MB"):
        _chunk_size = HF.parse_size(chunk_size)
        with open(path,"rb") as f:
            while True:
                value:bytes                     = f.read(_chunk_size)
                if not value:
                    break
                yield value
                
    @staticmethod
    def peers_from_str(peers_str:str,separator:str=" ")->Iterator[Peer]:
        splitted = peers_str.split(separator)
        splitted = map(lambda x: x.split(":"), splitted)
        return map(lambda x: Peer(peer_id=x[0],ip_addr=x[1], port=int(x[2])), splitted) 

    # mictlanx-peer-0:alpha.tamps.cinvestav.mx/v0/mictlanx:-1
    @staticmethod
    def peers_from_str_v2(peers_str:str,separator:str=" ",protocol:str="http")->Iterator[Peer]:
        splitted = peers_str.split(separator)
        splitted = map(lambda x: x.split(":"), splitted)
        def __inner(x:List[str]):
            return  Peer(peer_id=x[0],ip_addr=x[1], port=int(x[2]) ,protocol=protocol)
        return map(__inner, splitted) 
    
    @staticmethod
    def calculate_disk_uf(total:int,used:int,size:int = 0 ):
        return  1 - ((total - (used + size))/total)


if __name__ =="__main__":
    xs = map(lambda x: x. upadate_path_relative_to(relative_to="/sink/client1/bucket1"),Utils.get_checksums_and_sizes(path="/sink/client1"))
    for (path,checksum,size) in xs :
        print(path,checksum,size)
    # peers_strs = "mictlanx-peer-0:alpha.tamps.cinvestav.mx/v0/mictlanx/peer0:-1 mictlanx-peer-1:alpha.tamps.cinvestav.mx/v0/mictlanx/peer1:-1"
    # peers = Utils.peers_from_str_v2(peers_str=peers_strs,protocol="https")
    # for p in peers:
    #     print(p.base_url())