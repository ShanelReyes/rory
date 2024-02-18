from __future__ import annotations
from option import Option,NONE,Some
from typing import Dict,Iterator, List,Any,Callable,Tuple
from mictlanx.v3.interfaces.core import Metadata
import numpy as np
import numpy.typing as npt
import hashlib as H
import os


#
class Chunk(object):
    def __init__(self,group_id:str,index:int,data:bytes,chunk_id:Option[str]=NONE,metadata:Dict[str,str]={}):
        self.group_id = group_id
        self.index    = index
        self.size     = len(data)
        self.data     = data
        self.metadata = metadata
        hasher = H.sha256()
        hasher.update(self.data)
        self.checksum = hasher.hexdigest()
        self.chunk_id = chunk_id.unwrap_or(self.checksum)
    def __str__(self):
        return "Chunk(group_id={}, index={}, size={})".format(self.group_id,self.index,self.size)
    def from_ndarray(group_id:str,index:int,ndarray:npt.NDArray, metadata:Dict[str,str]={}, chunk_id:Option[str]=NONE):
        metadata["shape"] = str(ndarray.shape)
        metadata["attributes"] = str(ndarray.shape[1])
        metadata["records"] = str(ndarray.shape[0])
        metadata["dtype"] = str(ndarray.dtype)
        return Chunk(group_id=group_id,index= index, data = ndarray.tobytes(order="C"), metadata=metadata,chunk_id=chunk_id )
    def to_ndarray(self)->Option[npt.NDArray]:
        try:
            shape   = eval(self.metadata.get("shape"))
            dtype   = self.metadata.get("dtype","float64")
            ndarray = np.frombuffer(self.data,dtype=dtype).reshape(shape)
            # print(ndarray)
            return Some(ndarray)
        except Exception as e:
            return NONE


class Chunks(object):
    def __init__(self,chs:Iterator[Chunk],n:int ):
        self.chunks:List[Chunk] = list(chs)
        self.n:int = n 
    # def to_list(self)->List[Chunk]:
        # return list(self.chunks)
    def len(self)->int:
        return self.n
    def iter(self):
        return self.chunks
    
    def sorted_by(self,filter_by:Callable[[Chunk], Any] = lambda x:x.index)->Iterator[Chunk]:
        return sorted(self.chunks, key= filter_by)
    
    def _iter_to_chunks(group_id:str,iterable:Any,n:int,chunk_prefix:Option[str]=NONE,chunk_size:Option[int]=NONE,num_chunks:int =1):
        # THE RATIO OF RECORDS PER CHUNK (float)
        data_per_chunk     = chunk_size.unwrap_or(n / num_chunks)
        # Check if the data per chunk is lower or equal to the number of total elements. 
        assert(data_per_chunk <= n)
        # data per chunk but int
        data_per_chunk_int = int(data_per_chunk)
        # Total number of chunked elements (chunked = element that belongs to a specific chunk)
        total_chunked_elements = 0
        # Current chunk index
        i                      = 0 
        # Check that total number of chunked elements is lower than the total number of elements
        chunks = []
        while total_chunked_elements < n:
            # Difference between total number of elements and total chunked elements
            diff = n - total_chunked_elements
            # Chunk metadata 
            metadata = {"index": str(i)}
            # Check if diff is lower than -> if it is lower then drain all the iterable. 
            if diff < data_per_chunk:
                current_total_records_sent = data_per_chunk_int*i
                total_chunked_elements     += n - current_total_records_sent
                records_chunk              = iterable[current_total_records_sent:]
                chunk_metadata             = chunks[-1]
                if type(records_chunk) == np.ndarray:
                    chunk_metadata["data"] = np.concatenate([chunk_metadata["data"], records_chunk])
                else:
                    chunk_metadata["data"] = chunk_metadata["data"]+records_chunk
                if chunk_prefix.is_some:
                    chunk_metadata["chunk_id"] ="{}_{}".format(chunk_prefix.unwrap(),i-1)
            else:
                total_chunked_elements += data_per_chunk_int
                from_index             = i*data_per_chunk_int
                to_index               = ((i+1)*data_per_chunk_int)
                records_chunk          = iterable[from_index: to_index]
                chunk_metadata = {'group_id':group_id, 'index':i, 'data':records_chunk, 'metadata':metadata}
                if chunk_prefix.is_some:
                    chunk_metadata["chunk_id"] ="{}_{}".format(chunk_prefix.unwrap(),i)
                i+=1
                chunks.append(chunk_metadata)
        return chunks
   
    def iter_to_chunks(group_id:str,iterable:Any,n:int,chunk_prefix:Option[str]=NONE,chunk_size:Option[int]=NONE,num_chunks:int =1):
        # hashing
        # print("ITERABLE_TYPE",type(iterable))
        # hasher = H.sha256()
        # THE RATIO OF RECORDS PER CHUNK (float)
        data_per_chunk     = chunk_size.unwrap_or(n / num_chunks)
        # Check if the data per chunk is lower or equal to the number of total elements. 
        assert(data_per_chunk <= n)
        # data per chunk but int
        data_per_chunk_int = int(data_per_chunk)
        # Total number of chunked elements (chunked = element that belongs to a specific chunk)
        total_chunked_elements = 0
        # Current chunk index
        i                      = 0 
        # Check that total number of chunked elements is lower than the total number of elements
        while total_chunked_elements < n:
            # Difference between total number of elements and total chunked elements
            diff = n - total_chunked_elements
            # Chunk metadata 
            metadata = {"index": str(i)}
            # Check if diff is lower than -> if it is lower then drain all the iterable. 
            if diff < data_per_chunk:
                current_total_records_sent = data_per_chunk_int*i
                total_chunked_elements     += n - current_total_records_sent
                records_chunk              = iterable[current_total_records_sent:]
            else:
                total_chunked_elements += data_per_chunk_int
                from_index             = i*data_per_chunk_int
                to_index               = ((i+1)*data_per_chunk_int)
                records_chunk          = iterable[from_index: to_index]
            # hasher.update(records_chunk)
            chunk_metadata = {'group_id':group_id, 'index':i, 'data':records_chunk, 'metadata':metadata}
            if chunk_prefix.is_some:
                chunk_metadata["chunk_id"] ="{}_{}".format(chunk_prefix.unwrap(),i)

            i+=1
            yield chunk_metadata
        # return Chunks(chs=__inner(),n = n)    

            # The fractional part of the records per worker.

            # records_fraction = records_per_worker - records_per_worker_int 
            # sum_records_per_worker  = records_per_worker * workers 
            # assert(sum_records_per_worker<= records_len)
    

    def from_ndarray(ndarray:npt.NDArray, group_id:str,chunk_prefix:Option[str]=NONE,chunk_size:Option[int] = NONE,num_chunks:int = 1 )->Option[Chunks]:
        
        try:
            def __inner():
                xs= Chunks._iter_to_chunks(iterable=ndarray, group_id = group_id,n = ndarray.shape[0],num_chunks=num_chunks,chunk_size=chunk_size,chunk_prefix=chunk_prefix)
                for i,x in enumerate(xs):
                    chunk_id       = Some(x.get("chunk_id",None)).filter(lambda x: not x == None)
                    chunk          = Chunk.from_ndarray(group_id = group_id, index = x["index"], ndarray=x["data"],metadata = x['metadata'],chunk_id=chunk_id)
                    # chunk.chunk_id = x.get("chunk_id",chunk.chunk_id)
                    # chunk.chunk_id = chunk_prefix.map(lambda x: "{}_{}".format(x,chunk.index)).unwrap_or(chunk.chunk_id)
                    yield chunk
            return Some(Chunks(chs= __inner() , n = ndarray.shape[0]))
        except Exception as e:
            return NONE

    def from_file(path:str,group_id:str,chunk_size:Option[int] = NONE,num_chunks:int =1)->Option[Chunks]:
        try:
            n:int              = os.path.getsize(path)
            def __inner():
                with open(path,"rb") as f:
                    records_per_worker = chunk_size.unwrap_or(n / num_chunks)
                    assert(records_per_worker <= n)
                    records_per_worker_int = int(records_per_worker)
                    # x = n / records_per_worker_int
                    # print(x-int(x) > 0 , int(x) -1  )
                    # print(num_chunks)
                    i=0
                    while True:
                        # metadata = {"index":str(i)}
                        metadata = {}
                        if i >= num_chunks-1:
                            data = f.read()
                            if not data:
                                break
                            chunk_metadata = {'group_id':group_id, 'index':i, 'data':data, 'metadata':metadata}
                            chunk = Chunk(**chunk_metadata)
                            # i+=1
                            yield chunk
                        else:
                            data = f.read(records_per_worker_int)
                            if not data:
                                break
                            chunk_metadata = {'group_id':group_id, 'index':i, 'data':data, 'metadata':metadata}
                            chunk = Chunk(**chunk_metadata)
                            i+=1
                            yield chunk
                        # Chunk(group_id=group_id,index=i, )

            return Some(Chunks(chs=__inner(), n = n))


                    # data = f.read()
                    # records_len = 
                    # THE RATIO OF RECORDS PER WORKER (float)

        except Exception as e:
            return NONE

    def from_bytes(data:bytes,group_id:str,chunk_size:Option[int] = NONE,num_chunks:int =1)->Option[Chunks]:
        def __inner():
            xs = Chunks._iter_to_chunks(iterable = data,group_id = group_id,num_chunks = num_chunks,n=len(data),chunk_size=chunk_size) 
            for x in xs:
                chunk = Chunk(group_id = group_id,data=x["data"],index=x["index"], metadata = x["metadata"])
                yield chunk
        return Some(Chunks(chs = __inner(), n = len(data)))
        

    # GET ndarray and metadata
    def to_ndarray(self)->Option[Tuple[npt.NDArray,Metadata]]:
        try:
            result   = []
            metadata = Metadata(id="ID", size=0, checksum="",group_id="",tags={})
            hasher   = H.sha256()
            size     = 0
            for chunk in self.sorted_by(filter_by=lambda chunk:chunk.index):
                if not ("shape" in chunk.metadata or "dtype" in chunk.metadata):
                    return NONE
                dtype   = chunk.metadata.get("dtype","float64")
                shape   = eval(chunk.metadata.get("shape"))
                hasher.update(chunk.data)
                size    += len(chunk.data)
                ndarray = np.frombuffer(chunk.data,dtype= dtype).reshape(shape)
                result.append(ndarray)
            metadata.size     = size
            metadata.checksum = hasher.hexdigest()
            result = np.vstack(result)
            return Some((result,metadata))
        except Exception as e:
            print(e)
            return NONE
        
if __name__ == "__main__":
    pass
    # chs = [ 
        # Chunk.from_ndarray(ndarray=np.ones((5,5)), group_id="encrypted_matrix-0",index=0  ),
        # Chunk.from_ndarray(ndarray=np.ones((5,5)),group_id="encrypted_matrix-0" ,index=1 )
    # ]
    # cs = Chunks(chs=chs,n=2)
    # __________________

    # encrypted_matrix0 = np.random.random(size=(1000,10,3))

        # chunk.
    # cs1               = Chunks.from_ndarray(ndarray=encrypted_matrix0 ,num_chunks=3,group_id="encrypted_matrix-0",chunk_prefix=Some("chunk") ).unwrap()
    # for chunk in cs1.chunks:
        # print(chunk.chunk_id,chunk.checksum,chunk.metadata)
    # print(cs1.to_ndarray().unwrap()[0].shape)
    # encrypted_matrix0 = cs1.to_ndarray()
    # print("MATRIX",encrypted_matrix0)
    

    # cs = Chunks.from_bytes(group_id = "key",data=b"12345678900", num_chunks= 2).unwrap() 
    # cs = Chunks.from_file(path="/source/01.pdf",group_id="test",num_chunks=4).unwrap()
    # xs = cs.to_list()
    # h = H.sha256()
    # for c in cs.iter():
        # h.update(c.data)
        # print(c)
    # x = h.hexdigest()
    # print(x)
        # print(c.data)
        # print(x.checksum,x.size)
    # print(xs)
    # cs = Chunks.iter_to_chunks(
    #         group_id="test",
    #         iterable= [0,1,2,3,4,5,6,7],
    #         chunk_size= Some(2),
    #         workers=3, 
    #         n = 8
    #     )
    # for c in cs :
    #     print(c)
    # NDARRAY
    # ndarray = np.ones((100,500))
    # cs_ndarray = Chunks.from_ndarray(ndarray=ndarray,group_id="test",workers=5,chunk_size=Some(50))
    # x = cs_ndarray.sorted_by(filter_by=lambda chunk:chunk.index)
    # for chunk in cs_ndarray.iter():
        # print(str(chunk))
    # print(cs_ndarray.to_ndarray())

    # print(cs.)

