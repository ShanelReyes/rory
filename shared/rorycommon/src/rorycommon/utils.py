import numpy as np
from typing import Iterator,Optional
from mictlanx.utils.segmentation import Chunk
from option import Some
import hashlib as H
# import zarr 

class Utils:
    @staticmethod
    def read_chunks_numpy(filename: str,ball_id:str, row_chunk:int =10, max_columns:Optional[int]=None) -> Iterator[Chunk]:
        """
        Reads a .npy file chunk by chunk, with optional max number of columns.
        
        Parameters:
        - filename: path to .npy file
        - row_chunk: number of rows per chunk
        - max_columns: optional max number of columns per chunk (e.g. 100)
        
        Yields:
        - 2D NumPy arrays of shape (<=row_chunk, <=max_columns)
        """
        mmap_array = np.load(filename, mmap_mode='r')
        total_rows, total_cols = mmap_array.shape
        num_chunks = 0
        # (total_rows + row_chunk - 1) // row_chunk

        cols_to_read = min(total_cols, max_columns) if max_columns else total_cols
        h = H.sha256()
        for i in range(0, total_rows, row_chunk):
            chunk_data = mmap_array[i:i + row_chunk, 0:cols_to_read]
            h.update(chunk_data.tobytes())
            num_chunks +=1
            
        full_checksum = h.hexdigest()

        chunk_index =0
        for i in range(0, total_rows, row_chunk):
            x = mmap_array[i:i + row_chunk, 0:cols_to_read]
            
            chunk_id = f"{ball_id}_{chunk_index}"
            c = Chunk.from_ndarray(
                ndarray  = x,
                group_id = ball_id,
                index    = chunk_index,
                chunk_id = Some(chunk_id),
                metadata={
                    "full_shape":str((total_rows,total_cols)), 
                    "full_checksum":full_checksum,
                    "num_chunks":str(num_chunks)
                }
            )
            yield c
            chunk_index+=1