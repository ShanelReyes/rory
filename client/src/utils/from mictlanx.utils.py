from mictlanx.utils.segmentation import Chunks,Chunk

def chunks_to_bytes_gen(chs:Chunks): #-> Generator[bytes,None,None]:
    #iterarlos con su .iter
    for chunk in chs.iter():
        #convertirlos a bytes
        bytes_chunk = bytes(chunk)
        print(bytes_chunk)
    #Crear un generator de bytes


    