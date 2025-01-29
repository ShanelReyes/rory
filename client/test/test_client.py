# * This block of code MUST be executed first.  
# _______________________________________________________
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import unittest
import os
from flask import Blueprint,current_app,request,Response
from rory.core.security.pqc.dataowner import DataOwner as DataOwnerPQC
from rory.core.security.cryptosystem.pqc.ckks import Ckks
from mictlanx.utils.segmentation import Chunks,Chunk
from concurrent.futures import ProcessPoolExecutor
from typing import List,Awaitable,Generator
import numpy.typing as npt
from option import Some
import pickle
from mictlanx.v4.client import Client
from mictlanx.utils.index import Utils as MictlanXUtiks

# from ror
from Pyfhel import PyCtxt,Pyfhel

ckks = Ckks.from_pyfhel(
    _round   = False,
    decimals = 2,
    path               = os.environ.get("KEYS_PATH","/rory/keys"),
    ctx_filename       = os.environ.get("CTX_FILENAME","ctx"),
    pubkey_filename    = os.environ.get("PUBKEY_FILENAME","pubkey"),
    relinkey_filename  = os.environ.get("RELINKEY_FILENAME","relinkey"),
    rotatekey_filename = os.environ.get("ROTATE_KEY_FILENAME","rotatekey"),
    secretkey_filename = os.environ.get("SECRET_KEY_FILENAME","secretkey"),
)

dataowner   = DataOwnerPQC(scheme = ckks)
max_workers = 2
ppe         = ProcessPoolExecutor(max_workers=max_workers)
num_chunks  = max_workers
np_random   = True
key         = "test_chunks"

MICTLANX_CLIENT_ID           = os.environ.get("MICTLANX_CLIENT_ID","{}_mictlanx".format("client"))
MICTLANX_TIMEOUT             = int(os.environ.get("MICTLANX_TIMEOUT",3600))
MICTLANX_API_VERSION         = int(os.environ.get("MICTLANX_API_VERSION","3"))
MICTLANX_ROUTERS             = os.environ.get("MICTLANX_ROUTERS", "mictlanx-router-0:localhost:60666") #mictlanx-peer-2:localhost:7002")
MICTLANX_DEBUG               = bool(int(os.environ.get("MICTLANX_DEBUG",0)))
MICTLANX_DAEMON              = bool(int(os.environ.get("MICTLANX_DAEMON",1)))
MICTLANX_SHOW_METRICS        = bool(int(os.environ.get("MICTLANX_SHOW_METRICS",0)))
MICTLANX_DISABLED_LOG        = bool(int(os.environ.get("MICTLANX_DISABLED_LOG",0)))
MICTLANX_MAX_WORKERS         = int(os.environ.get("MICTLANX_MAX_WORKERS","12"))
MICTLANX_CLIENT_LB_ALGORITHM = os.environ.get("MICTLANX_CLIENT_LB_ALGORITHM","2CHOICES_UF")
MICTLANX_BUCKET_ID           = os.environ.get("MICTLANX_BUCKET_ID","rory") 
MICTLANX_OUTPUT_PATH         = os.environ.get("MICTLANX_OUTPUT_PATH","/rory/mictlanx")

STORAGE_CLIENT = Client(
    client_id       = MICTLANX_CLIENT_ID,
    bucket_id       = MICTLANX_BUCKET_ID,
    routers         = list(MictlanXUtiks.routers_from_str(MICTLANX_ROUTERS)),
    max_workers     = MICTLANX_MAX_WORKERS,
    lb_algorithm    = MICTLANX_CLIENT_LB_ALGORITHM,
    debug           = MICTLANX_DISABLED_LOG,
    log_output_path = MICTLANX_OUTPUT_PATH, 
 
)

#Cifrar
#serializar -- guardar en archivo
# leer el archivo --- from buffer
# convertir esos bytes a
# deberia dar una lista de pyctxt
# descifrar

##Como hacer lo mismo, pero ahora con pickle
##Serializar con pickle

#numpy -- ckks --
# 
class TestApp(unittest.TestCase):

    @unittest.skip("")
    def test_verify_multiplication(self):
        bucket_id = "rory"
        esm = "skmeanspqc1aencryptedshiftmatrix"

        

    @unittest.skip("")
    def test_resta(self):
        HE = Pyfhel()           # Creating empty Pyfhel object
        ckks_params = {
            'scheme': 'CKKS',   # can also be 'ckks'
            'n': 2**14,         # Polynomial modulus degree. For CKKS, n/2 values can be
                                #  encoded in a single ciphertext. 
                                #  Typ. 2^D for D in [10, 15]
            'scale': 2**30,     # All the encodings will use it for float->fixed point
                                #  conversion: x_fix = round(x_float * scale)
                                #  You can use this as default scale or use a different
                                #  scale on each operation (set in HE.encryptFrac)
            'qi_sizes': [60, 30, 30, 30, 60] # Number of bits of each prime in the chain. 
                                # Intermediate values should be  close to log2(scale)
                                # for each operation, to have small rounding errors.
        }
        HE.contextGen(**ckks_params)  # Generate context for ckks scheme
        HE.keyGen()

        vec1 = np.array([1.0, 2.0, 3.0, 4.0],  dtype=np.float64)
        vec2 = np.array([2.0, 3.0, 4.0, 5.0],  dtype=np.float64)

        # Cifrar los datos
        ctxt1 = HE.encryptFrac(vec1)
        ctxt2 = HE.encrypt(vec2)

        ctxt_subs = ctxt2 - ctxt1
        print(ctxt_subs)

        result_sub = HE.decryptFrac(ctxt_subs)
        print("Resta:", result_sub)

        # print("vec1 ", vec1,'ctxt_x ', ctxt1)
        # print("vec2 ", vec2,'ctxt_y ', ctxt2)


    @unittest.skip("")
    def test_verify_mean_decrypt(self):
        bucket_id = "rory"
        cent_i_id = "skmeanspqc1acenti"
        cent_j_id = "skmeanspqc1acentj"
        num_chunks = 2
        k = 2
        num_attributes = 5

        Cent_i_response = STORAGE_CLIENT.get_with_retry(bucket_id=bucket_id, key = cent_i_id)
        if Cent_i_response.is_err:
            return Response(response=f"GET Cent_i error [{cent_i_id}]", status=503)
        response = Cent_i_response.unwrap().value
        Cent_i = TestApp.bytes_to_pyctxt_list_v2(ckks = ckks, data=response)


        Cent_j_response = STORAGE_CLIENT.get_with_retry(bucket_id=bucket_id, key = cent_j_id)
        if Cent_j_response.is_err:
            return Response(response=f"GET Cent_j error [{cent_j_id}]", status=503)
        response = Cent_j_response.unwrap().value
        Cent_j = TestApp.bytes_to_pyctxt_list_v2(ckks = ckks, data=response)
        
        x = Cent_i[0]
        y = Cent_i[1]
        print("Cent_i",x , x.mod_level)
        print("Cent_j",y ,y.mod_level)

        old_matrix = ckks.decryptMatrix(
            ciphertext_matrix=Cent_i, 
            shape=[1,2],
        ) ## k x a
        
        new_matrix = ckks.decryptMatrix(
            ciphertext_matrix=Cent_j, 
            shape=[1,2],
        )
        mean_error = np.mean(np.abs((old_matrix - new_matrix) / old_matrix))
        print(mean_error)


    @staticmethod
    def bytes_to_pyctxt_list_v2(ckks:Ckks, data:bytes):
        xs = pickle.loads(data)
        scheme = ckks.he_object
        xx = [PyCtxt(None, scheme, None, x, "FRACTIONAL") for x in xs ]
        return xx

    @staticmethod
    def pyctxt_list_to_gen_bytes(ciphertext:List[PyCtxt]) -> Generator[bytes, None, None]:
        serialized_ciphertexts = [ctxt.to_bytes() for ctxt in ciphertext]
        xs= pickle.dumps(serialized_ciphertexts)
        for x in xs:
            yield x

    @unittest.skip("")
    def test_list_to_gen(self):

        bucket_id = "rory"
        key = "encryptedclusteringc0r10a5k20bb"
        num_chunks = 2
        k = 2
        num_attributes = 5
        shape = [k,num_attributes]
        encryptmatrix = TestApp.get_pyctxt_with_retry(
            STORAGE_CLIENT = STORAGE_CLIENT, 
            bucket_id=bucket_id, 
            num_chunks=num_chunks,
            key=key, 
            ckks = ckks)
        print(encryptmatrix)

        xs = TestApp.pyctxt_list_to_gen_bytes(ciphertext=encryptmatrix)
        for x in xs:
            print(x)
        print(x)

    @unittest.skip("")
    def test_decrypt(self):
        bucket_id = "rory"
        key = "encryptedclusteringc0r10a5k20bb"
        num_chunks = 2
        k = 2
        num_attributes = 5
        shape = [k,num_attributes]
        encryptmatrix = TestApp.get_pyctxt_with_retry(
            STORAGE_CLIENT = STORAGE_CLIENT, 
            bucket_id=bucket_id, 
            num_chunks=num_chunks,
            key=key, 
            ckks = ckks)
        print(encryptmatrix)

        # print("shape", len(encryptmatrix))

        # for x in encryptmatrix:
            # print(x)
            # print("*"*20)
        dec_matrix = ckks.decryptMatrix(
            ciphertext_matrix=encryptmatrix, 
            shape=shape,
        ) ## k x a
            # print(dec_matrix)
        print(dec_matrix)

    @staticmethod
    def get_pyctxt_with_retry(
            STORAGE_CLIENT,
            bucket_id:str, 
            num_chunks:int,
            key:str,
            ckks:Ckks,
            )-> List[PyCtxt]:
        
        x = STORAGE_CLIENT.get_with_retry(key = key, bucket_id=bucket_id)
        
        if x.is_err:
            e = x.unwrap_err()
            raise e
        serialized_ctxt_bytes = x.unwrap().value 
        chs = Chunks.from_bytes(data= serialized_ctxt_bytes, group_id="", num_chunks = num_chunks).unwrap()
    
        encryptedMatrix = TestApp.chunks_to_pyctxt_list(ckks= ckks, chunks= chs)

        return encryptedMatrix
    
    @unittest.skip("")
    def test_pyctxt_with_retry(self):
        bucket_id = "test"
        key = "test"
        num_chunks = 1
        init_shiftmatrix = TestApp.get_pyctxt_with_retry(
            STORAGE_CLIENT = STORAGE_CLIENT, 
            bucket_id=bucket_id, 
            num_chunks=num_chunks,
            key=key, 
            ckks = ckks)
        print(init_shiftmatrix)
    
    @staticmethod
    def chunks_to_pyctxt_list(chunks:Chunks, ckks:Ckks)->List[PyCtxt]:
        xs = []
        for ch in chunks.iter():
            # print("CHUNK",ch)
            x  = pickle.loads(ch.data)
            xx = TestApp.bytes_to_pyctxt_list(ckks=ckks,serialized_ctxt_bytes=x)
            # print("xx",xx)
            xs.extend(xx)
            # print("xs",xs)
        
        return xs
    
    @unittest.skip("")
    def test_local_read(self):
        # key = "encryptedskmeans123xxx888"
        key = "test"
        with open(f"/mnt/c/Users/isc_s/Downloads/{key}","rb") as f:
            data = f.read()
            chs = Chunks.from_bytes(data = data, group_id ="", num_chunks = 1).unwrap()
            for ch in chs.iter():
                print(ch)
                value = ch.data
                # x  = pickle.loads(value)
                # print(x)
                x = TestApp.chunks_to_pyctxt_list(ckks=ckks,serialized_ctxt_bytes=value)
                print(x)
                # print(value)

    @unittest.skip("")
    def test_download(self):
        bucket_id = "rory"
        key = "encryptedskmeans123xxx888"
        print("DOWNLOAD", bucket_id,key)
        res = STORAGE_CLIENT.get_with_retry(bucket_id=bucket_id,key=key,chunk_size="10MB")
        print("GET_RESULT",res)
        if res.is_ok:
            response = res.unwrap()
            data     = response.value
            result = Chunks.from_bytes(data = data, group_id = "",num_chunks=num_chunks).unwrap()
            
            print("CHUBK_RESULT", result)
            xx  = TestApp.chunks_to_pyctxt_list(chunks=result)
            print(xx)
            # print("RESULT", data,len(data))
        # res = STORAGE_CLIENT.get_metadata(bucket_id="test",key="test")
        # print("RESULT",res)
    @unittest.skip("")
    def test_put(self):
        plaintext_matrix = np.array([[1,2],[2,4],[4,5],[2,0],[2,3],[4,4]],dtype=np.float64)
        n           = plaintext_matrix.size
        _round      = False
        decimals    = 2
        path               = os.environ.get("KEYS_PATH","/rory/keys")
        ctx_filename       = os.environ.get("CTX_FILENAME","ctx")
        pubkey_filename    = os.environ.get("PUBKEY_FILENAME","pubkey")
        relinkey_filename  = os.environ.get("RELINKEY_FILENAME","relinkey")
        rotatekey_filename = os.environ.get("ROTATE_KEY_FILENAME","rotatekey")
        secretkey_filename = os.environ.get("SECRET_KEY_FILENAME","secretkey")
    
        result      = TestApp.segment_and_encrypt_ckks_with_executor(
            executor         = ppe,
            plaintext_matrix = plaintext_matrix,
            # dataowner        = dataowner,
            key              = key,
            n                = n,
            np_random        = np_random,
            num_chunks       = num_chunks,
            _round           = _round,
            decimals         = decimals,
            path               = path,
            ctx_filename       = ctx_filename,
            pubkey_filename    = pubkey_filename,
            relinkey_filename  = relinkey_filename,
            rotatekey_filename = rotatekey_filename,
            secretkey_filename = secretkey_filename

        )

        # xxx = result.iter();
        # print(xxx)
        # for ch in result.iter():
            # print(ch)
            # x = pickle.loads(ch.data)
            # xx = TestApp.bytes_to_pyctxt_list(ckks=ckks,serialized_ctxt_bytes=x)
        #     print("_"*40)
        #     print("X",xx)
        #     print("_"*40)
            # print(ch.data)
        # print(result)

        chunks_bytes = TestApp.chunks_to_bytes_gen(
            chs = result
        )
        
        res = STORAGE_CLIENT.put_chunked(
            chunks=chunks_bytes,
            bucket_id="test",
            key="test",
            # replication_factor=1
        )
        # print("Result", res)
        
        # x = TestApp.bytes_to_pyctxt_list(ckks,bytes(chunks_bytes))
        # print(x)
        # for ch in result.iter():
        #     x = TestApp.bytes_to_pyctxt_list_bytes(ckks=ckks,serialized_ctxt_bytes=ch.data)
        #     print("_"*40)
        #     print("X",x)
        #     print("_"*40)
            
    @staticmethod
    def chunks_to_bytes_gen(chs:Chunks) -> Generator[bytes,None,None]:
        for chunk in chs.iter():
            yield chunk.data

    @staticmethod
    def chunks_to_pyctxt_list_v1(
        ckks:Ckks,
        serialized_ctxt_bytes:bytes
    )->List[PyCtxt]:
        
        x  = pickle.loads(serialized_ctxt_bytes)
        xx = TestApp.bytes_to_pyctxt_list(ckks=ckks, serialized_ctxt_bytes=x)

        return xx

    @staticmethod
    def bytes_to_pyctxt_list(ckks:Ckks,serialized_ctxt_bytes:List[bytes])->List[PyCtxt]:
        scheme  = ckks.he_object
        xx      = list(map(lambda x: PyCtxt(None,scheme,None,x,'FRACTIONAL'), serialized_ctxt_bytes))
        # print("ddd",xx)
        return xx
    
    @unittest.skip("")
    def test_serialize_pyctxt(self):
        print("uno")
        plaintext_matrix = np.array([[1,2],[2,4],[4,5],[2,0],[2,3],[4,4]],dtype=np.float64)
        ckks = Ckks.from_pyfhel(
            _round   = False,
            decimals = 2,
            path               = os.environ.get("KEYS_PATH","/rory/keys"),
            ctx_filename       = os.environ.get("CTX_FILENAME","ctx"),
            pubkey_filename    = os.environ.get("PUBKEY_FILENAME","pubkey"),
            relinkey_filename  = os.environ.get("RELINKEY_FILENAME","relinkey"),
            rotatekey_filename = os.environ.get("ROTATE_KEY_FILENAME","rotatekey"),
            secretkey_filename = os.environ.get("SECRET_KEY_FILENAME","secretkey"),
        )
        # ckks = Ckks(he_object= he , _round = False, decimals = 2)
        # _______________________________________________________________________________
        # dataowner = DataOwnerPQC(scheme = ckks)

        encrypted = dataowner.outsourcedData(
            plaintext_matrix = plaintext_matrix,
            scheme = ckks
        )

        encrypted_matrix = encrypted.encrypted_matrix
        # print(encrypted_matrix)
        serialized_ciphertexts = [ctxt.to_bytes() for ctxt in encrypted_matrix]

        # with open("/rory/ciphertexts.bin", "wb") as f:
            # pickle.dump(serialized_ciphertexts, f)
        serialized_chunk = pickle.dumps(serialized_ciphertexts)
        print("dos")

    @unittest.skip("")
    def test_serialize_read(self):
        with open("/rory/ciphertexts.bin", "rb") as f:
            serialized_ciphertexts:List[bytes] = pickle.load(f)

        ckks = Ckks.from_pyfhel(
            _round   = False,
            decimals = 2,
            path               = os.environ.get("KEYS_PATH","/rory/keys"),
            ctx_filename       = os.environ.get("CTX_FILENAME","ctx"),
            pubkey_filename    = os.environ.get("PUBKEY_FILENAME","pubkey"),
            relinkey_filename  = os.environ.get("RELINKEY_FILENAME","relinkey"),
            rotatekey_filename = os.environ.get("ROTATE_KEY_FILENAME","rotatekey"),
            secretkey_filename = os.environ.get("SECRET_KEY_FILENAME","secretkey"),
        )
        print("SERIELIZED_LEN",len(serialized_ciphertexts))

        scheme = ckks.he_object
        xx = list(map(lambda x: PyCtxt(None,scheme,None,x,'FRACTIONAL'), serialized_ciphertexts))
        decrypt = ckks.decryptMatrix(xx, shape=[6,2])
        print(decrypt)
    

    @staticmethod
    def segment_and_encrypt_ckks_with_executor(
        executor:ProcessPoolExecutor,
        key:str,
        plaintext_matrix:npt.NDArray,
        n:int,
        np_random:bool,
        _round:bool, decimals:int, path:str, ctx_filename:str, 
        pubkey_filename:str, relinkey_filename:str, rotatekey_filename:str,secretkey_filename:str,
        num_chunks:int=2, 
    ):
        plaintext_matrix_chunks                 = Chunks.from_ndarray(ndarray= plaintext_matrix, group_id = key, num_chunks= num_chunks).unwrap()
        awaitable_chunks:List[Awaitable[Chunk]] = []
        for plaintext_matrix_chunk in plaintext_matrix_chunks.iter():
            future = executor.submit(
                TestApp.encrypt_chunk_ckks,
                key       = key,
                chunk     = plaintext_matrix_chunk,
                np_random = np_random,
                _round    = _round,
                decimals = decimals,
                path               = path,
                ctx_filename       = ctx_filename,
                pubkey_filename    = pubkey_filename,
                relinkey_filename  = relinkey_filename,
                rotatekey_filename = rotatekey_filename,
                secretkey_filename = secretkey_filename,

            )
            awaitable_chunks.append(future)
        return Chunks(chs= TestApp.to_chunks_generator(awaitable_chunks=awaitable_chunks),n =n)

        

    @staticmethod
    def encrypt_chunk_ckks(key:str, chunk:Chunk, np_random:bool, _round:bool, decimals:int, path:str, ctx_filename:str, 
                           pubkey_filename:str, relinkey_filename:str, rotatekey_filename:str,secretkey_filename:str)-> Chunk:
        try:
            dataowner = DataOwnerPQC(
                scheme= Ckks.from_pyfhel(
                    _round   = _round,
                    decimals = decimals,
                    path               = path,
                    ctx_filename       = ctx_filename,
                    pubkey_filename    = pubkey_filename,
                    relinkey_filename  = relinkey_filename,
                    rotatekey_filename = rotatekey_filename,
                    secretkey_filename = secretkey_filename,
                ) 
            )
            plaintext_matrix = chunk.to_ndarray().unwrap().copy()
            encyrpted_chunk:List[PyCtxt] = dataowner.ckks_encrypt_matrix_chunk(plaintext_matrix = plaintext_matrix, np_random = np_random)
            data = TestApp.pyctxt_list_to_bytes(ciphertext=encyrpted_chunk)
            return Chunk(
                group_id=key,
                index= chunk.index,
                data=data,
                chunk_id = Some("{}_{}".format(key,chunk.index))
            )
        except Exception as e:
            print("-"*50)
            print("ERRORRRORRR",e)
    @staticmethod
    def pyctxt_list_to_bytes(ciphertext:List[PyCtxt]):
        serialized_ciphertexts = [ctxt.to_bytes() for ctxt in ciphertext]
        return pickle.dumps(serialized_ciphertexts)

    
    @staticmethod
    def to_chunks_generator(awaitable_chunks:List[Awaitable[Chunk]]):
        try:
            xs = list(map(lambda fut: fut.result(), awaitable_chunks))
            return xs
        except Exception as e:
            print("TO_CHUNKS_EXCEPTION",e)
    
        
    @unittest.skip("")
    def test_segmentation(self):
        plaintext_matrix = np.array([[1,2],[2,4],[4,5],[2,0],[2,3],[4,4]],dtype=np.float64)
        n           = plaintext_matrix.size
        _round      = False
        decimals    = 2
        path               = os.environ.get("KEYS_PATH","/rory/keys")
        ctx_filename       = os.environ.get("CTX_FILENAME","ctx")
        pubkey_filename    = os.environ.get("PUBKEY_FILENAME","pubkey")
        relinkey_filename  = os.environ.get("RELINKEY_FILENAME","relinkey")
        rotatekey_filename = os.environ.get("ROTATE_KEY_FILENAME","rotatekey")
        secretkey_filename = os.environ.get("SECRET_KEY_FILENAME","secretkey")
    
        result      = TestApp.segment_and_encrypt_ckks_with_executor(
            executor         = ppe,
            plaintext_matrix = plaintext_matrix,
            # dataowner        = dataowner,
            key              = key,
            n                = n,
            np_random        = np_random,
            num_chunks       = num_chunks,
            _round           = _round,
            decimals         = decimals,
            path               = path,
            ctx_filename       = ctx_filename,
            pubkey_filename    = pubkey_filename,
            relinkey_filename  = relinkey_filename,
            rotatekey_filename = rotatekey_filename,
            secretkey_filename = secretkey_filename

        )

        return self.assertEqual(type(result), Chunks)
    

    @unittest.skip("")
    def test_ckss_encrypt_matrix_chunk(self):
        plaintext_matrix = np.array([[1,2],[2,4],[4,5],[2,0],[2,3],[4,4]],dtype=np.float64)
        result = dataowner.ckks_encrypt_matrix_chunk(
            plaintext_matrix = plaintext_matrix,
            np_random        = np_random
        )
        r1 = result[0]
        r1_bytes = r1.to_bytes()
        print("BYTES", r1_bytes)
        # xs = np.array([[1,2,],[2,3]])
        # chks = Chunk.from_ndarray(group_id = "x",index=0, ndarray=result,metadata={"shape":"(2,2)"})
        # print("CHUNK",chks)
        # print("RESULT=>", result,type(result))
        # print("XS",xs,type(xs))
    @unittest.skip("")
    def test_uno(self):
        pass


if __name__ == '__main__':
    unittest.main()