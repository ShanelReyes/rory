import hashlib as H
import humanfriendly as HF
from pathlib import Path
from Crypto.Cipher import AES
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey,X25519PublicKey
from cryptography.hazmat.primitives.asymmetric.types import PRIVATE_KEY_TYPES,PUBLIC_KEY_TYPES
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import serialization
from typing import Tuple,Type
from option import Result,Ok,Err,Option,Some,NONE
import os

class Utils:
    SECRET_PATH = os.environ.get("XOLO_SECRET_PATH","/mictlanx/xolo/.keys")

    
    @staticmethod
    def pbkdf2(password:str,key_length:int=32, iterations:int = 1000,salt_length:int = 16)->str:
        _pass      = password.encode("utf-8")
        salt       = os.urandom(salt_length)
        key        = H.pbkdf2_hmac('sha256', _pass, salt, iterations, key_length)
        # H.pbkdf2_hmac
        return "pbkdf2$l={},sl={},i={}${}/{}".format(key_length,salt_length,iterations,salt.hex(),key.hex())

    @staticmethod
    def check_password_hash(password:str, password_hash:str):
        (version,params,value) = password_hash.split("$")
        (key_length_var, salt_length_var, iterations_var) = params.split(",")
        (_, key_length)  = key_length_var.split("=")
        (_, salt_length) = salt_length_var.split("=")
        (_, iterations)  = iterations_var.split("=")
        (salt,_password_hash) = value.split("/")
        
        # print(version,params,salt,_password_hash)
        # x = bytes.fromhex(password)
        # print(x)
        _key        = H.pbkdf2_hmac('sha256',password.encode("utf-8"), bytes.fromhex(salt), int(iterations), int(key_length)).hex()
        return _key == _password_hash
        # print("LOCAL_KEY",_key.hex())



    @staticmethod
    def sha256(value:bytes)->str:
        h = H.sha256()
        h.update(value)
        return h.hexdigest()
    
    @staticmethod
    def sha256_file(path:str)->Tuple[str,int]:
        h = H.sha256()
        size = 0
        with open(path,"rb") as f:
            while True:
                data = f.read()
                if not data:
                    return (h.hexdigest(),size)
                size+= len(data)
                h.update(data)

    @staticmethod
    def extract_path_sha256_size(path:str)->Tuple[str,str,int]:
        h = H.sha256()
        size = 0
        with open(path,"rb") as f:
            while True:
                data = f.read()
                if not data:
                    return (path,h.hexdigest(),size)
                size+= len(data)
                h.update(data)
            
    @staticmethod
    def  X25519_key_pair_generator(filename:str):
        os.makedirs(Utils.SECRET_PATH, exist_ok=True)
        private_key = X25519PrivateKey.generate()
        pub_key     = private_key.public_key()
        private_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.PEM, 
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        public_bytes = pub_key.public_bytes(encoding=serialization.Encoding.PEM,format=serialization.PublicFormat.SubjectPublicKeyInfo)
        private_path = "{}/{}".format(Utils.SECRET_PATH,filename)
        public_path = "{}/{}.pub".format(Utils.SECRET_PATH,filename)
        with open(private_path,"wb") as f:
            f.write(private_bytes)
        with open(public_path,"wb") as f:
            f.write(public_bytes)

    @staticmethod
    def load_private_key(filename:str)->Result[Type[X25519PrivateKey],Exception]:
        try:
            private_path = "{}/{}".format(Utils.SECRET_PATH,filename)
            with open(private_path,"rb")  as f:
                x = f.read()
                private_key = serialization.load_pem_private_key(x,password=None)
            return Ok(private_key)
        except Exception as e:
            return Err(e)

    @staticmethod
    def load_public_key(filename:str)->Result[Type[X25519PublicKey],Exception]:
        try:
            public_path  = "{}/{}.pub".format(Utils.SECRET_PATH,filename)
            with open(public_path,"rb")  as f:
                public_key = serialization.load_pem_public_key(f.read())
            return Ok(public_key)
        except Exception as e:
            return Err(e)
        
    @staticmethod
    def load_key_pair(filename:str)->Result[Tuple[Type[X25519PrivateKey],Type[X25519PublicKey]],Exception]:
        try:
            private_key_result = Utils.load_private_key(filename=filename)
            public_key_result = Utils.load_public_key(filename=filename)
            return private_key_result.flatmap(lambda private_key: public_key_result.flatmap(lambda public_key: Ok((private_key,public_key))))
        except Exception as e:
            return Err(e)
            
    @staticmethod
    def encrypt_aes(key:bytes=None,data:bytes=None,header:Option[bytes]=NONE)->Result[bytes,Exception]:
        try:
            cipher         = AES.new(key=key,mode=AES.MODE_GCM)
            if(header.is_some):
                cipher.update(header.unwrap())
            ciphertext,tag = cipher.encrypt_and_digest(data)
            nonce          = cipher.nonce
            return Ok(tag+ciphertext+nonce)
        except Exception as e:
            return Err(e)
    
    @staticmethod
    def encrypt_file_aes(path:str,dest_path:str="/mictlanx/xolo/out",key:bytes=None,header:Option[bytes]=NONE,chunk_size:str="1MB")->Result[str,Exception]:
        if os.path.isdir(path):
            return Err(Exception("The path is a directory"))
        elif not os.path.exists(path=path):
            return Err(Exception("{} not exists".format(path)))
        _path = Path(path)
        filename = _path.name

        _chunk_size = HF.parse_size(chunk_size)
        os.makedirs(dest_path, exist_ok=True)
        try:
            cipher         = AES.new(key=key,mode=AES.MODE_GCM)
            if(header.is_some):
                cipher.update(header.unwrap())

            
            dest_path_file = "{}/{}.enc".format(dest_path,filename)

            with open(path,"rb") as file:
                with open(dest_path_file,"wb") as dest_file:
                    while True:
                        data = file.read(_chunk_size)
                        if len(data) == 0:
                            break
                        encrypted_data = cipher.encrypt(data)
                        dest_file.write(encrypted_data)
                        # ciphertext,tag = cipher.encrypt_and_digest(data)
                    nonce          = cipher.nonce
                    tag = cipher.digest()
                    dest_file.write(nonce)
                    dest_file.write(tag)
                    return Ok(dest_path_file)
                # return Ok(tag+ciphertext+nonce)
        except Exception as e:
            return Err(e)
        
    @staticmethod
    def decrypt_file_aes(path:str,dest_path:str,key:bytes=None,header:Option[bytes]=NONE,chunk_size:str="1MB",extra_bytes:int=32)->Result[str,Exception]:
        if os.path.isdir(path):
            return Err(Exception("The path is a directory"))
        elif not os.path.exists(path=path):
            return Err(Exception("{} not exists".format(path)))
        _path = Path(path)
        filename = _path.name

        _chunk_size = HF.parse_size(chunk_size)
        os.makedirs(dest_path, exist_ok=True)
        try:
            cipher         = AES.new(key=key,mode=AES.MODE_GCM)
            if(header.is_some):
                cipher.update(header.unwrap())

            
            _filename = ".".join(filename.split(".")[:-1])
            dest_path_file = "{}/{}".format(dest_path,_filename)
            # dest_path_file
            file = open(path,"rb")
            with open(dest_path_file,"wb") as dest_file:
                file.seek(0,os.SEEK_END)
                file_size = file.tell()-extra_bytes
                # _____________
                file.seek(-32, os.SEEK_END)
                last_chunk = file.read(32)
                nonce = last_chunk[:16]
                tag   = last_chunk[16:]
                cipher     =  AES.new(key=key,mode= AES.MODE_GCM,nonce=nonce)
                
                if(header.is_some):
                    cipher.update(header.unwrap())
                total_size_to_read = file_size
                file.seek(0)
                while total_size_to_read >0:
                    current_chunk_size = min(_chunk_size, file_size)
                    chunk              = file.read(current_chunk_size)
                    x                  = cipher.decrypt(chunk)
                    total_size_to_read-=current_chunk_size
                    dest_file.write(x)
            file.close()
            cipher.verify(received_mac_tag=tag)
            return Ok(dest_path_file)
        except Exception as e:
            return Err(e)
    @staticmethod
    def decrypt_aes(key:bytes=None,data:bytes=None,header:Option[bytes] =NONE)->Result[bytes,Exception]:
        # iterations = 1000
        try:
            tag        = data[:16] 
            ciphertext = data[16:len(data)-16]
            nonce      = data[-16:]
            cipher     =  AES.new(key=key,mode= AES.MODE_GCM,nonce=nonce)
            if(header.is_some):
                cipher.update(header.unwrap())
            return Ok(cipher.decrypt_and_verify(ciphertext=ciphertext,received_mac_tag=tag))
        except Exception as e:
            return Err(e)
        # iterations = 1000
if __name__ =="__main__":
    res  = Utils.load_key_pair(filename="jcastillo").unwrap()
    res2 = Utils.load_key_pair(filename="test").unwrap()
    shared_key = res[0].exchange(peer_public_key=res2[1])
    # Utils.X25519_key_pair_generator(
    #     filename="test"
    # )
    x = Utils.decrypt_file_aes(
        path="/mictlanx/xolo/out/01.pdf.enc",
        dest_path="/source/xolo",
        chunk_size="1MB",
        header=NONE,
        key=shared_key
    )
    # x = Utils.encrypt_file_aes(
    #     path="/source/01.pdf",
    #     dest_path="/mictlanx/xolo/out",
    #     key=shared_key,
    #     chunk_size="1MB",
    #     header=NONE
    # )
    print(x)
    # x = Utils.pbkdf2(password="xxx",key_length=32,iterations=1000, salt_length=16)
    # y = Utils.check_password_hash(password="xx",password_hash=x )
    # print(x,y)
    # pass = Utils.