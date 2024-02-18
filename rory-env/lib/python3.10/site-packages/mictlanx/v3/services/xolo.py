import os
import requests as R
from mictlanx.v3.interfaces.service import Service
from mictlanx.v3.interfaces.payloads import AuthTokenPayload ,SignUpPayload, VerifyTokenPayload,RefreshTokenPayload,LogoutPayload
from mictlanx.v3.interfaces.responses import AuthResponse,SignUpResponse,LogoutResponse,VerifyTokenResponse,RefreshTokenResponse
from option import Result,Ok,Err,Option,Some,NONE
from Crypto.Cipher import AES
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey,X25519PublicKey
from cryptography.hazmat.primitives.asymmetric.types import PRIVATE_KEY_TYPES,PUBLIC_KEY_TYPES
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import serialization
from typing import Type,Any,Tuple
from option import Option, Some,NONE


class Xolo(Service):
    def __init__(self,ip_addr:Option[str]=NONE, port:Option[int]=NONE, api_version:Option[int]=Some(3), secret_path:Option[str]=Some("/mictlanx/keys")):
        super(Xolo,self).__init__(ip_addr=ip_addr.unwrap_or("localhost"),port=port.unwrap_or(10000),api_version=api_version)
        self.signup_url  = "{}/signup".format(self.base_url)
        self.auth_url    = '{}/auth'.format(self.base_url)
        self.verify_url  = "{}/verify".format(self.base_url)
        self.refresh_url = "{}/refresh".format(self.base_url)
        self.logout_url  = "{}/logout".format(self.base_url)
        self.secret_path   = secret_path.unwrap_or("/mictlanx/keys")
        # kwargs.get("base_path","/mictlanx/keys")
        os.makedirs(self.secret_path, exist_ok=True)
    
    def  key_pair_gen(self,filename:str):
        private_key = X25519PrivateKey.generate()
        pub_key     = private_key.public_key()
        private_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.PEM, 
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        public_bytes = pub_key.public_bytes(encoding=serialization.Encoding.PEM,format=serialization.PublicFormat.SubjectPublicKeyInfo)
        private_path = "{}/{}".format(self.secret_path,filename)
        public_path = "{}/{}.pub".format(self.secret_path,filename)
        with open(private_path,"wb") as f:
            f.write(private_bytes)
        with open(public_path,"wb") as f:
            f.write(public_bytes)

    def load_private_key(self,filename:str)->Result[Type[X25519PrivateKey],Exception]:
        try:
            private_path = "{}/{}".format(self.secret_path,filename)
            with open(private_path,"rb")  as f:
                x = f.read()
                private_key = serialization.load_pem_private_key(x,password=None)
            return Ok(private_key)
        except Exception as e:
            return Err(e)

    def load_public_key(self,filename:str)->Result[Type[X25519PublicKey],Exception]:
        try:
            public_path  = "{}/{}.pub".format(self.secret_path,filename)
            with open(public_path,"rb")  as f:
                public_key = serialization.load_pem_public_key(f.read())
            return Ok(public_key)
        except Exception as e:
            return Err(e)
        
    def load_key_pair(self,filename:str)->Result[Tuple[Type[X25519PrivateKey],Type[X25519PublicKey]],Exception]:
        try:
            private_key_result = self.load_private_key(filename=filename)
            public_key_result = self.load_public_key(filename=filename)
            return private_key_result.flatmap(lambda private_key: public_key_result.flatmap(lambda public_key: Ok((private_key,public_key))))
        except Exception as e:
            return Err(e)
            
    def encrypt_aes(self,key:bytes=None,data:bytes=None,header:Option[bytes]=NONE)->Result[bytes,Exception]:
        try:
            cipher         = AES.new(key=key,mode=AES.MODE_GCM)
            if(header.is_some):
                cipher.update(header.unwrap())
            ciphertext,tag = cipher.encrypt_and_digest(data)
            nonce          = cipher.nonce
            return Ok(tag+ciphertext+nonce)
        except Exception as e:
            return Err(e)
        
    def decrypt_aes(self,key:bytes=None,data:bytes=None,header:Option[bytes] =NONE)->Result[bytes,Exception]:
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

    def signup(self,payload:SignUpPayload)->Result[SignUpResponse,R.RequestException]:
        try:
            response_data = payload.to_dict()
            response = R.post(self.signup_url,json=response_data)
            response.raise_for_status()
            response_data = SignUpResponse(**response.json())
            return Ok(response_data)
        except R.RequestException as e:
            return Err(e)
        
    def auth(self,payload:AuthTokenPayload)->Result[AuthResponse,R.RequestException]:
        try:
            response_data = payload.to_dict()
            response      = R.post(self.auth_url,json=response_data)
            response.raise_for_status()
            response_data = AuthResponse(**response.json())
            return Ok(response_data)
        except Exception as e:
            return Err(e)
        
    def authenticate_or_signup(self,payload:AuthTokenPayload,signup_payload:Option[SignUpPayload])->Result[AuthResponse,R.RequestException]:
        try:
            auth_result = self.auth(payload=payload)
            if(auth_result.is_ok):
                return auth_result
            else:
                auth_error = auth_result.unwrap_err()
                response:R.Response = auth_error.response
                if not type(response) ==R.Response:
                    return auth_result
                
                if(signup_payload.is_none):
                    return Err(Exception("No SignUpPyload provided."))
                if response.status_code != 401 | response.status_code != 403:  
                    signup_result = self.signup(signup_payload.unwrap())
                    if(signup_result.is_ok):
                        auth_result = self.auth(payload=payload)
                        return auth_result
                    return signup_result
                else:
                    return auth_result
        except Exception as e:
            return Err(e)

    def verify_token(self,payload:VerifyTokenPayload)->Result[VerifyTokenResponse,R.RequestException]:
        try:
            response_data = payload.to_dict()
            response      = R.post(self.verify_url,json=response_data)
            response.raise_for_status()
            response_data = VerifyTokenResponse(**response.json())
            return Ok(response_data)
        except R.RequestException as e:
            return Err(e)
        
    def refresh_token(self,payload:RefreshTokenPayload):
        try:
            response_data = payload.to_dict()
            response      = R.post(self.refresh_url,json=response_data)
            response.raise_for_status()
            response_data = RefreshTokenResponse(**response.json())
            return Ok(response_data)
        except Exception as e:
            return Err(e)
        
    def logout(self,payload:LogoutPayload):
        try:
            response_data = payload.to_dict()
            response      = R.post(self.logout_url,json=response_data)
            response.raise_for_status()
            response_data = LogoutResponse(**response.json())
            return Ok(response_data)
        except Exception as e:
            return Err(e)
        


if __name__ =="__main__":
    xolo                  = Xolo(ip_addr="",port=0)
    res  = xolo.load_key_pair(filename="mictlanx").unwrap()
    res2 = xolo.load_key_pair(filename="mictlanx2").unwrap()
    shared_key = res[0].exchange(peer_public_key=res2[1])
    # print("SHARED_KEY",shared_key,len(shared_key))
    # shared_key                   = bytes.fromhex("87a3bec68f8c004de4f26773a4000d7b04e77f56bd3b6d382ff464f621bda6f7")
    # shared_key                   = bytes.fromhex("1e490cd52d1e6b051f96edba2af2f7d53e266de9df150c26495f1511511222cc")
    encrypted_data_result = xolo.encrypt_aes(key=shared_key,data=b"Hello world")
    print("ENCRYPTED_DATA",encrypted_data_result)
    if(encrypted_data_result.is_ok):
        plaintext      = xolo.decrypt_aes(key=shared_key, data=encrypted_data_result.unwrap() )
        print("DECRYPTED_DATA",plaintext)