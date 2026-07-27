import os
import pytest
import numpy as np
# import pickle
from Pyfhel import Pyfhel

from rory.core.security.cryptosystem.pqc.ckks import Ckks, CkksModes

SCHEME = "CKKS"
MODE = CkksModes.DEFAULT
SECURITY_LEVEL = 128
RORY_KEYS_PATH = os.environ.get("RORY_KEYS_PATH", "/rory/keys/keys128/")

# @pytest.fixture
# def default_ckks_client():
#     """Provee un cliente CKKS básico para pruebas rápidas."""
#     return Ckks.from_pyfhel(path=RORY_KEYS_PATH, _round=True, decimals=2)

@pytest.fixture
def sink_path():
    return "/sink"

# --- TESTS ---

# @pytest.mark.skip(reason="generate keys")
# 
# @pytest.mark.skip(reason="read llaves")
def test_read_keys(ckks_client):
    ckks = ckks_client
    # ckks = Ckks.from_pyfhel(path=RORY_KEYS_PATH, _round=True, decimals=2)
    plaintext_matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    res = ckks.encryptMatrix(plaintext_matrix=plaintext_matrix)
    
    print(res)
    dres = ckks.adjust_matrix(
        ckks.decryptMatrix(ciphertext_matrix=res, adjust=False), 
        shape=plaintext_matrix.shape
    )
    print(dres)
    assert dres.shape == plaintext_matrix.shape

# @pytest.mark.skip(reason="encrypt list")
def test_encrypt_segment(ckks_client):
    ckks = ckks_client
    v1 = [2, 2, 3, 5, 4, 9, 1, 7, 7, 7, 4, 0]
    encode_v1 = ckks.encode_list(v1)
    encrypt_v1 = ckks.encrypt_list(encode_v1)
    decrypt_v1 = ckks.decrypt_list(encrypt_v1)
    
    print(decrypt_v1)
    assert len(decrypt_v1) >= len(v1)

# @pytest.mark.skip(reason="Encrypt and decrypt")
def test_cipher(sink_path):
    plaintext_matrix = np.array([[1,2],[2,4],[4,5],[2,0],[2,3],[4,4]])
    ckks = Ckks.from_pyfhel(path=RORY_KEYS_PATH, _round=False, decimals=2)

    em = ckks.encryptMatrix(plaintext_matrix=plaintext_matrix.astype(np.float32))
    x = ckks.decryptMatrix(ciphertext_matrix=em, shape=[6,2], adjust=True)
    assert x.shape == (6, 2)

# @pytest.mark.skip(reason="")
def test_load_ctx2(sink_path):
    HE = Ckks.load_pyfhel(path=RORY_KEYS_PATH)
    assert isinstance(HE, Pyfhel)

# @pytest.mark.skip(reason="")
def test_ckks_encrypt_decrypt_vector(ckks_client):
    ckks = ckks_client
    # ckks = Ckks.create_client(save=True)
    input_v = np.array([1,2,3], dtype=np.float32)
    res = ckks.encryptVector(plaintext_vector=input_v)
    dres = ckks.decryptVector(ciphertext_vector=res)
    assert np.allclose(input_v, dres[:3], atol=0.1)

# @pytest.mark.skip(reason="")
def test_save_ciphertext(sink_path):
    HE = Ckks.load_pyfhel(path=RORY_KEYS_PATH)
    integer1 = np.array([127, 2], dtype=np.int64)
    integer2 = np.array([-2, 1], dtype=np.int64)
    ctxt1 = HE.encrypt(integer1)
    ctxt2 = HE.encrypt(integer2)
    res = ctxt1 + ctxt2
    # res_bytes = res.to_bytes()
    # print(res_bytes)
    assert res.to_bytes() is not None