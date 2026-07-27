import os
import pytest
import time as T
import numpy as np
import phe
from rory.core.security.dataowner_paillier import DataOwner as DataownerPaillier
from rory.core.security.cryptosystem.paillier import Paillier

RORY_KEYS_PATH = os.environ.get("PAILLIER_KEYS_PATH", "/rory/keys/")
# --- Fixtures ---

@pytest.fixture
def matrix_3x3():
    return np.random.random(size=(3, 3))

@pytest.fixture
def matrix_large():
    return np.random.random(size=(1000, 10))

# --- Tests ---

# @pytest.mark.skip(reason="test phe directo")
def test_phe_encrypt_direct(matrix_3x3):
    t1 = T.time()
    pk, sk = phe.generate_paillier_keypair(n_length=128)
    
    for row in matrix_3x3:
        for v in row:
            res = pk.encrypt(v)
            # Verificamos que el objeto cifrado sea de la clase esperada
            assert isinstance(res, phe.paillier.EncryptedNumber)
            
    assert T.time() - t1 > 0

# @pytest.mark.skip(reason="test paillier core")
def test_core_encrypt_scalar(matrix_3x3, pallier_dataowner):
    do = pallier_dataowner
    paillier = Paillier(public_key=do.pk)

    t1 = T.time()
    for row in matrix_3x3:
        for v in row:
            result = paillier.encrypt_scalar(plaintext=v)
            assert result.data is not None

    assert T.time() - t1 > 0

# @pytest.mark.skip(reason="test phe directo")
def test_paillier_chunk_performance(matrix_large):
    t1 = T.time()
    do = DataownerPaillier(securitylevel=128)
    do.generate_keys(save=False)
    
    res = do.paillier_encrypt_matrix_chunk(plaintext_matrix=matrix_large)
    
    # Verificamos dimensiones del resultado
    assert res is not None
    assert len(res) == len(matrix_large)
    assert T.time() - t1 > 0

# @pytest.mark.skip(reason="generacion de llaves")
def test_key_generation_performance():
    t1 = T.time()
    do = DataownerPaillier(securitylevel=192)
    do.generate_keys(
        output_path=RORY_KEYS_PATH,
        filename="rory-phe-192",
        save=True
    )
    
    assert do.pk is not None
    assert do.sk is not None
    assert T.time() - t1 > 0



def test_encrypt_decrypt_scalar():
    paillier = Paillier()
    paillier.generate_keys(security_level=128)
    plaintext = 42.0
    result = paillier.encrypt_scalar(plaintext=plaintext)
    assert result.data is not None

    decrypted = paillier.decrypt_scalar(ciphertext=result.data)
    assert decrypted.data == plaintext


def test_encrypt_decrypt_vector():
    paillier = Paillier()
    paillier.generate_keys(security_level=128)
    plaintext_vector = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = paillier.encrypt_vector(plaintext_vector=plaintext_vector)
    assert len(result.data) == 5

    decrypted = paillier.decrypt_vector(ciphertext_vector=result.data)
    assert len(decrypted.data) == 5
    assert (decrypted.data == plaintext_vector).all()


def test_encrypt_decrypt_matrix():
    paillier = Paillier()
    paillier.generate_keys(security_level=128)
    plaintext_matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = paillier.encrypt_matrix(plaintext_matrix=plaintext_matrix)
    assert result.data is not None

    decrypted_result = paillier.decrypt_matrix(ciphertext_matrix=result.data)
    assert decrypted_result.data is not None


