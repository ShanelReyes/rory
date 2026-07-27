import pytest
import numpy as np


@pytest.fixture
def secret_key(liu_scheme):
    secret_key = liu_scheme.generate_secret_key()
    return secret_key

# --- Datos de prueba ---
PLAINTEXT_MATRIX = [
    [0.73, 8.84],
    [49.93, 34.44],
    [0.57, 65.04],
    [62.15, 32.29],
    [59.47, 36.04]
]

# --- Tests ---

def test_liu_encryption_decryption(liu_scheme, secret_key):

    encrypted_matrix = liu_scheme.encryptMatrix(
        plaintext_matrix = PLAINTEXT_MATRIX,
        secret_key       = secret_key
    )
    decrypted_matrix = liu_scheme.decryptMatrix(
        ciphertext_matrix = encrypted_matrix.data,
        secret_key        = secret_key
    )

    np.testing.assert_allclose(decrypted_matrix.data, PLAINTEXT_MATRIX, rtol=1e-5)

def test_nbits(liu_scheme):
    x = liu_scheme.generateSecretRandomBit(nbits=128)
    print("x", x)
    assert x is not None

def test_homomorphic_add(liu_scheme, secret_key):
    v1, v2 = 5, 2

    cipher1 = liu_scheme.encryptScalar(plaintext=v1, secret_key=secret_key).data
    cipher2 = liu_scheme.encryptScalar(plaintext=v2, secret_key=secret_key).data

    enc_add = liu_scheme.add(ciphertext_1=cipher1, ciphertext_2=cipher2)
    assert isinstance(enc_add, np.ndarray)

    decrypted_add = liu_scheme.decryptScalar(ciphertext=enc_add, secret_key=secret_key).data
    print("Decrypted Addition Result:", decrypted_add)
    assert decrypted_add == (v1 + v2)

def test_subtract(liu_scheme, secret_key):
    v1, v2 = 5, 2

    cipher1 = liu_scheme.encryptScalar(plaintext=v1, secret_key=secret_key).data
    cipher2 = liu_scheme.encryptScalar(plaintext=v2, secret_key=secret_key).data

    enc_sub = liu_scheme.subtract(ciphertext_1=cipher1, ciphertext_2=cipher2)
    assert isinstance(enc_sub, np.ndarray)

    result = liu_scheme.decryptScalar(ciphertext=enc_sub, secret_key=secret_key).data
    print("Decrypted Subtraction Result:", result)
    assert result == (v1 - v2)

def test_multiply(liu_scheme, secret_key):
    v1, v2 = 5, 2

    cipher1 = liu_scheme.encryptScalar(plaintext=v1, secret_key=secret_key).data
    cipher2 = liu_scheme.encryptScalar(plaintext=v2, secret_key=secret_key).data

    enc_mult = liu_scheme.multiply(ciphertext_1=cipher1, ciphertext_2=cipher2)

    result = liu_scheme.decryptMultiply(ciphertext=enc_mult, secret_key=secret_key).data
    print("Decrypted Multiplication Result:", result)
    assert result == (v1 * v2)

def test_multiply_scalar(liu_scheme, secret_key):
    v1, v2 = 5, 2

    cipher1 = liu_scheme.encryptScalar(plaintext=v1, secret_key=secret_key).data

    enc_mult = liu_scheme.multiply_c(ciphertext=cipher1, scalar=v2)
    assert isinstance(enc_mult, np.ndarray)

    result = liu_scheme.decryptScalar(ciphertext=enc_mult, secret_key=secret_key).data
    print("Decrypted Multiplication with Scalar Result:", result)
    assert result == (v1 * v2)


def test_encrypt_vector(liu_scheme, secret_key):
    plaintext_vector = [10.0, 20.0, 30.0]
    encrypted = liu_scheme.encryptVector(
        plaintext_vector=plaintext_vector,
        secret_key=secret_key,
    ).data
    assert encrypted is not None

    decrypted = liu_scheme.decryptVector(
        ciphertext_vector=encrypted,
        secret_key=secret_key,
    ).data
    assert len(decrypted) == 3


def test_encrypt_decrypt_scalar(liu_scheme, secret_key):
    plaintext = 42.0
    ciphertext = liu_scheme.encryptScalar(plaintext=plaintext, secret_key=secret_key).data
    assert ciphertext is not None

    decrypted = liu_scheme.decryptScalar(ciphertext=ciphertext, secret_key=secret_key).data
    assert decrypted == plaintext


def test_generate_random(liu_scheme):
    random_val = liu_scheme.generateRandom()
    assert random_val is not None


def test_generate_random_np(liu_scheme):
    result = liu_scheme.generate_random_np(low=0, high=10, size=5)
    assert len(result) == 5
    assert all(0 <= v <= 10 for v in result)
