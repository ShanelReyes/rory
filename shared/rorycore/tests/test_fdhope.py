import numpy as np
import pytest
from rory.core.security.cryptosystem.fdhope import Fdhope
from rory.core.interfaces.cipher_result import CipherResult


@pytest.fixture
def messagespace_and_cipherspace():
    dataset = np.array([[0.5, 1.2], [3.4, 2.1], [7.8, 6.5], [9.1, 8.3]], dtype=np.float64)
    ms, cs = Fdhope.keygen(
        dataset=dataset,
        minVal=0,
        max_range=8,
        proportion=15,
        range_limit=2,
        default_intervalLenght=0.001,
    )
    return ms, cs


def test_find_max():
    D = np.array([[-5.0, 3.0], [10.0, -2.0]], dtype=np.float64)
    max_val = Fdhope.findMax(D)
    assert max_val == 10.0


def test_find_max_with_negatives():
    D = np.array([[-15.0, -3.0], [-1.0, -7.0]], dtype=np.float64)
    max_val = Fdhope.findMax(D)
    assert max_val == 15.0


def test_generate_range_keys():
    keys = Fdhope.generate_range_keys(n_range=5)
    assert len(keys) == 5
    assert keys[0] == "RANGE_0"
    assert keys[4] == "RANGE_4"


def test_generate_range_values():
    range_ids = Fdhope.generate_range_keys(n_range=3)
    ranges_values = [0, 3.5, 10.0]
    result = Fdhope.generate_range_values(
        minValue=0,
        maxValue=10,
        n_range=3,
        range_ids=range_ids,
        ranges_values=ranges_values,
    )
    assert len(result) == 3
    assert "RANGE_0" in result
    assert result["RANGE_0"][0] == 0
    assert result["RANGE_0"][1] == 3.5


def test_get_interval_id():
    messagespace = {
        "RANGE_0": (0, 3),
        "RANGE_1": (3, 7),
        "RANGE_2": (7, 100),
    }
    key = Fdhope.getIntervalID(plaintext=2.5, messagespace=messagespace)
    assert key == "RANGE_0"

    key = Fdhope.getIntervalID(plaintext=5.0, messagespace=messagespace)
    assert key == "RANGE_1"


def test_get_interval_id_negative():
    messagespace = {
        "RANGE_0": (0, 3),
        "RANGE_1": (3, 10),
    }
    key = Fdhope.getIntervalID(plaintext=-2.5, messagespace=messagespace)
    assert key == "RANGE_0"


def test_get_boundary():
    space = {
        "RANGE_0": (0.0, 3.5),
        "RANGE_1": (3.5, 10.0),
    }
    lo, hi = Fdhope.getBoundary(interval_id="RANGE_0", space=space)
    assert lo == 0.0
    assert hi == 3.5


def test_calculate_dens_2d():
    dataset = np.array([[0.5, 1.2], [3.4, 2.1], [7.8, 6.5]], dtype=np.float64)
    messagespace = {
        "RANGE_0": (0, 3),
        "RANGE_1": (3, 10),
    }
    initial_dens = {"RANGE_0": 0, "RANGE_1": 0}
    density = Fdhope.calculate_dens(
        dataset=dataset,
        messagespace=messagespace,
        initial_dens=initial_dens,
    )
    assert isinstance(density, dict)
    assert density["RANGE_0"] + density["RANGE_1"] == 6


def test_calculate_dens_3d():
    dataset = np.array([
        [[0.5], [1.2]],
        [[3.4], [2.1]],
    ], dtype=np.float64)
    messagespace = {
        "RANGE_0": (0, 3),
        "RANGE_1": (3, 10),
    }
    initial_dens = {"RANGE_0": 0, "RANGE_1": 0}
    density = Fdhope.calculate_dens(
        dataset=dataset,
        messagespace=messagespace,
        initial_dens=initial_dens,
    )
    assert density["RANGE_0"] + density["RANGE_1"] == 4


def test_calculate_interval_length():
    density = {"RANGE_0": 5, "RANGE_1": 0, "RANGE_2": 3}
    result = Fdhope.calculate_intervalLength(
        density=density,
        lenTriangle=8,
        maxVal_cipherspace=100,
        default_intervalLenght=0.001,
    )
    assert result["RANGE_1"] == 0.001
    assert result["RANGE_0"] > result["RANGE_1"]


def test_keygen(messagespace_and_cipherspace):
    ms, cs = messagespace_and_cipherspace
    assert isinstance(ms, dict)
    assert isinstance(cs, dict)
    assert len(ms) > 0
    assert len(cs) > 0
    for key in ms:
        assert key in cs


def test_encrypt_scalar(messagespace_and_cipherspace):
    ms, cs = messagespace_and_cipherspace
    plaintext = 5.0
    ciphertext = Fdhope.encrypt(
        plaintext=plaintext,
        messagespace=ms,
        cipherspace=cs,
        sens=0.00001,
    )
    assert isinstance(ciphertext, float)
    assert ciphertext != plaintext


def test_encrypt_preserves_sign(messagespace_and_cipherspace):
    ms, cs = messagespace_and_cipherspace
    pos_ct = Fdhope.encrypt(plaintext=5.0, messagespace=ms, cipherspace=cs)
    neg_ct = Fdhope.encrypt(plaintext=-5.0, messagespace=ms, cipherspace=cs)
    assert pos_ct > 0
    assert neg_ct < 0


def test_encrypt_vector(messagespace_and_cipherspace):
    ms, cs = messagespace_and_cipherspace
    plaintext_vector = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    encrypted_vector = Fdhope.encryptVector(
        plaintext_vector=plaintext_vector,
        messagespace=ms,
        cipherspace=cs,
        sens=0.00001,
    )
    assert isinstance(encrypted_vector, CipherResult)
    assert len(encrypted_vector.data) == 3
    assert all(v != 0 for v in encrypted_vector.data)


def test_encrypt_matrix(messagespace_and_cipherspace):
    ms, cs = messagespace_and_cipherspace
    plaintext_matrix = np.array([[0.5, 1.2], [3.4, 2.1]], dtype=np.float64)
    result = Fdhope.encryptMatrix(
        plaintext_matrix=plaintext_matrix,
        messagespace=ms,
        cipherspace=cs,
        sens=0.00001,
    )
    assert isinstance(result, CipherResult)
    assert result.data.shape == (2, 2)


def test_encrypt_tensor(messagespace_and_cipherspace):
    ms, cs = messagespace_and_cipherspace
    plaintext_tensor = np.array([
        [[0.5], [1.2]],
        [[3.4], [2.1]],
    ], dtype=np.float64)
    result = Fdhope.encryptTensor(
        plaintext_tensor=plaintext_tensor,
        messagespace=ms,
        cipherspace=cs,
        sens=0.00001,
    )
    assert isinstance(result, CipherResult)


def test_encrypt_obeys_order(messagespace_and_cipherspace):
    ms, cs = messagespace_and_cipherspace
    small = abs(Fdhope.encrypt(plaintext=1.0, messagespace=ms, cipherspace=cs))
    large = abs(Fdhope.encrypt(plaintext=9.0, messagespace=ms, cipherspace=cs))
    assert small < large
