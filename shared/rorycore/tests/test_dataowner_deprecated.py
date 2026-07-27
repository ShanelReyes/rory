import numpy as np
import pytest
from rory.core.security.dataowner_deprecated import DataOwner
from rory.core.interfaces.outsourced_result import OutsourcedDataResult


@pytest.fixture
def data_owner(liu_scheme):
    return DataOwner(liu_scheme=liu_scheme, sens=0.00001)


@pytest.fixture
def plaintext_matrix():
    return np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0],
    ], dtype=np.float64)


def test_init(data_owner):
    assert data_owner.sens == 0.00001
    assert data_owner.liu_scheme is not None
    assert data_owner.sk is not None


def test_calculate_udm_diff(data_owner, plaintext_matrix):
    udm = data_owner.calculate_UDM(plaintext_matrix=plaintext_matrix, mode="diff")
    assert udm.shape == (5, 5, 3)
    assert np.allclose(udm[0, 0], [0, 0, 0])
    assert np.allclose(udm[0, 1], plaintext_matrix[0] - plaintext_matrix[1])


def test_calculate_udm_diff_abs(data_owner, plaintext_matrix):
    udm = data_owner.calculate_UDM(plaintext_matrix=plaintext_matrix, mode="diff-abs")
    assert udm.shape == (5, 5, 3)
    assert np.all(udm >= 0)


def test_calculate_dm(data_owner, plaintext_matrix):
    dm = data_owner.calculate_DM(plaintext_matrix=plaintext_matrix)
    assert dm.shape == (5, 5)
    assert dm[0, 0] == 0
    assert dm[0, 1] > 0


def test_liu_encrypt_matrix_chunk(data_owner, plaintext_matrix):
    encrypted = data_owner.liu_encrypt_matrix_chunk(
        plaintext_matrix=plaintext_matrix
    )
    assert encrypted is not None
    assert isinstance(encrypted, np.ndarray)


def test_get_u_skmeans(data_owner, plaintext_matrix):
    U = data_owner.get_U(plaintext_matrix=plaintext_matrix, algorithm="SKMEANS")
    assert U.shape == (5, 5, 3)


def test_get_u_dbskmeans(data_owner, plaintext_matrix):
    U = data_owner.get_U(plaintext_matrix=plaintext_matrix, algorithm="DBSKMEANS")
    assert U.shape == (5, 5, 3)
    assert len(data_owner.messageIntervals) > 0
    assert len(data_owner.cypherIntervals) > 0


def test_get_u_dbsnnc(data_owner, plaintext_matrix):
    U = data_owner.get_U(plaintext_matrix=plaintext_matrix, algorithm="DBSNNC")
    assert U.shape == (5, 5)
    assert len(data_owner.messageIntervals) > 0


def test_get_u_nnc(data_owner, plaintext_matrix):
    U = data_owner.get_U(plaintext_matrix=plaintext_matrix, algorithm="NNC")
    assert U.shape == (5, 5)


def test_get_u_unknown_algorithm(data_owner, plaintext_matrix):
    with pytest.raises(Exception):
        data_owner.get_U(plaintext_matrix=plaintext_matrix, algorithm="INVALID")


def test_encrypt_udm_chunks_dbskmeans(data_owner, plaintext_matrix):
    udm = data_owner.calculate_UDM(plaintext_matrix=plaintext_matrix)
    data_owner.messageIntervals, data_owner.cypherIntervals = {
        "RANGE_0": (0, 10),
        "RANGE_1": (10, 100),
    }, {
        "RANGE_0": (0, 50),
        "RANGE_1": (50, 150),
    }
    encrypted = data_owner.encrypt_udm_chunks(
        plaintext_matrix=udm, sens=0.0001, algorithm="DBSKMEANS"
    )
    assert encrypted is not None


def test_encrypt_udm_chunks_dbsnnc(data_owner, plaintext_matrix):
    dm = data_owner.calculate_DM(plaintext_matrix=plaintext_matrix)
    data_owner.messageIntervals, data_owner.cypherIntervals = {
        "RANGE_0": (0, 10),
        "RANGE_1": (10, 100),
    }, {
        "RANGE_0": (0, 50),
        "RANGE_1": (50, 150),
    }
    encrypted = data_owner.encrypt_udm_chunks(
        plaintext_matrix=dm, sens=0.0001, algorithm="DBSNNC"
    )
    assert encrypted is not None


def test_encrypt_udm_chunks_unknown(data_owner, plaintext_matrix):
    with pytest.raises(Exception):
        data_owner.encrypt_udm_chunks(
            plaintext_matrix=plaintext_matrix, algorithm="INVALID"
        )


def test_encrypt_threshold(data_owner):
    data_owner.messageIntervals = {"RANGE_0": (0, 10)}
    data_owner.cypherIntervals = {"RANGE_0": (0, 50)}

    encrypted = data_owner.encrypt_threshold(threshold=0.01)
    assert encrypted is not None


def test_encrypt_u_dbskmeans(data_owner, plaintext_matrix):
    udm = data_owner.calculate_UDM(plaintext_matrix=plaintext_matrix)
    data_owner.messageIntervals, data_owner.cypherIntervals = {
        "RANGE_0": (0, 5),
        "RANGE_1": (5, 20),
    }, {
        "RANGE_0": (0, 30),
        "RANGE_1": (30, 100),
    }
    encrypted_u = data_owner.encrypt_U(U=udm, algorithm="DBSKMEANS")
    assert encrypted_u is not None
    assert encrypted_u.shape == udm.shape


def test_encrypt_u_dbsnnc(data_owner, plaintext_matrix):
    dm = data_owner.calculate_DM(plaintext_matrix=plaintext_matrix)
    data_owner.messageIntervals, data_owner.cypherIntervals = {
        "RANGE_0": (0, 10),
        "RANGE_1": (10, 50),
    }, {
        "RANGE_0": (0, 50),
        "RANGE_1": (50, 200),
    }
    encrypted_u = data_owner.encrypt_U(U=dm, algorithm="DBSNNC")
    assert encrypted_u is not None
    assert encrypted_u.shape == dm.shape


def test_outsourced_data_skmeans(data_owner, plaintext_matrix):
    result = data_owner.outsourcedData(
        plaintext_matrix=plaintext_matrix,
        threshold=-1,
        algorithm="SKMEANS",
    )
    assert isinstance(result, OutsourcedDataResult)
    assert result.UDM is not None
    assert result.encrypted_matrix is not None


def test_outsourced_data_dbsnnc(data_owner, plaintext_matrix):
    result = data_owner.outsourcedData(
        plaintext_matrix=plaintext_matrix,
        threshold=5.0,
        algorithm="DBSNNC",
    )
    assert isinstance(result, OutsourcedDataResult)
    assert result.encrypted_threshold != 0
