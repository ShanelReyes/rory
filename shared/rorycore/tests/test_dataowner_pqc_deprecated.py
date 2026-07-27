import numpy as np
import pytest
from rory.core.security.pqc.dataowner_deprecated import DataOwner


@pytest.fixture(scope="module")
def data_owner(ckks_client):
    return DataOwner(scheme=ckks_client, sens=0.00001)


@pytest.fixture
def plaintext_matrix():
    return np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ], dtype=np.float32)


def test_init(data_owner):
    assert data_owner.sens == 0.00001
    assert data_owner.scheme is not None


def test_calculate_udm_diff(data_owner, plaintext_matrix):
    udm = data_owner.calculate_UDM(plaintext_matrix=plaintext_matrix, mode="diff")
    assert udm.shape == (3, 3, 2)
    assert np.allclose(udm[0, 0], [0, 0])


def test_calculate_udm_diff_abs(data_owner, plaintext_matrix):
    udm = data_owner.calculate_UDM(plaintext_matrix=plaintext_matrix, mode="diff-abs")
    assert udm.shape == (3, 3, 2)
    assert np.all(udm >= 0)


def test_get_u_skmeans_pqc(data_owner, plaintext_matrix):
    U = data_owner.get_U(plaintext_matrix=plaintext_matrix, algorithm="SKMEANS_PQC")
    assert U.shape == (3, 3, 2)


def test_get_u_dbskmeans_pqc(data_owner, plaintext_matrix):
    U = data_owner.get_U(plaintext_matrix=plaintext_matrix, algorithm="DBSKMEANS_PQC")
    assert U.shape == (3, 3, 2)
    assert len(data_owner.messageIntervals) > 0
    assert len(data_owner.cypherIntervals) > 0


def test_get_u_unknown(data_owner, plaintext_matrix):
    with pytest.raises(Exception):
        data_owner.get_U(plaintext_matrix=plaintext_matrix, algorithm="INVALID")


def test_ckks_encrypt_matrix_chunk(data_owner, plaintext_matrix):
    encrypted = data_owner.ckks_encrypt_matrix_chunk(plaintext_matrix=plaintext_matrix)
    assert encrypted is not None
    assert len(encrypted) == 3


def test_ckks_encrypt_encode_list_chunk(data_owner):
    chunk = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    encrypted = data_owner.ckks_encrypt_encode_list_chunk(plaintext_chunk=chunk)
    assert encrypted is not None
    assert len(encrypted) == 3


def test_ckks_encrypt_matrix_list_chunk(data_owner, plaintext_matrix):
    encrypted = data_owner.ckks_encrypt_matrix_list_chunk(plaintext_chunk=plaintext_matrix)
    assert encrypted is not None
    assert len(encrypted) == 3


def test_outsourced_data_skmeans_pqc(data_owner, plaintext_matrix):
    result = data_owner.outsourcedData(
        plaintext_matrix=plaintext_matrix,
        threshold=-1,
        algorithm="SKMEANS_PQC",
    )
    assert result is not None
    assert result.UDM is not None
    assert result.encrypted_matrix is not None
    assert result.num_attributes == 2


def test_outsourced_data_dbskmeans_pqc(data_owner, plaintext_matrix):
    result = data_owner.outsourcedData(
        plaintext_matrix=plaintext_matrix,
        threshold=-1,
        algorithm="DBSKMEANS_PQC",
    )
    assert result is not None
    assert result.encrypted_matrix is not None
