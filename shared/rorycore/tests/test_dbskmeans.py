import numpy as np
import pytest
from rory.core.clustering.secure.conventional.dbskmeans import DBSKMeans


@pytest.fixture(scope="module")
def encrypted_data(liu_scheme):
    sk = liu_scheme.generate_secret_key()
    m = liu_scheme.m
    plaintext = np.array([
        [1.0, 2.0, 3.0],
        [1.5, 2.5, 3.5],
        [10.0, 11.0, 12.0],
        [10.5, 11.5, 12.5],
        [20.0, 21.0, 22.0],
    ], dtype=np.float64)

    encrypt_result = liu_scheme.encryptMatrix(
        plaintext_matrix=plaintext,
        secret_key=sk,
    )
    encrypted = encrypt_result.data

    n = plaintext.shape[0]
    a = plaintext.shape[1]
    udm = np.zeros((n, n, a))
    for i in range(n):
        for j in range(n):
            udm[i, j] = plaintext[i] - plaintext[j]

    return encrypted, udm, sk, m, a, n


def test_compute_centroid_shift(liu_scheme, encrypted_data):
    encrypted, udm, sk, m, a, n = encrypted_data

    prev_centroids = [[encrypted[0, col].tolist() for col in range(a)]]
    curr_centroids = [[encrypted[0, col].tolist() for col in range(a)]]

    dbs = DBSKMeans()
    shift = dbs.compute_centroid_shift(
        previous_centroids=prev_centroids,
        current_centroids=curr_centroids,
        k=1,
        a=a,
        m=m,
    )
    assert shift is not None
    assert shift.shape == (1, a, m)


def test_execute_plaintext_phase(encrypted_data):
    encrypted, udm, sk, m, a, n = encrypted_data
    k = 2

    shift_matrix = np.zeros((k, a))

    dbs = DBSKMeans()
    updated = dbs.execute_plaintext_phase(
        k=k,
        udm=udm,
        num_attributes=a,
        shift_matrix=shift_matrix,
    )

    assert updated.shape == (n, k, a)
