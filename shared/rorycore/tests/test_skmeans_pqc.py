import numpy as np
from Pyfhel import PyCtxt
from rory.core.clustering.secure.pqc.skmeans import Skmeans


def test_init(ckks_client):
    init_shift = ckks_client.encryptVector(np.zeros(4, dtype=np.float32))

    skmeans = Skmeans(scheme=ckks_client, init_shiftmatrix=init_shift)
    assert skmeans.scheme is not None
    assert skmeans.init_shiftmatrix is not None


def test_compute_centroid_shift(ckks_client):
    skmeans = Skmeans(scheme=ckks_client, init_shiftmatrix=ckks_client.encryptVector(np.zeros(4, dtype=np.float32)))

    plaintext = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ], dtype=np.float32)
    encrypted = ckks_client.encryptMatrix(plaintext_matrix=plaintext)

    prev_centroids = [encrypted[0], encrypted[1]]
    curr_centroids = [encrypted[1], encrypted[0]]

    shift = skmeans.compute_centroid_shift(
        previous_centroids=prev_centroids,
        current_centroids=curr_centroids,
        k=2,
    )
    assert shift is not None
    assert len(shift) == 2
    assert isinstance(shift[0], PyCtxt) or isinstance(shift[1], PyCtxt)


def test_compute_centroids(ckks_client):
    skmeans = Skmeans(scheme=ckks_client, init_shiftmatrix=ckks_client.encryptVector(np.zeros(4, dtype=np.float32)))

    plaintext = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ], dtype=np.float32)
    encrypted = ckks_client.encryptMatrix(plaintext_matrix=plaintext)

    clusters = [[encrypted[0], encrypted[1]]]

    result = skmeans.compute_centroids(clusters=clusters)
    assert result is not None


def test_execute_plaintext_phase(ckks_client):
    init_shift = ckks_client.encryptVector(np.zeros(4, dtype=np.float32))
    skmeans = Skmeans(scheme=ckks_client, init_shiftmatrix=init_shift)

    k = 2
    num_attributes = 2
    n = 4
    udm = np.zeros((n, k, num_attributes))
    shift_matrix = np.zeros((k, num_attributes))

    updated = skmeans.execute_plaintext_phase(
        k=k,
        udm=udm,
        num_attributes=num_attributes,
        shift_matrix=shift_matrix,
    )
    assert updated.shape == (n, k, num_attributes)
