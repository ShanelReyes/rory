import numpy as np
import pytest
from rory.core.enums import Algorithm, Scheme
from rory.core.security.dataowner import DataOwner
from rory.core.security.scheme_params import LiuParams
from rory.core.security.cryptosystem.liu import Liu
from rory.core.clustering.secure.conventional.skmeans import SKMeans
from rory.core.utils.constants import Constants

PLAINTEXT_MATRIX = [
    [1.0, 2.0, 3.0],
    [1.5, 2.5, 3.5],
    [10.0, 11.0, 12.0],
    [10.5, 11.5, 12.5],
    [20.0, 21.0, 22.0],
]


@pytest.fixture(scope="module")
def liu_scheme():
    return Liu(
        _round=True,
        decimals=6,
        secure_random=False,
        seed=1,
        use_np_random=False,
        security_level=128,
    )


@pytest.fixture(scope="module")
def secret_key(liu_scheme):
    return liu_scheme.generate_secret_key()


@pytest.fixture(scope="module")
def encrypted_data(liu_scheme):
    sk = liu_scheme.generate_secret_key()
    m = liu_scheme.m
    plaintext = np.array(PLAINTEXT_MATRIX, dtype=np.float64)

    encrypt_result = liu_scheme.encryptMatrix(
        plaintext_matrix=plaintext, secret_key=sk,
    )
    encrypted = encrypt_result.data

    n = plaintext.shape[0]
    a = plaintext.shape[1]
    udm = np.zeros((n, n, a))
    for i in range(n):
        for j in range(n):
            udm[i, j] = plaintext[i] - plaintext[j]

    return encrypted, udm, sk, m, a, n


def test_execute_encrypted_phase_start(liu_scheme, encrypted_data):
    encrypted, udm, sk, m, a, n = encrypted_data
    k = 2

    skmeans = SKMeans()
    result = skmeans.execute_encrypted_phase(
        status=Constants.ClusteringStatus.START,
        k=k,
        encrypted_matrix=encrypted,
        udm=udm,
        num_attributes=a,
        m=m,
    )

    assert result.is_ok
    shift, prev_centroids, new_centroids, label_vector = result.unwrap()
    assert shift is not None
    assert shift.shape == (k, a, m)
    assert prev_centroids is not None
    assert new_centroids is not None
    assert len(label_vector) == n


def test_execute_encrypted_phase_work_in_progress(liu_scheme, encrypted_data):
    encrypted, udm, sk, m, a, n = encrypted_data
    k = 2

    skmeans = SKMeans()
    result1 = skmeans.execute_encrypted_phase(
        status=Constants.ClusteringStatus.START,
        k=k,
        encrypted_matrix=encrypted,
        udm=udm,
        num_attributes=a,
        m=m,
    )
    assert result1.is_ok
    _, _, centroids, _ = result1.unwrap()

    result2 = skmeans.execute_encrypted_phase(
        status=Constants.ClusteringStatus.WORK_IN_PROGRESS,
        k=k,
        encrypted_matrix=encrypted,
        udm=udm,
        num_attributes=a,
        m=m,
        centroids=centroids,
    )

    assert result2.is_ok
    shift, prev_centroids, new_centroids, label_vector = result2.unwrap()
    assert shift is not None
    assert shift.shape == (k, a, m)
    assert len(label_vector) == n


def test_execute_plaintext_phase(encrypted_data):
    encrypted, udm, sk, m, a, n = encrypted_data
    k = 2

    shift_matrix = np.zeros((k, a))

    skmeans = SKMeans()
    updated = skmeans.execute_plaintext_phase(
        k=k,
        udm=udm,
        num_attributes=a,
        shift_matrix=shift_matrix,
    )

    assert updated.shape == (n, k, a)


def test_compute_centroid_shift(liu_scheme, encrypted_data):
    encrypted, udm, sk, m, a, n = encrypted_data
    k = 2

    prev = np.array([encrypted[0], encrypted[1]])
    curr = np.array([encrypted[1], encrypted[0]])

    skmeans = SKMeans()
    shift = skmeans.compute_centroid_shift(
        previous_centroids=prev,
        current_centroids=curr,
        k=k,
        a=a,
        m=m,
    )

    assert shift is not None
    assert shift.shape == (k, a, m)


def test_fit(liu_scheme, secret_key):
    plaintext = np.array([
        [0.73, 8.84],
        [49.93, 34.44],
        [0.57, 65.04],
        [62.15, 32.29],
        [59.47, 36.04],
    ])

    dow0 = DataOwner.with_algorithm(Algorithm.SKMEANS) \
        .with_scheme(Scheme.LIU) \
        .with_scheme_params(LiuParams(
            security_level=128, _round=True, decimals=6, seed=1,
        )) \
        .build()
    outsourced = dow0.outsourcedData(
        plaintext_matrix=plaintext,
    )

    liu = dow0.primary_scheme
    skmeans = SKMeans()
    label_vector = skmeans.fit(
        status=Constants.ClusteringStatus.START,
        k=2,
        encrypted_matrix=outsourced.encrypted_matrix,
        UDM=outsourced.UDM,
        Cent_j=None,
        iterations=0,
        n_iterations=10,
        num_attributes=2,
        scheme=liu,
        sk=liu.sk,
        m=liu.m,
    )

    assert label_vector is not None
    assert len(label_vector) == len(plaintext)


def test_fit_manual_loop(liu_scheme, secret_key):
    plaintext = np.array([
        [0.73, 8.84],
        [49.93, 34.44],
        [0.57, 65.04],
        [62.15, 32.29],
        [59.47, 36.04],
    ])

    dow0 = DataOwner.with_algorithm(Algorithm.SKMEANS) \
        .with_scheme(Scheme.LIU) \
        .with_scheme_params(LiuParams(
            security_level=128, _round=True, decimals=6, seed=1,
        )) \
        .build()
    outsourced = dow0.outsourcedData(
        plaintext_matrix=plaintext,
    )

    liu = dow0.primary_scheme
    skmeans = SKMeans()
    k = 2
    a = len(plaintext[0])
    status = Constants.ClusteringStatus.START
    cent_j = None
    udm = outsourced.UDM
    max_iters = 10
    current_iter = 0
    label_vector = []

    while status != Constants.ClusteringStatus.COMPLETED and current_iter < max_iters:
        result = skmeans.execute_encrypted_phase(
            status=status,
            k=k,
            encrypted_matrix=outsourced.encrypted_matrix,
            udm=udm,
            num_attributes=a,
            m=liu.m,
            centroids=cent_j,
        )

        if not result.is_ok:
            pytest.fail(f"Failed at iteration {current_iter}: {result.unwrap_err()}")

        s1, _, _cent_j, label_vector = result.unwrap()
        dec_shift = liu.decryptMatrix(
            ciphertext_matrix=s1, secret_key=liu.sk,
        )

        udm = skmeans.execute_plaintext_phase(
            k=k,
            udm=udm,
            num_attributes=a,
            shift_matrix=np.array(dec_shift.data),
        )

        if np.max(np.abs(dec_shift.data)) <= 0.000001:
            status = Constants.ClusteringStatus.COMPLETED
        else:
            status = Constants.ClusteringStatus.WORK_IN_PROGRESS
            cent_j = _cent_j

        current_iter += 1

    assert status == Constants.ClusteringStatus.COMPLETED or current_iter == max_iters
    assert len(label_vector) == len(plaintext)
