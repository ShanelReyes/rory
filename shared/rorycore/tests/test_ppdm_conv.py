import pytest
import numpy as np
from rory.core.enums import Algorithm, Scheme
from rory.core.security.dataowner import DataOwner
from rory.core.security.scheme_params import LiuParams, LiuAndFdhopeParams
from rory.core.clustering.kmeans import KMeans
from rory.core.clustering.secure.conventional.skmeans import SKMeans
from rory.core.clustering.secure.conventional.dbskmeans import DBSKMeans
from rory.core.clustering.secure.conventional.dbsnnc import Dbsnnc
from rory.core.clustering.nnc import Nnc
from rory.core.security.cryptosystem.liu import Liu
from rory.core.utils.utils import Utils
from rory.core.utils.constants import Constants

PLAINTEXT_MATRIX = [
    [0.73, 8.84],
    [49.93, 34.44],
    [0.57, 65.04],
    [62.15, 32.29],
    [59.47, 36.04]
]

LIU_PARAMS = LiuParams(
    security_level=128, _round=True, decimals=6, seed=1,
)

LIU_FDHOPE_PARAMS = LiuAndFdhopeParams(
    security_level=128, _round=True, decimals=6, seed=1,
)


@pytest.fixture(scope="module")
def liu_scheme():
    return Liu(
        _round=True,
        decimals=6,
        secure_random=False,
        seed=1,
        use_np_random=False,
        security_level=128
    )


@pytest.fixture(scope="module")
def secret_key(liu_scheme):
    return liu_scheme.generate_secret_key()


def test_liu_operations(liu_scheme, secret_key):
    v1, v2 = 4, 10
    e1 = liu_scheme.encryptScalar(plaintext=v1, secret_key=secret_key).data
    e2 = liu_scheme.encryptScalar(plaintext=v2, secret_key=secret_key).data

    e3 = Liu.add(ciphertext_1=e1, ciphertext_2=e2)
    v3 = np.around(liu_scheme.decryptScalar(ciphertext=e3, secret_key=secret_key).data)

    assert v3 == (v1 + v2)


def test_skmeans_fit():
    do = DataOwner.with_algorithm(Algorithm.SKMEANS) \
        .with_scheme(Scheme.LIU) \
        .with_scheme_params(LIU_PARAMS) \
        .build()
    outsourced = do.outsourcedData(plaintext_matrix=np.array(PLAINTEXT_MATRIX))

    liu = do.primary_scheme
    skmeans = SKMeans()
    label_vector = skmeans.fit(
        status=Constants.ClusteringStatus.START,
        k=3,
        encrypted_matrix=outsourced.encrypted_matrix,
        UDM=outsourced.UDM,
        Cent_j=None,
        iterations=0,
        n_iterations=6,
        num_attributes=2,
        scheme=liu,
        sk=liu.sk,
        m=liu.m,
    )
    assert label_vector is not None
    assert len(label_vector) == len(PLAINTEXT_MATRIX)


def test_kmeans_baseline():
    result = KMeans().fit(plaintext_matrix=PLAINTEXT_MATRIX)
    assert hasattr(result, 'label_vector')
    assert len(result.label_vector) == len(PLAINTEXT_MATRIX)
    assert all(isinstance(label, (int, np.integer)) for label in result.label_vector)


def test_skmeans_loop():
    k = 2
    do = DataOwner.with_algorithm(Algorithm.SKMEANS) \
        .with_scheme(Scheme.LIU) \
        .with_scheme_params(LIU_PARAMS) \
        .build()
    outsourced = do.outsourcedData(plaintext_matrix=np.array(PLAINTEXT_MATRIX))

    liu = do.primary_scheme
    skmeans = SKMeans()
    num_attributes = len(PLAINTEXT_MATRIX[0])
    status = Constants.ClusteringStatus.START
    cent_j = None
    udm = outsourced.UDM
    max_iters = 10
    current_iter = 0

    while status != Constants.ClusteringStatus.COMPLETED and current_iter < max_iters:
        result1 = skmeans.execute_encrypted_phase(
            status=status,
            k=k,
            encrypted_matrix=outsourced.encrypted_matrix,
            udm=udm,
            num_attributes=num_attributes,
            m=liu.m,
            centroids=cent_j,
        )

        if not result1.is_ok:
            pytest.fail(f"Falla en la iteracion {current_iter}: {result1.unwrap_err()}")

        s1, cent_i, _cent_j, label_vector = result1.unwrap()
        shift_matrix_dec = liu.decryptMatrix(ciphertext_matrix=s1, secret_key=liu.sk)

        udm = skmeans.execute_plaintext_phase(
            k=k,
            udm=outsourced.UDM,
            num_attributes=num_attributes,
            shift_matrix=np.array(shift_matrix_dec.data),
        )

        if Utils.verifyZero(shift_matrix_dec.data):
            status = Constants.ClusteringStatus.COMPLETED
        else:
            status = Constants.ClusteringStatus.WORK_IN_PROGRESS
            cent_j = _cent_j

        current_iter += 1

    assert status == Constants.ClusteringStatus.COMPLETED or current_iter == max_iters
    assert len(label_vector) == len(PLAINTEXT_MATRIX)


def test_dbsnnc():
    do = DataOwner.with_algorithm(Algorithm.DBSNNC) \
        .with_scheme(Scheme.LIU_AND_FDHOPE) \
        .with_scheme_params(LIU_FDHOPE_PARAMS) \
        .build()
    outsourced = do.outsourcedData(
        plaintext_matrix=np.array(PLAINTEXT_MATRIX),
        threshold=1,
    )
    dbsnnc_res = Dbsnnc().fit(
        distance_matrix=outsourced.UDM,
        threshold=outsourced.encrypted_threshold,
    )
    assert dbsnnc_res.label_vector is not None
    assert len(dbsnnc_res.label_vector) == len(PLAINTEXT_MATRIX)


def test_nnc():
    do = DataOwner.with_algorithm(Algorithm.NNC) \
        .with_scheme(Scheme.NONE) \
        .with_scheme_params(None) \
        .build()
    outsourced = do.outsourcedData(
        plaintext_matrix=np.array(PLAINTEXT_MATRIX),
        threshold=1,
    )
    nnc_res = Nnc().fit(
        distance_matrix=outsourced.UDM,
        threshold=outsourced.encrypted_threshold,
    )
    assert nnc_res.label_vector is not None
    assert len(nnc_res.label_vector) == len(PLAINTEXT_MATRIX)


def test_dbskmeans_fit():
    do = DataOwner.with_algorithm(Algorithm.DBSKMEANS) \
        .with_scheme(Scheme.LIU_AND_FDHOPE) \
        .with_scheme_params(LIU_FDHOPE_PARAMS) \
        .build()
    outsourced = do.outsourcedData(plaintext_matrix=np.array(PLAINTEXT_MATRIX))

    liu = do.primary_scheme
    dbskmeans = DBSKMeans()
    label_vector = dbskmeans.fit(
        status=Constants.ClusteringStatus.START,
        k=3,
        encrypted_matrix=outsourced.encrypted_matrix,
        UDM=outsourced.UDM,
        Cent_j=None,
        iterations=0,
        n_iterations=6,
        num_attributes=2,
        scheme=liu,
        sk=liu.sk,
        m=liu.m,
    )
    assert label_vector is not None
    assert len(label_vector) == len(PLAINTEXT_MATRIX)
