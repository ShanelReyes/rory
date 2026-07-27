import numpy as np
import pytest
from rory.core.clustering.kmeans import KMeans
from rory.core.interfaces.rory_result import RoryResult


@pytest.fixture
def synthetic_data():
    rng = np.random.RandomState(42)
    X = rng.randn(30, 5).astype(np.float64)
    return X


def test_kmeans_default_k(synthetic_data):
    result = KMeans().fit(plaintext_matrix=synthetic_data)

    assert isinstance(result, RoryResult)
    assert len(result.label_vector) == synthetic_data.shape[0]
    assert result.n_iterations > 0
    assert result.response_time > 0
    assert result.service_time > 0
    assert len(set(result.label_vector)) == 2


def test_kmeans_k_3(synthetic_data):
    result = KMeans().fit(plaintext_matrix=synthetic_data, k=3)

    assert len(result.label_vector) == synthetic_data.shape[0]
    assert result.n_iterations > 0
    label_set = set(result.label_vector)
    assert len(label_set) == 3


def test_kmeans_k_4(synthetic_data):
    result = KMeans().fit(plaintext_matrix=synthetic_data, k=4)

    assert len(result.label_vector) == synthetic_data.shape[0]
    label_set = set(result.label_vector)
    assert len(label_set) == 4


def test_kmeans_deterministic_with_seed():
    rng = np.random.RandomState(99)
    X = rng.randn(20, 3).astype(np.float64)

    result1 = KMeans().fit(plaintext_matrix=X, k=2)
    result2 = KMeans().fit(plaintext_matrix=X, k=2)

    assert np.array_equal(result1.label_vector, result2.label_vector)
    assert result1.n_iterations == result2.n_iterations


def test_kmeans_service_time_less_than_response(synthetic_data):
    result = KMeans().fit(plaintext_matrix=synthetic_data, k=3)

    assert result.service_time <= result.response_time
