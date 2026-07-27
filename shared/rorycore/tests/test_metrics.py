import numpy as np
import pytest
from rory.core.validation_index.metrics_deprecated import internal_validation_indexes, external_validation_indexes
from rory.core.interfaces.metricsResult_internal import MetricsResultInternal
from rory.core.interfaces.metricsResult_external import MetricsResultExternal


@pytest.fixture
def synthetic_data():
    rng = np.random.RandomState(42)
    X = rng.randn(30, 5).astype(np.float64)
    y = rng.randint(0, 3, size=30)
    return X, y


def test_internal_validation_indexes(synthetic_data):
    X, y = synthetic_data
    result = internal_validation_indexes(plaintext_matrix=X, target=y)

    assert isinstance(result, MetricsResultInternal)
    assert isinstance(result.silhouette_coefficient, float)
    assert isinstance(result.davies_bouldin_index, float)
    assert isinstance(result.calinski_harabaz_index, float)
    assert isinstance(result.dunn_index, (int, float))
    assert -1 <= result.silhouette_coefficient <= 1
    assert result.davies_bouldin_index >= 0


def test_external_validation_indexes_binary(synthetic_data):
    X, y = synthetic_data
    y_binary = (y > 0).astype(int)
    pred = np.roll(y_binary, 1)

    result = external_validation_indexes(pred=pred, target=y_binary, k=2)

    assert isinstance(result, MetricsResultExternal)
    assert isinstance(result.adjusted_mutual_information, float)
    assert isinstance(result.fowlkes_mallows_index, float)
    assert isinstance(result.adjusted_rand_index, float)
    assert isinstance(result.jaccard_index, float)


def test_external_validation_indexes_multiclass(synthetic_data):
    X, y = synthetic_data
    pred = np.roll(y, 1)

    result = external_validation_indexes(pred=pred, target=y, k=3)

    assert isinstance(result, MetricsResultExternal)
    assert result.adjusted_rand_index <= 1.0
    assert result.adjusted_rand_index >= -1.0


def test_internal_validation_properties(synthetic_data):
    X, y = synthetic_data
    result = internal_validation_indexes(plaintext_matrix=X, target=y)

    assert result.__str__() is not None
    assert len(result.__str__()) > 0


def test_external_validation_properties(synthetic_data):
    X, y = synthetic_data
    y_binary = (y > 0).astype(int)
    pred = np.roll(y_binary, 1)

    result = external_validation_indexes(pred=pred, target=y_binary, k=2)

    assert result.__str__() is not None
    assert len(result.__str__()) > 0
