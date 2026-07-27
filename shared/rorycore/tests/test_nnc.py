import numpy as np
import pytest
from rory.core.clustering.nnc import Nnc
from rory.core.utils.utils import Utils
from rory.core.interfaces.rory_result import RoryResult


@pytest.fixture
def sample_distance_matrix():
    data = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 1.0, 4.0],
        [3.0, 4.0, 1.0],
    ])
    distances = np.zeros((3, 3))
    n = data.shape[0]
    for i in range(n):
        for j in range(n):
            distances[i, j] = np.sum(np.abs(data[i] - data[j]))
    return distances


def test_get_min_distance_in_clusters(sample_distance_matrix):
    c_indexes = [[0]]
    record_index = 1
    cluster_index, delta = Utils.getMinDistanceInClusters(
        c_indexes=c_indexes,
        record_index=record_index,
        distance_matrix=sample_distance_matrix
    )
    assert cluster_index == 0
    assert delta > 0


def test_get_min_distance_multiple_clusters(sample_distance_matrix):
    c_indexes = [[0], [1]]
    record_index = 2
    cluster_index, delta = Utils.getMinDistanceInClusters(
        c_indexes=c_indexes,
        record_index=record_index,
        distance_matrix=sample_distance_matrix
    )
    assert cluster_index in (0, 1)
    assert delta > 0


def test_nnc_fit(sample_distance_matrix):
    threshold = 5.0
    result = Nnc().fit(
        distance_matrix=sample_distance_matrix,
        threshold=threshold
    )

    assert isinstance(result, RoryResult)
    assert len(result.label_vector) == 3
    assert result.response_time > 0


def test_nnc_fit_high_threshold_single_cluster(sample_distance_matrix):
    threshold = 999.0
    result = Nnc().fit(
        distance_matrix=sample_distance_matrix,
        threshold=threshold
    )

    assert len(set(result.label_vector)) == 1
    assert result.label_vector[0] == 0


def test_nnc_fit_low_threshold_separate_clusters():
    data = np.array([
        [0.0, 0.0],
        [10.0, 10.0],
        [0.1, 0.1],
    ])
    n = data.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = np.linalg.norm(data[i] - data[j])

    threshold = 2.0
    result = Nnc().fit(
        distance_matrix=distances,
        threshold=threshold
    )

    labels = result.label_vector
    assert labels[0] == labels[2]
    assert labels[1] != labels[0]
