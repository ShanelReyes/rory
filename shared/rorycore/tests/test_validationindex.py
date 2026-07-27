import numpy as np
import pytest
from rory.core.validation_index.validationindex_deprecated import (
    delta,
    big_delta,
    dunn,
    delta_fast,
    big_delta_fast,
    dunn_fast,
    big_s,
    davisbouldin,
)
from sklearn.metrics.pairwise import euclidean_distances


@pytest.fixture
def cluster_list():
    c0 = np.array([[0.0, 0.0], [0.2, 0.1], [0.1, 0.3]])
    c1 = np.array([[5.0, 5.0], [5.2, 5.1], [4.9, 5.0]])
    c2 = np.array([[10.0, 10.0], [10.1, 9.9]])
    return [c0, c1, c2]


@pytest.fixture
def points_and_labels(cluster_list):
    points = np.vstack(cluster_list)
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    return points, labels


def test_delta(cluster_list):
    result = delta(cluster_list[0], cluster_list[1])
    assert result > 0
    assert result < 10.0


def test_big_delta(cluster_list):
    result = big_delta(cluster_list[0])
    assert result > 0
    assert result < 1.0


def test_dunn(cluster_list):
    result = dunn(cluster_list)
    assert result > 0


def test_delta_fast(points_and_labels):
    points, labels = points_and_labels
    distances = euclidean_distances(points)
    mask0 = labels == 0
    mask1 = labels == 1
    result = delta_fast(mask0, mask1, distances)
    assert result > 0
    assert result < 10.0


def test_big_delta_fast(points_and_labels):
    points, labels = points_and_labels
    distances = euclidean_distances(points)
    mask0 = labels == 0
    result = big_delta_fast(mask0, distances)
    assert result > 0


def test_dunn_fast(points_and_labels):
    points, labels = points_and_labels
    result = dunn_fast(points, labels)
    assert result > 0


def test_big_s(cluster_list):
    center = np.mean(cluster_list[0], axis=0)
    result = big_s(cluster_list[0], center)
    assert result > 0


def test_davisbouldin(cluster_list):
    centers = np.array([np.mean(c, axis=0) for c in cluster_list])
    result = davisbouldin(cluster_list, centers)
    assert result > 0


def test_dunn_well_separated_clusters():
    c0 = np.array([[0.0, 0.0], [0.1, 0.1]])
    c1 = np.array([[100.0, 100.0], [100.1, 99.9]])
    result = dunn([c0, c1])
    assert result > 1.0


def test_davisbouldin_identical_clusters():
    c0 = np.array([[0.0, 0.0], [0.1, 0.1]])
    c1 = np.array([[0.0, 0.0], [0.1, 0.1]])
    centers = np.array([[0.05, 0.05], [0.05, 0.05]])
    result = davisbouldin([c0, c1], centers)
    assert result >= 0
