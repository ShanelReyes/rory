import numpy as np
import pytest
from rory.core.classification.knn import KNearestNeighbors


@pytest.fixture
def model_and_dataset():
    model = np.array([
        [2, 2, 3, 5, 4, 9, 1, 7, 7, 7, 4, 0],
        [4, 8, 1, 9, 9, 6, 6, 10, 9, 2, 7, 1],
        [7, 7, 8, 5, 1, 4, 9, 2, 3, 6, 10, 0],
        [7, 8, 8, 8, 6, 6, 2, 9, 7, 8, 3, 1],
        [3, 9, 10, 3, 5, 5, 8, 8, 10, 8, 5, 0],
        [5, 4, 2, 2, 10, 9, 9, 9, 10, 3, 8, 1],
        [9, 8, 9, 4, 6, 4, 5, 3, 8, 6, 6, 0]
    ], dtype=np.float64)

    dataset = np.array([
        [7, 8, 8, 8, 6, 6, 2, 9, 7, 8, 3],
        [3, 9, 10, 3, 5, 5, 8, 8, 10, 8, 5],
        [5, 4, 2, 2, 10, 9, 9, 9, 10, 3, 8],
    ], dtype=np.float64)

    return model, dataset


def test_split_labelvector_from_data(model_and_dataset):
    model, _ = model_and_dataset
    data, labels = KNearestNeighbors.split_labelvector_from_data(dataset=model)

    assert data.shape == (7, 11)
    assert labels.shape == (7,)
    assert labels.tolist() == [0, 1, 0, 1, 0, 1, 0]


def test_manhathan_distance():
    x1 = np.array([1.0, 2.0, 3.0])
    x2 = np.array([4.0, 5.0, 6.0])
    dist = KNearestNeighbors.manhathan_distance(x1, x2)
    assert dist == 9.0


def test_euclidean():
    x1 = np.array([0.0, 3.0])
    x2 = np.array([4.0, 0.0])
    dist = KNearestNeighbors.euclidean(x1, x2)
    assert dist == 25.0


def test_get_distance_manhathan():
    x1 = np.array([1.0, 2.0])
    x2 = np.array([4.0, 6.0])
    dist = KNearestNeighbors.get_distance(x1, x2, distance="MANHATHAN")
    assert dist == 7.0


def test_get_distance_euclidean():
    x1 = np.array([0.0, 3.0])
    x2 = np.array([4.0, 0.0])
    dist = KNearestNeighbors.get_distance(x1, x2, distance="EUCLIDEAN")
    assert dist == 25.0


def test_calculate_distances_and_indexes(model_and_dataset):
    model, dataset = model_and_dataset
    model_data, _ = KNearestNeighbors.split_labelvector_from_data(dataset=model)

    min_indexes = KNearestNeighbors.calculate_distances_and_indexes(
        model=model_data,
        dataset=dataset,
        distance="MANHATHAN"
    )

    assert len(min_indexes) == 3
    assert all(0 <= idx < len(model_data) for idx in min_indexes)


def test_predict(model_and_dataset):
    model, dataset = model_and_dataset
    model_data, model_labels = KNearestNeighbors.split_labelvector_from_data(dataset=model)

    predictions = KNearestNeighbors.predict(
        dataset=dataset,
        model=model_data,
        model_labels=model_labels,
        distance="MANHATHAN"
    )

    assert len(predictions) == 3
    assert all(label in (0, 1) for label in predictions)


def test_predict_returns_correct_shape(model_and_dataset):
    model, dataset = model_and_dataset
    model_data, model_labels = KNearestNeighbors.split_labelvector_from_data(dataset=model)

    predictions = KNearestNeighbors.predict(
        dataset=dataset,
        model=model_data,
        model_labels=model_labels,
        distance="MANHATHAN"
    )

    assert predictions.shape == (dataset.shape[0],)


def test_fit_noop():
    model = np.array([[1.0, 2.0], [3.0, 4.0]])
    labels = np.array([0, 1])
    result = KNearestNeighbors.fit(model=model, model_labels=labels)
    assert result is None
