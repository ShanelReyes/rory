import numpy as np
from rory.core.classification.secure.pqc.sknn import SecureKNearestNeighbors
from rory.core.utils.utils import Utils


def test_split_labelvector_from_data():
    dataset = np.array([
        [1.0, 2.0, 0.0],
        [3.0, 4.0, 1.0],
        [5.0, 6.0, 0.0],
    ], dtype=np.float64)
    data, labels = SecureKNearestNeighbors.split_labelvector_from_data(dataset=dataset)
    assert data.shape == (3, 2)
    assert labels.shape == (3,)
    assert labels.tolist() == [0.0, 1.0, 0.0]


def test_get_label_vector():
    model_labels = np.array([10, 20, 30])
    min_indexes = np.array([0, 2, 1])
    result = Utils.get_label_vector(
        model_labels=model_labels,
        min_indexes=min_indexes,
    )
    assert result.tolist() == [10, 30, 20]


def test_fit_returns_none(ckks_client):
    model_plain = np.array([[1.0], [2.0]], dtype=np.float32)
    model_enc = ckks_client.encryptMatrix(plaintext_matrix=model_plain)
    labels = np.array([0, 1])

    result = SecureKNearestNeighbors.fit(model=model_enc, model_labels=labels)
    assert result is None


def test_euclidean(ckks_client):
    model_plain = np.array([[1.0, 2.0]], dtype=np.float32)
    model_enc = ckks_client.encryptMatrix(plaintext_matrix=model_plain)
    x1 = model_enc[0]

    dataset_plain = np.array([[0.0, 0.0]], dtype=np.float32)
    dataset_enc = ckks_client.encryptMatrix(plaintext_matrix=dataset_plain)
    x2 = dataset_enc[0]

    result = SecureKNearestNeighbors.euclidean(x1, x2)
    assert result is not None


def test_manhathan_distance(ckks_client):
    model_plain = np.array([[3.0, 4.0]], dtype=np.float32)
    model_enc = ckks_client.encryptMatrix(plaintext_matrix=model_plain)
    x1 = model_enc[0]

    dataset_plain = np.array([[1.0, 2.0]], dtype=np.float32)
    dataset_enc = ckks_client.encryptMatrix(plaintext_matrix=dataset_plain)
    x2 = dataset_enc[0]

    result = SecureKNearestNeighbors.manhathan_distance(x1, x2)
    assert result is not None


def test_get_distance(ckks_client):
    model_plain = np.array([[1.0, 2.0]], dtype=np.float32)
    model_enc = ckks_client.encryptMatrix(plaintext_matrix=model_plain)
    x1 = model_enc[0]

    dataset_plain = np.array([[0.0, 0.0]], dtype=np.float32)
    dataset_enc = ckks_client.encryptMatrix(plaintext_matrix=dataset_plain)
    x2 = dataset_enc[0]

    result = SecureKNearestNeighbors.get_distance(x1, x2, distance="EUCLIDEAN")
    assert result is not None


def test_calculate_distances(ckks_client):
    model_plain = np.array([
        [1.0, 2.0],
        [5.0, 6.0],
    ], dtype=np.float32)
    dataset_plain = np.array([
        [1.5, 2.5],
    ], dtype=np.float32)

    model_enc = ckks_client.encryptMatrix(plaintext_matrix=model_plain)
    dataset_enc = ckks_client.encryptMatrix(plaintext_matrix=dataset_plain)

    distances = SecureKNearestNeighbors.calculate_distances(
        model=model_enc,
        dataset=dataset_enc,
        model_shape=(2, 2),
        dataset_shape=(1, 2),
    )
    assert distances is not None
    assert distances.shape == (1, 2)


def test_predict(ckks_client):
    model_plain = np.array([
        [1.0, 2.0],
        [5.0, 6.0],
    ], dtype=np.float32)
    dataset_plain = np.array([
        [1.5, 2.5],
        [5.5, 6.5],
    ], dtype=np.float32)
    model_labels = np.array([0, 1], dtype=np.float64)

    model_enc = ckks_client.encryptMatrix(plaintext_matrix=model_plain)
    dataset_enc = ckks_client.encryptMatrix(plaintext_matrix=dataset_plain)

    predictions = SecureKNearestNeighbors.predict(
        dataset=dataset_enc, model=model_enc,
        model_labels=model_labels, distance="EUCLIDEAN",
        model_shape=(2, 2), dataset_shape=(2, 2),
        scheme=ckks_client,
    )
    assert predictions is not None
    assert len(predictions) == 2
