import numpy as np
import pytest
from rory.core.classification.secure.conventional.sknn import SecureKNearestNeighbors
from rory.core.utils.utils import Utils


@pytest.fixture(scope="module")
def encrypted_model_and_dataset(liu_scheme):
    sk = liu_scheme.generate_secret_key()

    model_labels = np.array([0, 1, 0], dtype=np.float64)

    model = np.array([
        [1.0, 2.0, 0.0],
        [5.0, 6.0, 1.0],
        [2.0, 3.0, 0.0],
    ], dtype=np.float64)

    dataset = np.array([
        [1.5, 2.5],
        [5.5, 6.5],
    ], dtype=np.float64)

    enc_model = liu_scheme.encryptMatrix(
        plaintext_matrix=model[:, :2], secret_key=sk
    ).data
    enc_dataset = liu_scheme.encryptMatrix(
        plaintext_matrix=dataset, secret_key=sk
    ).data

    return enc_model, model_labels, enc_dataset


def test_split_labelvector_from_data(liu_scheme):
    dataset = np.array([
        [1.0, 2.0, 0.0],
        [3.0, 4.0, 1.0],
    ], dtype=np.float64)
    data, labels = SecureKNearestNeighbors.split_labelvector_from_data(dataset=dataset)
    assert data.shape == (2, 2)
    assert labels.shape == (2,)
    assert labels.tolist() == [0.0, 1.0]


def test_fit_returns_none():
    model = np.array([[1.0], [2.0]])
    labels = np.array([0, 1])
    result = SecureKNearestNeighbors.fit(model=model, model_labels=labels)
    assert result is None


def test_get_label_vector():
    model_labels = np.array([10, 20, 30])
    min_indexes = np.array([0, 2, 1])
    result = Utils.get_label_vector(
        model_labels=model_labels,
        min_indexes=min_indexes,
    )
    assert result.tolist() == [10, 30, 20]


def test_manhathan_distance(liu_scheme):
    sk = liu_scheme.generate_secret_key()

    x1 = liu_scheme.encryptScalar(plaintext=5.0, secret_key=sk).data
    x2 = liu_scheme.encryptScalar(plaintext=2.0, secret_key=sk).data
    acum = np.zeros(liu_scheme.m).tolist()

    result = SecureKNearestNeighbors.manhathan_distance(x1=x1, x2=x2, acum=acum)
    assert result is not None


def test_euclidean(liu_scheme):
    sk = liu_scheme.generate_secret_key()

    x1 = liu_scheme.encryptScalar(plaintext=4.0, secret_key=sk).data
    x2 = liu_scheme.encryptScalar(plaintext=2.0, secret_key=sk).data
    acum = np.zeros(liu_scheme.m).tolist()

    result = SecureKNearestNeighbors.euclidean(x1=x1, x2=x2, acum=acum)
    assert result is not None


def test_get_distance_manhathan(liu_scheme):
    sk = liu_scheme.generate_secret_key()

    x1 = liu_scheme.encryptScalar(plaintext=5.0, secret_key=sk).data
    x2 = liu_scheme.encryptScalar(plaintext=3.0, secret_key=sk).data
    acum = np.zeros(liu_scheme.m).tolist()

    result = SecureKNearestNeighbors.get_distance(
        x1=x1, x2=x2, acum=acum, distance="MANHATHAN"
    )
    assert result is not None


def test_calculate_distances(liu_scheme, encrypted_model_and_dataset):
    enc_model, model_labels, enc_dataset = encrypted_model_and_dataset

    distances = SecureKNearestNeighbors.calculate_distances(
        model=enc_model,
        dataset=enc_dataset,
        distance="MANHATHAN",
    )

    assert distances is not None
    assert distances.shape == (2, 3, liu_scheme.m)


def test_predict(liu_scheme, encrypted_model_and_dataset):
    enc_model, model_labels, enc_dataset = encrypted_model_and_dataset
    sk = liu_scheme.generate_secret_key()

    predictions = SecureKNearestNeighbors.predict(
        dataset=enc_dataset, model=enc_model,
        model_labels=model_labels, distance="MANHATHAN",
        scheme=liu_scheme, sk=sk,
    )
    assert predictions is not None
    assert len(predictions) == 2
    assert all(label in model_labels for label in predictions)


def test_get_distance_euclidean(liu_scheme):
    sk = liu_scheme.generate_secret_key()
    x1 = liu_scheme.encryptScalar(plaintext=4.0, secret_key=sk).data
    x2 = liu_scheme.encryptScalar(plaintext=1.0, secret_key=sk).data
    acum = np.zeros(liu_scheme.m).tolist()

    result = SecureKNearestNeighbors.get_distance(
        x1=x1, x2=x2, acum=acum, distance="EUCLIDEAN"
    )
    assert result is not None
