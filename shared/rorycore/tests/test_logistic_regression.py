import time
import numpy as np
import os
import pytest
from rory.core.classification.logistic_regression import LogisticRegression





def test_experimentation(setup_paths, create_label_vector, create_datasets):
    (_, labelvector_path) = create_label_vector
    (dataset_train, dataset_train_path), (dataset_test, dataset_test_path) = create_datasets

    epochs = 3
    learning_rate = 0.1

    lr_result = logistic_regression_completed(
        source_path            = setup_paths["source_path"],
        dataset_train_filename = dataset_train_path,
        dataset_test_filename  = dataset_test_path,
        labelvector_filename   = labelvector_path,
        epochs                 = epochs,
        learning_rate          = learning_rate,
    )
    print(lr_result)


def logistic_regression_completed(
    source_path,
    dataset_train_filename,
    dataset_test_filename,
    labelvector_filename,
    epochs,
    learning_rate,
):
    start_time = time.time()
    plain_dataset_train = np.load(os.path.join(source_path, dataset_train_filename))
    plain_labelvector_train = np.load(
        os.path.join(source_path, labelvector_filename)
    )
    plain_dataset_test = np.load(os.path.join(source_path, dataset_test_filename))
    bias = 0.0
    weights_matrix = np.zeros(plain_dataset_train.shape[1], dtype=np.float32)

    start_train_time = time.time()
    weights, bias = LogisticRegression.fit(
        plaintext_matrix=plain_dataset_train,
        label_vector=plain_labelvector_train,
        epochs=epochs,
        learning_rate=learning_rate,
        bias=bias,
        weights=weights_matrix,
    )
    end_train_time = time.time() - start_train_time

    time.time()
    label_vector = LogisticRegression.predict(
        plaintext_matrix=plain_dataset_test,
        weights=weights,
        bias=bias,
    )
    end_predict_time = time.time() - start_train_time

    end_time = time.time() - start_time
    return {
        "label_vector": label_vector,
        "service_time": end_time,
        "training_time": end_train_time,
        "predict_time": end_predict_time,
    }


def test_sigmoid_poly_range():
    z = np.linspace(-1.5, 1.5, 100)
    result = LogisticRegression.sigmoid_poly(z)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.5)
    assert result.shape == z.shape


def test_sigmoid_poly_monotonic():
    z = np.linspace(-1, 1, 50)
    result = LogisticRegression.sigmoid_poly(z)
    for i in range(1, len(result)):
        assert result[i] >= result[i - 1]


def test_sigmoid_poly_zero():
    result = LogisticRegression.sigmoid_poly(np.array([0.0]))
    assert result[0] == pytest.approx(0.5, abs=0.01)


def test_train_binary_separable():
    rng = np.random.RandomState(42)
    n_samples = 50
    X_class0 = rng.randn(n_samples, 2) * 0.1 + np.array([-0.5, -0.5])
    X_class1 = rng.randn(n_samples, 2) * 0.1 + np.array([0.5, 0.5])
    X = np.vstack([X_class0, X_class1]).astype(np.float64)
    y = np.array([0.0] * n_samples + [1.0] * n_samples)

    weights, bias = LogisticRegression.fit(
        plaintext_matrix=X,
        label_vector=y,
        epochs=50,
        learning_rate=0.1,
    )

    predictions = LogisticRegression.predict(
        plaintext_matrix=X, weights=weights, bias=bias
    )
    accuracy = np.mean(np.array(predictions) == y.astype(int))
    assert accuracy > 0.6


def test_train_default_weights():
    rng = np.random.RandomState(42)
    X = rng.randn(20, 2).astype(np.float64)
    y = rng.binomial(1, 0.5, size=20).astype(np.float64)

    weights, bias = LogisticRegression.fit(
        plaintext_matrix=X,
        label_vector=y,
        epochs=5,
        learning_rate=0.1,
    )

    assert weights.shape == (2,)
    assert isinstance(bias, float)


def test_predict_returns_binary():
    rng = np.random.RandomState(42)
    X = rng.randn(10, 3).astype(np.float64)
    weights = np.zeros(3, dtype=np.float64)
    bias = 0.0

    predictions = LogisticRegression.predict(
        plaintext_matrix=X, weights=weights, bias=bias
    )

    assert len(predictions) == 10
    assert all(p in (0, 1) for p in predictions)


def test_train_reduces_loss():
    rng = np.random.RandomState(42)
    X = rng.randn(30, 4).astype(np.float64) * 0.1
    y = rng.binomial(1, 0.5, size=30).astype(np.float64)

    weights, bias = LogisticRegression.fit(
        plaintext_matrix=X,
        label_vector=y,
        epochs=0,
        learning_rate=0.1,
    )

    pred_init = np.array(
        LogisticRegression.predict(
            plaintext_matrix=X, weights=weights, bias=bias
        )
    )
    loss_init = np.mean((pred_init.astype(float) - y) ** 2)

    weights, bias = LogisticRegression.fit(
        plaintext_matrix=X,
        label_vector=y,
        epochs=10,
        learning_rate=0.5,
        weights=weights.copy(),
        bias=float(bias),
    )

    pred_final = np.array(
        LogisticRegression.predict(
            plaintext_matrix=X, weights=weights, bias=bias
        )
    )
    loss_final = np.mean((pred_final.astype(float) - y) ** 2)

    assert loss_final <= loss_init