import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


@pytest.mark.skip(reason="Integration test requiring running manager, worker, and MictlanX services")
def test_dataowner_logistic_regression_train():
    result = client.post("/machine-learning/logistic-regression/train", json={
        "experiment_id": "int-test-lr",
        "plaintext_matrix_train_id": "dataset1_train",
        "plaintext_label_vector_train_id": "label_vector_train1",
        "plaintext_matrix_train_filename": "dataset1_train",
        "plaintext_label_vector_train_filename": "label_vector_train",
        "extension": "npy",
        "epochs": 1,
        "learning_rate": 0.1,
    })
    if result.status_code != 200:
        print("Error:", result.status_code, result.text)
    else:
        print("Training response:", result.json())
    assert result.status_code == 200


@pytest.mark.skip(reason="Integration test requiring running services")
def test_dataowner_logistic_regression_predict():
    result = client.post("/machine-learning/logistic-regression/predict", json={
        "experiment_id": "int-test-lr",
        "plaintext_matrix_test_id": "dataset1_test",
        "plaintext_matrix_test_filename": "dataset1_train",
        "plaintext_matrix_train_id": "dataset1_train",
        "extension": "npy",
    })
    if result.status_code != 200:
        print("Error:", result.status_code, result.text)
    else:
        print("Prediction response:", result.json())
    assert result.status_code == 200


@pytest.mark.skip(reason="Integration test requiring running services")
def test_logistic_regression():
    test_dataowner_logistic_regression_train()
    test_dataowner_logistic_regression_predict()
