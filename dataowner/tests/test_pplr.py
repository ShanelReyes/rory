import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


@pytest.mark.skip(reason="Integration test requiring running manager, worker, and MictlanX services")
def test_dataowner_pplr_train():
    result = client.post("/machine-learning/pplr/train", json={
        "experiment_id": "int-test-pplr",
        "plaintext_matrix_train_id": "dataset1_train",
        "plaintext_label_vector_train_id": "label_vector_train1",
        "plaintext_matrix_train_filename": "dataset1_train",
        "plaintext_label_vector_train_filename": "label_vector_train",
        "extension": "npy",
        "epochs": 3,
        "learning_rate": 0.1,
    })
    if result.status_code != 200:
        print("Error:", result.status_code, result.text)
    else:
        print("Training response:", result.json())
    assert result.status_code == 200


@pytest.mark.skip(reason="Integration test requiring running services")
def test_dataowner_pplr_predict():
    result = client.post("/machine-learning/pplr/predict", json={
        "experiment_id": "int-test-pplr",
        "plaintext_matrix_test_id": "dataset1_test",
        "plaintext_matrix_test_filename": "dataset1_test",
        "extension": "npy",
        "plaintext_matrix_train_id": "dataset1_train",
    })
    if result.status_code != 200:
        print("Error:", result.status_code, result.text)
    else:
        print("Prediction response:", result.json())
    assert result.status_code == 200


@pytest.mark.skip(reason="Integration test requiring running services")
def test_pplr():
    test_dataowner_pplr_train()
    test_dataowner_pplr_predict()
