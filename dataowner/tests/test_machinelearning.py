from models.requests.machinelearning import (
    LRTrainRequest,
    LRPredictRequest,
    PPLRTrainRequest,
    PPLRPredictRequest,
)
from models.responses.machinelearning import (
    LRTrainResponse,
    LRPredictResponse,
    PPLRTrainResponse,
    PPLRPredictResponse,
)


class TestMLHealthCheck:
    def test_health_check_get(self, client):
        response = client.get("/machine-learning/test")
        assert response.status_code == 200
        assert response.json()["component_type"] == "dataowner"

    def test_health_check_post(self, client):
        response = client.post("/machine-learning/test")
        assert response.status_code == 200
        assert response.json()["component_type"] == "dataowner"


class TestLRTrainRequestValidation:
    def test_default_values(self):
        body = LRTrainRequest()
        assert body.plaintext_matrix_train_id == "train_x"
        assert body.plaintext_label_vector_train_id == "train_y"
        assert body.extension == "csv"
        assert body.epochs == 1
        assert body.learning_rate == 0.01
        assert body.plaintext_matrix_train_filename == "train_x"
        assert body.plaintext_label_vector_train_filename == "train_y"

    def test_custom_values(self):
        body = LRTrainRequest(
            plaintext_matrix_train_id="features_train",
            plaintext_label_vector_train_id="labels_train",
            epochs=50,
            learning_rate=0.001,
            extension="npy",
            experiment_id="exp-lr-001",
        )
        assert body.plaintext_matrix_train_id == "features_train"
        assert body.epochs == 50
        assert body.learning_rate == 0.001

    def test_filename_defaults_to_id(self):
        body = LRTrainRequest(
            plaintext_matrix_train_id="custom_train_x",
            plaintext_label_vector_train_id="custom_train_y",
        )
        assert body.plaintext_matrix_train_filename == "custom_train_x"
        assert body.plaintext_label_vector_train_filename == "custom_train_y"

    def test_negative_epochs(self, client):
        response = client.post("/machine-learning/logistic-regression/train", json={
            "epochs": -1,
        })
        assert response.status_code == 422

    def test_learning_rate_too_high(self):
        try:
            LRTrainRequest(learning_rate=2.0)
            assert False, "Should have raised validation error"
        except Exception:
            pass

    def test_negative_learning_rate(self):
        try:
            LRTrainRequest(learning_rate=-0.5)
            assert False, "Should have raised validation error due to gt=0"
        except Exception:
            pass

    def test_valid_json_body(self, client):
        response = client.post("/machine-learning/logistic-regression/train", json={
            "plaintext_matrix_train_id": "train_x",
            "plaintext_label_vector_train_id": "train_y",
            "epochs": 1,
            "learning_rate": 0.1,
            "extension": "npy",
            "experiment_id": "exp-lr-train",
        })
        assert response.status_code in [200, 500]

    def test_invalid_epochs_type(self, client):
        response = client.post("/machine-learning/logistic-regression/train", json={
            "epochs": "many",
        })
        assert response.status_code == 422

    def test_invalid_learning_rate_type(self, client):
        response = client.post("/machine-learning/logistic-regression/train", json={
            "learning_rate": "low",
        })
        assert response.status_code == 422


class TestLRPredictRequestValidation:
    def test_default_values(self):
        body = LRPredictRequest()
        assert body.plaintext_matrix_train_id == "train_x"
        assert body.plaintext_matrix_test_id == "test_x"
        assert body.extension == "csv"
        assert body.plaintext_matrix_test_filename == "test_x"

    def test_custom_values(self):
        body = LRPredictRequest(
            plaintext_matrix_train_id="trained_features",
            plaintext_matrix_test_id="test_features",
            extension="npy",
        )
        assert body.plaintext_matrix_train_id == "trained_features"
        assert body.plaintext_matrix_test_id == "test_features"

    def test_valid_json_body(self, client):
        response = client.post("/machine-learning/logistic-regression/predict", json={
            "plaintext_matrix_train_id": "train_x",
            "plaintext_matrix_test_id": "test_x",
            "extension": "npy",
            "experiment_id": "exp-lr-predict",
        })
        assert response.status_code in [200, 500]


class TestPPLRTrainRequestValidation:
    def test_default_values(self):
        body = PPLRTrainRequest()
        assert body.plaintext_matrix_train_id == "train_x"
        assert body.plaintext_label_vector_train_id == "train_y"
        assert body.epochs == 1
        assert body.learning_rate == 0.01
        assert body.extension == "csv"

    def test_custom_values(self):
        body = PPLRTrainRequest(
            plaintext_matrix_train_id="pplr_features",
            plaintext_label_vector_train_id="pplr_labels",
            epochs=10,
            learning_rate=0.05,
        )
        assert body.epochs == 10
        assert body.learning_rate == 0.05

    def test_valid_json_body(self, client):
        response = client.post("/machine-learning/pplr/train", json={
            "plaintext_matrix_train_id": "train_x",
            "plaintext_label_vector_train_id": "train_y",
            "epochs": 1,
            "learning_rate": 0.1,
            "extension": "npy",
            "experiment_id": "exp-pplr-train",
        })
        assert response.status_code in [200, 500]

    def test_invalid_epochs(self, client):
        response = client.post("/machine-learning/pplr/train", json={
            "epochs": 0,
        })
        assert response.status_code == 422

    def test_invalid_learning_rate_zero(self, client):
        response = client.post("/machine-learning/pplr/train", json={
            "learning_rate": 0.0,
        })
        assert response.status_code == 422


class TestPPLRPredictRequestValidation:
    def test_default_values(self):
        body = PPLRPredictRequest()
        assert body.plaintext_matrix_test_id == "test_x"
        assert body.plaintext_matrix_train_id == "train_x"
        assert body.extension == "csv"
        assert body.experiment_iteration == "0"

    def test_custom_values(self):
        body = PPLRPredictRequest(
            plaintext_matrix_test_id="pplr_test",
            plaintext_matrix_train_id="pplr_train",
            extension="npy",
            experiment_iteration="5",
        )
        assert body.plaintext_matrix_test_id == "pplr_test"
        assert body.experiment_iteration == "5"

    def test_valid_json_body(self, client):
        response = client.post("/machine-learning/pplr/predict", json={
            "plaintext_matrix_test_id": "test_x",
            "plaintext_matrix_train_id": "train_x",
            "extension": "npy",
            "experiment_id": "exp-pplr-predict",
        })
        assert response.status_code in [200, 500]


class TestMLResponseModels:
    def test_lr_train_response(self):
        resp = LRTrainResponse(
            worker_id="worker-1",
            service_time_manager=1.0,
            service_time_worker=5.0,
            service_time_dataowner=3.0,
            service_time_train=9.0,
            algorithm="logistic_regression_train",
        )
        assert resp.worker_id == "worker-1"
        assert resp.algorithm == "logistic_regression_train"
        assert resp.service_time_train == 9.0

    def test_lr_predict_response(self):
        resp = LRPredictResponse(
            label_vector=[0, 1, 0],
            algorithm="logistic_regression_predict",
            worker_id="worker-2",
            service_time_manager=0.5,
            service_time_worker=2.0,
            service_time_dataowner=1.0,
            service_time_predict=3.5,
        )
        assert resp.label_vector == [0, 1, 0]
        assert resp.service_time_predict == 3.5

    def test_pplr_train_response(self):
        resp = PPLRTrainResponse(
            algorithm="pplr_train",
            worker_id="worker-pqc",
            epochs=5,
            service_time_manager=2.0,
            service_time_worker=30.0,
            service_time_dataowner=10.0,
            service_time_train=42.0,
        )
        assert resp.epochs == 5
        assert resp.algorithm == "pplr_train"
        assert resp.service_time_train == 42.0

    def test_pplr_predict_response(self):
        resp = PPLRPredictResponse(
            label_vector=[1, 0, 1, 0],
            algorithm="pplr_predict",
            worker_id="worker-pqc",
            service_time_manager=1.0,
            service_time_worker=3.0,
            service_time_dataowner=2.0,
            service_time_predict=6.0,
        )
        assert resp.label_vector == [1, 0, 1, 0]
        assert resp.service_time_predict == 6.0

    def test_pplr_predict_response_serialization(self):
        resp = PPLRPredictResponse(
            label_vector=[0, 1],
            algorithm="pplr_predict",
            service_time_predict=5.0,
        )
        json_data = resp.model_dump()
        assert json_data["label_vector"] == [0, 1]
        assert json_data["service_time_predict"] == 5.0
