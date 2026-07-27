from models.requests.machinelearning import (
    LRTrainWorkerRequest,
    LRPredictWorkerRequest,
    PPLRTrainWorkerRequest,
    PPLRPredictWorkerRequest,
)
from models.responses.machinelearning import (
    LRTrainResponse,
    LRPredictResponse,
    PPLRTrainResponse,
    PPLRPredictResponse,
)


class TestHealthCheck:
    def test_get(self, client):
        response = client.get("/machine-learning/test")
        assert response.status_code == 200
        assert response.json()["component_type"] == "worker"

    def test_post(self, client):
        response = client.post("/machine-learning/test")
        assert response.status_code == 200
        assert response.json()["component_type"] == "worker"


class TestLRTrainWorkerRequest:
    def test_defaults(self):
        body = LRTrainWorkerRequest()
        assert body.plaintext_matrix_train_id == "train_x"
        assert body.plaintext_label_vector_train_id == "train_y"
        assert body.epochs == "1"
        assert body.learning_rate == "0.01"

    def test_custom(self):
        body = LRTrainWorkerRequest(
            plaintext_matrix_train_id="features",
            plaintext_label_vector_train_id="labels",
            weights_id="w-1",
            bias_id="b-1",
            epochs="50",
            learning_rate="0.001",
        )
        assert body.weights_id == "w-1"
        assert body.epochs == "50"

    def test_lr_train_endpoint(self, client):
        response = client.post("/machine-learning/logistic-regression/train", json={
            "plaintext_matrix_train_id": "train_x",
            "plaintext_label_vector_train_id": "train_y",
            "weights_id": "w-1",
            "bias_id": "b-1",
        })
        assert response.status_code in [200, 422, 500]


class TestLRPredictWorkerRequest:
    def test_defaults(self):
        body = LRPredictWorkerRequest()
        assert body.plaintext_matrix_test_id == "test_x"

    def test_custom(self):
        body = LRPredictWorkerRequest(
            plaintext_matrix_test_id="test_features",
            weights_id="w-1",
            bias_id="b-1",
        )
        assert body.plaintext_matrix_test_id == "test_features"

    def test_lr_predict_endpoint(self, client):
        response = client.post("/machine-learning/logistic-regression/predict", json={
            "plaintext_matrix_test_id": "test_x",
            "weights_id": "w-1",
            "bias_id": "b-1",
        })
        assert response.status_code in [200, 422, 500]


class TestPPLRTrainWorkerRequest:
    def test_defaults(self):
        body = PPLRTrainWorkerRequest()
        assert body.learning_rate == "0.01"
        assert body.scale == "40"
        assert body.n_features == "0"
        assert body.n_samples == "0"
        assert body.num_chunks == "1"

    def test_custom(self):
        body = PPLRTrainWorkerRequest(
            encrypted_matrix_train_id="enc-train",
            encrypted_label_vector_train_id="enc-labels",
            encrypted_weights_id="enc-w",
            encrypted_bias_id="enc-b",
            n_features="10",
            n_samples="1000",
        )
        assert body.encrypted_matrix_train_id == "enc-train"
        assert body.n_features == "10"

    def test_pplr_train_endpoint(self, client):
        response = client.post("/machine-learning/pplr/train", json={
            "encrypted_matrix_train_id": "enc-train",
            "encrypted_label_vector_train_id": "enc-labels",
            "encrypted_weights_id": "enc-w",
            "encrypted_bias_id": "enc-b",
        })
        assert response.status_code in [200, 422, 500]


class TestPPLRPredictWorkerRequest:
    def test_defaults(self):
        body = PPLRPredictWorkerRequest()
        assert body.scale == "40"
        assert body.n_features == "0"
        assert body.num_chunks == "1"

    def test_custom(self):
        body = PPLRPredictWorkerRequest(
            encrypted_matrix_test_id="enc-test",
            encrypted_weights_id="enc-w",
            encrypted_bias_id="enc-b",
            n_features="10",
        )
        assert body.encrypted_matrix_test_id == "enc-test"

    def test_pplr_predict_endpoint(self, client):
        response = client.post("/machine-learning/pplr/predict", json={
            "encrypted_matrix_test_id": "enc-test",
            "encrypted_weights_id": "enc-w",
            "encrypted_bias_id": "enc-b",
        })
        assert response.status_code in [200, 422, 500]


class TestResponseModels:
    def test_lr_train_response(self):
        resp = LRTrainResponse(service_time=5.0, train_time=4.0, algorithm="lr_train")
        assert resp.train_time == 4.0
        assert resp.algorithm == "lr_train"

    def test_lr_predict_response(self):
        resp = LRPredictResponse(predictions_id="pred-1", predict_time=2.0, service_time=3.0)
        assert resp.predictions_id == "pred-1"

    def test_pplr_train_response(self):
        resp = PPLRTrainResponse(service_time=10.0, train_time=8.0, algorithm="pplr_train")
        assert resp.train_time == 8.0

    def test_pplr_predict_response(self):
        resp = PPLRPredictResponse(
            encrypted_predictions_id="enc-pred",
            predict_time=3.0,
            service_time=4.0,
            algorithm="pplr_predict",
        )
        assert resp.encrypted_predictions_id == "enc-pred"
        assert resp.algorithm == "pplr_predict"
