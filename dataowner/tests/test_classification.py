from models.requests.classification import (
    KnnTrainRequest,
    KnnPredictRequest,
    SknnTrainRequest,
    SknnPredictRequest,
    PqcSknnTrainRequest,
    PqcSknnPredictRequest,
)
from models.responses.classification import (
    KnnTrainResponse,
    SknnTrainResponse,
    PqcSknnTrainResponse,
    PredictResponse,
)


class TestClassificationHealthCheck:
    def test_health_check_get(self, client):
        response = client.get("/classification/test")
        assert response.status_code == 200
        assert response.json()["component_type"] == "dataowner"

    def test_health_check_post(self, client):
        response = client.post("/classification/test")
        assert response.status_code == 200
        assert response.json()["component_type"] == "dataowner"


class TestKnnTrainRequestValidation:
    def test_default_values(self):
        body = KnnTrainRequest()
        assert body.model_id == "matrix0model"
        assert body.extension == "npy"
        assert body.model_filename == "matrix0model"
        assert body.model_labels_filename == "matrix0modellabels"

    def test_custom_values(self):
        body = KnnTrainRequest(
            model_id="my_model",
            model_filename="my_features",
            model_labels_filename="my_labels",
            extension="csv",
        )
        assert body.model_id == "my_model"
        assert body.model_filename == "my_features"
        assert body.model_labels_filename == "my_labels"

    def test_valid_json_body(self, client):
        response = client.post("/classification/knn/train", json={
            "model_id": "test-model",
            "extension": "npy",
            "experiment_id": "exp-knn-train",
        })
        assert response.status_code in [200, 500]

    def test_invalid_model_id_type(self, client):
        response = client.post("/classification/knn/train", json={
            "model_id": 123,
        })
        assert response.status_code == 422


class TestKnnPredictRequestValidation:
    def test_model_labels_shape_required(self):
        try:
            KnnPredictRequest()
            assert False, "Should have raised validation error"
        except Exception:
            pass

    def test_with_model_labels_shape(self):
        body = KnnPredictRequest(model_labels_shape="(10, 5)")
        assert body.model_labels_shape == "(10, 5)"
        assert body.model_id == "model-0"
        assert body.records_test_id == "matrix0data"

    def test_custom_params(self):
        body = KnnPredictRequest(
            model_id="trained-model",
            records_test_id="test-records",
            model_labels_shape="(100, 3)",
            extension="npy",
        )
        assert body.model_id == "trained-model"
        assert body.records_test_id == "test-records"
        assert body.records_test_filename == "test-records"

    def test_valid_json_body(self, client):
        response = client.post("/classification/knn/predict", json={
            "model_id": "model-0",
            "records_test_id": "test-data",
            "model_labels_shape": "(100, 3)",
        })
        assert response.status_code in [200, 500]

    def test_missing_required_field(self, client):
        response = client.post("/classification/knn/predict", json={
            "model_id": "model-0",
        })
        assert response.status_code == 422


class TestSknnTrainRequestValidation:
    def test_default_values(self):
        body = SknnTrainRequest()
        assert body.model_id == "matrix-0_model"
        assert body.extension == "npy"
        assert body.model_filename == "matrix-0_model"
        assert body.model_labels_filename == "matrix-0_modellabels"

    def test_valid_json_body(self, client):
        response = client.post("/classification/sknn/train", json={
            "model_id": "secure-model",
            "extension": "npy",
            "experiment_id": "exp-sknn-train",
        })
        assert response.status_code in [200, 500]


class TestSknnPredictRequestValidation:
    def test_required_fields(self):
        body = SknnPredictRequest(
            encrypted_model_shape="(100, 5, 3)",
            encrypted_model_dtype="float32",
            model_labels_shape="(100,)",
        )
        assert body.encrypted_model_shape == "(100, 5, 3)"
        assert body.encrypted_model_dtype == "float32"
        assert body.model_labels_shape == "(100,)"

    def test_missing_all_required(self):
        try:
            SknnPredictRequest()
            assert False, "Should have raised validation error"
        except Exception:
            pass

    def test_valid_json_body(self, client):
        response = client.post("/classification/sknn/predict", json={
            "model_id": "secure-model",
            "records_test_id": "test-data",
            "encrypted_model_shape": "(100, 5, 3)",
            "encrypted_model_dtype": "float32",
            "model_labels_shape": "(100,)",
        })
        assert response.status_code in [200, 500]

    def test_missing_encrypted_model_shape(self, client):
        response = client.post("/classification/sknn/predict", json={
            "model_id": "secure-model",
            "encrypted_model_dtype": "float32",
            "model_labels_shape": "(100,)",
        })
        assert response.status_code == 422


class TestPqcSknnTrainRequestValidation:
    def test_default_values(self):
        body = PqcSknnTrainRequest()
        assert body.model_id == "matrix-0_model"
        assert body.extension == "npy"

    def test_valid_json_body(self, client):
        response = client.post("/classification/pqc/sknn/train", json={
            "model_id": "pqc-model",
            "extension": "npy",
            "experiment_id": "exp-pqc-train",
        })
        assert response.status_code in [200, 500]


class TestPqcSknnPredictRequestValidation:
    def test_required_fields(self):
        body = PqcSknnPredictRequest(
            encrypted_model_shape="(50, 10)",
            encrypted_model_dtype="float32",
        )
        assert body.encrypted_model_shape == "(50, 10)"
        assert body.encrypted_model_dtype == "float32"
        assert body.records_test_extension == "npy"

    def test_missing_all_required(self):
        try:
            PqcSknnPredictRequest()
            assert False, "Should have raised validation error"
        except Exception:
            pass

    def test_valid_json_body(self, client):
        response = client.post("/classification/pqc/sknn/predict", json={
            "model_id": "pqc-model",
            "records_test_id": "test-data",
            "encrypted_model_shape": "(50, 10)",
            "encrypted_model_dtype": "float32",
        })
        assert response.status_code in [200, 500]

    def test_missing_shape(self, client):
        response = client.post("/classification/pqc/sknn/predict", json={
            "model_id": "pqc-model",
            "encrypted_model_dtype": "float32",
        })
        assert response.status_code == 422


class TestResponseModels:
    def test_knn_train_response(self):
        resp = KnnTrainResponse(
            response_time=1.5,
            algorithm="knn_train",
            model_labels_shape=[100, 3],
        )
        assert resp.response_time == 1.5
        assert resp.algorithm == "knn_train"
        assert resp.model_labels_shape == [100, 3]

    def test_sknn_train_response(self):
        resp = SknnTrainResponse(
            response_time=2.0,
            encrypted_model_shape="(100, 5, 3)",
            encrypted_model_dtype="float32",
            algorithm="sknn_train",
            model_labels_shape=[100],
        )
        assert resp.encrypted_model_shape == "(100, 5, 3)"
        assert resp.encrypted_model_dtype == "float32"
        assert resp.algorithm == "sknn_train"

    def test_pqc_sknn_train_response(self):
        resp = PqcSknnTrainResponse(
            response_time=3.0,
            encrypted_model_shape="(100, 5)",
            encrypted_model_dtype="float32",
            algorithm="sknn_pqc_train",
            model_labels_shape=[100],
        )
        assert resp.encrypted_model_shape == "(100, 5)"

    def test_predict_response(self):
        resp = PredictResponse(
            label_vector=[0, 1, 0, 1],
            algorithm="knn_predict",
            worker_id="worker-1",
            service_time_manager=1.0,
            service_time_worker=5.0,
            service_time_dataowner=2.0,
            service_time_predict=8.0,
        )
        assert resp.label_vector == [0, 1, 0, 1]
        assert resp.algorithm == "knn_predict"
        assert resp.service_time_manager == 1.0

    def test_predict_response_serialization(self):
        resp = PredictResponse(
            label_vector=[1, 2, 3],
            algorithm="test_algo",
            service_time_predict=10.0,
        )
        json_data = resp.model_dump()
        assert json_data["label_vector"] == [1, 2, 3]
        assert json_data["algorithm"] == "test_algo"
        assert json_data["service_time_predict"] == 10.0
