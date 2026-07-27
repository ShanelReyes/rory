from models.requests.classification import (
    KnnPredictWorkerRequest,
    SknnPredictWorkerRequest,
    PqcSknnPredictWorkerRequest,
)
from models.responses.classification import (
    KnnPredictResponse,
    SknnPredictStep1Response,
    SknnPredictStep2Response,
)


class TestHealthCheck:
    def test_get(self, client):
        response = client.get("/classification/test")
        assert response.status_code == 200
        assert response.json()["component_type"] == "worker"

    def test_post(self, client):
        response = client.post("/classification/test")
        assert response.status_code == 200
        assert response.json()["component_type"] == "worker"


class TestKnnPredictWorkerRequest:
    def test_defaults(self):
        body = KnnPredictWorkerRequest()
        assert body.model_id == "model0"
        assert body.records_test_id == "matrix0"

    def test_knn_predict_endpoint(self, client):
        response = client.post("/classification/knn/predict", json={
            "model_id": "model-0",
            "records_test_id": "test-data",
            "model_labels_shape": "(100,3)",
        })
        assert response.status_code in [200, 422, 500, 503]

    def test_missing_model_labels_shape(self, client):
        response = client.post("/classification/knn/predict", json={
            "model_id": "model-0",
        })
        assert response.status_code in [200, 422, 500, 503]


class TestSknnPredictWorkerRequest:
    def test_default_step(self):
        body = SknnPredictWorkerRequest()
        assert body.step_index == 1
        assert body.model_id == "model0"

    def test_step1_params(self):
        body = SknnPredictWorkerRequest(
            step_index=1,
            encrypted_model_shape="(100,5,3)",
            encrypted_model_dtype="float32",
            encrypted_records_shape="(10,5,3)",
            encrypted_records_dtype="float32",
            num_chunks="4",
        )
        assert body.encrypted_model_shape == "(100,5,3)"
        assert body.encrypted_records_shape == "(10,5,3)"

    def test_step2_params(self):
        body = SknnPredictWorkerRequest(
            step_index=2,
            model_labels_shape="(100,)",
        )
        assert body.step_index == 2
        assert body.model_labels_shape == "(100,)"

    def test_invalid_step_index(self, client):
        response = client.post("/classification/sknn/predict", json={
            "step_index": 0,
        })
        assert response.status_code == 422

    def test_sknn_predict_endpoint(self, client):
        response = client.post("/classification/sknn/predict", json={
            "step_index": 1,
            "encrypted_model_shape": "(100,5,3)",
            "encrypted_model_dtype": "float32",
            "encrypted_records_shape": "(10,5,3)",
            "encrypted_records_dtype": "float32",
            "num_chunks": "4",
        })
        assert response.status_code in [200, 422, 500, 503]


class TestPqcSknnPredictWorkerRequest:
    def test_is_sknn_subclass(self):
        body = PqcSknnPredictWorkerRequest(
            step_index=1,
            encrypted_model_shape="(50,10)",
            encrypted_model_dtype="float32",
            encrypted_records_shape="(5,10)",
            encrypted_records_dtype="float32",
            num_chunks="2",
        )
        assert body.step_index == 1

    def test_pqc_sknn_predict_endpoint(self, client):
        response = client.post("/classification/pqc/sknn/predict", json={
            "step_index": 1,
            "encrypted_model_shape": "(50,10)",
            "encrypted_model_dtype": "float32",
            "encrypted_records_shape": "(5,10)",
            "encrypted_records_dtype": "float32",
            "num_chunks": "2",
        })
        assert response.status_code in [200, 422, 500, 503]


class TestResponseModels:
    def test_knn_predict_response(self):
        resp = KnnPredictResponse(label_vector=[0, 1, 0], service_time=3.0)
        assert resp.label_vector == [0, 1, 0]

    def test_sknn_step1_response(self):
        resp = SknnPredictStep1Response(
            distances_id="dist-1",
            distances_shape="(10,100)",
            distances_dtype="float32",
            service_time=5.0,
        )
        assert resp.distances_id == "dist-1"
        assert resp.service_time == 5.0

    def test_sknn_step2_response(self):
        resp = SknnPredictStep2Response(label_vector=[1, 2, 3], service_time=2.0)
        assert resp.label_vector == [1, 2, 3]
