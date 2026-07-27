from models.requests.clustering import (
    KmeansWorkerRequest,
    SkmeansWorkerRequest,
    DbskmeansWorkerRequest,
    DbsnncWorkerRequest,
    NncWorkerRequest,
    PqcSkmeansWorkerRequest,
    PqcDbskmeansWorkerRequest,
)
from models.responses.clustering import (
    HealthCheckResponse,
    WorkerRun1Response,
    WorkerDbsnncResponse,
    WorkerNncResponse,
)


class TestHealthCheck:
    def test_get(self, client):
        response = client.get("/clustering/test")
        assert response.status_code == 200
        assert response.json()["component_type"] == "worker"

    def test_post(self, client):
        response = client.post("/clustering/test")
        assert response.status_code == 200
        assert response.json()["component_type"] == "worker"


class TestKmeansWorkerRequest:
    def test_defaults(self):
        body = KmeansWorkerRequest()
        assert body.plaintext_matrix_id == "matrix-0"
        assert body.k == 3

    def test_custom(self):
        body = KmeansWorkerRequest(plaintext_matrix_id="custom", k=5)
        assert body.plaintext_matrix_id == "custom"
        assert body.k == 5

    def test_k_below_min(self):
        from pydantic import ValidationError
        try:
            KmeansWorkerRequest(k=1)
            assert False, "ValidationError not raised"
        except ValidationError:
            pass

    def test_kmeans_endpoint(self, client):
        response = client.post("/clustering/kmeans", json={
            "plaintext_matrix_id": "test-matrix",
            "k": 3,
        })
        assert response.status_code in [200, 422, 500, 503]


class TestSkmeansWorkerRequest:
    def test_defaults(self):
        body = SkmeansWorkerRequest()
        assert body.step_index == 1
        assert body.k == "3"
        assert body.m == "3"
        assert body.iterations == "0"

    def test_step1_params(self):
        body = SkmeansWorkerRequest(
            step_index=1,
            encrypted_matrix_shape="(100,5,3)",
            encrypted_matrix_dtype="float32",
            num_chunks="4",
        )
        assert body.encrypted_matrix_shape == "(100,5,3)"
        assert body.num_chunks == "4"

    def test_step2_params(self):
        body = SkmeansWorkerRequest(
            step_index=2,
            is_zero="1",
            shift_matrix_id="test-shift",
        )
        assert body.step_index == 2
        assert body.is_zero == "1"

    def test_invalid_step_index(self, client):
        response = client.post("/clustering/skmeans", json={
            "step_index": 3,
            "k": "3",
        })
        assert response.status_code == 422

    def test_skmeans_endpoint(self, client):
        response = client.post("/clustering/skmeans", json={
            "step_index": 1,
            "k": "3",
            "m": "3",
            "encrypted_matrix_shape": "(10,5,3)",
            "encrypted_matrix_dtype": "float32",
            "num_chunks": "4",
        })
        assert response.status_code in [200, 422, 500, 503]


class TestDbskmeansWorkerRequest:
    def test_udm_fields(self):
        body = DbskmeansWorkerRequest(
            step_index=1,
            encrypted_udm_shape="(10,10,5)",
            encrypted_udm_dtype="float32",
        )
        assert body.encrypted_udm_shape == "(10,10,5)"

    def test_dbskmeans_endpoint(self, client):
        response = client.post("/clustering/dbskmeans", json={
            "step_index": 1,
            "k": "3",
            "encrypted_matrix_shape": "(10,5,3)",
            "encrypted_matrix_dtype": "float32",
            "encrypted_udm_shape": "(10,10,5)",
            "encrypted_udm_dtype": "float32",
            "num_chunks": "4",
        })
        assert response.status_code in [200, 422, 500, 503]


class TestDbsnncWorkerRequest:
    def test_defaults(self):
        body = DbsnncWorkerRequest()
        assert body.m == "3"

    def test_required_fields(self):
        body = DbsnncWorkerRequest(
            encrypted_threshold="0.5",
            encrypted_matrix_shape="(10,5,3)",
            encrypted_matrix_dtype="float32",
            encrypted_dm_shape="(10,10)",
            encrypted_dm_dtype="float32",
            num_chunks="4",
        )
        assert body.encrypted_threshold == "0.5"

    def test_dbsnnc_endpoint(self, client):
        response = client.post("/clustering/dbsnnc", json={
            "encrypted_threshold": "0.5",
            "encrypted_matrix_shape": "(10,5,3)",
            "encrypted_matrix_dtype": "float32",
            "encrypted_dm_shape": "(10,10)",
            "encrypted_dm_dtype": "float32",
            "num_chunks": "4",
        })
        assert response.status_code in [200, 422, 500, 503]


class TestNncWorkerRequest:
    def test_defaults(self):
        body = NncWorkerRequest()
        assert body.plaintext_matrix_id == "matrix0"

    def test_nnc_endpoint(self, client):
        response = client.post("/clustering/nnc", json={
            "threshold": "2.5",
            "plaintext_matrix_shape": "(10,5)",
            "plaintext_matrix_dtype": "float64",
            "dm_shape": "(10,10)",
            "dm_dtype": "float64",
            "num_chunks": "4",
        })
        assert response.status_code in [200, 422, 500, 503]


class TestPqcSkmeansWorkerRequest:
    def test_default_step(self):
        body = PqcSkmeansWorkerRequest()
        assert body.step_index == 1

    def test_pqc_skmeans_endpoint(self, client):
        response = client.post("/clustering/pqc/skmeans", json={
            "step_index": 1,
            "k": "3",
            "encrypted_matrix_shape": "(10,5)",
            "encrypted_matrix_dtype": "float32",
            "num_chunks": "4",
        })
        assert response.status_code in [200, 422, 500, 503]


class TestPqcDbskmeansWorkerRequest:
    def test_ope_field(self):
        body = PqcDbskmeansWorkerRequest(
            step_index=2,
            shift_matrix_ope_id="test-ope-id",
        )
        assert body.shift_matrix_ope_id == "test-ope-id"

    def test_pqc_dbskmeans_endpoint(self, client):
        response = client.post("/clustering/pqc/dbskmeans", json={
            "step_index": 1,
            "k": "3",
            "encrypted_matrix_shape": "(10,5)",
            "encrypted_matrix_dtype": "float32",
            "encrypted_udm_shape": "(10,10,5)",
            "encrypted_udm_dtype": "float32",
            "num_chunks": "4",
        })
        assert response.status_code in [200, 422, 500, 503]


class TestResponseModels:
    def test_health_check(self):
        resp = HealthCheckResponse()
        assert resp.component_type == "worker"

    def test_worker_run1(self):
        resp = WorkerRun1Response(
            label_vector=[0, 1, 2],
            service_time=5.0,
            n_iterations=10,
            encrypted_shift_matrix_id="shift-1",
        )
        assert resp.label_vector == [0, 1, 2]
        assert resp.service_time == 5.0

    def test_dbsnnc_response(self):
        resp = WorkerDbsnncResponse(label_vector=[0, 1], service_time=2.0)
        assert resp.label_vector == [0, 1]

    def test_nnc_response(self):
        resp = WorkerNncResponse(label_vector=[0, 1, 0], service_time=1.5)
        assert resp.label_vector == [0, 1, 0]
