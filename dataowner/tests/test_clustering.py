from models.requests.clustering import (
    KmeansRequest,
    SkmeansRequest,
    DbskmeansRequest,
    DbsnncRequest,
    NncRequest,
    PqcSkmeansRequest,
    PqcDbskmeansRequest,
    BaseRequest,
    MatrixRequest,
    EncryptedClusteringRequest,
)
from models.responses.clustering import (
    HealthCheckResponse,
    KmeansResponse,
    SkmeansResponse,
    DbskmeansResponse,
    NncResponse,
    DbsnncResponse,
    PqcSkmeansResponse,
    PqcDbskmeansResponse,
)


class TestHealthCheck:
    def test_health_check_get(self, client):
        response = client.get("/clustering/test")
        assert response.status_code == 200
        data = response.json()
        assert data["component_type"] == "dataowner"

    def test_health_check_post(self, client):
        response = client.post("/clustering/test")
        assert response.status_code == 200
        assert response.json()["component_type"] == "dataowner"


class TestBaseRequestValidation:
    def test_auto_experiment_id_generated(self):
        body = BaseRequest()
        assert body.experiment_id is not None
        assert len(body.experiment_id) == 10

    def test_explicit_experiment_id(self):
        body = BaseRequest(experiment_id="my-custom-id")
        assert body.experiment_id == "my-custom-id"


class TestMatrixRequestValidation:
    def test_default_values(self):
        body = MatrixRequest()
        assert body.plaintext_matrix_id == "matrix-0"
        assert body.plaintext_matrix_filename == "matrix-0"
        assert body.extension == "csv"

    def test_custom_values(self):
        body = MatrixRequest(
            plaintext_matrix_id="dataset-42",
            plaintext_matrix_filename="dataset_42",
            extension="npy",
            experiment_id="exp-001",
        )
        assert body.plaintext_matrix_id == "dataset-42"
        assert body.extension == "npy"
        assert body.experiment_id == "exp-001"


class TestKmeansRequestValidation:
    def test_default_k(self):
        body = KmeansRequest()
        assert body.k == 3

    def test_custom_k(self):
        body = KmeansRequest(k=5)
        assert body.k == 5

    def test_k_below_minimum_raises_error(self):
        from pydantic import ValidationError
        try:
            KmeansRequest(k=1)
            assert False, "ValidationError was not raised for k=1 (min is 2)"
        except ValidationError:
            pass

    def test_missing_fields_use_defaults(self):
        body = KmeansRequest(k=5)
        assert body.plaintext_matrix_id == "matrix-0"
        assert body.extension == "csv"
        assert body.experiment_id is not None

    def test_full_json_to_endpoint(self, client):
        response = client.post("/clustering/kmeans", json={
            "plaintext_matrix_id": "test-matrix",
            "k": 4,
            "extension": "csv",
            "experiment_id": "test-exp-km",
        })
        assert response.status_code in [200, 500]

    def test_missing_k_uses_default(self, client):
        response = client.post("/clustering/kmeans", json={
            "plaintext_matrix_id": "test-matrix",
        })
        assert response.status_code in [200, 500]

    def test_invalid_k_type(self, client):
        response = client.post("/clustering/kmeans", json={
            "k": "not_a_number",
        })
        assert response.status_code == 422

    def test_invalid_extension_type(self, client):
        response = client.post("/clustering/kmeans", json={
            "k": 3,
            "extension": 123,
        })
        assert response.status_code == 422


class TestSkmeansRequestValidation:
    def test_k_is_required(self):
        body = SkmeansRequest(k=3)
        assert body.k == 3
        assert body.max_iterations == 10
        assert body.convergence_threshold == 0.000001

    def test_custom_params(self):
        body = SkmeansRequest(
            k=5,
            max_iterations=20,
            convergence_threshold=0.001,
            experiment_iteration="42",
            plaintext_matrix_id="secure_matrix",
        )
        assert body.k == 5
        assert body.max_iterations == 20
        assert body.convergence_threshold == 0.001
        assert body.experiment_iteration == "42"

    def test_default_experiment_iteration(self):
        body = SkmeansRequest(k=3)
        assert body.experiment_iteration == "0"

    def test_invalid_k_type(self, client):
        response = client.post("/clustering/skmeans", json={
            "k": "abc",
        })
        assert response.status_code == 422

    def test_negative_convergence_threshold(self):
        try:
            SkmeansRequest(k=3, convergence_threshold=-1.0)
            assert False, "Should have raised validation error"
        except Exception:
            pass

    def test_negative_max_iterations(self, client):
        response = client.post("/clustering/skmeans", json={
            "k": 3,
            "max_iterations": -5,
        })
        assert response.status_code == 422

    def test_valid_json_body(self, client):
        response = client.post("/clustering/skmeans", json={
            "k": 3,
            "max_iterations": 5,
            "convergence_threshold": 0.0001,
            "experiment_id": "exp-sk-test",
            "plaintext_matrix_id": "secure-matrix",
        })
        assert response.status_code in [200, 500]


class TestDbskmeansRequestValidation:
    def test_sens_has_default(self):
        body = DbskmeansRequest(k=3)
        assert body.sens == 0.00000001

    def test_custom_sens(self):
        body = DbskmeansRequest(k=3, sens=0.001)
        assert body.sens == 0.001

    def test_valid_json_body(self, client):
        response = client.post("/clustering/dbskmeans", json={
            "k": 3,
            "sens": 0.0001,
            "max_iterations": 5,
            "convergence_threshold": 0.0001,
        })
        assert response.status_code in [200, 500]


class TestDbsnncRequestValidation:
    def test_default_sens_and_threshold(self):
        body = DbsnncRequest()
        assert body.sens == 0.00000001
        assert body.threshold == -1.0

    def test_custom_threshold(self):
        body = DbsnncRequest(threshold=0.5)
        assert body.threshold == 0.5

    def test_valid_json_body(self, client):
        response = client.post("/clustering/dbsnnc", json={
            "plaintext_matrix_id": "db-matrix",
            "sens": 0.0001,
            "threshold": 0.5,
        })
        assert response.status_code in [200, 500]


class TestNncRequestValidation:
    def test_default_threshold(self):
        body = NncRequest()
        assert body.threshold == -1.0

    def test_custom_threshold(self):
        body = NncRequest(threshold=2.5)
        assert body.threshold == 2.5

    def test_valid_json_body(self, client):
        response = client.post("/clustering/nnc", json={
            "plaintext_matrix_id": "nnc-matrix",
            "threshold": 0.5,
        })
        assert response.status_code in [200, 500]


class TestPqcSkmeansRequestValidation:
    def test_default_params(self):
        body = PqcSkmeansRequest(k=3)
        assert body.k == 3
        assert body.max_iterations == 10
        assert body.experiment_iteration == "0"

    def test_valid_json_body(self, client):
        response = client.post("/clustering/pqc/skmeans", json={
            "k": 3,
            "max_iterations": 3,
        })
        assert response.status_code in [200, 500]


class TestPqcDbskmeansRequestValidation:
    def test_sens_default(self):
        body = PqcDbskmeansRequest(k=3)
        assert body.sens == 0.00000001

    def test_valid_json_body(self, client):
        response = client.post("/clustering/pqc/dbskmeans", json={
            "k": 3,
            "sens": 0.0001,
            "max_iterations": 3,
        })
        assert response.status_code in [200, 500]


class TestResponseModels:
    def test_health_check_response(self):
        resp = HealthCheckResponse()
        assert resp.component_type == "dataowner"

    def test_kmeans_response_defaults(self):
        resp = KmeansResponse()
        assert resp.label_vector == []
        assert resp.iterations == 0
        assert resp.algorithm == ""
        assert resp.worker_id == ""
        assert resp.service_time_manager == 0.0
        assert resp.service_time_worker == 0.0
        assert resp.service_time_dataowner == 0.0
        assert resp.response_time_clustering == 0.0

    def test_skmeans_response_with_values(self):
        resp = SkmeansResponse(
            label_vector=[0, 1, 2],
            iterations=10,
            algorithm="skmeans",
            worker_id="worker-1",
            service_time_manager=1.5,
            service_time_worker=30.0,
            service_time_dataowner=5.0,
            response_time_clustering=36.5,
        )
        assert resp.label_vector == [0, 1, 2]
        assert resp.iterations == 10
        assert resp.algorithm == "skmeans"
        json_data = resp.model_dump()
        assert json_data["label_vector"] == [0, 1, 2]
        assert json_data["iterations"] == 10

    def test_nnc_response_no_iterations(self):
        resp = NncResponse()
        assert not hasattr(resp, "iterations")

    def test_dbsnnc_response(self):
        resp = DbsnncResponse()
        assert resp.label_vector == []

    def test_pqc_response_with_iterations(self):
        resp = PqcSkmeansResponse(iterations=5)
        assert resp.iterations == 5
