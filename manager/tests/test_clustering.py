from models.requests import SecureWorkerRequest, WorkerDeploymentConfig
from models.responses import HealthCheckResponse, SecureWorkerResponse


class TestHealthCheck:
    def test_health_check_get(self, client):
        response = client.get("/clustering/test")
        assert response.status_code == 200
        assert response.json()["component_type"] == "manager"

    def test_health_check_post(self, client):
        response = client.post("/clustering/test")
        assert response.status_code == 200
        assert response.json()["component_type"] == "manager"


class TestWorkerDeploymentConfig:
    def test_all_defaults(self):
        body = WorkerDeploymentConfig()
        assert body.worker_memory == "1000000000"
        assert body.worker_cpu == "1"
        assert body.debug == "0"
        assert body.reload == "0"
        assert body.liu_round == "1"
        assert body.sink_path == "/rory/sink"
        assert body.source_path == "/rory/source"
        assert body.log_path == "/rory/log"
        assert body.testing == "0"
        assert body.max_iterations == "10"
        assert body.m == "3"
        assert body.max_threads == "4"
        assert body.mictlanx_peers == "mictlanx-router-0:localhost:60666"
        assert body.mictlanx_lb_algorithm == "2CHOICES_UF"
        assert body.mictlanx_debug == "0"
        assert body.mictlanx_daemon == "0"
        assert body.mictlanx_show_metrics == "0"
        assert body.mictlanx_max_workers == "4"
        assert body.mictlanx_disabled_log == "1"

    def test_custom_values(self):
        body = WorkerDeploymentConfig(
            worker_memory="2000000000",
            worker_cpu="2",
            max_iterations="20",
        )
        assert body.worker_memory == "2000000000"
        assert body.worker_cpu == "2"
        assert body.max_iterations == "20"

    def test_optional_fields_are_none(self):
        body = WorkerDeploymentConfig()
        assert body.host_port is None
        assert body.container_id is None
        assert body.container_port is None


class TestSecureWorkerRequest:
    def test_defaults(self):
        body = SecureWorkerRequest()
        assert body.start_request_time == "0"
        assert body.get_worker_start_time == "0"
        assert body.matrix_id == "matrix0"

    def test_with_algorithm(self):
        body = SecureWorkerRequest(algorithm="KMEANS")
        assert body.algorithm == "KMEANS"

    def test_algorithm_is_optional(self):
        body = SecureWorkerRequest()
        assert body.algorithm is None

    def test_secure_endpoint_post_rejected(self, client):
        response = client.post("/clustering/secure", json={
            "algorithm": "KMEANS",
            "matrix_id": "test-matrix",
        })
        assert response.status_code == 403

    def test_secure_endpoint_get(self, client):
        response = client.get("/clustering/secure")
        assert response.status_code in [200, 403, 500]


class TestResponseModels:
    def test_health_check_response(self):
        resp = HealthCheckResponse()
        assert resp.component_type == "manager"

    def test_secure_worker_response(self):
        resp = SecureWorkerResponse(
            worker_id="worker-0",
            worker_port="9000",
            service_time=0.5,
        )
        assert resp.worker_id == "worker-0"
        assert resp.worker_port == "9000"
        assert resp.service_time == 0.5
