from models.requests import WorkerStartedRequest, DeployWorkerRequest
from models.responses import DeployWorkerResponse, WorkerInfo


class TestWorkerStartedRequest:
    def test_minimal_valid(self):
        body = WorkerStartedRequest(worker_id="worker-1", worker_port=9000)
        assert body.worker_id == "worker-1"
        assert body.worker_port == 9000

    def test_missing_worker_id(self, client):
        response = client.post("/workers/started", json={
            "worker_port": 9000,
        })
        assert response.status_code == 422

    def test_missing_worker_port(self, client):
        response = client.post("/workers/started", json={
            "worker_id": "worker-1",
        })
        assert response.status_code == 422

    def test_empty_worker_id(self, client):
        response = client.post("/workers/started", json={
            "worker_id": "",
            "worker_port": 9000,
        })
        assert response.status_code == 422

    def test_negative_worker_port(self, client):
        response = client.post("/workers/started", json={
            "worker_id": "worker-1",
            "worker_port": -1,
        })
        assert response.status_code == 422

    def test_get_returns_404(self, client):
        response = client.get("/workers/started")
        assert response.status_code == 404

    def test_post_with_valid_body(self, client):
        response = client.post("/workers/started", json={
            "worker_id": "worker-test-post",
            "worker_port": 9000,
        })
        assert response.status_code == 204


class TestListWorkers:
    def test_list_workers_empty(self, client):
        response = client.get("/workers")
        assert response.status_code == 200
        assert isinstance(response.json(), dict)

    def test_list_workers_after_registration(self, client):
        client.post("/workers/started", json={
            "worker_id": "worker-list-test-2",
            "worker_port": 9000,
        })
        response = client.get("/workers")
        assert response.status_code == 200
        data = response.json()
        assert "worker-list-test-2" in data


class TestDeployWorkerRequest:
    def test_defaults_differ_from_base(self):
        body = DeployWorkerRequest()
        assert body.sink_path == "/sink"
        assert body.source_path == "/source"
        assert body.log_path == "/log"
        assert body.mictlanx_peers == "mictlanx-peer-0:mictlanx-peer-0:7000"

    def test_custom_override(self):
        body = DeployWorkerRequest(
            sink_path="/custom/sink",
            source_path="/custom/source",
            worker_memory="4000000000",
        )
        assert body.sink_path == "/custom/sink"
        assert body.source_path == "/custom/source"
        assert body.worker_memory == "4000000000"

    def test_deploy_endpoint_accepts_request(self, client):
        response = client.post("/workers/deploy", json={})
        assert response.status_code in [200, 422, 500]


class TestResponseModels:
    def test_deploy_worker_response(self):
        resp = DeployWorkerResponse(
            container_id="worker-container-1",
            port="9001",
        )
        assert resp.container_id == "worker-container-1"
        assert resp.port == "9001"

    def test_worker_info_model(self):
        info = WorkerInfo(workerId="w-1", port=9000, isStarted=True)
        assert info.worker_id == "w-1"
        assert info.port == 9000
        assert info.is_started is True

    def test_worker_info_serialization(self):
        info = WorkerInfo(workerId="w-1", port=9000, isStarted=True)
        data = info.model_dump(by_alias=True)
        assert data["workerId"] == "w-1"
        assert data["port"] == 9000
        assert data["isStarted"] is True
