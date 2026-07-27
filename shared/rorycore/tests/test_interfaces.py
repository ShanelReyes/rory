import json
import numpy as np
from rory.core.interfaces.cipher_result import CipherResult
from rory.core.interfaces.outsourced_result import OutsourcedDataResult
from rory.core.interfaces.client_response import ClientResponse
from rory.core.interfaces.createroryworker import CreateRoryWorker
from rory.core.interfaces.logger_metrics import LoggerMetrics
from rory.core.interfaces.metricsResult_external import MetricsResultExternal
from rory.core.interfaces.metricsResult_internal import MetricsResultInternal
from rory.core.interfaces.rory_result import RoryResult
from rory.core.interfaces.rorymanagerrequest import RoryRequestManager
from rory.core.interfaces.roryrequest import RoryRequestClient
from rory.core.interfaces.worker import Worker


def test_cipher_result_init():
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = CipherResult(data=data)
    assert np.array_equal(result.data, data)


def test_cipher_result_defaults():
    data = np.zeros((2, 2))
    result = CipherResult(data=data)
    assert np.array_equal(result.data, data)


def test_client_result_init():
    result = OutsourcedDataResult(
        udm_time=0.5,
        UDM=np.array([[1, 2]]),
        encrypted_matrix=np.array([[3, 4]]),
        encrypted_matrix_time=0.3,
        messageIntervals={"k": (0, 1)},
        cypherIntervals={"k": (0, 10)},
        encrypted_threshold=5,
        num_attributes=3,
    )
    assert result.udm_time == 0.5
    assert result.num_attributes == 3
    assert result.encrypted_threshold == 5


def test_client_result_defaults():
    result = OutsourcedDataResult()
    assert result.udm_time == 0
    assert len(result.UDM) == 0
    assert len(result.encrypted_matrix) == 0
    assert result.encrypted_threshold == 0


def test_client_response_init():
    response = ClientResponse(
        label_vector=[0, 1, 0],
        service_time=2.0,
        response_time=1.5,
        algorithm="SKMEANS",
        headers={"x-custom": "val"},
        status=200,
    )
    assert response.label_vector == [0, 1, 0]
    assert response.service_time == 2.0
    assert response.algorithm == "SKMEANS"
    assert response.status == 200


def test_client_response_defaults():
    response = ClientResponse()
    assert response.label_vector == []
    assert response.service_time == 0
    assert response.status == 0


def test_createroryworker_init():
    worker = CreateRoryWorker(
        nodeId="node-1",
        nodeIndex=0,
        image="test_image",
        ports={"docker": 9000, "host": 9001},
    )
    assert worker.nodeId == "node-1"
    assert worker.nodeIndex == 0
    assert worker.image == "test_image"
    assert worker.ports == {"docker": 9000, "host": 9001}


def test_createroryworker_serialize():
    worker = CreateRoryWorker(
        nodeId="node-1",
        nodeIndex=0,
        ports={"docker": 9000, "host": 9001},
    )
    serialized = worker.serialize()
    parsed = json.loads(serialized)
    assert parsed["nodeId"] == "node-1"
    assert parsed["nodeIndex"] == 0


def test_createroryworker_str():
    worker = CreateRoryWorker(
        nodeId="test-node",
        ports={"docker": 9000, "host": 9001},
    )
    s = str(worker)
    assert "test-node" in s


def test_logger_metrics_init():
    lm = LoggerMetrics(
        operation_type="ENCRYPT",
        matrix_id="M1",
        worker_id="W1",
        algorithm="SKMEANS",
        arrival_time=100,
        end_time=200,
        service_time=50,
        latency=10,
        k_value=3,
        m_value=5,
        n_iterations=10,
    )
    assert lm.operation_type == "ENCRYPT"
    assert lm.matrix_id == "M1"
    assert lm.worker_id == "W1"
    assert lm.algorithm == "SKMEANS"
    assert lm.service_time == 50
    assert lm.latency == 10


def test_logger_metrics_str():
    lm = LoggerMetrics(
        operation_type="ENCRYPT",
        arrival_time=0,
        end_time=0,
        service_time=1,
    )
    s = str(lm)
    assert "ENCRYPT" in s


def test_logger_metrics_with_extra_kwargs():
    lm = LoggerMetrics(
        operation_type="ENCRYPT",
        test="EXTRA_KWARG",
    )
    s = str(lm)
    assert "ENCRYPT" in s


def test_metrics_result_external_init():
    mre = MetricsResultExternal(
        adjusted_mutual_information=0.8,
        fowlkes_mallows_index=0.7,
        adjusted_rand_index=0.6,
        jaccard_index=0.5,
    )
    assert mre.adjusted_mutual_information == 0.8
    assert mre.fowlkes_mallows_index == 0.7
    assert mre.adjusted_rand_index == 0.6
    assert mre.jaccard_index == 0.5


def test_metrics_result_external_str():
    mre = MetricsResultExternal(
        adjusted_mutual_information=0.8,
        fowlkes_mallows_index=0.7,
        adjusted_rand_index=0.6,
        jaccard_index=0.5,
    )
    s = str(mre)
    assert "0.8" in s
    assert "0.7" in s


def test_metrics_result_internal_init():
    mri = MetricsResultInternal(
        silhouette_coefficient=0.5,
        davies_bouldin_index=1.2,
        calinski_harabaz_index=100.0,
        dunn_index=0.3,
    )
    assert mri.silhouette_coefficient == 0.5
    assert mri.davies_bouldin_index == 1.2


def test_metrics_result_internal_to_dict():
    mri = MetricsResultInternal(silhouette_coefficient=0.5)
    d = mri.toDict()
    assert d["silhouette_coefficient"] == 0.5


def test_metrics_result_internal_to_json():
    mri = MetricsResultInternal(silhouette_coefficient=0.5)
    j = mri.toJson()
    parsed = json.loads(j)
    assert parsed["silhouette_coefficient"] == 0.5


def test_metrics_result_internal_str():
    mri = MetricsResultInternal(dunn_index=0.3)
    s = str(mri)
    assert "0.3" in s


def test_rory_result_init():
    result = RoryResult(
        label_vector=np.array([0, 1, 0]),
        n_iterations=5,
        response_time=1.0,
        service_time=0.8,
        clustering_time=0.6,
        udm_time=0.2,
        cipher_time=0.1,
    )
    assert np.array_equal(result.label_vector, np.array([0, 1, 0]))
    assert result.n_iterations == 5
    assert result.service_time == 0.8


def test_rory_result_defaults():
    result = RoryResult()
    assert len(result.label_vector) == 0
    assert result.n_iterations == 0


def test_rory_request_manager_init():
    rm = RoryRequestManager(
        requestId="req-1",
        algorithm="SKMEANS",
    )
    assert rm.requestId == "req-1"
    assert rm.algorithm == "SKMEANS"
    assert rm.encryptedMatrixId == "MATRIX_ID"


def test_rory_request_manager_latency():
    rm = RoryRequestManager(
        arrivalTime=100,
        startRequestTime=0,
    )
    assert rm.latency == 100


def test_rory_request_client_init():
    rc = RoryRequestClient(
        requestId="rc-1",
        algorithm="SKMEANS",
        m=5,
        k=3,
    )
    assert rc.requestId == "rc-1"
    assert rc.algorithm == "SKMEANS"
    assert rc.m == 5
    assert rc.k == 3


def test_rory_request_client_serialize():
    rc = RoryRequestClient(
        requestId="test-id",
        algorithm="KMEANS",
    )
    serialized = rc.serialize()
    parsed = json.loads(serialized)
    assert parsed["requestId"] == "test-id"
    assert parsed["algorithm"] == "KMEANS"


def test_worker_init():
    w = Worker(
        workerId="w-1",
        port=9000,
        balls=[1, 2, 3],
        isStarted=True,
    )
    assert w.workerId == "w-1"
    assert w.port == 9000
    assert w.balls == [1, 2, 3]
    assert w.isStarted is True


def test_worker_defaults():
    w = Worker()
    assert w.workerId is not None
    assert len(w.workerId) > 0
    assert w.isStarted is False


def test_worker_created_at():
    import time

    before = time.time()
    w = Worker()
    after = time.time()
    assert before <= w.createdAt <= after
