import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


@pytest.mark.skip(reason="Integration test requiring running manager, worker, and MictlanX services")
def test_dbskmeans():
    result = client.post("/clustering/dbskmeans", json={
        "k": 3,
        "max_iterations": 5,
        "convergence_threshold": 0.000001,
        "sens": 0.00000001,
        "plaintext_matrix_id": "matrix0",
        "plaintext_matrix_filename": "matrix0",
        "extension": "csv",
        "experiment_id": "int-test-dbskmeans",
    })
    if result.status_code != 200:
        print("Error:", result.status_code, result.text)
    else:
        print("DBSKMeans response:", result.json())
    assert result.status_code == 200
