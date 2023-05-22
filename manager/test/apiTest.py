import unittest
import os
import sys
import time
from pathlib import Path
path_root      = Path(__file__).parent.absolute()
(path_root, _) = os.path.split(path_root)
newPath = str(path_root)+"/src"
sys.path.append(newPath)
# ====================================================
import pytest
import json
from main import create_app


@pytest.fixture()
def client():
    app = create_app()
    app.config["TESTING"]= True
    with app.app_context():
        with app.test_client() as client:
            yield client

def test_clustering_get(client):
    response    = client.get("/clustering/secure")
    _response   = response
    status      = response.status
    status_code = response.status_code
    assert status_code == 200
def test_clustering_post(client):
    
    N = 5
    nodes_payload = [{ 
        "binId":"bin-"+str(i),
        "port": 3000+i,
        "balls":[]
    } for i in range(N)]
    
    requests = map(lambda np: {"data": json.dumps(np),"content_type": "application/json" }  ,nodes_payload)


    responses = [client.post("/bins", **req) for req in requests]
    statues   = lambda responses: list(map(lambda x: x.status , responses))
    print(statues(responses))
    # assert response.status_code == 200
# _________________________________
    REQUEST_COUNT = 10
    data        = [json.dumps(
        {
            "requestId":"request-"+str(i),
            "startRequestTime":time.time(),
            "algorithm":"ROUND_ROBIN",
            "dataset_uri":"http://localhost:6000",
            "metadata":{
                "k":3,
                "m":3
            }

        }
    ) for i in range(REQUEST_COUNT)]

    responses   = [client.post(
        "/clustering/secure",
        data = datum,
        content_type = "application/json"
        ) for datum in data
    ]
    print(statues(responses))
    # _response   = response
    # status      = response.status
    # status_code = response.status_code
    # print(status)
    # assert status_code == 200

    # return app.test_client()
