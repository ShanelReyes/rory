import requests as R
from uuid import uuid4

default_headers_skmeans = {
    "Plaintext-Matrix-Id":"dataset1_train",
    "Plaintext-Matrix-Filename":"dataset1_train",
    "Extension":"npy",
    "K":"2",
    "Experiment-Iteration":"1"
    ""
}

def test_skmeans():
    result = R.post(
        "http://localhost:3000/clustering/skmeans",
        headers=default_headers_skmeans
           )
    if result.status_code != 200:
        print("Error:", result.status_code, result.text)
    else:   
        print("Training response:", result.json())
    assert result.status_code == 200