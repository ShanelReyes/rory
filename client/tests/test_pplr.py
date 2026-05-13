import requests as R
from uuid import uuid4

default_headers_lr_train = {
    "Experiment-Id": uuid4().hex[:10],
    "Experiment-Iteration": "0",
    "Plaintext-Matrix-Train-Id": "dataset1_train",
    "Plaintext-Label-Vector-Train-Id": "label_vector_train1",
    "Plaintext-Matrix-Train-Filename": "dataset1_train",
    "Plaintext-Label-Vector-Train-Filename": "label_vector_train",
    "Extension": "npy",
    "Epochs": "1",
    "Learning-Rate": "0.01",
}

default_headers_lr_predict = {
    "Experiment-Id": uuid4().hex[:10],
    "Experiment-Iteration": "0",
    "Plaintext-Matrix-Test-Id": "dataset1_test",
    "Plaintext-Matrix-Test-Filename": "dataset1_test",
    "Plaintext-Matrix-Train-Id": "dataset1_train",
    "Extension": "npy",
}

default_headers_pplr_train = {
    "Experiment-Id": uuid4().hex[:10],
    "Experiment-Iteration": "0",
    "Plaintext-Matrix-Train-Id": "dataset1_train",
    "Plaintext-Label-Vector-Train-Id": "label_vector_train1",
    "Plaintext-Matrix-Train-Filename": "dataset1_train",
    "Plaintext-Label-Vector-Train-Filename": "label_vector_train",
    "Extension": "npy",
    "Epochs": "1",
    "Learning-Rate": "0.01",
    "Accuracy-Threshold": "0.80",
}

default_headers_pplr_predict = {
    "Experiment-Id": uuid4().hex[:10],
    "Experiment-Iteration": "0",
    "Plaintext-Matrix-Test-Id": "dataset1_test",
    "Plaintext-Matrix-Test-Filename": "dataset1_test",
    "Extension": "npy",
    "Epochs": "1",
    "Plaintext-Matrix-Train-Id": "dataset1_train",
    "Plaintext-Weight-Matrix-Id": "weight-matrix",
    "Plaintext-Bias-Vector-Id": "bias-vector",
    "Accuracy-Threshold": "0.80",
}

default_headers_pplr_predict_worker = {
    "Experiment-Id": uuid4().hex[:10],
    "Encrypted-Matrix-Test-Id": "encrypteddataset1_test",
    "Encrypted-Weights-Id": "dataset1_encryptedweights",
    "Encrypted-Bias-Id": "dataset1_encryptedbias",
}

def test_client_pplr_train():
    result = R.post(
        "http://localhost:3000/machine-learning/pplr/train",
        headers=default_headers_pplr_train
           )
    if result.status_code != 200:
        print("Error:", result.status_code, result.text)
    else:   
        print(result.json())

    assert result.status_code == 200

def test_client_pplr_predict():
    result = R.post(
        "http://localhost:3000/machine-learning/pplr/predict",
        headers=default_headers_pplr_predict
           )
    if result.status_code != 200:
        print("Error:", result.status_code, result.text)
    else:   
        print(result.json())

    assert result.status_code == 200

def test_worker_pplr_predict():
    result = R.post(
        "http://localhost:9000/machine-learning/pplr/predict",
        headers=default_headers_pplr_predict_worker
           )
    if result.status_code != 200:
        print("Error:", result.status_code, result.text)
    else:   
        print(result.json())

    assert result.status_code == 200

def test_client_logistic_regression_train():
    result = R.post(
        "http://localhost:3000/machine-learning/logistic-regression/train",
        headers=default_headers_lr_train
    )
    if result.status_code != 200:
        print("Error:", result.status_code, result.text)
    else:   
        print(result.json())

    assert result.status_code == 200

def test_client_logistic_regression_predict():
    result = R.post(
        "http://localhost:3000/machine-learning/logistic-regression/predict",
        headers=default_headers_lr_predict
    )
    if result.status_code != 200:
        print("Error:", result.status_code, result.text)
    else:   
        print(result.json())

    assert result.status_code == 200
