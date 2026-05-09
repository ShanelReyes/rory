import requests as R
from uuid import uuid4

default_headers_lr_train = {
    "Experiment-Id": uuid4().hex[:10],
    "Experiment-Iteration": "0",
    "Plaintext-Matrix-Train-Id": "dataset1_train",
    "Plaintext-Matrix-Train-Label-Id": "label_vector_train",
    "Plaintext-Matrix-Train-Filename": "dataset1_train",
    "Plaintext-Matrix-Train-Label-Filename": "label_vector_train",
    "Extension": "npy",
    "Epochs": "1",
    "Learning-Rate": "0.01",
}

default_headers_lr_predict = {
    "Experiment-Id": uuid4().hex[:10],
    "Experiment-Iteration": "0",
    "Plaintext-Matrix-Test-Id": "dataset1_test",
    "Plaintext-Matrix-Test-Filename": "dataset1_test",
    "Plaintext-Weight-Matrix-Id": "weight-matrix",
    "Plaintext-Weight-Matrix-Filename": "weight-matrix",
    "Plaintext-Bias-Vector-Id": "bias-vector",
    "Plaintext-Bias-Vector-Filename": "bias-vector",
    "Extension": "npy",
}

default_headers_pplr_train = {
    "Experiment-Id": uuid4().hex[:10],
    "Experiment-Iteration": "0",
    "Plaintext-Matrix-Train-Id": "dataset1_train",
    "Plaintext-Label-Vector-Train-Id": "label_vector_train1",
    # "Plaintext-Matrix-Test-Id": "dataset1_test",
    # "Plaintext-Matrix-Test-Label-Id": "label_vector_test",
    "Plaintext-Matrix-Train-Filename": "dataset1_train",
    # "Plaintext-Matrix-Test-Filename": "dataset1_test",
    "Plaintext-Label-Vector-Train-Filename": "label_vector_train",
    # "Plaintext-Matrix-Test-Label-Filename": "label_vector_test",
    "Extension": "npy",
    "Epochs": "1",
    "Learning-Rate": "0.01",
    "Accuracy-Threshold": "0.80",
    # "Plaintext-Weight-Id": "weight-matrix",
    # "Plaintext-Bias-Id": "bias-vector",
}

default_headers_pplr_predict = {
    "Experiment-Id": uuid4().hex[:10],
    "Experiment-Iteration": "0",
    "Plaintext-Matrix-Test-Id": "dataset1_test",
    "Plaintext-Matrix-Test-Filename": "dataset1_test",
    "Extension": "npy",
    "Epochs": "1",
    "Learning-Rate": "0.01",
    "Accuracy-Threshold": "0.80",
    "Plaintext-Weight-Matrix-Id": "weight-matrix",
    "Plaintext-Bias-Vector-Id": "bias-vector",
    "Accuracy-Threshold": "0.80",
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
