import requests as R
from uuid import uuid4

default_headers = {
    "Experiment-Id": uuid4().hex[:10],
    
    # IDs de Matrices
    "Plaintext-Matrix-Train-Id": "dataset1_train",
    "Plaintext-Matrix-Train-Label-Id": "label_vector_train",
    "Plaintext-Matrix-Test-Id": "dataset1_test",
    "Plaintext-Matrix-Test-Label-Id": "label_vector_test",
    
    # Nombres de Archivos y Extensión
    "Plaintext-Matrix-Train-Filename": "dataset1_train",
    "Plaintext-Matrix-Test-Filename": "dataset1_test",
    "Plaintext-Matrix-Train-Label-Filename": "label_vector_train",
    "Plaintext-Matrix-Test-Label-Filename": "label_vector_test",
    "Extension": "npy",
    
    # Hiperparámetros
    "Epochs": "1",
    "Learning-Rate": "0.01",
}


def test_client():
    result = R.post("http://localhost:3000/machinelearning/logisticregression", headers=default_headers
           )
    print(result.json())

    assert result.status_code == 200

# default_headers = {
#     "Experiment-Id": uuid4().hex[:10],
#     "Experiment-Iteration": "0",
    
#     # IDs de Matrices
#     "Plaintext-Matrix-Train-Id": "dataset1_train",
#     "Plaintext-Matrix-Train-Label-Id": "label_vector_train",
#     "Plaintext-Matrix-Test-Id": "dataset1_test",
#     "Plaintext-Matrix-Test-Label-Id": "label_vector_test",
    
#     # Nombres de Archivos y Extensión
#     "Plaintext-Matrix-Train-Filename": "dataset1_train",
#     "Plaintext-Matrix-Test-Filename": "dataset1_test",
#     "Plaintext-Matrix-Train-Label-Filename": "label_vector_train",
#     "Plaintext-Matrix-Test-Label-Filename": "label_vector_test",
#     "Extension": "npy",
    
#     # Hiperparámetros
#     "Epochs": "1",
#     "Learning-Rate": "0.01",
#     "Accuracy-Threshold": "0.80",
    
#     # Pesos y Bias
#     "Plaintext-Weight-Matrix-Id": "weight-matrix",
#     "Plaintext-Bias-Vector-Id": "bias-vector",
# }


# def test_client():
#     result = R.post("http://localhost:3000/machinelearning/pplr", headers=default_headers
#            )
#     print(result.json())

#     assert result.status_code == 200