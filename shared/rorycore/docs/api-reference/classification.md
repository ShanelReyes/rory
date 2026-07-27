# Classification

## Abstract Base Classes

### ClassificationTask

::: rory.core.tasks.ClassificationTask

### SecureClassificationTask

::: rory.core.tasks.SecureClassificationTask

## KNN

Plaintext K-Nearest Neighbors classifier.

::: rory.core.classification.knn.KNearestNeighbors
    options:
      members:
        - fit
        - predict
        - calculate_distances_and_indexes
        - get_distance
        - manhathan_distance
        - euclidean
        - split_labelvector_from_data

## Logistic Regression

Plaintext logistic regression model for binary classification.

::: rory.core.classification.logistic_regression.LogisticRegression
    options:
      members:
        - fit
        - predict
        - sigmoid_poly

## Secure KNN (Liu-based)

Secure KNN using Liu's symmetric homomorphic encryption scheme.

::: rory.core.classification.secure.conventional.sknn.SecureKNearestNeighbors
    options:
      members:
        - fit
        - get_label_vector
        - calculate_distances
        - get_distance
        - manhathan_distance
        - euclidean
        - split_labelvector_from_data

## Secure KNN (PQC — CKKS)

Secure KNN using CKKS fully homomorphic encryption.

::: rory.core.classification.secure.pqc.sknn.SecureKNearestNeighbors
    options:
      members:
        - fit
        - get_label_vector
        - calculate_distances
        - euclidean
        - split_labelvector_from_data

## PPLR (PQC — CKKS)

Privacy-Preserving Logistic Regression using CKKS homomorphic encryption.

::: rory.core.classification.secure.pqc.pplr.PPLR
    options:
      members:
        - fit
        - predict
        - sigmoid_poly
        - forward_and_error
        - encrypted_gradients
