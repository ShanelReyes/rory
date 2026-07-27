"""
Demo: Secure K-Nearest Neighbors PQC (CKKS-based).
"""
import numpy as np
from rory.core.enums import Algorithm, Scheme
from rory.core.security.dataowner import DataOwner
from rory.core.security.scheme_params import CkksParams
from rory.core.classification.secure.pqc.sknn import SecureKNearestNeighbors

print("=" * 60)
print("DEMO SKNN PQC — Secure KNN (CKKS)")
print("  Generating CKKS keys in memory...")
print("=" * 60)

model_plain = np.array([
    [1.0, 2.0],
    [5.0, 6.0],
], dtype=np.float32)

dataset_plain = np.array([
    [1.5, 2.5],
], dtype=np.float32)

model_labels = np.array([0, 1], dtype=np.float64)

do = DataOwner.with_algorithm(Algorithm.SKNN_PQC) \
    .with_scheme(Scheme.CKKS) \
    .with_scheme_params(CkksParams(
        security_level=128, enable_relinearize=True, enable_rotate=True,
    )) \
    .build()

model_result = do.outsourcedData(plaintext_matrix=model_plain)
dataset_result = do.outsourcedData(plaintext_matrix=dataset_plain)

ckks = do.primary_scheme
print(f"  Keys generated. n_features={ckks.n_features}")

model_shape = model_plain.shape
dataset_shape = dataset_plain.shape

SecureKNearestNeighbors.fit(
    model=model_result.encrypted_matrix, model_labels=model_labels,
)

predictions = SecureKNearestNeighbors.predict(
    dataset=dataset_result.encrypted_matrix,
    model=model_result.encrypted_matrix,
    model_labels=model_labels,
    distance="EUCLIDEAN",
    model_shape=model_shape,
    dataset_shape=dataset_shape,
    scheme=ckks,
)

print(f"Model: {model_plain.shape[0]} points x {model_plain.shape[1]} features")
print(f"Dataset: {dataset_plain.shape[0]} points")
print(f"Model labels: {model_labels.tolist()}")
print(f"Predictions (EUCLIDEAN): {predictions.tolist()}")

print("=" * 60)
print("SKNN PQC demo completed successfully.")
print("=" * 60)
