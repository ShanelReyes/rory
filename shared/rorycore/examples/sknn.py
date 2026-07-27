"""
Demo: Secure K-Nearest Neighbors (conventional, Liu-based).
"""
import numpy as np
from rory.core.enums import Algorithm, Scheme
from rory.core.security.dataowner import DataOwner
from rory.core.security.scheme_params import LiuParams
from rory.core.classification.secure.conventional.sknn import SecureKNearestNeighbors

print("=" * 60)
print("DEMO SKNN — Secure KNN (Liu)")
print("=" * 60)

model_plain = np.array([
    [1.0, 2.0, 0.0],
    [5.0, 6.0, 1.0],
    [2.0, 3.0, 0.0],
], dtype=np.float64)

dataset_plain = np.array([
    [1.5, 2.5],
    [5.5, 6.5],
], dtype=np.float64)

do = DataOwner.with_algorithm(Algorithm.SKNN) \
    .with_scheme(Scheme.LIU) \
    .with_scheme_params(LiuParams(
        security_level=128, _round=True, decimals=6, seed=1,
    )) \
    .build()

model_result = do.outsourcedData(plaintext_matrix=model_plain[:, :2])
dataset_result = do.outsourcedData(plaintext_matrix=dataset_plain)

liu = do.primary_scheme
model_labels = model_plain[:, 2].astype(np.float64)

SecureKNearestNeighbors.fit(
    model=model_result.encrypted_matrix, model_labels=model_labels,
)

predictions = SecureKNearestNeighbors.predict(
    dataset=dataset_result.encrypted_matrix,
    model=model_result.encrypted_matrix,
    model_labels=model_labels,
    distance="MANHATHAN",
    scheme=liu, sk=liu.sk,
)

print(f"Model: {model_plain.shape[0]} points x {model_plain.shape[1] - 1} features")
print(f"Dataset: {dataset_plain.shape[0]} points")
print(f"Model labels: {model_labels.tolist()}")
print(f"Predictions (MANHATHAN): {predictions.tolist()}")

print("=" * 60)
print("SKNN demo completed successfully.")
print("=" * 60)
