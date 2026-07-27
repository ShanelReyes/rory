"""
Demo: K-Nearest Neighbors (plaintext).
"""
import numpy as np
from rory.core.classification.knn import KNearestNeighbors

print("=" * 60)
print("DEMO KNN — K-Nearest Neighbors (plaintext)")
print("=" * 60)

model = np.array([
    [2, 2, 0],
    [4, 8, 1],
    [7, 7, 0],
    [7, 8, 1],
    [3, 9, 0],
], dtype=np.float64)

dataset = np.array([
    [7, 8],
    [3, 9],
    [5, 4],
], dtype=np.float64)

model_data, model_labels = KNearestNeighbors.split_labelvector_from_data(dataset=model)

KNearestNeighbors.fit(model=model_data, model_labels=model_labels)

predictions = KNearestNeighbors.predict(
    dataset      = dataset,
    model        = model_data,
    model_labels = model_labels,
    distance     = "MANHATHAN",
)

print(f"Model: {model.shape[0]} points x {model.shape[1] - 1} features")
print(f"Dataset: {dataset.shape[0]} points")
print(f"Model labels: {model_labels.tolist()}")
print(f"Predictions (MANHATHAN): {predictions.tolist()}")

predictions_euc = KNearestNeighbors.predict(
    dataset      = dataset,
    model        = model_data,
    model_labels = model_labels,
    distance     = "EUCLIDEAN",
)
print(f"Predictions (EUCLIDEAN): {predictions_euc.tolist()}")

print("=" * 60)
print("KNN demo completed successfully.")
print("=" * 60)
