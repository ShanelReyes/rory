"""
Demo: Nearest Neighbour Clustering (NNC) on plaintext data.
"""
import numpy as np
from rory.core.clustering.nnc import Nnc
from rory.core.utils.utils import Utils

print("=" * 60)
print("DEMO NNC — Nearest Neighbour Clustering")
print("=" * 60)

plain = np.array([
    [1.0, 2.0],
    [10.0, 11.0],
    [1.5, 2.5],
    [10.5, 11.5],
    [20.0, 21.0],
], dtype=np.float64)

dm = Utils.calculate_DM(plain)
threshold = Utils.get_threshold(dm)

print(f"Dataset: {plain.shape[0]} points x {plain.shape[1]} features")
print(f"Distance matrix:\n{dm}")
print(f"Threshold: {threshold:.4f}")

result = Nnc().fit(distance_matrix=dm, threshold=threshold)
print(f"\nLabels:      {result.label_vector}")
print(f"Response time: {result.response_time:.4f}s")
print("=" * 60)
print("NNC demo completed successfully.")
print("=" * 60)
