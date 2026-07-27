"""
Demo: Plain KMeans clustering (wrapper over sklearn).

Shows basic KMeans clustering on a toy dataset without encryption.
"""

import numpy as np
from rory.core.clustering.kmeans import KMeans

print("=" * 60)
print("DEMO KMeans — Plain Clustering (sklearn wrapper)")
print("=" * 60)

# ---- Toy dataset ----
np.random.seed(42)
dataset = np.random.rand(20, 2) * 10

print(f"\nDataset: {dataset.shape[0]} points, {dataset.shape[1]} features")
print(f"First 3 rows:\n{dataset[:3]}")

# ---- Fit ----
kmeans = KMeans()
result = kmeans.fit(plaintext_matrix=dataset, k=3)
label_vector = result.label_vector

print(f"\nClusters (k=3):")
for cluster_id in range(3):
    points = dataset[label_vector == cluster_id]
    c_x = float(points[:, 0].mean()) if len(points) > 0 else float('nan')
    c_y = float(points[:, 1].mean()) if len(points) > 0 else float('nan')
    print(f"  Cluster {cluster_id}: {len(points)} points — centroid ~({c_x:.2f}, {c_y:.2f})")

print(f"\nLabel vector: {label_vector}")
print(f"\n{'=' * 60}")
print("KMeans demo completed successfully.")
print("=" * 60)
