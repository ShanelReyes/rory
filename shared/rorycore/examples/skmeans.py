"""
Demo: Secure KMeans (conventional, Liu-based).
"""
import numpy as np
from rory.core.enums import Algorithm, Scheme
from rory.core.security.dataowner import DataOwner
from rory.core.security.scheme_params import LiuParams
from rory.core.clustering.secure.conventional.skmeans import SKMeans
from rory.core.utils.constants import Constants

print("=" * 60)
print("DEMO SKMeans — Secure KMeans (Liu)")
print("=" * 60)

plain = np.array([
    [1.0, 2.0],
    [10.0, 11.0],
    [1.5, 2.5],
    [10.5, 11.5],
    [20.0, 21.0],
], dtype=np.float64)

k = 2
n_attr = plain.shape[1]

do = DataOwner.with_algorithm(Algorithm.SKMEANS) \
    .with_scheme(Scheme.LIU) \
    .with_scheme_params(LiuParams(
        security_level=128, _round=True, decimals=6, seed=1,
    )) \
    .build()
outsourced = do.outsourcedData(plaintext_matrix=plain)

liu = do.primary_scheme

skmeans = SKMeans()
label_vector = skmeans.fit(
    status=Constants.ClusteringStatus.START,
    k=k,
    encrypted_matrix=outsourced.encrypted_matrix,
    UDM=outsourced.UDM,
    Cent_j=None,
    iterations=0,
    n_iterations=10,
    num_attributes=n_attr,
    scheme=liu,
    sk=liu.sk,
    m=liu.m,
)

print(f"Dataset: {plain.shape[0]} points x {plain.shape[1]} features, k={k}")
print(f"Labels: {label_vector}")
print("=" * 60)
print("SKMeans demo completed successfully.")
print("=" * 60)
