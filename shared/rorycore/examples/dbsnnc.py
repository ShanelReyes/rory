"""
Demo: Double-Blind Secure NNC (conventional, Liu + FDHOPE).
"""
import numpy as np
from rory.core.enums import Algorithm, Scheme
from rory.core.security.dataowner import DataOwner
from rory.core.security.scheme_params import LiuAndFdhopeParams
from rory.core.clustering.secure.conventional.dbsnnc import Dbsnnc

print("=" * 60)
print("DEMO DBSNNC — Double-Blind Secure NNC (Liu + FDHOPE)")
print("=" * 60)

plain = np.array([
    [1.0, 2.0],
    [10.0, 11.0],
    [1.5, 2.5],
    [10.5, 11.5],
    [20.0, 21.0],
], dtype=np.float64)

do = DataOwner.with_algorithm(Algorithm.DBSNNC) \
    .with_scheme(Scheme.LIU_AND_FDHOPE) \
    .with_scheme_params(LiuAndFdhopeParams(
        security_level=128, _round=True, decimals=6, seed=1,
    )) \
    .build()
outsourced = do.outsourcedData(plaintext_matrix=plain)

result = Dbsnnc().fit(
    distance_matrix = outsourced.UDM,
    threshold       = outsourced.encrypted_threshold,
)

print(f"Dataset: {plain.shape[0]} points x {plain.shape[1]} features")
print(f"Labels:        {result.label_vector}")
print(f"Response time: {result.response_time:.4f}s")
print("=" * 60)
print("DBSNNC demo completed successfully.")
print("=" * 60)
