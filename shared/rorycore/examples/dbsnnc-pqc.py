"""
Demo: Double-Blind Secure NNC PQC (CKKS + FDHOPE).
"""
import numpy as np
from rory.core.enums import Algorithm, Scheme
from rory.core.security.dataowner import DataOwner
from rory.core.security.scheme_params import CkksAndFdhopeParams
from rory.core.clustering.secure.pqc.dbsnnc import Dbsnnc

print("=" * 60)
print("DEMO DBSNNC PQC — Double-Blind Secure NNC (CKKS + FDHOPE)")
print("  Generating CKKS keys in memory...")
print("=" * 60)

plain = np.array([
    [1.0, 2.0],
    [10.0, 11.0],
    [1.5, 2.5],
    [10.5, 11.5],
    [20.0, 21.0],
], dtype=np.float64)

do = DataOwner.with_algorithm(Algorithm.DBSNNC_PQC) \
    .with_scheme(Scheme.CKKS_AND_FDHOPE) \
    .with_scheme_params(CkksAndFdhopeParams(
        security_level=128, enable_relinearize=True, enable_rotate=True,
    )) \
    .build()
outsourced = do.outsourcedData(plaintext_matrix=plain)

ckks = do.primary_scheme
print(f"  Keys generated. n_features={ckks.n_features}")

result = Dbsnnc().fit(
    distance_matrix = outsourced.UDM,
    threshold       = outsourced.encrypted_threshold,
)

print(f"Dataset: {plain.shape[0]} points x {plain.shape[1]} features")
print(f"Labels:        {result.label_vector}")
print(f"Response time: {result.response_time:.4f}s")
print("=" * 60)
print("DBSNNC PQC demo completed successfully.")
print("=" * 60)
