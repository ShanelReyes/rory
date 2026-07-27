"""
Demo: Double-Blind Secure KMeans PQC (CKKS + FDHOPE).
"""
import numpy as np
from rory.core.enums import Algorithm, Scheme
from rory.core.security.dataowner import DataOwner
from rory.core.security.scheme_params import CkksAndFdhopeParams
from rory.core.clustering.secure.pqc.dbskmeans import DBSKMeans
from rory.core.utils.constants import Constants

print("=" * 60)
print("DEMO DBSKMeans PQC — Double-Blind Secure KMeans (CKKS + FDHOPE)")
print("  Generating CKKS keys in memory...")
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
n_iterations = 3

do = DataOwner.with_algorithm(Algorithm.DBSKMEANS_PQC) \
    .with_scheme(Scheme.CKKS_AND_FDHOPE) \
    .with_scheme_params(CkksAndFdhopeParams(
        security_level=128, enable_relinearize=True, enable_rotate=True,
    )) \
    .build()
outsourced = do.outsourcedData(plaintext_matrix=plain)

ckks = do.primary_scheme

zero_shift = np.zeros((k, n_attr))
init_shift = ckks.encrypt_matrix(plaintext_matrix=zero_shift)

dbskmeans = DBSKMeans(scheme=ckks, init_shiftmatrix=init_shift.data)

label_vector = dbskmeans.fit(
    status           = Constants.ClusteringStatus.START,
    k                = k,
    encrypted_matrix = outsourced.encrypted_matrix,
    UDM              = outsourced.UDM,
    Cent_j           = None,
    iterations       = 0,
    n_iterations     = n_iterations,
    num_attributes   = n_attr,
    scheme           = ckks,
)

print(f"Dataset: {plain.shape[0]} points x {plain.shape[1]} features, k={k}")
print(f"Labels: {label_vector}")
print("=" * 60)
print("DBSKMeans PQC demo completed successfully.")
print("=" * 60)
