"""
Demo: Double-Blind Secure KMeans (conventional, Liu + FDHOPE).
"""
import numpy as np
from rory.core.enums import Algorithm, Scheme
from rory.core.security.dataowner import DataOwner
from rory.core.security.scheme_params import LiuAndFdhopeParams
from rory.core.clustering.secure.conventional.dbskmeans import DBSKMeans
from rory.core.utils.constants import Constants

print("=" * 60)
print("DEMO DBSKMeans — Double-Blind Secure KMeans (Liu + FDHOPE)")
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

do = DataOwner.with_algorithm(Algorithm.DBSKMEANS) \
    .with_scheme(Scheme.LIU_AND_FDHOPE) \
    .with_scheme_params(LiuAndFdhopeParams(
        security_level=128, _round=True, decimals=6, seed=1,
    )) \
    .build()
outsourced = do.outsourcedData(plaintext_matrix=plain)

liu = do.primary_scheme

dbskmeans = DBSKMeans()
label_vector = dbskmeans.fit(
    status           = Constants.ClusteringStatus.START,
    k                = k,
    encrypted_matrix = outsourced.encrypted_matrix,
    UDM              = outsourced.UDM,
    Cent_j           = None,
    iterations       = 0,
    n_iterations     = 10,
    num_attributes   = n_attr,
    scheme           = liu,
    sk               = liu.sk,
    m                = liu.m,
)

print(f"Dataset: {plain.shape[0]} points x {plain.shape[1]} features, k={k}")
print(f"Labels: {label_vector}")
print("=" * 60)
print("DBSKMeans demo completed successfully.")
print("=" * 60)
