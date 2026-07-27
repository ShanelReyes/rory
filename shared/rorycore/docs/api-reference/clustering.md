# Clustering

## Abstract Base Classes

### DataMiningTask

::: rory.core.tasks.DataMiningTask
    options:
      members:
        - fit
        - predict

### ClusteringTask

::: rory.core.tasks.ClusteringTask

### SecureClusteringTask

::: rory.core.tasks.SecureClusteringTask
    options:
      members:
        - execute_encrypted_phase
        - execute_plaintext_phase
        - compute_centroid_shift
        - fit

## KMeans

Plaintext KMeans clustering using scikit-learn.

::: rory.core.clustering.kmeans.KMeans
    options:
      members:
        - fit

## NNC

Nearest Neighbour Clustering on plaintext data.

::: rory.core.clustering.nnc.Nnc
    options:
      members:
        - fit
        - getMinDistanceInClusters

## SKMeans (Conventional — Liu)

Distributed Secure KMeans using Liu's symmetric homomorphic encryption.

::: rory.core.clustering.secure.conventional.skmeans.SKMeans
    options:
      members:
        - execute_encrypted_phase
        - execute_plaintext_phase
        - compute_centroid_shift
        - fit

## DBSKMeans (Conventional — Liu)

Double-Blind Secure KMeans using Liu's symmetric encryption.

::: rory.core.clustering.secure.conventional.dbskmeans.DBSKMeans
    options:
      members:
        - execute_encrypted_phase
        - execute_plaintext_phase
        - compute_centroid_shift

## DBSNNC (Conventional — Liu)

Double-Blind Secure Nearest Neighbour Clustering.

::: rory.core.clustering.secure.conventional.dbsnnc.Dbsnnc
    options:
      members:
        - fit
        - getMinDistanceInClusters

## SKMeans (PQC — CKKS)

Distributed Secure KMeans using CKKS fully homomorphic encryption.

::: rory.core.clustering.secure.pqc.skmeans.Skmeans
    options:
      members:
        - execute_encrypted_phase
        - execute_plaintext_phase
        - compute_centroid_shift
        - fit
        - calculateCentroidsObject
        - calculateCentroidsObjectv2
        - compute_centroids

## DBSKMeans (PQC — CKKS)

Double-Blind Secure KMeans using CKKS fully homomorphic encryption.

::: rory.core.clustering.secure.pqc.dbskmeans.DBSKMeans
    options:
      members:
        - execute_encrypted_phase
        - execute_plaintext_phase
        - compute_centroid_shift
        - fit
        - calculateCentroidsObject
