# Utils

## Utils

Common helper functions for matrix operations, clustering support, and
CKKS-level operations.

::: rory.core.utils.utils.Utils
    options:
      members:
        - get_threshold
        - get_labelvector_from_indexes
        - generate_centroids
        - getShapeOfMatrix
        - verifyZero
        - fillLabelVector
        - calculateSimilarity
        - populateClusters
        - populateClustersObject
        - calculateCentroids
        - compute_mean_relative_error
        - verify_mean_error
        - get_scale
        - align
        - rebind_ct
        - normalize_scale
        - safe_add
        - safe_sub
        - safe_multiply
        - mul_plain_scalar
        - add_plain_scalar
        - dot_cipher_garbage

## Constants

Namespace holding constant identifiers for clustering statuses, algorithms,
and ML algorithms.

::: rory.core.utils.constants.Constants
    options:
      members:
        - ClusteringStatus
        - ClusteringAlgorithms
        - ClassificationAlgorithms
        - MachineLearningAlgorithms

## Interfaces

Result objects and protocol contracts used across Rory layers.

### CipherschemeResult

::: rory.core.interfaces.cipherscheme_result.CipherschemeResult

### ClientResult

::: rory.core.interfaces.client_result.ClientResult

### RoryResult

::: rory.core.interfaces.rory_result.RoryResult

### RoryManager

::: rory.core.interfaces.rorymanager.RoryManager
    options:
      members:
        - getWorker

### RoryWorker

::: rory.core.interfaces.roryworker.RoryWorker
    options:
      members:
        - run

## Validation Index

Clustering validation metrics (deprecated).

### Internal Metrics

::: rory.core.validation_index.metrics_deprecated.internal_validation_indexes

### External Metrics

::: rory.core.validation_index.metrics_deprecated.external_validation_indexes

### Dunn Index

::: rory.core.validation_index.validationindex_deprecated.dunn

### Davis-Bouldin Index

::: rory.core.validation_index.validationindex_deprecated.davisbouldin
