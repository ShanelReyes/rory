from typing import List
from time import time
from rory.core.utils.utils import Utils
from rory.core.interfaces.rory_result import RoryResult
from rory.core.algorithms import StandardClustering
import numpy.typing as npt


class Nnc(StandardClustering):
    """Nearest Neighbour Clustering operating on plaintext data.

    Clusters records using a distance matrix and a threshold. Each record
    is assigned to the nearest cluster or forms a new cluster if the
    minimum distance exceeds the threshold.
    """

    def fit(
        self, distance_matrix: npt.NDArray, threshold: float
    ) -> RoryResult:
        """Run the single-pass nearest neighbour clustering on plaintext data.

        Iterates over records and assigns each to the nearest cluster
        if the minimum distance is within threshold, otherwise creates
        a new cluster.

        Args:
            distance_matrix: Pairwise distance matrix between records.
            threshold: Maximum distance for assigning to an existing cluster.

        Returns:
            RoryResult: Result container with label vector and service time.
        """
        startTime = time()
        distance_matrix_shape = Utils.getShapeOfMatrix(distance_matrix)
        c_indexes: List[List[int]] = [[0]]
        for record_index in range(1, distance_matrix_shape[0]):
            cluster_m_index, delta = Utils.getMinDistanceInClusters(
                c_indexes, record_index, distance_matrix
            )
            if delta <= threshold:
                c_indexes[cluster_m_index].append(record_index)
            else:
                c_indexes.append([record_index])
        label_vector = Utils.get_labelvector_from_indexes(
            shape=distance_matrix_shape[0], c_indexes=c_indexes
        )
        return self._build_result(label_vector, startTime)
