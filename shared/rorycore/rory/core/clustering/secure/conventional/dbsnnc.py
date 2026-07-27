from typing import List
from time import time
from rory.core.utils.utils import Utils
from rory.core.interfaces.rory_result import RoryResult
from rory.core.algorithms import ConventionalClustering
import numpy.typing as npt
from option import Result


class Dbsnnc(ConventionalClustering):
    """Double Blind Secure Nearest Neighbour Clustering (DBSNNC).

    Clusters an encrypted dataset using a secure nearest neighbour
    approach. Operates on a distance matrix with an encrypted threshold.
    This is a single-pass algorithm (no iterative phases).
    """

    def execute_encrypted_phase(
        self, *args, **kwargs
    ) -> Result:
        """Not applicable: DBSNNC is a single-pass algorithm.

        The double-blind secure NNC operates on a distance matrix and
        does not require iterative encrypted phases.

        Args:
            *args: Ignored.
            **kwargs: Ignored.

        Returns:
            Result: Never returns normally.

        Raises:
            NotImplementedError: Always raised.
        """
        raise NotImplementedError

    def execute_plaintext_phase(
        self, *args, **kwargs
    ) -> npt.NDArray:
        """Not applicable: DBSNNC is a single-pass algorithm.

        The double-blind secure NNC operates on a distance matrix and
        does not require iterative plaintext phases.

        Args:
            *args: Ignored.
            **kwargs: Ignored.

        Returns:
            npt.NDArray: Never returns normally.

        Raises:
            NotImplementedError: Always raised.
        """
        raise NotImplementedError

    def compute_centroid_shift(
        self, *args, **kwargs
    ) -> npt.NDArray:
        """Not applicable: DBSNNC is not centroid-based.

        Nearest neighbour clustering uses distance thresholds rather
        than centroid shifts.

        Args:
            *args: Ignored.
            **kwargs: Ignored.

        Returns:
            npt.NDArray: Never returns normally.

        Raises:
            NotImplementedError: Always raised.
        """
        raise NotImplementedError

    def fit(
        self, distance_matrix: npt.NDArray, threshold: float
    ) -> RoryResult:
        """Run the single-pass double-blind secure nearest neighbour clustering.

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
