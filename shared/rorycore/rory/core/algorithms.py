from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List, Optional, TYPE_CHECKING
import numpy.typing as npt
from time import time
from option import Result
from rory.core.utils.utils import Utils
from rory.core.interfaces.rory_result import RoryResult
if TYPE_CHECKING:
	from Pyfhel import PyCtxt


# --- Clustering ---


class ClusteringAlgorithm(ABC):
    """Root for all clustering algorithms in Rory.

    Provides common utilities: result building and label vector generation.

    Subclasses are grouped into three families:
      - StandardClustering  — plaintext algorithms
      - ConventionalClustering — secure algorithms (Liu / FDHOPE)
      - PqcClustering       — post-quantum secure algorithms (CKKS / FDHOPE)
    """

    @abstractmethod
    def fit(self, **kwargs) -> RoryResult:
        """Execute the clustering algorithm.

        Returns:
            RoryResult with label_vector and response_time.
        """
        ...

    def _build_result(
        self,
        label_vector: List[int],
        start_time: float,
        n_iterations: int = 0,
        service_time: float = 0,
    ) -> RoryResult:
        """Build a RoryResult with response time and optional extra fields.

        Args:
            label_vector: Cluster label assignments.
            start_time: Monotonic timestamp from time().
            n_iterations: Number of iterations performed.
            service_time: Service execution time.

        Returns:
            RoryResult with response_time computed automatically.
        """
        return RoryResult(
            label_vector=label_vector,
            response_time=time() - start_time,
            n_iterations=n_iterations,
            service_time=service_time,
        )

    @staticmethod
    def get_labelvector_from_indexes(
        shape: int, c_indexes: List[List[int]]
    ) -> List[int]:
        """Generate a label vector from cluster indexes.

        Delegates to Utils.get_labelvector_from_indexes.

        Args:
            shape: Total number of records.
            c_indexes: List of clusters, each containing record indices.

        Returns:
            Label vector with the cluster assignment for each record.
        """
        return Utils.get_labelvector_from_indexes(
            shape=shape, c_indexes=c_indexes
        )


class StandardClustering(ClusteringAlgorithm):
    """Base for plaintext clustering algorithms (KMeans, Nnc)."""


class ConventionalClustering(ClusteringAlgorithm):
    """Base for conventional secure clustering (Liu / FDHOPE).

    Iterative KMeans variants implement the three-phase pattern:
      1. execute_encrypted_phase  — cluster assignment + centroid update + shift
      2. execute_plaintext_phase  — update distance matrix with decrypted shift
      3. compute_centroid_shift   — encrypted difference between centroid sets

    Single-pass algorithms (Dbsnnc) override fit() directly and
    leave the phase methods as stubs.
    """

    @abstractmethod
    def execute_encrypted_phase(
        self,
        status: int,
        k: int,
        encrypted_matrix: npt.NDArray,
        udm: npt.NDArray,
        num_attributes: int,
        centroids: Optional[npt.NDArray] = None,
        **kwargs,
    ) -> Result:
        """Execute one encrypted clustering iteration.

        Args:
            status: ClusteringStatus (START or WORK_IN_PROGRESS).
            k: Number of clusters.
            encrypted_matrix: Encrypted dataset as NDArray.
            udm: Updatable distance matrix.
            num_attributes: Number of attributes per record.
            centroids: Previous centroid set. None on first call.
            **kwargs: Additional parameters (e.g. m).

        Returns:
            Result with Ok((shift_matrix, prev_centroids,
            new_centroids, label_vector)) or Err(exception).
        """
        ...

    @abstractmethod
    def execute_plaintext_phase(
        self,
        k: int,
        udm: npt.NDArray,
        num_attributes: int,
        shift_matrix: npt.NDArray,
        **kwargs,
    ) -> npt.NDArray:
        """Apply decrypted shift values to update the UDM.

        Args:
            k: Number of clusters.
            udm: Current UDM matrix.
            num_attributes: Number of attributes per record.
            shift_matrix: Decrypted shift values.
            **kwargs: Additional parameters.

        Returns:
            Updated UDM as npt.NDArray.
        """
        ...

    @abstractmethod
    def compute_centroid_shift(
        self,
        previous_centroids: npt.NDArray,
        current_centroids: npt.NDArray,
        k: int,
        **kwargs,
    ) -> npt.NDArray:
        """Compute encrypted shift between two centroid sets.

        Args:
            previous_centroids: Centroids from previous iteration.
            current_centroids: Centroids from current iteration.
            k: Number of clusters.
            **kwargs: Additional parameters (a, m).

        Returns:
            Encrypted shift matrix (NDArray).
        """
        ...


class PqcClustering(ClusteringAlgorithm):
    """Base for post-quantum secure clustering (CKKS / FDHOPE).

    Iterative KMeans variants implement the three-phase pattern:
      1. execute_encrypted_phase  — cluster assignment + centroid update + shift
      2. execute_plaintext_phase  — update distance matrix with decrypted shift
      3. compute_centroid_shift   — encrypted difference between centroid sets

    Single-pass algorithms (Dbsnnc) override fit() directly and
    leave the phase methods as stubs.
    """

    @abstractmethod
    def execute_encrypted_phase(
        self,
        status: int,
        k: int,
        encrypted_matrix: 'List[PyCtxt]',
        udm: npt.NDArray,
        num_attributes: int,
        centroids: 'Optional[List[PyCtxt]]' = None,
        **kwargs,
    ) -> Result:
        """Execute one encrypted CKKS-based clustering iteration.

        Args:
            status: ClusteringStatus (START or WORK_IN_PROGRESS).
            k: Number of clusters.
            encrypted_matrix: Encrypted dataset as List[PyCtxt].
            udm: Updatable distance matrix.
            num_attributes: Number of attributes per record.
            centroids: Previous centroid set. None on first call.
            **kwargs: Additional parameters.

        Returns:
            Result with Ok((shift_matrix, prev_centroids,
            new_centroids, label_vector)) or Err(exception).
        """
        ...

    @abstractmethod
    def execute_plaintext_phase(
        self,
        k: int,
        udm: npt.NDArray,
        num_attributes: int,
        shift_matrix: npt.NDArray,
        **kwargs,
    ) -> npt.NDArray:
        """Apply decrypted shift values to update the UDM.

        Args:
            k: Number of clusters.
            udm: Current UDM matrix.
            num_attributes: Number of attributes per record.
            shift_matrix: Decrypted shift values.
            **kwargs: Additional parameters.

        Returns:
            Updated UDM as npt.NDArray.
        """
        ...

    @abstractmethod
    def compute_centroid_shift(
        self,
        previous_centroids: 'List[PyCtxt]',
        current_centroids: 'List[PyCtxt]',
        k: int,
        **kwargs,
    ) -> 'List[PyCtxt]':
        """Compute encrypted shift between two CKKS centroid sets.

        Args:
            previous_centroids: Centroids from previous iteration.
            current_centroids: Centroids from current iteration.
            k: Number of clusters.
            **kwargs: Additional parameters.

        Returns:
            Encrypted shift matrix as List[PyCtxt].
        """
        ...


# --- Classification ---


class ClassificationAlgorithm(ABC):
    """Root for all classification algorithms in Rory.

    Subclasses are grouped into three families:
      - StandardClassification      — plaintext algorithms
      - ConventionalClassification  — secure algorithms (Liu-based)
      - PqcClassification           — post-quantum secure algorithms (CKKS-based)
    """

    @staticmethod
    def fit(**kwargs: Any) -> Any:
        """Execute the classification algorithm.

        Default is no-op (KNN does not train). Subclasses that
        train (LogisticRegression, PPLR) override this.

        Returns:
            Task-specific result (weights, model, None, etc.).
        """
        pass

    @abstractmethod
    def predict(self, **kwargs: Any) -> Any:
        """Apply the trained model to new data.

        Returns:
            Predictions (label vector, encrypted predictions, etc.).
        """
        ...


class StandardClassification(ClassificationAlgorithm):
    """Base for plaintext classification algorithms
    (KNearestNeighbors, LogisticRegression).
    """


class ConventionalClassification(ClassificationAlgorithm):
    """Base for conventional secure classification algorithms
    (Liu-based SecureKNearestNeighbors).
    """


class PqcClassification(ClassificationAlgorithm):
    """Base for post-quantum secure classification algorithms
    (CKKS-based SecureKNearestNeighbors, PPLR).
    """
