from __future__ import annotations
from typing import List, Tuple, TYPE_CHECKING
from rory.core.utils.utils import Utils
from rory.core.algorithms import ConventionalClustering
import numpy.typing as npt
from option import Result
if TYPE_CHECKING:
	from rory.core.security.cryptosystem.liu import Liu


class DBSKMeans(ConventionalClustering):
    """Distributed Double-Blind Secure KMeans using Liu's cryptosystem.

    Iterative algorithm divided into three phases per iteration:
      1. execute_encrypted_phase  — cluster assignment + centroid update + shift
      2. Decrypt shift matrix (client-side)
      3. execute_plaintext_phase  — update UDM with decrypted shift values
    """

    def execute_encrypted_phase(
        self,
        status: int,
        k: int,
        encrypted_matrix: npt.NDArray,
        udm: npt.NDArray,
        num_attributes: int,
        centroids: npt.NDArray = None,
        m: int = 3,
        **kwargs,
    ) -> Result:
        """Perform cluster assignment, centroid update, and shift over encrypted data.

        Delegates to the Liu-based encrypted-phase utility for the
        double-blind secure KMeans protocol.

        Args:
            status: Clustering status indicating first or subsequent iteration.
            k: Number of clusters.
            encrypted_matrix: Encrypted data matrix.
            udm: Updatable Distance Matrix.
            num_attributes: Number of attributes per record.
            centroids: Current centroids. Defaults to None.
            m: Liu scheme parameter for key size. Defaults to 3.

        Returns:
            Result containing shift matrix, previous centroids,
            current centroids, and label vector.
        """
        return Utils.execute_encrypted_phase_liu(
            status=status,
            k=k,
            encrypted_matrix=encrypted_matrix,
            udm=udm,
            num_attributes=num_attributes,
            centroids=centroids,
            m=m,
        )

    def execute_plaintext_phase(
        self,
        k: int,
        udm: npt.NDArray,
        num_attributes: int,
        shift_matrix: npt.NDArray,
        **kwargs,
    ) -> npt.NDArray:
        """Update the UDM with decrypted shift values.

        Applies the decrypted shift matrix to the Updatable Distance
        Matrix in plaintext.

        Args:
            k: Number of clusters.
            udm: Current Updatable Distance Matrix.
            num_attributes: Number of attributes per record.
            shift_matrix: Decrypted shift values from the encrypted phase.

        Returns:
            npt.NDArray: Updated UDM after applying the shift matrix.
        """
        return Utils.execute_plaintext_phase(
            k=k,
            udm=udm,
            num_attributes=num_attributes,
            shift_matrix=shift_matrix,
        )

    def compute_centroid_shift(
        self,
        previous_centroids: npt.NDArray,
        current_centroids: npt.NDArray,
        k: int,
        a: int = 3,
        m: int = 3,
        **kwargs,
    ) -> npt.NDArray:
        """Compute the encrypted shift between two centroid sets.

        Delegates to the Liu-based utility for the double-blind protocol.

        Args:
            previous_centroids: Centroids from the previous iteration.
            current_centroids: Centroids from the current iteration.
            k: Number of clusters.
            a: Scheme parameter. Defaults to 3.
            m: Liu scheme parameter. Defaults to 3.

        Returns:
            npt.NDArray: Encrypted centroid shift matrix.
        """
        return Utils.compute_centroid_shift_liu(
            previous_centroids=previous_centroids,
            current_centroids=current_centroids,
        )

    def fit(
        self,
        status: int,
        k: int,
        m: int,
        encrypted_matrix: npt.NDArray,
        UDM: npt.NDArray,
        num_attributes: int,
        Cent_j: npt.NDArray,
        iterations: int,
        n_iterations: int,
        scheme: 'Liu',
        sk: 'List[Tuple[float, float, float]]',
        min_error: float = 0.000001,
    ) -> List[int]:
        """Run the iterative double-blind secure KMeans clustering.

        Delegates to the conventional iterative fitting utility for the
        double-blind protocol, alternating between encrypted and plaintext
        phases until convergence.

        Args:
            status: Clustering status.
            k: Number of clusters.
            m: Liu scheme parameter.
            encrypted_matrix: Encrypted data matrix.
            UDM: Updatable Distance Matrix.
            num_attributes: Number of attributes per record.
            Cent_j: Initial centroids.
            iterations: Maximum number of iterations.
            n_iterations: Counter for current iteration.
            scheme: Liu cryptosystem instance.
            sk: Secret key as list of (k, s, t) tuples.
            min_error: Convergence threshold. Defaults to 0.000001.

        Returns:
            List[int]: Final label vector assigning each record to a cluster.
        """
        return Utils.fit_conventional_iterative(
            instance=self,
            status=status,
            k=k,
            m=m,
            encrypted_matrix=encrypted_matrix,
            UDM=UDM,
            num_attributes=num_attributes,
            Cent_j=Cent_j,
            iterations=iterations,
            n_iterations=n_iterations,
            scheme=scheme,
            sk=sk,
            min_error=min_error,
        )
