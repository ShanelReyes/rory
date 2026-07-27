import copy
from typing import List
import numpy.typing as npt
from option import Result, Ok, Err
from Pyfhel import PyCtxt
from rory.core.security.cryptosystem.pqc.ckks import Ckks
from rory.core.utils.utils import Utils
from rory.core.utils.constants import Constants
from rory.core.algorithms import PqcClustering


class DBSKMeans(PqcClustering):
    """Distributed Double-Blind Secure KMeans PQC using CKKS encryption.
    """

    def __init__(self, scheme: Ckks, init_shiftmatrix: PyCtxt):
        """Initialize the PQC double-blind secure KMeans instance.

        Args:
            scheme: CKKS cryptosystem instance for homomorphic operations.
            init_shiftmatrix: Pre-allocated all-zero ciphertext used as
                the initial shift matrix during centroid updates.
        """
        self.scheme: Ckks = scheme
        self.init_shiftmatrix = init_shiftmatrix

    def execute_encrypted_phase(
        self,
        status: int,
        k: int,
        encrypted_matrix: List[PyCtxt],
        udm: npt.NDArray,
        num_attributes: int,
        centroids: List[PyCtxt] = None,
        **kwargs,
    ) -> Result:
        """Execute the encrypted phase of CKKS-based double-blind KMeans.

        Performs cluster assignment, centroid computation via the CKKS
        utility, and encrypted shift matrix calculation.

        Args:
            status: Clustering status (START or CONTINUE).
            k: Number of clusters.
            encrypted_matrix: List of CKKS-encrypted records.
            udm: Updatable Distance Matrix.
            num_attributes: Number of attributes per record.
            centroids: Current encrypted centroids. Defaults to None.

        Returns:
            Result containing (shift_matrix, previous_centroids,
            current_centroids, label_vector) on success.
        """
        try:
            if status == Constants.ClusteringStatus.START:
                C = [[encrypted_matrix[i]] for i in range(k)]
                start_record = k
                cent_i = [encrypted_matrix[i] for i in range(k)]
            else:
                C = Utils.empty_cluster(k=k)
                start_record = 0
                cent_i = copy.copy(centroids)

            populate_result = Utils._populate_clusters(
                record_id=start_record,
                UDM=udm,
                num_clusters=k,
                num_attributes=num_attributes,
                ciphertext_matrix=encrypted_matrix,
                append_fn=lambda cl, idx, rec: cl[idx].append(rec),
                clusters=C,
            )
            
            if populate_result.is_err:
                return Err(populate_result.unwrap_err())
            C, label_vector = populate_result.unwrap()

            centroids_result = Utils.calculateCentroidsCkks(
                scheme=self.scheme, clusters=C, k=k,
            )
            if centroids_result.is_err:
                return Err(centroids_result.unwrap_err())
            cent_j_raw = centroids_result.unwrap()
            cent_j = [
                (cent_j_raw[i] if cent_j_raw[i] is not None else cent_i[i])
                for i in range(k)
            ]

            for ct in cent_j:
                self.scheme._try_rescale_next(ct)

            S1 = Utils.compute_centroid_shift_ckks(
                previous_centroids=cent_i,
                current_centroids=cent_j,
                init_shiftmatrix=self.init_shiftmatrix,
            )
            
            if status == Constants.ClusteringStatus.START:
                label_vector = Utils.fillLabelVector(
                    label_vector=label_vector, k=k
                )

            return Ok((S1, cent_i, cent_j, label_vector))
        except Exception as e:
            return Err(e)

    def execute_plaintext_phase(
        self,
        k: int,
        udm: npt.NDArray,
        num_attributes: int,
        shift_matrix: npt.NDArray,
        **kwargs,
    ) -> npt.NDArray:
        """Update the UDM with decrypted shift values.

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
        previous_centroids: List[PyCtxt],
        current_centroids: List[PyCtxt],
        k: int,
        a: int = 3,
        **kwargs,
    ) -> List[PyCtxt]:
        """Compute the encrypted shift between two centroid sets using CKKS.

        Args:
            previous_centroids: Encrypted centroids from the previous iteration.
            current_centroids: Encrypted centroids from the current iteration.
            k: Number of clusters.
            a: Scheme parameter. Defaults to 3.

        Returns:
            List[PyCtxt]: Encrypted centroid shift values.
        """
        return Utils.compute_centroid_shift_ckks(
            previous_centroids=previous_centroids,
            current_centroids=current_centroids,
            init_shiftmatrix=self.init_shiftmatrix,
        )

    def fit(
        self,
        status: int,
        k: int,
        encrypted_matrix: List[PyCtxt],
        UDM: npt.NDArray,
        num_attributes: int,
        Cent_j: List[PyCtxt],
        iterations: int,
        n_iterations: int,
        scheme: Ckks,
        min_error: float = 0.000001,
    ) -> List[int]:
        """Run the iterative PQC double-blind secure KMeans clustering.

        Alternates between encrypted CKKS operations and plaintext UDM
        updates until convergence or max iterations.

        Args:
            status: Clustering status.
            k: Number of clusters.
            encrypted_matrix: CKKS-encrypted data matrix.
            UDM: Updatable Distance Matrix.
            num_attributes: Number of attributes per record.
            Cent_j: Initial encrypted centroids.
            iterations: Maximum number of iterations.
            n_iterations: Counter for current iteration.
            scheme: CKKS cryptosystem instance.
            min_error: Convergence threshold. Defaults to 0.000001.

        Returns:
            List[int]: Final label vector assigning each record to a cluster.
        """
        return Utils.fit_pqc_iterative(
            instance         = self,
            status           = status,
            k                = k,
            encrypted_matrix = encrypted_matrix,
            UDM              = UDM,
            num_attributes   = num_attributes,
            Cent_j           = Cent_j,
            iterations       = iterations,
            n_iterations     = n_iterations,
            scheme           = scheme,
            min_error        = min_error,
        )
