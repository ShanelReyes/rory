import math
from typing import List, Optional
import numpy as np
import numpy.typing as npt
from option import Result, Ok, Err
from Pyfhel import PyCtxt
from rory.core.security.cryptosystem.pqc.ckks import Ckks
from rory.core.utils.utils import Utils
from rory.core.algorithms import PqcClustering


class Skmeans(PqcClustering):
    """Distributed Secure KMeans PQC using CKKS homomorphic encryption.

    Iterative algorithm divided into three phases per iteration:
      1. execute_encrypted_phase  — cluster assignment + centroid update + shift
      2. Decrypt shift matrix (client-side, via Ckks scheme)
      3. execute_plaintext_phase  — update UDM with decrypted shift values
    """

    def __init__(self, scheme: Ckks, init_shiftmatrix: PyCtxt):
        """Initialize the PQC secure KMeans instance.

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
        """Execute the encrypted phase of CKKS-based secure KMeans.

        Performs cluster assignment via UDM, computes centroids with
        rescale-aware averaging, and calculates the encrypted centroid
        shift matrix.

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
        import copy
        from rory.core.utils.constants import Constants
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

            centroids_result = self.compute_centroids(clusters=C)
            if centroids_result.is_err:
                return Err(centroids_result.unwrap_err())
            cent_j_raw = centroids_result.unwrap()
            cent_j = [
                (cent_j_raw[i] if cent_j_raw[i] is not None else cent_i[i])
                for i in range(k)
            ]

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
        **kwargs,
    ) -> List[PyCtxt]:
        """Compute the encrypted shift between two centroid sets using CKKS.

        Args:
            previous_centroids: Encrypted centroids from the previous iteration.
            current_centroids: Encrypted centroids from the current iteration.
            k: Number of clusters.

        Returns:
            List[PyCtxt]: Encrypted centroid shift values.
        """
        return Utils.compute_centroid_shift_ckks(
            previous_centroids=previous_centroids,
            current_centroids=current_centroids,
            init_shiftmatrix=self.init_shiftmatrix,
        )

    def compute_centroids(
        self, clusters: List[List[PyCtxt]]
    ) -> Result[List[Optional[PyCtxt]], Exception]:
        """Compute centroids using rescale-aware homomorphic averaging.

        For each cluster, sums all ciphertexts, rescales to level 0,
        then multiplies by 1/cluster_size to obtain the encrypted
        arithmetic mean.

        Args:
            clusters: List of k clusters, each a list of CKKS ciphertexts.

        Returns:
            Result containing list of k encrypted centroid ciphertexts
            (None for empty clusters).
        """
        try:
            k = len(clusters)
            centroids: List[Optional[PyCtxt]] = [None] * k
            slots = self.scheme.n_features
            scale = self.scheme.scale
            scale_bits = int(math.log2(scale))
            for j, cluster in enumerate(clusters):
                cj_len = len(cluster)
                if cj_len == 0:
                    centroids[j] = None
                    continue
                total_ctxt: PyCtxt = cluster[0]
                for ctxt in cluster[1:]:
                    total_ctxt = self.scheme.add(total_ctxt, ctxt)
                while total_ctxt.mod_level > 0:
                    total_ctxt = self.scheme.he_object.rescale_to_next(total_ctxt)
                factor = 1.0 / cj_len
                vec = np.full(slots, factor, dtype=np.float64)
                ptxt_factor = self.scheme.he_object.encodeFrac(
                    vec, scale=scale, scale_bits=scale_bits
                )
                avg_ctxt: PyCtxt = self.scheme.he_object.multiply_plain(
                    total_ctxt, ptxt_factor, in_new_ctxt=True
                )
                avg_ctxt = self.scheme.he_object.rescale_to_next(avg_ctxt)
                centroids[j] = avg_ctxt
            return Ok(centroids)
        except Exception as e:
            return Err(e)

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
    ) -> List[int]:
        """Run the iterative PQC secure KMeans clustering.

        Delegates to the PQC iterative fitting utility, alternating
        between encrypted CKKS operations and plaintext UDM updates
        until convergence or max iterations.

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

        Returns:
            List[int]: Final label vector assigning each record to a cluster.
        """
        return Utils.fit_pqc_iterative(
            instance=self,
            status=status,
            k=k,
            encrypted_matrix=encrypted_matrix,
            UDM=UDM,
            num_attributes=num_attributes,
            Cent_j=Cent_j,
            iterations=iterations,
            n_iterations=n_iterations,
            scheme=scheme,
            min_error=0.0015,
        )
