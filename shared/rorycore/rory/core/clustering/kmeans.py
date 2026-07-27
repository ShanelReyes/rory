from time import time
from sklearn.cluster import KMeans as _SklearnKMeans
from rory.core.utils.utils import Utils
from rory.core.interfaces.rory_result import RoryResult
from rory.core.algorithms import StandardClustering
import numpy.typing as npt


class KMeans(StandardClustering):
    """Plaintext KMeans clustering via scikit-learn.

    Wraps sklearn.cluster.KMeans with Rory's result interface.
    """

    def fit(self, plaintext_matrix: npt.NDArray, k: int = 2) -> RoryResult:
        """Execute KMeans clustering on plaintext data.

        Args:
            plaintext_matrix: Input data matrix.
            k: Number of clusters. Defaults to 2.

        Returns:
            RoryResult with label_vector, n_iterations, response_time,
            and service_time.
        """
        startTime          = time()
        centroids          = Utils.generate_centroids(k=k, plain_matrix=plaintext_matrix)
        start_service_time = time()
        model              = _SklearnKMeans(n_clusters=k, init=centroids)
        model.fit(plaintext_matrix)
        end_service_time = time()
        service_time     = end_service_time - start_service_time
        return self._build_result(
            model.labels_,
            startTime,
            n_iterations = model.n_iter_,
            service_time = service_time,
        )
