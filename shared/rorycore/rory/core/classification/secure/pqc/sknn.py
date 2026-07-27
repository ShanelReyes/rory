from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple, TYPE_CHECKING
import numpy.typing as npt
from Pyfhel import PyCtxt
from rory.core.utils.utils import Utils
from rory.core.algorithms import PqcClassification
if TYPE_CHECKING:
	from rory.core.security.cryptosystem.pqc.ckks import Ckks


class SecureKNearestNeighbors(PqcClassification):
    """Secure KNN on encrypted data using PQC (CKKS) homomorphic encryption."""

    @staticmethod
    def predict(dataset: List[PyCtxt], model: List[PyCtxt],
                model_labels: npt.NDArray, distance: str = "EUCLIDEAN",
                model_shape: Tuple[int, int] = None,
                dataset_shape: Tuple[int, int] = None,
                scheme: Optional['Ckks'] = None) -> npt.NDArray:
        """Classify PQC-encrypted dataset records using secure KNN.

        Computes encrypted distances using CKKS operations and returns
        model labels for the closest model point. Optionally decrypts
        distances if a CKKS scheme is provided.

        Args:
            dataset: CKKS-encrypted records to classify.
            model: CKKS-encrypted model records.
            model_labels: Labels associated with each model record.
            distance: Distance metric name. Defaults to "EUCLIDEAN".
            model_shape: Shape (rows, cols) of the model. Defaults to None.
            dataset_shape: Shape (rows, cols) of the dataset. Defaults to None.
            scheme: CKKS cryptosystem for decryption. Defaults to None.

        Returns:
            npt.NDArray: Predicted labels for each dataset record.
        """
        distances = SecureKNearestNeighbors.calculate_distances(
            model         = model,
            dataset       = dataset,
            model_shape   = model_shape,
            dataset_shape = dataset_shape,
            distance      = distance,
        )
        if scheme is not None:
            decrypted = []
            for row in distances:
                row_vals = []
                for ct in row:
                    vals = scheme.decryptVector(ct)
                    row_vals.append(sum(vals))
                decrypted.append(row_vals)
            min_indexes = np.argmin(decrypted, axis=1)
        else:
            min_indexes = [
                min(enumerate(row), key=lambda x: x[1])[0]
                for row in distances
            ]
        return Utils.get_label_vector(
            model_labels=model_labels, min_indexes=min_indexes
        )

    @staticmethod
    def calculate_distances(model: List[PyCtxt], dataset: List[PyCtxt],
                            model_shape: Tuple[int, int],
                            dataset_shape: Tuple[int, int],
                            distance: str = "EUCLIDEAN") -> List[PyCtxt]:
        """Compute PQC-encrypted pairwise distances between dataset and model.

        Iterates over all dataset-model record pairs using CKKS-encrypted
        distance computations.

        Args:
            model: CKKS-encrypted model records.
            dataset: CKKS-encrypted records to classify.
            model_shape: Shape (rows, cols) of the model.
            dataset_shape: Shape (rows, cols) of the dataset.
            distance: Distance metric name. Defaults to "EUCLIDEAN".

        Returns:
            List[PyCtxt]: Encrypted distance matrix.
        """
        all_distances = []
        for i in range(dataset_shape[0]):
            distances_record = []
            for j in range(model_shape[0]):
                d = SecureKNearestNeighbors.get_distance(
                    x1       = dataset[i],
                    x2       = model[j],
                    distance = distance
                )
                distances_record.append(d)
            all_distances.append(distances_record)
        return np.array(all_distances)

    @staticmethod
    def get_distance(x1: PyCtxt, x2: PyCtxt, distance: str = "EUCLIDEAN"):
        """Dispatch to the correct CKKS-encrypted distance metric.

        Args:
            x1: First CKKS-encrypted data point.
            x2: Second CKKS-encrypted data point.
            distance: Metric name ("EUCLIDEAN" or "MANHATHAN").

        Returns:
            PyCtxt: Encrypted distance ciphertext.

        Raises:
            ValueError: If the distance metric is unknown.
        """
        if distance == "EUCLIDEAN":
            return SecureKNearestNeighbors.euclidean(x1, x2)
        if distance == "MANHATHAN":
            return SecureKNearestNeighbors.manhathan_distance(x1, x2)
        raise ValueError(f"Unknown distance: {distance}")

    @staticmethod
    def euclidean(x1: PyCtxt, x2: PyCtxt):
        """CKKS-encrypted squared Euclidean distance.

        Computes (x1 - x2)^2 using homomorphic operations.

        Args:
            x1: First CKKS-encrypted data point.
            x2: Second CKKS-encrypted data point.

        Returns:
            PyCtxt: Encrypted squared Euclidean distance.
        """
        diff = x1 - x2
        return diff * diff

    @staticmethod
    def manhathan_distance(x1: PyCtxt, x2: PyCtxt):
        """CKKS-encrypted Manhattan distance (signed difference).

        Args:
            x1: First CKKS-encrypted data point.
            x2: Second CKKS-encrypted data point.

        Returns:
            PyCtxt: Encrypted difference (x1 - x2).
        """
        return x1 - x2

    @staticmethod
    def split_labelvector_from_data(dataset: npt.NDArray):
        """Extract the label vector from a dataset with inline labels.

        Args:
            dataset: Dataset with labels in the last column.

        Returns:
            npt.NDArray: Labels extracted from the dataset.
        """
        return Utils.split_labelvector_from_data(dataset=dataset)
