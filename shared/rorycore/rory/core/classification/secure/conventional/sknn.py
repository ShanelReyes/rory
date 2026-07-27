from __future__ import annotations
import numpy as np
from rory.core.utils.utils import Utils
from rory.core.security.cryptosystem.liu import Liu
from rory.core.algorithms import ConventionalClassification
from typing import List, Optional, Tuple
import numpy.typing as npt


class SecureKNearestNeighbors(ConventionalClassification):
    """Secure KNN on encrypted data using Liu's symmetric homomorphic encryption."""

    @staticmethod
    def predict(dataset: npt.NDArray, model: npt.NDArray,
                model_labels: npt.NDArray, distance: str = "MANHATHAN",
                scheme: Optional['Liu'] = None,
                sk: Optional['List[Tuple[float, float, float]]'] = None) -> npt.NDArray:
        """Classify encrypted dataset records using secure nearest-neighbour.

        Computes encrypted distances and returns model labels for the
        closest model point. Optionally decrypts distances using the
        Liu scheme if a scheme and secret key are provided.

        Args:
            dataset: Encrypted records to classify.
            model: Encrypted model records.
            model_labels: Labels associated with each model record.
            distance: Distance metric name. Defaults to "MANHATHAN".
            scheme: Liu cryptosystem for decryption. Defaults to None.
            sk: Secret key. Defaults to None.

        Returns:
            npt.NDArray: Predicted labels for each dataset record.
        """
        distances = SecureKNearestNeighbors.calculate_distances(
            model=model, dataset=dataset, distance=distance
        )
        if scheme is not None and sk is not None:
            decrypted = scheme.decryptMatrix(
                ciphertext_matrix = distances,
                secret_key        = sk
            ).data
            min_indexes = np.argmin(decrypted, axis=1)
        else:
            min_indexes = [
                min(enumerate(row), key=lambda x: np.sum(x[1]))[0]
                for row in distances
            ]
        return Utils.get_label_vector(
            model_labels=model_labels, min_indexes=min_indexes
        )

    @staticmethod
    def calculate_distances(model: npt.NDArray, dataset: npt.NDArray,
                            distance: str = "MANHATHAN") -> npt.NDArray:
        """Compute encrypted pairwise distances between dataset and model.

        Iterates over all dataset-model record pairs and accumulates
        the encrypted distance across attributes using Liu operations.

        Args:
            model: Encrypted model records.
            dataset: Encrypted records to classify.
            distance: Distance metric name. Defaults to "MANHATHAN".

        Returns:
            npt.NDArray: Encrypted distance matrix of shape
                (num_dataset, num_model, m).
        """
        dataset_shape = Utils.getShapeOfMatrix(dataset)
        model_shape = Utils.getShapeOfMatrix(model)
        attributes = dataset_shape[1]
        m_value = dataset_shape[2]
        all_distances = []
        for i in range(dataset_shape[0]):
            distances_record = []
            for j in range(model_shape[0]):
                acum = np.zeros(m_value).tolist()
                for k in range(attributes):
                    acum = SecureKNearestNeighbors.get_distance(
                        x1=dataset[i][k], x2=model[j][k],
                        distance=distance, acum=acum
                    )
                distances_record.append(acum)
            all_distances.append(distances_record)
        return np.array(all_distances)

    @staticmethod
    def get_distance(x1: npt.NDArray, x2: npt.NDArray, acum,
                     distance: str = "MANHATHAN"):
        """Dispatch to the correct encrypted distance metric.

        Args:
            x1: First encrypted data point.
            x2: Second encrypted data point.
            acum: Liu-encrypted accumulator.
            distance: Metric name ("MANHATHAN" or "EUCLIDEAN").

        Returns:
            Liu-encrypted accumulated distance value.

        Raises:
            ValueError: If the distance metric is unknown.
        """
        if distance == "MANHATHAN":
            return SecureKNearestNeighbors.manhathan_distance(x1, x2, acum)
        if distance == "EUCLIDEAN":
            return SecureKNearestNeighbors.euclidean(x1, x2, acum)
        raise ValueError(f"Unknown distance: {distance}")

    @staticmethod
    def manhathan_distance(x1: npt.NDArray, x2: npt.NDArray, acum):
        """Encrypted Manhattan distance accumulation using Liu's scheme.

        Computes abs(x1 - x2) and adds to the accumulator homomorphically.

        Args:
            x1: First encrypted data point.
            x2: Second encrypted data point.
            acum: Liu-encrypted accumulator.

        Returns:
            Updated Liu-encrypted accumulator.
        """
        manh = np.abs(Liu.subtract(ciphertext_1=x1, ciphertext_2=x2))
        acum = Liu.add(ciphertext_1=acum, ciphertext_2=manh)
        return acum

    @staticmethod
    def euclidean(x1: npt.NDArray, x2: npt.NDArray, acum):
        """Encrypted squared Euclidean distance accumulation.

        Computes (x1 - x2)^2 and adds to the accumulator homomorphically.

        Args:
            x1: First encrypted data point.
            x2: Second encrypted data point.
            acum: Liu-encrypted accumulator.

        Returns:
            Updated Liu-encrypted accumulator.
        """
        manh = np.power(Liu.subtract(ciphertext_1=x1, ciphertext_2=x2), 2)
        acum = Liu.add(ciphertext_1=acum, ciphertext_2=manh)
        return acum

    @staticmethod
    def split_labelvector_from_data(dataset: npt.NDArray):
        """Extract the label vector from a dataset with inline labels.

        Args:
            dataset: Dataset with labels in the last column.

        Returns:
            npt.NDArray: Labels extracted from the dataset.
        """
        return Utils.split_labelvector_from_data(dataset=dataset)
