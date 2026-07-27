from rory.core.utils.utils import Utils
from rory.core.algorithms import StandardClassification
from typing import List
import numpy.typing as npt
import numpy as np


class KNearestNeighbors(StandardClassification):
    """A K-Nearest Neighbors classifier operating on plaintext data."""

    @staticmethod
    def predict(dataset: npt.NDArray, model: npt.NDArray,
                model_labels: npt.NDArray, distance: str) -> npt.NDArray:
        """Classify dataset records using nearest-neighbour lookup.

        Computes distances to model points, finds the minimum-index
        model record for each dataset row, and returns the corresponding
        labels.

        Args:
            dataset: Plaintext records to classify.
            model: Reference model records.
            model_labels: Labels associated with each model record.
            distance: Distance metric name ("MANHATHAN" or "EUCLIDEAN").

        Returns:
            npt.NDArray: Predicted labels for each dataset record.
        """
        min_indexes = KNearestNeighbors.calculate_distances_and_indexes(
            dataset  = dataset,
            model    = model,
            distance = distance
        )
        return Utils.get_label_vector(model_labels=model_labels,
                                      min_indexes=min_indexes)

    @staticmethod
    def calculate_distances_and_indexes(model: npt.NDArray,
                                        dataset: npt.NDArray,
                                        distance: str = "MANHATHAN") -> List[int]:
        """Compute minimum-model indices for each dataset record.

        For each dataset row, calculates the distance to every model
        record and returns the index of the closest model point.

        Args:
            model: Reference model records.
            dataset: Plaintext records to classify.
            distance: Distance metric name. Defaults to "MANHATHAN".

        Returns:
            List[int]: Index of the closest model record per dataset row.
        """
        dataset_shape = Utils.getShapeOfMatrix(dataset)
        model_shape = Utils.getShapeOfMatrix(model)
        mins = []
        for i in range(dataset_shape[0]):
            distances = []
            for j in range(model_shape[0]):
                dist = KNearestNeighbors.get_distance(
                    x1       = dataset[i],
                    x2       = model[j],
                    distance = distance
                )
                distances.append((j, dist))
            min_distance = min(distances, key=lambda x: x[1])
            mins.append(min_distance[0])
        return mins

    @staticmethod
    def get_distance(x1: npt.NDArray, x2: npt.NDArray, distance: str):
        """Dispatch to the correct distance metric.

        Args:
            x1: First data point.
            x2: Second data point.
            distance: Metric name ("MANHATHAN" or "EUCLIDEAN").

        Returns:
            float: Distance between x1 and x2.

        Raises:
            ValueError: If the distance metric is unknown.
        """
        if distance == "MANHATHAN":
            return KNearestNeighbors.manhathan_distance(x1, x2)
        if distance == "EUCLIDEAN":
            return KNearestNeighbors.euclidean(x1, x2)
        raise ValueError(f"Unknown distance: {distance}")

    @staticmethod
    def manhathan_distance(x1: npt.NDArray, x2: npt.NDArray):
        """Compute the Manhattan (L1) distance between two points.

        Args:
            x1: First data point.
            x2: Second data point.

        Returns:
            float: Sum of absolute differences.
        """
        return np.sum(np.abs(x1 - x2))

    @staticmethod
    def euclidean(x1: npt.NDArray, x2: npt.NDArray):
        """Compute the squared Euclidean (L2^2) distance between two points.

        Args:
            x1: First data point.
            x2: Second data point.

        Returns:
            float: Sum of squared differences.
        """
        return np.sum(np.power((x1 - x2), 2))

    @staticmethod
    def split_labelvector_from_data(dataset: npt.NDArray):
        """Extract the label vector from a dataset with inline labels.

        Args:
            dataset: Plaintext dataset with labels in the last column.

        Returns:
            npt.NDArray: Array of labels extracted from the dataset.
        """
        return Utils.split_labelvector_from_data(dataset=dataset)
