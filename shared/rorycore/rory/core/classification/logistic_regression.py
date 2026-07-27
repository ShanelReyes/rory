import numpy as np
from typing import Tuple, List
from rory.core.algorithms import StandardClassification


class LogisticRegression(StandardClassification):
    """A plaintext logistic regression model for binary classification."""

    @staticmethod
    def sigmoid_poly(z_input: np.ndarray) -> np.ndarray:
        """Polynomial approximation of the sigmoid function.

        Uses a third-degree Taylor expansion of the sigmoid:
        sigmoid(x) = 0.5 + 0.25*x - 0.02083*x^3.

        Args:
            z_input: Linear combination values.

        Returns:
            np.ndarray: Approximated sigmoid output for each element.
        """
        sigmoid = 0.5 + 0.25 * z_input - 0.02083 * (z_input ** 3)
        return sigmoid

    @staticmethod
    def fit(plaintext_matrix: np.ndarray, label_vector: np.ndarray,
            epochs: int = 10, learning_rate: float = 0.1,
            bias: float = 0.0,
            weights: np.ndarray = None) -> Tuple[np.ndarray, float]:
        """Train a logistic regression model using gradient descent.

        Performs binary classification training by minimizing the error
        between polynomial-sigmoid predictions and true labels.

        Args:
            plaintext_matrix: Training data matrix.
            label_vector: Binary labels (0 or 1).
            epochs: Number of training iterations. Defaults to 10.
            learning_rate: Step size for gradient descent. Defaults to 0.1.
            bias: Initial bias term. Defaults to 0.0.
            weights: Initial weight vector. Defaults to zeros if None.

        Returns:
            Tuple[np.ndarray, float]: Trained weights and bias.
        """
        n_samples = plaintext_matrix.shape[0]
        if weights is None:
            weights = np.zeros(plaintext_matrix.shape[1])
        for _ in range(epochs):
            linear_output    = np.dot(plaintext_matrix, weights) + bias
            predictions      = LogisticRegression.sigmoid_poly(linear_output)
            error            = predictions - label_vector
            weight_gradients = (1 / n_samples) * np.dot(
                plaintext_matrix.T, error
            )
            bias_gradient = (1 / n_samples) * np.sum(error)
            weights -= learning_rate * weight_gradients
            bias    -= learning_rate * bias_gradient
        return weights, bias

    @staticmethod
    def predict(plaintext_matrix: np.ndarray, weights: np.ndarray,
                bias: float) -> List[int]:
        """Generate binary predictions using trained logistic regression.

        Computes linear combination, applies polynomial sigmoid, and
        thresholds at 0.5 for binary classification.

        Args:
            plaintext_matrix: Data matrix to predict.
            weights: Trained weight vector.
            bias: Trained bias term.

        Returns:
            List[int]: Binary predictions (0 or 1) for each record.
        """
        linear_output = np.dot(plaintext_matrix, weights) + bias
        predictions = LogisticRegression.sigmoid_poly(linear_output)
        return [1 if p >= 0.5 else 0 for p in predictions]
