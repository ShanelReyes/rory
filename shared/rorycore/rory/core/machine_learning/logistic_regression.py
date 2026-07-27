import numpy as np
from typing import Tuple


class LogisticRegression:
	"""A plaintext logistic regression model for binary classification.

	Implements polynomial sigmoid approximation, gradient-descent
	training, and threshold-based prediction on unencrypted data.
	"""

	@staticmethod
	def sigmoid_poly(z_input: np.ndarray) -> np.ndarray:
		"""Computes a 3rd-degree polynomial approximation of the sigmoid function.

		Uses the Taylor expansion: f(z) = 0.5 + 0.25*z - 0.02083*z^3.

		Args:
			z_input (np.ndarray): The linear combination z = X*w + b.

		Returns:
			np.ndarray: Approximated sigmoid activation values.
		"""
		sigmoid = 0.5 + 0.25 * z_input - 0.02083 * (z_input ** 3)
		return sigmoid

	@staticmethod
	def train(plaintext_matrix: np.ndarray, label_vector: np.ndarray, epochs: int = 10, learning_rate: float = 0.1, bias: float = 0.0, weights: np.ndarray = None) -> Tuple[np.ndarray, float]:
		"""Trains the logistic regression model using gradient descent.

		Args:
			plaintext_matrix (np.ndarray): Training data of shape (n_samples,
				n_features).
			label_vector (np.ndarray): True binary labels (0 or 1).
			epochs (int, optional): Number of training epochs. Defaults to
				10.
			learning_rate (float, optional): Gradient descent step size.
				Defaults to 0.1.
			bias (float, optional): Initial bias term. Defaults to 0.0.
			weights (np.ndarray, optional): Initial weight vector. If None,
				initialized to zeros. Defaults to None.

		Returns:
			Tuple[np.ndarray, float]: The trained weights vector and bias.
		"""
		n_samples = plaintext_matrix.shape[0]

		if weights is None:
			weights = np.zeros(plaintext_matrix.shape[1])

		for _ in range(epochs):
			linear_output = np.dot(plaintext_matrix, weights) + bias
			predictions   = LogisticRegression.sigmoid_poly(linear_output)

			error            = predictions - label_vector
			weight_gradients = (1 / n_samples) * np.dot(plaintext_matrix.T, error)
			bias_gradient    = (1 / n_samples) * np.sum(error)

			weights -= learning_rate * weight_gradients
			bias    -= learning_rate * bias_gradient

		return weights, bias

	@staticmethod
	def predict(plaintext_matrix: np.ndarray, weights: np.ndarray, bias: float) -> np.array:
		"""Generates binary predictions using the trained model.

		Computes the linear output, applies polynomial sigmoid, and
		thresholds at 0.5 to produce class labels.

		Args:
			plaintext_matrix (np.ndarray): Test data of shape (n_samples,
				n_features).
			weights (np.ndarray): Trained weight vector.
			bias (float): Trained bias term.

		Returns:
			List[int]: Predicted binary labels (0 or 1).
		"""
		linear_output = np.dot(plaintext_matrix, weights) + bias
		predictions = LogisticRegression.sigmoid_poly(linear_output)
		return [1 if p >= 0.5 else 0 for p in predictions]