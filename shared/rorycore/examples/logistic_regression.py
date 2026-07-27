"""
Demo: Logistic Regression (plaintext).
"""
import numpy as np
from rory.core.classification.logistic_regression import LogisticRegression

print("=" * 60)
print("DEMO Logistic Regression (plaintext)")
print("=" * 60)

rng = np.random.RandomState(42)
n_samples = 30
X_class0 = rng.randn(n_samples, 2) * 0.15 + np.array([-0.5, -0.5])
X_class1 = rng.randn(n_samples, 2) * 0.15 + np.array([0.5, 0.5])
X = np.vstack([X_class0, X_class1]).astype(np.float64)
y = np.array([0.0] * n_samples + [1.0] * n_samples)

print(f"Dataset: {X.shape[0]} points x {X.shape[1]} features")
print(f"Labels: {n_samples} zeros, {n_samples} ones")

weights, bias = LogisticRegression.fit(
    plaintext_matrix = X,
    label_vector     = y,
    epochs           = 30,
    learning_rate    = 0.1,
)

predictions = LogisticRegression.predict(
    plaintext_matrix = X,
    weights          = weights,
    bias             = bias,
)

accuracy = np.mean(np.array(predictions) == y.astype(int))
print(f"Predictions (first 10): {predictions[:10]}")
print(f"Accuracy: {accuracy:.2%}")

print("=" * 60)
print("Logistic Regression demo completed successfully.")
print("=" * 60)
