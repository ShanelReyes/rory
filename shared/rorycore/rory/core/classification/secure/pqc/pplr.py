import gc
from typing import List, Tuple
from rory.core.algorithms import PqcClassification
from rory.core.security.cryptosystem.pqc.ckks import Ckks
from Pyfhel import PyCtxt


class PPLR(PqcClassification):
    """Privacy Preserving Logistic Regression using CKKS encryption."""

    @staticmethod
    def sigmoid_poly(scheme: Ckks, z_input: PyCtxt, scale: int,
                     n_features: int) -> PyCtxt:
        """Polynomial approximation of the sigmoid for CKKS ciphertexts.

        Uses a third-degree Taylor expansion: sigmoid(x) = 0.5 + 0.25*x - 0.02083*x^3,
        with scale normalization for each homomorphic operation.

        Args:
            scheme: CKKS cryptosystem instance.
            z_input: Encrypted linear combination value.
            scale: CKKS scale factor.
            n_features: Number of features for slot alignment.

        Returns:
            PyCtxt: Encrypted polynomial sigmoid output.
        """
        normalized_z      = scheme.normalize_scale(z_input, scale)
        z_squared         = scheme.multiply(normalized_z, normalized_z)
        z_cubed           = scheme.multiply(z_squared, normalized_z)
        term_degree_1     = scheme.multiply_scalar(0.25, normalized_z)
        term_degree_3     = scheme.multiply_scalar(-0.02083, z_cubed)
        temp_sum          = scheme.add(term_degree_1, term_degree_3)
        activation_output = scheme.add_plain_scalar(temp_sum, 0.5)
        return scheme.normalize_scale(activation_output, scale)

    @staticmethod
    def forward_and_error(scheme: Ckks, encrypted_x: PyCtxt,
                          encrypted_y: PyCtxt,
                          encrypted_weights: PyCtxt,
                          encrypted_bias: PyCtxt,
                          n_features: int, scale: int) -> PyCtxt:
        """Compute encrypted forward pass and prediction error.

        Calculates dot(x, w) + b, applies polynomial sigmoid, and
        returns the difference from the true label.

        Args:
            scheme: CKKS cryptosystem instance.
            encrypted_x: Encrypted feature vector.
            encrypted_y: Encrypted label.
            encrypted_weights: Encrypted weight vector.
            encrypted_bias: Encrypted bias term.
            n_features: Number of features.
            scale: CKKS scale factor.

        Returns:
            PyCtxt: Encrypted prediction error (prediction - y).
        """
        linear_output = scheme.dot_product(encrypted_x, encrypted_weights)
        linear_output = scheme.add(linear_output, encrypted_bias)
        linear_output = scheme.normalize_scale(linear_output, scale)
        prediction = PPLR.sigmoid_poly(
            scheme     = scheme,
            z_input    = linear_output,
            scale      = scale,
            n_features = n_features
        )
        return scheme.subtract(prediction, encrypted_y)

    @staticmethod
    def encrypted_gradients(scheme: Ckks,
                            encrypted_X_batch: List[PyCtxt],
                            encrypted_y_batch: List[PyCtxt],
                            encrypted_weights: PyCtxt,
                            encrypted_bias: PyCtxt,
                            n_features: int,
                            scale: int) -> Tuple[PyCtxt, PyCtxt]:
        """Compute encrypted weight and bias gradients over a batch.

        Accumulates gradients for each sample in the batch using
        homomorphic operations.

        Args:
            scheme: CKKS cryptosystem instance.
            encrypted_X_batch: List of encrypted feature vectors.
            encrypted_y_batch: List of encrypted labels.
            encrypted_weights: Encrypted weight vector.
            encrypted_bias: Encrypted bias term.
            n_features: Number of features.
            scale: CKKS scale factor.

        Returns:
            Tuple[PyCtxt, PyCtxt]: Accumulated weight and bias gradients.
        """
        weight_gradient_accumulator = None
        bias_gradient_accumulator   = None
        for idx, (curr_x, curr_y) in enumerate(zip(encrypted_X_batch, encrypted_y_batch)):
            sample_error = PPLR.forward_and_error(
                scheme            = scheme,
                encrypted_x       = curr_x,
                encrypted_y       = curr_y,
                encrypted_weights = encrypted_weights,
                encrypted_bias    = encrypted_bias,
                n_features        = n_features,
                scale             = scale
            )
            gradient_term = scheme.multiply(sample_error, curr_x)
            if idx == 0:
                weight_gradient_accumulator = gradient_term
                bias_gradient_accumulator = sample_error
            else:
                weight_gradient_accumulator = scheme.add(
                    weight_gradient_accumulator, gradient_term
                )
                bias_gradient_accumulator = scheme.add(
                    bias_gradient_accumulator, sample_error
                )
        return weight_gradient_accumulator, bias_gradient_accumulator

    @staticmethod
    def fit(scheme: Ckks, learning_rate: float,
            encrypted_weights: PyCtxt, encrypted_bias: PyCtxt,
            encrypted_matrix: List[PyCtxt],
            encrypted_labelvector: List[PyCtxt],
            n_features: int, scale: int,
            n_samples: int) -> Tuple[PyCtxt, PyCtxt]:
        """Perform one training step of encrypted logistic regression.

        Computes encrypted gradients over the full dataset and updates
        weights and bias using gradient descent. Applies scale
        normalization after each update.

        Args:
            scheme: CKKS cryptosystem instance.
            learning_rate: Learning rate for gradient descent.
            encrypted_weights: Current encrypted weight vector.
            encrypted_bias: Current encrypted bias term.
            encrypted_matrix: List of encrypted training feature vectors.
            encrypted_labelvector: List of encrypted labels.
            n_features: Number of features.
            scale: CKKS scale factor.
            n_samples: Number of training samples.

        Returns:
            Tuple[PyCtxt, PyCtxt]: Updated encrypted weights and bias.
        """
        combined_lr_m = learning_rate / float(n_samples)
        sum_dw, sum_db = PPLR.encrypted_gradients(
            scheme            = scheme,
            encrypted_X_batch = encrypted_matrix,
            encrypted_y_batch = encrypted_labelvector,
            encrypted_weights = encrypted_weights,
            encrypted_bias    = encrypted_bias,
            n_features        = n_features,
            scale             = scale
        )
        step_weights      = scheme.multiply_scalar(combined_lr_m, sum_dw)
        step_bias         = scheme.multiply_scalar(combined_lr_m, sum_db)
        encrypted_weights = scheme.subtract(encrypted_weights, step_weights)
        encrypted_bias    = scheme.subtract(encrypted_bias, step_bias)
        encrypted_weights = scheme.normalize_scale(encrypted_weights, scale)
        encrypted_bias    = scheme.normalize_scale(encrypted_bias, scale)
        gc.collect()
        return encrypted_weights, encrypted_bias

    @staticmethod
    def predict(scheme: Ckks, encrypted_matrix: List[PyCtxt],
                encrypted_weights: PyCtxt, encrypted_bias: PyCtxt,
                scale: int, n_features: int) -> List[PyCtxt]:
        """Generate encrypted predictions using trained PPLR model.

        Computes dot(x, w) + b for each sample, applies polynomial
        sigmoid, and returns encrypted prediction values.

        Args:
            scheme: CKKS cryptosystem instance.
            encrypted_matrix: List of encrypted test feature vectors.
            encrypted_weights: Trained encrypted weight vector.
            encrypted_bias: Trained encrypted bias term.
            scale: CKKS scale factor.
            n_features: Number of features.

        Returns:
            List[PyCtxt]: Encrypted prediction values for each sample.
        """
        encrypted_predictions: List[PyCtxt] = []
        enc_w_norm = scheme.normalize_scale(encrypted_weights, scale)
        enc_b_norm = scheme.normalize_scale(encrypted_bias, scale)
        for current_x in encrypted_matrix:
            linear_z = scheme.dot_product(current_x, enc_w_norm)
            linear_z = scheme.add(linear_z, enc_b_norm)
            linear_z = scheme.normalize_scale(linear_z, scale)
            prediction_h = PPLR.sigmoid_poly(
                scheme            = scheme,
                z_input           = linear_z,
                scale             = scale,
                n_features        = n_features
            )
            encrypted_predictions.append(prediction_h)
        return encrypted_predictions
