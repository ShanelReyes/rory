import gc
from typing import List, Tuple
from rory.core.security.cryptosystem.pqc.ckks import Ckks
from Pyfhel import PyCtxt


class PPLR:
    """Privacy Preserving Logistic Regression using Homomorphic Encryption.

    Provides static utility methods for training and evaluation on encrypted
    vectors via a Ckks scheme instance.
    """

    @staticmethod
    def sigmoid_poly(scheme: Ckks, z_input: PyCtxt, scale: int, n_features: int) -> PyCtxt:
        """3rd-degree polynomial approximation of the Sigmoid: f(z) = 0.5 + 0.25*z - 0.02083*z^3."""
        normalized_z      = scheme.normalize_scale(z_input, scale)
        z_squared         = scheme.multiply(normalized_z, normalized_z)
        z_cubed           = scheme.multiply(z_squared, normalized_z)
        term_degree_1     = scheme.multiply_scalar(0.25, normalized_z)
        term_degree_3     = scheme.multiply_scalar(-0.02083, z_cubed)
        temp_sum          = scheme.add(term_degree_1, term_degree_3)
        activation_output = scheme.add_plain_scalar(temp_sum, 0.5)
        del z_squared, z_cubed, term_degree_1, term_degree_3, temp_sum
        return scheme.normalize_scale(activation_output, scale)

    @staticmethod
    def forward_and_error(scheme: Ckks, encrypted_x: PyCtxt, encrypted_y: PyCtxt, encrypted_weights: PyCtxt, encrypted_bias: PyCtxt, n_features: int, scale: int) -> PyCtxt:
        """Forward pass: linear combination + sigmoid activation, returns prediction error."""
        linear_output = scheme.dot_product(encrypted_x, encrypted_weights)
        linear_output = scheme.add(linear_output, encrypted_bias)
        linear_output = scheme.normalize_scale(linear_output, scale)
        prediction    = PPLR.sigmoid_poly(scheme=scheme, z_input=linear_output, scale=scale, n_features=n_features)
        error_diff    = scheme.subtract(prediction, encrypted_y)
        del linear_output, prediction
        return error_diff

    @staticmethod
    def encrypted_gradients(scheme: Ckks, encrypted_X_batch: List[PyCtxt], encrypted_y_batch: List[PyCtxt], encrypted_weights: PyCtxt, encrypted_bias: PyCtxt, n_features: int, scale: int) -> Tuple[PyCtxt, PyCtxt]:
        """Accumulate encrypted gradients across a data batch."""
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
                old_weight_acc              = weight_gradient_accumulator
                old_bias_acc                = bias_gradient_accumulator
                weight_gradient_accumulator = scheme.add(weight_gradient_accumulator, gradient_term)
                bias_gradient_accumulator   = scheme.add(bias_gradient_accumulator, sample_error)
                del old_weight_acc, old_bias_acc
            del sample_error, gradient_term
        return weight_gradient_accumulator, bias_gradient_accumulator

    @staticmethod
    def train(scheme: Ckks, learning_rate: float, encrypted_weights: PyCtxt, encrypted_bias: PyCtxt, encrypted_X: List[PyCtxt], encrypted_y: List[PyCtxt], n_features: int, scale: int, n_samples: int) -> Tuple[PyCtxt, PyCtxt]:
        """Single training epoch for Privacy Preserving Logistic Regression."""
        combined_lr_m = learning_rate / float(n_samples)
        sum_dw, sum_db = PPLR.encrypted_gradients(
            scheme            = scheme,
            encrypted_X_batch = encrypted_X,
            encrypted_y_batch = encrypted_y,
            encrypted_weights = encrypted_weights,
            encrypted_bias    = encrypted_bias,
            n_features        = n_features,
            scale             = scale,
        )
        step_weights      = scheme.multiply_scalar(combined_lr_m, sum_dw)
        step_bias         = scheme.multiply_scalar(combined_lr_m, sum_db)
        encrypted_weights = scheme.subtract(encrypted_weights, step_weights)
        encrypted_bias    = scheme.subtract(encrypted_bias, step_bias)
        encrypted_weights = scheme.normalize_scale(encrypted_weights, scale)
        encrypted_bias    = scheme.normalize_scale(encrypted_bias, scale)
        del sum_dw, sum_db, step_weights, step_bias
        gc.collect()
        return encrypted_weights, encrypted_bias

    @staticmethod
    def predict(scheme: Ckks, encrypted_X_test: List[PyCtxt], encrypted_weights: PyCtxt, encrypted_bias: PyCtxt, scale: int, n_features: int) -> List[PyCtxt]:
        """Generate encrypted classification probabilities on a test dataset."""
        encrypted_predictions: List[PyCtxt] = []
        enc_w_norm = scheme.normalize_scale(encrypted_weights, scale)
        enc_b_norm = scheme.normalize_scale(encrypted_bias, scale)
        for idx, current_x in enumerate(encrypted_X_test):
            curr_w = enc_w_norm.copy()
            curr_b = enc_b_norm.copy()
            linear_z = scheme.dot_product(current_x, curr_w)
            old_z = linear_z
            linear_z = scheme.add(linear_z, curr_b)
            del old_z
            old_z2 = linear_z
            linear_z = scheme.normalize_scale(linear_z, scale)
            del old_z2
            prediction_h = PPLR.sigmoid_poly(
                scheme      = scheme,
                z_input     = linear_z,
                scale       = scale,
                n_features  = n_features
            )
            encrypted_predictions.append(prediction_h)
            del linear_z, curr_w, curr_b, prediction_h
        return encrypted_predictions
