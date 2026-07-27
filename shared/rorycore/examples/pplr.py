"""
Demo: PPLR — Privacy Preserving Logistic Regression (CKKS).
"""
import time
import numpy as np
from rory.core.enums import Algorithm, Scheme
from rory.core.security.dataowner import DataOwner
from rory.core.security.scheme_params import CkksParams
from rory.core.security.cryptosystem.pqc.ckks import Ckks, CkksModes
from rory.core.classification.secure.pqc.pplr import PPLR

print("=" * 60)
print("DEMO PPLR — Privacy Preserving Logistic Regression (CKKS)")
print("  Generating CKKS keys in memory...")
print("=" * 60)

mode = CkksModes.ML
security_level = 128

rng = np.random.RandomState(42)
n_samples = 8
n_features = 2
X = rng.randn(n_samples, n_features).astype(np.float64)
y = rng.binomial(1, 0.5, size=n_samples).astype(np.float64)

epochs = 2
learning_rate = 0.1

do = DataOwner.with_algorithm(Algorithm.PPLR) \
    .with_scheme(Scheme.CKKS) \
    .with_scheme_params(CkksParams(
        security_level=security_level, mode="ml",
        enable_relinearize=True, enable_rotate=True,
    )) \
    .build()

result = do.outsourcedData(
    plaintext_matrix = X,
    label_vector     = y,
    n_features       = n_features,
)

ckks = do.primary_scheme
scale = ckks.SECURITY_LEVELS[mode.value][security_level]["scale"]
print(f"  Keys generated. n_features={ckks.n_features}")

enc_weights = result.encrypted_weights
enc_bias = result.encrypted_bias


start_time = time.time()

for epoch in range(epochs):
    fresh_X = ckks.encrypt_matrix(X).data
    fresh_y = ckks.encrypt_matrix(y).data

    enc_weights, enc_bias = PPLR.fit(
        scheme                = ckks,
        learning_rate         = learning_rate,
        encrypted_weights     = enc_weights.data,
        encrypted_bias        = enc_bias.data,
        encrypted_matrix      = fresh_X,
        encrypted_labelvector = fresh_y,
        n_features            = n_features,
        scale                 = scale,
        n_samples             = n_samples,
    )
    # enc_weights = enc_weights.data
    # enc_bias = enc_bias.data
    print("fuera de PPLR.fit")
    weight_plain = ckks.decrypt_vector(enc_weights)
    bias_plain   = ckks.decrypt_vector(enc_bias)

    # print("weights plain type:", type(weight_plain.data))
    # print("bias plain type:", type(bias_plain.data))

    weight_clean = Ckks.post_process(weight_plain.data)
    bias_clean   = Ckks.post_process(bias_plain.data)

    # print("weight_clean type:", type(weight_clean))
    # print("bias_clean type:", type(bias_clean))
    
    enc_weights  = ckks.encrypt_vector(weight_clean)
    enc_bias     = ckks.encrypt_vector(bias_clean)

    # print("encrypted_weights type:", type(enc_weights.data))
    # print("encrypted_bias type:", type(enc_bias.data))

    print(f"  Epoch {epoch + 1}/{epochs} completed")

enc_test = ckks.encrypt_matrix(X)
enc_predictions = PPLR.predict(
    scheme            = ckks,
    encrypted_matrix  = enc_test.data,
    encrypted_weights = enc_weights.data,
    encrypted_bias    = enc_bias.data,
    scale             = scale,
    n_features        = n_features,
)

decrypted = [ckks.decrypt_vector(p).data[0] for p in enc_predictions]
labels = [1 if v >= 0.5 else 0 for v in decrypted]

print(f"Labels:      {labels}")
print(f"Service time: {time.time() - start_time:.2f}s")
print("=" * 60)
print("PPLR demo completed successfully.")
print("=" * 60)
