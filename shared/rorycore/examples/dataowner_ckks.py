"""
DataOwner Demo — Liu & CKKS encryption/decryption + homomorphic ops.

Shows how to use DataOwner with Algorithm.NONE (scheme-only mode)
to encrypt, decrypt, and perform basic homomorphic operations
(add, multiply_scalar) using both the Liu and CKKS schemes.
"""

import numpy as np
import sys
from rory.core.enums import Algorithm, Scheme
from rory.core.security.dataowner import DataOwner
from rory.core.security.scheme_params import LiuParams, CkksParams
from rory.core.security.cryptosystem.pqc.ckks import Ckks
from rory.core.interfaces.outsourced_result import OutsourcedDataResult

PLAINTEXT = np.array([
    [0.73, 8.84],
    [49.93, 34.44],
    [0.57, 65.04],
    [11.00, 22.50],
], dtype=np.float64)



print("=" * 60)
print("DEMO CKKS DataOwner")
print("=" * 60)

print("\n[1] Building DataOwner with CKKS scheme...")
do = DataOwner.with_algorithm(Algorithm.NONE) \
    .with_scheme(Scheme.CKKS) \
    .with_scheme_params(CkksParams(
        security_level=128,
        enable_relinearize=True,
        enable_rotate=True,
        decimals=6,
    )) \
    .build()

print("\n[2] Outsourcing plaintext matrix via DataOwner...")
osdr = do.outsourcedData(plaintext_matrix=PLAINTEXT)
assert isinstance(osdr, OutsourcedDataResult)
ckks:Ckks = do.primary_scheme

print("\n[3] Scalar encrypt / decrypt")
scalar = 3.14
ct = ckks.encrypt_scalar(scalar).data
pt = ckks.decrypt_scalar(ct).data
print(f"    plaintext={scalar} -> decrypt -> {pt:.4f}")

print("\n[4] Vector encrypt / decrypt")
vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
ct_vec = ckks.encrypt_vector(vec).data
pt_vec = ckks.decrypt_vector(ct_vec).data[:len(vec)]
print(f"    plaintext= {vec}")
print(f"    decrypted= {np.round(pt_vec, 4)}")
print(f"    match: {np.allclose(pt_vec, vec, rtol=1e-2)}")

print("\n[5] Matrix encrypt / decrypt")
ct_mat = ckks.encrypt_matrix(PLAINTEXT).data
pt_mat = ckks.decrypt_matrix(ct_mat, adjust=True, shape=PLAINTEXT.shape).data
print(f"    plaintext shape={PLAINTEXT.shape}, encrypted count={len(ct_mat)}")
print(f"    decrypted:\n{np.round(pt_mat, 4)}")
print(f"    match: {np.allclose(pt_mat, PLAINTEXT, rtol=1e-2)}")

print("\n[6] Homomorphic ADD: E(v1) + E(v2)")
v1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
v2 = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)
ct_a = ckks.encrypt_vector(v1).data
ct_b = ckks.encrypt_vector(v2).data
enc_add = ckks.add(ciphertext_1=ct_a, ciphertext_2=ct_b)
result = np.round(ckks.decrypt_vector(enc_add).data, decimals=4)[:len(v1)]
expected = np.round(v1 + v2, decimals=4)
print(f"    decrypted= {result}")
print(f"    expected = {expected}")
print(f"    match: {np.allclose(result, expected, rtol=1e-2)}")

print("\n[7] Homomorphic MULTIPLY_SCALAR: E(v1) * 2.5")
enc_ms = ckks.multiply_scalar(scalar=2.5, ciphertext=ct_a)
result = np.round(ckks.decrypt_vector(enc_ms).data, decimals=4)[:len(v1)]
expected = np.round(v1 * 2.5, decimals=4)
print(f"    decrypted= {result}")
print(f"    expected = {expected}")
print(f"    match: {np.allclose(result, expected, rtol=1e-2)}")

print("\nCKKS demo completed.\n")


# if __name__ == "__main__":
#     demo_liu()

#     # try:
#     #     demo_ckks()
#     # except ImportError as e:
#     #     print(f"CKKS requires Pyfhel. Import error: {e}")
#     #     print("Install with: poetry install -E pqc")
#     #     sys.exit(1)
#     # except Exception as e:
#     #     print(f"CKKS demo failed: {type(e).__name__}: {e}")
#     #     print("Note: CKKS requires pre-generated keys or a compatible environment.")
#     #     sys.exit(1)

#     print("=" * 60)
#     print("DataOwner demo completed successfully.")
#     print("=" * 60)
