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

PLAINTEXT = np.array([
    [0.73, 8.84],
    [49.93, 34.44],
    [0.57, 65.04],
    [11.00, 22.50],
], dtype=np.float64)


# def demo_liu():
print("=" * 60)
print("DEMO LIU DataOwner")
print("=" * 60)

print("\n[1] Building DataOwner with Liu scheme...")
do = DataOwner.with_algorithm(Algorithm.NONE) \
    .with_scheme(Scheme.LIU) \
    .with_scheme_params(LiuParams(
        security_level=128, _round=True, decimals=6, seed=1,
    )) \
    .build()

print("\n[2] Outsourcing plaintext matrix via DataOwner...")
do.outsourcedData(plaintext_matrix=PLAINTEXT)
liu = do.primary_scheme


print("\n[3] Scalar encrypt / decrypt")
scalar = 42.0
ct = liu.encrypt_scalar(scalar).data
pt = liu.decrypt_scalar(ct).data
print(f"    plaintext={scalar} -> decrypt -> {pt:.6f}")

print("\n[4] Vector encrypt / decrypt")
vec = np.array([10.0, 20.0, 30.0], dtype=np.float64)
ct_vec = liu.encrypt_vector(vec).data
pt_vec = liu.decrypt_vector(ct_vec).data
print(f"    plaintext= {vec}")
print(f"    decrypted= {pt_vec}")
print(f"    shapes: encrypted={ct_vec.shape}, decrypted={pt_vec.shape}")

print("\n[5] Matrix encrypt / decrypt")
ct_mat = liu.encrypt_matrix(PLAINTEXT).data
pt_mat = liu.decrypt_matrix(ct_mat).data
print(f"    plaintext shape={PLAINTEXT.shape} -> encrypted shape={ct_mat.shape} -> decrypted shape={pt_mat.shape}")
print(f"    match: {np.allclose(pt_mat, PLAINTEXT, rtol=1e-4)}")

print("\n[6] Homomorphic ADD: E(5) + E(2)")
c1 = liu.encrypt_scalar(5.0).data
c2 = liu.encrypt_scalar(2.0).data
enc_add = liu.add(ciphertext_1=c1, ciphertext_2=c2)
result = liu.decrypt_scalar(enc_add).data
print(f"    5.0 + 2.0 = {result:.6f}  ok={abs(result - 7.0) < 1e-5}")

print("\n[7] Homomorphic MULTIPLY_SCALAR: E(5) * 3")
enc_ms = liu.multiply_scalar(scalar=3.0, ciphertext=c1)
result = liu.decrypt_scalar(enc_ms).data
print(f"    5.0 * 3.0 = {result:.6f}  ok={abs(result - 15.0) < 1e-5}")

print("\nLiu demo completed.\n")
