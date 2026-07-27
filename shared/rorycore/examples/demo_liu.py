"""
Demo: Liu — Symmetric Fully Homomorphic Encryption.

Shows:
  - Key generation
  - Scalar, vector, and matrix encrypt/decrypt
  - Homomorphic add, subtract, multiply, multiply_scalar, decrypt_multiply
  - Key persistence (save/load)
"""

import numpy as np
import os
from rory.core.security.cryptosystem.liu import Liu

# ---- Toy dataset ----
PLAINTEXT_MATRIX = np.array([
    [0.73, 8.84],
    [49.93, 34.44],
    [0.57, 65.04],
], dtype=np.float64)

print("=" * 60)
print("DEMO Liu — Symmetric Fully Homomorphic Encryption")
print("=" * 60)

# ---- 1. Key generation ----
print("\n[1] Generating keys (security_level=128, m=3)...")
liu = Liu(_round=True, decimals=6, security_level=128)
liu.generate_keys(security_level=128)
print(f"    Secret key generated: m={liu.m}, len(sk)={len(liu.sk)}")

# ---- 2. Scalar encrypt / decrypt ----
print("\n[2] Scalar encrypt / decrypt")
scalar = 42.0
ct_scalar = liu.encrypt_scalar(scalar).data
pt_scalar = liu.decrypt_scalar(ct_scalar).data
print(f"    plaintext={scalar} → encrypt → decrypt → {pt_scalar:.6f}")
assert abs(pt_scalar - scalar) < 1e-5

# ---- 3. Vector encrypt / decrypt ----
print("\n[3] Vector encrypt / decrypt")
vector = np.array([10.0, 20.0, 30.0], dtype=np.float64)
ct_vec = liu.encrypt_vector(vector).data
pt_vec = liu.decrypt_vector(ct_vec).data
print(f"    plaintext={vector}")
print(f"    decrypted ={pt_vec}")
print(f"    shapes: encrypted={ct_vec.shape}, decrypted={pt_vec.shape}")

# ---- 4. Matrix encrypt / decrypt ----
print("\n[4] Matrix encrypt / decrypt")
ct_mat = liu.encrypt_matrix(PLAINTEXT_MATRIX).data
pt_mat = liu.decrypt_matrix(ct_mat).data
print(f"    plaintext shape={PLAINTEXT_MATRIX.shape} → encrypted shape={ct_mat.shape} → decrypted shape={pt_mat.shape}")
print(f"    Decryption matches: {np.allclose(pt_mat, PLAINTEXT_MATRIX, rtol=1e-4)}")

# ---- 5. Homomorphic ADD ----
print("\n[5] Homomorphic ADD: E(5) + E(2)")
v1, v2 = 5.0, 2.0
ct1 = liu.encrypt_scalar(v1).data
ct2 = liu.encrypt_scalar(v2).data
enc_add = liu.add(ciphertext_1=ct1, ciphertext_2=ct2)
result_add = liu.decrypt_scalar(enc_add).data
print(f"    {v1} + {v2} = {result_add:.6f}  (ok={abs(result_add - (v1+v2)) < 1e-5})")

# ---- 6. Homomorphic SUBTRACT ----
print("\n[6] Homomorphic SUBTRACT: E(5) - E(2)")
enc_sub = liu.subtract(ciphertext_1=ct1, ciphertext_2=ct2)
result_sub = liu.decrypt_scalar(enc_sub).data
print(f"    {v1} - {v2} = {result_sub:.6f}  (ok={abs(result_sub - (v1-v2)) < 1e-5})")

# ---- 7. Homomorphic MULTIPLY (ciphertext × ciphertext) ----
print("\n[7] Homomorphic MULTIPLY: E(5) * E(2)")
enc_mul = liu.multiply(ciphertext_1=ct1, ciphertext_2=ct2)
result_mul = liu.decrypt_multiply(enc_mul)
print(f"    {v1} * {v2} = {result_mul:.6f}  (ok={abs(result_mul - (v1*v2)) < 1e-5})")

# ---- 8. Homomorphic MULTIPLY_SCALAR ----
print("\n[8] Homomorphic MULTIPLY_SCALAR: E(5) * 3")
scalar_factor = 3.0
enc_ms = liu.multiply_scalar(scalar=scalar_factor, ciphertext=ct1)
result_ms = liu.decrypt_scalar(enc_ms).data
print(f"    {v1} * {scalar_factor} = {result_ms:.6f}  (ok={abs(result_ms - (v1*scalar_factor)) < 1e-5})")

# ---- 9. Key persistence ----
print("\n[9] Key persistence (save / load)")
tmp_path = "/tmp/liu_demo_keys"
liu.save_keys(tmp_path)
print(f"    Keys saved to {tmp_path}")

liu2 = Liu(_round=True, decimals=6)
liu2.load_keys(tmp_path)
ct_reloaded = liu2.encrypt_scalar(99.0).data
pt_reloaded = liu2.decrypt_scalar(ct_reloaded).data
print(f"    Reloaded instance: E(99.0) → decrypt → {pt_reloaded:.6f}")

os.remove(f"{tmp_path}/liu_sk.pkl")
os.rmdir(tmp_path)

print("\n" + "=" * 60)
print("Liu demo completed successfully.")
print("=" * 60)
