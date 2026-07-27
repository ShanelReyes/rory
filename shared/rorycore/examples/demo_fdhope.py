"""
Demo: FDHOPE — Frequency Concealment and Distribution Order-Preserving Encryption.

Shows:
  - Key generation from a dataset
  - Scalar, vector, matrix, and tensor encryption
  - Order preservation
  - Key persistence (save/load)
  - Decrypt: NotImplementedError (expected)
"""

import numpy as np
import os
from rory.core.security.cryptosystem.fdhope import Fdhope

# ---- Toy dataset ----
dataset = np.array([
    [0.5, 1.2],
    [3.4, 2.1],
    [7.8, 6.5],
    [9.1, 8.3],
], dtype=np.float64)

print("=" * 60)
print("DEMO FDHOPE — Order-Preserving Encryption")
print("=" * 60)

# ---- 1. Key generation ----
print("\n[1] Generating keys from dataset (shape={})...".format(dataset.shape))
fdhope = Fdhope(seed=42)
messagespace, cipherspace = fdhope.generate_keys(
    dataset=dataset,
    minVal=0,
    max_range=8,
    proportion=15,
    range_limit=2,
    default_intervalLenght=0.001,
)
print("    Messagespace ranges:", list(messagespace.keys()))
print("    Cipherspace  ranges:", list(cipherspace.keys()))

# ---- 2. Scalar encryption ----
print("\n[2] Scalar encryption")
scalar_pos = 5.0
scalar_neg = -3.0
ct_pos = fdhope.encrypt_scalar(scalar_pos).data
ct_neg = fdhope.encrypt_scalar(scalar_neg).data
print(f"    plaintext={scalar_pos}  → ciphertext={ct_pos:.4f}")
print(f"    plaintext={scalar_neg} → ciphertext={ct_neg:.4f}")
print(f"    Sign preserved: {scalar_neg} < 0 and ct < 0 → {ct_neg < 0}")

# ---- 3. Vector encryption ----
print("\n[3] Vector encryption")
vector = np.array([1.0, 2.0, 3.0], dtype=np.float64)
ct_vec = fdhope.encrypt_vector(vector).data
print(f"    plaintext={vector}")
print(f"    ciphertext={ct_vec}")

# ---- 4. Matrix encryption ----
print("\n[4] Matrix encryption")
matrix = np.array([[0.5, 1.2], [3.4, 2.1]], dtype=np.float64)
ct_mat = fdhope.encrypt_matrix(matrix).data
print(f"    plaintext shape={matrix.shape} → ciphertext shape={ct_mat.shape}")
print(f"    ciphertext:\n{ct_mat}")

# ---- 5. Tensor encryption (3D) ----
print("\n[5] Tensor encryption (3D)")
tensor = np.array([[[0.5], [1.2]], [[3.4], [2.1]]], dtype=np.float64)
ct_tensor = fdhope.encrypt_tensor(tensor).data
print(f"    plaintext shape={tensor.shape} → ciphertext shape={ct_tensor.shape}")

# ---- 6. Order preservation ----
print("\n[6] Order preservation check")
small = fdhope.encrypt_scalar(1.0).data
large = fdhope.encrypt_scalar(9.0).data
print(f"    E(1.0) = {small:.4f}, E(9.0) = {large:.4f}")
print(f"    abs(E(1.0)) < abs(E(9.0)) → {abs(small) < abs(large)}")

# ---- 7. Decrypt = NotImplementedError ----
print("\n[7] Decrypt (expected: NotImplementedError)")
try:
    fdhope.decrypt_scalar(ct_pos)
    print("    [UNEXPECTED] decrypt_scalar did NOT raise!")
except NotImplementedError as e:
    print(f"    OK — {e}")

# ---- 8. Key persistence ----
print("\n[8] Key persistence (save / load)")
tmp_path = "/tmp/fdhope_demo_keys"
fdhope.save_keys(tmp_path)
print(f"    Keys saved to {tmp_path}")

fdhope2 = Fdhope()
fdhope2.load_keys(tmp_path)
ct_reloaded = fdhope2.encrypt_scalar(5.0).data
print(f"    Reloaded instance: E(5.0) = {ct_reloaded:.4f}")

# Cleanup
os.remove(f"{tmp_path}/fdhope_keys.pkl")
os.rmdir(tmp_path)

print("\n" + "=" * 60)
print("FDHOPE demo completed successfully.")
print("=" * 60)
