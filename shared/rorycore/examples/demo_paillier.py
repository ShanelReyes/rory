"""
Demo: Paillier — Partially Homomorphic Encryption.

Shows:
  - Key generation
  - Scalar, vector, and matrix encrypt/decrypt
  - Homomorphic add and multiply_scalar
  - Multiply(c1,c2) NOT supported (PHE limitation)
  - Key persistence (save/load)
"""

import numpy as np
import os
from rory.core.security.cryptosystem.paillier import Paillier

# ---- Toy dataset ----
MATRIX = np.array([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
], dtype=np.float64)

print("=" * 60)
print("DEMO Paillier — Partially Homomorphic Encryption")
print("=" * 60)

# ---- 1. Key generation ----
print("\n[1] Generating 128-bit keypair...")
paillier = Paillier()
paillier.generate_keys(security_level=128)
print(f"    Public  key n bits: {paillier.public_key.n.bit_length()}")
print(f"    Private key generated: {paillier.private_key is not None}")

# ---- 2. Scalar encrypt / decrypt ----
print("\n[2] Scalar encrypt / decrypt")
scalar = 42.0
ct_scalar = paillier.encrypt_scalar(scalar).data
pt_scalar = paillier.decrypt_scalar(ct_scalar).data
print(f"    plaintext={scalar} → encrypt → decrypt → {pt_scalar}")
assert abs(pt_scalar - scalar) < 1e-10

# ---- 3. Vector encrypt / decrypt ----
print("\n[3] Vector encrypt / decrypt")
vector = [1.0, 2.0, 3.0, 4.0, 5.0]
ct_vec = paillier.encrypt_vector(vector).data
pt_vec = paillier.decrypt_vector(ct_vec).data
print(f"    plaintext={vector}")
print(f"    decrypted ={pt_vec}")
assert np.allclose(pt_vec, vector)

# ---- 4. Matrix encrypt / decrypt ----
print("\n[4] Matrix encrypt / decrypt")
ct_mat = paillier.encrypt_matrix(MATRIX).data
pt_mat = paillier.decrypt_matrix(ct_mat).data
print(f"    plaintext shape={MATRIX.shape}")
print(f"    decrypted:\n{pt_mat}")
assert np.allclose(pt_mat, MATRIX)

# ---- 5. Homomorphic ADD ----
print("\n[5] Homomorphic ADD: E(10) + E(20)")
v1, v2 = 10.0, 20.0
ct1 = paillier.encrypt_scalar(v1).data
ct2 = paillier.encrypt_scalar(v2).data
enc_add = paillier.add(ciphertext_1=ct1, ciphertext_2=ct2)
result_add = paillier.decrypt_scalar(enc_add).data
print(f"    {v1} + {v2} = {result_add}  (ok={abs(result_add - (v1+v2)) < 1e-10})")

# ---- 6. Homomorphic MULTIPLY_SCALAR ----
print("\n[6] Homomorphic MULTIPLY_SCALAR: E(7) * 4")
ct3 = paillier.encrypt_scalar(7.0).data
enc_ms = paillier.multiply_scalar(scalar=4.0, ciphertext=ct3)
result_ms = paillier.decrypt_scalar(enc_ms).data
print(f"    7 * 4 = {result_ms}  (ok={abs(result_ms - 28.0) < 1e-10})")

# ---- 7. multiply(c1,c2) NOT supported ----
print("\n[7] MULTIPLY(c1, c2) — NOT supported (PHE limitation)")
try:
    paillier.multiply(ciphertext_1=ct1, ciphertext_2=ct2)
    print("    [UNEXPECTED] multiply() succeeded!")
except AttributeError:
    print("    OK — multiply(c1,c2) not available in PHE (AttributeError as expected)")

# ---- 8. Key persistence ----
print("\n[8] Key persistence (save / load)")
tmp_path = "/tmp/paillier_demo_keys"
paillier.save_keys(tmp_path, filename="demo-phe")
print(f"    Keys saved to {tmp_path}/demo-phe.{{pub,priv}}")

paillier2 = Paillier()
paillier2.load_keys(tmp_path, filename="demo-phe")
ct_reloaded = paillier2.encrypt_scalar(100.0).data
pt_reloaded = paillier2.decrypt_scalar(ct_reloaded).data
print(f"    Reloaded instance: E(100.0) → decrypt → {pt_reloaded}")

os.remove(f"{tmp_path}/demo-phe.pub")
os.remove(f"{tmp_path}/demo-phe.priv")
os.rmdir(tmp_path)

print("\n" + "=" * 60)
print("Paillier demo completed successfully.")
print("=" * 60)
