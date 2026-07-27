"""
Demo: CKKS — Fully Homomorphic Encryption (PQC variant).

Shows:
  - Key generation
  - Scalar, vector, and matrix encrypt/decrypt
  - Homomorphic add, subtract, multiply, multiply_scalar
  - dot_product, add_plain_scalar, normalize_scale
  - Key persistence (save/load)

Requires: Pyfhel with CKKS support (pip install Pyfhel)
"""

import numpy as np
import os
import sys
from rory.core.security.cryptosystem.pqc.ckks import Ckks, CkksModes
from Pyfhel import PyCtxt
from rory.core.security.cryptosystem.pqc.ckks import Ckks as CkksClass

def demo_ckks():
    # ---- Toy dataset ----
    MATRIX = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
    ], dtype=np.float64)

    VEC_A = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    VEC_B = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)

    print("=" * 60)
    print("DEMO CKKS — Fully Homomorphic Encryption (PQC)")
    print("=" * 60)

    # ---- 1. Key generation ----
    print("\n[1] Generating CKKS context and keys (security_level=128, mode=DEFAULT)...")
    ckks = Ckks()
    ckks.generate_keys(
        security_level=128,
        mode=CkksModes.DEFAULT,
        enable_relinearize=True,
        enable_rotate=True,
    )
    print(f"    n_features={ckks.n_features}, scale={ckks.scale}")
    print(f"    he_object configured: {ckks.he_object is not None}")

    # ---- 2. Scalar encrypt / decrypt ----
    print("\n[2] Scalar encrypt / decrypt")
    scalar = 3.14
    ct_scalar = ckks.encrypt_scalar(scalar).data
    pt_scalar = ckks.decrypt_scalar(ct_scalar).data
    print(f"    plaintext={scalar} → encrypt → decrypt → {pt_scalar:.4f}")

    # ---- 3. Vector encrypt / decrypt ----
    print("\n[3] Vector encrypt / decrypt")
    ct_vec = ckks.encrypt_vector(VEC_A).data
    pt_vec = ckks.decrypt_vector(ct_vec).data[:len(VEC_A)]
    print(f"    plaintext={VEC_A}")
    print(f"    decrypted ={np.round(pt_vec, 4)}")
    print(f"    match: {np.allclose(pt_vec, VEC_A, rtol=1e-2)}")

    # ---- 4. Matrix encrypt / decrypt ----
    print("\n[4] Matrix encrypt / decrypt")
    ct_mat = ckks.encrypt_matrix(MATRIX).data
    pt_mat = ckks.decrypt_matrix(ct_mat, adjust=True, shape=MATRIX.shape).data
    print(f"    plaintext shape={MATRIX.shape}, encrypted count={len(ct_mat)}")
    print(f"    decrypted:\n{np.round(pt_mat, 4)}")
    print(f"    match: {np.allclose(pt_mat, MATRIX, rtol=1e-2)}")

    # ---- 5. Homomorphic ADD ----
    print("\n[5] Homomorphic ADD: E(v1) + E(v2)")
    ct_a = ckks.encrypt_vector(VEC_A).data
    ct_b = ckks.encrypt_vector(VEC_B).data
    enc_add = ckks.add(ciphertext_1=ct_a, ciphertext_2=ct_b)
    result_add = np.round(ckks.decrypt_vector(enc_add).data, decimals=4)[:len(VEC_A)]
    expected_add = np.round(VEC_A + VEC_B, decimals=4)
    print(f"    decrypted = {result_add }")
    print(f"    expected  = {expected_add}")
    print(f"    match: {np.allclose(result_add, expected_add, rtol=1e-2)}")

    # ---- 6. Homomorphic SUBTRACT ----
    print("\n[6] Homomorphic SUBTRACT: E(v1) - E(v2)")
    ct_a2 = ckks.encrypt_vector(VEC_A).data
    ct_b2 = ckks.encrypt_vector(VEC_B).data
    enc_sub = ckks.subtract(ciphertext_1=ct_a2, ciphertext_2=ct_b2)
    result_sub = np.round(ckks.decrypt_vector(enc_sub).data, decimals=4)[:len(VEC_A)]
    expected_sub = np.round(VEC_A - VEC_B, decimals=4)
    print(f"    decrypted = {result_sub}")
    print(f"    expected  = {expected_sub}")
    print(f"    match: {np.allclose(result_sub, expected_sub, rtol=1e-2)}")

    # ---- 7. Homomorphic MULTIPLY ----
    print("\n[7] Homomorphic MULTIPLY: E(v1) * E(v2)")
    ct_a3 = ckks.encrypt_vector(VEC_A).data
    ct_b3 = ckks.encrypt_vector(VEC_B).data
    enc_mul = ckks.multiply(ciphertext_1=ct_a3, ciphertext_2=ct_b3)
    result_mul = np.round(ckks.decrypt_vector(enc_mul).data, decimals=4)
    expected_mul = np.round(VEC_A * VEC_B, decimals=4)
    print(f"    decrypted = {result_mul[:4]} (first 4 slots)")
    print(f"    expected  = {expected_mul[:4]}")
    print(f"    match (first 4): {np.allclose(result_mul[:4], expected_mul[:4], rtol=1e-2)}")

    # ---- 8. Homomorphic MULTIPLY_SCALAR ----
    print("\n[8] Homomorphic MULTIPLY_SCALAR: E(v1) * 2.5")
    ct_a4 = ckks.encrypt_vector(VEC_A).data
    enc_ms = ckks.multiply_scalar(scalar=2.5, ciphertext=ct_a4)
    result_ms = np.round(ckks.decrypt_vector(enc_ms).data, decimals=4)[:len(VEC_A)]
    expected_ms = np.round(VEC_A * 2.5, decimals=4)
    print(f"    decrypted = {result_ms}")
    print(f"    expected  = {expected_ms}")
    print(f"    match: {np.allclose(result_ms, expected_ms, rtol=1e-2)}")

    # ---- 9. DOT PRODUCT ----
    print("\n[9] DOT PRODUCT: E(v1) · E(v2)")
    ct_a5 = ckks.encrypt_vector(VEC_A).data
    ct_b5 = ckks.encrypt_vector(VEC_B).data
    enc_dot = ckks.dot_product(ciphertext_1=ct_a5, ciphertext_2=ct_b5)
    result_dot = ckks.decrypt_scalar(enc_dot).data
    expected_dot = np.dot(VEC_A, VEC_B)
    print(f"    decrypted = {result_dot:.4f}")
    print(f"    expected  = {expected_dot:.4f}")
    print(f"    match: {abs(result_dot - expected_dot) < 0.1}")

    # ---- 10. ADD_PLAIN_SCALAR ----
    print("\n[10] ADD_PLAIN_SCALAR: E(v1) + 1.0")
    ct_a6 = ckks.encrypt_vector(VEC_A).data
    enc_aps = ckks.add_plain_scalar(ciphertext=ct_a6, scalar=1.0)
    result_aps = np.round(ckks.decrypt_vector(enc_aps).data, decimals=4)[:len(VEC_A)]
    expected_aps = np.round(VEC_A + 1.0, decimals=4)
    print(f"    decrypted = {result_aps}")
    print(f"    expected  = {expected_aps}")
    print(f"    match: {np.allclose(result_aps, expected_aps, rtol=1e-2)}")

    # ---- 11. NORMALIZE_SCALE ----
    print("\n[11] NORMALIZE_SCALE after multiply")
    ct_a7 = ckks.encrypt_vector(VEC_A).data
    ct_b7 = ckks.encrypt_vector(VEC_B).data
    enc_mul2 = ckks.multiply(ct_a7, ct_b7)
    enc_norm = ckks.normalize_scale(enc_mul2, scale=ckks.scale)
    print(f"    Scale normalized (CKKS noise management)")

    # ---- 12. Key persistence ----
    print("\n[12] Key persistence (save / load)")
    tmp_path = "/tmp/ckks_demo_keys"
    ckks.save_keys(tmp_path)
    print(f"    Keys saved to {tmp_path}/")

    ckks2 = Ckks.load_pyfhel_client(path=tmp_path)
    ckks2 = CkksClass(he_object=ckks2, n_features=ckks2.n // 2)
    # print("A")
    # print(f"    Keys loaded from {tmp_path}/, n_features={ckks2.n_features}")
    ct_reloaded = ckks2.encrypt_scalar(42.0).data
    # print(f"    Reloaded instance: E(42.0) → encrypt → {ct_reloaded}")
    pt_reloaded = ckks2.decrypt_scalar(ct_reloaded).data
    # print(f"    Reloaded instance: E(42.0) → decrypt → {pt_reloaded:.4f}")

    for fname in ["ctx", "pubkey", "secretkey","relinkey","rotatekey"]:
        os.remove(f"{tmp_path}/{fname}")
    os.rmdir(tmp_path)

    print("\n" + "=" * 60)
    print("CKKS demo completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        demo_ckks()
    except ImportError as e:
        print(f"CKKS requires Pyfhel. Import error: {e}")
        print("Install with: poetry install -E pqc")
        sys.exit(1)
    except Exception as e:
        print(f"CKKS demo failed: {type(e).__name__}: {e}")
        print("Note: CKKS requires pre-generated keys or a compatible environment.")
        sys.exit(1)
