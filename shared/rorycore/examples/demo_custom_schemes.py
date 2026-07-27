"""
Demo: Custom Homomorphic Encryption Schemes using Abstract Classes.

Shows how to define fictional HE schemes by implementing only the
required abstract methods from the abstract base classes.

Scheme hierarchy used:
  Cipher → HomomorphicCipher → PartiallyHomomorphicCipher / FullyHomomorphicCipher

Fictional schemes defined here:
  1. FictionalXorCipher(Cipher)        — XOR-based, no homomorphic ops
  2. FictionalAdditivePHE(PartiallyHomomorphicCipher) — additive PHE
  3. FictionalSimpleFHE(FullyHomomorphicCipher)       — additive + multiplicative FHE

Each scheme demonstrates:
  - Minimal abstract method implementation
  - Inherited default behaviour (encrypt_vector, encrypt_matrix, subtract)
  - save_keys / load_keys
  - Polymorphism via a single polymorphic_demo() function
"""

import os
import random
import pickle
from rory.core.security.cryptosystem.abstract import (
    Cipher,
    HomomorphicCipher,
    PartiallyHomomorphicCipher,
    FullyHomomorphicCipher,
)
from rory.core.interfaces.cipher_result import CipherResult


# ============================================================
# 1. FictionalXorCipher — Minimal Cipher (no homomorphic ops)
# ============================================================

class FictionalXorCipher(Cipher):
    """Toy XOR-based cipher: E(x) = x ^ key, D(c) = c ^ key.

    The simplest possible Cipher implementation. Only 3 abstract
    methods needed (generate_keys, encrypt_scalar, decrypt_scalar).
    encrypt_vector, encrypt_matrix, decrypt_vector, decrypt_matrix
    all come for free from the Cipher base class defaults.
    """

    def __init__(self):
        self.key = 0

    # ---- Required abstract methods ----

    def generate_keys(self, *args, **kwargs):
        """Generate a random integer XOR key."""
        self.key = random.randint(1, 2**32 - 1)
        print(f"    [XOR] Generated key: {self.key}")

    def encrypt_scalar(self, plaintext):
        """E(x) = x ^ key"""
        result = int(plaintext) ^ self.key
        return CipherResult(data=result)

    def decrypt_scalar(self, ciphertext):
        """D(c) = c ^ key"""
        result = int(ciphertext) ^ self.key
        return CipherResult(data=result)

    def save_keys(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/xor_key.pkl", "wb") as f:
            pickle.dump({"key": self.key}, f)

    def load_keys(self, path: str):
        with open(f"{path}/xor_key.pkl", "rb") as f:
            self.key = pickle.load(f)["key"]


# ============================================================
# 2. FictionalAdditivePHE — Partially Homomorphic
# ============================================================

class FictionalAdditivePHE(PartiallyHomomorphicCipher):
    """Toy additive PHE: E(x) = x + key, D(c) = c - key.

    Implements 5 abstract methods (3 from Cipher + 2 from
    HomomorphicCipher). Inherits subtract() for free.
    multiply(c1,c2) is NOT available — that's the definition of PHE.
    """

    def __init__(self):
        self.key = 0

    # ---- Required abstract methods (Cipher) ----

    def generate_keys(self, *args, **kwargs):
        self.key = random.randint(1000, 9999)
        print(f"    [AdditivePHE] Generated key: {self.key}")

    def encrypt_scalar(self, plaintext):
        """E(x) = x + key"""
        return CipherResult(data=plaintext + self.key)

    def decrypt_scalar(self, ciphertext):
        """D(c) = c - key"""
        return CipherResult(data=ciphertext - self.key)

    # ---- Required abstract methods (HomomorphicCipher) ----

    def add(self, ciphertext_1, ciphertext_2):
        """E(a) + E(b) = (a+key) + (b+key) - key = a + b + key"""
        return ciphertext_1 + ciphertext_2 - self.key

    def multiply_scalar(self, scalar, ciphertext):
        """E(a) * s = (a+key)*s - (s-1)*key = a*s + key"""
        return scalar * ciphertext - (scalar - 1) * self.key

    def save_keys(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/additive_phe_key.pkl", "wb") as f:
            pickle.dump({"key": self.key}, f)

    def load_keys(self, path: str):
        with open(f"{path}/additive_phe_key.pkl", "rb") as f:
            self.key = pickle.load(f)["key"]


# ============================================================
# 3. FictionalSimpleFHE — Fully Homomorphic
# ============================================================

class FictionalSimpleFHE(FullyHomomorphicCipher):
    """Toy FHE: E(x) = x + KEY, supports add, multiply, multiply_scalar.

    Implements 6 abstract methods. Inherits subtract() for free.
    multiply(c1,c2) is available — that's the definition of FHE.
    """

    KEY = 1000000  # constant for simplicity

    def generate_keys(self, *args, **kwargs):
        print(f"    [SimpleFHE] Key (constant): {self.KEY}")

    def encrypt_scalar(self, plaintext):
        """E(x) = x + KEY"""
        return CipherResult(data=plaintext + self.KEY)

    def decrypt_scalar(self, ciphertext):
        """D(c) = c - KEY"""
        return CipherResult(data=ciphertext - self.KEY)

    def add(self, ciphertext_1, ciphertext_2):
        """E(a) + E(b) = (a+K)+(b+K) - K = a+b+K"""
        return ciphertext_1 + ciphertext_2 - self.KEY

    def multiply_scalar(self, scalar, ciphertext):
        """E(a) * s = s*(a+K) - (s-1)*K = a*s + K"""
        return scalar * ciphertext - (scalar - 1) * self.KEY

    def multiply(self, ciphertext_1, ciphertext_2):
        """E(a) * E(b) = (a+K)*(b+K) - K*a - K*b - K^2 + K"""
        a_plus_key = ciphertext_1
        b_plus_key = ciphertext_2
        raw_product = a_plus_key * b_plus_key
        return raw_product - self.KEY * a_plus_key - self.KEY * b_plus_key + self.KEY**2 + self.KEY

    def save_keys(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/simple_fhe_key.pkl", "wb") as f:
            pickle.dump({"KEY": self.KEY}, f)

    def load_keys(self, path: str):
        with open(f"{path}/simple_fhe_key.pkl", "rb") as f:
            self.KEY = pickle.load(f)["KEY"]


# ============================================================
# Polymorphic demo — works with ANY Cipher subclass
# ============================================================

def polymorphic_demo(scheme: Cipher, name: str):
    """Run a standard test suite on any Cipher instance.

    Demonstrates that the same code works for ALL scheme types
    because they share the Cipher abstract interface.
    """
    print(f"\n{'=' * 60}")
    print(f"Polymorphic demo: {name}")
    print(f"{'=' * 60}")

    # generate_keys is abstract → all schemes have it
    scheme.generate_keys()

    # ---- scalar ----
    print("\n  [Scalar encrypt / decrypt]")
    ct_s = scheme.encrypt_scalar(7)
    pt_s = scheme.decrypt_scalar(ct_s.data)
    print(f"    E(7) = {ct_s.data}, D(c) = {pt_s.data}  (ok={pt_s.data == 7})")

    # ---- vector (inherited default from Cipher) ----
    print("\n  [Vector encrypt / decrypt (inherited default)]")
    vec = [1, 2, 3]
    ct_v = scheme.encrypt_vector(vec)
    pt_v = scheme.decrypt_vector(ct_v.data)
    print(f"    E({vec}) = {ct_v.data}")
    print(f"    D(c)     = {pt_v.data}")

    # ---- matrix (inherited default from Cipher) ----
    print("\n  [Matrix encrypt / decrypt (inherited default)]")
    mat = [[10, 20], [30, 40]]
    ct_m = scheme.encrypt_matrix(mat)
    pt_m = scheme.decrypt_matrix(ct_m.data)
    print(f"    plaintext  = {mat}")
    print(f"    decrypted  = {pt_m.data}")

    # ---- homomorphic ops (only if HomomorphicCipher) ----
    if isinstance(scheme, HomomorphicCipher):
        print(f"\n  [Homomorphic ADD: E(3) + E(5)]")
        c1 = scheme.encrypt_scalar(3).data
        c2 = scheme.encrypt_scalar(5).data
        enc_add = scheme.add(c1, c2)
        pt_add = scheme.decrypt_scalar(enc_add).data
        print(f"    3 + 5 = {pt_add}  (ok={pt_add == 8})")

        print(f"\n  [Homomorphic MULTIPLY_SCALAR: E(3) * 4]")
        enc_ms = scheme.multiply_scalar(4, c1)
        pt_ms = scheme.decrypt_scalar(enc_ms).data
        print(f"    3 * 4 = {pt_ms}  (ok={pt_ms == 12})")

        # subtract is inherited (free) from HomomorphicCipher
        print(f"\n  [Homomorphic SUBTRACT: E(5) - E(3) (inherited)]")
        enc_sub = scheme.subtract(c2, c1)
        pt_sub = scheme.decrypt_scalar(enc_sub).data
        print(f"    5 - 3 = {pt_sub}  (ok={pt_sub == 2})")

    # ---- multiply (only if FullyHomomorphicCipher) ----
    if isinstance(scheme, FullyHomomorphicCipher):
        print(f"\n  [Homomorphic MULTIPLY: E(3) * E(5)]")
        enc_mul = scheme.multiply(c1, c2)
        pt_mul = scheme.decrypt_scalar(enc_mul).data
        print(f"    3 * 5 = {pt_mul}  (ok={pt_mul == 15})")
    elif isinstance(scheme, PartiallyHomomorphicCipher):
        print(f"\n  [Homomorphic MULTIPLY: E(3) * E(5)]")
        print(f"    NOT available (PHE limitation — expected)")

    # ---- persistence ----
    if not isinstance(scheme, FictionalSimpleFHE):
        print(f"\n  [Key persistence (save / load)]")
        tmp_path = f"/tmp/{name.lower().replace(' ', '_')}_keys"
        try:
            scheme.save_keys(tmp_path)
            # Create a new instance of the same class
            new_scheme = type(scheme)()
            new_scheme.load_keys(tmp_path)
            ct_r = new_scheme.encrypt_scalar(99)
            pt_r = new_scheme.decrypt_scalar(ct_r.data)
            print(f"    Reloaded: E(99) → D = {pt_r.data}  (ok={pt_r.data == 99})")
        finally:
            import glob as _glob
            for _f in _glob.glob(f"{tmp_path}/*"):
                os.remove(_f)
            os.rmdir(tmp_path)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 72)
    print("DEMO: Custom Homomorphic Encryption Schemes using Abstract Classes")
    print("=" * 72)
    print()
    print("Three fictional schemes are defined:")
    print("  1. FictionalXorCipher(Cipher)              — XOR-based, no HE ops")
    print("  2. FictionalAdditivePHE(PartiallyHomomorphicCipher) — additive PHE")
    print("  3. FictionalSimpleFHE(FullyHomomorphicCipher)       — additive + multiplicative FHE")
    print()
    print("All use the SAME polymorphic_demo() function.")
    print()

    # ---- Scheme 1: XOR Cipher ----
    xor = FictionalXorCipher()
    polymorphic_demo(xor, "FictionalXorCipher (Cipher)")

    # ---- Scheme 2: Additive PHE ----
    phe = FictionalAdditivePHE()
    polymorphic_demo(phe, "FictionalAdditivePHE (PartiallyHomomorphicCipher)")

    # ---- Scheme 3: Simple FHE ----
    fhe = FictionalSimpleFHE()
    polymorphic_demo(fhe, "FictionalSimpleFHE (FullyHomomorphicCipher)")

    print(f"\n{'=' * 72}")
    print("Custom schemes demo completed.")
    print()

    # ---- Bonus: type hierarchy verification ----
    print("Type hierarchy verification:")
    print(f"  FictionalXorCipher    is Cipher:       {isinstance(xor, Cipher)}")
    print(f"  FictionalAdditivePHE  is Cipher:       {isinstance(phe, Cipher)}")
    print(f"  FictionalAdditivePHE  is HomomorphicCipher: {isinstance(phe, HomomorphicCipher)}")
    print(f"  FictionalSimpleFHE    is Cipher:       {isinstance(fhe, Cipher)}")
    print(f"  FictionalSimpleFHE    is HomomorphicCipher: {isinstance(fhe, HomomorphicCipher)}")
    print(f"  FictionalSimpleFHE    is FullyHomomorphicCipher: {isinstance(fhe, FullyHomomorphicCipher)}")
    print()
    print("Abstract methods required per scheme type:")
    print(f"  Cipher (minimal):          3 abstract methods")
    print(f"  PartiallyHomomorphicCipher: 5 abstract methods (+ subtract inherited)")
    print(f"  FullyHomomorphicCipher:    6 abstract methods (+ subtract inherited)")
    print(f"{'=' * 72}")