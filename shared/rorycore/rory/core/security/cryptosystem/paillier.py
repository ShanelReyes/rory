import warnings
import numpy as np
import os
from phe import paillier
from rory.core.interfaces.cipher_result import CipherResult
from rory.core.security.cryptosystem.abstract import PartiallyHomomorphicCipher
from typing import Tuple, List
import numpy.typing as npt


class Paillier(PartiallyHomomorphicCipher):
	"""A wrapper around the Paillier partially homomorphic encryption scheme.

	Provides methods for key generation, encryption, decryption,
	homomorphic operations (addition, scalar multiplication), and
	key persistence (save/load).

	Args:
		public_key: Optional Paillier public key.
		private_key: Optional Paillier private key.
	"""

	def __init__(self, public_key: 'paillier.PaillierPublicKey' = None, private_key: 'paillier.PaillierPrivateKey' = None):
		self.public_key = public_key
		self.private_key = private_key

	# ---- Standardized interface ----

	def generate_keys(self, security_level: int = 128, output_path: str = "/sink", filename: str = "rory-phe", save: bool = False):
		"""Generate Paillier keypair and store internally.

		Args:
			security_level: Key size in bits (128, 192, or 256).
			output_path: Directory for saving keys.
			filename: Base filename for key files.
			save: If True, persists keys to disk.

		Returns:
			Tuple[PaillierPublicKey, PaillierPrivateKey]: Generated keypair.
		"""
		self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=security_level)
		if save:
			self.save_keys(output_path, filename)
		return self.public_key, self.private_key

	def encrypt_scalar(self, plaintext: float):
		"""Encrypt a single plaintext value.

		Args:
			plaintext: Plaintext value to encrypt.

		Returns:
			CipherResult: Container with the EncryptedNumber.
		"""
		result = self.public_key.encrypt(plaintext)
		return CipherResult(data=result)

	def encrypt_vector(self, plaintext_vector: npt.NDArray):
		"""Encrypt a vector of plaintext values.

		Args:
			plaintext_vector: Vector of plaintext values.

		Returns:
			CipherResult: Container with the encrypted vector.
		"""
		result = self._encryptVector(plaintext_vector=plaintext_vector, public_key=self.public_key)
		return CipherResult(data=result)

	def encrypt_matrix(self, plaintext_matrix: npt.NDArray):
		"""Encrypt a matrix of plaintext values.

		Args:
			plaintext_matrix: Matrix of plaintext values.

		Returns:
			CipherResult: Container with the encrypted matrix.
		"""
		result = self._encryptMatrix(plaintext_matrix=plaintext_matrix, public_key=self.public_key)
		return CipherResult(data=result)

	def decrypt_scalar(self, ciphertext):
		"""Decrypt a single ciphertext value.

		Args:
			ciphertext: EncryptedNumber to decrypt.

		Returns:
			CipherResult: Container with the decrypted plaintext.
		"""
		result = self.private_key.decrypt(ciphertext)
		return CipherResult(data=result)

	def decrypt_vector(self, ciphertext_vector):
		"""Decrypt a vector of ciphertext values.

		Args:
			ciphertext_vector: Vector of EncryptedNumber objects.

		Returns:
			CipherResult: Container with the decrypted vector.
		"""
		result = self._decryptVector(ciphertext_vector=ciphertext_vector, private_key=self.private_key)
		return CipherResult(data=result)

	def decrypt_matrix(self, ciphertext_matrix):
		"""Decrypt a matrix of ciphertext values.

		Args:
			ciphertext_matrix: Matrix of EncryptedNumber objects.

		Returns:
			CipherResult: Container with the decrypted matrix.
		"""
		result = self._decryptMatrix(ciphertext_matrix=ciphertext_matrix, private_key=self.private_key)
		return CipherResult(data=result)

	# Homomorphic operations

	@classmethod
	def add(cls, ciphertext_1, ciphertext_2):
		"""Homomorphic addition: E(a) + E(b) = E(a + b).

		Args:
			ciphertext_1: First encrypted operand.
			ciphertext_2: Second encrypted operand.

		Returns:
			EncryptedNumber: Encrypted sum.
		"""
		return ciphertext_1 + ciphertext_2

	@classmethod
	def multiply_scalar(cls, scalar: float, ciphertext):
		"""Multiply ciphertext by a plaintext scalar: E(a) * b = E(a * b).

		Args:
			scalar: Plaintext multiplier.
			ciphertext: Encrypted operand.

		Returns:
			EncryptedNumber: Encrypted product.
		"""
		return ciphertext * scalar

	def save_keys(self, path: str, filename: str = "rory-phe"):
		"""Save keypair to disk.

		Args:
			path: Directory for key files.
			filename: Base filename (without extension).
		"""
		os.makedirs(path, exist_ok=True)
		n_bytes = self.public_key.n.to_bytes((self.public_key.n.bit_length() + 7) // 8, byteorder='big')
		with open(f"{path}/{filename}.pub", "wb") as pub_file:
			pub_file.write(n_bytes)
		p_bytes = self.private_key.p.to_bytes((self.private_key.p.bit_length() + 7) // 8, byteorder='big')
		q_bytes = self.private_key.q.to_bytes((self.private_key.q.bit_length() + 7) // 8, byteorder='big')
		n_pub_bytes = self.private_key.public_key.n.to_bytes((self.private_key.public_key.n.bit_length() + 7) // 8, byteorder='big')

		with open(f"{path}/{filename}.priv", "wb") as priv_file:
			for component in [p_bytes, q_bytes, n_pub_bytes]:
				priv_file.write(len(component).to_bytes(4, byteorder='big'))
				priv_file.write(component)

	def load_keys(self, path: str, filename: str = "rory-phe"):
		"""Load keypair from disk.

		Args:
			path: Directory where key files are stored.
			filename: Base filename (without extension).
		"""
		with open(f"{path}/{filename}.pub", "rb") as pub_file:
			n_bytes = pub_file.read()
			self.public_key = paillier.PaillierPublicKey(int.from_bytes(n_bytes, byteorder='big'))

		with open(f"{path}/{filename}.priv", "rb") as priv_file:
			p = Paillier._read_component(priv_file)
			q = Paillier._read_component(priv_file)
			public_key_n = Paillier._read_component(priv_file)

			if public_key_n != self.public_key.n:
				raise ValueError("Public key mismatch between .pub and .priv files.")

			self.private_key = paillier.PaillierPrivateKey(self.public_key, p, q)

	# ---- Internal implementations ----

	def _encryptMatrix(self, plaintext_matrix: npt.NDArray, public_key) -> npt.NDArray:
		"""Encrypt a matrix element-wise using the public key.

		Args:
			plaintext_matrix: Matrix of plaintext values.
			public_key: Paillier public key.

		Returns:
			npt.NDArray: Matrix of EncryptedNumber objects.
		"""
		return np.array([self._encryptVector(plaintext_vector=v, public_key=public_key) for v in plaintext_matrix])

	def _encryptVector(self, plaintext_vector: npt.NDArray, public_key) -> npt.NDArray:
		"""Encrypt a vector element-wise using the public key.

		Args:
			plaintext_vector: Vector of plaintext values.
			public_key: Paillier public key.

		Returns:
			npt.NDArray: Vector of EncryptedNumber objects.
		"""
		return np.array([public_key.encrypt(v) for v in plaintext_vector])

	def _decryptMatrix(self, ciphertext_matrix, private_key) -> npt.NDArray:
		"""Decrypt a matrix element-wise using the private key.

		Args:
			ciphertext_matrix: Matrix of EncryptedNumber objects.
			private_key: Paillier private key.

		Returns:
			npt.NDArray: Decrypted plaintext matrix.
		"""
		return np.array([self._decryptVector(ciphertext_vector=c, private_key=private_key) for c in ciphertext_matrix])

	def _decryptVector(self, ciphertext_vector, private_key) -> npt.NDArray:
		"""Decrypt a vector element-wise using the private key.

		Args:
			ciphertext_vector: Vector of EncryptedNumber objects.
			private_key: Paillier private key.

		Returns:
			npt.NDArray: Decrypted plaintext vector.
		"""
		return np.array([private_key.decrypt(c) for c in ciphertext_vector])

	@staticmethod
	def _read_component(file_handle):
		"""Read a length-prefixed integer component from a binary file.

		Args:
			file_handle: Open file handle positioned at the start of
				a 4-byte length prefix followed by the integer bytes.

		Returns:
			int: The deserialized integer.
		"""
		length_bytes = file_handle.read(4)
		length = int.from_bytes(length_bytes, byteorder='big')
		return int.from_bytes(file_handle.read(length), byteorder='big')

	# ---- Deprecated static methods (kept for backward compatibility) ----

	@staticmethod
	def generate_keypair(security_level: int = 128, save: bool = False, output_path: str = "/sink", filename: str = "rory-phe") -> Tuple:
		"""@deprecated: Use instance method generate_keys() instead."""
		warnings.warn("Paillier.generate_keypair() is deprecated, use instance method generate_keys()", DeprecationWarning, stacklevel=2)
		public_key, private_key = paillier.generate_paillier_keypair(n_length=security_level)
		if save:
			Paillier.save_paillier_keys(public_key, private_key, output_path, filename)
		return public_key, private_key

	@staticmethod
	def generate_keypair_by_sl(security_level: int = 128, output_path: str = "/sink", filename: str = "rory-phe", save: bool = False) -> Tuple:
		"""@deprecated: Use instance method generate_keys() instead."""
		return Paillier.generate_keypair(security_level=security_level, save=save, output_path=output_path, filename=filename)

	@staticmethod
	def save_paillier_keys(public_key, private_key, output_path: str, filename: str):
		"""@deprecated: Use instance method save_keys() instead."""
		warnings.warn("Paillier.save_paillier_keys() is deprecated, use instance method save_keys()", DeprecationWarning, stacklevel=2)
		Paillier(public_key=public_key, private_key=private_key).save_keys(output_path, filename)

	@staticmethod
	def load_paillier_keys(path: str, filename: str):
		"""@deprecated: Use instance method load_keys() instead."""
		warnings.warn("Paillier.load_paillier_keys() is deprecated, use instance method load_keys()", DeprecationWarning, stacklevel=2)
		inst = Paillier()
		inst.load_keys(path, filename)
		return inst.public_key, inst.private_key

	@staticmethod
	def encryptMatrix(plaintext_matrix: npt.NDArray, public_key) -> CipherResult:
		"""@deprecated: Use instance method encrypt_matrix() instead."""
		warnings.warn("Paillier.encryptMatrix() is deprecated, use instance method encrypt_matrix()", DeprecationWarning, stacklevel=2)
		return CipherResult(data=np.array([[public_key.encrypt(x) for x in row] for row in plaintext_matrix]))

	@staticmethod
	def encryptVector(plaintext_vector: npt.NDArray, public_key) -> npt.NDArray:
		"""@deprecated: Use instance method encrypt_vector() instead."""
		warnings.warn("Paillier.encryptVector() is deprecated, use instance method encrypt_vector()", DeprecationWarning, stacklevel=2)
		return np.array([public_key.encrypt(v) for v in plaintext_vector])

	@staticmethod
	def encryptScalar(plaintext: float, public_key):
		"""@deprecated: Use instance method encrypt_scalar() instead."""
		warnings.warn("Paillier.encryptScalar() is deprecated, use instance method encrypt_scalar()", DeprecationWarning, stacklevel=2)
		return public_key.encrypt(plaintext)

	@staticmethod
	def decryptMatrix(ciphertext_matrix: npt.NDArray, private_key) -> npt.NDArray:
		"""@deprecated: Use instance method decrypt_matrix() instead."""
		warnings.warn("Paillier.decryptMatrix() is deprecated, use instance method decrypt_matrix()", DeprecationWarning, stacklevel=2)
		return np.array([[private_key.decrypt(x) for x in row] for row in ciphertext_matrix])

	@staticmethod
	def decryptVector(ciphertext_vector: npt.NDArray, private_key) -> npt.NDArray:
		"""@deprecated: Use instance method decrypt_vector() instead."""
		warnings.warn("Paillier.decryptVector() is deprecated, use instance method decrypt_vector()", DeprecationWarning, stacklevel=2)
		return np.array([private_key.decrypt(c) for c in ciphertext_vector])

	@staticmethod
	def decryptScalar(ciphertext, private_key) -> float:
		"""@deprecated: Use instance method decrypt_scalar() instead."""
		warnings.warn("Paillier.decryptScalar() is deprecated, use instance method decrypt_scalar()", DeprecationWarning, stacklevel=2)
		return private_key.decrypt(ciphertext)
