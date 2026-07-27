from abc import ABC, abstractmethod
import numpy.typing as npt
from rory.core.interfaces.cipher_result import CipherResult


class Cipher(ABC):
	"""Abstract base for all cryptographic schemes.

	Defines the minimal contract: key generation, encryption,
	decryption, and key persistence.
	"""

	@abstractmethod
	def generate_keys(self, *args, **kwargs):
		"""Generate and store scheme-specific keys internally."""
		...

	@abstractmethod
	def encrypt_scalar(self, plaintext: float) -> CipherResult:
		"""Encrypt a single plaintext value.

		Returns:
			CipherResult: Container with the encrypted data.
		"""
		...

	def encrypt_vector(self, plaintext_vector: npt.NDArray) -> CipherResult:
		"""Encrypt a vector of plaintext values.

		Default implementation iterates over encrypt_scalar.

		Returns:
			CipherResult: Container with the encrypted vector.
		"""
		results = [self.encrypt_scalar(v).data for v in plaintext_vector]
		return CipherResult(data=results)

	def encrypt_matrix(self, plaintext_matrix: npt.NDArray) -> CipherResult:
		"""Encrypt a matrix of plaintext values.

		Default implementation iterates over encrypt_vector.

		Returns:
			CipherResult: Container with the encrypted matrix.
		"""
		results = [self.encrypt_vector(row).data for row in plaintext_matrix]
		return CipherResult(data=results)

	@abstractmethod
	def decrypt_scalar(self, ciphertext) -> CipherResult:
		"""Decrypt a single ciphertext value.

		Returns:
			CipherResult: Container with the decrypted plaintext.
		"""
		...

	def decrypt_vector(self, ciphertext_vector) -> CipherResult:
		"""Decrypt a vector of ciphertext values.

		Default implementation iterates over decrypt_scalar.

		Returns:
			CipherResult: Container with the decrypted vector.
		"""
		results = [self.decrypt_scalar(c).data for c in ciphertext_vector]
		return CipherResult(data=results)

	def decrypt_matrix(self, ciphertext_matrix) -> CipherResult:
		"""Decrypt a matrix of ciphertext values.

		Default implementation iterates over decrypt_vector.

		Returns:
			CipherResult: Container with the decrypted matrix.
		"""
		results = [self.decrypt_vector(row).data for row in ciphertext_matrix]
		return CipherResult(data=results)

	def save_keys(self, path: str) -> None:
		"""Persist keys to disk.

		Raises:
			NotImplementedError: If the scheme does not support persistence.
		"""
		raise NotImplementedError

	def load_keys(self, path: str) -> None:
		"""Load keys from disk.

		Raises:
			NotImplementedError: If the scheme does not support persistence.
		"""
		raise NotImplementedError


class HomomorphicCipher(Cipher):
	"""Abstract base for homomorphic encryption schemes.

	Adds homomorphic addition, subtraction, and scalar multiplication.
	"""

	@abstractmethod
	def add(self, ciphertext_1, ciphertext_2):
		"""Homomorphic addition: E(a) + E(b) = E(a + b)."""
		...

	def subtract(self, ciphertext_1, ciphertext_2):
		"""Homomorphic subtraction: E(a) - E(b) = E(a - b).

		Derived from add and multiply_scalar(-1).
		"""
		neg = self.multiply_scalar(-1, ciphertext_2)
		return self.add(ciphertext_1, neg)

	@abstractmethod
	def multiply_scalar(self, scalar: float, ciphertext):
		"""Multiply ciphertext by a plaintext scalar: E(a) * b = E(a * b)."""
		...


class PartiallyHomomorphicCipher(HomomorphicCipher):
	"""Abstract base for Partially Homomorphic Encryption (PHE) schemes.

	Supports addition and scalar multiplication on ciphertexts, but NOT
	ciphertext-ciphertext multiplication.
	"""
	pass


class FullyHomomorphicCipher(HomomorphicCipher):
	"""Abstract base for Fully Homomorphic Encryption (FHE) schemes.

	Supports both addition and ciphertext-ciphertext multiplication.
	"""

	@abstractmethod
	def multiply(self, ciphertext_1, ciphertext_2):
		"""Homomorphic multiplication: E(a) * E(b) = E(a * b)."""
		...
