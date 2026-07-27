import warnings
import random
import pickle
import os
import numpy as np
from rory.core.interfaces.cipher_result import CipherResult
from rory.core.security.cryptosystem.abstract import FullyHomomorphicCipher
from typing import Tuple, List
import numpy.typing as npt
import secrets

"""
Description: 
	class to represent a symmetric encryption scheme
Attributes:
    _round: Flag to know if decryption needs to be rounded based on plaintext type
"""


class Liu(FullyHomomorphicCipher):
	"""
	Represents a symmetric encryption scheme based on Liu's cryptographic approach.

    This class implements Liu's cryptographic scheme, supporting key generation, encryption,
    decryption, and homomorphic operations (addition, subtraction, multiplication).
	Additionally, it offers configurable options for random number generation,
    allowing the use of either cryptographically secure or standard random generators, as well as
    support for reproducibility via seeding and optional use of NumPy's random module.

    Attributes:
        round (bool): Determines if decryption should be rounded.
        decimals (int): Number of decimal places for rounding.
        secure_random (bool): Specifies whether to use a cryptographically secure random generator.
        seed (int or None): Seed for the random generator to ensure reproducibility; if None, randomness is maintained.
        use_np_random (bool): If True, NumPy's random generator is used instead of Python's built-in random module.
        SECURITY_LEVELS (dict): Maps security levels to scheme parameters (m, nbits).
        sk_parameters (tuple): Scheme parameters (m, nbits) derived from the chosen security level.
        m (int): The 'm' parameter extracted from the security level parameters.
        sk (list): List for storing secret keys.
    """

	def __init__(self, _round: bool = False, decimals: int = 2, secure_random: bool = False, seed: int = None, use_np_random: bool = False, security_level: int = 128):
		self.round = _round
		self.seed = seed
		self.decimals = decimals
		self.secure_random = secure_random
		self.use_np_random = use_np_random
		self._py_rng = random.Random(seed) if seed is not None else random.Random()
		self._np_rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
		self.SECURITY_LEVELS = {128: (3, 15), 192: (4, 16), 256: (6, 15)}  # level:(m,nbits)
		self.sk_parameters = self.SECURITY_LEVELS.get(security_level)
		self.m = self.sk_parameters[0]
		self.sk = []

	# ---- Standardized interface ----

	def generate_keys(self, security_level: int = 128):
		"""Generate and store secret key internally.

		Args:
			security_level: 128, 192, or 256.

		Returns:
			List[Tuple[float, float, float]]: The generated secret key.
		"""
		self.sk_parameters = self.SECURITY_LEVELS.get(security_level)
		self.m = self.sk_parameters[0]
		self.sk = []
		self.__generate_secret_key(
			m     = self.m,
			nbits = self.sk_parameters[1],
		)
		return self.sk

	def reseed(self, seed: int = None):
		"""Reset encryption randomness without changing the secret key.

		This is intended for worker-local cipher copies.  When ``seed`` is
		omitted, fresh operating-system entropy is used so separate workers do
		not repeat the same encryption random stream.

		Args:
			seed: Optional deterministic seed, primarily for tests.

		Returns:
			Liu: This cipher instance.
		"""
		if seed is None:
			seed = secrets.randbits(128)
		self.seed = seed
		self._py_rng = random.Random(seed)
		self._np_rng = np.random.default_rng(seed)
		return self

	def encrypt_scalar(self, plaintext: float):
		"""Encrypt a single plaintext value using the internally stored secret key.

		Args:
			plaintext: Plaintext value to encrypt.

		Returns:
			CipherResult: Container with the encrypted scalar as an ndarray.
		"""
		result = self._encryptScalar(plaintext=plaintext, secret_key=self.sk)
		return CipherResult(data=result)

	def encrypt_vector(self, plaintext_vector: npt.NDArray):
		"""Encrypt a vector of plaintext values using the internally stored secret key.

		Args:
			plaintext_vector: Vector of plaintext values.

		Returns:
			CipherResult: Container with the encrypted vector.
		"""
		result = self._encryptVector(plaintext_vector=plaintext_vector, secret_key=self.sk)
		return CipherResult(data=result)

	def encrypt_matrix(self, plaintext_matrix: npt.NDArray):
		"""Encrypt a matrix of plaintext values using the internally stored secret key.

		Args:
			plaintext_matrix: Matrix of plaintext values.

		Returns:
			CipherResult: Container with the encrypted matrix.
		"""
		result = self._encryptMatrix(plaintext_matrix=plaintext_matrix, secret_key=self.sk)
		return CipherResult(data=result)

	def decrypt_scalar(self, ciphertext):
		"""Decrypt a single ciphertext value using the internally stored secret key.

		Args:
			ciphertext: Ciphertext vector to decrypt.

		Returns:
			CipherResult: Container with the decrypted plaintext.
		"""
		result = self._decryptScalar(ciphertext=ciphertext, secret_key=self.sk)
		return CipherResult(data=result)

	def decrypt_vector(self, ciphertext_vector):
		"""Decrypt a vector of ciphertext values using the internally stored secret key.

		Args:
			ciphertext_vector: Vector of ciphertext values.

		Returns:
			CipherResult: Container with the decrypted vector.
		"""
		result = self._decryptVector(ciphertext_vector=ciphertext_vector, secret_key=self.sk)
		return CipherResult(data=result)

	def decrypt_matrix(self, ciphertext_matrix):
		"""Decrypt a matrix of ciphertext values using the internally stored secret key.

		Args:
			ciphertext_matrix: Matrix of ciphertext values.

		Returns:
			CipherResult: Container with the decrypted matrix.
		"""
		result = self._decryptMatrix(ciphertext_matrix=ciphertext_matrix, secret_key=self.sk)
		return CipherResult(data=result)

	# Homomorphic operations (instance methods, abstract interface)

	@classmethod
	def add(cls, ciphertext_1, ciphertext_2):
		"""Homomorphic addition: E(a) + E(b) = E(a + b).

		Args:
			ciphertext_1: First encrypted operand.
			ciphertext_2: Second encrypted operand.

		Returns:
			ndarray: Encrypted sum.
		"""
		return np.array(ciphertext_1) + np.array(ciphertext_2)

	@classmethod
	def multiply(cls, ciphertext_1, ciphertext_2):
		"""Homomorphic multiplication: E(a) * E(b) = E(a * b).

		Args:
			ciphertext_1: First encrypted operand.
			ciphertext_2: Second encrypted operand.

		Returns:
			ndarray: Encrypted outer product, flattened.
		"""
		return np.outer(np.array(ciphertext_1), np.array(ciphertext_2)).ravel()

	@classmethod
	def multiply_scalar(cls, scalar: float, ciphertext):
		"""Multiply ciphertext by a plaintext scalar: E(a) * b = E(a * b).

		Args:
			scalar: Plaintext multiplier.
			ciphertext: Encrypted operand.

		Returns:
			ndarray: Encrypted product.
		"""
		return np.array(ciphertext) * scalar

	@classmethod
	def subtract(cls, ciphertext_1, ciphertext_2):
		"""Homomorphic subtraction: E(a) - E(b) = E(a - b).

		Args:
			ciphertext_1: First encrypted operand.
			ciphertext_2: Second encrypted operand.

		Returns:
			ndarray: Encrypted difference.
		"""
		neg = cls.multiply_scalar(-1, ciphertext_2)
		return cls.add(ciphertext_1, neg)

	def decrypt_multiply(self, ciphertext, secret_key: List[Tuple[float, float, float]] = None) -> float:
		"""Decrypt the result of a homomorphic multiplication.

		The method splits the input ciphertext into groups of size m,
		decrypts each group individually, and then decrypts the aggregated
		results to obtain the final plaintext value.

		Args:
			ciphertext: The ciphertext representing the product of two ciphertexts.
			secret_key: The secret key. Defaults to internally stored key.

		Returns:
			float: The final decrypted plaintext value.
		"""
		if secret_key is None:
			secret_key = self.sk
		m = self.sk_parameters[0]
		v1, v = [], []
		for i in range(len(ciphertext)):
			v1.append(ciphertext[i])
			if ((i % m) == (m - 1)):
				v.append(self._decryptScalar(ciphertext=v1, secret_key=secret_key))
				v1 = []
		E3 = self._decryptScalar(ciphertext=v, secret_key=secret_key)
		return E3

	def save_keys(self, path: str):
		"""Save the secret key to disk.

		Args:
			path: Directory path to save keys.
		"""
		os.makedirs(path, exist_ok=True)
		with open(f"{path}/liu_sk.pkl", "wb") as f:
			pickle.dump({"sk": self.sk, "m": self.m, "sk_parameters": self.sk_parameters}, f)

	def load_keys(self, path: str):
		"""Load the secret key from disk.

		Args:
			path: Directory path where keys are stored.
		"""
		with open(f"{path}/liu_sk.pkl", "rb") as f:
			data = pickle.load(f)
		self.sk = data["sk"]
		self.m = data["m"]
		self.sk_parameters = data["sk_parameters"]

	# ---- Legacy methods (deprecated, kept for backward compatibility) ----

	def generateSecretRandomBit(self, nbits: int) -> int:
		"""
		Generates a random integer with the specified number of bits.

		This function uses either a secure or non-secure random generator depending on the
		value of the `secure_random` attribute. If `secure_random` is True, it employs 
		`secrets.randbits(nbits)` for cryptographic security; otherwise, it uses 
		`random.getrandbits(nbits)`.

		Args:
			nbits (int): The number of bits for the random integer to be generated.

		Returns:
			int: A random integer consisting of the specified number of bits.
		"""
		if self.secure_random:
			r = secrets.randbits(nbits)
		else:
			r = random.getrandbits(nbits)
		return r

	def generate_random_np(self, low: int, high: int, size: int) -> npt.NDArray:
		"""@deprecated: Internal helper."""
		return self._np_rng.uniform(low=low, high=high, size=size)

	def generateRandom(self) -> float:
		"""@deprecated: Internal helper."""
		return self._py_rng.uniform(0, 1)

	def __generate_secret_key(self, m: int = 3, nbits: int = 15) -> List[Tuple[float, float, float]]:
		"""@deprecated: Internal helper for key generation."""
		for i in range(m):
			tp = self.generateSecretRandomBit(nbits=nbits), self.generateSecretRandomBit(nbits=nbits), self.generateSecretRandomBit(nbits=nbits)
			self.sk.append(tp)
		return self.sk

	def generate_secret_key(self):
		"""@deprecated: Use generate_keys() instead."""
		warnings.warn("generate_secret_key() is deprecated, use generate_keys()", DeprecationWarning, stacklevel=2)
		self.sk = []
		self.m = self.sk_parameters[0]
		self.__generate_secret_key(
			m=self.m,
			nbits=self.sk_parameters[1],
		)
		return self.sk

	# Deprecated encrypt/decrypt methods with explicit key parameter

	def encryptMatrix(self, plaintext_matrix: npt.NDArray, secret_key: List[Tuple[float, float, float]]) -> CipherResult:
		"""@deprecated: Use encrypt_matrix() instead."""
		warnings.warn("encryptMatrix() is deprecated, use encrypt_matrix()", DeprecationWarning, stacklevel=2)
		return CipherResult(data=self._encryptMatrix(plaintext_matrix=plaintext_matrix, secret_key=secret_key))

	def encryptVector(self, plaintext_vector: npt.NDArray, secret_key: List[Tuple[float, float, float]]) -> CipherResult:
		"""@deprecated: Use encrypt_vector() instead."""
		warnings.warn("encryptVector() is deprecated, use encrypt_vector()", DeprecationWarning, stacklevel=2)
		return CipherResult(data=self._encryptVector(plaintext_vector=plaintext_vector, secret_key=secret_key))

	def encryptScalar(self, plaintext: float, secret_key: List[Tuple[float, float, float]]) -> CipherResult:
		"""@deprecated: Use encrypt_scalar() instead."""
		warnings.warn("encryptScalar() is deprecated, use encrypt_scalar()", DeprecationWarning, stacklevel=2)
		return CipherResult(data=self._encryptScalar(plaintext=plaintext, secret_key=secret_key))

	def decryptMatrix(self, ciphertext_matrix: npt.NDArray, secret_key: List[Tuple[float, float, float]]) -> CipherResult:
		"""@deprecated: Use decrypt_matrix() instead."""
		warnings.warn("decryptMatrix() is deprecated, use decrypt_matrix()", DeprecationWarning, stacklevel=2)
		return CipherResult(data=self._decryptMatrix(ciphertext_matrix=ciphertext_matrix, secret_key=secret_key))

	def decryptVector(self, ciphertext_vector: npt.NDArray, secret_key: List[Tuple[float, float, float]]) -> CipherResult:
		"""@deprecated: Use decrypt_vector() instead."""
		warnings.warn("decryptVector() is deprecated, use decrypt_vector()", DeprecationWarning, stacklevel=2)
		return CipherResult(data=self._decryptVector(ciphertext_vector=ciphertext_vector, secret_key=secret_key))

	def decryptScalar(self, ciphertext: List[float], secret_key: List[Tuple[float, float, float]]) -> CipherResult:
		"""@deprecated: Use decrypt_scalar() instead."""
		warnings.warn("decryptScalar() is deprecated, use decrypt_scalar()", DeprecationWarning, stacklevel=2)
		return CipherResult(data=self._decryptScalar(ciphertext=ciphertext, secret_key=secret_key))

	def decryptMultiply(self, ciphertext, secret_key: List[Tuple[float, float, float]]) -> CipherResult:
		"""@deprecated: Use decrypt_multiply() instead."""
		warnings.warn("decryptMultiply() is deprecated, use decrypt_multiply()", DeprecationWarning, stacklevel=2)
		return CipherResult(data=self.decrypt_multiply(ciphertext=ciphertext, secret_key=secret_key))

	# ---- Internal implementation methods (preserve original logic) ----

	def _encryptMatrix(self, plaintext_matrix: npt.NDArray, secret_key: List[Tuple[float, float, float]]) -> npt.NDArray:
		"""Encrypt a matrix using the given secret key.

		Args:
			plaintext_matrix: Matrix of plaintext values.
			secret_key: Secret key as list of (k, s, t) tuples.

		Returns:
			npt.NDArray: Encrypted matrix.
		"""
		return np.array([self._encryptVector(plaintext_vector=v, secret_key=secret_key) for v in plaintext_matrix])

	def _encryptVector(self, plaintext_vector: npt.NDArray, secret_key: List[Tuple[float, float, float]]) -> npt.NDArray:
		"""Encrypt a vector using the given secret key.

		Args:
			plaintext_vector: Vector of plaintext values.
			secret_key: Secret key as list of (k, s, t) tuples.

		Returns:
			npt.NDArray: Encrypted vector.
		"""
		return np.array([self._encryptScalar(plaintext=v, secret_key=secret_key) for v in plaintext_vector])

	def _encryptScalar(self, plaintext: float, secret_key: List[Tuple[float, float, float]]) -> npt.NDArray:
		"""Encrypt a single plaintext using Liu's symmetric scheme.

		Generates m random values and uses the secret key components
		(k, s, t) to produce a ciphertext vector of size m.

		Args:
			plaintext: Plaintext value to encrypt.
			secret_key: Secret key as list of (k, s, t) tuples.

		Returns:
			npt.NDArray: Ciphertext vector of length m.
		"""
		m = self.m
		plaintext = np.round(plaintext, decimals=self.decimals) if self.round else plaintext
		E = np.empty(m)
		R = self.generate_random_np(low=0, high=1, size=m) if self.use_np_random else [self.generateRandom() for _ in range(m)]

		E[0] = self.__eEncrypt(
			ki     = secret_key[0][0],
			ti     = secret_key[0][2],
			v      = plaintext,
			si     = secret_key[0][1],
			rm     = R[m - 1],
			rrdiff = (R[0] - R[m - 2]),
		)
		for i in range(1, m - 1):
			E[i] = self.__eEncrypt(
				ki     = secret_key[i][0],
				ti     = secret_key[i][2],
				v      = plaintext,
				si     = secret_key[i][1],
				rm     = R[m - 1],
				rrdiff = (R[i] - R[i - 1]),
			)
		E[m - 1] = (secret_key[m - 1][0] + secret_key[m - 1][1] + secret_key[m - 1][2]) * R[m - 1]
		return E

	def __eEncrypt(self, ki: float, ti: float, v: float, si: float, rm: float, rrdiff: float) -> float:
		"""Core encryption formula for Liu's symmetric scheme.

		Computes: ki * ti * v + si * rm + ki * rrdiff.

		Args:
			ki: First key component.
			ti: Third key component (t).
			v: Plaintext value.
			si: Second key component (s).
			rm: Random value for masking.
			rrdiff: Random value difference for obfuscation.

		Returns:
			float: Single ciphertext component.
		"""
		return ki * ti * v + si * rm + ki * rrdiff

	def _decryptMatrix(self, ciphertext_matrix: npt.NDArray, secret_key: List[Tuple[float, float, float]]) -> npt.NDArray:
		"""Decrypt a matrix using the given secret key.

		Args:
			ciphertext_matrix: Matrix of ciphertext values.
			secret_key: Secret key as list of (k, s, t) tuples.

		Returns:
			npt.NDArray: Decrypted matrix.
		"""
		return np.array([self._decryptVector(ciphertext_vector=c, secret_key=secret_key) for c in ciphertext_matrix])

	def _decryptVector(self, ciphertext_vector: npt.NDArray, secret_key: List[Tuple[float, float, float]]) -> npt.NDArray:
		"""Decrypt a vector using the given secret key.

		Args:
			ciphertext_vector: Vector of ciphertext values.
			secret_key: Secret key as list of (k, s, t) tuples.

		Returns:
			npt.NDArray: Decrypted vector.
		"""
		return np.array([self._decryptScalar(ciphertext=c, secret_key=secret_key) for c in ciphertext_vector])

	def _decryptScalar(self, ciphertext: List[float], secret_key: List[Tuple[float, float, float]]) -> float:
		"""Decrypt a single ciphertext using Liu's symmetric scheme.

		Recovers the plaintext from the ciphertext vector using the
		secret key components (k, s, t) and optional rounding.

		Args:
			ciphertext: Ciphertext vector of length m.
			secret_key: Secret key as list of (k, s, t) tuples.

		Returns:
			float: Decrypted plaintext value.
		"""
		t = sum(secret_key[i][2] for i in range(self.m - 1))
		s = ciphertext[self.m - 1] / (secret_key[self.m - 1][0] + secret_key[self.m - 1][1] + secret_key[self.m - 1][2])
		e = sum((ciphertext[i] - s * secret_key[i][1]) / secret_key[i][0] for i in range(self.m - 1))
		v = e / t
		return np.around(v, decimals=self.decimals) if self.round else float(v)

	# ---- Deprecated static homomorphic methods ----

	@staticmethod
	def multiply_c(scalar: float, ciphertext):
		"""@deprecated: Use instance method multiply_scalar() instead."""
		return Liu.multiply_scalar(scalar, ciphertext)
