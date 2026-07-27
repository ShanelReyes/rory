import warnings
import numpy as np
from Pyfhel import Pyfhel, PyCtxt, PyPtxt
from rory.core.utils.utils import Utils
from rory.core.interfaces.cipher_result import CipherResult
from rory.core.security.cryptosystem.abstract import FullyHomomorphicCipher
from typing import Tuple, List, Union, Optional
import numpy.typing as npt
from enum import Enum


class CkksModes(Enum):
	DEFAULT = "default"
	ML = "ml"


class Ckks(FullyHomomorphicCipher):
	"""
    Represents a CKKS-based homomorphic encryption scheme.
    This class serves as a wrapper around a Pyfhel object configured with CKKS settings. It maintains whether decrypted values should be rounded, and to
    what number of decimals. It also stores static identifiers for context, public key, secret key, rotation key, and relinearization key.

    Attributes
    ----------
    _ctx_id : str
        The identifier for the context file (used when saving/loading).
    _public_key_id : str
        The identifier for the public key file (used when saving/loading).
    _secret_key_id : str
        The identifier for the secret key file (used when saving/loading).
    _rorate_key_id : str
        The identifier for the rotation key file (used when saving/loading).
    _relin_key_id : str
        The identifier for the relinearization key file (used when saving/loading).
    round : bool
        A flag indicating whether decrypted results should be rounded.
    decimals : int
        The number of decimal places to retain if rounding is enabled.
    he_object : Pyfhel
        A Pyfhel object configured with CKKS for homomorphic encryption operations.
    """
	_ctx_id         = "ctx"
	_public_key_id  = "pubkey"
	_secret_key_id  = "secretkey"
	_rotate_key_id  = "rotatekey"
	_relin_key_id   = "relinkey"
	SECURITY_LEVELS = {"default": {128: {"n": 2**13, "scale": 2**40, "qi_sizes": [60, 40, 40, 60]},
							192: {"n": 2**14, "scale": 2**40, "qi_sizes": [60, 40, 40, 40, 60]},
							256: {"n": 2**15, "scale": 2**60, "qi_sizes": [60, 50, 50, 50, 60]},
						},
						"ml": {
							128: {"n": 2**14, "scale": 2**40, "qi_sizes": [60] + [40]*7 + [60]},
							192: {"n": 2**15, "scale": 2**40, "qi_sizes": [60] + [40]*7 + [60]},
							256: {"n": 2**15, "scale": 2**40, "qi_sizes": [60] + [40]*7 + [60]}
						}
					}

	def __init__(self, he_object: Pyfhel=None, _round: bool = False, decimals: int = 2, security_level: int = 128, n_features: int = 8192):
		"""
        Initializes the Ckks class with a Pyfhel object and optional rounding parameters.

        Parameters
        ----------
        he_object : Pyfhel
            The underlying Pyfhel instance configured for CKKS encryption.
        _round : bool, optional
            Indicates whether decrypted results should be rounded.
        decimals : int, optional
            The number of decimal places to keep if rounding is enabled.
        n_features : int, optional
            The number of features/slots to use for scalar encoding and dot products.
        """
		self.round = _round
		self.decimals = decimals
		self.he_object: Pyfhel = he_object
		self.n_features = n_features
		if he_object is not None:
			self.scale = int(he_object.scale)

	# ---- Standardized interface ----

	def generate_keys(self, scheme: str = "CKKS", mode: CkksModes = CkksModes.DEFAULT, security_level: int = 128, _round: bool = False, decimals: int = 2, output_path: str = "/sink", save: bool = False, enable_relinearize: bool = True, enable_rotate: bool = True):
		"""Generate CKKS keys and store them internally.

		Args:
			scheme: Homomorphic encryption scheme. Defaults to "CKKS".
			mode: CkksModes.DEFAULT or CkksModes.ML.
			security_level: 128, 192, or 256.
			_round: Whether to enable rounding on decryption.
			decimals: Number of decimal places to preserve.
			output_path: Directory for saving keys.
			save: If True, persists keys to disk.
			enable_relinearize: Generate relinearization keys.
			enable_rotate: Generate rotation keys.

		Returns:
			Ckks: Self, for method chaining.
		"""
		self.round      = _round
		self.decimals   = decimals
		sk_parameters   = Ckks.SECURITY_LEVELS.get(mode.value).get(security_level)
		n               = sk_parameters.get("n", 2**13)
		self.scale      = int(sk_parameters.get("scale", 2**40))
		qi_sizes        = sk_parameters.get("qi_sizes", [60, 50, 50, 50, 60])
		self.n_features = n // 2
		HE              = Pyfhel()
		HE.contextGen(
			scheme   = scheme,
			n        = n,
			scale    = self.scale,
			qi_sizes = qi_sizes
		)
		HE.keyGen()
		if enable_relinearize:
			HE.relinKeyGen()
		if enable_rotate:
			HE.rotateKeyGen()
		if save:
			HE.save_context("{}/{}".format(output_path, Ckks._ctx_id))
			HE.save_public_key("{}/{}".format(output_path, Ckks._public_key_id))
			HE.save_secret_key("{}/{}".format(output_path, Ckks._secret_key_id))
			if enable_relinearize:
				HE.save_relin_key(f"{output_path}/{Ckks._relin_key_id}")
			if enable_rotate:
				HE.save_rotate_key(f"{output_path}/{Ckks._rotate_key_id}")
		self.he_object = HE
		return self

	def encrypt_scalar(self, plaintext: float):
		"""Encrypt a single plaintext value.

		Args:
			plaintext: Plaintext value to encrypt.

		Returns:
			CipherResult: Container with the encrypted scalar.
		"""
		ptxt   = self.he_object.encodeFrac(np.array([plaintext], dtype=np.float64), scale=self.scale)
		result = self.he_object.encrypt(ptxt)
		return CipherResult(data=result)

	def encrypt_vector(self, plaintext_vector: npt.NDArray):
		"""Encrypt a plaintext vector.

		Args:
			plaintext_vector: Vector of plaintext values.

		Returns:
			CipherResult: Container with the encrypted vector.
		"""
		result = self._encryptVector(plaintext_vector=plaintext_vector)
		return CipherResult(data=result)

	def encrypt_matrix(self, plaintext_matrix: npt.NDArray):
		"""Encrypt a matrix of plaintext values.

		Args:
			plaintext_matrix: Matrix of plaintext values.

		Returns:
			CipherResult: Container with the encrypted matrix.
		"""
		result = self._encryptMatrix(plaintext_matrix=plaintext_matrix)
		return CipherResult(data=result)

	def decrypt_scalar(self, ciphertext):
		"""Decrypt a single ciphertext value.

		Args:
			ciphertext: CKKS ciphertext to decrypt.

		Returns:
			CipherResult: Container with the decrypted plaintext.
		"""
		result       = self.he_object.decryptFrac(ciphertext)[0]
		round_result = np.round(result, decimals=self.decimals) if self.round else result
		return CipherResult(data=round_result)

	def decrypt_vector(self, ciphertext_vector):
		"""Decrypt a vector of ciphertext values.

		Args:
			ciphertext_vector: Vector of CKKS ciphertexts.

		Returns:
			CipherResult: Container with the decrypted vector.
		"""
		result = self._decryptVector(ciphertext_vector=ciphertext_vector)
		return CipherResult(data=result)

	def decrypt_matrix(self, ciphertext_matrix, adjust: bool = True, shape: Tuple[int, int] = (1, 1)):
		"""Decrypt a matrix of ciphertext values.

		Args:
			ciphertext_matrix: Matrix of CKKS ciphertexts.
			adjust: Whether to reshape decrypted rows. Defaults to True.
			shape: Target shape (rows, cols) for adjustment. Defaults to (1, 1).

		Returns:
			CipherResult: Container with the decrypted matrix.
		"""
		result = self._decryptMatrix(ciphertext_matrix=ciphertext_matrix, adjust=adjust, shape=shape)
		return CipherResult(data=result)

	# Homomorphic operations

	def add(self, ciphertext_1, ciphertext_2):
		"""Homomorphic addition: E(a) + E(b) = E(a + b).

		Aligns modulus levels before addition.

		Args:
			ciphertext_1: First CKKS ciphertext operand.
			ciphertext_2: Second CKKS ciphertext operand.

		Returns:
			PyCtxt: Encrypted sum.

		Raises:
			RuntimeError: If the homomorphic addition fails.
		"""
		lvl_a = ciphertext_1.mod_level
		lvl_b = ciphertext_2.mod_level

		if lvl_a > lvl_b:
			for _ in range(lvl_a - lvl_b):
				self.he_object.mod_switch_to_next(ciphertext_2)
		elif lvl_b > lvl_a:
			for _ in range(lvl_b - lvl_a):
				self.he_object.mod_switch_to_next(ciphertext_1)

		ciphertext_2.scale = ciphertext_1.scale

		try:
			return self.he_object.add(ciphertext_1, ciphertext_2, in_new_ctxt=True)
		except Exception as e:
			raise RuntimeError(f"Error in CKKS add: {e}")

	def subtract(self, ciphertext_1: PyCtxt, ciphertext_2: PyCtxt):
		"""Homomorphic subtraction: E(a) - E(b) = E(a - b).

		Aligns modulus levels before subtraction.

		Args:
			ciphertext_1: First CKKS ciphertext operand.
			ciphertext_2: Second CKKS ciphertext operand.

		Returns:
			PyCtxt: Encrypted difference.

		Raises:
			RuntimeError: If the homomorphic subtraction fails.
		"""
		lvl_a = ciphertext_1.mod_level
		lvl_b = ciphertext_2.mod_level

		if lvl_a < lvl_b:
			for _ in range(lvl_b - lvl_a):
				self.he_object.mod_switch_to_next(ciphertext_1)
		elif lvl_b < lvl_a:
			for _ in range(lvl_a - lvl_b):
				self.he_object.mod_switch_to_next(ciphertext_2)

		ciphertext_2.scale = ciphertext_1.scale

		try:
			return self.he_object.sub(ciphertext_1, ciphertext_2, in_new_ctxt=True)
		except Exception as e:
			raise RuntimeError(f"Error in CKKS subtract: {e}")

	def multiply(self, ciphertext_1: PyCtxt, ciphertext_2: PyCtxt):
		"""Homomorphic multiplication: E(a) * E(b) = E(a * b).

		Aligns levels, applies relinearization and rescaling.

		Args:
			ciphertext_1: First CKKS ciphertext operand.
			ciphertext_2: Second CKKS ciphertext operand.

		Returns:
			PyCtxt: Encrypted product.

		Raises:
			RuntimeError: If the homomorphic multiplication fails.
		"""
		lvl_a = ciphertext_1.mod_level
		lvl_b = ciphertext_2.mod_level

		if lvl_a > lvl_b:
			for _ in range(lvl_a - lvl_b):
				self.he_object.mod_switch_to_next(ciphertext_2)
		elif lvl_b > lvl_a:
			for _ in range(lvl_b - lvl_a):
				self.he_object.mod_switch_to_next(ciphertext_1)

		try:
			res = self.he_object.multiply(ciphertext_1, ciphertext_2, in_new_ctxt=True)
			self._relinearize_if_possible(res)
			self._try_rescale_next(res)
			self._normalize_scale(res, scale=self.scale)
			return res
		except Exception as e:
			raise RuntimeError(f"Error in CKKS multiply: {e}")

	def multiply_scalar(self, scalar: float, ciphertext):
		"""Multiply ciphertext by a plaintext scalar: E(a) * b = E(a * b).

		Encodes the scalar as a plaintext and multiplies.

		Args:
			scalar: Plaintext multiplier.
			ciphertext: CKKS ciphertext operand.

		Returns:
			PyCtxt: Encrypted product.

		Raises:
			RuntimeError: If the scalar multiplication fails.
		"""
		exact_scale = ciphertext.scale
		pt_scalar = self._ptxt_from_scalar(val=scalar, n_features=self.n_features, scale=exact_scale)
		try:
			_, pt_scalar_aligned = self._align(a=ciphertext, b=pt_scalar, only_mod=True)
			res = self.he_object.multiply_plain(ciphertext, pt_scalar_aligned, in_new_ctxt=True)
			self._try_rescale_next(res)
			return res
		except Exception as e:
			raise RuntimeError(f"CKKS multiply_scalar error: {e}")

	def dot_product(self, ciphertext_1, ciphertext_2):
		"""Compute encrypted dot product between two ciphertext vectors.

		Executes element-wise multiplication followed by log-step rotations
		and additions to accumulate the sum across all slots.

		Args:
			ciphertext_1: Input feature vector ciphertext.
			ciphertext_2: Weight vector ciphertext.

		Returns:
			PyCtxt: Ciphertext with dot product total replicated across all slots.
		"""
		elementwise_product = self.multiply(ciphertext_1, ciphertext_2)
		accumulator = elementwise_product.copy()
		del elementwise_product

		step = 1
		while step < self.n_features:
			try:
				rotated = self.he_object.rotate(accumulator, step, in_new_ctxt=True)
				old_accumulator = accumulator
				accumulator = self.add(accumulator, rotated)
				del rotated, old_accumulator
			except Exception as e:
				raise RuntimeError(f"Error in dot_product rotation: {e}")
			step <<= 1

		return accumulator

	def add_plain_scalar(self, ciphertext, scalar: float):
		"""Add a plaintext scalar to a ciphertext: E(a) + b = E(a + b).

		Encodes the scalar as a plaintext vector matching the ciphertext's
		features and performs homomorphic addition.

		Args:
			ciphertext: CKKS ciphertext operand.
			scalar: Plaintext value to add.

		Returns:
			PyCtxt: Encrypted sum.
		"""
		exact_scale = ciphertext.scale
		pt_scalar = self._ptxt_from_scalar(val=scalar, n_features=self.n_features, scale=exact_scale)
		_, pt_scalar_aligned = self._align(a=ciphertext, b=pt_scalar, only_mod=True)
		return self.he_object.add_plain(ciphertext, pt_scalar_aligned, in_new_ctxt=True)

	def normalize_scale(self, ct, scale: int, ratio: float = 1.25):
		"""Iteratively rescale ciphertext until scale stabilizes.

		Args:
			ct: CKKS ciphertext to normalize.
			scale: Target scale value.
			ratio: Scale tolerance ratio. Defaults to 1.25.

		Returns:
			PyCtxt: Normalized ciphertext.
		"""
		return self._normalize_scale(ct, scale=scale, ratio=ratio)

	def save_keys(self, path: str):
		"""Save context and all keys to disk.

		Args:
			path: Directory path to save keys.
		"""
		import os
		os.makedirs(path, exist_ok=True)
		self.he_object.save_context("{}/{}".format(path, Ckks._ctx_id))
		self.he_object.save_public_key("{}/{}".format(path, Ckks._public_key_id))
		try:
			self.he_object.save_secret_key("{}/{}".format(path, Ckks._secret_key_id))
		except Exception:
			pass
		try:
			self.he_object.save_relin_key(f"{path}/{Ckks._relin_key_id}")
		except Exception:
			pass
		try:
			self.he_object.save_rotate_key(f"{path}/{Ckks._rotate_key_id}")
		except Exception:
			pass

	def load_keys(self, path: str):
		"""Load context and keys from disk.

		Args:
			path: Directory path where keys are stored.
		"""
		HE = self.he_object
		HE.load_context("{}/{}".format(path, Ckks._ctx_id))
		HE.load_public_key("{}/{}".format(path, Ckks._public_key_id))
		try:
			HE.load_secret_key("{}/{}".format(path, Ckks._secret_key_id))
		except Exception:
			pass
		try:
			HE.load_relin_key(f"{path}/{Ckks._relin_key_id}")
		except Exception:
			pass
		try:
			HE.load_rotate_key(f"{path}/{Ckks._rotate_key_id}")
		except Exception:
			pass

	# ---- Internal helpers (moved from Utils) ----

	@staticmethod
	def _get_scale(ct) -> Optional[float]:
		"""Safely extract scale from a ciphertext.

		Args:
			ct: CKKS ciphertext or plaintext object.

		Returns:
			Optional[float]: Scale value, or None if extraction fails.
		"""
		try:
			return float(ct.scale)
		except Exception:
			try:
				return float(ct.get_scale())
			except Exception:
				return None

	def _ptxt_from_scalar(self, val: float, n_features: int, scale: int):
		"""Encode a scalar into a plaintext repeated n_features times.

		Args:
			val: Scalar value to encode.
			n_features: Number of slots in the plaintext.
			scale: CKKS scale factor.

		Returns:
			PyPtxt: Encoded plaintext with the scalar replicated.
		"""
		arr = np.full(n_features, float(val), dtype=np.float64)
		return self.he_object.encodeFrac(arr, scale=scale)

	def _relinearize_if_possible(self, ct):
		"""Apply relinearization if keys are available.

		Args:
			ct: CKKS ciphertext to relinearize.

		Returns:
			PyCtxt: Relinearized (or original) ciphertext.
		"""
		try:
			self.he_object.relinearize(ct)
		except Exception:
			pass
		return ct

	def _try_rescale_next(self, ct):
		"""Attempt to rescale ciphertext to next level.

		Args:
			ct: CKKS ciphertext to rescale.

		Returns:
			PyCtxt: Rescaled (or original) ciphertext.
		"""
		try:
			self.he_object.rescale_to_next(ct)
		except Exception:
			pass
		return ct

	def _align(self, a=None, b=None, only_mod: bool = False):
		"""Align modulus chain levels and scales of two elements.

		Args:
			a: First CKKS ciphertext or plaintext.
			b: Second CKKS ciphertext or plaintext.
			only_mod: If True, only align modulus levels. Defaults to False.

		Returns:
			Tuple: Aligned copies of (a, b).
		"""
		a_al, b_al = self.he_object.align_mod_n_scale(
			a, b, copy_this=True, copy_other=True, only_mod=only_mod
		)
		return a_al, b_al

	def _normalize_scale(self, ct, scale: int, ratio: float = 1.25):
		"""Iteratively rescale ciphertext until scale stabilizes.

		Args:
			ct: CKKS ciphertext to normalize.
			scale: Target scale value.
			ratio: Scale tolerance ratio. Defaults to 1.25.

		Returns:
			PyCtxt: Normalized ciphertext.
		"""
		current_scale = Ckks._get_scale(ct)
		if current_scale is None:
			return ct

		changed = True
		while current_scale > ratio * scale and changed:
			previous_scale = current_scale
			changed = False
			try:
				self.he_object.rescale_to_next(ct)
				current_scale = Ckks._get_scale(ct)
				if current_scale is not None and current_scale < previous_scale:
					changed = True
			except Exception:
				break
		return ct

	# ---- Factory methods ----

	@staticmethod
	def _create_he_context(scheme: str = "CKKS", mode: CkksModes = CkksModes.DEFAULT, security_level: int = 128, enable_relinearize: bool = False, enable_rotate: bool = False):
		"""Create a configured Pyfhel context with generated keys.

		Args:
			scheme: HE scheme name. Defaults to "CKKS".
			mode: CKKS configuration mode. Defaults to DEFAULT.
			security_level: Security level (128, 192, 256). Defaults to 128.
			enable_relinearize: Generate relinearization keys. Defaults to False.
			enable_rotate: Generate rotation keys. Defaults to False.

		Returns:
			Tuple[Pyfhel, int, int]: (HE context, n_features, scale).
		"""
		sk_parameters = Ckks.SECURITY_LEVELS.get(mode.value).get(security_level)
		n        = sk_parameters.get("n", 2**13)
		scale    = sk_parameters.get("scale", 2**40)
		qi_sizes = sk_parameters.get("qi_sizes", [60, 50, 50, 50, 60])
		n_feat   = n // 2
		HE = Pyfhel()
		HE.contextGen(scheme=scheme, n=n, scale=scale, qi_sizes=qi_sizes)
		HE.keyGen()
		if enable_relinearize:
			HE.relinKeyGen()
		if enable_rotate:
			HE.rotateKeyGen()
		return HE, n_feat, scale

	@staticmethod
	def create_server(scheme: str = "CKKS", mode: CkksModes = CkksModes.DEFAULT, security_level: int = 128, _round: bool = False, decimals: int = 2, output_path: str = "/sink", save: bool = False) -> 'Ckks':
		"""Create a CKKS context for server usage (no secret key).

		Generates a Pyfhel context for performing homomorphic operations
		without access to decryption.

		Args:
			scheme: HE scheme name. Defaults to "CKKS".
			mode: CKKS configuration mode. Defaults to DEFAULT.
			security_level: Security level. Defaults to 128.
			_round: Enable rounding on decryption. Defaults to False.
			decimals: Decimal places for rounding. Defaults to 2.
			output_path: Directory for saving keys. Defaults to "/sink".
			save: Persist keys to disk. Defaults to False.

		Returns:
			Ckks: Configured instance without secret key.
		"""
		HE, n_feat, scale = Ckks._create_he_context(scheme=scheme, mode=mode, security_level=security_level)
		if save:
			HE.save_context(f"{output_path}/{Ckks._ctx_id}")
			HE.save_public_key(f"{output_path}/{Ckks._public_key_id}")
		return Ckks(he_object=HE, _round=_round, decimals=decimals, n_features=n_feat)

	@staticmethod
	def create_client(scheme: str = "CKKS", mode: CkksModes = CkksModes.DEFAULT, security_level: int = 128, _round: bool = False, decimals: int = 2, output_path: str = "/sink", save: bool = False, enable_relinearize: bool = False, enable_rotate: bool = False) -> 'Ckks':
		"""Create a CKKS context for client usage (full key access).

		Generates a Pyfhel context with secret key for encryption
		and decryption capabilities.

		Args:
			scheme: HE scheme name. Defaults to "CKKS".
			mode: CKKS configuration mode. Defaults to DEFAULT.
			security_level: Security level. Defaults to 128.
			_round: Enable rounding on decryption. Defaults to False.
			decimals: Decimal places for rounding. Defaults to 2.
			output_path: Directory for saving keys. Defaults to "/sink".
			save: Persist keys to disk. Defaults to False.
			enable_relinearize: Generate relinearization keys. Defaults to False.
			enable_rotate: Generate rotation keys. Defaults to False.

		Returns:
			Ckks: Configured instance with full key access.
		"""
		HE, n_feat, scale = Ckks._create_he_context(
			scheme=scheme, mode=mode, security_level=security_level,
			enable_relinearize=enable_relinearize, enable_rotate=enable_rotate,
		)
		if save:
			HE.save_context(f"{output_path}/{Ckks._ctx_id}")
			HE.save_public_key(f"{output_path}/{Ckks._public_key_id}")
			HE.save_secret_key(f"{output_path}/{Ckks._secret_key_id}")
			if enable_relinearize:
				HE.save_relin_key(f"{output_path}/{Ckks._relin_key_id}")
			if enable_rotate:
				HE.save_rotate_key(f"{output_path}/{Ckks._rotate_key_id}")
		return Ckks(he_object=HE, _round=_round, decimals=decimals, security_level=security_level, n_features=n_feat)

	@staticmethod
	def load_pyfhel_client(path: str = "/sink", ctx_filename: str = "ctx", pubkey_filename: str = "pubkey", secretkey_filename: str = "secretkey", relinkey_filename: str = "", rotatekey_filename: str = "") -> Pyfhel:
		"""Load a Pyfhel object with all keys (client-side).

		Loads context, public key, and secret key from disk. Optionally
		loads relinearisation and rotation keys.

		Args:
			path: Directory containing key files. Defaults to "/sink".
			ctx_filename: Context filename. Defaults to "ctx".
			pubkey_filename: Public key filename. Defaults to "pubkey".
			secretkey_filename: Secret key filename. Defaults to "secretkey".
			relinkey_filename: Relinearisation key filename. Defaults to "".
			rotatekey_filename: Rotation key filename. Defaults to "".

		Returns:
			Pyfhel: Initialised Pyfhel object with context and keys loaded.
		"""
		return Ckks.load_pyfhel(
			path               = path,
			ctx_filename       = ctx_filename,
			pubkey_filename    = pubkey_filename,
			secretkey_filename = secretkey_filename,
			relinkey_filename  = relinkey_filename,
			rotatekey_filename = rotatekey_filename
		)

	@staticmethod
	def load_pyfhel_server(path: str = "/sink", ctx_filename: str = "ctx", pubkey_filename: str = "pubkey", relinkey_filename: str = "", rotatekey_filename: str = "") -> Pyfhel:
		"""Load a Pyfhel object with public key only (server-side).

		Loads context and public key from disk. Secret key is NOT loaded.
		Optionally loads relinearisation and rotation keys.

		Args:
			path: Directory containing key files. Defaults to "/sink".
			ctx_filename: Context filename. Defaults to "ctx".
			pubkey_filename: Public key filename. Defaults to "pubkey".
			relinkey_filename: Relinearisation key filename. Defaults to "".
			rotatekey_filename: Rotation key filename. Defaults to "".

		Returns:
			Pyfhel: Initialised Pyfhel object with context and public key.
		"""
		HE = Pyfhel()
		HE.load_context("{}/{}".format(path, ctx_filename))
		HE.load_public_key("{}/{}".format(path, pubkey_filename))
		if not relinkey_filename == "":
			HE.load_relin_key(f"{path}/{relinkey_filename}")
		if not rotatekey_filename == "":
			HE.load_rotate_key(f"{path}/{rotatekey_filename}")
		return HE

	@staticmethod
	def load_pyfhel(path: str = "/sink", ctx_filename: str = "ctx", pubkey_filename: str = "pubkey", secretkey_filename: str = "secretkey", relinkey_filename: str = "", rotatekey_filename: str = "") -> Pyfhel:
		"""Load an existing Pyfhel context and keys from disk.

		Args:
			path: Directory containing key files. Defaults to "/sink".
			ctx_filename: Context filename. Defaults to "ctx".
			pubkey_filename: Public key filename. Defaults to "pubkey".
			secretkey_filename: Secret key filename. Defaults to "secretkey".
			relinkey_filename: Relinearization key filename. Defaults to "".
			rotatekey_filename: Rotation key filename. Defaults to "".

		Returns:
			Pyfhel: Initialised Pyfhel object with context and keys loaded.
		"""
		HE = Pyfhel()
		HE.load_context("{}/{}".format(path, ctx_filename))
		HE.load_public_key("{}/{}".format(path, pubkey_filename))
		HE.load_secret_key("{}/{}".format(path, secretkey_filename))
		if not relinkey_filename == "":
			HE.load_relin_key(f"{path}/{relinkey_filename}")
		if not rotatekey_filename == "":
			HE.load_rotate_key(f"{path}/{rotatekey_filename}")
		return HE

	@staticmethod
	def from_pyfhel_client(_round: bool = False, decimals: int = 2, path: str = "/sink", ctx_filename: str = "ctx", pubkey_filename: str = "pubkey", secretkey_filename: str = "secretkey", relinkey_filename: str = "", rotatekey_filename: str = "") -> 'Ckks':
		"""Create a Ckks instance with full key access (client-side).

		Loads context and all keys from disk and returns a configured
		Ckks object.

		Args:
			_round: Whether to round decrypted values. Defaults to False.
			decimals: Number of decimal places for rounding. Defaults to 2.
			path: Directory containing key files. Defaults to "/sink".
			ctx_filename: Context filename. Defaults to "ctx".
			pubkey_filename: Public key filename. Defaults to "pubkey".
			secretkey_filename: Secret key filename. Defaults to "secretkey".
			relinkey_filename: Relinearisation key filename. Defaults to "".
			rotatekey_filename: Rotation key filename. Defaults to "".

		Returns:
			Ckks: Configured CKKS instance with full key access.
		"""
		return Ckks.from_pyfhel(_round=_round, decimals=decimals, path=path, ctx_filename=ctx_filename, pubkey_filename=pubkey_filename, secretkey_filename=secretkey_filename, relinkey_filename=relinkey_filename, rotatekey_filename=rotatekey_filename)

	@staticmethod
	def from_pyfhel_server(_round: bool = False, decimals: int = 2, path: str = "/sink", ctx_filename: str = "ctx", pubkey_filename: str = "pubkey", relinkey_filename: str = "", rotatekey_filename: str = "") -> 'Ckks':
		"""Create a Ckks instance with public key only (server-side).

		Loads context and public key from disk. The secret key is NOT
		loaded, making this suitable for server-side operations that
		only perform encryption and homomorphic computation.

		Args:
			_round: Whether to round decrypted values. Defaults to False.
			decimals: Number of decimal places for rounding. Defaults to 2.
			path: Directory containing key files. Defaults to "/sink".
			ctx_filename: Context filename. Defaults to "ctx".
			pubkey_filename: Public key filename. Defaults to "pubkey".
			relinkey_filename: Relinearisation key filename. Defaults to "".
			rotatekey_filename: Rotation key filename. Defaults to "".

		Returns:
			Ckks: Configured CKKS instance with public-key-only access.
		"""
		HE = Pyfhel()
		HE.load_context(f"{path}/{ctx_filename}")
		HE.load_public_key(f"{path}/{pubkey_filename}")
		if relinkey_filename:
			HE.load_relin_key(f"{path}/{relinkey_filename}")
		if rotatekey_filename:
			HE.load_rotate_key(f"{path}/{rotatekey_filename}")
		return Ckks(
			he_object  = HE,
			_round     = _round,
			decimals   = decimals,
			n_features = int(HE.get_nSlots()) if hasattr(HE, 'get_nSlots') else HE.n // 2,
		)

	@staticmethod
	def from_pyfhel(_round: bool = False, decimals: int = 2, path: str = "/sink", ctx_filename: str = "ctx", pubkey_filename: str = "pubkey", secretkey_filename: str = "secretkey", relinkey_filename: str = "", rotatekey_filename: str = "") -> 'Ckks':
		"""Create a Ckks instance from existing Pyfhel context files on disk.

		Args:
			_round: Whether to round decrypted values. Defaults to False.
			decimals: Number of decimal places for rounding. Defaults to 2.
			path: Directory containing key files. Defaults to "/sink".
			ctx_filename: Context filename. Defaults to "ctx".
			pubkey_filename: Public key filename. Defaults to "pubkey".
			secretkey_filename: Secret key filename. Defaults to "secretkey".
			relinkey_filename: Relinearization key filename. Defaults to "".
			rotatekey_filename: Rotation key filename. Defaults to "".

		Returns:
			Ckks: Configured CKKS instance.
		"""
		HE = Pyfhel()
		HE.load_context("{}/{}".format(path, ctx_filename))
		HE.load_public_key("{}/{}".format(path, pubkey_filename))
		HE.load_secret_key("{}/{}".format(path, secretkey_filename))
		if not relinkey_filename == "":
			HE.load_relin_key(f"{path}/{relinkey_filename}")
		if not rotatekey_filename == "":
			HE.load_rotate_key(f"{path}/{rotatekey_filename}")

		return Ckks(
			he_object=HE,
			_round=_round,
			decimals=decimals,
		)

	# ---- Legacy encode/encrypt/decrypt methods ----

	def encode_list(self, xs: List[float] = []):
		"""Encode a list of floats into CKKS plaintext objects.

		Args:
			xs: List of floating-point values to encode.

		Returns:
			List[PyPtxt]: List of encoded plaintext objects.
		"""
		try:
			res = []
			for x in xs:
				res.append(self.he_object.encode(x))
			return res
		except Exception as e:
			raise e

	def encrypt_list(self, plaintext_vector: List[PyPtxt] = []):
		"""Encrypt a list of plaintext objects into ciphertexts.

		Args:
			plaintext_vector: List of PyPtxt plaintext objects.

		Returns:
			List[PyCtxt]: List of encrypted ciphertexts.
		"""
		try:
			res = []
			for x in plaintext_vector:
				res.append(self.he_object.encrypt(x))
			return res

		except Exception as e:
			raise e

	def encrypt_matrix_list(self, plaintext_matrix: List[List[PyPtxt]] = []):
		"""Encrypt a matrix (list of lists) of plaintext values.

		Args:
			plaintext_matrix: List of lists of plaintext values.

		Returns:
			List[List[PyCtxt]]: Encrypted matrix.
		"""
		try:
			res = []
			for x in plaintext_matrix:
				res.append(self.encrypt_list(plaintext_vector=self.encode_list(x)))
			return res

		except Exception as e:
			raise e

	def decrypt_matrix_list(self, xs: List[List[PyCtxt]] = [], take: int = 1):
		"""Decrypt a matrix of ciphertexts with optional slicing.

		Args:
			xs: List of lists of CKKS ciphertexts.
			take: Number of elements to take per decryption. Defaults to 1.

		Returns:
			List: Decrypted matrix.
		"""
		try:
			res = []
			for x in xs:
				res.append(self.decrypt_list(xs=x, take=take))
			return res

		except Exception as e:
			raise e

	def decrypt_list_v2(self, xs: List[PyCtxt] = [], take: int = 0):
		"""Decrypt a list of ciphertexts extracting a single value per entry.

		Args:
			xs: List of CKKS ciphertexts.
			take: Index of the value to extract. Defaults to 0.

		Returns:
			List[float]: Decrypted values at the specified index.
		"""
		try:
			res = []
			for x in xs:
				result = self.he_object.decrypt(x)[take]
				round_result = np.round(result, decimals=self.decimals) if self.round else result
				res.append(round_result)
			return res
		except Exception as e:
			raise e

	def decrypt_list(self, xs: List[PyCtxt] = [], take: int = 0):
		"""Decrypt a list of ciphertexts returning slices of each vector.

		Args:
			xs: List of CKKS ciphertexts.
			take: Number of elements to slice. Defaults to 0 (all).

		Returns:
			List: Sliced decrypted vectors.
		"""
		try:
			res = []
			for x in xs:
				result = self.he_object.decrypt(x)[:take]
				round_result = np.round(result, decimals=self.decimals) if self.round else result
				res.append(round_result)
			return res
		except Exception as e:
			raise e

	# ---- Internal encrypt/decrypt implementations ----

	def _encryptMatrix(self, plaintext_matrix: npt.NDArray):
		"""Encrypt each row of a plaintext matrix using CKKS.

		Args:
			plaintext_matrix: Matrix of plaintext values.

		Returns:
			List[PyCtxt]: List of encrypted ciphertexts per row.
		"""
		results = []
		for v in plaintext_matrix:
			result = self._encryptVector(plaintext_vector=v)
			results.append(result)
		return results

	def _encryptVector(self, plaintext_vector: npt.NDArray):
		"""Encrypt a plaintext vector using CKKS.

		Args:
			plaintext_vector: Vector of plaintext values.

		Returns:
			PyCtxt: Encrypted ciphertext.
		"""
		result = self.he_object.encrypt(self.he_object.encode(plaintext_vector))
		return result

	def _decryptMatrix(self, ciphertext_matrix: npt.NDArray, adjust: bool = True, shape: Tuple[int, int] = (1, 1)) -> npt.NDArray:
		"""Decrypt each row of a ciphertext matrix with optional reshape.

		Args:
			ciphertext_matrix: Matrix of CKKS ciphertexts.
			adjust: Whether to reshape decrypted rows. Defaults to True.
			shape: Target shape (rows, cols) for adjustment. Defaults to (1, 1).

		Returns:
			npt.NDArray: Decrypted plaintext matrix.
		"""
		results = []
		for c in ciphertext_matrix:
			result = self._decryptVector(ciphertext_vector=c)
			results.append(result)
		return self.adjust_matrix(plaintext=results, shape=shape) if adjust else np.array(results)

	def _decryptVector(self, ciphertext_vector: npt.NDArray) -> List:
		"""Decrypt a single ciphertext vector using CKKS.

		Args:
			ciphertext_vector: CKKS ciphertext to decrypt.

		Returns:
			List: Decrypted plaintext vector.
		"""
		result = self.he_object.decryptFrac(ciphertext_vector)
		round_result = np.round(result, decimals=self.decimals) if self.round else result
		return round_result

	# ---- Legacy deprecated methods ----

	def encryptMatrix(self, plaintext_matrix: npt.NDArray):
		"""@deprecated: Use encrypt_matrix() instead."""
		warnings.warn("encryptMatrix() is deprecated, use encrypt_matrix()", DeprecationWarning, stacklevel=2)
		return self._encryptMatrix(plaintext_matrix=plaintext_matrix)

	def encryptVector(self, plaintext_vector: npt.NDArray):
		"""@deprecated: Use encrypt_vector() instead."""
		warnings.warn("encryptVector() is deprecated, use encrypt_vector()", DeprecationWarning, stacklevel=2)
		return self._encryptVector(plaintext_vector=plaintext_vector)

	def decryptMatrix(self, ciphertext_matrix: npt.NDArray, adjust: bool = True, shape: Tuple[int, int] = (1, 1)) -> npt.NDArray:
		"""@deprecated: Use decrypt_matrix() instead."""
		warnings.warn("decryptMatrix() is deprecated, use decrypt_matrix()", DeprecationWarning, stacklevel=2)
		return self._decryptMatrix(ciphertext_matrix=ciphertext_matrix, adjust=adjust, shape=shape)

	def decryptVector(self, ciphertext_vector: npt.NDArray) -> List:
		"""@deprecated: Use decrypt_vector() instead."""
		warnings.warn("decryptVector() is deprecated, use decrypt_vector()", DeprecationWarning, stacklevel=2)
		return self._decryptVector(ciphertext_vector=ciphertext_vector)

	# ---- Utility methods ----

	def __adjust_decrypt(self, plaintext, decryptext):
		"""Remove extra elements from decrypted rows to match plaintext shape.

		Args:
			plaintext: Original plaintext matrix for shape reference.
			decryptext: Decrypted output to trim.

		Returns:
			List: Trimmed decrypted rows matching plaintext column count.
		"""
		plaintext_shape = Utils.getShapeOfMatrix(plaintext)
		return [array[:plaintext_shape[1]].tolist() for array in decryptext]

	def adjust_vector_decrypt(self, decryptext: npt.NDArray, n_elements: int = 1):
		"""Truncate a decrypted vector to a specified number of elements.

		Args:
			decryptext: Decrypted vector to truncate.
			n_elements: Number of elements to keep. Defaults to 1.

		Returns:
			npt.NDArray: Truncated vector.
		"""
		return decryptext[:n_elements]

	def adjust_matrix(self, plaintext: npt.NDArray, shape: Tuple[int, int]):
		"""Truncate each row of a matrix to the specified shape.

		Args:
			plaintext: Matrix to reshape.
			shape: Target (rows, cols) shape.

		Returns:
			npt.NDArray: Reshaped matrix.
		"""
		xs = []
		for i in range(shape[0]):
			x = plaintext[i][:shape[1]]
			xs.append(x)
		return np.array(xs)

	@staticmethod
	def post_process(matrix, threshold=1e-5):
		"""Apply threshold-based postprocessing, setting small values to zero.

		Args:
			matrix: Input matrix to process.
			threshold: Values below this become zero. Defaults to 1e-5.

		Returns:
			ndarray: Matrix with small values zeroed out.
		"""
		return np.where(np.abs(matrix) < threshold, 0, matrix)
	
	