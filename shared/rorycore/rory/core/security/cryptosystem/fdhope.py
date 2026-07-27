import warnings
import random
import pickle
import os
import numpy as np


from collections import Counter
from rory.core.utils.utils import Utils
from rory.core.interfaces.cipher_result import CipherResult
from rory.core.security.cryptosystem.abstract import Cipher
from typing import Tuple, Dict, List
import numpy.typing as npt


"""
Description: 
    A class to represent Frequency Concealment and Distribution OPE (FDH-OPE) scheme, 
    used to facilitate the required UDM operation.
"""


class Fdhope(Cipher):
	"""Frequency Concealment and Distribution OPE (FDH-OPE) scheme.

	An Order-Preserving Encryption scheme used for encrypted comparison
	operations on Updatable Distance Matrices (UDM). Does not support
	decryption (ciphertexts preserve plaintext ordering for comparison).

	Attributes:
		messagespace: Mapping of range IDs to (lower, upper) intervals.
		cipherspace: Mapping of range IDs to (lower, upper) intervals.
	"""

	def __init__(self, seed: int = 42):
		"""Initialize the FDHOPE scheme.

		Args:
			seed: Seed for the random number generator. Defaults to 42.
		"""
		self.seed = seed
		self.messagespace = {}
		self.cipherspace = {}

	# ---- Standardized interface ----

	def generate_keys(self, dataset: npt.NDArray, minVal: int = 0, max_range: int = 8, proportion: int = 15, range_limit: int = 2, default_intervalLenght: float = 0.001):
		"""Generate OPE key intervals from a dataset.

		Args:
			dataset: The dataset from which secret keys will be derived.
			minVal: Starting value for the range space. Defaults to 0.
			max_range: Maximum number of ranges to generate. Defaults to 8.
			proportion: Scaling factor for cipher space. Defaults to 15.
			range_limit: Offset for message space max value. Defaults to 2.
			default_intervalLenght: Default interval length. Defaults to 0.001.

		Returns:
			Tuple[Dict, Dict]: (messagespace, cipherspace) mappings.
		"""
		self.messagespace, self.cipherspace = self._keygen(
			dataset                = dataset,
			minVal                 = minVal,
			max_range              = max_range,
			proportion             = proportion,
			range_limit            = range_limit,
			default_intervalLenght = default_intervalLenght,
		)
		return self.messagespace, self.cipherspace

	def encrypt_scalar(self, plaintext: float, sens: float = 0.00001):
		"""Encrypt a single plaintext value using stored key spaces.

		Args:
			plaintext: Plaintext value to encrypt.
			sens: Sensitivity parameter for noise. Defaults to 0.00001.

		Returns:
			CipherResult: Container with the encrypted value.
		"""
		result = self._encrypt(
			plaintext    = plaintext,
			messagespace = self.messagespace,
			cipherspace  = self.cipherspace,
			sens         = sens,
		)
		return CipherResult(data=result)

	def encrypt_vector(self, plaintext_vector: npt.NDArray, sens: float = 0.00001):
		"""Encrypt a vector of plaintext values using stored key spaces.

		Args:
			plaintext_vector: Vector of plaintext values.
			sens: Sensitivity parameter for noise. Defaults to 0.00001.

		Returns:
			CipherResult: Container with the encrypted vector.
		"""
		result = self._encryptVector(
			plaintext_vector = plaintext_vector,
			messagespace     = self.messagespace,
			cipherspace      = self.cipherspace,
			sens             = sens,
		)
		return CipherResult(data=result)

	def encrypt_matrix(self, plaintext_matrix: npt.NDArray, sens: float = 0.00001):
		"""Encrypt a matrix of plaintext values using stored key spaces.

		Args:
			plaintext_matrix: Matrix of plaintext values.
			sens: Sensitivity parameter for noise. Defaults to 0.00001.

		Returns:
			CipherResult: Container with the encrypted matrix.
		"""
		result = self._encryptMatrix(
			plaintext_matrix = plaintext_matrix,
			messagespace     = self.messagespace,
			cipherspace      = self.cipherspace,
			sens             = sens,
		)
		return CipherResult(data=result)

	def encrypt_tensor(self, plaintext_tensor: npt.NDArray, sens: float = 0.00001):
		"""Encrypt a 3D tensor using stored key spaces (FDHOPE-specific).

		Args:
			plaintext_tensor: 3D tensor of plaintext values.
			sens: Sensitivity parameter for noise. Defaults to 0.00001.

		Returns:
			CipherResult: Container with the encrypted tensor.
		"""
		result = self._encryptTensor(
			plaintext_tensor = plaintext_tensor,
			messagespace     = self.messagespace,
			cipherspace      = self.cipherspace,
			sens             = sens,
		)
		return CipherResult(data=result)

	def decrypt_scalar(self, ciphertext):
		"""FDHOPE does not support decryption.

		Args:
			ciphertext: Ignored.

		Raises:
			NotImplementedError: Always raised.
		"""
		raise NotImplementedError("FDHOPE is an Order-Preserving Encryption scheme and does not support decryption")

	def decrypt_vector(self, ciphertext_vector):
		"""FDHOPE does not support decryption.

		Args:
			ciphertext_vector: Ignored.

		Raises:
			NotImplementedError: Always raised.
		"""
		raise NotImplementedError("FDHOPE is an Order-Preserving Encryption scheme and does not support decryption")

	def decrypt_matrix(self, ciphertext_matrix):
		"""FDHOPE does not support decryption.

		Args:
			ciphertext_matrix: Ignored.

		Raises:
			NotImplementedError: Always raised.
		"""
		raise NotImplementedError("FDHOPE is an Order-Preserving Encryption scheme and does not support decryption")

	def save_keys(self, path: str):
		"""Save OPE key intervals to disk.

		Args:
			path: Directory path to save keys.
		"""
		os.makedirs(path, exist_ok=True)
		with open(f"{path}/fdhope_keys.pkl", "wb") as f:
			pickle.dump({
				"messagespace": self.messagespace,
				"cipherspace": self.cipherspace,
			}, f)

	def load_keys(self, path: str):
		"""Load OPE key intervals from disk.

		Args:
			path: Directory path where keys are stored.
		"""
		with open(f"{path}/fdhope_keys.pkl", "rb") as f:
			data = pickle.load(f)
		self.messagespace = data["messagespace"]
		self.cipherspace  = data["cipherspace"]

	# ---- Internal implementations ----

	def _keygen(self, dataset: npt.NDArray, minVal: int = 0, max_range: int = 8, proportion: int = 15, range_limit: int = 2, default_intervalLenght: float = 0.001) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]:
		"""Generate OPE key intervals from a dataset.

		Calculates density-based message space and cipher space
		intervals for order-preserving encryption.

		Args:
			dataset: Dataset for density calculation.
			minVal: Minimum range value. Defaults to 0.
			max_range: Maximum number of ranges. Defaults to 8.
			proportion: Cipher space scaling factor. Defaults to 15.
			range_limit: Message space max offset. Defaults to 2.
			default_intervalLenght: Default interval length. Defaults to 0.001.

		Returns:
			Tuple containing (messagespace, cipherspace) interval mappings.
		"""
		Dshape              = Utils.getShapeOfMatrix(dataset)
		maxVal_messagespace = round(Fdhope.findMax(dataset)) + range_limit
		maxVal_cipherspace  = maxVal_messagespace * proportion
		lenTriangle         = Dshape[0] * Dshape[1]
		n_range             = random.randint(3, max_range)

		range_ids = Fdhope.generate_range_keys(n_range=n_range)
		ranges_values = [minVal] + sorted([np.random.uniform(minVal + 0.2, maxVal_messagespace) for _ in range(n_range - 1)]) + [maxVal_messagespace]

		messagespace = Fdhope.generate_range_values(
			minValue      = minVal,
			maxValue      = maxVal_messagespace,
			n_range       = n_range,
			range_ids     = range_ids,
			ranges_values = ranges_values
		)
		initial_dens = dict(zip(range_ids, [0] * n_range))

		density = Fdhope.calculate_dens(
			dataset      = dataset,
			messagespace = messagespace,
			initial_dens = initial_dens
		)

		intervalLength = Fdhope.calculate_intervalLength(
			density                = density,
			lenTriangle            = lenTriangle,
			maxVal_cipherspace     = maxVal_cipherspace,
			default_intervalLenght = default_intervalLenght
		)

		cipherspace = Fdhope.generate_range_values(
			minValue      = minVal,
			maxValue      = maxVal_cipherspace,
			n_range       = len(intervalLength),
			range_ids     = intervalLength.keys(),
			ranges_values = np.cumsum([0] + list(intervalLength.values()))
		)
		return messagespace, cipherspace

	def _encryptTensor(self, plaintext_tensor: npt.NDArray, messagespace: Dict[str, Tuple[float, float]], cipherspace: Dict[str, Tuple[float, float]], sens: float = 0.00001) -> npt.NDArray:
		"""Encrypt a 3D tensor element-wise using OPE.

		Args:
			plaintext_tensor: 3D tensor of plaintext values.
			messagespace: Message space interval mappings.
			cipherspace: Cipher space interval mappings.
			sens: Sensitivity parameter for noise. Defaults to 0.00001.

		Returns:
			npt.NDArray: Encrypted 3D tensor.
		"""
		results = []
		for plaintext_matrix in plaintext_tensor:
			x = self._encryptMatrix(
				plaintext_matrix = plaintext_matrix,
				messagespace     = messagespace,
				cipherspace      = cipherspace,
				sens             = sens
			)
			results.append(x)
		return np.array(results)

	def _encryptMatrix(self, plaintext_matrix: npt.NDArray, messagespace: Dict[str, Tuple[float, float]], cipherspace: Dict[str, Tuple[float, float]], sens: float = 0.00001) -> npt.NDArray:
		"""Encrypt a matrix element-wise using OPE.

		Args:
			plaintext_matrix: Matrix of plaintext values.
			messagespace: Message space interval mappings.
			cipherspace: Cipher space interval mappings.
			sens: Sensitivity parameter for noise. Defaults to 0.00001.

		Returns:
			npt.NDArray: Encrypted matrix.
		"""
		results = []
		for plaintext_vector in plaintext_matrix:
			x = self._encryptVector(
				plaintext_vector = plaintext_vector,
				messagespace     = messagespace,
				cipherspace      = cipherspace,
				sens             = sens
			)
			results.append(x)
		return np.array(results)

	def _encryptVector(self, plaintext_vector: npt.NDArray, messagespace: Dict[str, Tuple[float, float]], cipherspace: Dict[str, Tuple[float, float]], sens: float = 0.00001) -> npt.NDArray:
		"""Encrypt a vector element-wise using OPE.

		Args:
			plaintext_vector: Vector of plaintext values.
			messagespace: Message space interval mappings.
			cipherspace: Cipher space interval mappings.
			sens: Sensitivity parameter for noise. Defaults to 0.00001.

		Returns:
			npt.NDArray: Encrypted vector.
		"""
		results = []
		for plaintext in plaintext_vector:
			x = self._encrypt(
				plaintext    = plaintext,
				messagespace = messagespace,
				cipherspace  = cipherspace,
				sens         = sens
			)
			results.append(x)
		return np.array(results)

	def _encrypt(self, plaintext: float, messagespace: Dict[str, Tuple[float, float]], cipherspace: Dict[str, Tuple[float, float]], sens: float = 0.00001) -> float:
		"""Encrypt a single plaintext value using OPE.

		Maps the plaintext to a message space interval, then scales
		it into the corresponding cipher space interval with added
		random noise for frequency concealment.

		Args:
			plaintext: Plaintext value to encrypt.
			messagespace: Message space interval mappings.
			cipherspace: Cipher space interval mappings.
			sens: Sensitivity parameter for noise. Defaults to 0.00001.

		Returns:
			float: Order-preserving ciphertext.
		"""
		interval_id = Fdhope.getIntervalID(
			plaintext    = plaintext,
			messagespace = messagespace
		)
		messagespace_min, messagespace_max = Fdhope.getBoundary(
			interval_id = interval_id,
			space       = messagespace
		)
		cipherspace_min, cipherspace_max = Fdhope.getBoundary(
			interval_id = interval_id,
			space       = cipherspace,
		)
		scale      = (cipherspace_max - cipherspace_min) / (messagespace_max - messagespace_min)
		delta      = float(random.uniform(0, sens * scale))
		ciphertext = cipherspace_min + scale * (abs(plaintext) - messagespace_min) + delta

		if (plaintext < 0):
			ciphertext = ciphertext * (-1.0)
		return ciphertext
	
	
	# ---- Deprecated static methods ----

	@staticmethod
	def keygen(dataset: npt.NDArray, minVal: int = 0, max_range: int = 8, proportion: int = 15, range_limit: int = 2, default_intervalLenght: float = 0.001) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]:
		"""@deprecated: Use instance method generate_keys() instead."""
		warnings.warn("Fdhope.keygen() is deprecated, use instance method generate_keys()", DeprecationWarning, stacklevel=2)
		fd = Fdhope()
		return fd._keygen(
			dataset                = dataset,
			minVal                 = minVal,
			max_range              = max_range,
			proportion             = proportion,
			range_limit            = range_limit,
			default_intervalLenght = default_intervalLenght,
		)

	@staticmethod
	def encryptTensor(plaintext_tensor: npt.NDArray, messagespace: Dict[str, Tuple[float, float]], cipherspace: Dict[str, Tuple[float, float]], sens: float = 0.00001) -> CipherResult:
		"""@deprecated: Use instance method encrypt_tensor() instead."""
		warnings.warn("Fdhope.encryptTensor() is deprecated, use instance method encrypt_tensor()", DeprecationWarning, stacklevel=2)
		fd = Fdhope()
		result = fd._encryptTensor(
			plaintext_tensor = plaintext_tensor,
			messagespace     = messagespace,
			cipherspace      = cipherspace,
			sens             = sens,
		)
		return CipherResult(data=result)

	@staticmethod
	def encryptMatrix(plaintext_matrix: npt.NDArray, messagespace: Dict[str, Tuple[float, float]], cipherspace: Dict[str, Tuple[float, float]], sens: float = 0.00001) -> CipherResult:
		"""@deprecated: Use instance method encrypt_matrix() instead."""
		warnings.warn("Fdhope.encryptMatrix() is deprecated, use instance method encrypt_matrix()", DeprecationWarning, stacklevel=2)
		fd = Fdhope()
		result = fd._encryptMatrix(
			plaintext_matrix = plaintext_matrix,
			messagespace     = messagespace,
			cipherspace      = cipherspace,
			sens             = sens,
		)
		return CipherResult(data=result)

	@staticmethod
	def encryptVector(plaintext_vector: npt.NDArray, messagespace: Dict[str, Tuple[float, float]], cipherspace: Dict[str, Tuple[float, float]], sens: float = 0.00001) -> CipherResult:
		"""@deprecated: Use instance method encrypt_vector() instead."""
		warnings.warn("Fdhope.encryptVector() is deprecated, use instance method encrypt_vector()", DeprecationWarning, stacklevel=2)
		fd = Fdhope()
		result = fd._encryptVector(
			plaintext_vector = plaintext_vector,
			messagespace     = messagespace,
			cipherspace      = cipherspace,
			sens             = sens,
		)
		return CipherResult(data=result)

	@staticmethod
	def encrypt(plaintext: float, messagespace: Dict[str, Tuple[float, float]], cipherspace: Dict[str, Tuple[float, float]], sens: float = 0.00001) -> float:
		"""@deprecated: Use instance method encrypt_scalar() instead."""
		warnings.warn("Fdhope.encrypt() is deprecated, use instance method encrypt_scalar()", DeprecationWarning, stacklevel=2)
		fd = Fdhope()
		return fd._encrypt(
			plaintext    = plaintext,
			messagespace = messagespace,
			cipherspace  = cipherspace,
			sens         = sens,
		)

	# ---- Static utility methods (keep as-is, internal helpers) ----

	@staticmethod
	def calculate_dens(dataset: npt.NDArray, messagespace: Dict[str, Tuple[float, float]], initial_dens: Dict[str, int]) -> Dict[int, int]:
		"""Compute density for each message space interval.

		Counts how many values from the dataset fall into each interval.

		Args:
			dataset: Plaintext dataset.
			messagespace: Message space interval mappings.
			initial_dens: Initial density dictionary with zero counts.

		Returns:
			Dict[int, int]: Density count per interval ID.
		"""
		dens   = [Fdhope.getIntervalID(plaintext=v, messagespace=messagespace) for v in np.ravel(dataset)]
		conteo = Counter(dens)
		return {**initial_dens, **dict(conteo.items())}

	@staticmethod
	def calculate_intervalLength(density: Dict[str, str], lenTriangle: int, maxVal_cipherspace: int, default_intervalLenght: float) -> Dict[str, float]:
		"""Compute cipher space interval lengths proportional to density.

		Allocates cipher space proportionally to the density of each
		message space interval, with a default for empty intervals.

		Args:
			density: Density count per interval.
			lenTriangle: Product of dataset dimensions.
			maxVal_cipherspace: Maximum value in cipher space.
			default_intervalLenght: Default length for empty intervals.

		Returns:
			Dict[str, float]: Interval length per interval ID.
		"""
		intervalLenght: Dict[str, float] = {}
		for key, value in density.items():
			if value == 0:
				intervalLenght[key] = default_intervalLenght
			else:
				ratio = value / lenTriangle
				intervalLenght[key] = maxVal_cipherspace * ratio
		return intervalLenght

	@staticmethod
	def generate_range_keys(n_range: int) -> List[str]:
		"""Generate string identifiers for the requested number of ranges.

		Args:
			n_range: Number of ranges to create.

		Returns:
			List[str]: Range identifiers (e.g. "RANGE_0", "RANGE_1", ...).
		"""
		return [f'RANGE_{i}' for i in range(n_range)]

	@staticmethod
	def generate_range_values(minValue: int, maxValue: int, n_range: int, range_ids: List[str], ranges_values: List[float]) -> Dict[str, Tuple[float, float]]:
		"""Build the (min, max) boundary mapping for each range ID.

		Args:
			minValue: Minimum value for the first range.
			maxValue: Maximum value for the last range.
			n_range: Number of ranges.
			range_ids: List of range identifiers.
			ranges_values: Sorted list of range boundary values.

		Returns:
			Dict[str, Tuple[float, float]]: Mapping from range ID to
				(min, max) boundary tuple.
		"""
		rangos = {}
		for index, range_id in enumerate(range_ids):
			minVal = ranges_values[index]
			maxVal = float(maxValue + 1) if (index == n_range - 1) else ranges_values[index + 1]
			rangos[range_id] = (minVal, maxVal)
		return rangos

	@staticmethod
	def findMax(D: npt.NDArray) -> float:
		"""Find the maximum absolute value in a dataset.

		Args:
			D: Dataset array of any dimension.

		Returns:
			float: Maximum absolute value.
		"""
		return np.max(np.abs(np.ravel(D)))

	@staticmethod
	def getIntervalID(plaintext: float, messagespace: Dict[str, Tuple[float, float]]) -> str:
		"""Map a plaintext value to its message space interval ID.

		Searches intervals for the one containing the absolute plaintext
		value.

		Args:
			plaintext: Plaintext value to map.
			messagespace: Message space interval mappings.

		Returns:
			str: Interval ID that contains the plaintext value.
		"""
		plaintext = abs(plaintext)
		sorted_keys = sorted(messagespace.keys())
		
		for i, key in enumerate(sorted_keys):
			min_val, max_val = messagespace[key]
			if i == len(sorted_keys) - 1:
				if min_val <= plaintext:
					return key
			else:
				if min_val <= plaintext < max_val:
					return key
		return sorted_keys[0]

	@staticmethod
	def getBoundary(interval_id: int, space: Dict[str, Tuple[float, float]]) -> Tuple[float, float]:
		"""Retrieve the (min, max) boundaries for a given interval.

		Args:
			interval_id: Interval identifier.
			space: Interval space mapping (message or cipher).

		Returns:
			Tuple[float, float]: (min, max) boundary values.
		"""
		return space.get(interval_id)
