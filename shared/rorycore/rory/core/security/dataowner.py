import numpy as np
import numpy.typing as npt
from time import time
from typing import Optional

from rory.core.enums.algorithms import Algorithm
from rory.core.enums.schemes import Scheme
from rory.core.security.recipe import ALGORITHM_RECIPES, RecipeStep
from rory.core.security.cryptosystem.fdhope import Fdhope
from rory.core.utils.utils import Utils
from rory.core.interfaces.outsourced_result import OutsourcedDataResult
from rory.core.security.scheme_params import (
	SchemeParams,
	LiuParams,
	CkksParams,
	PaillierParams,
	LiuAndFdhopeParams,
	CkksAndFdhopeParams,
)
from rory.core.security.cryptosystem.abstract import HomomorphicCipher

_SCHEME_PARAMS_MAP: dict[Scheme, SchemeParams] = {
	Scheme.LIU:             LiuParams(),
	Scheme.CKKS:            CkksParams(),
	Scheme.PAILLIER:        PaillierParams(),
	Scheme.LIU_AND_FDHOPE:  LiuAndFdhopeParams(),
	Scheme.CKKS_AND_FDHOPE: CkksAndFdhopeParams(),
}

_VALID_COMBINATIONS = {
	Algorithm.SKMEANS:               {Scheme.LIU, Scheme.LIU_AND_FDHOPE},
	Algorithm.DBSKMEANS:             {Scheme.LIU_AND_FDHOPE},
	Algorithm.DBSNNC:                {Scheme.LIU_AND_FDHOPE},
	Algorithm.NNC:                   {Scheme.NONE},
	Algorithm.SKMEANS_PQC:           {Scheme.CKKS, Scheme.CKKS_AND_FDHOPE},
	Algorithm.DBSKMEANS_PQC:         {Scheme.CKKS_AND_FDHOPE},
	Algorithm.DBSNNC_PQC:            {Scheme.CKKS_AND_FDHOPE},
	Algorithm.SKNN:                  {Scheme.LIU, Scheme.CKKS},
	Algorithm.SKNN_PQC:              {Scheme.CKKS},
	Algorithm.KMEANS:                {Scheme.NONE},
	Algorithm.KNN:                   {Scheme.NONE},
	Algorithm.LOGISTIC_REGRESSION:   {Scheme.NONE},
	Algorithm.PPLR:                  {Scheme.CKKS},
	Algorithm.NONE:                  {Scheme.CKKS, Scheme.LIU, Scheme.PAILLIER},
}


def _is_fdhope_scheme(scheme: Scheme) -> bool:
	"""Check if a scheme includes FDHOPE for order-preserving encryption.

	Args:
		scheme: Scheme enum value.

	Returns:
		bool: True if the scheme bundles FDHOPE.
	"""
	return scheme in {Scheme.LIU_AND_FDHOPE, Scheme.CKKS_AND_FDHOPE}


def _is_ckks_scheme(scheme: Scheme) -> bool:
	"""Check if a scheme includes CKKS homomorphic encryption.

	Args:
		scheme: Scheme enum value.

	Returns:
		bool: True if the scheme includes CKKS.
	"""
	return scheme in {Scheme.CKKS, Scheme.CKKS_AND_FDHOPE}


def _is_liu_scheme(scheme: Scheme) -> bool:
	"""Check if a scheme includes Liu homomorphic encryption.

	Args:
		scheme: Scheme enum value.

	Returns:
		bool: True if the scheme includes Liu.
	"""
	return scheme in {Scheme.LIU, Scheme.LIU_AND_FDHOPE}


def _is_pq_algorithm(algorithm: Algorithm) -> bool:
	"""Check if an algorithm is post-quantum (uses CKKS).

	Args:
		algorithm: Algorithm enum value.

	Returns:
		bool: True if the algorithm requires PQC support.
	"""
	return algorithm in {
		Algorithm.SKMEANS_PQC,
		Algorithm.DBSKMEANS_PQC,
		Algorithm.DBSNNC_PQC,
		Algorithm.SKNN_PQC,
	}


def _is_dbskmeans_algorithm(algorithm: Algorithm) -> bool:
	"""Check if an algorithm is a variant of double-blind secure KMeans.

	Args:
		algorithm: Algorithm enum value.

	Returns:
		bool: True if the algorithm is DBSKMeans.
	"""
	return algorithm in {Algorithm.DBSKMEANS, Algorithm.DBSKMEANS_PQC}


def _is_dbsnnc_algorithm(algorithm: Algorithm) -> bool:
	"""Check if an algorithm is a variant of double-blind secure NNC.

	Args:
		algorithm: Algorithm enum value.

	Returns:
		bool: True if the algorithm is DBSNNC.
	"""
	return algorithm in {Algorithm.DBSNNC, Algorithm.DBSNNC_PQC}


class _DataOwnerBuilder:
	def __init__(self):
		"""Initialise a DataOwner builder with empty configuration.

		Sets algorithm, scheme, and scheme_params to None. Use
		with_algorithm(), with_scheme(), and with_scheme_params()
		to configure before calling build().
		"""
		self._algorithm: Optional[Algorithm] = None
		self._scheme: Optional[Scheme] = None
		self._scheme_params: Optional[SchemeParams] = None

	def with_algorithm(self, algorithm: Algorithm):
		"""Set the algorithm for the DataOwner being built.

		Args:
			algorithm: Algorithm enum value.

		Returns:
			_DataOwnerBuilder: Self for method chaining.
		"""
		self._algorithm = algorithm
		return self

	def with_scheme(self, scheme: Scheme):
		"""Set the cryptographic scheme for the DataOwner being built.

		Args:
			scheme: Scheme enum value.

		Returns:
			_DataOwnerBuilder: Self for method chaining.
		"""
		self._scheme = scheme
		return self

	def with_scheme_params(self, params):
		"""Set custom scheme parameters for the DataOwner being built.

		Args:
			params: SchemeParams dataclass instance.

		Returns:
			_DataOwnerBuilder: Self for method chaining.
		"""
		self._scheme_params = params
		return self

	def build(self) -> 'DataOwner':
		"""Build and return a validated DataOwner instance.

		Validates the algorithm-scheme combination and applies
		default parameters when none are specified.

		Returns:
			DataOwner: Configured DataOwner instance.

		Raises:
			ValueError: If the algorithm-scheme combination is invalid
				or required parameters are missing.
		"""
		algorithm = self._algorithm
		scheme    = self._scheme
		params    = self._scheme_params

		if algorithm is None and scheme is None:
			raise ValueError(
				"At least one of algorithm or scheme must be set"
			)

		if algorithm is None:
			algorithm = Algorithm.NONE
		if scheme is None:
			scheme = Scheme.NONE

		if scheme not in _VALID_COMBINATIONS.get(algorithm, set()):
			raise ValueError(
				f"Invalid combination: algorithm={algorithm.value}, "
				f"scheme={scheme.value}"
			)

		if scheme != Scheme.NONE and params is None:
			defaults = _SCHEME_PARAMS_MAP.get(scheme)
			if defaults is not None:
				params = defaults
			else:
				raise ValueError(
					f"Scheme {scheme.value} requires scheme_params"
				)

		return DataOwner(algorithm, scheme, params)


class DataOwner:
	def __init__(
		self,
		algorithm: Algorithm,
		scheme: Scheme,
		scheme_params: Optional[SchemeParams],
	):
		"""Initialise a DataOwner with algorithm, scheme, and parameters.

		Args:
			algorithm: Algorithm enum value.
			scheme: Scheme enum value.
			scheme_params: SchemeParams dataclass instance or None.
		"""
		self._algorithm: Algorithm = algorithm
		self._scheme: Scheme = scheme
		self._scheme_params: Optional[SchemeParams] = scheme_params
		self.primary_scheme:Optional[HomomorphicCipher] = None
		self._fdhope: Optional[Fdhope] = None
		self._sens: Optional[float] = None
		self._messageIntervals = {}
		self._cypherIntervals = {}
		self._encrypted_threshold = 0
		self._threshold = 0
		self._schemes_initialized = False

	@property
	def algorithm(self) -> Algorithm:
		"""Configured algorithm."""
		return self._algorithm

	@property
	def scheme(self) -> Scheme:
		"""Configured cryptographic scheme."""
		return self._scheme

	@property
	def scheme_params(self) -> Optional[SchemeParams]:
		"""Configured scheme parameters."""
		return self._scheme_params

	def initialize(self) -> 'DataOwner':
		"""Initialize configured schemes without processing plaintext data."""
		self._init_schemes(self._scheme, self._scheme_params)
		return self

	def reseed(self, seed: Optional[int] = None) -> 'DataOwner':
		"""Give an initialized worker owner an independent random stream.

		The primary cipher must expose a public ``reseed`` method.  Liu supports
		this operation while preserving its existing secret key.
		"""
		self.initialize()
		reseed = getattr(self.primary_scheme, "reseed", None)
		if reseed is None:
			raise ValueError(
				f"Scheme {self._scheme.value} does not support worker reseeding"
			)
		reseed(seed)
		return self

	@staticmethod
	def with_algorithm(algorithm: Algorithm) -> _DataOwnerBuilder:
		"""Start building a DataOwner configured for a specific algorithm.

		Args:
			algorithm: Algorithm enum value.

		Returns:
			_DataOwnerBuilder: Builder instance for method chaining.
		"""
		return _DataOwnerBuilder().with_algorithm(algorithm)

	@staticmethod
	def with_scheme(scheme: Scheme) -> _DataOwnerBuilder:
		"""Start building a DataOwner configured for a specific scheme.

		Args:
			scheme: Scheme enum value.

		Returns:
			_DataOwnerBuilder: Builder instance for method chaining.
		"""
		return _DataOwnerBuilder().with_scheme(scheme)

	def outsourcedData(
		self,
		plaintext_matrix: npt.NDArray,
		threshold: float = -1,
		label_vector: Optional[npt.NDArray] = None,
		n_features: Optional[int] = None,
		**kwargs,
	) -> OutsourcedDataResult:
		"""Process plaintext data through the algorithm recipe.

		Follows the recipe steps associated with the configured
		algorithm: encryption, UDM/DM generation, FDHOPE keygen,
		threshold encryption, and weight/label initialisation.

		Args:
			plaintext_matrix: Plaintext data matrix.
			threshold: Clustering threshold (-1 auto-computes). Defaults to -1.
			label_vector: Optional label vector for supervised algorithms.
			n_features: Number of features for weight initialisation.

		Returns:
			ClientResult: Container with encrypted matrix, UDM, keys,
				threshold, and metadata.

		Raises:
			RuntimeError: If FDHOPE keygen is required but not initialised.
		"""
		plaintext_matrix = np.asarray(plaintext_matrix)
		algorithm = self._algorithm
		scheme    = self._scheme
		params    = self._scheme_params
		if plaintext_matrix.ndim == 0:
			raise ValueError("plaintext_matrix must be a vector or matrix")
		if plaintext_matrix.ndim == 1 and algorithm != Algorithm.NONE:
			raise ValueError(
				"One-dimensional input is supported only for Algorithm.NONE"
			)

		self._init_schemes(scheme, params)

		recipe = ALGORITHM_RECIPES.get(algorithm, [])

		Dshape = Utils.getShapeOfMatrix(plaintext_matrix)
		attributes = Dshape[1] if len(Dshape) > 1 else Dshape[0]

		encrypted_matrix      = np.array([])
		U                     = np.array([])
		udm_time              = 0.0
		encrypted_matrix_time = 0.0
		encrypted_weights     = np.array([])
		encrypted_bias        = np.array([])
		encrypted_labels      = np.array([])

		if algorithm == Algorithm.NONE and scheme != Scheme.NONE:
			start                 = time()
			encryption_result     = self._encrypt_plaintext(plaintext_matrix)
			encrypted_matrix      = self._cipher_data_to_array(encryption_result.data)
			encrypted_matrix_time = time() - start

		for step in recipe:
			if step == RecipeStep.ENCRYPT_DATASET:
				start                 = time()
				encryption_result     = self._encrypt_plaintext(plaintext_matrix)
				encrypted_matrix      = self._cipher_data_to_array(encryption_result.data)
				encrypted_matrix_time = time() - start

			elif step == RecipeStep.GENERATE_UDM:
				start    = time()
				U        = Utils.calculate_UDM(plaintext_matrix = plaintext_matrix)
				udm_time = time() - start

			elif step == RecipeStep.GENERATE_DM:
				start    = time()
				U        = Utils.calculate_DM(plaintext_matrix = plaintext_matrix)
				udm_time = time() - start

			elif step == RecipeStep.FDHOPE_KEYGEN:
				if self._fdhope is None:
					raise RuntimeError("FDHOPE scheme not initialized")
				self._messageIntervals, self._cypherIntervals = (
					self._fdhope.generate_keys(
						dataset                = U,
						minVal                 = self._params_int("fdhope_min_val", 0),
						max_range              = self._params_int("fdhope_max_range", 8),
						proportion             = self._params_int("fdhope_proportion", 15),
						range_limit            = self._params_int("fdhope_range_limit", 2),
						default_intervalLenght = self._params_float("fdhope_interval_length", 0.001),
					)
				)

			elif step == RecipeStep.ENCRYPT_U:
				U = self._encrypt_U(U, algorithm)

			elif step == RecipeStep.GENERATE_THRESHOLD:
				if threshold == -1:
					self._threshold = Utils.get_threshold(distance_matrix=U)
				else:
					self._threshold = threshold

			elif step == RecipeStep.ENCRYPT_THRESHOLD:
				self._encrypted_threshold = self._fdhope._encrypt(
					plaintext    = self._threshold,
					messagespace = self._messageIntervals,
					cipherspace  = self._cypherIntervals,
				)

			elif step == RecipeStep.INIT_WEIGHTS:
				if n_features is None:
					n_features = attributes
				plaintext_weights = np.zeros(n_features)

			elif step == RecipeStep.INIT_BIAS:
				plaintext_bias = np.zeros((1,))

			elif step == RecipeStep.ENCRYPT_WEIGHTS:
				encrypted_weights = self.primary_scheme.encrypt_vector(
					plaintext_vector=plaintext_weights
				)

			elif step == RecipeStep.ENCRYPT_BIAS:
				encrypted_bias = self.primary_scheme.encrypt_vector(
					plaintext_vector=plaintext_bias
				)

			elif step == RecipeStep.ENCRYPT_LABELS:
				if label_vector is not None:
					encrypted_labels = np.array(
						self.primary_scheme.encrypt_vector(
							plaintext_vector=label_vector
						).data
					)

		return OutsourcedDataResult(
			UDM                   = U,
			udm_time              = udm_time,
			encrypted_matrix      = encrypted_matrix,
			encrypted_matrix_time = encrypted_matrix_time,
			messageIntervals      = self._messageIntervals,
			cypherIntervals       = self._cypherIntervals,
			encrypted_threshold   = self._encrypted_threshold,
			num_attributes        = attributes,
			encrypted_weights     = encrypted_weights,
			encrypted_bias        = encrypted_bias,
			encrypted_labels      = encrypted_labels,
		)

	def _encrypt_plaintext(self, plaintext: npt.NDArray):
		"""Encrypt a vector or matrix through the configured primary scheme."""
		if plaintext.ndim == 1:
			return self.primary_scheme.encrypt_vector(plaintext_vector=plaintext)
		return self.primary_scheme.encrypt_matrix(plaintext_matrix=plaintext)

	@staticmethod
	def _cipher_data_to_array(data) -> npt.NDArray:
		"""Normalize scalar and collection cipher results for ClientResult."""
		if isinstance(data, np.ndarray):
			return data
		if isinstance(data, (list, tuple)):
			return np.array(data)
		result = np.empty(1, dtype=object)
		result[0] = data
		return result

	def _init_schemes(self, scheme: Scheme, params):
		"""Lazily initialise cryptographic schemes from parameters.

		Calls params.create_scheme() and unpacks the result into
		primary_scheme and optional FDHOPE/sens attributes.

		Args:
			scheme: Scheme enum value.
			params: SchemeParams dataclass instance.

		Raises:
			ValueError: If params is None and the scheme is not NONE.
		"""
		if hasattr(self, '_schemes_initialized') and self._schemes_initialized:
			return

		if scheme == Scheme.NONE:
			self._schemes_initialized = True
			return

		if params is None:
			raise ValueError(
				f"scheme_params is required for scheme {scheme.value}"
			)

		schemes = params.create_scheme()

		if _is_fdhope_scheme(scheme):
			self.primary_scheme, self._fdhope = schemes
			print("FDHOPE scheme initialized:", schemes)
			self._sens                = self._params_float("sens", 0.00001)
			self._fdhope.messagespace = {}
			self._fdhope.cipherspace  = {}
		else:
			self.primary_scheme = schemes
			self._fdhope = None
			self._sens = None

		self._schemes_initialized = True

	def _params_float(self, name: str, default: float) -> float:
		"""Get a float parameter from scheme_params with a fallback default.

		Args:
			name: Attribute name on the SchemeParams instance.
			default: Fallback value if the attribute is absent.

		Returns:
			float: Parameter value.
		"""
		if self._scheme_params is not None and hasattr(self._scheme_params, name):
			return float(getattr(self._scheme_params, name))
		return default

	def _params_int(self, name: str, default: int) -> int:
		"""Get an integer parameter from scheme_params with a fallback default.

		Args:
			name: Attribute name on the SchemeParams instance.
			default: Fallback value if the attribute is absent.

		Returns:
			int: Parameter value.
		"""
		if self._scheme_params is not None and hasattr(self._scheme_params, name):
			return int(getattr(self._scheme_params, name))
		return default

	def _encrypt_U(self, U: npt.NDArray, algorithm: Algorithm) -> npt.NDArray:
		"""Encrypt the UDM using FDHOPE order-preserving encryption.

		Encrypts the lower-triangular portion of the distance matrix
		and mirrors values across the diagonal. Handles 3D tensors
		for DBSKMeans and 2D matrices for DBSNNC.

		Args:
			U: Updatable Distance Matrix or distance matrix.
			algorithm: Algorithm enum value determining the UDM structure.

		Returns:
			npt.NDArray: FDHOPE-encrypted matrix.
		"""
		Ushape = Utils.getShapeOfMatrix(U)
		sens = self._sens if self._sens is not None else 0.00001

		for x in range(Ushape[0]):
			for y in range(x):
				if _is_dbskmeans_algorithm(algorithm):
					for z in range(Ushape[2]):
						U[x][y][z] = self._fdhope._encrypt(
							plaintext    = U[x][y][z],
							sens         = sens,
							messagespace = self._messageIntervals,
							cipherspace  = self._cypherIntervals,
						)
						U[y][x][z] = U[x][y][z]
				elif _is_dbsnnc_algorithm(algorithm):
					U[x][y] = self._fdhope._encrypt(
						plaintext    = U[x][y],
						messagespace = self._messageIntervals,
						cipherspace  = self._cypherIntervals,
					)
					U[y][x] = U[x][y]
		return np.array(U)
