import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Union, Optional, TYPE_CHECKING
from option import Result,Ok,Err
from rory.core.security.cryptosystem.liu import Liu
if TYPE_CHECKING:
	from rory.core.algorithms import ConventionalClustering, PqcClustering
	from rory.core.security.cryptosystem.pqc.ckks import Ckks
	from Pyfhel import PyCtxt
from rory.core.utils.deprecation import deprecated
from Pyfhel import PyCtxt, Pyfhel, PyPtxt
from rory.core.utils.constants import Constants

class Utils:
	"""
	A utility class providing common helper functions for matrix operations, clustering support, and error verification.
	"""
	
	@staticmethod
	def get_threshold(distance_matrix:npt.NDArray) -> float:
		"""
		Computes the threshold value from a given distance matrix.

		This function filters out the zero entries from the distance matrix and returns the minimum non-zero
		distance as the threshold. This threshold can be used for various purposes, such as determining the
		minimal separation between distinct clusters.

		Args:
			distance_matrix (npt.NDArray): A NumPy array representing pairwise distances between elements.

		Returns:
			float: The smallest non-zero value in the distance matrix, which serves as the threshold.
		"""
		matrix_without_zeros = distance_matrix[distance_matrix != 0]
		threshold = np.min(matrix_without_zeros)
		return threshold
	
	@staticmethod
	def get_labelvector_from_indexes(shape:int,c_indexes:List[List[int]]) -> List[int]:
		"""
		Generates a label vector from the provided cluster indexes.

		This function creates a label vector of a specified size (shape).

		Args:
			shape (int): The total number of records, which determines the size of the label vector.
			c_indexes (List[List[int]]): A list of clusters, with each cluster containing record indices.

		Returns:
			List[int]: A label vector with the cluster assignment for each record.
		"""
		label_vector = [-1]*shape #fill label vector with -1
		
		for index,record_index in enumerate(c_indexes):
			for index_r,record in enumerate(record_index):
				label_vector[record] = index
		return label_vector
	
	@staticmethod
	def split_labelvector_from_data(dataset: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
		"""Split the last column of a dataset as labels.

		Args:
			dataset: Array where the last column contains labels.

		Returns:
			Tuple of (data, labels) as NDArrays.
		"""
		data   = dataset[:, :-1]
		labels = dataset[:, -1]
		return data, labels

	@staticmethod
	def get_label_vector(
		model_labels: npt.NDArray,
		min_indexes: npt.NDArray
	) -> "np.ndarray":
		"""Map minimum-distance indexes to their corresponding labels.

		Args:
			model_labels: Array of labels for the model records.
			min_indexes: Array of indexes pointing to the nearest model record.

		Returns:
			NDArray of predicted labels.
		"""
		import numpy as np
		label_vector = [model_labels[index] for index in min_indexes]
		return np.array(label_vector)
	
	@staticmethod
	def getMinDistanceInClusters(
		c_indexes: List[List[int]],
		record_index: int,
		distance_matrix: npt.NDArray
	) -> Tuple[int, float]:
		"""Determine the cluster with the minimum distance to a record.

		Uses numpy indexing for performance instead of per-element
		Python loops.

		Args:
			c_indexes: List of clusters (each a list of record indices).
			record_index: Index of the record being evaluated.
			distance_matrix: Pairwise distances between records.

		Returns:
			Tuple of (cluster_index, minimum_distance).
		"""
		cluster_m_index = -1
		delta = None
		for current_cluster_index, cluster_k in enumerate(c_indexes):
			delta_min = np.min(distance_matrix[record_index, cluster_k])
			if (delta is None) or (delta_min < delta):
				delta = delta_min
				cluster_m_index = current_cluster_index
		return cluster_m_index, delta

	@staticmethod
	def generate_centroids(k:int, plain_matrix:npt.NDArray) -> npt.NDArray:
		"""
		Generates initial centroids from a plaintext matrix.

		This function selects the first k records from the plaintext matrix to serve as the initial centroids.

		Args:
			k (int): The number of centroids (clusters) to generate.
			plain_matrix (npt.NDArray): The plaintext dataset, where each row represents a record.

		Returns:
			npt.NDArray: A NumPy array of shape (k, columns) containing the initial centroids.
		"""
		centroids    = []
		for x in range(k):
			centroids.append(plain_matrix[x])
		columns = Utils.getShapeOfMatrix(plain_matrix)[1]
		return np.array(centroids).reshape(k, columns)
	
	
	@staticmethod
	def getShapeOfMatrix(xs: npt.NDArray) -> Tuple[int, ...]:
		"""
		Returns the shape of a matrix-like object.

		This function checks if the input is a NumPy ndarray and returns its shape directly.
		If the input is not a NumPy array, it converts it into one and then returns its shape.
		
		Args:
			xs: A matrix-like object (e.g., a list of lists or a NumPy array).

		Returns:
			tuple: A tuple representing the dimensions (shape) of the matrix.
		"""
		if hasattr(xs, 'shape'):
			return xs.shape
		return np.array(xs).shape

	@staticmethod
	def verifyZero(S) -> bool:
		"""
		Checks whether all elements in the shift matrix S are zero.

		Args:
			S: The shift matrix to be checked.

		Returns:
			bool: True if all elements in S are zero, False otherwise.
		"""
		return np.all(np.asarray(S) == 0)

	@staticmethod
	def fillLabelVector(label_vector:List[int]=[],k=2) -> List[int]:
		"""
		Adds initial cluster labels to the provided label vector.

		This function generates a list of labels from 0 to k-1 and then concatenates it 
		with the given label_vector. It is useful for ensuring that the label vector starts 
		with the initial cluster labels.

		Args:
			label_vector (List[int], optional): A list of existing labels. Defaults to an empty list.
			k (int, optional): The number of clusters.

		Returns:
			List[int]: The resulting label vector with initial labels from 0 to k-1 prepended.
		"""
		return list(range(k)) + label_vector

	@staticmethod
	def calculateSimilarity(UDM:npt.NDArray,limit:int,sim:float, xy:Tuple[int,int])->float:
		"""
		Calculates the similarity measure between two records using the UDM matrix.

		This function iterates over the first 'limit' dimensions (attributes) of the UDM matrix for the given 
		record pair (x, y).

		Args:
			UDM (npt.NDArray): A 3D matrix representing pairwise differences between records for each attribute.
			limit (int): The number of dimensions (attributes) to consider in the similarity calculation.
			sim (float): The initial similarity value, which is incremented by the differences.
			xy (Tuple[int, int]): A tuple containing the indices (x, y) of the two records to compare.

		Returns:
			float: The computed similarity measure between the two records.
		"""
		x,y  = xy
		return sim + float(np.sum(np.abs(np.asarray(UDM[x][y][:limit]))))
	
	@staticmethod
	def compute_centroid_shift_liu(
		previous_centroids: npt.NDArray,
		current_centroids: npt.NDArray,
	) -> npt.NDArray:
		"""
		Compute encrypted shift between two centroid sets using Liu's subtraction.

		Args:
			previous_centroids: Centroids from previous iteration.
			current_centroids: Centroids from current iteration.

		Returns:
			npt.NDArray: Encrypted shift matrix.
		"""
		S1 = []
		for i in range(len(previous_centroids)):
			row = []
			for j in range(len(previous_centroids[i])):
				row.append(Liu.subtract(
					ciphertext_1=previous_centroids[i][j],
					ciphertext_2=current_centroids[i][j],
				))
			S1.append(row)
		return np.array(S1)

	@staticmethod
	def _populate_clusters(
		record_id: int,
		UDM: npt.NDArray,
		num_clusters: int,
		num_attributes: int,
		ciphertext_matrix: npt.NDArray,
		append_fn,
		clusters: list,
	) -> Result[Tuple[list, List[int]], Exception]:
		"""
		Core cluster assignment: assigns each record from record_id onward to the
		nearest cluster based on the UDM, using the provided append_fn callback.

		Args:
			record_id: Starting record index.
			UDM: Updatable distance matrix.
			num_clusters: Number of clusters.
			num_attributes: Number of attributes per record.
			ciphertext_matrix: Encrypted dataset.
			append_fn: Callable(clusters, cluster_idx, record) to add a record.
			clusters: Cluster container (list-based or PyCtxt-based).

		Returns:
			Ok((clusters, label_vector)) or Err(exception).
		"""
		try:
			UDM_arr = np.asarray(UDM)
			label_vector: List[int] = []
			for x in range(record_id, len(ciphertext_matrix)):
				sim1 = []
				for y in range(num_clusters):
					if y > x:
						sim = Utils.calculateSimilarity(
							UDM=UDM_arr, limit=num_attributes, sim=0, xy=(y, x),
						)
					else:
						sim = Utils.calculateSimilarity(
							UDM=UDM_arr, limit=num_attributes, sim=0, xy=(x, y),
						)
					sim1.append(sim)
				min_index = sim1.index(min(sim1))
				label_vector.append(min_index)
				append_fn(clusters, min_index, ciphertext_matrix[x])
			return Ok((clusters, label_vector))
		except Exception as e:
			return Err(e)

	
	@deprecated(replacement="_populate_clusters()")
	@staticmethod
	def populateClusters(record_id:int,UDM:npt.NDArray, clusters: List[List[int]], ciphertext_matrix:npt.NDArray)->Result[Tuple[List[List[int]],List[int]], Exception]:
		"""
		Assigns the remaining records of the encrypted dataset (D1) to clusters.

		Args:
			record_id: Starting record index (typically k).
			UDM: Updatable distance matrix.
			clusters: Current set of clusters.
			ciphertext_matrix: Encrypted dataset.

		Returns:
			Ok((clusters, label_vector)) or Err(exception).
		"""
		shape = Utils.getShapeOfMatrix(ciphertext_matrix)
		return Utils._populate_clusters(
			record_id=record_id,
			UDM=UDM,
			num_clusters=len(clusters),
			num_attributes=shape[1],
			ciphertext_matrix=ciphertext_matrix,
			append_fn=lambda cl, idx, rec: cl[idx].append(rec.tolist()),
			clusters=clusters,
		)

	@deprecated(replacement="_populate_clusters()")
	@staticmethod
	def populateClustersObject(record_id:int, UDM:npt.NDArray,clusters: List[PyCtxt], ciphertext_matrix:npt.NDArray, num_attributes: int)->Result[
		Tuple[List[PyCtxt],List[int]], Exception]:
		"""
		Assigns the remaining records of the encrypted dataset (D1) to clusters
		represented as PyCtxt objects.

		Args:
			record_id: Starting record index.
			UDM: Updatable distance matrix.
			clusters: List of clusters (each as PyCtxt).
			ciphertext_matrix: Encrypted dataset.
			num_attributes: Number of attributes.

		Returns:
			Ok((clusters, label_vector)) or Err(exception).
		"""
		return Utils._populate_clusters(
			record_id=record_id,
			UDM=UDM,
			num_clusters=len(clusters),
			num_attributes=num_attributes,
			ciphertext_matrix=ciphertext_matrix,
			append_fn=lambda cl, idx, rec: cl[idx].append(rec),
			clusters=clusters,
		)
	

	@staticmethod
	def calculateCentroids(clusters:List[int], k:int, attributes:int,m:int)->Result[npt.NDArray,Exception]:
		"""
		Generates a new set of centroids from the given clusters using homomorphic operations.

		This function computes the centroid for each cluster by calculating the homomorphic average of the 
		records in that cluster. The centroids are represented as a 3D ndarray with dimensions (k, attributes, m), where:
		- k is the number of clusters.
		- attributes is the number of attributes of the dataset (D1).
		- m is the number of attributes of the secret key (SK).

		For each cluster:
		- If the cluster is empty, its centroid is set to a zero matrix.
		- Otherwise, the function sums the records in the cluster using a homomorphic addition (via Liu.add) 
			and then multiplies the sum by 1/(number of records) using homomorphic multiplication (via Liu.multiply_c) 
			to compute the average.

		Args:
			clusters (List[int]): The set of clusters, where each cluster is a list of records (each record is a ciphertext).
			k (int): The number of clusters.
			attributes (int): The number of attributes in the dataset D1.
			m (int): The number of attributes of the secret key (SK).

		Returns:
			Result[npt.NDArray, Exception]:
				On success, returns an Ok result containing the ndarray of new centroids.
				On failure, returns an Err containing the encountered Exception.
		"""
		try:
			cent = np.zeros((k, attributes, m))
			for j in range(k):
				average = np.zeros((attributes, m))
				cjLen = len(clusters[j])
				if cjLen == 0:
					cent[j] = np.zeros((attributes, m))
				else:
					for i in range(cjLen):
						rec1 = clusters[j][i]
						for q in range(attributes):
							average[q] = Liu.add(
								ciphertext_1=average[q],
								ciphertext_2=rec1[q],
							)
					for q in range(attributes):
						cent[j][q] = Liu.multiply_c(
							scalar=1 / cjLen,
							ciphertext=average[q],
						)
			return Ok(cent)
		except Exception as e:
			return Err(e)
		
	def distance(matrix: npt.NDArray, mode:str= "diff") -> npt.NDArray:
		"""
		Calculates a pairwise distance matrix from the input matrix.

		Args:
			matrix (np.ndarray): A NumPy array with shape (n, a) representing the dataset, where n is the number of records and a is the number of attributes.
			mode (str, optional): The mode of difference calculation. 
				- "diff": Returns the raw differences.
				- "diff-abs": Returns the absolute differences.
				Defaults to "diff".

		Returns:
			np.ndarray: A 3D NumPy array with shape (n, n, a) containing the pairwise differences.
		"""
		matrix.shape[0]
		expanded_matrix  = np.expand_dims(matrix, axis=1)  # Shape: (n, 1, a)
		if mode =="diff":
			result = expanded_matrix - matrix  # Shape: (n, n, a)
		elif mode =="diff-abs":
			result = np.abs(expanded_matrix - matrix)  # Shape: (n, n, a)
		else:
			result  = expanded_matrix - matrix  # Shape: (n, n, a)
		return result

	def empty_cluster(k:int=3) -> List[List[int]]:
		"""
		Creates a list of empty clusters.

		This function generates a list containing k empty lists, where k represents the number of clusters.
		It is typically used to initialize cluster structures before assigning records to clusters.

		Keyword Args:
			k (int): The number of clusters to initialize.

		Returns:
			List[List[Any]]: A list of k empty lists.
		"""
		return [[] for _ in range(k)]
	@staticmethod
	def compute_mean_relative_error(
		old: npt.NDArray[np.floating],
		new: npt.NDArray[np.floating],
		eps: float = 0.0
	) -> float:
		"""
		Compute the Mean Relative Error (MRE) between two arrays in a numerically robust way.

		The element-wise relative error is:
			r_i = |(old_i - new_i) / denom_i|
		where
			denom_i = old_i               if |old_i| >= eps,
					eps * sign(old_i)    if 0 < |old_i| < eps,
					eps                  if old_i == 0 (treat zero-denominator as eps).

		We then:
		- treat old==0 and new==0 entries as zero error,
		- clamp very small |old| to eps to avoid blowing up the ratio,
		- mask out any infinite or NaN ratios before averaging,
		- if no valid entries remain (i.e. old all zero but new ≠ 0 somewhere), return +∞.

		Args:
			old:  original array (denominator for relative error)
			new:  new/updated array
			eps:  small smoothing constant; if >0, any |old_i|<eps is treated as eps

		Returns:
			The mean of the finite, absolute relative errors, or +∞ if all errors are invalid.
		"""
		# 1) Build denominator, clamping very small old values to ±eps
		if eps > 0:
			denom = np.where(np.abs(old) < eps, eps * np.sign(old), old)
		else:
			denom = old

		# 2) Compute raw ratio with warnings suppressed
		with np.errstate(divide='ignore', invalid='ignore'):
			ratio = (old - new) / denom

		# 3) Absolute value of the ratio
		abs_ratio = np.abs(ratio)

		# 4) Treat 0→0 as zero error
		mask_zero_zero = (old == 0) & (new == 0)
		abs_ratio[mask_zero_zero] = 0.0

		# 5) Mask out any non-finite entries (old==0 & new!=0)
		valid = np.isfinite(abs_ratio)
		if not np.any(valid):
			# No valid entries—error is effectively infinite
			return np.inf

		# 6) Mean over valid entries
		return float(np.mean(abs_ratio[valid]))
	
	@staticmethod
	def verify_mean_error(
		old_matrix: npt.NDArray,
		new_matrix: npt.NDArray,
		min_error: float = 0.15,
		eps: float = 0.0
	) -> bool:
		"""
		Check if the mean relative error between two arrays is within a given threshold.

		Args:
			old:             original array
			new:             new/updated array
			max_mean_error:  maximum allowed mean relative error
			eps:             small smoothing constant (see compute_mean_relative_error)

		Returns:
			True if MRE <= max_mean_error, False otherwise.
		"""
		mre = Utils.compute_mean_relative_error(old_matrix, new_matrix, eps=eps)
		return (mre <= min_error)

	@staticmethod
	def get_scale(ct: PyCtxt) -> Optional[float]:
		"""Safely extracts the scale parameter from a ciphertext object.

        Attempts to retrieve the scale directly from the object's attribute,
        falling back to its native getter method if an exception occurs.

        Args:
            ct (PyCtxt): The ciphertext object from which to extract the scale.

        Returns:
            Optional[float]: The scale value as a float if successfully 
                retrieved; otherwise, None.
        """
		try:
			return float(ct.scale)
		except Exception:
			try:
				return float(ct.get_scale())
			except Exception:
				return None

	@staticmethod
	def ptxt_from_scalar(HE: Pyfhel, val: float, n_features: int, scale: int) -> PyPtxt:
		"""Encodes a scalar value into a plaintext repeated n times.

        Creates a one-dimensional array filled with the specified scalar value
        matching the number of features, then encodes it using Pyfhel's 
        fractional encoding scheme.

        Args:
            HE (Pyfhel): The homomorphic encryption context instance.
            val (float): The scalar value to replicate across the array.
            n_features (int): The size of the array (number of features).
            scale (int): The scale parameter used for encoding.

        Returns:
            PyPtxt: The resulting plaintext object containing the encoded data.
        """
		arr = np.full(n_features, float(val), dtype=np.float64)
		return HE.encodeFrac(arr, scale=scale)

	@staticmethod
	def relinearize_if_possible(HE: Pyfhel, ct: PyCtxt) -> PyCtxt:
		"""Applies relinearization on a ciphertext if the execution state permits.

        Attempts to perform the relinearization operation within the Pyfhel context
        to reduce ciphertext size after a multiplication. If the operation fails 
        due to parameter or key constraints, the exception is caught and the 
        ciphertext is returned unmodified.

        Args:
            HE (Pyfhel): The homomorphic encryption context instance.
            ct (PyCtxt): The ciphertext to relinearize.

        Returns:
            PyCtxt: The resulting ciphertext, either relinearized or unaltered.
        """
		try:
			HE.relinearize(ct)
		except Exception:
			pass
		return ct

	@staticmethod
	def try_rescale_next(HE: Pyfhel, ct: PyCtxt) -> PyCtxt:
		"""Attempts to rescale a ciphertext to the next level in the modulus chain.

        Safely executes Pyfhel's downward rescaling operation to manage noise 
        growth. If the constraints of the current level or modulus chain prevent 
        rescaling, the exception is bypassed.

        Args:
            HE (Pyfhel): The homomorphic encryption context instance.
            ct (PyCtxt): The ciphertext whose modulus and scale are to be reduced.

        Returns:
            PyCtxt: The ciphertext at the new level in the chain, or unaltered.
        """
		try:
			HE.rescale_to_next(ct)
		except Exception:
			pass
		return ct

	@staticmethod
	def align(HE: Pyfhel, a: Union[PyCtxt, PyPtxt], b: Union[PyCtxt, PyPtxt], only_mod: bool = False) -> tuple[PyCtxt, PyPtxt]:
		"""Aligns the modulus chain levels and scales of two encrypted or encoded elements.

        Generates deep copies of both input elements and aligns them into a 
        compatible state to enable subsequent mathematical operations between them.

        Args:
            HE (Pyfhel): The homomorphic encryption context instance.
            a (Union[PyCtxt, PyPtxt]): The first element (ciphertext or plaintext) to align.
            b (Union[PyCtxt, PyPtxt]): The second element (ciphertext or plaintext) to align.
            only_mod (bool, optional): If True, the alignment is strictly restricted 
                to the modulus, skipping scale adjustment. Defaults to False.

        Returns:
            Tuple[Union[PyCtxt, PyPtxt], Union[PyCtxt, PyPtxt]]: A tuple containing 
                both aligned elements ready for operations.
        """
		a_al, b_al = HE.align_mod_n_scale(
			a, b, copy_this=True, copy_other=True, only_mod=only_mod
		)
		return a_al, b_al

	@staticmethod
	def rebind_ct(ct: PyCtxt) -> PyCtxt:
		"""Creates a deep copy of a ciphertext object to handle independent references.

        Args:
            ct (PyCtxt): The ciphertext object to be copied.

        Returns:
            PyCtxt: A new, independent instance of the ciphertext containing the same data.
        """
		return ct.copy()

	@staticmethod
	def normalize_scale(HE: Pyfhel, ct: PyCtxt, scale: int, ratio: float = 1.25) -> PyCtxt:
		"""Iteratively rescales a ciphertext down until its scale stabilizes within a target range.

        Checks the current scale against the target scale multiplied by a specific 
        tolerance ratio. It continuously triggers downward rescaling as long as the 
        scale exceeds the threshold and continues to actively change.

        Args:
            HE (Pyfhel): The homomorphic encryption context instance.
            ct (PyCtxt): The ciphertext whose scale needs to be normalized.
            scale (int): The target base scale value.
            ratio (float, optional): The upper bound tolerance ratio for scale comparison. 
                Defaults to 1.25.

        Returns:
            PyCtxt: The ciphertext with its scale normalized down as close to the target 
                as parameters permit.
        """
		current_scale = Utils.get_scale(ct)
		if current_scale is None:
			return ct
		
		changed = True
		while current_scale > ratio * scale and changed:
			previous_scale = current_scale
			changed = False
			try:
				HE.rescale_to_next(ct)
				current_scale = Utils.get_scale(ct)
				if current_scale is not None and current_scale < previous_scale:
					changed = True
			except Exception:
				break
		return ct

	@staticmethod
	def safe_add(HE: Pyfhel, a: PyCtxt, b: PyCtxt) -> PyCtxt:
		"""Safely performs homomorphic addition after aligning the modulus levels and scales.

        Compares the modulus levels of both ciphertexts, performing modulus switching 
        steps on the one with a higher level until they match. It then forces the scales 
        to match before executing the addition to prevent cryptographic mismatches.

        Args:
            HE (Pyfhel): The homomorphic encryption context instance.
            a (PyCtxt): The first encrypted operand.
            b (PyCtxt): The second encrypted operand.

        Raises:
            RuntimeError: If the alignment or the underlying homomorphic addition fails.

        Returns:
            PyCtxt: A new ciphertext containing the encrypted sum of the operands.
        """
		lvl_a = a.mod_level
		lvl_b = b.mod_level

		if lvl_a > lvl_b:
			for _ in range(lvl_a - lvl_b):
				HE.mod_switch_to_next(b)
		elif lvl_b > lvl_a:
			for _ in range(lvl_b - lvl_a):
				HE.mod_switch_to_next(a)

		b.scale = a.scale

		try:
			return HE.add(a, b, in_new_ctxt=True)
		except Exception as e:
			raise RuntimeError(f"Error en suma: {e}")

	@staticmethod
	def safe_sub(HE: Pyfhel, a: PyCtxt, b: PyCtxt) -> PyCtxt:
		"""Safely performs homomorphic subtraction after aligning the modulus levels and scales.

        Compares the modulus levels of both ciphertexts, forcing modulus switching 
        downward on the operand with the higher level until they reach parity. It then 
        unifies their scale attributes and executes the subtraction safely.

        Args:
            HE (Pyfhel): The homomorphic encryption context instance.
            a (PyCtxt): The encrypted minuend.
            b (PyCtxt): The encrypted subtrahend.

        Raises:
            RuntimeError: If the alignment or the underlying homomorphic subtraction fails.

        Returns:
            PyCtxt: A new ciphertext containing the encrypted result of the subtraction.
        """
		lvl_a = a.mod_level
		lvl_b = b.mod_level

		if lvl_a < lvl_b:
			for _ in range(lvl_b - lvl_a):
				HE.mod_switch_to_next(a)
		elif lvl_b < lvl_a:
			for _ in range(lvl_a - lvl_b):
				HE.mod_switch_to_next(b)

		b.scale = a.scale

		try:
			return HE.sub(a, b, in_new_ctxt=True)
		except Exception as e:
			raise RuntimeError(f"Error en resta: {e}")

	@staticmethod
	def safe_multiply(HE: Pyfhel, a: PyCtxt, b: PyCtxt, scale: int) -> PyCtxt:
		"""Safely executes homomorphic multiplication, including relinearization and scale management.

        Aligns the modulus level of the two target ciphertexts, multiplies them into 
        a new context, relinearizes the resulting cipher text to minimize its size, 
        and drops down one modulus level. Finally, it uses scale normalization to 
        safeguard against excessive noise expansion.

        Args:
            HE (Pyfhel): The homomorphic encryption context instance.
            a (PyCtxt): The first encrypted multiplier.
            b (PyCtxt): The second encrypted multiplicand.
            scale (int): The target base scale for the resulting normalization.

        Raises:
            RuntimeError: If any sequence of the multiplication, relinearization, 
                or rescaling pipeline fails.

        Returns:
            PyCtxt: A managed, relinearized, and normalized ciphertext containing 
                the encrypted product.
        """
		lvl_a = a.mod_level
		lvl_b = b.mod_level

		if lvl_a > lvl_b:
			for _ in range(lvl_a - lvl_b):
				HE.mod_switch_to_next(b)
		elif lvl_b > lvl_a:
			for _ in range(lvl_b - lvl_a):
				HE.mod_switch_to_next(a)    

		try:
			res = HE.multiply(a, b, in_new_ctxt=True)
			HE.relinearize(res)
			HE.rescale_to_next(res)
			return Utils.normalize_scale(HE=HE, ct=res, scale=scale)
			
		except Exception as e:
			raise RuntimeError(f"Critical error in multiplication: {e}")

	@staticmethod
	def mul_plain_scalar(HE: Pyfhel, ct: PyCtxt, scalar: float, scale: int, n_features: int) -> PyCtxt:
		"""Multiplies an encrypted ciphertext by an unencrypted plaintext scalar.

        Extracts the exact current scale of the ciphertext to encode the scalar,
        aligns only the modulus level to bypass potential scale mismatch errors,
        performs plaintext multiplication, drops one modulus level via rescaling,
        and normalizes the final scale.

        Args:
            HE (Pyfhel): The homomorphic encryption context instance.
            ct (PyCtxt): The ciphertext multiplier operand.
            scalar (float): The scalar value to encode and multiply.
            scale (int): The target base scale for final normalization.
            n_features (int): The length of the underlying slot array.

        Raises:
            RuntimeError: If modulus alignment, multiplication, or subsequent 
                rescaling steps fail.

        Returns:
            PyCtxt: The rescaled and normalized ciphertext product.
        """
		exact_scale = ct.scale
		pt_scalar = Utils.ptxt_from_scalar(HE=HE, val=scalar, n_features=n_features, scale=exact_scale)
		try:
			_, pt_scalar_aligned = Utils.align(HE=HE, a=ct, b=pt_scalar, only_mod=True)
			
			res = HE.multiply_plain(ct, pt_scalar_aligned, in_new_ctxt=True)
			HE.rescale_to_next(res)
			return Utils.normalize_scale(HE=HE, ct=res, scale=scale)
		except Exception as e:
			raise RuntimeError(f"Plaintext scalar multiplication error: {e}")

	@staticmethod
	def add_plain_scalar(HE: Pyfhel, ct: PyCtxt, scalar: float, n_features: int, scale: int) -> PyCtxt:
		"""Adds an unencrypted plaintext scalar to a ciphertext.

        Manually extracts the exact current scale of the ciphertext to encode the 
        scalar, aligns only the modulus level to bypass potential 'round_scale' 
        crashes, and computes the homomorphic addition.

        Args:
            HE (Pyfhel): The homomorphic encryption context instance.
            ct (PyCtxt): The ciphertext augend operand.
            scalar (float): The scalar value to encode and add.
            n_features (int): The length of the underlying slot array.
            scale (int): The target scale (unused directly, as ct.scale is preferred).

        Raises:
            RuntimeError: If modulus alignment or plaintext addition fails.

        Returns:
            PyCtxt: A new ciphertext containing the sum of the operands.
        """
		exact_scale = ct.scale 
		pt_scalar = Utils.ptxt_from_scalar(HE=HE, val=scalar, n_features=n_features, scale=exact_scale)
		
		try:
			_, pt_scalar_aligned = Utils.align(HE=HE, a=ct, b=pt_scalar, only_mod=True)

			return HE.add_plain(ct, pt_scalar_aligned, in_new_ctxt=True)
		except Exception as e:
			raise RuntimeError(f"Plaintext scalar addition error: {e}")

	@staticmethod
	def dot_cipher_garbage(HE: Pyfhel, x_ct: PyCtxt, w_ct: PyCtxt, n_features: int, scale: float) -> PyCtxt:
		"""Computes a dot product between two ciphertexts.

        Executes element-wise multiplication followed by log-step left rotations 
        and additions to accumulate the sum across all slots.

        Args:
            HE (Pyfhel): The homomorphic encryption context instance.
            x_ct (PyCtxt): The ciphertext representing the input feature vector.
            w_ct (PyCtxt): The ciphertext representing the weight vector.
            n_features (int): The length of the vector (bounds the active rotation steps).
            scale (float): The target base scale parameter for tracking and normalization.

        Raises:
            RuntimeError: If element-wise multiplication or rotation/accumulation
                encounters a cryptographic error.

        Returns:
            PyCtxt: A ciphertext containing the dot product total replicated across
                all slots.
        """
		elementwise_product = Utils.safe_multiply(HE, x_ct, w_ct, scale=int(scale))
		accumulator = elementwise_product.copy()
		del elementwise_product

		step = 1
		while step < n_features:
			try:
				rotated = HE.rotate(accumulator, step, in_new_ctxt=True)
				old_accumulator = accumulator
				accumulator = Utils.safe_add(HE=HE, a=accumulator, b=rotated)
				del rotated, old_accumulator
			except Exception as e:
				raise RuntimeError(f"Error occurred while rotating left: {e}")
			step <<= 1

		return Utils.normalize_scale(HE=HE, ct=accumulator, scale=int(scale))

	@staticmethod
	def execute_plaintext_phase(
		k: int,
		udm: npt.NDArray,
		num_attributes: int,
		shift_matrix: npt.NDArray,
	) -> npt.NDArray:
		"""Apply decrypted shift values to update the UDM via numpy broadcasting.

		Updates the Updatable Distance Matrix (UDM) element-wise by
		incorporating decrypted centroid shift values. Shared by both
		conventional (Liu) and PQC (CKKS) secure KMeans variants.

		Args:
			k: Number of clusters.
			udm: Current UDM matrix.
			num_attributes: Number of attributes per record.
			shift_matrix: Decrypted shift values.

		Returns:
			Updated UDM as npt.NDArray with shape (n, k, num_attributes).
		"""
		U = np.asarray(udm)
		S = np.asarray(shift_matrix)
		n = U.shape[0]
		result = np.empty_like(U, shape=(n, k, num_attributes))
		for y in range(k):
			result[y:, y, :] = U[y:, y, :] + S[y, :]
			if y > 0:
				result[:y, y, :] = -U[y, :y, :] + S[y, :]
		return result

	@staticmethod
	def calculate_UDM(plaintext_matrix: npt.NDArray, mode: str = "diff") -> npt.NDArray:
		"""Compute the Updatable Distance Matrix (pairwise record differences).

		Expands the input matrix to shape (n, 1, a) and subtracts the
		original to produce an (n, n, a) tensor of pairwise differences.

		Args:
			plaintext_matrix: Numeric dataset with shape (n, a), where n is
				the number of records and a is the number of attributes.
			mode: "diff" for raw difference, "diff-abs" for absolute
				difference. Defaults to "diff".

		Returns:
			npt.NDArray with shape (n, n, a) containing pairwise differences.
		"""
		expanded = np.expand_dims(plaintext_matrix, axis=1)
		if mode == "diff-abs":
			return np.abs(expanded - plaintext_matrix)
		return expanded - plaintext_matrix

	@staticmethod
	def calculate_DM(plaintext_matrix: npt.NDArray) -> npt.NDArray:
		"""Compute the Distance Matrix (Manhattan distance between records).

		Args:
			plaintext_matrix: Numeric dataset with shape (n, a).

		Returns:
			npt.NDArray with shape (n, n) where entry [x][y] is the
			Manhattan distance between the x-th and y-th records.
		"""
		plaintext_matrix = np.asarray(plaintext_matrix)
		n = plaintext_matrix.shape[0]
		result = np.zeros((n, n))
		for x in range(n):
			for y in range(n):
				result[x, y] = np.sum(
					np.abs(plaintext_matrix[x] - plaintext_matrix[y])
				)
		return result

	@staticmethod
	def compute_centroid_shift_ckks(
		previous_centroids: List[PyCtxt],
		current_centroids: List[PyCtxt],
		init_shiftmatrix: List[PyCtxt] = None,
	) -> List[PyCtxt]:
		"""Compute encrypted shift between two CKKS centroid sets.

		Performs element-wise homomorphic subtraction between previous
		and current centroids. Falls back to init_shiftmatrix on subtraction
		failure.

		Args:
			previous_centroids: Centroids from previous iteration.
			current_centroids: Centroids from current iteration.
			init_shiftmatrix: Optional fallback shift matrix for bootstrapping.

		Returns:
			Encrypted shift matrix as List[PyCtxt].
		"""
		k = len(previous_centroids)
		S1: List[PyCtxt] = [None] * k
		for i in range(k):
			try:
				S1[i] = previous_centroids[i] - current_centroids[i]
			except Exception:
				S1[i] = init_shiftmatrix[i] if init_shiftmatrix is not None else None
		return S1

	@staticmethod
	def calculateCentroidsCkks(
		scheme,
		clusters: List[List[PyCtxt]],
		k: int,
	) -> Result[List[Optional[PyCtxt]], Exception]:
		"""Compute homomorphic average (centroid) for each CKKS cluster.

		Args:
			scheme: Ckks scheme instance.
			clusters: List of clusters, each a list of encrypted records.
			k: Number of clusters.

		Returns:
			Result with Ok(List[Optional[PyCtxt]] centroids) or Err(exception).
			Empty clusters yield None for that centroid.
		"""
		try:
			centroids: List[Optional[PyCtxt]] = [None] * k
			for j in range(k):
				cj_len = len(clusters[j])
				if cj_len == 0:
					centroids[j] = None
					continue
				average = clusters[j][0]
				for rec in clusters[j][1:]:
					average = scheme.add(average, rec)
				scheme._relinearize_if_possible(average)
				centroids[j] = scheme.multiply_scalar(1.0 / cj_len, average)
			return Ok(centroids)
		except Exception as e:
			return Err(e)

	@staticmethod
	def execute_encrypted_phase_liu(
		status: int,
		k: int,
		encrypted_matrix: npt.NDArray,
		udm: npt.NDArray,
		num_attributes: int,
		centroids: npt.NDArray = None,
		m: int = 3,
	) -> Result:
		"""Execute one encrypted KMeans iteration using Liu's scheme.

		Shared implementation for SKMeans and DBSKMeans.

		Args:
			status: ClusteringStatus (START or WORK_IN_PROGRESS).
			k: Number of clusters.
			encrypted_matrix: Encrypted dataset as NDArray.
			udm: Updatable distance matrix.
			num_attributes: Number of attributes per record.
			centroids: Previous centroids. None on first call.
			m: Number of secret key attributes. Defaults to 3.

		Returns:
			Result with Ok((shift_matrix, prev_centroids,
			new_centroids, label_vector)) or Err(exception).
		"""
		import copy
		from rory.core.utils.constants import Constants
		try:
			if status == Constants.ClusteringStatus.START:
				C = [[encrypted_matrix[i].tolist()] for i in range(k)]
				start_record = k
				cent_i = np.array([encrypted_matrix[i] for i in range(k)])
			else:
				C = Utils.empty_cluster(k=k)
				start_record = 0
				cent_i = copy.copy(centroids)

			populate_result = Utils._populate_clusters(
				record_id=start_record,
				UDM=udm,
				num_clusters=k,
				num_attributes=num_attributes,
				ciphertext_matrix=encrypted_matrix,
				append_fn=lambda cl, idx, rec: cl[idx].append(rec.tolist()),
				clusters=C,
			)
			if populate_result.is_err:
				return Err(populate_result.unwrap_err())
			C, label_vector = populate_result.unwrap()

			centroids_result = Utils.calculateCentroids(
				clusters=C, k=k, attributes=num_attributes, m=m,
			)
			if centroids_result.is_err:
				return Err(centroids_result.unwrap_err())
			cent_j = centroids_result.unwrap()

			S1 = Utils.compute_centroid_shift_liu(
				previous_centroids=cent_i,
				current_centroids=cent_j,
			)

			if status == Constants.ClusteringStatus.START:
				label_vector = Utils.fillLabelVector(
					label_vector=label_vector, k=k
				)

			return Ok((S1, cent_i, cent_j, label_vector))
		except Exception as e:
			return Err(e)

	@staticmethod
	def fit_conventional_iterative(
		instance: 'ConventionalClustering',
		status: int,
		k: int,
		m: int,
		encrypted_matrix: npt.NDArray,
		UDM: npt.NDArray,
		num_attributes: int,
		Cent_j: npt.NDArray,
		iterations: int,
		n_iterations: int,
		scheme: 'Liu',
		sk: npt.NDArray,
		min_error: float = 0.000001,
	) -> List[int]:
		"""Run the Liu-based iterative KMeans clustering loop.

		Shared by SKMeans and DBSKMeans (conventional).

		Args:
			instance: The clustering algorithm instance (self).
			status: Current ClusteringStatus.
			k: Number of clusters.
			m: Number of secret key attributes.
			encrypted_matrix: Encrypted data matrix.
			UDM: Updatable distance matrix.
			num_attributes: Number of attributes.
			Cent_j: Current centroids.
			iterations: Current iteration counter.
			n_iterations: Maximum iterations.
			scheme: Liu scheme for decryption.
			sk: Secret key for decryption.
			min_error: Convergence threshold. Defaults to 1e-6.

		Returns:
			List[int]: Final cluster label assignments.
		"""
		label_vector = []
		while (
			status != Constants.ClusteringStatus.COMPLETED
			and iterations < n_iterations
		):
			j = instance.execute_encrypted_phase(
				status=status,
				k=k,
				encrypted_matrix=encrypted_matrix,
				udm=UDM,
				num_attributes=num_attributes,
				centroids=Cent_j,
				m=m,
			)
			if j.is_ok:
				_run1 = j.unwrap()
			else:
				raise RuntimeError(
					f"Encrypted phase failed: {j.unwrap_err()}"
				)
			shiftmatrix = _run1[0]
			_Cent_j = _run1[2]
			label_vector = _run1[3]

			dec_shiftmatrix = scheme.decryptMatrix(
				ciphertext_matrix=shiftmatrix, secret_key=sk
			).data

			UDM = instance.execute_plaintext_phase(
				k=k,
				udm=UDM,
				num_attributes=num_attributes,
				shift_matrix=dec_shiftmatrix,
			)

			if np.max(np.abs(dec_shiftmatrix)) <= min_error:
				status = Constants.ClusteringStatus.COMPLETED
			else:
				status = Constants.ClusteringStatus.WORK_IN_PROGRESS
				Cent_j = _Cent_j
			iterations += 1
		return label_vector

	@staticmethod
	def fit_pqc_iterative(
		instance: 'PqcClustering',
		status: int,
		k: int,
		encrypted_matrix: 'List[PyCtxt]',
		UDM: npt.NDArray,
		num_attributes: int,
		Cent_j: 'Optional[List[PyCtxt]]',
		iterations: int,
		n_iterations: int,
		scheme: 'Ckks',
		min_error: float = 0.000001,
	) -> List[int]:
		"""Run the CKKS-based iterative KMeans clustering loop.

		Shared by Skmeans and DBSKMeans (PQC).

		Args:
			instance: The clustering algorithm instance (self).
			status: Current ClusteringStatus.
			k: Number of clusters.
			encrypted_matrix: CKKS encrypted data matrix.
			UDM: Updatable distance matrix.
			num_attributes: Number of attributes.
			Cent_j: Current centroids in encrypted form.
			iterations: Current iteration counter.
			n_iterations: Maximum iterations.
			scheme: Ckks scheme for decryption.
			min_error: Convergence threshold. Defaults to 1e-6.

		Returns:
			List[int]: Final cluster label assignments.
		"""
		label_vector = []
		while (
			status != Constants.ClusteringStatus.COMPLETED
			and iterations < n_iterations
		):
			j = instance.execute_encrypted_phase(
				status           = status,
				k                = k,
				encrypted_matrix = encrypted_matrix,
				udm              = UDM,
				num_attributes   = num_attributes,
				centroids        = Cent_j,
			)
			if j.is_ok:
				_run1 = j.unwrap()
			else:
				raise RuntimeError(
					f"Encrypted phase failed: {j.unwrap_err()}"
				)
			
			shiftmatrix = _run1[0]
			_Cent_j = _run1[2]
			label_vector = _run1[3]

			dec_shiftmatrix = scheme.decryptMatrix(
				ciphertext_matrix = shiftmatrix,
				shape             = [k, num_attributes],
			)

			UDM = instance.execute_plaintext_phase(
				k              = k,
				udm            = UDM,
				num_attributes = num_attributes,
				shift_matrix   = dec_shiftmatrix,
			)

			if np.max(np.abs(dec_shiftmatrix)) <= min_error:
				status = Constants.ClusteringStatus.COMPLETED
			else:
				status = Constants.ClusteringStatus.WORK_IN_PROGRESS
				Cent_j = _Cent_j
			iterations += 1
		return label_vector