import numpy as np
import numpy.typing as npt
import warnings
from time import time
from rory.core.utils.utils import Utils
from rory.core.security.cryptosystem.fdhope import Fdhope
from rory.core.interfaces.outsourced_result import OutsourcedDataResult
from rory.core.security.cryptosystem.pqc.ckks import Ckks
from typing import Tuple,Dict,List
from Pyfhel import PyCtxt 

class DataOwner(object):
	"""
    A class representing the preparation step performed by data owners to securely externalize their data 
    to the TPDM that provides DMaaS.

    Attributes:
        scheme (CKKS): An instance of CKKS scheme used for encrypting data.
        sens (float): Sensitivity parameter used for FDH-OPE encryption (default is 0.00001).
        messageIntervals (Dict[str, Tuple[float, float]]): A dictionary storing the message intervals for FDH-OPE.
        cypherIntervals (Dict[str, Tuple[float, float]]): A dictionary storing the cipher intervals for FDH-OPE.
        encrypted_threshold: The threshold value after encryption (DBSNNC only).
        threshold: The plaintext threshold value (DBSNNC only).
    """

	def __init__(self, scheme:Ckks, sens:float=0.00001):
		warnings.warn(
			"rory.core.security.pqc.dataowner.DataOwner (PQC) is deprecated, "
			"use the unified DataOwner (from rory.core.security.dataowner) with "
			"fluent API instead",
			DeprecationWarning, stacklevel=2
		)
		self.sens = sens
		self.scheme:Ckks = scheme
		self.messageIntervals:Dict[str,Tuple[float,float]] = {} 
		self.cypherIntervals:Dict[str,Tuple[float,float]]  = {}
		self.encrypted_threshold = 0
		self.threshold = 0


	def outsourcedData(self, plaintext_matrix:npt.NDArray, threshold:float = -1, algorithm:str= "SKMEANS_PQC", np_random:bool  = False) -> OutsourcedDataResult:
		"""
		Prepares and encrypts the data for secure outsourcing.

		This method processes the original numerical dataset by determining its shape to
		extract the number of attributes, then encrypts the entire dataset using CKKS encryption scheme
		with the stored secret key. The parameters 'threshold' and 'algorithm' are provided to
		specify clustering criteria for further processing (e.g., grouping records that are sufficiently similar),
		although the clustering functionality may be applied later in the data preparation workflow.

		Parameters:
			plaintext_matrix (npt.NDArray): The original numerical dataset to be encrypted.
			threshold (float, optional): The similarity threshold that defines how close records must be
				to belong to the same cluster. Defaults to -1.
			algorithm (str, optional): The clustering algorithm to use. Defaults to "SKMEANS_PQC".

		Returns:
			ClientResult: An object containing the encrypted dataset and associated metadata.
		"""		
		Dshape     = Utils.getShapeOfMatrix(plaintext_matrix)
		attributes = Dshape[1]
		
		encryption_result = self.scheme.encryptMatrix(
			plaintext_matrix = plaintext_matrix
		)
		start_time_udm = time() 
		
		U = self.get_U ( #U is generated according to the chosen algorithm
			algorithm         = algorithm,
			plaintext_matrix  = plaintext_matrix
		)
		udm_time = time() - start_time_udm

		if(algorithm == "DBSNNC_PQC"):
			if(threshold == -1):
				self.threshold = Utils.get_threshold(
					distance_matrix = U
				)
			else:
				self.threshold = threshold
			self.encrypted_threshold = self.encrypt_threshold(
				threshold = self.threshold
			)

		return OutsourcedDataResult(
			UDM                 = U,
			udm_time            = udm_time, 
			encrypted_matrix    = np.array(encryption_result),
			num_attributes      = attributes,
			messageIntervals    = self.messageIntervals,
			cypherIntervals     = self.cypherIntervals,
			encrypted_threshold = self.encrypted_threshold
		)


	def get_U(self, plaintext_matrix:npt.NDArray, algorithm:str,**kwargs) -> npt.NDArray:
		"""
		Generates the matrix U based on the specified clustering algorithm.

		Depending on the chosen algorithm, this function computes the appropriate U matrix from the
		original numerical dataset (plaintext_matrix).

		Parameters:
			plaintext_matrix (npt.NDArray): The original numerical dataset.
			algorithm (str): The clustering algorithm to use (e.g., "SKMEANS_PQC", "DBSKMEANS_PQC").

		Returns:
			npt.NDArray: The computed U matrix based on the specified algorithm.

		Raises:
			Exception: If an unknown algorithm is provided.
		"""
		if (algorithm == "SKMEANS_PQC"): 
			U  = self.calculate_UDM(
				plaintext_matrix = plaintext_matrix
				)
		elif(algorithm == "DBSKMEANS_PQC"):
			U  = self.calculate_UDM(
				plaintext_matrix = plaintext_matrix
				)
			self.messageIntervals, self.cypherIntervals = Fdhope.keygen( #the intervals (SK) of each space are generated
				dataset = U
			)
		elif(algorithm == "DBSNNC_PQC"):
			U = self.calculate_DM(
				plaintext_matrix = plaintext_matrix
			)
			self.messageIntervals, self.cypherIntervals = Fdhope.keygen(
				dataset = U
			)
		else:
			raise Exception("UKNOWN ALGORITHM: {}" .format(algorithm))
		return U


	def calculate_UDM(self, plaintext_matrix:npt.NDArray, mode:str="diff") -> npt.NDArray:
		"""Calculates the UDM (Update Distance Matrix) from a numerical dataset.

		Thin wrapper delegating to Utils.calculate_UDM.

		Parameters:
			plaintext_matrix (npt.NDArray): The numeric dataset to process, typically with shape (n, a),
											where n is the number of records and a is the number of attributes.
			mode (str, optional): The mode of difference calculation. Options are:
				- "diff": Computes the raw difference.
				- "diff-abs": Computes the absolute difference.
				Defaults to "diff".

		Returns:
			npt.NDArray: The computed UDM matrix with shape (n, n, a), containing pairwise differences
						(or absolute differences) of the dataset.
		"""
		return Utils.calculate_UDM(plaintext_matrix=plaintext_matrix, mode=mode)

	def calculate_DM(self, plaintext_matrix:npt.NDArray) -> npt.NDArray:
		"""Calculates the DM (Distance Matrix) for the given numeric dataset.

		Thin wrapper delegating to Utils.calculate_DM.

		Parameters:
			plaintext_matrix (npt.NDArray): The numeric dataset with shape (n, a).

		Returns:
			npt.NDArray: The resulting distance matrix with shape (n, n).
		"""
		return Utils.calculate_DM(plaintext_matrix=plaintext_matrix)

	def encrypt_threshold(self, threshold: float = 0.01):
		"""Encrypts the threshold value using Fdhope.

		Parameters:
			threshold: The threshold value to be encrypted.

		Returns:
			The encrypted threshold value.
		"""
		encrypted_threshold = Fdhope.encrypt(
			plaintext=threshold,
			messagespace=self.messageIntervals,
			cipherspace=self.cypherIntervals,
		)
		return encrypted_threshold
		

	def ckks_encrypt_matrix_chunk(self, plaintext_matrix:npt.NDArray) -> List[PyCtxt]:
		"""
		The method delegates the actual encoding and encryption work to the underlying `ckks.encryptMatrix`, 
		which treats each row of the NumPy matrix as an independent plaintext vector.  
		The resulting ciphertext objects are returned as a flat list that preserves the original row ordering.

		Parameters
		----------
		plaintext_matrix : npt.NDArray
			A NumPy 2D array where each row is a plaintext vector to be encrypted.

		Returns
		-------
		List[PyCtxt]
			A list of ciphertext objects, one per row of the input matrix, in the same
			order as they appear in `plaintext_matrix`.
		"""
		encryption_result = self.scheme.encryptMatrix(
			plaintext_matrix = plaintext_matrix,
		)
		return encryption_result
	

	def ckks_encrypt_encode_list_chunk(self, plaintext_chunk:npt.NDArray) -> List[PyCtxt]:
		"""
		This helper first converts each scalar in `plaintext_chunk` to a CKKS plaintext via `scheme.encode_list`, 
		then encrypts each encoded value with `scheme.encrypt_list`. It is useful when the caller already has a chunk of data that should be protected.

		Parameters
		----------
		plaintext_chunk : npt.NDArray
			A NumPy array of floats to be encoded and encrypted element wise.

		Returns
		-------
		List[PyCtxt]
			A list of ciphertexts, one for every element in `plaintext_chunk`, preserving the original order.
		"""
		encode_plaintext_chunk = self.scheme.encode_list(plaintext_chunk)
		encryption_result      = self.scheme.encrypt_list(
			plaintext_vector = encode_plaintext_chunk,
		)
		return encryption_result
	

	def ckks_encrypt_matrix_list_chunk(self, plaintext_chunk:npt.NDArray) -> List[PyCtxt]:
		"""Encrypts a matrix chunk by encoding rows and then encrypting them.

		Delegates to the underlying CKKS scheme's encrypt_matrix_list, which
		encodes each row of the plaintext chunk and encrypts the resulting
		plaintexts.

		Args:
			plaintext_chunk (npt.NDArray): A NumPy 2D array whose rows are
				plaintext vectors to be encrypted.

		Returns:
			List[PyCtxt]: A list of ciphertexts, one per row.
		"""
		encryption_result      = self.scheme.encrypt_matrix_list(
			plaintext_matrix = plaintext_chunk,
		)
		return encryption_result


	def encrypt_U(self, U:npt.NDArray, algorithm:str) -> npt.NDArray:
		"""
		Encrypts the U matrix based on the specified clustering algorithm.

		This method encrypts the lower triangle of the U matrix using the Fdhope encryption function,
		and then mirrors the encrypted value to the corresponding upper triangle position to maintain symmetry.
		
		For the "DBSKMEANS_PQC" algorithm, it assumes U is a 3D tensor and encrypts each element across the third dimension.
		For the "DBSNNC_PQC" algorithm, it assumes U is a 2D matrix and encrypts each element directly.

		Parameters:
			U (npt.NDArray): The matrix (or tensor) to be encrypted.
			algorithm (str): The clustering algorithm to use ("DBSKMEANS" or "DBSNNC").

		Returns:
			npt.NDArray: The encrypted U matrix as a NumPy array.
		"""
		Ushape = Utils.getShapeOfMatrix(U)
		
		for x in range(Ushape[0]): 
			for y in range(x):
				if (algorithm == "DBSKMEANS_PQC"): #if the algorithm is dbskmeans, one more dimension needs to be traversed
					for z in range(Ushape[2]):
						U[x][y][z] = Fdhope.encrypt( #the lower triangle of U is encrypted
							plaintext    = U[x][y][z], 
							sens         = self.sens, 
							messagespace = self.messageIntervals, 
							cipherspace  = self.cypherIntervals
						)
						U[y][x][z] = U[x][y][z] #the equivalent position is obtained to fill the upper triangle
				elif(algorithm == "DBSNNC_PQC"):
					U[x][y] = Fdhope.encrypt( #the lower triangle of U is encrypted
						plaintext    = U[x][y], 
						messagespace = self.messageIntervals, 
						cipherspace  = self.cypherIntervals
					)
					U[y][x] = U[x][y] #the equivalent position is obtained to fill the upper triangle
		return np.array(U)
	
if __name__ =="__main__":
	ckks = Ckks.create_client(save=False,_round= True)

	do = DataOwner()

	vec1 = np.array([[4.5,1.3,2.1,5],[2.4,3.7,1.8,6]],  dtype=np.float32)
	
	encrypted = do.outsourcedData(
		plaintext_matrix = vec1,
	)

	print("uno")
	print(encrypted.encrypted_matrix)
	# print(encrypted.UDM)