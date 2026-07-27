import numpy as np
import numpy.typing as npt
import warnings
from time import time
from rory.core.utils.utils import Utils
from rory.core.security.cryptosystem.fdhope import Fdhope
from rory.core.interfaces.outsourced_result import OutsourcedDataResult
from rory.core.security.cryptosystem.liu import Liu
from typing import Tuple,Dict

class DataOwner(object):
	"""
    A class representing the preparation step performed by data owners to securely externalize their data 
    to the TPDM that provides DMaaS.

    Attributes:
        liu_scheme (Liu): An instance of Liu's symmetric encryption scheme used for encrypting data.
        sens (float): Sensitivity parameter used for FDH-OPE encryption (default is 0.00001).
        sk: The secret key generated using the Liu scheme.
        m: Number of secret key attributes from the Liu scheme.
        messageIntervals (Dict[str, Tuple[float, float]]): A dictionary storing the message intervals for FDH-OPE.
        cypherIntervals (Dict[str, Tuple[float, float]]): A dictionary storing the cipher intervals for FDH-OPE.
        encrypted_threshold: The threshold value after encryption (DBSNNC only).
        threshold: The plaintext threshold value (DBSNNC only).
    """
	def __init__(self, liu_scheme:Liu, sens:float=0.00001):
		warnings.warn(
			"rory.core.security.dataowner.DataOwner (conventional) is deprecated, "
			"use the unified DataOwner (from rory.core.security.dataowner) with "
			"fluent API instead",
			DeprecationWarning, stacklevel=2
		)
		self.sens           = sens
		self.liu_scheme:Liu = liu_scheme
		self.sk             = self.liu_scheme.generate_secret_key()
		self.m              = self.liu_scheme.m
		self.messageIntervals:Dict[str,Tuple[float,float]] = {} 
		self.cypherIntervals:Dict[str,Tuple[float,float]]  = {}
		self.encrypted_threshold = 0
		self.threshold = 0
		

	def liu_encrypt_matrix_chunk(self, plaintext_matrix:npt.NDArray) -> npt.NDArray :
		"""
		Encrypts a matrix chunk using Liu's encryption scheme.

		This method encrypts the provided plaintext matrix by delegating the operation to the Liu scheme's
		encryptMatrix method with the stored secret key. The resulting encrypted matrix is returned as a NumPy array.

		Parameters:
			plaintext_matrix (npt.NDArray): The matrix of plaintext values to encrypt.
			np_random (bool, optional): Flag to indicate whether to use NumPy's random generator. Defaults to False.

		Returns:
			npt.NDArray: The encrypted matrix resulting from the encryption process.
		"""
		encryption_result = self.liu_scheme.encryptMatrix(
			plaintext_matrix = plaintext_matrix,
			secret_key       = self.sk,
		)
		return encryption_result.data


	def outsourcedData(self, plaintext_matrix:npt.NDArray, threshold:float=-1, algorithm:str= "SKMEANS") -> OutsourcedDataResult:
		"""
		Prepares and encrypts the data for secure outsourcing.

		This method processes the original numerical dataset by determining its shape to
		extract the number of attributes, then encrypts the entire dataset using Liu's encryption scheme
		with the stored secret key. The parameters 'threshold' and 'algorithm' are provided to
		specify clustering criteria for further processing (e.g., grouping records that are sufficiently similar),
		although the clustering functionality may be applied later in the data preparation workflow.

		Parameters:
			plaintext_matrix (npt.NDArray): The original numerical dataset to be encrypted.
			threshold (float, optional): The similarity threshold that defines how close records must be
				to belong to the same cluster. Defaults to -1.
			algorithm (str, optional): The clustering algorithm to use (e.g., "SKMEANS"). Defaults to "SKMEANS".

		Returns:
			ClientResult: An object containing the encrypted dataset and associated metadata.
		"""
		encryption_result = self.liu_scheme.encryptMatrix(
			plaintext_matrix = plaintext_matrix,
			secret_key       = self.sk
		)

		start_time_udm = time() 
		U = self.get_U(
			algorithm        = algorithm,
			plaintext_matrix = plaintext_matrix
		)

		if(algorithm == "DBSNNC"):
			if(threshold==-1):
				self.threshold = Utils.get_threshold(
					distance_matrix = U
				)
			else:
				self.threshold=threshold
			self.encrypted_threshold = self.encrypt_threshold(
				threshold = self.threshold
			)
		udm_time = time() - start_time_udm

		return OutsourcedDataResult(
			UDM                   = U,
			udm_time              = udm_time, 
			encrypted_matrix      = encryption_result.data,
			encrypted_matrix_time = 0,
			messageIntervals      = self.messageIntervals,
			cypherIntervals       = self.cypherIntervals,
			encrypted_threshold   = self.encrypted_threshold
		)


	def get_U(self, plaintext_matrix:npt.NDArray, algorithm:str) -> npt.NDArray:
		"""
		Generates the matrix U based on the specified clustering algorithm.

		Depending on the chosen algorithm, this function computes the appropriate U matrix from the
		original numerical dataset (plaintext_matrix).

		Parameters:
			plaintext_matrix (npt.NDArray): The original numerical dataset.
			algorithm (str): The clustering algorithm to use (e.g., "SKMEANS", "DBSKMEANS", "DBSNNC", "NNC").

		Returns:
			npt.NDArray: The computed U matrix based on the specified algorithm.

		Raises:
			Exception: If an unknown algorithm is provided.
		"""
		if (algorithm == "SKMEANS"): 
			U  = self.calculate_UDM(
				plaintext_matrix = plaintext_matrix
				)
		elif(algorithm == "DBSKMEANS"):
			U  = self.calculate_UDM(
				plaintext_matrix = plaintext_matrix
				)
			self.messageIntervals, self.cypherIntervals = Fdhope.keygen(
				dataset = U
			)
		elif(algorithm == "DBSNNC"):
			U  = self.calculate_DM(
				plaintext_matrix = plaintext_matrix
			)
			self.messageIntervals, self.cypherIntervals = Fdhope.keygen(
				dataset = U
			)
		elif(algorithm == "NNC"):
			U  = self.calculate_DM(
				plaintext_matrix = plaintext_matrix
			)
		else:
			raise Exception("UKNOWN ALGORITHM: {}" .format(algorithm))
		return U


	def encrypt_udm_chunks(self,plaintext_matrix:npt.NDArray,sens:float=0.0001, algorithm:str="DBSKMEANS") -> npt.NDArray:
		"""
		Encrypts chunks of the UDM (or ED) matrix based on the specified algorithm.

		This method encrypts the lower triangle of the input plaintext matrix by applying the Fdhope
		encryption functions with the provided sensitivity and precomputed message and cipher intervals.
		Depending on the algorithm type:
		- "DBSKMEANS": Uses Fdhope.encryptTensor, which handles an additional dimension.
		- "DBSNNC": Uses Fdhope.encryptMatrix to encrypt the lower triangle.
		
		Parameters:
			plaintext_matrix (npt.NDArray): The input plaintext matrix (or tensor) to be encrypted.
			sens (float, optional): The sensitivity parameter for the encryption process. Defaults to 0.0001.
			algorithm (str, optional): The algorithm specifying the encryption method ("DBSKMEANS" or "DBSNNC"). Defaults to "DBSKMEANS".

		Returns:
			npt.NDArray: The result of the encryption containing the encrypted matrix (or tensor) and associated metadata.

		Raises:
			Exception: If an unknown algorithm is specified.
		"""
		if (algorithm == "DBSKMEANS"):
			U = Fdhope.encryptTensor(
				plaintext_tensor = plaintext_matrix, 
				sens             = sens, 
				messagespace     = self.messageIntervals, 
				cipherspace      = self.cypherIntervals
			)
		elif(algorithm == "DBSNNC"):
			U = Fdhope.encryptMatrix(
				plaintext_matrix = plaintext_matrix, 
				sens             = sens, 
				messagespace     = self.messageIntervals, 
				cipherspace      = self.cypherIntervals
			)
		else:
			raise Exception("UKNOWN ALGORITHM: {}" .format(algorithm))
		return U

	
	def encrypt_U(self, U:npt.NDArray, algorithm:str) -> npt.NDArray:
		"""
		Encrypts the U matrix based on the specified clustering algorithm.

		This method encrypts the lower triangle of the U matrix using the Fdhope encryption function,
		and then mirrors the encrypted value to the corresponding upper triangle position to maintain symmetry.
		
		For the "DBSKMEANS" algorithm, it assumes U is a 3D tensor and encrypts each element across the third dimension.
		For the "DBSNNC" algorithm, it assumes U is a 2D matrix and encrypts each element directly.

		Parameters:
			U (npt.NDArray): The matrix (or tensor) to be encrypted.
			algorithm (str): The clustering algorithm to use ("DBSKMEANS" or "DBSNNC").

		Returns:
			npt.NDArray: The encrypted U matrix as a NumPy array.
		"""
		Ushape    = Utils.getShapeOfMatrix(U)
		for x in range(Ushape[0]): 
			for y in range(x):
				if (algorithm == "DBSKMEANS"): #if the algorithm is dbskmeans, one more dimension needs to be traversed
					for z in range(Ushape[2]):
						U[x][y][z] = Fdhope.encrypt( #the lower triangle of U is encrypted
							plaintext    = U[x][y][z], 
							sens         = self.sens, 
							messagespace = self.messageIntervals, 
							cipherspace  = self.cypherIntervals
						)
						U[y][x][z] = U[x][y][z] #the equivalent position is obtained to fill the upper triangle
				elif(algorithm == "DBSNNC"):
					U[x][y] = Fdhope.encrypt( #the lower triangle of U is encrypted
						plaintext    = U[x][y], 
						messagespace = self.messageIntervals, 
						cipherspace  = self.cypherIntervals
					)
					U[y][x] = U[x][y] #the equivalent position is obtained to fill the upper triangle
		return np.array(U)


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
		"""Calculates the EDM (Encrypted Distance Matrix) for the given numeric dataset.

		Thin wrapper delegating to Utils.calculate_DM.

		Parameters:
			plaintext_matrix (npt.NDArray): The numeric dataset with shape (n, a), where n is the number of
											records and a is the number of attributes.

		Returns:
			npt.NDArray: The resulting distance matrix with shape (n, n), where each entry is the Manhattan distance
						between corresponding records.
		"""
		return Utils.calculate_DM(plaintext_matrix=plaintext_matrix)
		
	

	def encrypt_threshold(self, threshold:int=0.01):
		"""
		Encrypts the threshold value using Fdhope's encryption function.

		This method encrypts the given threshold by applying Fdhope.encrypt with the stored message and cipher intervals.

		Parameters:
			threshold (int, optional): The threshold value to be encrypted. Defaults to 0.01.

		Returns:
			The encrypted threshold value.
		"""
		encrypted_threshold = Fdhope.encrypt(
				plaintext    = threshold,
				messagespace = self.messageIntervals, 
				cipherspace  = self.cypherIntervals
			)
		return encrypted_threshold