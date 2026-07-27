import numpy as np
import numpy.typing as npt
import warnings
from rory.core.security.cryptosystem.paillier import Paillier


class DataOwner:
	"""Represents a data owner that prepares data for secure outsourcing
	using the Paillier homomorphic encryption scheme.

	Generates a Paillier keypair at construction time and provides methods
	to encrypt matrix chunks using the public key.

	Args:
		securitylevel (int, optional): Security level in bits (128, 192,
			256). Defaults to 128.
	"""
	def __init__(self, securitylevel: int = 128):
		warnings.warn(
			"rory.core.security.dataowner_paillier.DataOwner is deprecated, "
			"use the unified DataOwner (from rory.core.security.dataowner) with "
			"fluent API instead",
			DeprecationWarning, stacklevel=2
		)
		self.security_level = securitylevel
		self.pk = None
		self.sk = None



	def generate_keys(self, output_path:str="/sink", filename:str="rory-phe", save:bool=False):
		"""Generates a Paillier keypair and stores it in the instance.

		Args:
			output_path (str, optional): Directory for saving keys.
				Defaults to "/sink".
			filename (str, optional): Base filename for key files.
				Defaults to "rory-phe".
			save (bool, optional): If True, persists keys to disk.
				Defaults to False.
		"""
		self.paillier = Paillier()
		self.pk, self.sk = self.paillier.generate_keys(
			security_level = self.security_level,
			save           = save,
			filename       = filename,
			output_path    = output_path
		)

	@staticmethod
	def from_keys(path:str="/sink",filename:str="rory-phe")->'DataOwner':
		"""Creates a DataOwner instance from existing Paillier key files.

		Args:
			path (str, optional): Directory where the key files are stored.
				Defaults to "/sink".
			filename (str, optional): Base filename (without extension).
				Defaults to "rory-phe".

		Returns:
			DataOwner: A new instance loaded with the persisted keypair.
		"""
		do = DataOwner()
		paillier_scheme = Paillier()
		paillier_scheme.load_keys(path=path, filename=filename)
		do.pk = paillier_scheme.public_key
		do.sk = paillier_scheme.private_key
		do.paillier = paillier_scheme
		return do		

	def paillier_encrypt_matrix_chunk(self, plaintext_matrix: npt.NDArray):
		"""Encrypts a plaintext matrix chunk using the Paillier public key.

		Args:
			plaintext_matrix (npt.NDArray): The plaintext matrix to encrypt.

		Returns:
			npt.NDArray: The encrypted matrix.
		"""
		encryption_result = self.paillier.encrypt_matrix(
			plaintext_matrix = plaintext_matrix,
		)
		return encryption_result.data
	
if __name__ == "__main__":
	plaintext_matrix = np.zeros((3,3))
	dataowner = DataOwner(securitylevel=128)
	encrypted_chunk = dataowner.paillier_encrypt_matrix_chunk(plaintext_matrix)
	print(encrypted_chunk)