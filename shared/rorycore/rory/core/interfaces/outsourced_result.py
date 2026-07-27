import numpy as np


class OutsourcedDataResult:
	"""Interface representing the result of a data owner's preparation step.

	Encapsulates the encrypted dataset, the Updatable Distance Matrix (UDM),
	timing metadata, key intervals for FDH-OPE, and optional encrypted
	threshold used before outsourcing data to the TPDM.

	Keyword Args:
		udm_time (float, optional): Time taken to compute the UDM. Defaults
			to 0.
		UDM (npt.NDArray, optional): The Updatable Distance Matrix. Defaults
			to an empty array.
		encrypted_matrix (npt.NDArray, optional): The encrypted dataset
			matrix. Defaults to an empty array.
		encrypted_matrix_time (float, optional): Time taken to encrypt the
			matrix. Defaults to 0.
		messageIntervals (dict, optional): FDH-OPE message space intervals.
			Defaults to {}.
		cypherIntervals (dict, optional): FDH-OPE cipher space intervals.
			Defaults to {}.
		encrypted_threshold (float, optional): Encrypted threshold for
			clustering. Defaults to 0.
		num_attributes (int, optional): Number of attributes in the dataset.
			Defaults to 0.
		encrypted_weights (npt.NDArray, optional): Encrypted weights for
			PPLR. Defaults to an empty array.
		encrypted_bias (npt.NDArray, optional): Encrypted bias for PPLR.
			Defaults to an empty array.
		encrypted_labels (npt.NDArray, optional): Encrypted label vector
			for PPLR. Defaults to an empty array.
	"""
	def __init__(self,**kwargs):
		self.udm_time              = kwargs.get("udm_time",0)
		self.UDM                   = kwargs.get("UDM",np.array([]))
		self.encrypted_matrix      = kwargs.get("encrypted_matrix",np.array([]))
		self.encrypted_matrix_time = kwargs.get("encrypted_matrix_time",np.array([]))
		self.messageIntervals      = kwargs.get("messageIntervals",{})
		self.cypherIntervals       = kwargs.get("cypherIntervals",{})
		self.encrypted_threshold   = kwargs.get("encrypted_threshold",0)
		self.num_attributes        = kwargs.get("num_attributes",0)
		self.encrypted_weights     = kwargs.get("encrypted_weights",np.array([]))
		self.encrypted_bias        = kwargs.get("encrypted_bias",np.array([]))
		self.encrypted_labels      = kwargs.get("encrypted_labels",np.array([]))