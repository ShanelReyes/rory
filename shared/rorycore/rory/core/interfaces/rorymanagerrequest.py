from uuid import uuid4
import time


class RoryRequestManager:
	"""Data transfer object representing a request sent to the TPDM manager.

	Carries the request ID, arrival time, latency, target algorithm, and
	the identifier of the encrypted matrix along with any extra metadata.

	Keyword Args:
		requestId (str, optional): Unique request identifier. Defaults to
			a UUID4.
		arrivalTime (int, optional): Unix timestamp when the request
			arrived. Defaults to the current time.
		startRequestTime (int, optional): Unix timestamp when the request
			was initiated. Defaults to 0.
		algorithm (str, optional): Clustering algorithm to execute.
			Defaults to "SKMEANS".
		encryptedMatrixId (str, optional): Identifier of the encrypted
			matrix. Defaults to "MATRIX_ID".
		metadata (dict, optional): Additional metadata. Defaults to {}.
	"""
	def __init__(self,**kwargs):
		self.requestId         = kwargs.get("requestId",str(uuid4()))
		self.arrivalTime       = kwargs.get("arrivalTime", int(time.time()) )
		self.startRequestTime  = kwargs.get("startRequestTime",0)
		self.latency           = self.arrivalTime - self.startRequestTime
		self.algorithm         = kwargs.get("algorithm","SKMEANS")
		self.encryptedMatrixId = kwargs.get("encryptedMatrixId","MATRIX_ID")
		self.metadata          = kwargs.get("metadata",{})