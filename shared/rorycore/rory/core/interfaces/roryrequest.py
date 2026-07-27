from uuid import uuid4
import time
import json


class RoryRequestClient:
	"""Data transfer object representing a request sent from a client.

	Carries the request ID, start time, dataset identifier, target
	algorithm, and clustering parameters (k, m).

	Keyword Args:
		requestId (str, optional): Unique request identifier. Defaults to
			a UUID4.
		startRequestTime (float, optional): Unix timestamp when the
			request was initiated. Defaults to the current time.
		encryptedDatasetId (str, optional): Identifier of the encrypted
			dataset. Defaults to "encrypted-dataset-0".
		algorithm (str, optional): Clustering or classification algorithm.
			Defaults to "SKMEANS".
		m (int, optional): Number of secret key attributes. Defaults to 3.
		k (int, optional): Number of clusters. Defaults to 3.
	"""
	def __init__(self,**kwargs):
		self.requestId        = kwargs.get("requestId",str(uuid4()))
		self.startRequestTime = kwargs.get("startRequestTime",time.time())
		self.datasetId        = kwargs.get("encryptedDatasetId","encrypted-dataset-0")
		self.algorithm        = kwargs.get("algorithm","SKMEANS")
		self.m                = kwargs.get("m",3)
		self.k                = kwargs.get("k",3)

	def serialize(self):
		"""Serializes the request to a JSON string.

		Returns:
			str: JSON representation of the request object.
		"""
		return json.dumps(self.__dict__)
    