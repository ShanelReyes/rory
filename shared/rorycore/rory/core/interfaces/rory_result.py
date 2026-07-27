import numpy as np


class RoryResult:
	"""Interface for results returned by clustering and classification algorithms.

	Encapsulates the cluster/class label vector, iteration count, and timing
	metadata across different phases (response, service, clustering, UDM,
	and cipher operations).

	Keyword Args:
		label_vector (np.ndarray, optional): Cluster or class labels assigned
			to each data point. Defaults to an empty array.
		n_iterations (int, optional): Number of iterations performed by the
			algorithm. Defaults to 0.
		response_time (float, optional): Total end-to-end response time in
			seconds. Defaults to 0.
		service_time (float, optional): Time spent exclusively by the
			service. Defaults to 0.
		clustering_time (float, optional): Time spent on clustering
			operations. Defaults to 0.
		udm_time (float, optional): Time spent computing the UDM. Defaults
			to 0.
		cipher_time (float, optional): Time spent on cryptographic
			operations. Defaults to 0.
	"""
	def __init__(self,**kwargs):
		self.label_vector    = kwargs.get("label_vector",np.array(()))
		self.n_iterations    = kwargs.get("n_iterations",0)
		self.response_time   = kwargs.get("response_time",0)
		self.service_time    = kwargs.get("service_time",0)
		self.clustering_time = kwargs.get("clustering_time",0)
		self.udm_time        = kwargs.get("udm_time",0)
		self.cipher_time     = kwargs.get("cipher_time",0)