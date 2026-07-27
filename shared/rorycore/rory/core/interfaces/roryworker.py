from requests import Response
from rory.core.utils.constants import Constants


class DumbRoryWorker:
	"""A dummy worker implementation used for manual testing.

	Keyword Args:
		workerId (str): Worker identifier.
		port (int, optional): Worker port. Defaults to 9000.
	"""
	def __init__(self,**kwargs):
		self.workerId  = kwargs.get("workerId")
		self.port      = kwargs.get("port",9000)

	def kmeans(self,**kwargs) -> Response:
		"""Mocks a K-Means clustering request.

		Returns:
			Response: An HTTP POST response.
		"""
		return self.session.post(
			"http://{}:{}/clustering/kmeans".format(self.workerId,self.port),
			 headers = kwargs,
		)

	def DBSkMeans(self,**kwargs):
		"""Mocks a Double-Blind Secure K-Means request.

		Returns:
			Response: An empty HTTP response.
		"""
		return Response()

	def SKMeans(self,**kwargs) -> Response:
		"""Mocks a Secure K-Means request.

		Returns:
			Response: An empty HTTP response.
		"""
		return Response()


class RoryWorker:
	"""Client-side proxy for invoking clustering and classification
	algorithms on a remote TPDM worker.

	Selects the appropriate REST endpoint based on the configured
	algorithm and forwards the request via an HTTP session.

	Keyword Args:
		workerId (str, optional): Worker hostname or IP. Defaults to
			"localhost".
		port (int, optional): Worker port. Defaults to 9000.
		session: An HTTP session object for sending requests.
		algorithm (str, optional): Algorithm identifier from
			Constants.ClusteringAlgorithms, Constants.ClassificationAlgorithms,
			or Constants.MachineLearningAlgorithms. Defaults to SKMEANS.
	"""
	def __init__(self,**kwargs):
		self.workerId  = kwargs.get("workerId","localhost")
		self.port      = kwargs.get("port",9000)
		self.session   = kwargs.get("session")
		self.algorithm = kwargs.get("algorithm",Constants.ClusteringAlgorithms.SKMEANS)  

	def run(self,*args,timeout:int = 120, **kwargs) -> Response:
		"""Dispatches a request to the worker based on the configured algorithm.

		Args:
			*args: Positional arguments forwarded to the endpoint method.
			timeout (int, optional): Request timeout in seconds. Defaults
				to 120.
			**kwargs: Keyword arguments forwarded to the endpoint method.

		Returns:
			Response: The HTTP response from the worker.
		"""
		if(self.algorithm == Constants.ClusteringAlgorithms.SKMEANS):
			return self.__skmeans(timeout, **kwargs)
		elif(self.algorithm == Constants.ClusteringAlgorithms.KMEANS):
			return self.__kmeans(timeout, **kwargs)
		elif (self.algorithm == Constants.ClusteringAlgorithms.DBSKMEANS): 
			return self.__dbskmeans(timeout, **kwargs)
		elif (self.algorithm == Constants.ClusteringAlgorithms.DBSNNC):
			return self.__dbsnnc(timeout, **kwargs)
		elif (self.algorithm == Constants.ClusteringAlgorithms.NNC):
			return self.__nnc(timeout, **kwargs)
		elif (self.algorithm == Constants.ClassificationAlgorithms.KNN_TRAIN):
			return self.__knn_train(timeout, **kwargs)
		elif (self.algorithm == Constants.ClassificationAlgorithms.KNN_PREDICT):
			return self.__knn_predict(timeout, **kwargs)
		elif (self.algorithm == Constants.ClassificationAlgorithms.SKNN_TRAIN):
			return self.__sknn_train(timeout, **kwargs)
		elif (self.algorithm == Constants.ClassificationAlgorithms.SKNN_PREDICT):
			return self.__sknn_predict(timeout, **kwargs)
		elif (self.algorithm == Constants.ClusteringAlgorithms.SKMEANS_PQC):
			return self.__pqc_skmeans(timeout, **kwargs)
		elif (self.algorithm == Constants.ClusteringAlgorithms.DBSKMEANS_PQC):
			return self.__pqc_dbskmeans(timeout, **kwargs)
		elif (self.algorithm == Constants.ClusteringAlgorithms.DBSNNC_PQC):
			return self.__pqc_dbsnnc(timeout, **kwargs)
		elif (self.algorithm == Constants.ClassificationAlgorithms.SKNN_PQC_TRAIN):
			return self.__pqc_sknn_train(timeout, **kwargs)
		elif (self.algorithm == Constants.ClassificationAlgorithms.SKNN_PQC_PREDICT):
			return self.__pqc_sknn_predict(timeout, **kwargs)
		elif (self.algorithm == Constants.MachineLearningAlgorithms.LOGISTIC_REGRESSION_TRAIN):
			return self.__logistic_regression_train(timeout, **kwargs)
		elif (self.algorithm == Constants.MachineLearningAlgorithms.LOGISTIC_REGRESSION_PREDICT):
			return self.__logistic_regression_predict(timeout, **kwargs)
		elif (self.algorithm == Constants.MachineLearningAlgorithms.PPLR_TRAIN):
			return self.__pplr_train(timeout, **kwargs)
		elif (self.algorithm == Constants.MachineLearningAlgorithms.PPLR_PREDICT):
			return self.__pplr_predict(timeout, **kwargs)
		else:
			return Response()


	def __kmeans(self,timeout:int = 120, **kwargs) -> Response:
		"""POSTs a K-Means clustering request to the worker.

		Args:
			timeout (int, optional): Request timeout in seconds. Defaults
				to 120.

		Returns:
			Response: The HTTP response.

		Raises:
			Exception: If the request fails.
		"""
		try:
			response:Response = self.session.post(
				"http://{}:{}/clustering/kmeans".format(self.workerId,self.port),
					headers = kwargs.get("headers",{}),
					timeout = timeout
			)
			response.raise_for_status()
			return response
		except Exception as e:
			raise e  

	def __skmeans(self,timeout:int = 120, **kwargs) -> Response:
		"""POSTs a Secure K-Means clustering request to the worker.

		Args:
			timeout (int, optional): Request timeout in seconds. Defaults
				to 120.

		Returns:
			Response: The HTTP response.

		Raises:
			Exception: If the request fails.
		"""
		try:
			response:Response = self.session.post(
				"http://{}:{}/clustering/skmeans".format(self.workerId,self.port),
				headers = kwargs.get("headers",{}),
				timeout = timeout
			)
			response.raise_for_status()
			return response
		except Exception as e:
			raise e

	def __dbskmeans(self, timeout:int = 120, **kwargs) -> Response:
		"""POSTs a Double-Blind Secure K-Means request to the worker.

		Args:
			timeout (int, optional): Request timeout in seconds. Defaults
				to 120.

		Returns:
			Response: The HTTP response.

		Raises:
			Exception: If the request fails.
		"""
		try:
			response:Response = self.session.post(
				"http://{}:{}/clustering/dbskmeans".format(self.workerId,self.port),
				headers = kwargs.get("headers",{}),
				timeout = timeout
			)
			response.raise_for_status()
			return response
		except Exception as e:
			raise e
	

	def __dbsnnc(self,timeout:int = 120, **kwargs) -> Response:
		"""POSTs a Double-Blind Secure NNC request to the worker.

		Args:
			timeout (int, optional): Request timeout in seconds. Defaults
				to 120.

		Returns:
			Response: The HTTP response.

		Raises:
			Exception: If the request fails.
		"""
		try:
			response:Response = self.session.post(
				"http://{}:{}/clustering/dbsnnc".format(self.workerId,self.port),
				headers = kwargs.get("headers",{}),
				timeout = timeout
			)
			response.raise_for_status()
			return response
		except Exception as e:
			raise e
		
	def __nnc(self,timeout:int = 120, **kwargs) -> Response:
		"""POSTs an NNC clustering request to the worker.

		Args:
			timeout (int, optional): Request timeout in seconds. Defaults
				to 120.

		Returns:
			Response: The HTTP response.

		Raises:
			Exception: If the request fails.
		"""
		try:
			response:Response = self.session.post(
				"http://{}:{}/clustering/nnc".format(self.workerId,self.port),
				headers = kwargs.get("headers",{}),
				timeout = timeout
			)
			response.raise_for_status()
			return response
		except Exception as e:
			raise e
		
	def __knn_train(self,timeout:int = 120, **kwargs) -> Response:
		"""POSTs a KNN training request to the worker.

		Args:
			timeout (int, optional): Request timeout in seconds. Defaults
				to 120.

		Returns:
			Response: The HTTP response.

		Raises:
			Exception: If the request fails.
		"""
		try:
			response:Response = self.session.post(
				"http://{}:{}/classification/knn/train".format(self.workerId,self.port),
				headers = kwargs.get("headers",{}),
				timeout = timeout
			)
			response.raise_for_status()
			return response
		except Exception as e:
			raise e

	def __knn_predict(self,timeout:int = 120, **kwargs) -> Response:
		"""POSTs a KNN prediction request to the worker.

		Args:
			timeout (int, optional): Request timeout in seconds. Defaults
				to 120.

		Returns:
			Response: The HTTP response.

		Raises:
			Exception: If the request fails.
		"""
		try:
			response:Response = self.session.post(
				"http://{}:{}/classification/knn/predict".format(self.workerId,self.port),
				headers = kwargs.get("headers",{}),
				timeout = timeout
			)
			response.raise_for_status()
			return response
		except Exception as e:
			raise e


	def __sknn_train(self,timeout:int = 120, **kwargs) -> Response:
		"""POSTs a Secure KNN training request to the worker.

		Args:
			timeout (int, optional): Request timeout in seconds. Defaults
				to 120.

		Returns:
			Response: The HTTP response.

		Raises:
			Exception: If the request fails.
		"""
		try:
			response:Response = self.session.post(
				"http://{}:{}/classification/sknn/train".format(self.workerId,self.port),
				headers = kwargs.get("headers",{}),
				timeout = timeout
			)
			response.raise_for_status()
			return response
		except Exception as e:
			raise e

	def __sknn_predict(self,timeout:int = 120, **kwargs) -> Response:
		"""POSTs a Secure KNN prediction request to the worker.

		Args:
			timeout (int, optional): Request timeout in seconds. Defaults
				to 120.

		Returns:
			Response: The HTTP response.

		Raises:
			Exception: If the request fails.
		"""
		try:
			response:Response = self.session.post(
				"http://{}:{}/classification/sknn/predict".format(self.workerId,self.port),
				headers = kwargs.get("headers",{}),
				timeout = timeout
			)
			response.raise_for_status()
			return response
		except Exception as e:
			raise e

	def __pqc_skmeans(self,timeout:int = 120, **kwargs) -> Response:
		"""POSTs a PQC-based Secure K-Means request to the worker.

		Args:
			timeout (int, optional): Request timeout in seconds. Defaults
				to 120.

		Returns:
			Response: The HTTP response.

		Raises:
			Exception: If the request fails.
		"""
		try:
			response:Response = self.session.post(
				"http://{}:{}/clustering/pqc/skmeans".format(self.workerId,self.port),
				headers = kwargs.get("headers",{}),
				timeout = timeout
			)
			response.raise_for_status()
			return response
		except Exception as e:
			raise e
		
	def __pqc_dbskmeans(self,timeout:int = 120, **kwargs) -> Response:
		"""POSTs a PQC-based Double-Blind Secure K-Means request to the
		worker.

		Args:
			timeout (int, optional): Request timeout in seconds. Defaults
				to 120.

		Returns:
			Response: The HTTP response.

		Raises:
			Exception: If the request fails.
		"""
		try:
			response:Response = self.session.post(
				"http://{}:{}/clustering/pqc/dbskmeans".format(self.workerId,self.port),
				headers = kwargs.get("headers",{}),
				timeout = timeout
			)
			response.raise_for_status()
			return response
		except Exception as e:
			raise e

	def __pqc_dbsnnc(self,timeout:int = 120, **kwargs) -> Response:
		"""POSTs a PQC-based Double-Blind Secure NNC request to the
		worker.

		Args:
			timeout (int, optional): Request timeout in seconds. Defaults
				to 120.

		Returns:
			Response: The HTTP response.

		Raises:
			Exception: If the request fails.
		"""
		try:
			response:Response = self.session.post(
				"http://{}:{}/clustering/pqc/dbsnnc".format(self.workerId,self.port),
				headers = kwargs.get("headers",{}),
				timeout = timeout
			)
			response.raise_for_status()
			return response
		except Exception as e:
			raise e
		
	def __pqc_sknn_train(self,timeout:int = 120, **kwargs) -> Response:
		"""POSTs a PQC-based Secure KNN training request to the worker.

		Args:
			timeout (int, optional): Request timeout in seconds. Defaults
				to 120.

		Returns:
			Response: The HTTP response.

		Raises:
			Exception: If the request fails.
		"""
		try:
			response:Response = self.session.post(
				"http://{}:{}/classification/pqc/sknn/train".format(self.workerId,self.port),
				headers = kwargs.get("headers",{}),
				timeout = timeout
			)
			response.raise_for_status()
			return response
		except Exception as e:
			raise e

	def __pqc_sknn_predict(self,timeout:int = 120, **kwargs) -> Response:
		"""POSTs a PQC-based Secure KNN prediction request to the worker.

		Args:
			timeout (int, optional): Request timeout in seconds. Defaults
				to 120.

		Returns:
			Response: The HTTP response.

		Raises:
			Exception: If the request fails.
		"""
		try:
			response:Response = self.session.post(
				"http://{}:{}/classification/pqc/sknn/predict".format(self.workerId,self.port),
				headers = kwargs.get("headers",{}),
				timeout = timeout
			)
			response.raise_for_status()
			return response
		except Exception as e:
			raise e
		
	def __logistic_regression_train(self,timeout:int = 120, **kwargs) -> Response:
		"""POSTs a Logistic Regression training request to the worker.

		Args:
			timeout (int, optional): Request timeout in seconds. Defaults
				to 120.

		Returns:
			Response: The HTTP response.

		Raises:
			Exception: If the request fails.
		"""
		try:
			response:Response = self.session.post(
				"http://{}:{}/machine-learning/logistic-regression/train".format(self.workerId,self.port),
				headers = kwargs.get("headers",{}),
				timeout = timeout
			)
			response.raise_for_status()
			return response
		except Exception as e:
			raise e
		
	def __logistic_regression_predict(self,timeout:int = 120, **kwargs) -> Response:
		"""POSTs a Logistic Regression prediction request to the worker.

		Args:
			timeout (int, optional): Request timeout in seconds. Defaults
				to 120.

		Returns:
			Response: The HTTP response.

		Raises:
			Exception: If the request fails.
		"""
		try:
			response:Response = self.session.post(
				"http://{}:{}/machine-learning/logistic-regression/predict".format(self.workerId,self.port),
				headers = kwargs.get("headers",{}),
				timeout = timeout
			)
			response.raise_for_status()
			return response
		except Exception as e:
			raise e
	
	def __pplr_train(self,timeout:int = 120, **kwargs) -> Response:
		"""POSTs a PQC-based PPLR training request to the worker.

		Args:
			timeout (int, optional): Request timeout in seconds. Defaults
				to 120.

		Returns:
			Response: The HTTP response.

		Raises:
			Exception: If the request fails.
		"""
		try:
			response:Response = self.session.post(
				"http://{}:{}/machine-learning/pplr/train".format(self.workerId,self.port),
				headers = kwargs.get("headers",{}),
				timeout = timeout
			)
			response.raise_for_status()
			return response
		except Exception as e:
			raise e
	
	def __pplr_predict(self,timeout:int = 120, **kwargs) -> Response:
		"""POSTs a PQC-based PPLR prediction request to the worker.

		Args:
			timeout (int, optional): Request timeout in seconds. Defaults
				to 120.

		Returns:
			Response: The HTTP response.

		Raises:
			Exception: If the request fails.
		"""
		try:
			response:Response = self.session.post(
				"http://{}:{}/machine-learning/pplr/predict".format(self.workerId,self.port),
				headers = kwargs.get("headers",{}),
				timeout = timeout
			)
			response.raise_for_status()
			return response
		except Exception as e:
			raise e