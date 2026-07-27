import requests
from rory.core.interfaces.createroryworker import CreateRoryWorker


class DumbReplicator:
	"""A dummy replicator used for manual testing.

	Does not contact any real orchestrator; returns a placeholder string
	for node deployments.
	"""
	def __init__(self,**kwargs):
		pass

	def deploy(self,**kwargs):
		"""Mocks a worker node deployment.

		Keyword Args:
			Any additional keyword arguments (ignored).

		Returns:
			str: A placeholder deployment confirmation string.
		"""
		return "FAKE_NODE_DEPLOYMENT"


class SecureReplicator:
	"""Interacts with the TPDM orchestrator to deploy and remove workers.

	Constructs the base REST API URL from the hostname, port, protocol,
	and API version, then exposes methods to create and remove worker
	nodes via HTTP POST requests.

	Keyword Args:
		hostname (str, optional): Orchestrator hostname. Defaults to
			"localhost".
		port (int, optional): Orchestrator port. Defaults to 1025.
		protocol (str, optional): Protocol scheme. Defaults to "http".
		apiVersion (int, optional): API version number. Defaults to 2.
	"""
	def __init__(self,**kws):
		self.hostname   = kws.get("hostname","localhost")
		self.port       = kws.get("port",1025)
		self.protocol   = kws.get("protocol","http")
		self.apiVersion = kws.get("apiVersion",2)
		self.url = "{}://{}:{}/api/v{}".format(self.protocol,self.hostname,self.port,self.apiVersion)
		self.createWorkerURL = "{}/generic/create".format(self.url)
		self.removeWorkerURL = lambda workerId: "{}/generic/remove/{}".format(self.url,workerId)

	def deploy(self,createWorker:CreateRoryWorker):
		"""Deploys a new worker node via the orchestrator API.

		Args:
			createWorker (CreateRoryWorker): The DTO describing the worker
				to be provisioned.

		Returns:
			Response: The HTTP response from the orchestrator.
		"""
		data      = createWorker.serialize()
		response  = requests.post( self.createWorkerURL,
			data =  data, 
			headers = {
				"Content-Type":"application/json",
				"Deferred": "true"
			}
		)
		return response

	def remove(self,**kws):
		"""Removes a worker node via the orchestrator API.

		Keyword Args:
			workerId (str): The identifier of the worker to remove.

		Returns:
			Response: The HTTP response from the orchestrator.
		"""
		workerId    = kws.get("workerId")
		response = requests.post( self.removeWorkerURL(workerId))
		return response