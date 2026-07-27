import requests
import json
from retry.api import retry_call
from typing import Dict,Tuple
from option import Result,Ok,Err


class Text:
	"""Helper class that wraps a worker ID into a JSON payload.

	Keyword Args:
		workerId (str): The identifier of the target worker.
	"""
	def __init__(self,**kwargs):
		workerId = kwargs.get("workerId")
		self.text = json.dumps({"workerId":workerId})


class DumbRoryManager:
	"""A dummy manager implementation used for manual testing.

	Instead of contacting a real TPDM manager service, it returns a Text
	object wrapping the supplied worker ID.
	"""
	def __init__(self):
		pass

	def sendRoryRequest(self,**kwargs):
		"""Mocks sending a clustering request to the manager.

		Keyword Args:
			workerId (str): Worker identifier. Defaults to "localhost".

		Returns:
			Text: A JSON-serialized response containing the worker ID.
		"""
		return Text(workerId = kwargs.get("workerId","localhost"))


class RoryManager:
	"""Manages the connection between the client and the TPDM manager.

	Constructs the base URL from hostname, port, and protocol, and exposes
	methods to discover available workers for secure clustering tasks.

	Available port ranges:
		- Client:  3000-5999
		- Manager: 6000-8999
		- Workers: 9000+

	Keyword Args:
		hostname (str): Manager hostname or IP. Defaults to "localhost".
		port (int): Manager port. Defaults to 6000.
		protocol (str): Protocol scheme. Defaults to "http".
		apiVersion (int): API version number. Defaults to 2.
	"""
	def __init__(self,**kwargs):
		self.hostname      = kwargs.get("hostname","localhost")
		self.port          = kwargs.get("port",6000)
		self.protocol      = kwargs.get("protocol","http")
		self.apiVersion    = kwargs.get("apiVersion",2)
		
		self.baseUrl       = "{}://{}:{}".format(self.protocol,self.hostname,self.port)
		self.clusteringURL = "{}/clustering/secure".format(self.baseUrl)
		

	def __get_worker(self,headers:Dict[str,str]):
		"""Sends a GET request to discover an available worker.

		Args:
			headers (Dict[str,str]): Additional HTTP headers to include.

		Returns:
			Response: The HTTP response from the manager.

		Raises:
			Exception: If the request fails.
		"""
		try:
			response = requests.get(
				self.clusteringURL,
				headers= {
					"Content-Type":"application/json",**headers
				}
			)
			return response
		except Exception as e:
			raise e

	def getWorker(self,**kwargs)->Result[Tuple[str,int],Exception]:
		"""Retrieves an available worker with retry logic.

		Calls __get_worker up to 100 times with exponential back-off.
		On success, parses the JSON response and returns the worker ID
		and port.

		Keyword Args:
			headers (dict): HTTP headers for the request.

		Returns:
			Result[Tuple[str,int], Exception]:
				On success, Ok((worker_id, worker_port)).
				On failure, Err(exception).
		"""
		try:
			headers        = kwargs.get("headers")
			res            = retry_call(self.__get_worker, fkwargs={"headers":headers} ,tries=100, delay=1, max_delay=3, jitter=0.3)
			res.raise_for_status()
			stringResponse = res.content.decode("utf-8")
			jsonResponse   = json.loads(stringResponse)
			return Ok((jsonResponse["worker_id"],jsonResponse["worker_port"]))
		except Exception as e:
			return Err(e)

