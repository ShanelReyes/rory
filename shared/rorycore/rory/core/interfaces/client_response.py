import numpy as np
import json
from requests import Response


class ClientResponse:
	"""Interface representing the result returned from a TPDM worker.

	Encapsulates the clustering/classification result along with metadata
	such as timing, algorithm used, HTTP headers, and status code.

	Keyword Args:
		label_vector (list, optional): Cluster or class labels assigned to
			each data point. Defaults to [].
		service_time (float, optional): Time spent by the service. Defaults
			to 0.
		response_time (float, optional): Total end-to-end response time.
			Defaults to 0.
		algorithm (str, optional): The algorithm used. Defaults to None.
		headers (dict, optional): HTTP response headers. Defaults to {}.
		status (int, optional): HTTP status code. Defaults to 0.
	"""
	def __init__(self,**kwargs):
		self.label_vector  = kwargs.get("label_vector",[])
		self.service_time  = kwargs.get("service_time",0)
		self.response_time = kwargs.get("response_time",0)
		self.algorithm    = kwargs.get("algorithm",None)
		self.headers      = kwargs.get("headers",{})
		self.status       = kwargs.get("status",0)

	@staticmethod
	def fromResponse(response:Response) -> 'ClientResponse':
		"""Parses an HTTP response into a ClientResponse instance.

		Decodes the JSON body, converts the label_vector entry to a NumPy
		array, and preserves the HTTP headers and status code.

		Args:
			response (Response): An HTTP response object from the requests
				library.

		Returns:
			ClientResponse: A fully populated instance built from the
				response content and metadata.
		"""
		jsonString = response.content.decode("utf-8")
		jsonResponse = json.loads(jsonString)
		jsonResponse["label_vector"] = np.array(jsonResponse.get("label_vector",[]))

		return ClientResponse(
			**jsonResponse, 
			headers = response.headers,
		    status = response.status_code
		)

