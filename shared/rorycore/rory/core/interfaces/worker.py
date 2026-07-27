from uuid import uuid4 
import time


class Worker:
	"""Represents a TPDM worker node and its runtime state.

	Keyword Args:
		workerId (str, optional): Unique worker identifier. Defaults to
			a UUID4 string.
		port (int): Port the worker listens on.
		balls (list, optional): Arbitrary list payload. Defaults to [].
		isStarted (bool, optional): Whether the worker has been started.
			Defaults to False.
		createdAt (float, optional): Unix timestamp when the worker was
			created. Defaults to the current time.
	"""
	def __init__(self,**kws):
		self.workerId  = kws.get("workerId",str(uuid4))
		self.port      = kws.get("port")
		self.balls     = kws.get("balls", [])
		self.isStarted = kws.get("isStarted",False)
		self.createdAt = kws.get("createdAt",time.time())