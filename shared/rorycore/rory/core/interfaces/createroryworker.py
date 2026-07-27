import json


class CreateRoryWorker:
	"""Data Transfer Object (DTO) for provisioning a new worker node.

	Describes the Docker container configuration — image, network, ports,
	volumes, resource limits, environment variables, and labels — used by
	the TPDM orchestrator to deploy a new worker.

	Keyword Args:
		nodeId (str): Unique node identifier.
		nodeIndex (int, optional): Numeric index of the node. Defaults to 0.
		image (str, optional): Docker image name. Defaults to
			"shanelreyes/rory:worker".
		network (dict, optional): Docker network configuration. Defaults to
			{"name":"test","driver":"bridge"}.
		ports (dict): Port mapping between host and container.
		DOCKER_SINK_PATH (str, optional): Sink directory inside the
			container. Defaults to "{node_path}/sink".
		DOCKER_LOG_PATH (str, optional): Logs directory inside the
			container. Defaults to "{node_path}/logs".
		HOST_LOG_PATH (str, optional): Host-side logs directory. Defaults
			to "{node_path}/log".
		HOST_SINK_PATH (str, optional): Host-side sink directory. Defaults
			to "{node_path}/test/sink/{nodeId}".
		RORY_MANAGER_HOSTNAME (str, optional): Manager hostname. Defaults
			to "scm-0".
		RORY_MANAGER_PORT (str, optional): Manager port. Defaults to "6000".
		envs (dict, optional): Additional environment variables. Merged
			with defaults.
		labels (dict, optional): Docker labels. Defaults to {}.
		volumes (dict, optional): Host-to-container volume bindings.
		resources (dict, optional): CPU and memory resource constraints.
	"""
	def __init__(self,**kws):
		self.nodeId      = kws.get("nodeId")
		self.nodeIndex   = kws.get("nodeIndex",0)
		self.image       = kws.get("image","shanelreyes/rory:worker")
		self.network     = kws.get("network", {"name":"test","driver":"bridge"})
		self.ports       = kws.get("ports")
		self.node_path   = "/rory/{}".format(self.nodeId)
		DOCKER_SINK_PATH = kws.get("DOCKER_SINK_PATH","{}/sink".format(self.node_path))
		DOCKER_LOG_PATH  = kws.get("DOCKER_LOG_PATH","{}/logs".format(self.node_path))
		HOST_LOG_PATH    = kws.get("HOST_LOG_PATH","{}/log".format(self.node_path))
		HOST_SINK_PATH   = kws.get("HOST_SINK_PATH","{}/test/sink/".format(self.node_path))+self.nodeId

		default_envs   = {
			"NODE_ID": self.nodeId,
			"NODE_PORT": str(self.ports["docker"]),
			"HOST_PORT": str(self.ports["host"]),
			"NODE_INDEX": str(self.nodeIndex),
			"RORY_MANAGER_HOSTNAME": kws.get("RORY_MANAGER_HOSTNAME","scm-0"),
			"RORY_MANAGER_PORT": str(kws.get("RORY_MANAGER_PORT","6000")),
			"LOG_PATH": DOCKER_LOG_PATH,
			"SINK_PATH": DOCKER_SINK_PATH
		}

		_envs          = kws.get("envs",{})
		self.envs      = {**default_envs,**_envs}
		self.labels    = kws.get("labels",{})
		self.volumes   = kws.get("volumes",{
			 HOST_LOG_PATH: DOCKER_LOG_PATH,
			HOST_SINK_PATH: DOCKER_SINK_PATH
		})
		self.resources = kws.get("resources",{
			"cpuCount":1,
			"cpuPeriod":0,
			"cpuQuota":0,
			"memory":1000000000
		})

	def serialize(self):
		"""Serializes the DTO to a JSON string for transmission.

		Returns:
			str: JSON representation of the worker creation request.
		"""
		return json.dumps(self.__dict__)


	def __str__(self):
		"""Returns a human-readable string representation.

		Returns:
			str: The dictionary representation as a string.
		"""
		return str(self.__dict__)
