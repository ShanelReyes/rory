class LoggerMetrics:
	"""Structured log entry capturing timing and metadata of a TPDM operation.

	Records the operation type, matrix and worker identifiers, algorithm
	name, arrival/end/service times, latency, and clustering parameters
	(k, m, n_iterations). Additional positional and keyword arguments are
	preserved for future extensibility.

	Args:
		operation_type (str): Label identifying the operation (e.g.
			"ENCRYPT"). Defaults to "OPERATION_TYPE".
		matrix_id (str): Identifier of the matrix being processed. Defaults
			to "MATRIX".
		worker_id (str): Identifier of the worker node. Defaults to
			"WORKER_ID".
		algorithm (str): Algorithm name. Defaults to "ALGORITHM".
		arrival_time (int): Request arrival timestamp. Defaults to 0.
		end_time (int): Request completion timestamp. Defaults to 0.
		service_time (int): Time spent servicing the request. Defaults to 0.
		latency (int): Network latency. Defaults to 0.
		k_value (int): Number of clusters. Defaults to 0.
		m_value (int): Number of SK attributes. Defaults to 0.
		n_iterations (int): Number of iterations executed. Defaults to 0.
		*args: Additional positional arguments stored as-is.
		**kwargs: Additional keyword arguments stored as-is.
	"""
	def __init__(self,
				 *args,
				 operation_type:str = "OPERATION_TYPE",
				 matrix_id:str      = "MATRIX",
				 worker_id:str      = "WORKER_ID",
				 algorithm:str      = "ALGORITHM",
				 arrival_time:int   = 0, 
				 end_time:int       = 0, 
				 service_time:int   = 0,
				 latency:int        = 0,
				 k_value:int        = 0,
				 m_value:int        = 0,
				 n_iterations:int   = 0,
				 **kwargs,

				 
				 ):
		self.operation_type = operation_type
		self.matrix_id      = matrix_id
		self.worker_id      = worker_id
		self.algorithm      = algorithm 
		self.arrival_time   = arrival_time
		self.end_time       = end_time
		self.service_time   = service_time
		self.latency        = latency
		self.k_value        = k_value
		self.m_value        = m_value
		self.n_iterations   = n_iterations
		self.args           = args
		self.kwargs         = kwargs
	def __str__(self):
		"""Returns a CSV representation of the log entry.

		Returns:
			str: Comma-separated values of all fields, followed by any
				extra positional and keyword arguments.
		"""
		value:str = "{},{},{},{},{},{},{},{},{},{},{}".format(
			self.operation_type,
			self.matrix_id,
			self.worker_id,
			self.algorithm,
			self.arrival_time, 
			self.end_time, 
			self.service_time,
			self.latency,
			self.k_value,
			self.m_value,
			self.n_iterations
			)
		y       = ",".join(self.args)
		value_y = value if len(self.args) == 0 else "{},{}".format(value,y)
		values  = list(map(str,self.kwargs.values()))
		x       = ",".join(values)
		value_x = value if len(values) == 0 else "{},{}".format(value_y,x)
		return value_x


if __name__ == "__main__":
	lm = LoggerMetrics(operation_type="ENCRYPT",arrival_time=0, end_time=0, service_time=1,test="AL_ULTIMO_KWARGS")