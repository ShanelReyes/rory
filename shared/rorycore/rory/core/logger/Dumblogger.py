class DumbLogger:
	"""A no-op logger that suppresses all log output.

	Provides the standard logging interface (debug, info, error) but
	discards all messages. Useful for testing or when logging is disabled.
	"""
	def debug(self,x):
		"""Suppresses a debug-level log message.

		Args:
			x: The message to log (ignored).
		"""
		return

	def info(self,x):
		"""Suppresses an info-level log message.

		Args:
			x: The message to log (ignored).
		"""
		return

	def error(self,x):
		"""Suppresses an error-level log message.

		Args:
			x: The message to log (ignored).
		"""
		return