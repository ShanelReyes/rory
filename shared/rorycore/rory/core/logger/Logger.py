import sys
import logging


def create_logger(**kwargs):
	"""Creates a configured logger with console and file handlers.

	Sets up a Python logging.Logger with a console stream handler
	(filtered to DEBUG level) and a file handler writing CSV-formatted
	logs. An optional error log file handler can also be added.

	Keyword Args:
		name (str, optional): Logger name. Defaults to "default".
		LOG_PATH (str): Directory where log files are written.
		LOG_FILENAME (str): Base filename for the output files.
		add_error_log (bool, optional): Whether to create a separate
			error log file. Defaults to True.
		console_handler_filter (callable, optional): Filter function for
			console output. Defaults to DEBUG-only.
		file_handler_filter (callable, optional): Filter function for
			file output. Defaults to INFO-only.

	Returns:
		logging.Logger: The configured logger instance.
	"""
	name                   = kwargs.get("name","default")
	LOG_PATH               = kwargs.get("LOG_PATH")
	LOG_FILENAME           = kwargs.get("LOG_FILENAME")
	add_error_log          = kwargs.get("add_error_log",True)
	console_handler_filter = kwargs.get("console_handler_filter", lambda record: record.levelno == logging.DEBUG)
	file_handler_filter    = kwargs.get("file_handler_filter", lambda record: record.levelno == logging.INFO)
	FORMAT         = '%(threadName)s,%(message)s'
	formatter      = logging.Formatter(FORMAT,"%Y-%m-%d,%H:%M:%S")
	filename       = "{}/{}.csv".format(LOG_PATH,LOG_FILENAME)
	errorFilename  = "{}/{}-error.log".format(LOG_PATH,LOG_FILENAME)
	logger         = logging.getLogger(name)
	consolehanlder = logging.StreamHandler(sys.stdout)
	consolehanlder.setFormatter(formatter)
	consolehanlder.setLevel(logging.DEBUG)
	consolehanlder.addFilter(console_handler_filter)
	filehandler = logging.FileHandler(filename= filename)
	filehandler.setFormatter(formatter)
	filehandler.setLevel(logging.INFO)
	filehandler.addFilter(file_handler_filter)
	if(add_error_log):
		errorFilehandler = logging.FileHandler(filename=errorFilename)
		errorFilehandler.setFormatter(formatter)
		errorFilehandler.setLevel(logging.ERROR)
		errorFilehandler.addFilter(lambda record: record.levelno == logging.ERROR)
		logger.addHandler(errorFilehandler)
	logger.addHandler(filehandler)
	logger.addHandler(consolehanlder)
	logger.setLevel(logging.DEBUG)
	return logger