import os
PORT       = int(os.environ.get("NODE_PORT","6000"))
bind       = "0.0.0.0:{}".format(PORT)
threads    = int(os.environ.get("GUNICORN_MAX_THREADS","1"))
workers    = int(os.environ.get("GUNICORN_WORKERS","2"))

timeout    = int(os.environ.get("GUNICORN_WORKER_TIMEOUT",3600))