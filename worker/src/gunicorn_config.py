import os
PORT       = int(os.environ.get("NODE_PORT","9000"))
bind       = "0.0.0.0:{}".format(PORT)
threads    = int(os.environ.get("GUNICORN_MAX_THREADS","1"))
workers    = int(os.environ.get("GUNICORN_WORKERS","2"))
timeout    = int(os.environ.get("GUNICORN_WORKER_TIMEOUT",3600))
print("Starting gunicorn on port {} with {} workers and {} threads".format(PORT, workers, threads))