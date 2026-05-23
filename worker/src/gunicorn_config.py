import os
from dotenv import load_dotenv

x= os.environ.get("RORY_WORKER_ENV_FILE_PATH",".env")
if os.path.exists(x):
    load_dotenv(x)

PORT       = int(os.environ.get("NODE_PORT","9000"))
bind       = "0.0.0.0:{}".format(PORT)
threads    = int(os.environ.get("GUNICORN_MAX_THREADS","1"))
workers    = int(os.environ.get("GUNICORN_WORKERS","1"))
timeout    = int(os.environ.get("GUNICORN_WORKER_TIMEOUT",3600))
max_requests = int(os.environ.get("RORY_MAX_REQUESTS", "100"))
print("Starting gunicorn on port {} with {} workers and {} threads".format(PORT, workers, threads))