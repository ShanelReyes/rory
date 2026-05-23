import os

from dotenv import load_dotenv

x= os.environ.get("RORY_MANAGER_ENV_FILE_PATH",".env")
if os.path.exists(x):
    load_dotenv(x)


PORT       = int(os.environ.get("NODE_PORT","6000"))
bind       = "0.0.0.0:{}".format(PORT)
threads    = int(os.environ.get("GUNICORN_MAX_THREADS","1"))
workers    = int(os.environ.get("GUNICORN_WORKERS","1"))

timeout    = int(os.environ.get("GUNICORN_WORKER_TIMEOUT",3600))