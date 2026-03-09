import os
ENV_FILE_PATH = os.environ.get("ENV_FILE_PATH",".env.dev")
if os.path.exists(ENV_FILE_PATH):
    from dotenv import load_dotenv
    load_dotenv(ENV_FILE_PATH)
    
PORT       = int(os.environ.get("NODE_PORT","3001"))
bind       = "0.0.0.0:{}".format(PORT)
threads    = int(os.environ.get("GUNICORN_MAX_THREADS","1"))
workers    = int(os.environ.get("GUNICORN_WORKERS","1"))

timeout    = int(os.environ.get("GUNICORN_WORKER_TIMEOUT",3600))
