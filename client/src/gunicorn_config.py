import os
PORT       = int(os.environ.get("NODE_PORT","3000"))
bind       = "0.0.0.0:{}".format(PORT)
pythonpath = os.environ.get("CUSTOM_PYTHON_PATH",'/app')  
threads    = int(os.environ.get("MAX_THREADS","1"))
timeout    = 300