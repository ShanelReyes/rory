import os, logging, requests, sys
from threading import Thread
from flask import Flask,current_app
from routes.clustering import clustering
from routes.classification import classification
from mictlanx.v4.client import Client
from mictlanx.utils.index import Utils
from dotenv import load_dotenv
from retry.api import retry_call
from mictlanx.logger.log import Log
app = Flask(__name__)
DEBUG                 = bool(int(os.environ.get("RORY_DEBUG",1)))
if DEBUG:
    load_dotenv(os.environ.get("ENV_FILE_PATH","/rory/envs/.worker.env"))

NODE_ID              = os.environ.get("NODE_ID","rory-worker-0") 
PORT                 = int(os.environ.get("NODE_PORT",9000))
NODE_INDEX           = int(os.environ.get("NODE_INDEX",0))
HOST_PORT            = os.environ.get("HOST_PORT",PORT + NODE_INDEX)
MAX_RETRIES          = int(os.environ.get("MAX_RETRIES",100))
RORY_MANAGER_PORT    = int(os.environ.get("RORY_MANAGER_PORT",6000))
RELOAD               = bool(int(os.environ.get("RELOAD",0)))
RORY_MANAGER_IP_ADDR = os.environ.get("RORY_MANAGER_IP_ADDR","localhost")
IP_ADDR              = os.environ.get("NODE_IP_ADDR",NODE_ID)
SERVER_IP_ADDR       = os.environ.get("SERVER_IP_ADDR","0.0.0.0")

#CREAR FOLDERS
SOURCE_PATH      = os.environ.get("SOURCE_PATH","/rory/source")
SINK_PATH        = os.environ.get("SINK_PATH","/rory/sink")
LOG_PATH         = os.environ.get("LOG_PATH","/rory/log")
try:
    os.makedirs(SOURCE_PATH,exist_ok = True)
    os.makedirs(SINK_PATH,  exist_ok = True)
    os.makedirs(LOG_PATH,   exist_ok = True)
except Exception as e:
    print("MAKE_FOLDER_ERROR",e)

MICTLANX_TIMEOUT             = int(os.environ.get("MICTLANX_TIMEOUT",120))
MICTLANX_APP_ID              = os.environ.get("MICTLANX_APP_ID" "APP_ID")
MICTLANX_CLIENT_ID           = os.environ.get("MICTLANX_CLIENT_ID","{}_mictlanx".format(NODE_ID))
MICTLANX_SECRET              = os.environ.get("MICTLANX_SECRET","SECRET")
MICTLANX_XOLO_IP_ADDR        = os.environ.get("MICTLANX_XOLO_IP_ADDR","localhost")
MICTLANX_XOLO_PORT           = int(os.environ.get("MICTLANX_XOLO_PORT","10000"))
MICTLANX_API_VERSION         = int(os.environ.get("MICTLANX_API_VERSION","3"))
MICTLANX_EXPIRES_IN          = os.environ.get("MICTLANX_EXPIRES_IN","15d")
MICTLANX_ROUTERS               = os.environ.get("MICTLANX_ROUTERS", "mictlanx-router-0:localhost:60666")
MICTLANX_DEBUG               = bool(int(os.environ.get("MICTLANX_DEBUG",0)))
MICTLANX_DAEMON              = bool(int(os.environ.get("MICTLANX_DAEMON",1)))
MICTLANX_SHOW_METRICS        = bool(int(os.environ.get("MICTLANX_SHOW_METRICS",0)))
MICTLANX_MAX_WORKERS         = int(os.environ.get("MICTLANX_MAX_WORKERS","4"))
MICTLANX_CLIENT_LB_ALGORITHM = os.environ.get("MICTLANX_CLIENT_LB_ALGORITHM","2CHOICES_UF")

STORAGE_CLIENT  = Client(
    client_id       = MICTLANX_CLIENT_ID,
    routers         = list(Utils.routers_from_str(MICTLANX_ROUTERS)),
    debug           = MICTLANX_DEBUG,
    max_workers     = MICTLANX_MAX_WORKERS,
    lb_algorithm    = MICTLANX_CLIENT_LB_ALGORITHM,
    bucket_id       = os.environ.get("MICTLANX_BUCKET_ID","rory"),
    log_output_path = os.environ.get("MICTLANX_OUTPUT_PATH","/rory/mictlanx")
)

def console_handler_filter(record:logging.LogRecord):
    if DEBUG:
        return True
    elif not DEBUG and (record.levelno == logging.INFO or record.levelno == logging.ERROR):
        return True
    else:
        return False
    
LOGGER = Log(
    name                   = NODE_ID,
    path                   = LOG_PATH,
    console_handler_filter = console_handler_filter,
    interval               = 24,
    when                   = "h"
)

"""
Description:
  Function that create a context using Flask. Establishes the connection between client, manager and worker. 
"""
def create_app():
    # Register blueprints
    app.register_blueprint(clustering) # SkMeans routes / DBSkmeans routes
    app.register_blueprint(classification)
    with app.app_context():
        current_app.config["request_counter"]  = 0
        current_app.config["NODE_PORT"]        = PORT
        current_app.config["SINK_PATH"]        = SINK_PATH
        current_app.config["SOURCE_PATH"]      = SOURCE_PATH
        current_app.config["logger"]           = LOGGER
        current_app.config["NODE_ID"]          = NODE_ID
        current_app.config["events"]           = {}
        current_app.config["LOG_PATH"]         = LOG_PATH
        current_app.config["STORAGE_CLIENT"]   = STORAGE_CLIENT
        current_app.config["MICTLANX_TIMEOUT"] = MICTLANX_TIMEOUT

"""
Description:
  Initialize worker
"""
def started_completed():
  def __inner():
    try:
      response = requests.post(
            "http://{}:{}/workers/started".format(RORY_MANAGER_IP_ADDR,RORY_MANAGER_PORT),
            headers = {"Worker-Id":NODE_ID,"Worker-Port":str(PORT)},
            timeout = 300
      )
      response.raise_for_status()
      LOGGER.debug({
         "event":"MANAGER.STARTED_COMPLETED",
         "manager_ip_addr":RORY_MANAGER_IP_ADDR,
         "manager_port":RORY_MANAGER_PORT,
         "node_id":NODE_ID,
         "port":PORT
      })
      return response
    except Exception as e:
      LOGGER.error({
         "msg":str(e)
      })
      raise e
  result = retry_call(__inner, tries=MAX_RETRIES, delay=1,backoff=1)

if __name__ == 'main' or __name__ == "__main__":
  try:
    LOGGER.debug({
        "event":"WORKER_STARTED",
        "node_id":NODE_ID,
        "port":PORT,
        "debug":DEBUG,
        "mictlanx_max_workers":MICTLANX_MAX_WORKERS,
        "mictlanx_timeout":MICTLANX_TIMEOUT,
    })
    create_app()
    t1 = Thread(target= started_completed, daemon= True, args = () )
    t1.start()
  except Exception as e:
    print(e)
    sys.exit(1)