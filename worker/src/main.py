import os, logging, requests, sys
from threading import Thread
from flask import Flask,current_app
from routes.clustering import clustering
from rory.core.logger.Logger import create_logger
from mictlanx.v4.client import Client
from mictlanx.utils.index import Utils
from option import Some
from dotenv import load_dotenv
from retry.api import retry_call
app = Flask(__name__)
load_dotenv()

NODE_ID              = os.environ.get("NODE_ID","rory-worker-0") 
PORT                 = int(os.environ.get("NODE_PORT",9000))
NODE_INDEX           = int(os.environ.get("NODE_INDEX",0))
HOST_PORT            = os.environ.get("HOST_PORT",PORT + NODE_INDEX)
MAX_RETRIES          = int(os.environ.get("MAX_RETRIES",100))
RORY_MANAGER_PORT    = int(os.environ.get("RORY_MANAGER_PORT",6000))
DEBUG                = bool(int(os.environ.get("DEBUG",0)))
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

MICTLANX_TIMEOUT                 = int(os.environ.get("MICTLANX_TIMEOUT",120))
MICTLANX_APP_ID                  = os.environ.get("MICTLANX_APP_ID" "APP_ID")
MICTLANX_CLIENT_ID               = os.environ.get("MICTLANX_CLIENT_ID",NODE_ID)
MICTLANX_SECRET                  = os.environ.get("MICTLANX_SECRET","SECRET")
MICTLANX_PROXY_IP_ADDR           = os.environ.get("MICTLANX_PROXY_IP_ADDR","localhost")
MICTLANX_PROXY_PORT              = int(os.environ.get("MICTLANX_PROXY_PORT","8080"))
MICTLANX_XOLO_IP_ADDR            = os.environ.get("MICTLANX_XOLO_IP_ADDR","localhost")
MICTLANX_XOLO_PORT               = int(os.environ.get("MICTLANX_XOLO_PORT","10000"))
MICTLANX_REPLICA_MANAGER_IP_ADDR = os.environ.get("MICTLANX_REPLICA_MANAGER_IP_ADDR", "localhost")
MICTLANX_REPLICA_MANAGER_PORT    = int(os.environ.get("MICTLANX_REPLICA_MANAGER_PORT", "20000"))
MICTLANX_API_VERSION             = int(os.environ.get("MICTLANX_API_VERSION","3"))
MICTLANX_EXPIRES_IN              = os.environ.get("MICTLANX_EXPIRES_IN","15d")

STORAGE_CLIENT  = Client(
    client_id       = MICTLANX_CLIENT_ID,
    peers           = list(Utils.peers_from_str(os.environ.get("MICTLANX_PEERS", "mictlanx-peer-0:localhost:7000"))),
    debug           = False,
    daemon          = False,
    max_workers     = int(os.environ.get("MICTLANX_CLIENT_WORKERS","4"))
)
LOGGER = create_logger (
    name                   = NODE_ID,
    LOG_FILENAME           = NODE_ID,
    LOG_PATH               = LOG_PATH,
    console_handler_filter = lambda record: record.levelno == logging.INFO or record.levelno == logging.ERROR or record.levelno == logging.DEBUG,
    file_handler_filter    = lambda record: record.levelno == logging.DEBUG or record.levelno == logging.INFO,
)


"""
Description:
  Function that create a context using Flask. Establishes the connection between client, manager and worker. 
"""
def create_app():
    
    # Register blueprints
    app.register_blueprint(clustering) # SkMeans routes / DBSkmeans routes
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
    # return app


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
      LOGGER.debug("STARTED_RESPONSE {}".format(response.content))
      return response
    except Exception as e:
      print("ERROR",e)
      LOGGER.error(str(e))
      raise e
  result = retry_call(__inner, tries=MAX_RETRIES, delay=1,backoff=1)
  LOGGER.debug("WORKER_STARTED_RESPONSE {}".format(result))

if __name__ == 'main' or __name__ == "__main__":
  try:
    create_app()
    t1 = Thread(target= started_completed, daemon= True, args = () )
    t1.start()
  except Exception as e:
    print(e)
    sys.exit(1)
  # finally: