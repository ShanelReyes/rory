import os, logging, requests
from threading import Thread
from flask import Flask,current_app
from routes.clustering import clustering
from rory.core.logger.Logger import create_logger
from mictlanx.v3.client import Client 
from mictlanx.v3.services.xolo import Xolo
from mictlanx.v3.services.proxy import Proxy
from mictlanx.v3.services.replica_manger import ReplicaManager
from option import Some
from dotenv import load_dotenv
from retry.api import retry_call
load_dotenv()

NODE_ID                            = os.environ.get("NODE_ID","scw-0") 
PORT                               = int(os.environ.get("NODE_PORT",9000))
NODE_INDEX                         = int(os.environ.get("NODE_INDEX",0))
IP_ADDR                               = os.environ.get("NODE_IP_ADDR","0.0.0.0")
HOST_PORT                          = os.environ.get("HOST_PORT",PORT + NODE_INDEX)
# __________________________________________________

MAX_RETRIES     = int(os.environ.get("MAX_RETRIES",100))
SECURE_CLUSTERING_MANAGER_IP_ADDR = os.environ.get("SECURE_CLUSTERING_MANAGER_IP_ADDR","localhost")
SECURE_CLUSTERING_MANAGER_PORT     = int(os.environ.get("SECURE_CLUSTERING_MANAGER_PORT",6000))
LOG_PATH                           = os.environ.get("LOG_PATH","/log")
SINK_PATH                          = os.environ.get("SINK_PATH","/sink")
SOURCE_PATH                        = os.environ.get("SOURCE_PATH","/source")
DEBUG                              = bool(int(os.environ.get("DEBUG",0)))
RELOAD                             = bool(int(os.environ.get("RELOAD",0)))
# SECRET_KEY                         = os.environ.get("SECRET_KEY","schanel")
# alpha.tamps.cinvestav.mx

MICTLANX_APP_ID                    = os.environ.get("MICTLANX_APP_ID")
MICTLANX_CLIENT_ID                 = os.environ.get("MICTLANX_CLIENT_ID")
MICTLANX_SECRET                    = os.environ.get("MICTLANX_SECRET")
MICTLANX_PROXY_IP_ADDR             = os.environ.get("MICTLANX_PROXY_IP_ADDR","localhost")
MICTLANX_PROXY_PORT                = int(os.environ.get("MICTLANX_PROXY_PORT","8080"))
MICTLANX_XOLO_IP_ADDR              = os.environ.get("MICTLANX_XOLO_IP_ADDR","localhost")
MICTLANX_XOLO_PORT                 = int(os.environ.get("MICTLANX_XOLO_PORT","10000"))
MICTLANX_REPLICA_MANAGER_IP_ADDR   = os.environ.get("MICTLANX_REPLICA_MANAGER_IP_ADDR", "localhost")
MICTLANX_REPLICA_MANAGER_PORT      = int(os.environ.get("MICTLANX_REPLICA_MANAGER_PORT", "20000"))
MICTLANX_API_VERSION               = int(os.environ.get("MICTLANX_API_VERSION","3"))
MICTLANX_EXPIRES_IN                = os.environ.get("MICTLANX_EXPIRES_IN","15d")

replica_manager = ReplicaManager(ip_addr = MICTLANX_REPLICA_MANAGER_IP_ADDR, port = MICTLANX_REPLICA_MANAGER_PORT, api_version = Some(MICTLANX_API_VERSION))
xolo            = Xolo(ip_addr = MICTLANX_XOLO_IP_ADDR, port = MICTLANX_XOLO_PORT, api_version = Some(MICTLANX_API_VERSION))
proxy           = Proxy(ip_addr = MICTLANX_PROXY_IP_ADDR, port = MICTLANX_PROXY_PORT, api_version = Some(MICTLANX_API_VERSION))
STORAGE_CLIENT  = Client(
    app_id          = MICTLANX_APP_ID,
    client_id       = Some(MICTLANX_CLIENT_ID),
    secret          = MICTLANX_SECRET,
    replica_manager = replica_manager, 
    xolo            = xolo, 
    proxy           = proxy, 
    expires_in      = Some(MICTLANX_EXPIRES_IN)
)



LOGGER = create_logger (
  name                   = NODE_ID,
  LOG_FILENAME           = NODE_ID,
  LOG_PATH               = LOG_PATH,
  console_handler_filter = lambda record: record.levelno == logging.INFO or record.levelno == logging.ERROR,
  file_handler_filter    = lambda record: record.levelno == logging.DEBUG or record.levelno == logging.INFO,
)


"""
Description:
  Function that create a context using Flask. Establishes the connection between client, manager and worker. 
"""
def create_app():
    app = Flask(__name__)
    # Register blueprints
    app.register_blueprint(clustering) # SkMeans routes / DBSkmeans routes
    #app.register_blueprint(index)
    # app.secret_key = SECRET_KEY
    with app.app_context():
        current_app.config["request_counter"]         = 0
        current_app.config["NODE_PORT"]               = PORT
        current_app.config["SINK_PATH"]               = SINK_PATH
        current_app.config["SOURCE_PATH"]             = SOURCE_PATH
        current_app.config["logger"]                  = LOGGER
        # current_app.config["STORAGE_SYSTEM_HOSTNAME"] = STORAGE_SYSTEM_HOSTNAME
        # current_app.config["STORAGE_SYSTEM_PORT"]     = STORAGE_SYSTEM_PORT
        current_app.config["NODE_ID"]                 = NODE_ID
        current_app.config["events"]                  = {}
        # current_app.config["SESSION_TYPE"]            = "filesystem"
        current_app.config["LOG_PATH"]                = LOG_PATH
        current_app.config["STORAGE_CLIENT"]          = STORAGE_CLIENT
    return app


"""
Description:
  Initialize worker
"""

def started_completed():
  def __inner():
    try:
      response = requests.post(
            "http://{}:{}/workers/started".format(SECURE_CLUSTERING_MANAGER_IP_ADDR,SECURE_CLUSTERING_MANAGER_PORT),
            headers = {"Worker-Id":NODE_ID,"Worker-Port":str(HOST_PORT)}
      )
      print("RESPONSE",response)
      response.raise_for_status()
      LOGGER.debug("STARTED_RESPONSE {}".format(response.content))
      return response
    except Exception as e:
      print("ERROR",e)
      LOGGER.error(str(e))
      raise e
  result = retry_call(__inner, tries=MAX_RETRIES, delay=1,backoff=1)
  print("RESILT", result)
  LOGGER.debug("WORKER_STARTED_RESPONSE {}".format(result))
  # print("WORKER")


if __name__ == '__main__':
  app = create_app()
  t1 = Thread(target= started_completed, daemon= True, args = () )
  t1.start()
  app.run(host = IP_ADDR, port = PORT,debug = DEBUG,use_reloader = RELOAD)