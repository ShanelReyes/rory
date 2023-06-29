import os, logging, sys
from flask import Flask,current_app
from rory.core.logger.Logger import create_logger
from rory.core.security.cryptosystem.liu import Liu
from rory.core.security.dataowner import DataOwner
from rory.core.interfaces.rorymanager import RoryManager,DumbRoryManager
from routes.clustering import clustering
from mictlanx.v3.client import Client 
from mictlanx.v3.services.xolo import Xolo
from mictlanx.v3.services.proxy import Proxy
from mictlanx.v3.services.replica_manger import ReplicaManager
from option import Some
from dotenv import load_dotenv
load_dotenv()

NODE_ID              = os.environ.get("NODE_ID","rory-client-0")
NODE_ID_METRICS      = "{}-Metrics".format(NODE_ID)
PORT                 = int(os.environ.get("NODE_PORT",3000))
IP_ADDR              = os.environ.get("NODE_IP_ADDR",NODE_ID)
RORY_MANAGER_IP_ADDR = os.environ.get("RORY_MANAGER_IP_ADDR","localhost")
RORY_MANAGER_PORT    = int(os.environ.get("RORY_MANAGER_PORT",6000))
DEBUG                = bool(int(os.environ.get("DEBUG",0)))
RELOAD               = bool(int(os.environ.get("RELOAD",0)))
LIU_ROUND            = bool(int(os.environ.get("LIU_ROUND","1")))
SERVER_IP_ADDR       = os.environ.get("SERVER_IP_ADDR","0.0.0.0")

#CREAR FOLDERS
# SINK_FOLDER   = "/rory/{}/sink".format(NODE_ID)
# SOURCE_FOLDER = "/rory/{}/source".format(NODE_ID)
# LOG_FOLDER    = "/rory/{}/log".format(NODE_ID)

SOURCE_PATH      = os.environ.get("SOURCE_PATH","/rory/source")
SINK_PATH        = os.environ.get("SINK_PATH","/rory/sink")
LOG_PATH         = os.environ.get("LOG_PATH","/rory/log")
try:
    os.makedirs(SOURCE_PATH,exist_ok = True)
    os.makedirs(SINK_PATH,  exist_ok = True)
    os.makedirs(LOG_PATH,   exist_ok = True)
except Exception as e:
    print("MAKE_FOLDER_ERROR",e)

LOGGER_NAME      = os.environ.get("LOGGER_NAME","rory-client-0")
MAX_ITERATIONS   = int(os.environ.get("MAX_ITERATIONS",10))
TESTING          = bool(int(os.environ.get("TESTING","0")))
M                = int(os.environ.get("M","3"))

# MICTLANX
MICTLANX_APP_ID                  = os.environ.get("MICTLANX_APP_ID")
MICTLANX_CLIENT_ID               = os.environ.get("MICTLANX_CLIENT_ID")
MICTLANX_SECRET                  = os.environ.get("MICTLANX_SECRET")
MICTLANX_PROXY_IP_ADDR           = os.environ.get("MICTLANX_PROXY_IP_ADDR","localhost")
MICTLANX_PROXY_PORT              = int(os.environ.get("MICTLANX_PROXY_PORT","8080"))
MICTLANX_XOLO_IP_ADDR            = os.environ.get("MICTLANX_XOLO_IP_ADDR","localhost")
MICTLANX_XOLO_PORT               = int(os.environ.get("MICTLANX_XOLO_PORT","10000"))
MICTLANX_REPLICA_MANAGER_IP_ADDR = os.environ.get("MICTLANX_REPLICA_MANAGER_IP_ADDR","localhost")
MICTLANX_REPLICA_MANAGER_PORT    = int(os.environ.get("MICTLANX_REPLICA_MANAGER_PORT","20000"))
MICTLANX_API_VERSION             = int(os.environ.get("MICTLANX_API_VERSION","3"))
MICTLANX_EXPIRES_IN              = os.environ.get("MICTLANX_EXPIRES_IN","15d")

replica_manager = ReplicaManager(
    ip_addr     = MICTLANX_REPLICA_MANAGER_IP_ADDR, 
    port        = MICTLANX_REPLICA_MANAGER_PORT, 
    api_version = Some(MICTLANX_API_VERSION)
)
xolo            = Xolo(
    ip_addr     = MICTLANX_XOLO_IP_ADDR, 
    port        = MICTLANX_XOLO_PORT, 
    api_version = Some(MICTLANX_API_VERSION)
)
proxy           = Proxy(
    ip_addr     = MICTLANX_PROXY_IP_ADDR, 
    port        = MICTLANX_PROXY_PORT, 
    api_version = Some(MICTLANX_API_VERSION)
)
STORAGE_CLIENT  = Client(
    app_id          = MICTLANX_APP_ID,
    client_id       = Some(MICTLANX_CLIENT_ID),
    secret          = MICTLANX_SECRET,
    replica_manager = replica_manager, 
    xolo            = xolo, 
    proxies         = [proxy], 
    expires_in      = Some(MICTLANX_EXPIRES_IN)
)
MANAGER = DumbRoryManager() if(TESTING) else RoryManager(
    hostname = RORY_MANAGER_IP_ADDR,
    port     = RORY_MANAGER_PORT,
)
LOGGER = create_logger(
    name                   = LOGGER_NAME,
    LOG_FILENAME           = NODE_ID,
    LOG_PATH               = LOG_PATH,
    console_handler_filter = lambda record: record.levelno == logging.DEBUG or record.levelno == logging.INFO or record.levelno == logging.ERROR,
    file_handler_filter    = lambda record:  record.levelno == logging.INFO,
)
# METRICSLOGGER = create_logger(
#     name                   = NODE_ID_METRICS,
#     LOG_FILENAME           = NODE_ID_METRICS,
#     LOG_PATH               = LOG_PATH,
#     console_handler_filter = lambda record: record.levelno == logging.DEBUG or record.levelno == logging.INFO,
#     file_handler_filter    = lambda record:  record.levelno == logging.INFO,
# )
LIU  = Liu(
    round = LIU_ROUND
)
DATAOWNER = DataOwner(
    m          = M,
    liu_scheme = LIU,
)

"""
Description:
    Function that create a context using Flask. Establishes the connection between client, manager and worker. 
"""
def create_app(*args):
    app = Flask(__name__)
    # Register blueprints
    app.register_blueprint(clustering)
    with app.app_context():
        current_app.config["request_counter"] = 0
        current_app.config["logger"]          = LOGGER
        #current_app.config["metricslogger"]   = METRICSLOGGER
        current_app.config["manager"]         = MANAGER
        current_app.config["liu"]             = LIU
        current_app.config["dataowner"]       = DATAOWNER
        current_app.config["SOURCE_PATH"]     = SOURCE_PATH
        current_app.config["SINK_PATH"]       = SINK_PATH
        current_app.config["NODE_ID"]         = NODE_ID
        current_app.config["LOG_PATH"]        = LOG_PATH
        current_app.config["MAX_ITERATIONS"]  = MAX_ITERATIONS
        current_app.config["STORAGE_CLIENT"]  = STORAGE_CLIENT
        # print(">>>>>>>>>>TESTING",TESTING)
        current_app.config["TESTING"]         = TESTING
    return app

"""
Description:
    Initialize create_app
"""
def start_app(*args):
    app = create_app(*args)
    app.run(host = SERVER_IP_ADDR, port = PORT,debug = DEBUG,use_reloader = RELOAD)

if __name__ == '__main__':
    try:
        start_app()
    except Exception as e:
        print(e)
        sys.exit(1)
    finally:
        STORAGE_CLIENT.logout()