import os, logging, sys, time
from flask import Flask,current_app
from rory.core.logger.Logger import create_logger
from rory.core.security.cryptosystem.liu import Liu
from rory.core.security.dataowner import DataOwner
from rory.core.interfaces.rorymanager import RoryManager,DumbRoryManager
from routes.clustering import clustering
from mictlanx.v4.client import Client
from mictlanx.utils.index import Utils
from mictlanx.v3.services.xolo import Xolo
from option import Some
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor
app = Flask(__name__)
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
NUM_CHUNKS           = int(os.environ.get("NUM_CHUNKS",4)) #Chunks for dataset
MAX_WORKERS          = int(os.environ.get("MAX_WORKERS",4)) #Total of process for encryption

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

LOGGER_NAME    = os.environ.get("LOGGER_NAME","rory-client-0")
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS",10))
TESTING_ENV    = os.environ.get("TESTING","1")
TESTING        = bool(int(TESTING_ENV))
M              = int(os.environ.get("M","3"))

# MICTLANX
MICTLANX_TIMEOUT                 = int(os.environ.get("MICTLANX_TIMEOUT",120))
MICTLANX_APP_ID                  = os.environ.get("MICTLANX_APP_ID","APP_ID")
MICTLANX_CLIENT_ID               = os.environ.get("MICTLANX_CLIENT_ID",NODE_ID)
MICTLANX_SECRET                  = os.environ.get("MICTLANX_SECRET","SECRET")
MICTLANX_XOLO_IP_ADDR            = os.environ.get("MICTLANX_XOLO_IP_ADDR","localhost")
MICTLANX_XOLO_PORT               = int(os.environ.get("MICTLANX_XOLO_PORT","10000"))
MICTLANX_API_VERSION             = int(os.environ.get("MICTLANX_API_VERSION","3"))
MICTLANX_EXPIRES_IN              = os.environ.get("MICTLANX_EXPIRES_IN","15d")
MICTLANX_PEERS                   = os.environ.get("MICTLANX_PEERS", "mictlanx-peer-0:localhost:7000")

xolo            = Xolo(
    ip_addr     = Some(MICTLANX_XOLO_IP_ADDR), 
    port        = Some(MICTLANX_XOLO_PORT), 
    api_version = Some(MICTLANX_API_VERSION)
)
STORAGE_CLIENT  = Client(
    client_id   = MICTLANX_CLIENT_ID,
    peers       = list(Utils.peers_from_str(MICTLANX_PEERS)),
    daemon      = False,
    debug       = False,
    max_workers = int(os.environ.get("MICTLANX_CLIENT_WORKERS","4"))
)
MANAGER = RoryManager(
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
LIU  = Liu(
    round = LIU_ROUND
)
DATAOWNER = DataOwner(
    m          = M,
    liu_scheme = LIU,
)

cores                   = os.cpu_count()
max_workers             = cores if MAX_WORKERS > cores else MAX_WORKERS
executor = ProcessPoolExecutor(max_workers=max_workers)
"""
Description:
    Function that create a context using Flask. Establishes the connection between client, manager and worker. 
"""
def create_app(*args):
    
    # Register blueprints
    app.register_blueprint(clustering)
    with app.app_context():
        current_app.config["request_counter"]  = 0
        current_app.config["logger"]           = LOGGER
        current_app.config["manager"]          = MANAGER
        current_app.config["liu"]              = LIU
        current_app.config["dataowner"]        = DATAOWNER
        current_app.config["SOURCE_PATH"]      = SOURCE_PATH
        current_app.config["SINK_PATH"]        = SINK_PATH
        current_app.config["NODE_ID"]          = NODE_ID
        current_app.config["LOG_PATH"]         = LOG_PATH
        current_app.config["MAX_ITERATIONS"]   = MAX_ITERATIONS
        current_app.config["STORAGE_CLIENT"]   = STORAGE_CLIENT
        current_app.config["TESTING"]          = TESTING
        current_app.config["NUM_CHUNKS"]       = NUM_CHUNKS
        current_app.config["MAX_WORKES"]       = max_workers
        current_app.config["MICTLANX_TIMEOUT"] = MICTLANX_TIMEOUT
        current_app.config["executor"]         = executor
    # return app

"""
Description:
    Initialize create_app
"""
if __name__ == 'main' or __name__ == "__main__":
    try:
        create_app()
    except Exception as e:
        print(e)
        executor.shutdown()
        STORAGE_CLIENT.shutdown()
        sys.exit(1)