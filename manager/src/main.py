import os, logging
from flask import Flask,current_app
from routes.clustering import clustering
from routes.workers import workers
from load_balancing.round_robin import RoundRobin
from load_balancing.two_choices import TwoChoices
from load_balancing.random import  Random
from deployworkers import deploy_nodes
from mictlanx.v3.services.summoner import Summoner
from mictlanx.v3.services.xolo import Xolo 
from mictlanx.v3.interfaces.payloads import SummonContainerPayload,ExposedPort
from option import NONE,Some
from rory.core.logger.Logger import create_logger
from dotenv import load_dotenv
load_dotenv()

NODE_ID           = os.environ.get("NODE_ID","rory-manager-0")
IP_ADDR           = os.environ.get("NODE_IP_ADDR",NODE_ID)
PORT              = int(os.environ.get("NODE_PORT",6000))
NODE_PREFIX       = os.environ.get("NODE_PREFIX","rory-worker-")
init_workers      = int(os.environ.get("INIT_WORKERS",1))
MAX_WORKERS       = int(os.environ.get("MAX_WORKERS",4))
DOCKER_IMAGE_NAME = os.environ.get("DOCKER_IMAGE_NAME","shanelreyes/rory")
DOCKER_IMAGE_TAG  = os.environ.get("DOCKER_IMAGE_TAG","worker")
DOCKER_IMAGE      = os.environ.get("DOCKER_IMAGE","{}:{}".format(DOCKER_IMAGE_NAME,DOCKER_IMAGE_TAG))
DOCKER_NETWORK_ID = os.environ.get("DOCKER_NETWORK_ID","mictlanx") 
init_port         = int(os.environ.get("WORKER_INIT_PORT",3000))
DEBUG             = bool(int(os.environ.get("DEBUG",0)))
RELOAD            = bool(int(os.environ.get("RELOAD",0)))
SERVER_IP_ADDR    = os.environ.get("SERVER_IP_ADDR","0.0.0.0")

#CREAR FOLDERS
SINK_FOLDER   = "/rory/{}/sink".format(NODE_ID)
SOURCE_FOLDER = "/rory/{}/source".format(NODE_ID)
LOG_FOLDER    = "/rory/{}/log".format(NODE_ID)
os.makedirs(SINK_FOLDER,  exist_ok = True)
os.makedirs(SOURCE_FOLDER,exist_ok = True)
os.makedirs(LOG_FOLDER,   exist_ok = True)

LOG_PATH       = os.environ.get("LOG_PATH",LOG_FOLDER)
SINK_PATH      = os.environ.get("SINK_PATH",SINK_FOLDER)
SOURCE_PATH    = os.environ.get("SOURCE_PATH",SOURCE_FOLDER)

TESTING        = bool(int(os.environ.get("TESTING","1")))
MAX_RETRIES    = int(os.environ.get("MAX_RETRIES",100))
LOAD_BALANCING = int(os.environ.get("LOAD_BALANCING","0"))

MICTLANX_SUMMONER_IP_ADDR        = os.environ.get("MICTLANX_SUMMONER_IP_ADDR","localhost")
MICTLANX_SUMMONER_PORT           = os.environ.get("MICTLANX_SUMMONER_PORT",1025)
MICTLANX_API_VERSION             = int(os.environ.get("MICTLANX_API_VERSION",3))
MICTLANX_APP_ID                  = os.environ.get("MICTLANX_APP_ID","APP_ID")
MICTLANX_CLIENT_ID               = os.environ.get("MICTLANX_CLIENT_ID","CLIENT_ID")
MICTLANX_SECRET                  = os.environ.get("MICTLANX_SECRET","SECRET")
MICTLANX_PROXY_IP_ADDR           = os.environ.get("MICTLANX_PROXY_IP_ADDR","localhost")
MICTLANX_PROXY_PORT              = int(os.environ.get("MICTLANX_PROXY_PORT","8080"))
MICTLANX_XOLO_IP_ADDR            = os.environ.get("MICTLANX_XOLO_IP_ADDR","localhost")
MICTLANX_XOLO_PORT               = int(os.environ.get("MICTLANX_XOLO_PORT","10000"))
MICTLANX_REPLICA_MANAGER_IP_ADDR = os.environ.get("MICTLANX_REPLICA_MANAGER_IP_ADDR", "localhost")
MICTLANX_REPLICA_MANAGER_PORT    = int(os.environ.get("MICTLANX_REPLICA_MANAGER_PORT", "20000"))
MICTLANX_EXPIRES_IN              = os.environ.get("MICTLANX_EXPIRES_IN","15d")

REPLICATOR = Summoner(
    ip_addr     = MICTLANX_SUMMONER_IP_ADDR,
    port        = MICTLANX_SUMMONER_PORT,
    api_version = Some(MICTLANX_API_VERSION)
)
xolo = Xolo(
    ip_addr     = MICTLANX_XOLO_IP_ADDR ,
    port        = MICTLANX_XOLO_PORT,
    api_version = Some(MICTLANX_API_VERSION) 
)

# DEPLOY_NODES
deploy_nodes(
    summoner                         = REPLICATOR,
    xolo                             = xolo,
    init_workers                     = init_workers,
    NODE_ID                          = NODE_ID,
    PORT                             = str(PORT),
    NODE_PREFIX                      = NODE_PREFIX,
    init_port                        = init_port,
    DOCKER_IMAGE                     = DOCKER_IMAGE,
    DOCKER_NETWORK_ID                = DOCKER_NETWORK_ID,
    MICTLANX_APP_ID                  = MICTLANX_APP_ID,
    MICTLANX_CLIENT_ID               = MICTLANX_CLIENT_ID,
    MICTLANX_EXPIRES_IN              = MICTLANX_EXPIRES_IN,
    MICTLANX_PROXY_IP_ADDR           = MICTLANX_PROXY_IP_ADDR,
    MICTLANX_PROXY_PORT              = str(MICTLANX_PROXY_PORT),
    MICTLANX_REPLICA_MANAGER_IP_ADDR = MICTLANX_REPLICA_MANAGER_IP_ADDR,
    MICTLANX_REPLICA_MANAGER_PORT    = str(MICTLANX_REPLICA_MANAGER_PORT),
    MICTLANX_XOLO_IP_ADDR            = MICTLANX_XOLO_IP_ADDR,
    MICTLANX_XOLO_PORT               = str(MICTLANX_XOLO_PORT),
    MICTLANX_API_VERSION             = MICTLANX_API_VERSION,
    MICTLANX_SECRET                  = MICTLANX_SECRET,
)

LOGGER = create_logger(
    name                   = NODE_ID,
    LOG_FILENAME           = NODE_ID,
    LOG_PATH               = LOG_PATH,
    console_handler_filter = lambda record: record.levelno == logging.DEBUG or record.levelno == logging.INFO or record.levelno == logging.ERROR,
    file_handler_filter    = lambda record: record.levelno == logging.DEBUG or record.levelno == logging.INFO,
)

"""
Description:
    Function that create a context using Flask. Establishes the connection between client, manager and worker. 
"""
def create_app(*args):
    app = Flask(__name__)
    app.register_blueprint(clustering) # SkMeans routes / DBSkmeans routes
    app.register_blueprint(workers) # node replication
    balancers = [
        RoundRobin(n = init_workers,prefix = NODE_PREFIX),
        TwoChoices(n = init_workers,prefix = NODE_PREFIX),
        Random    (n = init_workers,prefix = NODE_PREFIX)
    ]
    with app.app_context():
        current_app.config["lb"]                               = balancers[LOAD_BALANCING]
        current_app.config["workers"]                          = {}
        current_app.config["replicator"]                       = REPLICATOR
        current_app.config["xolo"]                             = xolo
        current_app.config["NODE_ID"]                          = NODE_ID
        current_app.config["NODE_PORT"]                        = PORT
        current_app.config["NODE_PREFIX"]                      = NODE_PREFIX
        current_app.config["DOCKER_IMAGE_NAME"]                = DOCKER_IMAGE_NAME
        current_app.config["DOCKER_IMAGE_TAG"]                 = DOCKER_IMAGE_TAG
        current_app.config["DOCKER_IMAGE"]                     = "{}:{}".format(DOCKER_IMAGE_NAME,DOCKER_IMAGE_TAG)
        current_app.config["logger"]                           = LOGGER
        current_app.config["DEPLOY_START_TIMES"]               = {}
        #current_app.config["STORAGE_CLIENT"]                   = STORAGE_CLIENT
        current_app.config["MICTLANX_APP_ID"]                  = MICTLANX_APP_ID
        current_app.config["MICTLANX_SECRET"]                  = MICTLANX_SECRET
        current_app.config["MICTLANX_PROXY_IP_ADDR"]           = MICTLANX_PROXY_IP_ADDR
        current_app.config["MICTLANX_PROXY_PORT"]              = MICTLANX_PROXY_PORT
        current_app.config["MICTLANX_XOLO_IP_ADDR"]            = MICTLANX_XOLO_IP_ADDR
        current_app.config["MICTLANX_XOLO_PORT"]               = MICTLANX_XOLO_PORT
        current_app.config["MICTLANX_REPLICA_MANAGER_IP_ADDR"] = MICTLANX_REPLICA_MANAGER_IP_ADDR
        current_app.config["MICTLANX_REPLICA_MANAGER_PORT"]    = MICTLANX_REPLICA_MANAGER_PORT
        current_app.config["MICTLANX_API_VERSION"]             = MICTLANX_API_VERSION
        current_app.config["MICTLANX_EXPIRES_IN"]              = MICTLANX_EXPIRES_IN
        current_app.config["MICTLANX_CLIENT_ID"]               = MICTLANX_CLIENT_ID
        current_app.config["DOCKER_NETWORK_ID"]                = DOCKER_NETWORK_ID

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host = SERVER_IP_ADDR, port = PORT,debug = DEBUG,use_reloader = RELOAD)