import os, logging, sys
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
# print("INIT_MANAGER")

app = Flask(__name__)
DEBUG                 = bool(int(os.environ.get("DEBUG",0)))
if DEBUG:
    load_dotenv(os.environ.get("ENV_FILE_PATH","/rory/envs/.manager.env"))

NODE_ID               = os.environ.get("NODE_ID","rory-manager-0")
IP_ADDR               = os.environ.get("NODE_IP_ADDR",NODE_ID)
PORT                  = int(os.environ.get("NODE_PORT",6000))
NODE_PREFIX           = os.environ.get("NODE_PREFIX","rory-worker-")
init_workers          = int(os.environ.get("INIT_WORKERS","0")) #worker iniciales que se levantan
DOCKER_IMAGE_NAME     = os.environ.get("DOCKER_IMAGE_NAME","shanelreyes/rory")
DOCKER_IMAGE_TAG      = os.environ.get("DOCKER_IMAGE_TAG","worker")
DOCKER_IMAGE          = os.environ.get("DOCKER_IMAGE","{}:{}".format(DOCKER_IMAGE_NAME,DOCKER_IMAGE_TAG))
DOCKER_NETWORK_ID     = os.environ.get("DOCKER_NETWORK_ID","mictlanx") 
init_port             = int(os.environ.get("WORKER_INIT_PORT",3000))
RELOAD                = bool(int(os.environ.get("RELOAD",0)))
SERVER_IP_ADDR        = os.environ.get("SERVER_IP_ADDR","0.0.0.0")
XOLO_ENABLE           = bool(int(os.environ.get("XOLO_ENABLE","0")))
WORKER_MAX_THREADS    = int(os.environ.get("WORKER_MAX_THREADS",2)) #Cantidad de threats para gunicorn
WORKER_MEMORY         = os.environ.get("WORKER_MEMORY","1000000000")
WORKER_CPU            = os.environ.get("WORKER_CPU",2)
WORKER_MICTLANX_PEERS = os.environ.get("WORKER_MICTLANX_PEERS")
WORKER_TIMEOUT        = int(os.environ.get("WORKER_TIMEOUT",300))

#CREAR FOLDERS
SOURCE_PATH = os.environ.get("SOURCE_PATH","/rory/source")
SINK_PATH   = os.environ.get("SINK_PATH","/rory/sink")
LOG_PATH    = os.environ.get("LOG_PATH","/rory/log")
try:
    os.makedirs(SOURCE_PATH,exist_ok = True)
    os.makedirs(SINK_PATH,  exist_ok = True)
    os.makedirs(LOG_PATH,   exist_ok = True)
except Exception as e:
    print("MAKE_FOLDER_ERROR",e)

TESTING                      = bool(int(os.environ.get("TESTING","1")))
MAX_RETRIES                  = int(os.environ.get("MAX_RETRIES",100))
LOAD_BALANCING               = int(os.environ.get("LOAD_BALANCING","0"))
MICTLANX_SUMMONER_IP_ADDR    = os.environ.get("MICTLANX_SUMMONER_IP_ADDR","localhost")
MICTLANX_SUMMONER_PORT       = os.environ.get("MICTLANX_SUMMONER_PORT",15000)
MICTLANX_SUMMONER_MODE       = os.environ.get("MICTLANX_SUMMONER_MODE","docker")
MICTLANX_API_VERSION         = int(os.environ.get("MICTLANX_API_VERSION",3))
MICTLANX_APP_ID              = os.environ.get("MICTLANX_APP_ID","APP_ID")
MICTLANX_CLIENT_ID           = os.environ.get("MICTLANX_CLIENT_ID","CLIENT_ID")
MICTLANX_SECRET              = os.environ.get("MICTLANX_SECRET","SECRET")
MICTLANX_XOLO_IP_ADDR        = os.environ.get("MICTLANX_XOLO_IP_ADDR","localhost")
MICTLANX_XOLO_PORT           = int(os.environ.get("MICTLANX_XOLO_PORT","10000"))
MICTLANX_EXPIRES_IN          = os.environ.get("MICTLANX_EXPIRES_IN","15d")
MICTLANX_PEERS               = os.environ.get("MICTLANX_PEERS", "mictlanx-peer-0:localhost:7000")
MICTLANX_TIMEOUT             = int(os.environ.get("MICTLANX_TIMEOUT",120))
MICTLANX_CLIENT_LB_ALGORITHM = os.environ.get("MICTLANX_CLIENT_LB_ALGORITHM","2CHOICES_UF")
MICTLANX_MAX_WORKERS         = int(os.environ.get("MICTLANX_MAX_WORKERS",12))
MICTLANX_DEBUG               = bool(int(os.environ.get("MICTLANX_DEBUG",0)))
MICTLANX_DAEMON              = bool(int(os.environ.get("MICTLANX_DAEMON",0)))
MICTLANX_SHOW_METRICS        = bool(int(os.environ.get("MICTLANX_SHOW_METRICS",0)))
MICTLANX_DISABLED_LOG        = bool(int(os.environ.get("MICTLANX_DISABLED_LOG",0)))


REPLICATOR = Summoner(
    ip_addr     = MICTLANX_SUMMONER_IP_ADDR,
    port        = MICTLANX_SUMMONER_PORT,
    api_version = Some(MICTLANX_API_VERSION)
)
xolo = Xolo(
    ip_addr     = Some(MICTLANX_XOLO_IP_ADDR) ,
    port        = Some(MICTLANX_XOLO_PORT),
    api_version = Some(MICTLANX_API_VERSION) 
)
# DEPLOY_NODES
deploy_nodes(
    summoner               = REPLICATOR,
    NODE_ID                = NODE_ID,
    PORT                   = str(PORT),
    WORKER_MAX_THREADS     = WORKER_MAX_THREADS,
    DOCKER_IMAGE           = DOCKER_IMAGE,
    DOCKER_NETWORK_ID      = DOCKER_NETWORK_ID,
    MICTLANX_APP_ID        = MICTLANX_APP_ID,
    MICTLANX_CLIENT_ID     = MICTLANX_CLIENT_ID,
    MICTLANX_SECRET        = MICTLANX_SECRET,
    xolo                   = xolo,
    MICTLANX_SUMMONER_MODE = MICTLANX_SUMMONER_MODE,
    init_workers           = init_workers,
    MICTLANX_EXPIRES_IN    = MICTLANX_EXPIRES_IN,
    NODE_PREFIX            = NODE_PREFIX,
    init_port              = init_port,
    XOLO_ENABLE            = XOLO_ENABLE,
    WORKER_MEMORY          = WORKER_MEMORY,
    WORKER_CPU             = WORKER_CPU,
    WORKER_MICTLANX_PEERS  = WORKER_MICTLANX_PEERS,
    MICTLANX_CLIENT_LB_ALGORITHM = MICTLANX_CLIENT_LB_ALGORITHM,
    MICTLANX_DEBUG         = MICTLANX_DEBUG,
    MICTLANX_DAEMON        = MICTLANX_DAEMON,
    MICTLANX_SHOW_METRICS  = MICTLANX_SHOW_METRICS,
    MICTLANX_MAX_WORKERS   = MICTLANX_MAX_WORKERS,
    MICTLANX_DISABLED_LOG  = MICTLANX_DISABLED_LOG
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
   
    app.register_blueprint(clustering) # SkMeans routes / DBSkmeans routes
    app.register_blueprint(workers) # node replication
    balancers = [
        RoundRobin(n = init_workers,prefix = NODE_PREFIX),
        TwoChoices(n = init_workers,prefix = NODE_PREFIX),
        Random    (n = init_workers,prefix = NODE_PREFIX)
    ]
    with app.app_context():
        current_app.config["lb"]                 = balancers[LOAD_BALANCING]
        current_app.config["workers"]            = {}
        current_app.config["replicator"]         = REPLICATOR
        current_app.config["xolo"]               = xolo
        current_app.config["NODE_ID"]            = NODE_ID
        current_app.config["NODE_PORT"]          = PORT
        current_app.config["NODE_PREFIX"]        = NODE_PREFIX
        current_app.config["DOCKER_IMAGE_NAME"]  = DOCKER_IMAGE_NAME
        current_app.config["DOCKER_IMAGE_TAG"]   = DOCKER_IMAGE_TAG
        current_app.config["DOCKER_IMAGE"]       = "{}:{}".format(DOCKER_IMAGE_NAME,DOCKER_IMAGE_TAG)
        current_app.config["logger"]             = LOGGER
        current_app.config["DEPLOY_START_TIMES"] = {}
        current_app.config["DOCKER_NETWORK_ID"]  = DOCKER_NETWORK_ID
        current_app.config["MICTLANX_TIMEOUT"]   = MICTLANX_TIMEOUT
        current_app.config["WORKER_TIMEOUT"]     = WORKER_TIMEOUT

if __name__ == 'main' or __name__ == "__main__":
    try:
        #print("ANTES DE INICIAR LA APP")
        create_app()
    except Exception as e:
        print(e)
        sys.exit(1)