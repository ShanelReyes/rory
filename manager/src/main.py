import os, sys
from flask import Flask,current_app
from routes.clustering import clustering
from routes.workers import workers
from mictlanx.v4.summoner.summoner import Summoner
from load_balancing.round_robin import RoundRobin
from load_balancing.two_choices import TwoChoices
from load_balancing.random import  Random
from deployworkers import deploy_nodes
from mictlanx.logger.log import Log
from option import Some
from dotenv import load_dotenv
import logging
import time

app = Flask(__name__)
DEBUG              = bool(int(os.environ.get("RORY_DEBUG",0)))
if DEBUG:
    load_dotenv(os.environ.get("ENV_FILE_PATH","/rory/envs/.manager.env"))

NODE_ID            = os.environ.get("NODE_ID","rory-manager-0")
IP_ADDR            = os.environ.get("NODE_IP_ADDR",NODE_ID)
PORT               = int(os.environ.get("NODE_PORT",6000))
SERVER_IP_ADDR     = os.environ.get("SERVER_IP_ADDR","0.0.0.0")
NODE_PREFIX        = os.environ.get("NODE_PREFIX","rory-worker-")
FOLDER_KEYS        = os.environ.get("FOLDER_KEYS","keys128")
init_workers       = int(os.environ.get("INIT_WORKERS","0")) #worker iniciales que se levantan
init_port          = int(os.environ.get("WORKER_INIT_PORT",9000))
DOCKER_IMAGE_NAME  = os.environ.get("DOCKER_IMAGE_NAME","shanelreyes/rory")
DOCKER_IMAGE_TAG   = os.environ.get("DOCKER_IMAGE_TAG","worker")
DOCKER_IMAGE       = os.environ.get("DOCKER_IMAGE","{}:{}".format(DOCKER_IMAGE_NAME,DOCKER_IMAGE_TAG))
DOCKER_NETWORK_ID  = os.environ.get("DOCKER_NETWORK_ID","mictlanx")
MAX_RETRIES        = int(os.environ.get("MAX_RETRIES",100))
LOAD_BALANCING     = int(os.environ.get("LOAD_BALANCING","0"))
WORKER_MAX_THREADS = int(os.environ.get("WORKER_MAX_THREADS",2)) #Cantidad de threats para gunicorn
WORKER_MEMORY      = os.environ.get("WORKER_MEMORY","1000000000")
WORKER_CPU         = os.environ.get("WORKER_CPU",2)
WORKER_TIMEOUT     = int(os.environ.get("WORKER_TIMEOUT",300))
SWARM_NODES        = os.environ.get("SWARM_NODES","2,3,4,8").split(",")
LIU_ROUND          = int(os.environ.get("LIU_ROUND","2"))
# NUM_CHUNKS         = int(os.environ.get("NUM_CHUNKS",4)) #Chunks for mixtlanx

DISTANCE            = os.environ.get("DISTANCE","MANHATHAN")
MIN_ERROR           = float(os.environ.get("MIN_ERROR",0.015))
CKKS_ROUND          = int(os.environ.get("CKKS_ROUND",0))
CKKS_DECIMALS       = int(os.environ.get("CKKS_DECIMALS",2))
CTX_FILENAME        = os.environ.get("CTX_FILENAME","ctx")
PUBKEY_FILENAME     = os.environ.get("PUBKEY_FILENAME","pubkey")
SECRET_KEY_FILENAME = os.environ.get("SECRET_KEY_FILENAME","secretkey")
RELINKEY_FILENAME   = os.environ.get("RELINKEY_FILENAME","relinkey")

SOURCE_PATH = os.environ.get("SOURCE_PATH","/rory/source")
SINK_PATH   = os.environ.get("SINK_PATH","/rory/sink")
LOG_PATH    = os.environ.get("LOG_PATH","/rory/log")
try:
    os.makedirs(SOURCE_PATH,exist_ok = True)
    os.makedirs(SINK_PATH,  exist_ok = True)
    os.makedirs(LOG_PATH,   exist_ok = True)
except Exception as e:
    print("MAKE_FOLDER_ERROR",e)

MICTLANX_SUMMONER_IP_ADDR    = os.environ.get("MICTLANX_SUMMONER_IP_ADDR","localhost")
MICTLANX_SUMMONER_PORT       = int(os.environ.get("MICTLANX_SUMMONER_PORT",15000))
MICTLANX_SUMMONER_MODE       = os.environ.get("MICTLANX_SUMMONER_MODE","docker")
MICTLANX_API_VERSION         = int(os.environ.get("MICTLANX_API_VERSION",3))
MICTLANX_CLIENT_ID           = os.environ.get("MICTLANX_CLIENT_ID","CLIENT_ID")
MICTLANX_ROUTERS             = os.environ.get("MICTLANX_ROUTERS", "mictlanx-router-0:localhost:60666")
MICTLANX_TIMEOUT             = int(os.environ.get("MICTLANX_TIMEOUT",120))
MICTLANX_MAX_WORKERS         = int(os.environ.get("MICTLANX_MAX_WORKERS",12))
MICTLANX_DEBUG               = bool(int(os.environ.get("MICTLANX_DEBUG",0)))
MICTLANX_LOG_PATH            = os.environ.get("MICTLANX_LOG_PATH","/rory/mictlanx")
MICTLANX_LOG_INTERVAL        = os.environ.get("MICTLANX_LOG_INTERVAL","24")
MICTLANX_LOG_WHEN            = os.environ.get("MICTLANX_LOG_WHEN","h")
MICTLANX_PROTOCOL            = os.environ.get("MICTLANX_PROTOCOL","https")
MICTLANX_BUCKET_ID           = os.environ.get("MICTLANX_BUCKET_ID","rory")
MICTLANX_DELAY               = int(os.environ.get("MICTLANX_DELAY","2"))
MICTLANX_BACKOFF_FACTOR      = float(os.environ.get("MICTLANX_BACKOFF_FACTOR","0.5"))
MICTLANX_MAX_RETRIES         = int(os.environ.get("MICTLANX_MAX_RETRIES","10"))
MICTLANX_CHUNK_SIZE          = os.environ.get("MICTLANX_CHUNK_SIZE","256kb")
MICTLANX_MAX_PARALELL_GETS   = int(os.environ.get("MICTLANX_MAX_PARALELL_GETS","2"))

REPLICATOR = Summoner(
    ip_addr     = MICTLANX_SUMMONER_IP_ADDR,
    port        = MICTLANX_SUMMONER_PORT,
    api_version = Some(MICTLANX_API_VERSION)
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
if init_workers > 0:
    deploy_workers_start_time = time.time()
    LOGGER.debug({
        "event":"DEPLOY_NODES",
        "node_id":NODE_ID,
        "port":PORT,
        "init_workers":init_workers,
        "worker_memory":WORKER_MEMORY,
        "worker_cpu":WORKER_CPU,
        "init_port":init_port,
        "docker_image":DOCKER_IMAGE,
        "routers":MICTLANX_ROUTERS,
        "swarm_nodes":",".join(SWARM_NODES)
    })
    
    deploy_nodes_result = deploy_nodes(
        log                        = LOGGER,
        summoner                   = REPLICATOR,
        NODE_ID                    = NODE_ID,
        PORT                       = str(PORT),
        WORKER_MAX_THREADS         = WORKER_MAX_THREADS,
        DOCKER_IMAGE               = DOCKER_IMAGE,
        DOCKER_NETWORK_ID          = DOCKER_NETWORK_ID,
        MICTLANX_CLIENT_ID         = MICTLANX_CLIENT_ID,
        MICTLANX_SUMMONER_MODE     = MICTLANX_SUMMONER_MODE,
        init_workers               = init_workers,
        NODE_PREFIX                = NODE_PREFIX,
        FOLDER_KEYS                = FOLDER_KEYS,
        init_port                  = init_port,
        WORKER_MEMORY              = WORKER_MEMORY,
        WORKER_CPU                 = WORKER_CPU,
        WORKER_MICTLANX_ROUTERS    = MICTLANX_ROUTERS,
        MICTLANX_DEBUG             = MICTLANX_DEBUG,
        MICTLANX_MAX_WORKERS       = MICTLANX_MAX_WORKERS,
        swarm_nodes                = SWARM_NODES,
        SERVER_IP_ADDR             = SERVER_IP_ADDR,
        MAX_RETRIES                = MAX_RETRIES,
        # NUM_CHUNKS                 = NUM_CHUNKS,
        DISTANCE                   = DISTANCE,
        MIN_ERROR                  = MIN_ERROR,
        CKKS_ROUND                 = CKKS_ROUND,
        CKKS_DECIMALS              = CKKS_DECIMALS,
        CTX_FILENAME               = CTX_FILENAME,
        PUBKEY_FILENAME            = PUBKEY_FILENAME,
        SECRET_KEY_FILENAME        = SECRET_KEY_FILENAME,
        RELINKEY_FILENAME          = RELINKEY_FILENAME,
        MICTLANX_TIMEOUT           = MICTLANX_TIMEOUT,
        MICTLANX_API_VERSION       = MICTLANX_API_VERSION,
        MICTLANX_PROTOCOL          = MICTLANX_PROTOCOL,
        MICTLANX_LOG_PATH          = MICTLANX_LOG_PATH,
        MICTLANX_LOG_INTERVAL      = MICTLANX_LOG_INTERVAL,
        MICTLANX_LOG_WHEN          = MICTLANX_LOG_WHEN,
        MICTLANX_BUCKET_ID         = MICTLANX_BUCKET_ID,
        MICTLANX_DELAY             = MICTLANX_DELAY,
        MICTLANX_BACKOFF_FACTOR    = MICTLANX_BACKOFF_FACTOR,
        MICTLANX_MAX_RETRIES       = MICTLANX_MAX_RETRIES,
        MICTLANX_CHUNK_SIZE        = MICTLANX_CHUNK_SIZE,
        MICTLANX_MAX_PARALELL_GETS = MICTLANX_MAX_PARALELL_GETS,
        LIU_ROUND                  = LIU_ROUND
    )
    if deploy_nodes_result.is_err:
        LOGGER.error({
            "msg":str(deploy_nodes_result.unwrap_err())
        })
        sys.exit(1)
    else:
        LOGGER.info({
            "event":"DEPLOY_NODES",
            "node_id":NODE_ID,
            "port":PORT,
            "init_workers":init_workers,
            "worker_memory":WORKER_MEMORY,
            "worker_cpu":WORKER_CPU,
            "folder_keys":FOLDER_KEYS,
            "init_port":init_port,
            "docker_image":DOCKER_IMAGE,
            "peers":MICTLANX_ROUTERS,
            "swarm_nodes":",".join(SWARM_NODES),
            "service_time":time.time() - deploy_workers_start_time
        })
        

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
        current_app.config["NODE_ID"]            = NODE_ID
        current_app.config["NODE_PORT"]          = PORT
        current_app.config["NODE_PREFIX"]        = NODE_PREFIX
        current_app.config["FOLDER_KEYS"]        = FOLDER_KEYS
        current_app.config["DOCKER_IMAGE_NAME"]  = DOCKER_IMAGE_NAME
        current_app.config["DOCKER_IMAGE_TAG"]   = DOCKER_IMAGE_TAG
        current_app.config["DOCKER_IMAGE"]       = "{}:{}".format(DOCKER_IMAGE_NAME,DOCKER_IMAGE_TAG)
        current_app.config["logger"]             = LOGGER
        current_app.config["DEPLOY_START_TIMES"] = {}
        current_app.config["DOCKER_NETWORK_ID"]  = DOCKER_NETWORK_ID
        current_app.config["MICTLANX_TIMEOUT"]   = MICTLANX_TIMEOUT
        current_app.config["WORKER_TIMEOUT"]     = WORKER_TIMEOUT
        current_app.config["INIT_WORKER_PORT"]   = init_port

if __name__ == 'main' or __name__ == "__main__":
    try:
        LOGGER.debug({
            "event":"MANAGER_STARTED",
            "load_balancing_algorithm":LOAD_BALANCING,
            "node_id":NODE_ID,
            "port":PORT,
            "node_prefix":NODE_PREFIX,
            "debug":DEBUG,
            "docker_image_name":DOCKER_IMAGE_NAME,
            "docker_image_tag":DOCKER_IMAGE_TAG,
            "docker_image":DOCKER_IMAGE,
            "docker_network_id":DOCKER_NETWORK_ID,
            "worker_timeout":WORKER_TIMEOUT,
            "worker_memory":WORKER_MEMORY,
            "worker_cpu":WORKER_CPU,
            "worker_max_threads":WORKER_MAX_THREADS,
            "mictlanx_max_workers":MICTLANX_MAX_WORKERS,
            "mictlanx_timeout":MICTLANX_TIMEOUT,
        })
        create_app()
    except Exception as e:
        print(e)
        sys.exit(1)