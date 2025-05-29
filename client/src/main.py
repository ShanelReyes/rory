import os, logging, sys
from flask import Flask,current_app
from option import Some
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor
from mictlanx.utils.index import Utils
from rory.core.security.cryptosystem.liu import Liu
from rory.core.security.dataowner import DataOwner
from rory.core.security.pqc.dataowner import DataOwner as DataOwnerPQC
from rory.core.interfaces.rorymanager import RoryManager
from routes.clustering import clustering
from routes.classification import classification
from mictlanx.logger.log import Log
from mictlanx import Client,AsyncClient

app = Flask(__name__)


ENV_FILE_PATH = os.environ.get("ENV_FILE_PATH","/rory/envs/.client.env")
STR_DEBUG     = os.environ.get("RORY_DEBUG",0) 
DEBUG         = bool(int(STR_DEBUG))

if DEBUG:
    load_dotenv(ENV_FILE_PATH)

NODE_ID              = os.environ.get("NODE_ID","rory-client-0")
NODE_IP_ADDR         = os.environ.get("NODE_IP_ADDR",NODE_ID)
NODE_PORT            = int(os.environ.get("NODE_PORT",3000))
SERVER_IP_ADDR       = os.environ.get("SERVER_IP_ADDR","0.0.0.0")
RORY_MANAGER_IP_ADDR = os.environ.get("RORY_MANAGER_IP_ADDR","localhost")
RORY_MANAGER_PORT    = int(os.environ.get("RORY_MANAGER_PORT",6000))
MAX_WORKERS          = int(os.environ.get("MAX_WORKERS",2)) #Total of process for encryption
WORKER_TIMEOUT       = int(os.environ.get("WORKER_TIMEOUT",300))
MAX_ITERATIONS       = int(os.environ.get("MAX_ITERATIONS",10))

LIU_SECURITY_LEVEL = int(os.environ.get("LIU_SECURITY_LEVEL","128")) #128, 192, 256
LIU_SECURE_RANDOM  = bool(int(os.environ.get("LIU_SECURE_RANDOM","0")))
LIU_SEED           = int(os.environ.get("LIU_SEED","123"))
LIU_USE_NP_RANDOM  = bool(int(os.environ.get("LIU_USE_NP_RANDOM","1")))
LIU_ROUND          = bool(int(os.environ.get("LIU_ROUND","0")))
LIU_DECIMALS       = int(os.environ.get("LIU_DECIMALS",6))

CKKS_ROUND          = bool(int(os.environ.get("CKKS_ROUND",0)))
CKKS_DECIMALS       = int(os.environ.get("CKKS_DECIMALS",2))
CTX_FILENAME        = os.environ.get("CTX_FILENAME","ctx")
PUBKEY_FILENAME     = os.environ.get("PUBKEY_FILENAME","pubkey")
SECRET_KEY_FILENAME = os.environ.get("SECRET_KEY_FILENAME","secretkey")
RELINKEY_FILENAME   = os.environ.get("RELINKEY_FILENAME","relinkey")

RELOAD        = bool(int(os.environ.get("RELOAD",0)))
NP_RANDOM     = bool(int(os.environ.get("NP_RANDOM","1")))
TESTING_ENV   = os.environ.get("TESTING","1")

LOGGER_NAME   = os.environ.get("LOGGER_NAME","rory-client-0")
SOURCE_PATH   = os.environ.get("SOURCE_PATH","/rory/source")
SINK_PATH     = os.environ.get("SINK_PATH","/rory/sink")
LOG_PATH      = os.environ.get("LOG_PATH","/rory/log")
KEYS_PATH     = os.environ.get("KEYS_PATH","/rory/keys")
TESTING       = bool(int(TESTING_ENV))

try:
    os.makedirs(SOURCE_PATH,exist_ok = True)
    os.makedirs(SINK_PATH,  exist_ok = True)
    os.makedirs(LOG_PATH,   exist_ok = True)
except Exception as e:
    print("MAKE_FOLDER_ERROR",e)

MICTLANX_CLIENT_ID = os.environ.get("MICTLANX_CLIENT_ID","{}_mictlanx".format(NODE_ID))
MICTLANX_TIMEOUT   = int(os.environ.get("MICTLANX_TIMEOUT",3600))
MICTLANX_ROUTERS   = os.environ.get("MICTLANX_ROUTERS", "mictlanx-router-0:localhost:60666") #mictlanx-peer-2:localhost:7002")
MICTLANX_MAX_WORKERS    = int(os.environ.get("MICTLANX_MAX_WORKERS","12"))
MICTLANX_BUCKET_ID      = os.environ.get("MICTLANX_BUCKET_ID","rory")
MICTLANX_LOG_PATH       = os.environ.get("MICTLANX_LOG_PATH","/rory/mictlanx")
MICTLANX_LOG_INTERVAL   = int(os.environ.get("MICTLANX_LOG_INTERVAL","24"))
MICTLANX_LOG_WHEN       = os.environ.get("MICTLANX_LOG_WHEN","h")
MICTLANX_DELAY          = int(os.environ.get("MICTLANX_DELAY","2"))
MICTLANX_BACKOFF_FACTOR = float(os.environ.get("MICTLANX_BACKOFF_FACTOR","0.5"))
MICTLANX_MAX_RETRIES    = int(os.environ.get("MICTLANX_MAX_RETRIES","10"))
MICTLANX_PROTOCOL       = os.environ.get("MICTLANX_PROTOCOL","https")

ASYNC_STORAGE_CLIENT = AsyncClient(
    client_id        = MICTLANX_CLIENT_ID,
    capacity_storage = "200mb",
    debug            = False,
    eviction_policy  = "LRU",
    max_workers      = MICTLANX_MAX_WORKERS,
    routers          = list(Utils.routers_from_str(routers_str=MICTLANX_ROUTERS,protocol=MICTLANX_PROTOCOL)),
    verify           = False,
    log_output_path  = MICTLANX_LOG_PATH,
    log_interval     = MICTLANX_LOG_INTERVAL,
    log_when         = MICTLANX_LOG_WHEN
)

MANAGER = RoryManager(
    hostname = RORY_MANAGER_IP_ADDR,
    port     = RORY_MANAGER_PORT,
)

def console_handler_filter(record:logging.LogRecord):
    if DEBUG:
        return True
    elif not DEBUG and (record.levelno == logging.INFO or record.levelno == logging.ERROR):
        return True
    else:
        return False

LOGGER = Log(
    name                   = LOGGER_NAME,
    path                   = LOG_PATH,
    console_handler_filter = console_handler_filter,
    interval               = 24,
    when                   = "h"
)

LIU  = Liu(
    _round         = LIU_ROUND,
    decimals       = LIU_DECIMALS,
    secure_random  = LIU_SECURE_RANDOM,
    security_level = LIU_SECURITY_LEVEL,
    seed           = LIU_SEED,
    use_np_random  = LIU_USE_NP_RANDOM
)
DATAOWNER = DataOwner(
    # m             = M,
    securitylevel = LIU_SECURITY_LEVEL,
    liu_scheme    = LIU
)


cores       = os.cpu_count()
max_workers = cores if MAX_WORKERS > cores else MAX_WORKERS
# print("MAX_WORKERS", max_workers)
executor    = ProcessPoolExecutor(max_workers=max_workers)

"""
Description:
    Function that create a context using Flask. Establishes the connection between client, manager and worker. 
"""
def create_app(*args):
    
    # Register blueprints
    app.register_blueprint(clustering)
    app.register_blueprint(classification)
    with app.app_context():
        current_app.config["request_counter"]         = 0
        current_app.config["logger"]                  = LOGGER
        current_app.config["manager"]                 = MANAGER
        current_app.config["liu"]                     = LIU
        current_app.config["dataowner"]               = DATAOWNER
        current_app.config["SOURCE_PATH"]             = SOURCE_PATH
        current_app.config["SINK_PATH"]               = SINK_PATH
        current_app.config["NODE_ID"]                 = NODE_ID
        current_app.config["LOG_PATH"]                = LOG_PATH
        current_app.config["MAX_ITERATIONS"]          = MAX_ITERATIONS
        current_app.config["ASYNC_STORAGE_CLIENT"]    = ASYNC_STORAGE_CLIENT
        current_app.config["TESTING"]                 = TESTING
        current_app.config["MAX_WORKES"]              = max_workers
        current_app.config["executor"]                = executor
        current_app.config["WORKER_TIMEOUT"]          = WORKER_TIMEOUT
        current_app.config["np_random"]               = NP_RANDOM
        current_app.config["LIU_SECURITY_LEVEL"]      = LIU_SECURITY_LEVEL
        current_app.config["BUCKET_ID"]               = MICTLANX_BUCKET_ID
        current_app.config["MICTLANX_TIMEOUT"]        = MICTLANX_TIMEOUT
        current_app.config["MICTLANX_DELAY"]          = MICTLANX_DELAY
        current_app.config["MICTLANX_BACKOFF_FACTOR"] = MICTLANX_BACKOFF_FACTOR
        current_app.config["MICTLANX_MAX_RETRIES"]    = MICTLANX_MAX_RETRIES
        current_app.config["_round"]                  = CKKS_ROUND
        current_app.config["DECIMALS"]                = CKKS_DECIMALS
        current_app.config["KEYS_PATH"]               = KEYS_PATH
        current_app.config["CTX_FILENAME"]            = CTX_FILENAME
        current_app.config["PUBKEY_FILENAME"]         = PUBKEY_FILENAME
        current_app.config["SECRET_KEY_FILENAME"]     = SECRET_KEY_FILENAME
        current_app.config["RELINKEY_FILENAME"]       = RELINKEY_FILENAME
    # return app


"""
Description:
    Initialize create_app
"""
if __name__ == 'main' or __name__ == "__main__":
    try:
        LOGGER.debug({
            "event":"CLIENT_STARTED",
            "soruce_path":SOURCE_PATH,
            "sink_path":SINK_PATH,
            "node_id":NODE_ID,
            "log_path":LOG_PATH,
            "debug":DEBUG,
            "max_iterations":MAX_ITERATIONS,
            "testing":TESTING,
            # "num_chunks":NUM_CHUNKS,
            "mictlanx_timeout":MICTLANX_TIMEOUT,
            "worker_timeout":WORKER_TIMEOUT 
        })
        create_app()
    except Exception as e:
        LOGGER.error({
            "msg":str(e)
        })
        executor.shutdown()
        # STORAGE_CLIENT.shutdown()
        sys.exit(1)