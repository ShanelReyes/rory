# import sys
# sys.path.append("/home/sreyes/rory/client/src")
# from utils.utils import Utils
import os
from mictlanx.v4.client import Client
from mictlanx.utils.index import Utils
# res = Utils.get_with_retry()/
MICTLANX_CLIENT_ID           = os.environ.get("MICTLANX_CLIENT_ID","test")
MICTLANX_TIMEOUT             = int(os.environ.get("MICTLANX_TIMEOUT",3600))
MICTLANX_API_VERSION         = int(os.environ.get("MICTLANX_API_VERSION","3"))
MICTLANX_ROUTERS             = os.environ.get("MICTLANX_ROUTERS", "mictlanx-router-0:localhost:60666") #mictlanx-peer-2:localhost:7002")
MICTLANX_DEBUG               = bool(int(os.environ.get("MICTLANX_DEBUG",0)))
MICTLANX_DAEMON              = bool(int(os.environ.get("MICTLANX_DAEMON",1)))
MICTLANX_SHOW_METRICS        = bool(int(os.environ.get("MICTLANX_SHOW_METRICS",0)))
MICTLANX_DISABLED_LOG        = bool(int(os.environ.get("MICTLANX_DISABLED_LOG",0)))
MICTLANX_MAX_WORKERS         = int(os.environ.get("MICTLANX_MAX_WORKERS","12"))
MICTLANX_CLIENT_LB_ALGORITHM = os.environ.get("MICTLANX_CLIENT_LB_ALGORITHM","2CHOICES_UF")
MICTLANX_BUCKET_ID           = os.environ.get("MICTLANX_BUCKET_ID","rory") 
MICTLANX_OUTPUT_PATH         = os.environ.get("MICTLANX_OUTPUT_PATH","/rory/mictlanx")

STORAGE_CLIENT = Client(
    client_id       = MICTLANX_CLIENT_ID,
    bucket_id       = MICTLANX_BUCKET_ID,
    routers         = list(Utils.routers_from_str(MICTLANX_ROUTERS)),
    max_workers     = MICTLANX_MAX_WORKERS,
    lb_algorithm    = MICTLANX_CLIENT_LB_ALGORITHM,
    debug           = MICTLANX_DISABLED_LOG,
    log_output_path = MICTLANX_OUTPUT_PATH, 
 
)

res = STORAGE_CLIENT.get(
    bucket_id="rory",
    key="encryptedsknnmodel1ca",
    headers={"Accept-Encoding":"identity"}
)
print(res)