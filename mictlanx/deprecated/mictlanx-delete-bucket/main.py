import os
import sys
import time as T
from mictlanx.v4.client import Client
from concurrent.futures import as_completed
from option import Result,Some,NONE
from mictlanx.utils.index import Utils
from typing import Tuple
import dotenv 
from concurrent.futures import ThreadPoolExecutor,as_completed
dotenv.load_dotenv()

    

def delete_object(client:Client,bucket_id:str,key:str) -> Tuple[str,float,bool]:
    start_time = T.time()
    del_result = client.delete(key=key,bucket_id=bucket_id)
    response_time = T.time() - start_time
    if del_result.is_ok:
        return (key, response_time,True)
        # print("delete {} SUCCESSFULLY - {}".format(key, response_time))
    else:
        return (key, response_time,False)

        # print("DELETE {} FAILED".format(key))

def example_run():
    args = sys.argv[1:]
    if(len(args) >= 2  or len(args)==0):
        raise Exception("Please try to pass a valid file path: python examples/v4/14_get.py <BUCKET_ID>")
    routers        =  Utils.routers_from_str(routers_str=os.environ.get("MICTLANX_ROUTERS","mictlanx-router-0:localhost:60666"))
    bucket_id = Utils.get_or_default(iterator=args, i = 0, default="mictlanx").unwrap()
    client            = Client(
        client_id   = "client-example-0",
        routers       = list(routers),
        debug        = True,
        max_workers  = 1,
        log_when     = "m",
        log_interval = 20,
        bucket_id    = bucket_id,
        log_output_path= os.environ.get("MICTLANX_CLIENT_LOG_PATH","/mictlanx/client")
    )
    MAX_WORKERS = int(os.environ.get("MICTLANX_MAX_WORKERS","6"))
    futures = []
    with ThreadPoolExecutor(max_workers= MAX_WORKERS) as tp:
        for metadata in client.get_all_bucket_metadata(bucket_id=bucket_id):
            for ball in metadata.balls:
                fut = tp.submit(delete_object, key = ball.key, bucket_id = bucket_id, client = client)
                futures.append(fut)
        for fut in as_completed(futures):
            key, response_time, completed = fut.result()
            if completed:
                print("delete {} in {} seconds [SUCCESSFULLY]".format(key,response_time))
            else:
                print("[FAILED] delete {} in {} seconds".format(key,response_time))

              


if __name__ == "__main__":
    example_run()
