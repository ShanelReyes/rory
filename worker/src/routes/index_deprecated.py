import time
from flask import Blueprint,current_app,request,send_from_directory,send_file,Response
from os.path import exists

index = Blueprint("index",__name__,url_prefix="/")


"""
Description:
    Read files from a specific path
"""
@index.route("download",methods = ["GET"])
def dataset():
    logger     = current_app.config["logger"]
    nodeId     = current_app.config["NODE_ID"]
    args       = request.args
    filePath   = args.get("file-path","")      
    fileExists = exists(filePath)
    logger.debug("FILE_EXISTS {} {}".format(filePath,fileExists))
    if(filePath == "" or fileExists):
        logger.error("DOWNLOAD_FILE_FAILED node-id={} file-path={} file-exists={}".format(nodeId,filePath,fileExists))
        return Response(None,204)
    else:
        arrivalTime    = time.time()
        headers        = request.headers
        extension      = headers.get("File-Extension","")
        response       = send_file(filePath)
        serviceTime    = time.time() - arrivalTime
        logger.debug("PULL_FILE,{},{},{}".format(fileExists,serviceTime))
        logger.debug("_"*100)
        return response