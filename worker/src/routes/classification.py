import time, json
import numpy as np
#from typing import Awaitable
from flask import Blueprint,current_app,request,Response
#from rory.core.utils.Utils import Utils
#from rory.core.utils.SegmentationUtils import Segmentation
from rory.core.utils.constants import Constants
from rory.core.classification.secure.sknn import SecureKNearestNeighbors as SKNN
#from rory.core.classification.knn import KNearestNeighbors as KNN
from sklearn.neighbors import KNeighborsClassifier
from mictlanx.v4.client import Client as V4Client
#from option import Result
import pickle
# f
from mictlanx.v4.interfaces.responses import GetNDArrayResponse,GetBytesResponse
#from mictlanx.v3.interfaces.payloads import PutNDArrayPayload
from rory.core.interfaces.logger_metrics import LoggerMetrics
from option import Option,NONE,Some,Result
# 
classification = Blueprint("classification",__name__,url_prefix = "/classification")


@classification.route("/test",methods=["GET","POST"])
def test():
    return Response(
        response = json.dumps({
            "component_type":"worker"
        }),
        status   = 200,
        headers  = {
            "Component-Type":"worker"
        }
    )


@classification.route("/sknn/train",methods = ["POST"])
def sknn_train():
    pass

@classification.route("/sknn/predict",methods = ["POST"])
def sknn_pedict():
    arrivalTime               = time.time() #Worker start time
    headers                   = request.headers
    headers                   = request.headers
    to_remove_headers         = ["User-Agent","Accept-Encoding","Connection"]
    filteredHeaders           = dict(list(filter(lambda x: not x[0] in to_remove_headers, headers.items())))
    logger                    = current_app.config["logger"]
    workerId                  = current_app.config["NODE_ID"] # Get the node_id from the global configuration
    STORAGE_CLIENT:V4Client   = current_app.config["STORAGE_CLIENT"]
    m                         = int(filteredHeaders.get("M",3))
    model_id                 = filteredHeaders.get("Model-Id","model-0") #iris
    #model_id                  = "{}_model".format(matrix_id) #iris_model
    encrypted_model_id        = "encrypted-{}".format(model_id) #encrypted-iris_model
    model_labels_id           = "{}_labels".format(model_id) #iris_model_labels
    records_test_id           = filteredHeaders.get("Records-Test-Id","matrix-0")
    encrypted_records_test_id = "encrypted-{}".format(records_test_id) # The id of the encrypted matrix is built
    algorithm                 = Constants.ClassificationAlgorithms.SKNN_PREDICT
    responseHeaders           = {}

    try:
        logger.debug("Worker starts SKNN_PREDICT process -> {}".format(model_id))
        responseHeaders["Start-Time"] = str(arrivalTime)

        encryptedModel_response = STORAGE_CLIENT.get_and_merge_ndarray(key = encrypted_model_id )
        x                       = encryptedModel_response.result()
        encryptedModel_response:GetNDArrayResponse = x.unwrap()
        encryptedModel                             = encryptedModel_response.value
        responseHeaders["Encrypted-Model-Dtype"]   = encryptedModel_response.metadata.tags.get("dtype",encryptedModel.dtype) #["tags"]["dtype"] #Save the data type
        responseHeaders["Encrypted-Model-Shape"]   = encryptedModel_response.metadata.tags.get("shape",encryptedModel.shape) #Save the shape
        
        model_labels_result = STORAGE_CLIENT.get_ndarray(key = model_labels_id).result()
        model_labels        = model_labels_result.unwrap().value

        encryptedRecords_response = STORAGE_CLIENT.get_and_merge_ndarray(key = encrypted_records_test_id)
        y                         = encryptedRecords_response.result()
        encryptedRecords_response:GetNDArrayResponse = y.unwrap()

        encryptedRecords                         = encryptedRecords_response.value
        responseHeaders["Encrypted-Model-Dtype"] = encryptedRecords_response.metadata.tags.get("dtype",encryptedRecords.dtype) #["tags"]["dtype"] #Save the data type
        responseHeaders["Encrypted-Model-Shape"] = encryptedRecords_response.metadata.tags.get("shape",encryptedRecords.shape) #Save the shape
        
        label_vector = SKNN.predict(
            dataset      = encryptedRecords,
            model        = encryptedModel,
            model_labels = model_labels
        )
        endTime                         = time.time()
        serviceTime                     = endTime - arrivalTime
        responseHeaders["Service-Time"] = str(serviceTime)

        logger_metrics = LoggerMetrics(
            operation_type = algorithm, 
            matrix_id      = records_test_id, 
            worker_id      = workerId,
            algorithm      = algorithm, 
            arrival_time   = arrivalTime, 
            end_time       = endTime, 
            service_time   = serviceTime)
        logger.info(str(logger_metrics))

        return Response( #Returns the final response as a label vector + the headers
            response = json.dumps({"labelVector":label_vector.tolist()}),
            status   = 200,
            headers  = {**responseHeaders}
        )
    except Exception as e:
        print(e)
        return Response(
            response = None,
            status   = 503,
            headers  = {"Error-Message":str(e)}
        )


@classification.route("/knn/train",methods = ["POST"])
def knn_train():
    arrivalTime             = time.time() #Worker start time
    headers                 = request.headers
    headers                 = request.headers
    to_remove_headers       = ["User-Agent","Accept-Encoding","Connection"]
    filteredHeaders         = dict(list(filter(lambda x: not x[0] in to_remove_headers, headers.items())))
    logger                  = current_app.config["logger"]
    workerId                = current_app.config["NODE_ID"] # Get the node_id from the global configuration
    STORAGE_CLIENT:V4Client = current_app.config["STORAGE_CLIENT"]
    BUCKET_ID:str           = current_app.config.get("BUCKET_ID","rory")    
    model_id                = filteredHeaders.get("Model-Id","model-0") #iris
    model_object_id         = "{}_object".format(model_id)
    model_labels_id         = "{}_labels".format(model_id) #iris_model_labels
    extension               = filteredHeaders.get("Extension","npy")
    algorithm               = Constants.ClassificationAlgorithms.KNN_TRAIN
    responseHeaders         = {}

    try:
        logger.debug("Worker starts KNN_TRAIN process -> {}".format(model_id))
        responseHeaders["Start-Time"] = str(arrivalTime)

        model_result        = STORAGE_CLIENT.get_ndarray(key = model_id).result()
        model               = model_result.unwrap().value
        model_labels_result = STORAGE_CLIENT.get_ndarray(key = model_labels_id).result()
        model_labels        = model_labels_result.unwrap().value
        
        metric      = "manhattan"
        n_neighbors =  1
        # knn         = KNN(
        #     model       = NONE,
        #     n_neighbors = Some(n_neighbors),
        #     metric      = Some(metric)
        # )

        knn = KNeighborsClassifier(
            n_neighbors = n_neighbors, 
            metric      = metric,
            weights     = "uniform",
            algorithm   = "brute"
        )
        knn.fit(model, model_labels)
        model_bytes = pickle.dumps(knn)

        x = STORAGE_CLIENT.put(
            key   = model_object_id,
            value = model_bytes,
            tags  = {
                "n_neighbors":str(n_neighbors),
                "metric":metric
            },
            bucket_id = BUCKET_ID
        )

        x           = x.result()
        endTime     = time.time()
        serviceTime = endTime - arrivalTime
        
        logger_metrics = LoggerMetrics(
            operation_type = algorithm, 
            matrix_id      = model_id, 
            worker_id      = workerId,
            algorithm      = algorithm, 
            arrival_time   = arrivalTime, 
            end_time       = endTime, 
            service_time   = serviceTime,
        )
        logger.info(str(logger_metrics))
      
        return Response( #Returns the final response as a label vector + the headers
            response = json.dumps({
                "serviceTime" : endTime,
                "responseTime": str(serviceTime),
                "algorithm"   : algorithm,
            }),
            status   = 200,
            headers  = responseHeaders
        )
    except Exception as e:
        print(e)
        return Response(
            response = None,
            status   = 503,
            headers  = {"Error-Message":str(e)}
        )


@classification.route("/knn/predict",methods = ["POST"])
def knn_predict():
    arrivalTime             = time.time() #Worker start time
    headers                 = request.headers
    headers                 = request.headers
    to_remove_headers       = ["User-Agent","Accept-Encoding","Connection"]
    filteredHeaders         = dict(list(filter(lambda x: not x[0] in to_remove_headers, headers.items())))
    logger                  = current_app.config["logger"]
    workerId                = current_app.config["NODE_ID"] # Get the node_id from the global configuration
    STORAGE_CLIENT:V4Client = current_app.config["STORAGE_CLIENT"]
    model_id                = filteredHeaders.get("Model-Id","model-0") #iris
    model_object_id         = "{}_object".format(model_id)
    model_labels_id         = "{}_labels".format(model_id) #iris_model_labels
    records_test_id         = filteredHeaders.get("Records-Test-Id","matrix-0")
    algorithm               = Constants.ClassificationAlgorithms.KNN_PREDICT
    responseHeaders         = {}
    
    try:
        logger.debug("Worker starts KNN_PREDICT process -> {}".format(model_id))
        responseHeaders["Start-Time"] = str(arrivalTime)
        
        model_bytes_result:Result[GetBytesResponse,Exception] =  STORAGE_CLIENT.get(
            key = model_object_id
        ).result()

        if model_bytes_result.is_err:
            return Response("MODEL_BYTES_NOT_FOUND", status=500)
        
        model_bytes_response:GetBytesResponse = model_bytes_result.unwrap()

        records_test_result = STORAGE_CLIENT.get_ndarray(
            key = records_test_id
        ).result()
        records_test             = records_test_result.unwrap().value
        knn:KNeighborsClassifier = pickle.loads(model_bytes_response.value)
        predicted_labels = knn.predict(
            records_test
        )
        endTime                         = time.time()
        serviceTime                     = endTime - arrivalTime
        responseHeaders["Service-Time"] = str(serviceTime)

        logger_metrics = LoggerMetrics(
            operation_type = algorithm, 
            matrix_id      = records_test_id, 
            worker_id      = workerId,
            algorithm      = algorithm, 
            arrival_time   = arrivalTime, 
            end_time       = endTime, 
            service_time   = serviceTime)
        logger.info(str(logger_metrics))

        return Response( #Returns the final response as a label vector + the headers
            response = json.dumps({"labelVector":predicted_labels.tolist()}),
            status   = 200,
            headers  = {**responseHeaders}
        )
    except Exception as e:
        logger.error(str(e))
        return Response(
            response = None,
            status   = 503,
            headers  = {"Error-Message":str(e)}
        )