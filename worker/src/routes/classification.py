import time, json, operator
import numpy as np
import numpy.typing as npt
from typing import Tuple
from flask import Blueprint,current_app,request,Response
from rory.core.utils.constants import Constants
from rory.core.classification.secure.sknn import SecureKNearestNeighbors as SKNN
from rory.core.classification.knn import KNearestNeighbors as KNN
#from sklearn.neighbors import KNeighborsClassifier
from mictlanx.v4.client import Client as V4Client
from mictlanx.v4.interfaces.responses import Metadata
import pickle
from mictlanx.v4.interfaces.responses import GetNDArrayResponse,GetBytesResponse
from rory.core.interfaces.logger_metrics import LoggerMetrics
from option import Option,NONE,Some,Result
from functools import reduce

classification = Blueprint("classification",__name__,url_prefix = "/classification")

def get_and_merge_ndarray(STORAGE_CLIENT:V4Client,bucket_id:str, key:str,num_chunks:int, shape:tuple,dtype:str)->Tuple[npt.NDArray,Metadata]:
    encryptedMatrix_result:Result[GetBytesResponse,Exception] = STORAGE_CLIENT.get_and_merge_with_num_chunks(bucket_id=bucket_id,key=key,num_chunks=num_chunks).result()
    if encryptedMatrix_result.is_err:
        raise Exception("{} not found".format(key))
    
    encryptedMatrix_response = encryptedMatrix_result.unwrap()
    _encryptedMatrix         = np.frombuffer(encryptedMatrix_response.value,dtype=dtype)
    expected_shape           = reduce(operator.mul,shape)
    if not _encryptedMatrix.size == expected_shape:
        raise Exception("Matrix sizes are not equal: calculated: {} != expected: {}".format(_encryptedMatrix.size, expected_shape ))
    
    encryptedMatrix = _encryptedMatrix.reshape(shape)
    return (encryptedMatrix,encryptedMatrix_response.metadata)

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
    arrivalTime              = time.time() #Worker start time
    headers                  = request.headers
    to_remove_headers        = ["User-Agent","Accept-Encoding","Connection"]
    filteredHeaders          = dict(list(filter(lambda x: not x[0] in to_remove_headers, headers.items())))
    logger                   = current_app.config["logger"]
    workerId                 = current_app.config["NODE_ID"] # Get the node_id from the global configuration
    STORAGE_CLIENT:V4Client  = current_app.config["STORAGE_CLIENT"]
    BUCKET_ID:str            = current_app.config.get("BUCKET_ID","rory")
    m                        = int(filteredHeaders.get("M",3))
    model_id                 = filteredHeaders.get("Model-Id","model-0") #iris
    encrypted_model_id       = "encrypted-{}".format(model_id) #encrypted-iris_model
    model_labels_id          = "{}_labels".format(model_id) #iris_model_labels
    records_test_id          = filteredHeaders.get("Records-Test-Id","matrix-0")
    encrypted_records_id     = "encrypted-{}".format(records_test_id) # The id of the encrypted matrix is built
    algorithm                = Constants.ClassificationAlgorithms.SKNN_PREDICT
    _encrypted_model_shape   = filteredHeaders.get("Encrypted-Model-Shape",-1)
    _encrypted_model_dtype   = filteredHeaders.get("Encrypted-Model-Dtype",-1)
    _encrypted_records_shape = filteredHeaders.get("Encrypted-Records-Shape",-1)
    _encrypted_records_dtype = filteredHeaders.get("Encrypted-Records-Dtype",-1)

    logger.debug(str(filteredHeaders))
    logger.debug("SKNN_PREDICT algorithm={}, m={}, model_id={}, encrypted_model_id={}, records_test_id={}, encrypted_records_id={}, ems={}, emd={}".format(algorithm,m,model_id,encrypted_model_id,records_test_id,encrypted_records_id,_encrypted_records_shape,_encrypted_records_dtype))
    
    if _encrypted_model_dtype == -1:
        return Response("Encrypted-Model-Dtype", status=500)
    if _encrypted_model_shape == -1 :
        return Response("Encrypted-Model-Shape header is required", status=500)
    
    if _encrypted_records_dtype == -1:
        return Response("Encrypted-Records-Dtype", status=500)
    if _encrypted_records_shape == -1 :
        return Response("Encrypted-Records-Shape header is required", status=500)
    
    encrypted_model_shape:tuple   = eval(_encrypted_model_shape)
    encrypted_records_shape:tuple = eval(_encrypted_records_shape)
    num_chunks                    = int(filteredHeaders.get("Num-Chunks",-1))
    responseHeaders               = {}

    if num_chunks == -1:
        return Response("Num-Chunks header is required", status=503)

    try:
        # logger.debug("Worker starts SKNN_PREDICT process -> {}".format(model_id))
        logger.debug("get encrypted model {}".format(encrypted_model_id))
        responseHeaders["Start-Time"] = str(arrivalTime)

        # encryptedModel_response = STORAGE_CLIENT.get_and_merge_ndarray(key = encrypted_model_id )
        # x                       = encryptedModel_response.result()
        # encryptedModel_response:GetNDArrayResponse = x.unwrap()
        # encryptedModel                             = encryptedModel_response.value
        (encryptedModel, encrypted_model_metadata) = get_and_merge_ndarray(
            STORAGE_CLIENT = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID, 
            key            = encrypted_model_id,
            num_chunks     = num_chunks, 
            shape          = encrypted_model_shape,
            dtype          = _encrypted_model_dtype
        )

        responseHeaders["Encrypted-Model-Dtype"] = encrypted_model_metadata.tags.get("dtype",encryptedModel.dtype) #["tags"]["dtype"] #Save the data type
        responseHeaders["Encrypted-Model-Shape"] = encrypted_model_metadata.tags.get("shape",encryptedModel.shape) #Save the shape
        
        logger.debug("ENCRYPTED_MODEL GET SUCCESSFULLY")

        model_labels_result = STORAGE_CLIENT.get_ndarray(key = model_labels_id).result()
        model_labels        = model_labels_result.unwrap().value
        logger.debug("MODEL_LABELS GET SUCCESSFULLY")

        (encryptedRecords, encrypted_records_metadata) = get_and_merge_ndarray(
            STORAGE_CLIENT = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = encrypted_records_id,
            num_chunks     = num_chunks,
            shape          = encrypted_records_shape,
            dtype          = _encrypted_records_dtype
        )

        responseHeaders["Encrypted-Records-Test-Dtype"] = encrypted_records_metadata.tags.get("dtype",encryptedRecords.dtype) #["tags"]["dtype"] #Save the data type
        responseHeaders["Encrypted-Records-Test-Shape"] = encrypted_records_metadata.tags.get("shape",encryptedRecords.shape) #Save the shape
        logger.debug("ENCRYPTED_RECORDS GET SUCCESSFULLY")

        label_vector = SKNN.predict(
            dataset      = encryptedRecords,
            model        = encryptedModel,
            model_labels = model_labels
        )
        logger.debug("SKNN_PREDICT COMPLETED SUCCESSFULLY")
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
    pass
    # arrivalTime             = time.time() #Worker start time
    # headers                 = request.headers
    # headers                 = request.headers
    # to_remove_headers       = ["User-Agent","Accept-Encoding","Connection"]
    # filteredHeaders         = dict(list(filter(lambda x: not x[0] in to_remove_headers, headers.items())))
    # logger                  = current_app.config["logger"]
    # workerId                = current_app.config["NODE_ID"] # Get the node_id from the global configuration
    # STORAGE_CLIENT:V4Client = current_app.config["STORAGE_CLIENT"]
    # BUCKET_ID:str           = current_app.config.get("BUCKET_ID","rory")    
    # model_id                = filteredHeaders.get("Model-Id","model-0") #iris
    # model_object_id         = "{}_object".format(model_id)
    # model_labels_id         = "{}_labels".format(model_id) #iris_model_labels
    # extension               = filteredHeaders.get("Extension","npy")
    # algorithm               = Constants.ClassificationAlgorithms.KNN_TRAIN
    # responseHeaders         = {}

    # try:
    #     logger.debug("KNN_TRAIN algorithm={}, model_id={}".format(algorithm,model_id))
    #     #logger.debug("KNN_TRAIN process -> {}".format(model_id))
    #     responseHeaders["Start-Time"] = str(arrivalTime)

    #     model_result        = STORAGE_CLIENT.get_ndarray(key = model_id).result()
    #     model               = model_result.unwrap().value
    #     logger.debug("MODEL GET SUCCESSFULLY")

    #     model_labels_result = STORAGE_CLIENT.get_ndarray(key = model_labels_id).result()
    #     model_labels        = model_labels_result.unwrap().value
    #     logger.debug("MODEL_LABELS GET SUCCESSFULLY")
        
    #     metric      = "manhattan"
    #     n_neighbors =  1
        
    #     knn = KNeighborsClassifier(
    #         n_neighbors = n_neighbors, 
    #         metric      = metric,
    #         weights     = "uniform",
    #         algorithm   = "brute"
    #     )
    #     logger.debug("INIT KNeighborsClassifier COMPLETED SUCCESSFULLY")

    #     knn.fit(model, model_labels)
    #     logger.debug("KNN_FIT COMPLETED SUCCESSFULLY")

    #     model_bytes = pickle.dumps(knn)

    #     x = STORAGE_CLIENT.put(
    #         key   = model_object_id,
    #         value = model_bytes,
    #         tags  = {
    #             "n_neighbors":str(n_neighbors),
    #             "metric":metric
    #         },
    #         bucket_id = BUCKET_ID
    #     )
    #     logger.debug("MODEL_OBJECT PUT SUCCESSFULLY")

    #     x           = x.result()
    #     endTime     = time.time()
    #     serviceTime = endTime - arrivalTime
        
    #     logger_metrics = LoggerMetrics(
    #         operation_type = algorithm, 
    #         matrix_id      = model_id, 
    #         worker_id      = workerId,
    #         algorithm      = algorithm, 
    #         arrival_time   = arrivalTime, 
    #         end_time       = endTime, 
    #         service_time   = serviceTime,
    #     )
    #     logger.info(str(logger_metrics))
      
    #     return Response( #Returns the final response as a label vector + the headers
    #         response = json.dumps({
    #             "serviceTime" : endTime,
    #             "responseTime": str(serviceTime),
    #             "algorithm"   : algorithm,
    #         }),
    #         status   = 200,
    #         headers  = responseHeaders
    #     )
    # except Exception as e:
    #     print(e)
    #     return Response(
    #         response = None,
    #         status   = 503,
    #         headers  = {"Error-Message":str(e)}
    #     )


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
    #model_id                = "{}_obj".format(model_id)
    model_labels_id         = "{}_labels".format(model_id) #iris_model_labels
    records_test_id         = filteredHeaders.get("Records-Test-Id","matrix-0")
    algorithm               = Constants.ClassificationAlgorithms.KNN_PREDICT
    responseHeaders         = {}
    
    try:
        logger.debug("KNN_PREDICT algorithm={}, model_id={}, records_test_id={}".format(algorithm,model_id,records_test_id))
        #logger.debug("Worker starts KNN_PREDICT process -> {}".format(model_id))
        responseHeaders["Start-Time"] = str(arrivalTime)

        model_result = STORAGE_CLIENT.get_ndarray(key = model_id).result()
        model        = model_result.unwrap().value
        logger.debug("MODEL GET SUCCESSFULLY")

        model_labels_result = STORAGE_CLIENT.get_ndarray(key = model_labels_id).result()
        model_labels  = model_labels_result.unwrap().value
        logger.debug("MODEL_LABELS GET SUCCESSFULLY")

        record_result = STORAGE_CLIENT.get_ndarray(key = records_test_id).result()
        records  = record_result.unwrap().value
        logger.debug("RECORDS GET SUCCESSFULLY")

        label_vector = KNN.predict(
            dataset      = records,
            model        = model,
            model_labels = model_labels
        )
        logger.debug("KNN_PREDICT COMPLETED SUCCESSFULLY")
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
        logger.error(str(e))
        return Response(
            response = None,
            status   = 503,
            headers  = {"Error-Message":str(e)}
        )