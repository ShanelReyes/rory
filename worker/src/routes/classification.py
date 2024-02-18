import time, json
from flask import Blueprint,current_app,request,Response
from rory.core.utils.constants import Constants
from rory.core.classification.secure.sknn import SecureKNearestNeighbors as SKNN
from rory.core.classification.knn import KNearestNeighbors as KNN
from mictlanx.v4.client import Client as V4Client
from rory.core.interfaces.logger_metrics import LoggerMetrics
from utils.utils import Utils
from mictlanx.v4.interfaces.responses import GetNDArrayResponse
import numpy.typing as npt
from option import Result

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

@classification.route("/sknn/predict",methods = ["POST"])
def sknn_pedict():
    local_start_time              = time.time() #Worker start time
    headers                  = request.headers
    to_remove_headers        = ["User-Agent","Accept-Encoding","Connection"]
    filtered_headers          = dict(list(filter(lambda x: not x[0] in to_remove_headers, headers.items())))
    logger                   = current_app.config["logger"]
    worker_id                 = current_app.config["NODE_ID"] # Get the node_id from the global configuration
    STORAGE_CLIENT:V4Client  = current_app.config["STORAGE_CLIENT"]
    BUCKET_ID:str            = current_app.config.get("BUCKET_ID","rory")
    # m                        = int(filtered_headers.get("M",3))
    model_id                 = filtered_headers.get("Model-Id","model-0") #iris
    encrypted_model_id       = "encrypted-{}".format(model_id) #encrypted-iris_model
    model_labels_id          = "{}_labels".format(model_id) #iris_model_labels
    records_test_id          = filtered_headers.get("Records-Test-Id","matrix-0")
    encrypted_records_id     = "encrypted-{}".format(records_test_id) # The id of the encrypted matrix is built
    algorithm                = Constants.ClassificationAlgorithms.SKNN_PREDICT
    _encrypted_model_shape   = filtered_headers.get("Encrypted-Model-Shape",-1)
    _encrypted_model_dtype   = filtered_headers.get("Encrypted-Model-Dtype",-1)
    _encrypted_records_shape = filtered_headers.get("Encrypted-Records-Shape",-1)
    _encrypted_records_dtype = filtered_headers.get("Encrypted-Records-Dtype",-1)

    # logger.debug(str(filtered_headers))
    # logger.debug("SKNN_PREDICT algorithm={}, m={}, model_id={}, encrypted_model_id={}, records_test_id={}, encrypted_records_id={}, ems={}, emd={}".format(algorithm,m,model_id,encrypted_model_id,records_test_id,encrypted_records_id,_encrypted_records_shape,_encrypted_records_dtype))
    
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
    num_chunks                    = int(filtered_headers.get("Num-Chunks",-1))
    response_headers               = {}

    logger.debug({
        "event":"SKNN.PREDICT.STARTED",
        "worker_id":worker_id,
        "model_id":model_id,
        "encrypted_model_id":encrypted_model_id,
        "models_labels_id":model_labels_id,
        "algorithm":algorithm,
        "encrypted_records_id":encrypted_records_id,
        "encrypted_model_shape":_encrypted_model_shape,
        "encrypted_model_dtype":_encrypted_model_dtype,
        "encrypted_records_shape":_encrypted_records_shape,
        "encrypted_records_dtype":_encrypted_records_dtype,
        "num_chunks":num_chunks
    })
    if num_chunks == -1:
        return Response("Num-Chunks header is required", status=503)
    try:
        response_headers["Start-Time"] = str(local_start_time)
        logger.debug({
            "event":"GET.MERGE.NDARRAY.BEFORE",
            "bucket_id":BUCKET_ID,
            "key":encrypted_model_id,
            "shape":_encrypted_model_shape,
            "dtype":_encrypted_model_dtype,
            "num_chunks":num_chunks
        })
        get_merge_encrypted_model_start_time = time.time()
        (encrypted_model, encrypted_model_metadata) = Utils.get_and_merge_ndarray(
            STORAGE_CLIENT = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID, 
            key            = encrypted_model_id,
            num_chunks     = num_chunks, 
            shape          = encrypted_model_shape,
            dtype          = _encrypted_model_dtype
        )

        get_merge_encrypted_model_st = time.time()- get_merge_encrypted_model_start_time
        logger.info({
            "event":"GET.MERGE.NDARRAY",
            "bucket_id":BUCKET_ID,
            "key":encrypted_model_id,
            "num_chunks":num_chunks,
            "shape":_encrypted_model_shape,
            "dtype":_encrypted_model_dtype,
            "service_time":get_merge_encrypted_model_st
        })

        response_headers["Encrypted-Model-Dtype"] = encrypted_model_metadata.tags.get("dtype",encrypted_model.dtype) #["tags"]["dtype"] #Save the data type
        response_headers["Encrypted-Model-Shape"] = encrypted_model_metadata.tags.get("shape",encrypted_model.shape) #Save the shape
        

        # ___________________________________________________________________

        logger.debug({
            "event":"GET.NDARRAY.BEFORE",
            "key":model_labels_id,
            "bucket_id":BUCKET_ID,
        })
        model_labels_get_start_time = time.time()
        model_labels= Utils.get_matrix_or_error(
            client=STORAGE_CLIENT,
            key = model_labels_id,
            bucket_id=BUCKET_ID
        ).value

        model_labels_get_st = time.time() - model_labels_get_start_time
        logger.debug({
            "event":"GET.NDARRAY",
            "bucket_id":BUCKET_ID,
            "key":model_labels_id,
            "shape":str(model_labels.shape), 
            "dtype":str(model_labels.dtype),
            "service_time":model_labels_get_st
        })
        # ___________________________________________________________________
        logger.debug({
            "event":"GET.MERGE.NDARRAY.BEFORE",
            "bucket_id":BUCKET_ID,
            "key":encrypted_model_id,
            "shape":_encrypted_records_shape,
            "dtype":_encrypted_records_dtype,
            "num_chunks":num_chunks
        })
        get_merge_encrypted_records_start_time = time.time()
        (encrypted_records, encrypted_records_metadata) = Utils.get_and_merge_ndarray(
            STORAGE_CLIENT = STORAGE_CLIENT,
            bucket_id      = BUCKET_ID,
            key            = encrypted_records_id,
            num_chunks     = num_chunks,
            shape          = encrypted_records_shape,
            dtype          = _encrypted_records_dtype
        )
        response_headers["Encrypted-Records-Test-Dtype"] = encrypted_records_metadata.tags.get("dtype",encrypted_records.dtype) #["tags"]["dtype"] #Save the data type
        response_headers["Encrypted-Records-Test-Shape"] = encrypted_records_metadata.tags.get("shape",encrypted_records.shape) #Save the shape
        get_merge_encrypted_records_st = time.time()- get_merge_encrypted_records_start_time
        logger.info({
            "event":"GET.MERGE.NDARRAY",
            "bucket_id":BUCKET_ID,
            "key":encrypted_records_id,
            "num_chunks":num_chunks,
            "shape":_encrypted_model_shape,
            "dtype":_encrypted_model_dtype,
            "service_time":get_merge_encrypted_records_st
        })
        # ___________________________________________________________________
        # logger.debug("ENCRYPTED_RECORDS GET SUCCESSFULLY")

        # time.sleep(1000)
        logger.debug({
            "event":"SKNN.PREDICT.BEFORE",
            "encrypted_records_shape":str(encrypted_records.shape),
            "encrypted_records_dtype":str(encrypted_records.dtype),
            "encrypted_model_shape":str(encrypted_model.shape),
            "encrypted_model_dtype":str(encrypted_model.dtype),
            "models_labels_shape":str(model_labels.shape),
            "models_labels_dtype":str(model_labels.dtype),
        })
        sknn_predict_start_time = time.time()
        label_vector = SKNN.predict(
            dataset      = encrypted_records,
            model        = encrypted_model,
            model_labels = model_labels
        )
        sknn_pedict_st = time.time() - sknn_predict_start_time
        logger.info({
            "event":"SKNN.PREDICT",
            "encrypted_records_shape":str(encrypted_records.shape),
            "encrypted_records_dtype":str(encrypted_records.dtype),
            "encrypted_model_shape":str(encrypted_model.shape),
            "encrypted_model_dtype":str(encrypted_model.dtype),
            "models_labels_shape":str(model_labels.shape),
            "models_labels_dtype":str(model_labels.dtype),
            "service_time": sknn_pedict_st
        })

        # logger.debug("SKNN_PREDICT COMPLETED SUCCESSFULLY")
        end_time                         = time.time()
        service_time                     = end_time - local_start_time
        # response_headers["Service-Time"] = str(service_time)
        logger.info({
            "event":"SKNN.PREDICT.COMPLETED",
            "algorithm":algorithm,
            "predict_service_time":sknn_pedict_st,
            "get_encrypted_records_service_time":get_merge_encrypted_records_st,
            "get_model_service_time":get_merge_encrypted_model_st,
            "get_model_labels_service_time":model_labels_get_st,
            "service_time":service_time
        })

        # logger_metrics = LoggerMetrics(
        #     operation_type = algorithm, 
        #     matrix_id      = records_test_id, 
        #     worker_id      = worker_id,
        #     algorithm      = algorithm, 
        #     arrival_time   = local_start_time, 
        #     end_time       = end_time, 
        #     service_time   = service_time)
        # logger.info(str(logger_metrics))

        return Response( #Returns the final response as a label vector + the headers
            response = json.dumps({
                "label_vector":label_vector.tolist(),
                "service_time":service_time
            }),
            status   = 200,
            # headers  = {**response_headers}
        )
    except Exception as e:
        logger.error({
            "msg":str(e)
        })
        # print(e)
        return Response(
            response = None,
            status   = 503,
            headers  = {"Error-Message":str(e)}
        )



@classification.route("/knn/predict",methods = ["POST"])
def knn_predict():
    local_start_time             = time.time() #Worker start time
    headers                      = request.headers
    headers                      = request.headers
    to_remove_headers            = ["User-Agent","Accept-Encoding","Connection"]
    filtered_headers             = dict(list(filter(lambda x: not x[0] in to_remove_headers, headers.items())))
    logger                       = current_app.config["logger"]
    worker_id                    = current_app.config["NODE_ID"] # Get the node_id from the global configuration
    BUCKET_ID:str                = current_app.config.get("BUCKET_ID","rory")
    STORAGE_CLIENT:V4Client      = current_app.config["STORAGE_CLIENT"]
    model_id                     = filtered_headers.get("Model-Id","model-0") #iris
    model_labels_id              = "{}_labels".format(model_id) #iris_model_labels
    records_test_id              = filtered_headers.get("Records-Test-Id","matrix-0")
    algorithm                    = Constants.ClassificationAlgorithms.KNN_PREDICT
    response_headers             = {}
    
    logger.debug({
        "event":"KNN.PREDICT.STARTED",
        "algorithm":algorithm,
        "worker_id":worker_id,
        "model_id":model_id,
        "model_labels_id":model_labels_id,
        "records_test_id":records_test_id,
    })
    try:
        response_headers["Start-Time"] = str(local_start_time)
        logger.debug({
            "event":"GET.NDARRAY.BEFORE",
            "key":model_id,
        })
        get_model_start_time = time.time()
        model_result = STORAGE_CLIENT.get_ndarray(key = model_id,bucket_id=BUCKET_ID).result()
        model        = model_result.unwrap().value
        get_model_st = time.time() - get_model_start_time
        logger.info({
            "event":"GET.NDARRAY",
            "key":model_id,
            "service_time":get_model_st
        })
        # __________________________________________
        
        logger.debug({
            "event":"GET.NDARRAY.BEFORE",
            "key":model_labels_id,
        })
        get_model_labels_start_time = time.time()
        model_labels_result:Result[GetNDArrayResponse,Exception] = STORAGE_CLIENT.get_ndarray(key = model_labels_id,bucket_id=BUCKET_ID).result()
        model_labels:npt.NDArray  = model_labels_result.unwrap().value
        get_model_labels_st = time.time() - get_model_labels_start_time
        logger.info({
            "event":"GET.NDARRAY",
            "key":model_labels_id,
            "service_time":get_model_labels_st
        })
        # __________________________________________

        logger.debug({
            "event":"GET.NDARRAY.BEFORE",
            "key":records_test_id,
        })
        get_records_start_time = time.time()
        record_result:Result[GetNDArrayResponse,Exception] = STORAGE_CLIENT.get_ndarray(key = records_test_id,bucket_id=BUCKET_ID).result()
        records:npt.NDArray  = record_result.unwrap().value
        get_records_st = time.time() - get_records_start_time
        logger.info({
            "event":"GET.NDARRAY",
            "key":records_test_id,
            "service_time":get_records_st
        })

        logger.debug({
            "event":"KNN.PREDICT.BEFORE",
            "records_shape":str(records.shape),
            "records_dtype":str(records.dtype),
            "models_labels_shape":str(model_labels.shape),
            "models_labels_shape":str(model_labels.dtype)
        })
        knn_predict_start_time = time.time()
        label_vector = KNN.predict(
            dataset      = records,
            model        = model,
            model_labels = model_labels
        )
        knn_predict_st = time.time() - knn_predict_start_time
        logger.info({
            "event":"KNN.PREDICT",
            "records_shape":str(records.shape),
            "records_dtype":str(records.dtype),
            "models_labels_shape":str(model_labels.shape),
            "models_labels_shape":str(model_labels.dtype),
            "service_time":knn_predict_st
        })
        # __________________________________________
        end_time                         = time.time()
        service_time                     = end_time - local_start_time
        response_headers["Service-Time"] = str(service_time)
        
        logger.info({
            "event":"KNN.PREDICT.COMPLETED",
            "service_time":time.time() - local_start_time
        })
        return Response( #Returns the final response as a label vector + the headers
            response = json.dumps({
                "label_vector":label_vector.tolist(),
                "service_time":service_time
            }),
            status   = 200,
            headers  = {**response_headers}
        )
    except Exception as e:
        logger.error({
            "msg":str(e)
        })
        return Response(
            response = None,
            status   = 503,
            headers  = {"Error-Message":str(e)}
        )