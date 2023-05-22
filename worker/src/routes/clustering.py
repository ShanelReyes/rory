import time, json
import numpy as np
from flask import Blueprint,current_app,request,Response
from rory.core.clustering.kmeans import kmeans as kMeans
from rory.core.clustering.secure.local.dbsnnc import Dbsnnc
from rory.core.utils.Utils import Utils
from rory.core.utils.constants import Constants
from rory.core.clustering.secure.distributed.skmeans import SKMeans
from rory.core.clustering.secure.distributed.dbskmeans import DBSKMeans
from mictlanx.v3.client import Client
from mictlanx.v3.interfaces.payloads import PutNDArrayPayload

clustering = Blueprint("clustering",__name__,url_prefix = "/clustering")

"""
Description:
    First part of the skmeans process. 
    It stops where client interaction is required and writes the centroids and matrix S to disk.
"""
def skmeans_1(requestHeaders) -> Response:
    arrivalTime           = time.time() #Worker start time
    logger                = current_app.config["logger"]
    workerId              = current_app.config["NODE_ID"] # Get the node_id from the global configuration
    STORAGE_CLIENT:Client = current_app.config["STORAGE_CLIENT"]
    status                = int(requestHeaders.get("Clustering-Status", Constants.ClusteringStatus.START)) 
    isStartStatus         = status == Constants.ClusteringStatus.START #if status is start save it to isStartStatus
    k                     = int(requestHeaders.get("K",3)) # It is passed to integer because the headers are strings
    m                     = int(requestHeaders.get("M",3))
    plainTextMatrixID     = requestHeaders.get("Plaintext-Matrix-Id")
    encryptedMatrixId     = requestHeaders.get("Encrypted-Matrix-Id","")
    UDMId                 = "{}-UDM".format(plainTextMatrixID) 
    encryptedShiftMatrixId = "{}-EncryptedShiftMatrix".format(plainTextMatrixID) #Build the id of Encrypted Shift Matrix
    skmeans               = SKMeans()
    responseHeaders       = {} 
    try:
        responseHeaders["Start-Time"]             = str(arrivalTime)
        encryptedMatrix_response                  = STORAGE_CLIENT.get_ndarray(key = encryptedMatrixId,cache=True,force =isStartStatus).unwrap()# Extract the encrypted dataset
        encryptedMatrix                           = encryptedMatrix_response.value
        responseHeaders["Encrypted-Matrix-Dtype"] = encryptedMatrix_response.metadata.get("dtype",encryptedMatrix.dtype) #["tags"]["dtype"] #Save the data type
        responseHeaders["Encrypted-Matrix-Shape"] = encryptedMatrix_response.metadata.get("shape",encryptedMatrix.shape) #Save the shape
        udm_matrix_response                       = STORAGE_CLIENT.get_ndarray(key = UDMId,force = True, cache=True).unwrap() #Gets the UDM of the storage system
        UDMMatrix                                 = udm_matrix_response.value
        responseHeaders["Udm-Matrix-Dtype"]       = udm_matrix_response.metadata.get("dtype",UDMMatrix.dtype) # Extract the type
        responseHeaders["Udm-Matrix-Shape"]       = udm_matrix_response.metadata.get("shape",UDMMatrix.shape) # Extract the shape
        Cent_iId                                  = "{}-Cent_i".format(plainTextMatrixID) #Build the id of Cent_i
        Cent_jId                                  = "{}-Cent_j".format(plainTextMatrixID) #Build the id of Cent_j
            
        if(isStartStatus): #if the status is start
            __Cent_j = None #There is no Cent_j
        else: 
            Cent_j_response = STORAGE_CLIENT.get_ndarray(key = Cent_iId).unwrap() #Cent_J is extracted from the storage system
            __Cent_j          = Cent_j_response.value
        
        S1,Cent_i,Cent_j,label_vector = skmeans.run1( # The first part of the skmeans is done
            status          = status,
            k               = k,
            m               = m,
            encryptedMatrix = encryptedMatrix, 
            UDM             = UDMMatrix,
            Cent_j          = __Cent_j
        )        
        centI_put_payload = PutNDArrayPayload(key = Cent_iId, ndarray = np.array(Cent_i))
        _  = STORAGE_CLIENT.put_ndarray(centI_put_payload,update=True).unwrap() # Saving Cent_i to storage
        centJ_put_payload = PutNDArrayPayload(key = Cent_jId, ndarray = np.array(Cent_j))
        _  = STORAGE_CLIENT.put_ndarray(centJ_put_payload,update=True).unwrap() # Saving Cent_j to storage
        encrypted_shiftmatrix_put_payload = PutNDArrayPayload(key = encryptedShiftMatrixId, ndarray = np.array(S1))
        _  = STORAGE_CLIENT.put_ndarray(encrypted_shiftmatrix_put_payload,update=True).unwrap() # Saving S1 matrix to storage
        serviceTime                                  = time.time() - arrivalTime
        responseHeaders["Service-Time"]              = str(serviceTime)
        responseHeaders["Iterations"]                = str(int(requestHeaders.get("Iterations",0)) + 1) #Saves the number of iterations in the header
        responseHeaders["Encrypted-Shift-Matrix-Id"] = encryptedShiftMatrixId #Save the id of the encrypted shift matrix    
                
        logger.info("SKMEANS_1 {} {} {} {} {} {}".format(#Show the final result in a logger
            workerId,
            plainTextMatrixID,
            # encryptedMatrixId,
            k,
            m,
            responseHeaders.get("Iterations",0),
            serviceTime
        ))
                
        return Response( #Returns the final response as a label vector + the headers
            response = json.dumps({"labelVector":label_vector}),
            status   = 200,
            headers  = responseHeaders
            # {**requestHeaders, **responseHeaders}
        )
    except Exception as e:
        logger.error( encryptedMatrixId+" "+str(e) )
        print("_"*20)
        return Response(None,status = 503,headers = {"Error-Message":str(e)} )

"""
Description:
    Second part of the skmeans process. 
    It starts when it receives S (decrypted matrix) from the client.
    If S is zero process ends
"""
def skmeans_2(requestHeaders):
    arrivalTime           = time.time()
    logger                = current_app.config["logger"]
    workerId              = current_app.config["NODE_ID"]
    STORAGE_CLIENT:Client = current_app.config["STORAGE_CLIENT"]
    status                = int(requestHeaders.get("Clustering-Status",Constants.ClusteringStatus.START ))
    k                     = int(requestHeaders.get("K",3))
    m                     = int(requestHeaders.get("M",3))
    encryptedMatrixId     = requestHeaders["Encrypted-Matrix-Id"]
    plainTextMatrixID     = requestHeaders["Plaintext-Matrix-Id"]
    shiftMatrixId         = requestHeaders.get("Shift-Matrix-Id","{}-Shift-Matrix".format(plainTextMatrixID))
    UDM_id = "{}-UDM".format(plainTextMatrixID)
    iterations            = int(requestHeaders.get("Iterations",0))
    responseHeaders       = {}
    start_time            = requestHeaders.get("Start-Time","0.0")
    try:
        # UDM                    = np.fromfile(UDMPath,dtype = requestHeaders["Udm-Matrix-Dtype"]).reshape(eval(requestHeaders["Udm-Matrix-Shape"]))
        UDM_response             = STORAGE_CLIENT.get_ndarray(key = UDM_id,cache= True).unwrap()
        UDM                      = UDM_response.value
        shiftMatrix_get_response = STORAGE_CLIENT.get_ndarray(key = shiftMatrixId).unwrap()
        shiftMatrix              = shiftMatrix_get_response.value
        isZero                   = Utils.verifyZero(shiftMatrix)
        if(isZero): #If Shift matrix is zero
            responseHeaders["Clustering-Status"]  = Constants.ClusteringStatus.COMPLETED #Change the status to COMPLETED
            totalServiceTime                      = time.time() - float(start_time) #The service time is calculated
            responseHeaders["Total-Service-Time"] = str(totalServiceTime) #Save the service time

            logger.info("SKMEANS_2 {} {} {} {} {} {} ".format( #Show the final result in a logger
                workerId,
                encryptedMatrixId,
                k,
                m,
                iterations,
                totalServiceTime
            ))
            print("_"*50)
            return Response( #Return none and headers
                response = None, 
                status   = 204, 
                # headers  = {**requestHeaders, **responseHeaders}
                headers  = responseHeaders
            )
        else: #If Shift matrix is not zero
            skmeans         = SKMeans() 
            responseHeaders["Clustering-Status"] = Constants.ClusteringStatus.WORK_IN_PROGRESS #The status is changed to WORK IN PROGRESS
            attibutes_shape = eval(requestHeaders["Encrypted-Matrix-Shape"]) # extract the attributes of shape
            _UDM            = skmeans.run_2( # The second part of the skmeans starts
                status      = status,
                k           = k,
                UDM         = UDM,
                attributes  = attibutes_shape[1],
                shiftMatrix = shiftMatrix,
            )
            udm_put_payload = PutNDArrayPayload( key = UDM_id, ndarray = np.array(_UDM))
            _                               = STORAGE_CLIENT.put_ndarray(udm_put_payload).unwrap() # UDM is extracted from the storage system
            end_time                        = time.time()
            service_time                    = end_time - arrivalTime  #Service time is calculated
            responseHeaders["End-Time"]     = str(end_time)
            responseHeaders["Service-Time"] = str(service_time)
            logger.info("SKMEANS_2 {} {} {} {} {} {}".format( #Show the final result in a logger
                workerId,
                plainTextMatrixID,
                # encryptedMatrixId,
                k,
                m,
                iterations,
                service_time
            ))
            print("_"*50)
            return Response( #Return none and headers
                response = None,
                status   = 204, 
                headers  = responseHeaders
            )
    except Exception as e:
        logger.error(encryptedMatrixId+" "+str(e))
        print("_"*40)
        return Response(
            response = None,
            status   = 503,
            headers  = {"Error-Message":str(e)} 
        )

@clustering.route("/skmeans",methods = ["POST"])
def skmeans():
    headers = request.headers
    head = ["User-Agent","Accept-Encoding","Connection"]
    filteredHeaders = dict(list(filter(lambda x: not x[0] in head, headers.items())))
    step_index = int(filteredHeaders.get("Step-Index",1))
    response = Response()
    if step_index == 1:
        return skmeans_1(filteredHeaders)
    elif step_index == 2:
        return skmeans_2(filteredHeaders)
    else:
        return response

@clustering.route("/kmeans",methods = ["POST"])
def kmeans():
    arrivalTime           = time.time() #System startup time
    headers               = request.headers
    to_remove_headers      = ["User-Agent","Accept-Encoding","Connection"]
    filteredHeaders       = dict(list(filter(lambda x: not x[0] in to_remove_headers, headers.items())))
    logger                = current_app.config["logger"]
    workerId              = current_app.config["NODE_ID"] # Get the node_id from the global configuration
    STORAGE_CLIENT:Client = current_app.config["STORAGE_CLIENT"]
    plainTextMatrixId     = filteredHeaders.get("Plaintext-Matrix-Id")
    k                     = int(filteredHeaders.get("K",3))
    responseHeaders       = {}
    try:
        plainTextMatrix_response        = STORAGE_CLIENT.get_ndarray(key = plainTextMatrixId, cache=True).unwrap()
        plainTextMatrix                 = plainTextMatrix_response.value
        result                          = kMeans(k = k, plaintext_matrix = plainTextMatrix)
        serviceTime                     = time.time() - arrivalTime
        responseHeaders["Service-Time"] = str(serviceTime)
        
        logger.info("KMEANS {} {} {} {} {} {}".format(
            workerId,
            plainTextMatrixId,
            k,
            0,
            result.n_iterations,
            serviceTime
        ))
        return Response( #Returns the final response as a label vector + the headers
            response = json.dumps({"labelVector":result.label_vector.tolist()}),
            status   = 200,
            headers  = {**responseHeaders}
        )
    except Exception as e:
        print(e)
        return Response(
            response = None,
            status   = 503,
            headers = {"Error-Message":str(e)}
        )


"""
Description:
    First part of the dbskmeans process. 
    It stops where client interaction is required and writes the centroids and matrix S to disk.
"""
def dbskmeans_1(requestHeaders) -> Response:
    arrivalTime           = time.time() #System startup time
    logger                = current_app.config["logger"]
    workerId              = current_app.config["NODE_ID"] # Get the node_id from the global configuration
    STORAGE_CLIENT:Client = current_app.config["STORAGE_CLIENT"]
    status                = int(requestHeaders.get("Clustering-Status", Constants.ClusteringStatus.START)) 
    isStartStatus         = status == Constants.ClusteringStatus.START #if status is start save it to isStartStatus
    k                     = int(requestHeaders.get("K",3)) # It is passed to integer because the headers are strings
    m                     = int(requestHeaders.get("M",3))

    plainTextMatrixID     = requestHeaders.get("Plaintext-Matrix-Id")
    encryptedMatrixId     = requestHeaders.get("Encrypted-Matrix-Id","")
    UDMId                 = "{}-encrypted-UDM".format(plainTextMatrixID) 
    Cent_iId                            = "{}-Cent_i".format(plainTextMatrixID) #Build the id of Cent_i
    Cent_jId                            = "{}-Cent_j".format(plainTextMatrixID) #Build the id of Cent_j
    encryptedShiftMatrixId = "{}-EncryptedShiftMatrix".format(plainTextMatrixID) #Build the id of Encrypted Shift Matrix

    dbskmeans             = DBSKMeans()
    responseHeaders       = {}
    try:
        if(isStartStatus):
            responseHeaders["Start-Time"]      = arrivalTime
        
        encryptedMatrix_response = STORAGE_CLIENT.get_ndarray(key = encryptedMatrixId,cache=True, force=isStartStatus).unwrap() # Extract the encrypted dataset
        encryptedMatrix          = encryptedMatrix_response.value
        responseHeaders["Encrypted-Matrix-Dtype"] = encryptedMatrix_response.metadata.get("dtype",encryptedMatrix.dtype) #Save the data type
        responseHeaders["Encrypted-Matrix-Shape"] = encryptedMatrix_response.metadata.get("shape",encryptedMatrix.shape) #Save the shape
        
        UDMMatrix_response                  = STORAGE_CLIENT.get_ndarray(key= UDMId).unwrap() #Gets the UDM of the storage system
        UDMMatrix                           = UDMMatrix_response.value
        responseHeaders["Udm-Matrix-Dtype"] = UDMMatrix_response.metadata.get("dtype",UDMMatrix.dtype) # Extract the type
        responseHeaders["Udm-Matrix-Shape"] = UDMMatrix_response.metadata.get("shape",UDMMatrix.shape) # Extract the shape

        if(isStartStatus): #if the status is start
            __Cent_j = None #There is no Cent_j
        else: 
            __Cent_j_response = STORAGE_CLIENT.get_ndarray(key = Cent_iId).unwrap() #Cent_J is extracted from the storage system
            __Cent_j = __Cent_j_response.value
        
        S1,Cent_i,Cent_j,label_vector = dbskmeans.run1(
            status           = status,
            k                = k,
            m                = m,
            encryptedMatrix  = encryptedMatrix,
            UDM              = UDMMatrix,
            Cent_j           = __Cent_j
        )    
        cent_i_put_payload           = PutNDArrayPayload(key = Cent_iId, ndarray = np.array(Cent_i))
        x                            = STORAGE_CLIENT.put_ndarray(payload= cent_i_put_payload, update=True) # Saving Cent_i to storage
        print("Cent_i",x)
        cent_j_put_payload           = PutNDArrayPayload(key = Cent_jId, ndarray = np.array(Cent_j))
        x                            = STORAGE_CLIENT.put_ndarray(payload=cent_j_put_payload,update=True) # Saving Cent_j to storage
        print("Cent_j",x)
        encrypted_matrix_put_payload = PutNDArrayPayload(key = encryptedShiftMatrixId,  ndarray = np.array(S1)) 
        x                      = STORAGE_CLIENT.put_ndarray(payload=encrypted_matrix_put_payload,update=True) # Saving S1 matrix to storage
        print("S1",x)
        serviceTime            = time.time() - arrivalTime
        responseHeaders["Service-Time"] = str(serviceTime)
        
        responseHeaders["Iterations"]                = str(int(requestHeaders.get("Iterations",0)) + 1) #Saves the number of iterations in the header
        responseHeaders["Encrypted-Shift-Matrix-Id"] = encryptedShiftMatrixId #Save the id of the encrypted shift matrix
        responseHeaders["Service-Time"]              = str(serviceTime) #Save the service time
    
        logger.info("DBSKMEANS_1 {} {} {} {} {} {}".format( #Show the final result in a logger
            workerId,
            plainTextMatrixID,
            # encryptedMatrixId,
            k,
            m,
            responseHeaders.get("Iterations",0),
            serviceTime
        ))
        
        return Response( #Returns the final response as a label vector + the headers
            response = json.dumps({"labelVector":label_vector }),
            status   = 200,
            headers  = {**requestHeaders, **responseHeaders}
        )
    except Exception as e:
        print("ERROR {}".format(e))
        logger.error( encryptedMatrixId+" "+str(e) )
        print("_"*20)
        return Response(None,status = 503,headers = {"error-Message":str(e)} )

    


    

"""
Description:
    Second part of the dbskmeans process. 
    It starts when it receives S (decrypted matrix) from the client.
    If S is zero process ends
"""
def dbskmeans_2(requestHeaders):
    logger                = current_app.config["logger"]
    workerId              = current_app.config["NODE_ID"]
    STORAGE_CLIENT:Client = current_app.config["STORAGE_CLIENT"]
    arrivalTime           = time.time()
    status                = int(requestHeaders.get("Clustering-Status",Constants.ClusteringStatus.START ))
    k                     = requestHeaders.get("K",3)
    m                     = requestHeaders.get("M",3)
    iterations            = requestHeaders.get("Iterations",0)
    start_time            = requestHeaders.get("Start-Time","0.0")
    responseHeaders       = {}
    encryptedMatrixId     = requestHeaders["Encrypted-Matrix-Id"]
    plainTextMatrixID     = requestHeaders["Plaintext-Matrix-Id"]
    shiftMatrixId         = requestHeaders.get("Shift-Matrix-Id","{}-ShiftMatrix".format(plainTextMatrixID))
    shiftMatrixOpeId      = requestHeaders.get("Shift-Matrix-Ope-Id","{}-ShiftMatrixOpe".format(plainTextMatrixID))
    UDM_id              = "{}-encrypted-UDM".format(plainTextMatrixID)
    try:
        shiftMatrix_response = STORAGE_CLIENT.get_ndarray(key = shiftMatrixId).unwrap()
        shiftMatrix          = shiftMatrix_response.value
        isZero               = Utils.verifyZero(shiftMatrix)
        if(isZero): #If Shift matrix is zero
            responseHeaders["Clustering-Status"]  = Constants.ClusteringStatus.COMPLETED #Change the status to COMPLETED
            totalServiceTime                      = time.time() - float(start_time) #The service time is calculated
            responseHeaders["Total-Service-Time"] = str(totalServiceTime) #Save the service time
            logger.info("DBSKMEANS_2 {} {} {} {} {} {} ".format( #Show the final result in a logger
                workerId,
                encryptedMatrixId,
                k,
                m,
                iterations,
                totalServiceTime
            ))
            print("_"*50)
            return Response( #Return none and headers
                response = None, 
                status   = 204, 
                headers  = {**requestHeaders, **responseHeaders}
            )
        else: #If Shift matrix is not zero
            UDMMatrix               = STORAGE_CLIENT.get_ndarray(key = UDM_id,cache=True).unwrap()
            responseHeaders["Clustering-Status"] = Constants.ClusteringStatus.WORK_IN_PROGRESS #The status is changed to WORK IN PROGRESS
            dbskmeans               = DBSKMeans() 
            attibutes_shape         = eval(requestHeaders["Encrypted-Matrix-Shape"]) # extract the attributes of shape
            shiftMatrixOpe_response = STORAGE_CLIENT.get_ndarray(key = shiftMatrixOpeId).unwrap()
            shiftMatrixOpe          = shiftMatrixOpe_response.value
            _UDMMatrix              = dbskmeans.run_2( # The second part of the skmeans starts
                status           = status,
                k                = k,
                UDM              = UDMMatrix.value,
                attributes       = attibutes_shape[1],
                shiftMatrix      = shiftMatrixOpe,
            )

            udm_put_payload = PutNDArrayPayload( key = UDM_id, ndarray= np.array(_UDMMatrix))
            _                               = STORAGE_CLIENT.put_ndarray(payload=udm_put_payload, update=True).unwrap() # UDM is extracted from the storage system
            end_time                        = time.time()
            service_time                    = end_time - arrivalTime  #Service time is calculated
            responseHeaders["End-Time"]     = str(end_time)
            responseHeaders["Service-Time"] = str(service_time)

            logger.info("DBSKMEANS_2 {} {} {} {} {} {}".format( #Show the final result in a logger
                workerId,
                plainTextMatrixID,
                # encryptedMatrixId,
                k,
                m,
                iterations,
                service_time
            ))
            print("_"*50)
            return Response( #Return none and headers
                response = None,
                status   = 204, 
                headers  = {**requestHeaders, **responseHeaders}
            )
    except Exception as e:
        logger.error(encryptedMatrixId+" "+str(e))
        print("_"*40)
        return Response(
            response = None,
            status   = 503,
            headers  = {"Error-Message":str(e)} 
        )

"""
Description:
    DBSKMEANS algorithm
"""
@clustering.route("/dbskmeans", methods = ["POST"])
def dbskmeans():
    headers = request.headers
    head = ["User-Agent","Accept-Encoding","Connection"]
    filteredHeaders = dict(list(filter(lambda x: not x[0] in head, headers.items())))
    step_index = int(filteredHeaders.get("Step-Index",1))
    response = Response()
    if step_index == 1:
        return dbskmeans_1(filteredHeaders)
    elif step_index == 2:
        return dbskmeans_2(filteredHeaders)
    else:
        return response


@clustering.route("/dbsnnc", methods = ["POST"])
def dbsnnc():
    arrivalTime         = time.time() #System startup time
    headers             = request.headers
    head                = ["User-Agent","Accept-Encoding","Connection"]
    filteredHeaders     = dict(list(filter(lambda x: not x[0] in head, headers.items())))
    logger              = current_app.config["logger"]
    STORAGE_CLIENT:Client = current_app.config["STORAGE_CLIENT"]
    workerId            = current_app.config["NODE_ID"] # Get the node_id from the global configuration
    plainTextMatrixId   = filteredHeaders.get("Plaintext-Matrix-Id")
    encryptedMatrixId   = filteredHeaders.get("Encrypted-Matrix-Id","")
    encrypted_threshold = filteredHeaders.get("Encrypted-Threshold")
    UDMId               = "{}-encrypted-UDM".format(plainTextMatrixId)  
    responseHeaders     = {}
    try:        
        UDMMatrix_response  = STORAGE_CLIENT.get_ndarray(key = UDMId).unwrap()
        UDMMatrix           = UDMMatrix_response.value
        result = Dbsnnc.run(
            EDM                 = UDMMatrix,
            encrypted_threshold = float(encrypted_threshold)
        )
        serviceTime                     = time.time() - arrivalTime
        responseHeaders["Service-Time"] = str(serviceTime)
        
        logger.info("DBSNNC {} {} {} {} {} {} ".format(
            workerId,
            plainTextMatrixId,
            # encryptedMatrixId,
            0,
            0,
            0,
            serviceTime
        ))
        return Response( #Returns the final response as a label vector + the headers
            response = json.dumps({"labelVector":result.label_vector}),
            status   = 200,
            headers  = responseHeaders
        )
    except Exception as e:
        print(e)
        return Response(
            response = None,
            status   = 503,
            headers  = {"Error-Message":e})
    

    """
    try:
        if(isStartStatus): #If status is start
            responseHeaders["Start-Time"]             = arrivalTime
            encryptedMatrix_response = STORAGE_CLIENT.get_ndarray(key = encryptedMatrixId)# Extract the encrypted dataset
            encryptedMatrix          = encryptedMatrix_response.value
            responseHeaders["Encrypted-Matrix-Dtype"] = encryptedMatrix_response.metadata.tags.get("dtype",encryptedMatrix.dtype) #Save the data type
            responseHeaders["Encrypted-Matrix-Shape"] = encryptedMatrix_response.metadata.tags.get("shape",encryptedMatrix.shape) #Save the shape
        else: 
            path            = "{}/{}".format(SINK_PATH,encryptedMatrixId) # Build the path to get the dataset
            encryptedMatrix = np.fromfile(path, dtype = requestHeaders["Encrypted-Matrix-Dtype"]).reshape(eval(requestHeaders["Encrypted-Matrix-Shape"]))
        UDMMatrix_response                  = STORAGE_CLIENT.get_ndarray(id = UDMId, sink_path = SINK_PATH, delete = False) #Gets the UDM of the storage system
        UDMMatrix                           = UDMMatrix_response.value
        responseHeaders["Udm-Matrix-Dtype"] = UDMMatrix_response.metadata.tags.get("dtype",UDMMatrix.dtype) # Extract the type
        responseHeaders["Udm-Matrix-Shape"] = UDMMatrix_response.metadata.tags.get("shape",UDMMatrix.shape) # Extract the shape
        Cent_iId                            = "{}-Cent_i".format(plainTextMatrixID) #Build the id of Cent_i
        Cent_jId                            = "{}-Cent_j".format(plainTextMatrixID) #Build the id of Cent_j
        
        if(isStartStatus): #if the status is start
            __Cent_j = None #There is no Cent_j
        else: 
            __Cent_j_response = STORAGE_CLIENT.get_ndarray(id = Cent_iId, sink_path = SINK_PATH, delete = True) #Cent_J is extracted from the storage system
            __Cent_j = __Cent_j_response.value

        S1,Cent_i,Cent_j,label_vector = dbskmeans.run1(
            status           = status,
            k                = k,
            m                = m,
            encryptedMatrix  = encryptedMatrix,
            UDM              = UDMMatrix,
            Cent_j           = __Cent_j
        )    
        _                      = STORAGE_CLIENT.put_ndarray(id = Cent_iId, matrix = np.array(Cent_i)) # Saving Cent_i to storage
        _                      = STORAGE_CLIENT.put_ndarray(id = Cent_jId, matrix = np.array(Cent_j)) # Saving Cent_j to storage
        encryptedShiftMatrixId = "{}-EncryptedShiftMatrix".format(plainTextMatrixID) #Build the id of Encrypted Shift Matrix
        _                      = STORAGE_CLIENT.put_ndarray(id = encryptedShiftMatrixId, matrix = np.array(S1)) # Saving S1 matrix to storage
        serviceTime            = time.time() - arrivalTime
        responseHeaders["Service-Time"] = str(serviceTime)
        
        responseHeaders["Iterations"]                = str(int(requestHeaders.get("Iterations",0)) + 1) #Saves the number of iterations in the header
        responseHeaders["Encrypted-Shift-Matrix-Id"] = encryptedShiftMatrixId #Save the id of the encrypted shift matrix
        responseHeaders["Service-Time"]              = str(serviceTime) #Save the service time
    
        logger.info("DBSKMEANS_1 {} {} {} {} {} {}".format( #Show the final result in a logger
            workerId,
            encryptedMatrixId,
            k,
            m,
            responseHeaders.get("Iterations",0),
            serviceTime
        ))
        
        return Response( #Returns the final response as a label vector + the headers
            response = json.dumps({"labelVector":label_vector.tolist()}),
            status   = 200,
            headers  = {**requestHeaders, **responseHeaders}
        )
    except Exception as e:
        print("ERROR {}".format(e))
        logger.error( encryptedMatrixId+" "+str(e) )
        print("_"*20)
        return Response(None,status = 503,headers = {"error-Message":str(e)} )
           """