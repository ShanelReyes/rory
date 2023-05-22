from sklearn import preprocessing
from time import time
from security.cryptosystem.dataowner import DataOwner
import pandas as pd
import numpy as np

from sklearn import preprocessing
from time import time
# 
from security.cryptosystem.dataowner import DataOwner
from clustering.secure.skmeans import SKMeans
from clustering.secure.dbskmeans import DBSKMeans

from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score,davies_bouldin_score,calinski_harabasz_score
# from sklearn.metrics import adjusted_mutual_info_score,fowlkes_mallows_score,adjusted_rand_score,jaccard_score
# 
from validationindex.helpers import internal_validation_indexes,external_validation_indexes
# from validationindex.validationindex import dunn_fast
# 
from utils.Utils import Utils
from utils.constants import Constants

from scripts.declarations import ExperimentOutputRow
# from utils.Utils import


import warnings
warnings.filterwarnings("ignore")

def remove_string_columns(df):
    drop_columns = []
    for col in df.columns:
        try:
            _ = df[col].astype(float)
        except Exception as e:
            drop_columns.append(col)
    return df.drop(columns= drop_columns)

def remove_headers(df,rm):
    if(rm):
        df.columns = range(df.shape[1])
        return df
    else:
        return df
    
def remove_vector_class(df,index):
    vector_class = df[df.columns[index]]
    df           = remove_columns(df,[index])
    return (df,vector_class)
    
def remove_columns(df,columns):
    df.drop(columns = [df.columns[int(float(i))] for i in columns], inplace = True )
    return df

def generate_dataset_description_report(**kwargs):
    BASE_PATH            = kwargs.get("BASE_PATH","/test/datasets")
    filename             = kwargs.get("filename")
    extension            = kwargs.get("extension","txt")
    sink_path            = kwargs.get("sinkPath","/test/sink")
    write                = kwargs.get("write",False)
    full_path            = "{}/{}.{}".format(sink_path,filename,extension)
    dataset_descriptions = kwargs.get("dataset_descriptions",{})
    f                    = open(full_path,"w") if(write)  else None
    for index,(key,value) in enumerate(dataset_descriptions.items()):
        separator = value.separator
        le = preprocessing.LabelEncoder()
        if(separator):
            df_path         = "{}/{}".format(BASE_PATH,value.fullname)
            df               = pd.read_csv(df_path,sep = separator )
            df               = remove_headers(df,value.remove_headers)
            df,vector_class  = remove_vector_class(df,value.vector_class_index)
            df               = remove_string_columns(df)
            raw_vector_class = vector_class.values.flatten()[1:]
            le.fit(raw_vector_class)
            target           = le.transform(raw_vector_class)
            vector_class     = pd.DataFrame(
                {
                    "target":target
                }
            )
            df              = df.apply(pd.to_numeric,errors="coerce").fillna(0)
            x               = vector_class.value_counts()
            xx              = pd.DataFrame( {"CLASS":list(range(x.size)),"COUNT":x.values } )
            if(write):
                f.write(key+"\n")
                f.write("&"+str(df.shape[0])+"\n")
                f.write("&"+str(df.shape[1])+"\n")
                f.write("&"+str(len(x))+"\n")
                f.write("\n")
            df.to_csv("{}/{}.{}".format(sink_path,value.filename,"csv"),index=False,header=None)
            xx.to_csv("{}/{}.{}".format(sink_path,value.filename+"_counter","csv"),index=False)
            vector_class.to_csv("{}/{}.{}".format(sink_path,value.filename+"_target","csv"),index=False)
            print("PROCESSED_SUCCESSFULLY",value)
            print("_"*30)
    if(write):
        f.close()


def clustering(**kwargs):
    algorithm             = kwargs.get("algorithm","SKMEANS")
    m                     = kwargs.get("m",3)
    k                     = kwargs.get("k",2)
    plaintext_matrix      = kwargs.get("plaintext_matrix",np.array([]))
    experiment_output_row = kwargs.get("experiment_output_row",{})
    LIU                   = kwargs.get("LIU",None)
    L                     = kwargs.get("logger",None)
    result                = None

    L.debug("INIT_CLUSTERING algorithm={} k={} m={}".format(algorithm,k,m)) 
    

    if(algorithm =="SKMEANS"):
        result      = skmeans(
            k                = k,
            m                = m,
            plaintext_matrix = plaintext_matrix,
            LIU              = LIU,
            logger           = L
        )
    elif(algorithm == "KMEANS"):
        result = kmeans(
            k                = k,
            plaintext_matrix =  plaintext_matrix
        )
    elif(algorithm == "DBSKMEANS"):
        result = {}
    else:
        result = {}

    experiment_output_row.metadata                    = {"pred":result["labels_vector"] }
    experiment_output_row.n_iterations                = result.get("n_iterations",0)
    experiment_output_row.cipher_time                 = result.get("cipher_time",0.0)
    # experiment_output_row.udm_calculation             = result.get("udm_calculation",0.0)
    # experiment_output_row.udm_encryption              = result.get("udm_encryption",0.0)
    experiment_output_row.udm_time                    = result.get("udm_time",0.0)
    # experiment_output_row.clustering_time             = result.get("clustering_time",0.0)
    experiment_output_row.service_time                = result.get("service_time",0.0)
    experiment_output_row.response_time               = result.get("response_time",0.0)
    return experiment_output_row


def validation_indexes(**kwargs) -> ExperimentOutputRow:
        x                = kwargs.get("experiment_output_row")
        k                = kwargs.get("k")
        plaintext_matrix = kwargs.get("plaintext_matrix",np.array([]))
        pred             = kwargs.get("pred",x.metadata.get("pred",np.array([])) )
        target           = kwargs.get("target",np.array([]))  
        internal_vi    = internal_validation_indexes(
            plaintext_matrix = plaintext_matrix,
            target = target
        )
        external_vi    =  external_validation_indexes(
            target = target,
            pred   = pred, 
            k = k
        )
        x.silhouette_coefficient      = internal_vi.get("silhouette_coefficient",0.0)
        x.davies_bouldin_index        = internal_vi.get("davies_bouldin_index",0.0)
        x.calinski_harabaz_index      = internal_vi.get("calinski_harabaz_index",0.0)
        x.dunn_index                  = internal_vi.get("dunn_index",0.0)
        x.adjusted_mutual_information = external_vi.get("adjusted_mutual_information",0.0)
        x.fowlkes_mallows_index       = external_vi.get("fowlkes_mallows_index",0.0)
        x.adjusted_rand_index         = external_vi.get("adjusted_rand_index",0.0)
        x.jaccard_index               = external_vi.get("jaccard_index",0.0)
        return x




# 
# def generate_centroids(**kwargs):
#     k            = kwargs.get("k",3)
#     plain_matrix = kwargs.get("plain_matrix")
#     centroids    = []
#     for x in range(k):
#         centroids.append(plain_matrix[x])
#     columns = Utils.getShapeOfMatrix(plain_matrix)[1]

#     return np.array(centroids).reshape(k,columns)




def kmeans(**kwargs):
    startTime            = time()
    k                    = kwargs.get("k",2)
    plain_matrix         = kwargs.get("plaintext_matrix")
    # Utils.generate_centroids
    centroids            = Utils.generate_centroids(k = k,plain_matrix = plain_matrix)
    start_service_time   = time()
    kmeans               = KMeans(n_clusters=k,init=centroids)
    kmeans.fit(plain_matrix)
    end_service_time     = time()
    service_time = end_service_time - start_service_time
    response_time        = time() - startTime
    
    return {
         "labels_vector"     :kmeans.labels_,
         "n_iterations" :kmeans.n_iter_,
         "response_time"     :response_time,
         "service_time"      :service_time,
    }

def skmeans(**kwargs):
    response_time_start = time()
    plaintext_matrix = kwargs.get("plaintext_matrix")
    # print("plain_matrix",len(plaintext_matrix))
    m                = kwargs.get("m")
    k                = kwargs.get("k")
    LIU              = kwargs.get("LIU")
    L                = kwargs.get("logger",None)
    startTime        = time()
    # ___________________________________________________
    dow0             = DataOwner(
        m          = m,
        liu_scheme = LIU,
    )
    # ___________________________________________________
    # Cipher time
    cipher_start_time    = time()
    # print("PLAIN_MATRIX_SHAPE ",len(plaintext_matrix))
    outsource_data_stats = dow0.outsourcedDataAndStats(
        plaintext_matrix = plaintext_matrix,
    )
    # print("AAAAA ",outsource_data_stats.encrypted_matrix)
    UDM              = outsource_data_stats.UDM
    ciphertextMatrix = outsource_data_stats.encrypted_matrix
    # 
    # cipher_time = time() - cipher_start_time
    # START SERVICE TIME
    service_time_start_time = time()
    skmeans = SKMeans(
        ciphertext_matrix = ciphertextMatrix,
        UDM               = UDM,
        k                 = k,
        m                 = m,
        dataowner         = dow0,
        max_iterations    = 100,
        logger            = L
    )
    end_time       = time()
    # _______________________________________________________ 
    service_time   = end_time - service_time_start_time 
    response_time_end = time()
    response_time = response_time_end - response_time_start
    # response_time  = end_time - startTime
    # ________________________________________________________
    return {
         "labels_vector"     : np.array(skmeans.label_vector),
         "n_iterations" : skmeans.iteration_counter,
         "response_time"     : response_time,
         "service_time"      : service_time,
        #  _________________________________________________
         "udm_time"          : outsource_data_stats.udm_time,
         "cipher_time" : outsource_data_stats.encrypted_matrix_time
    }