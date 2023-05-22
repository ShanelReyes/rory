# * This block of code MUST be executed first.  
# _______________________________________________________
import os
import sys
from pathlib import Path
import time
from sklearn.metrics import silhouette_score

path_root      = Path(__file__).parent.absolute()
(path_root, _) = os.path.split(path_root)
sys.path.append(str(path_root))
# ______________________________________________________
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor,wait
import unittest

from src.modules.distributedskmeans import DistributedSkMeans
from clustering.secure.skmeans import SKMeans
from clustering.secure.dbskmeans import Dbskmeans
from clustering.secure.dbsnnc import Dbsnnc
# from src.modules.SKMeans_deprecated import SKMeans
# from src.modules.DBSKMeans_deprecated import DBSKMeans

from security.cryptosystem.liu import Liu
from security.cryptosystem.dataowner import DataOwner
from security.cryptosystem.FDHOpe import Fdhope

from utils.Utils import Utils
from utils.constants import Constants

SECRETKEY = [[3.2,2.7,0],[9.1,3.1,1.5],[3.6,7.9,0]]
liu        = Liu(round = True)        
m          = 3
k          = 3
sk         = liu.secretKey( m = m )
#R_random_numbers = [0.5134455324953191, 0.040472390165429584, 0.25352312974495494]
plaintext  = int(123125234)
plaintext_vector = [1,2,3,4,5]
#plaintext_matrix = np.random.rand(100,100)
plaintext_matrix = [
    [0.73,8.84],
    [49.93,34.44],
    [0.57,65.04],
    [62.15,32.29],
    [59.47,36.04]
]
plaintext_matrix_01 = pd.DataFrame({}) 

# plaintext_matrix_01 = Data.load_csv(path = "data/lung-cancer1.data")
# plaintext_matrix_01 = pd.read_csv("/test/source/dataset-0.csv",header=None)
# plaintext_matrix_01 = pd.read_csv("H:/My Drive/Doctorado-Privado/datasets/datasets/iris.csv" ,header=None).values.tolist()
plaintext_matrix_01 = np.array([])

# _____________________________________ 
import os

def empty_clusters(**kwargs):
    k = kwargs.get("k",1)
    xs = np.zeros((k,1))
    xs[:] = np.nan
    return xs 


class TestApp(unittest.TestCase):

    def test_dbsnnc(self):
        dow0 = DataOwner(
            m = m,
            liu_scheme = liu,
        )
        outsource_data_result, udm = dow0.outsourceDataDbsnnc(
            plaintext_matrix = plaintext_matrix
        )
        nnc = Dbsnnc(
            ciphertext_matrix = plaintext_matrix
        )
        #print(outsource_data_result, udm)

    @unittest.skip("")
    def test_dbskmeans_final(self):
        dow0 = DataOwner(
            m = m,
            liu_scheme = liu,
        )
        outsource_data_result = dow0.outsourceDataDBS(
            plaintext_matrix = plaintext_matrix
        )

        dbsskmeans = Dbskmeans(
            ciphertext_matrix = outsource_data_result.encrypted_matrix,
            UDM               = outsource_data_result.UDM,
            k                 = k,
            m                 = m,
            dataowner         = dow0,
            messageIntervals  = outsource_data_result.messageIntervals,
            cypherIntervals   = outsource_data_result.cypherIntervals,
            sens              = 0.01 
        )

        print(dbsskmeans.label_vector)


    @unittest.skip("")
    def test_sskmeans(self):    
        ptm = pd.read_csv("/test/sink/food_category.csv").to_numpy()
        dow0 = DataOwner(
            m = m,
            liu_scheme = liu,
        )

        outsource_data_result = dow0.outsourcedDataVectorizeAndStats(
            plaintext_matrix = ptm,
        )
        print("UDM_TIME",outsource_data_result.udm_time)
        print("ENCRYPTION_TIME",outsource_data_result.encrypted_matrix_time)
        
    @unittest.skip("")
    def test_optimize(self):
        k = 3
        es = empty_clusters(k = k)
        print(es.shape)
        c1 = np.array([ 0,np.nan,np.nan]).reshape(k,1)
        c2 = np.array([np.nan,1,np.nan]).reshape(k,1)
        print(c1.shape)
        es = np.hstack((es,c1))
        es = np.hstack((es,c2)) 
        print(es.shape)

    @unittest.skip("")
    def test_optimize_encryption(self):
        vectorize_times = []
        list_times      = []
        experiment_iterations = 1
        # food-category / 
        rows = 60000
        cols = 39
        with ThreadPoolExecutor(max_workers=4) as executor :
            futures = []
            for i in range(0,experiment_iterations):
                future = None
                M               = np.random.randint(0, 10, (rows,cols))
                encryption_res  = liu.vectorizeEncryptMatrix(plaintext_matrix =  M, sk = sk, m = m)
                print(encryption_res)
                decryption_res = liu.vectorizeDecryptMatrix(ciphertext_matrix =encryption_res.matrix, sk= sk, m=m )
                print(decryption_res)
            wait(futures)
        df = pd.DataFrame({"VT":vectorize_times,"LT":list_times})
        df.to_csv("/test/sink/times_{}_{}.csv".format(rows,cols))

    @unittest.skip("")
    def test_experiments(self):
        SINK_PATH         = "/test/sink"
        _dataset_fullnames = os.listdir(SINK_PATH)
        only_csv          = lambda x: x.split(".")[1]=="csv"
        only_datasets     = lambda x: len(x.split("_counter"))==1
        dataset_fullnames = list(filter(only_csv , _dataset_fullnames ))
        dataset_fullnames = list(filter(only_datasets,dataset_fullnames))
        dataset_fullnames.sort()
        counters_fullnames = list(set(_dataset_fullnames).difference(set(dataset_fullnames)))
        counters_fullnames = list(filter(only_csv,counters_fullnames))
        counters_fullnames.sort()
        # 
        for dataset_fullname,counter_fullname in zip(dataset_fullnames,counters_fullnames):
            dataset_full_path = "{}/{}".format(SINK_PATH,dataset_fullname)
            counter_full_path = "{}/{}".format(SINK_PATH,counter_fullname)
            
            dataset_df        = pd.read_csv(dataset_full_path) 
            counter_df        = pd.read_csv(counter_full_path)
            print(dataset_df.shape[0],counter_df["COUNT"].sum())

        print(counters_fullnames)

    @unittest.skip("Todavia no sirve el DBSkmeans")
    def test_dbskmeans(self):
        startTime = time.time()
        dow0 = DataOwner(
            m = m,
            liu_scheme = liu,
        )

        ciphertextMatrix,UDM, messageIntervals, cypherIntervals = dow0.outsourceDataDBS(
            plaintext_matrix = plaintext_matrix_01,
        )

        dbsskmeans = DBSKMeans(
            ciphertext_matrix = ciphertextMatrix,
            UDM               = UDM,
            k                 = k,
            m                 = m,
            dataowner         = dow0,
            messageIntervals  = messageIntervals,
            cypherIntervals   = cypherIntervals,
            sens              = 0.01
            
        )
        responseTime = time.time() - startTime
       
        plainClusters = dow0.verify(
            cipher_clusters = dbsskmeans.C
        )
        counterElementsInCluster = list(map(lambda xs: len(xs),plainClusters))
        print(counterElementsInCluster)
        
    @unittest.skip("")
    def test_skmeansComplete(self):
        startTime = time.time()
                
        dow0 = DataOwner(
            m = m,
            liu_scheme = liu,
        )
        ciphertextMatrix,UDM = dow0.outsourcedData(
            plaintext_matrix = plaintext_matrix_01,
        )
        
        skmeans = SKMeans(
            ciphertext_matrix = ciphertextMatrix,
            UDM               = UDM,
            k                 = k,
            m                 = m,
            dataowner         = dow0   
        )
        responseTime = time.time() - startTime
        
        plainClusters = dow0.verify(
            cipher_clusters = skmeans.C
            #round = True
        )
        counterElementsInCluster = list(map(lambda xs: len(xs),plainClusters))
        score_silhouette = silhouette_score(plaintext_matrix_01, skmeans.label_vector, metric='euclidean')
        print("_"*20)
        print("Registros dentro de cada cluster",counterElementsInCluster)
        print("Tiempo de ejecucion", responseTime)
        print("Numero de iteraciones", skmeans.iteration_counter)
        print("score_silhouette",score_silhouette)

    @unittest.skip("")
    def test_Dataowner(self):
        dow0 = DataOwner(
            m = m,
            liu_scheme = liu,
        )

        dow0.setSecretKey(SECRETKEY)

        ciphertextMatrix,UDM = dow0.outsourcedData(
            plaintext_matrix = plaintext_matrix,
        )        
        print('V', ciphertextMatrix)
        print("_"*20)
        print("_"*20)
    
    @unittest.skip("")
    def test_run1(self):
        dow0 = DataOwner(
            m = m,
            liu_scheme = liu,
        )
        ciphertextMatrix,UDM = dow0.outsourcedData(
            plaintext_matrix = plaintext_matrix_01
        )
        distributedSkMeans = DistributedSkMeans()
        status             = Constants.ClusteringStatus.START
        S1,Cent_i,Cent_j   = distributedSkMeans.run_1(
            
            status           = status,
            k                = k,
            m                = m,
            ciphertextMatrix = np.array(ciphertextMatrix),
            UDM              = np.array(UDM),
            Cent_j           = None
        )

    @unittest.skip("")
    def test_skmeans_1(self):
        for i in range(100):
            dow0 = DataOwner(
                m = m,
                liu_scheme = liu,
            )
            D1,U = dow0.outsourcedData(
                plaintext_matrix = plaintext_matrix_01
            )
            print("Dataset-{}".format(i),Utils.getShapeOfMatrix(D1))
  
    @unittest.skip("")
    def test_dataOwner(self):
        # Create Data owner
        dow0 = DataOwner(
            m = m,
            liu_scheme = liu,
        )

        D1,U = dow0.outsourcedData(
            plaintext_matrix = plaintext_matrix_01
        )
        skmeans = SKMeans(
            ciphertext_matrix = D1,
            UDM               = U,
            k                 = k,
            m                 = m,
            dataowner         = dow0
        )
        C = skmeans.run()
        plaintext_clusters = dow0.verify(
            cipher_clusters = C
        )
        print("_"*60)
        print(Utils.prettyprint(plaintext_clusters))
        print("_"*50)

    @unittest.skip("")
    def test_create_UDM(self):
        udm = Utils.create_UDM(plaintext_matrix = plaintext_matrix_01)
        print(udm)
        
    @unittest.skip("")
    def test_Liu_encrypt_matrix(self):
        ciphertext_matrix = liu.encryptMatrix(
            plaintext_matrix  = plaintext_matrix,
            secret_key        = sk,
            m                 = m
        )
        _plaintext_matrix = liu.decryptMatrix(
            ciphertext_matrix = ciphertext_matrix,
            secret_key        = sk,
            m                 = m
        )
        print(np.array(ciphertext_matrix).shape)
        print(np.array(_plaintext_matrix))

    @unittest.skip("")
    def test_Liu_encrypt_vector(self):
        ciphertext_vector = liu.encryptVector(
            plaintext_vector  = plaintext_vector,
            secret_key = sk,
            m          = m
        )
        _plaintext_vector = liu.decryptVector(
            ciphertext_vector = ciphertext_vector,
            secret_key        = sk,
            m                 = m
        )
        print("CIPHERTEXT_VECTOR")
        print(ciphertext_vector)
        print("_"*100)
        print("PLAINTEXT_VECTOR")
        print(_plaintext_vector)

    @unittest.skip("")
    def test_Liu_encrypt_sclar(self):
        ciphertext = liu.encryptScalar(
            plaintext  = plaintext,
            secret_key = sk,
            m          = m
        )
        _plaintext = liu.decryptScalar(
            ciphertext = ciphertext,
            secret_key = sk,
            m          = m
        )
        print("PLAINTEXT = {}\nCIPHER_TEXT = {}\n_PLAINTEXT = {}".format(plaintext,ciphertext,_plaintext))
        self.assertEqual(plaintext,_plaintext)


if __name__ == '__main__':
    unittest.main()