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
from src.modules.SKMeans_deprecated import SKMeans
from src.modules.DBSKMeans_deprecated import DBSKMeans

from security.cryptosystem.liu import Liu
from security.cryptosystem.dataowner import DataOwner
from security.cryptosystem.FDHOpe import FDHOpe

from utils.Utils import Utils
from utils.constants import Constants

SECRETKEY = [[3.2,2.7,0],[9.1,3.1,1.5],[3.6,7.9,0]]
#R         = [0.22269171, 0.08139647, 0.01490155]
liu       = Liu(round = True)        
m         = 3
k         = 3
plaintext_matrix = pd.read_csv("C:/test/sources/taller1.csv" ,header=None).values.tolist()


class TestApp(unittest.TestCase):

    @unittest.skip("DEMO 1")
    def test_liu_encrypt(self):
        ciphertext_matrix = liu.encryptMatrix(
            plaintext_matrix  = plaintext_matrix,
            secret_key        = SECRETKEY,
            m                 = m
        )
        _plaintext_matrix = liu.decryptMatrix(
            ciphertext_matrix = ciphertext_matrix,
            secret_key        = SECRETKEY,
            m                 = m
        )
        print("PLAIN TEXT")
        print(np.array(plaintext_matrix))
        print("_"*20)

        print("CIPHER TEXT")
        print(np.array(ciphertext_matrix))
        print("_"*20)
        
        print("PLAIN TEXT")
        print(np.array(_plaintext_matrix))

    @unittest.skip("DEMO 2")
    def test_homorphic_propierties(self):
        
       

if __name__ == '__main__':
    unittest.main()
