import requests as R
import time as T
from option import Result,Ok,Err
from abc import ABC
from typing import List,Tuple
from dataclasses import dataclass,field
class ClusteringResponse(ABC):
    def __init__(self, label_vector:List[int], iterations:int, worker_service_time:float, response_time:float, algorithm:str) -> None:
        self.label_vector = label_vector
        self.iterations = iterations
        self.worker_service_time = worker_service_time
        self.response_time = response_time
        self.algorithm = algorithm

@dataclass 
class SknnPQCResponse:
    algorithm:str
    response_time:float = field(default=-1)
    encrypted_model_dtype: str = field(default="float64")
    encrypted_model_shape:str = field(default="(0,0)")
    _encrypted_model_shape:Tuple[int,int] = field(init=False)
    def __post_init__(self):
        self._encrypted_model_shape = eval(self.encrypted_model_shape)


class KmeansResponse(ClusteringResponse):
    def __init__(self,label_vector: List[int], iterations: int, worker_service_time: float, response_time: float, algorithm: str= "KMEANS") -> None:
        super().__init__(label_vector, iterations, worker_service_time, response_time, algorithm)



class RoryClient(object):
    def __init__(self, hostname:str="localhost",port:int = 9000):
        self.uri = "http://{}:{}".format(hostname,port)
        self.clustering_url = "{}/clustering".format(self.uri)
        self.classification_url= f"{self.uri}/classification"
        self.kmeans_url = f"{self.clustering_url}/kmeans"
        self.knn_pqc_url=f"{self.classification_url}/pqc/sknn"


    def sknn_pqc_train(self,
        id:str, 
        model_filename:str,
        model_labels_filename:str, 
        record_tests_filename:str,
        num_chunks:int=2,extension:str="npy"
    ):
        try:
            model_id = f"{id}model"
            record_test_id = f"{id}recordtest"
            headers = {
                "Model-Filename": model_filename,
                "Model-Id":model_id,
                "Model-Labels-Filename": model_labels_filename,
                "Num-Chunks":str(num_chunks)
            }
            response = R.post(f"{self.knn_pqc_url}/train", headers=headers)
            response.raise_for_status()
            data = SknnPQCResponse(**response.json())
            return Ok(data)
        except Exception as e:
            return Err(e)

    def sknn_pqc_predict(self,
                id:str, 
                model_filename:str,
                model_labels_filename:str, 
                record_tests_filename:str,
                encrypted_model_shape: str,
                num_chunks:int=2,
                extension:str="npy",
                encrypted_model_dtype:str ="float64"
    ):
        try:
            model_id = f"{id}model"
            record_test_id = f"{id}recordtest"
        
            predict_response = R.post(
                f"{self.knn_pqc_url}/predict",
                headers={
                    "Extension": extension,
                    "Model-Id":model_id,
                    "Model-Filename": model_filename,
                    "Model-Labels-Filename": model_labels_filename,
                    "Records-Test-Id":record_test_id,
                    "Records-Test-Filename":record_tests_filename,
                    "Num-Chunks":str(num_chunks),
                    "Encrypted-Model-Shape": encrypted_model_shape,
                    "Encrypted-Model-Dtype":encrypted_model_dtype

                }
            )
            predict_response.raise_for_status()
            predict_data = predict_response.json()
            return Ok(predict_data)
        except Exception as e:
            return Err(e)

    def sknn_pqc(self,
                id:str, 
                model_filename:str,
                model_labels_filename:str, 
                record_tests_filename:str,
                num_chunks:int=2,extension:str="npy"
    ):
        try:
            model_id = f"{id}model"
            record_test_id = f"{id}recordtest"
            headers = {
                "Model-Filename": model_filename,
                "Model-Id":model_id,
                "Model-Labels-Filename": model_labels_filename,
                "Num-Chunks":str(num_chunks)
            }
            response = R.post(f"{self.knn_pqc_url}/train", headers=headers)
            response.raise_for_status()
            data = SknnPQCResponse(**response.json())
            predict_response = R.post(
                f"{self.knn_pqc_url}/predict",
                headers={
                    "Extension": extension,
                    "Model-Id":model_id,
                    "Model-Filename": model_filename,
                    "Model-Labels-Filename": model_labels_filename,
                    "Records-Test-Id":record_test_id,
                    "Records-Test-Filename":record_tests_filename,
                    "Num-Chunks":str(num_chunks),
                    "Encrypted-Model-Shape": data.encrypted_model_shape,
                    "Encrypted-Model-Dtype":data.encrypted_model_dtype

                }
            )
            predict_response.raise_for_status()
            predict_data = predict_response.json()
            print("PREDICT_DATA", predict_data)
            # print("DATA", data)

            return Ok(data)
        except Exception as e:
            return Err(e)

    def kmeans(self,plaintext_matrix_id:str, plaintext_matrix_filename:str,k:int = 2, extension:str = "npy"):
        try:
            headers = {
                "K":str(k),
                "Plaintext-Matrix-Id": plaintext_matrix_id,
                "Plaintext-Matrix-Filename": plaintext_matrix_filename,
                "Extension": extension
            }
            response = R.post(self.kmeans_url, headers= headers)
            response.raise_for_status()
            data = KmeansResponse(**response.json())
            return Ok(data)
        except Exception as e:
            return Err(e)


if __name__ == "__main__":
    rc = RoryClient(hostname="localhost",port=3000)
    
    result = rc.kmeans(
        k=3,
        plaintext_matrix_id="testing",
        plaintext_matrix_filename= "audit_data_data",
        extension="npy"
    )
    if result.is_ok:
        print(result.unwrap().__dict__)