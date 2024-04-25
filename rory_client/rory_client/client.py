import requests as R
import time as T
from option import Result,Ok,Err
from abc import ABC
from typing import List

class ClusteringResponse(ABC):
    def __init__(self, label_vector:List[int], iterations:int, worker_service_time:float, response_time:float, algorithm:str) -> None:
        self.label_vector = label_vector
        self.iterations = iterations
        self.worker_service_time = worker_service_time
        self.response_time = response_time
        self.algorithm = algorithm

class KmeansResponse(ClusteringResponse):
    def __init__(self,label_vector: List[int], iterations: int, worker_service_time: float, response_time: float, algorithm: str= "KMEANS") -> None:
        super().__init__(label_vector, iterations, worker_service_time, response_time, algorithm)


class RoryClient(object):
    def __init__(self, hostname:str="localhost",port:int = 9000):
        self.uri = "http://{}:{}".format(hostname,port)
        self.clustering_url = "{}/clustering".format(self.uri)
        self.kmeans_url = "{}/kmeans".format(self.clustering_url)

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