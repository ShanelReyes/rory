import requests as R
import time as T
from option import Result,Ok,Err
from abc import ABC
from typing import List,Tuple
from dataclasses import dataclass,field

# class ClusteringResponse(ABC):
#     def __init__(self, label_vector:List[int], iterations:int, worker_service_time:float, response_time:float, algorithm:str) -> None:
#         self.label_vector = label_vector
#         self.iterations   = iterations
#         self.worker_service_time = worker_service_time
#         self.response_time = response_time
#         self.algorithm = algorithm

@dataclass
class KnnTrainResponse:
    response_time:float
    algorithm:str

@dataclass
class KnnPredictResponse:
    label_vector:List[int]
    algorithm:str
    worker_id:str
    service_time_manager:float
    service_time_worker:float
    service_time_client:float
    service_time_predict:float

@dataclass
class SknnTrainResponse:
    response_time:str
    encrypted_model_shape:str
    encrypted_model_dtype:str
    algorithm:str


@dataclass
class KmeansResponse:
    label_vector:List[int]
    iterations:int
    algorithm:str
    worker_id:str
    service_time_manager:float
    service_time_worker:float
    service_time_client:float
    response_time_clustering:float


@dataclass
class NncResponse:
    label_vector:List[int]
    algorithm:str
    worker_id:str
    service_time_manager:float
    service_time_worker:float
    service_time_client:float
    response_time_clustering:float

class RoryClient(object):
    def __init__(self, hostname:str="localhost",port:int = 9000):
        self.uri                = "http://{}:{}".format(hostname,port)
        self.clustering_url     = "{}/clustering".format(self.uri)
        self.classification_url = f"{self.uri}/classification"
        self.kmeans_url         = f"{self.clustering_url}/kmeans"
        self.skmeans_url        = f"{self.clustering_url}/skmeans"
        self.dbskmeans_url      = f"{self.clustering_url}/dbskmeans"
        self.skmeans_pqc_url    = f"{self.clustering_url}/pqc/skmeans"
        self.dbskmeans_pqc_url  = f"{self.clustering_url}/pqc/dbskmeans"
        self.nnc_url            = f"{self.clustering_url}/nnc"
        self.dbsnnc_url         = f"{self.clustering_url}/dbsnnc"
        self.knn_url            = f"{self.classification_url}/knn"
        self.sknn_url           = f"{self.classification_url}/sknn"
        self.knn_pqc_url        = f"{self.classification_url}/pqc/knn"
        self.sknn_pqc_url       = f"{self.classification_url}/pqc/sknn"

    def kmeans(self, 
            plaintext_matrix_id:str,
            plaintext_matrix_filename:str, 
            k:int = 2, 
            extension:str = "npy"):
        """
        Sends a POST request to the /clustering/kmeans endpoint with the specified headers.

        Parameters:
            k (int): Number of clusters.
            matrix_filename (str): Name of the plaintext matrix file.
            matrix_id (str): Identifier of the matrix.
            extension (str): File extension (default is "npy").

        Returns:
            dict: A dictionary containing:
                - "label_vector" (list): The resulting label vector.
                - "iterations" (int): Number of iterations performed.
                - "algorithm" (str): Algorithm used.
                - "worker_id" (str): Identifier of the worker.
                - "service_time_manager" (float): Service time from the manager.
                - "service_time_worker" (float): Service time from the worker.
                - "service_time_client" (float): Service time from the client.
                - "response_time_clustering" (float): Clustering response time.
        """
        try:
            headers = {
                "Extension": extension,
                "K": str(k),
                "Plaintext-Matrix-Filename": plaintext_matrix_filename,
                "Plaintext-Matrix-Id": plaintext_matrix_id,
            }
            
            response = R.post(f"{self.kmeans_url}", headers = headers)
            response.raise_for_status()
            data = KmeansResponse(**response.json())

            return Ok(data)
        except Exception as e:
            return Err(e)        

    def skmeans(self, 
            plaintext_matrix_id: str, 
            plaintext_matrix_filename: str, 
            experiment_iteration:int = 0, 
            num_chunks:int = 2, 
            k: int = 2, 
            max_iterations:int = 5, 
            extension: str = "npy"
        ):
        """
        Sends a POST request to the /clustering/skmeans endpoint with the specified headers.

        Parameters:
            plaintext_matrix_id (str): Identifier of the matrix for the skmeans algorithm.
            plaintext_matrix_filename (str): Name of the plaintext matrix file.
            experiment_iteration (int): Id of current operation.
            num_chunks (int): Number of chunks in which the dataset is split.
            max_iterations (int): Max. number of iterations before stopping the process.
            k (int): Number of clusters (default is 2).
            extension (str): File extension (default is "npy").

        Returns:
            dict: A dictionary containing:
                - label_vector (list): The resulting label vector.
                - iterations (int): Number of iterations performed.
                - algorithm (str): Algorithm used.
                - worker_id (str): Identifier of the worker.
                - service_time_manager (float): Service time from the manager.
                - service_time_worker (float): Service time from the worker.
                - service_time_client (float): Service time from the client.
                - response_time_clustering (float): Clustering response time.
        """
        try:
            headers = {
                "Experiment-Iteration": str(experiment_iteration),
                "Extension": extension,
                "K": str(k),
                "Max-Iterations": str(max_iterations),
                "Num-Chunks": str(num_chunks),
                "Plaintext-Matrix-Filename": plaintext_matrix_filename,
                "Plaintext-Matrix-Id": plaintext_matrix_id,
            }
            
            response = R.post(f"{self.skmeans_url}", headers=headers)
            response.raise_for_status()
            data = KmeansResponse(**response.json())
            return Ok(data)
        except Exception as e:
            return Err(e)

    def dbskmeans(self, 
            plaintext_matrix_id: str, 
            plaintext_matrix_filename: str, 
            k: int = 2, 
            experiment_iteration:int = 0, 
            num_chunks:int = 2, 
            max_iterations:int = 5, 
            extension: str = "npy", 
            sens: str = "0.00000000001"
        ):
        """
        Sends a POST request to the /clustering/dbskmeans endpoint with the specified headers.

        Parameters:
            plaintext_matrix_id (str): Identifier of the matrix for the dbskmeans algorithm.
            plaintext_matrix_filename (str): Name of the plaintext matrix file.
            k (int): Number of clusters (default is 2).
            experiment_iteration (int): Id of current operation.
            num_chunks (int): Number of chunks in which the dataset is split.
            max_iterations (int): Max. number of iterations before stopping the process.
            extension (str): File extension (default is "npy").
            sens (str): Sensitivity value (default is "0.00000000001").

        Returns:
            KmeansResponse: A dataclass instance containing:
                - label_vector (list): The resulting label vector.
                - iterations (int): Number of iterations performed.
                - algorithm (str): Algorithm used.
                - worker_id (str): Identifier of the worker.
                - service_time_manager (float): Service time from the manager.
                - service_time_worker (float): Service time from the worker.
                - service_time_client (float): Service time from the client.
                - response_time_clustering (float): Clustering response time.
        """
        try:
            headers = {
                "Experiment-Iteration": str(experiment_iteration),
                "Extension": extension,
                "K": str(k),
                "Max-Iterations": str(max_iterations),
                "Num-Chunks": str(num_chunks),
                "Plaintext-Matrix-Filename": plaintext_matrix_filename,
                "Plaintext-Matrix-Id": plaintext_matrix_id,
                "Sens": str(sens)
            }
            
            response = R.post(f"{self.dbskmeans_url}", headers=headers)
            response.raise_for_status()
            data = KmeansResponse(**response.json())
            return Ok(data)
        except Exception as e:
            return Err(e)

    def skmeans_pqc(self, 
            plaintext_matrix_id: str, 
            plaintext_matrix_filename: str, 
            k: int = 2, 
            experiment_iteration:int = 0, 
            num_chunks:int = 2, 
            max_iterations:int = 5, 
            extension: str = "npy"
        ):
        """
        Sends a POST request to the /clustering/pqc/skmeans endpoint with the specified headers.

        Parameters:
            plaintext_matrix_id (str): Identifier of the matrix for the skmeans_pqc algorithm.
            plaintext_matrix_filename (str): Name of the plaintext matrix file.
            k (int): Number of clusters (default is 2).
            experiment_iteration (int): Id of current operation.
            num_chunks (int): Number of chunks in which the dataset is split.
            max_iterations (int): Max. number of iterations before stopping the process.
            extension (str): File extension (default is "npy").

        Returns:
            KmeansResponse: A dataclass instance containing:
                - label_vector (list): The resulting label vector.
                - iterations (int): Number of iterations performed.
                - algorithm (str): Algorithm used.
                - worker_id (str): Identifier of the worker.
                - service_time_manager (float): Service time from the manager.
                - service_time_worker (float): Service time from the worker.
                - service_time_client (float): Service time from the client.
                - response_time_clustering (float): Clustering response time.
        """
        try:
            headers = {
                "Experiment-Iteration": str(experiment_iteration),
                "Extension": extension,
                "K": str(k),
                "Max-Iterations": str(max_iterations),
                "Num-Chunks": str(num_chunks),
                "Plaintext-Matrix-Filename": plaintext_matrix_filename,
                "Plaintext-Matrix-Id": plaintext_matrix_id,
            }
            response = R.post(f"{self.skmeans_pqc_url}", headers=headers)
            response.raise_for_status()
            data = KmeansResponse(**response.json())
            return Ok(data)
        except Exception as e:
            return Err(e)
        
    def dbskmeans_pqc(self, 
            plaintext_matrix_id: str, 
            plaintext_matrix_filename: str, 
            k: int = 2, 
            experiment_iteration:int = 0, 
            num_chunks:int = 2, 
            max_iterations:int = 5, 
            extension: str = "npy", 
            sens: str = "0.00000000001"
        ):
        """
        Sends a POST request to the /clustering/pqc/dbskmeans endpoint with the specified headers.

        Parameters:
            plaintext_matrix_id (str): Identifier of the matrix for the skmeans_pqc algorithm.
            plaintext_matrix_filename (str): Name of the plaintext matrix file.
            k (int): Number of clusters (default is 2).
            experiment_iteration (int): Id of current operation.
            num_chunks (int): Number of chunks in which the dataset is split.
            max_iterations (int): Max. number of iterations before stopping the process.
            extension (str): File extension (default is "npy").
            sens (str): Sensitivity value (default is "0.00000000001").

        Returns:
            KmeansResponse: A dataclass instance containing:
                - label_vector (list): The resulting label vector.
                - iterations (int): Number of iterations performed.
                - algorithm (str): Algorithm used.
                - worker_id (str): Identifier of the worker.
                - service_time_manager (float): Service time from the manager.
                - service_time_worker (float): Service time from the worker.
                - service_time_client (float): Service time from the client.
                - response_time_clustering (float): Clustering response time.
        """
        try:
            headers = {
                "Experiment-Iteration": str(experiment_iteration),
                "Extension": extension,
                "K": str(k),
                "Max-Iterations": str(max_iterations),
                "Num-Chunks": str(num_chunks),
                "Plaintext-Matrix-Filename": plaintext_matrix_filename,
                "Plaintext-Matrix-Id": plaintext_matrix_id,
                "Sens": str(sens)
            }  
            response = R.post(f"{self.dbskmeans_pqc_url}", headers=headers)
            response.raise_for_status()
            data = KmeansResponse(**response.json())
            return Ok(data)
        except Exception as e:
            return Err(e)

    def nnc(self, 
        plaintext_matrix_id: str, 
        plaintext_matrix_filename: str, 
        extension: str = "npy", 
        threshold: float = 1.4
    ):
        """
        Sends a POST request to the /clustering/nnc endpoint with the specified headers.

        Parameters:
            plaintext_matrix_id (str): Identifier of the matrix for the nnc algorithm.
            plaintext_matrix_filename (str): Name of the plaintext matrix file.
            extension (str): File extension (default is "npy").
            threshold (str): Threshold value (default is "1.4").

        Returns:
            KmeansResponse: A dataclass instance containing:
                - label_vector (list): The resulting label vector.
                - iterations (int): Number of iterations performed.
                - algorithm (str): Algorithm used.
                - worker_id (str): Identifier of the worker.
                - service_time_manager (float): Service time from the manager.
                - service_time_worker (float): Service time from the worker.
                - service_time_client (float): Service time from the client.
                - response_time_clustering (float): Clustering response time.
        """
        try:
            headers = {
                "Extension": extension,
                "Plaintext-Matrix-Filename": plaintext_matrix_filename,
                "Plaintext-Matrix-Id": plaintext_matrix_id,
                "Threshold": str(threshold),
            }
            response = R.post(f"{self.nnc_url}", headers=headers)
            response.raise_for_status()
            data = NncResponse(**response.json())
            return Ok(data)
        except Exception as e:
            return Err(e)
        
    def dbsnnc(self, 
        plaintext_matrix_id: str, 
        plaintext_matrix_filename: str, 
        extension: str = "npy", 
        threshold: float = 1.4,
        num_chunks:int = 2, 
        sens: str = "0.00000000001"
        ):
        """
        Sends a POST request to the /clustering/nnc endpoint with the specified headers.

        Parameters:
            plaintext_matrix_id (str): Identifier of the matrix for the nnc algorithm.
            plaintext_matrix_filename (str): Name of the plaintext matrix file.
            extension (str): File extension (default is "npy").
            threshold (str): Threshold value (default is "1.4").
            num_chunks (int): Number of chunks in which the dataset is split.
            sens (str): Sensitivity value (default is "0.00000000001").

        Returns:
            KmeansResponse: A dataclass instance containing:
                - label_vector (list): The resulting label vector.
                - iterations (int): Number of iterations performed.
                - algorithm (str): Algorithm used.
                - worker_id (str): Identifier of the worker.
                - service_time_manager (float): Service time from the manager.
                - service_time_worker (float): Service time from the worker.
                - service_time_client (float): Service time from the client.
                - response_time_clustering (float): Clustering response time.
        """
        try:
            headers = {
                "Extension": extension,
                "Plaintext-Matrix-Filename": plaintext_matrix_filename,
                "Plaintext-Matrix-Id": plaintext_matrix_id,
                "Threshold": str(threshold),
                "Num-Chunks": str(num_chunks),
                "Sens": str(sens)
            }
            response = R.post(f"{self.nnc_url}", headers=headers)
            response.raise_for_status()
            data = NncResponse(**response.json())
            return Ok(data)
        except Exception as e:
            return Err(e)

#### CLASSIFICATION

    def knn_train(self,
        model_id:str, 
        model_filename:str, 
        model_labels_filename:str,
        extension:str="npy"         
        ):
        try:
            headers = {
                "Model-Filename": model_filename,
                "Model-Id":model_id,
                "Model-Labels-Filename": model_labels_filename,
                "Extension":extension
            }
            response = R.post(f"{self.knn_url}/train", headers=headers)
            response.raise_for_status()
            data = KnnTrainResponse(**response.json())
            return Ok(data)
        except Exception as e:
            return Err(e)

    def knn_predict(self,
        model_id:str, 
        model_filename:str, 
        model_labels_filename:str,
        record_test_id:str,
        record_test_filename:str,
        extension:str="npy"         
        ):
        try:
            headers = {
                "Model-Filename": model_filename,
                "Model-Id":model_id,
                "Model-Labels-Filename": model_labels_filename,
                "Records-Test-Id":record_test_id,
                "Records-Test-Filename":record_test_filename,
                "Extension":extension
            }
            response = R.post(f"{self.knn_url}/predict", headers=headers)
            response.raise_for_status()
            data = KnnPredictResponse(**response.json())
            return Ok(data)
        except Exception as e:
            return Err(e)  
        
    def knn(self,
        model_id:str, 
        model_filename:str, 
        model_labels_filename:str,
        record_test_id:str,
        record_test_filename:str,
        extension:str="npy" 
        )->Result[KnnPredictResponse, Exception]:
        try:
            knn_train_result = self.knn_train(
                model_id= model_id,
                model_filename=model_filename,
                model_labels_filename=model_labels_filename,
                extension=extension
            )
            if knn_train_result.is_err:
                return knn_train_result
            train_response = knn_train_result.unwrap()
            # train_response.res
            predict_result = self.knn_predict(
                model_id=model_id,
                model_filename=model_filename,
                model_labels_filename=model_labels_filename, 
                record_test_id=record_test_id,
                record_test_filename=record_test_filename,
                extension=extension
            )
            return Ok(predict_result)
        except Exception as e:
            return Err(e)

    def sknn_train(self,
        model_id:str, 
        model_filename:str,
        model_labels_filename:str,
        num_chunks:int=2,
        extension:str="npy"
    ):
        try:
            headers = {
                "Model-Filename": model_filename,
                "Model-Id":model_id,
                "Model-Labels-Filename": model_labels_filename,
                "Num-Chunks":str(num_chunks),
                "Extension":extension
            }
            response = R.post(f"{self.sknn_url}/train", headers=headers)
            response.raise_for_status()
            data = SknnTrainResponse(**response.json())
            return Ok(data)
        except Exception as e:
            return Err(e)

    def sknn_predict(self,
        model_id:str, 
        model_filename:str, 
        model_labels_filename:str,
        record_test_id:str,
        record_test_filename:str,
        encrypted_model_shape:str,
        encrypted_model_dtype:str = "float64",
        num_chunks:int=2,
        extension:str="npy",
        ):
        try:
            headers = {
                "Model-Filename": model_filename,
                "Model-Id":model_id,
                "Model-Labels-Filename": model_labels_filename,
                "Records-Test-Id":record_test_id,
                "Records-Test-Filename":record_test_filename,
                "Extension":extension,
                "Num-Chunks":str(num_chunks),
                "Encrypted-Model-Shape": encrypted_model_shape,
                "Encrypted-Model-Dtype":encrypted_model_dtype
            }
            response = R.post(f"{self.sknn_url}/predict", headers=headers)
            response.raise_for_status()
            data = KnnPredictResponse(**response.json())
            return Ok(data)
        except Exception as e:
            return Err(e) 

    def sknn(self,
        model_id:str, 
        model_filename:str, 
        model_labels_filename:str,
        record_test_id:str,
        record_test_filename:str,       
        # encrypted_model_shape:str,
        # encrypted_model_dtype:str = "float64",
        num_chunks:int=2,
        extension:str="npy",
        ):
        try:
            sknn_train_result = self.sknn_train(
                model_id              = model_id,
                model_filename        = model_filename,
                model_labels_filename = model_labels_filename,
                extension             = extension,
                num_chunks            = num_chunks
            ) 
            if sknn_train_result.is_err:
                return sknn_train_result
            train_response = sknn_train_result.unwrap()
            # train_response_data = train_response.
            predict_result = self.sknn_predict(
                model_id              = model_id,
                model_filename        = model_filename,
                model_labels_filename = model_labels_filename, 
                record_test_id        = record_test_id,
                record_test_filename  = record_test_filename,
                extension             = extension,
                num_chunks            = num_chunks,
                encrypted_model_shape = train_response.encrypted_model_shape,
                encrypted_model_dtype = train_response.encrypted_model_dtype
            )
            return Ok(predict_result)
        except Exception as e:
            return Err(e)    

    def sknn_pqc_train(self,
        model_id:str, 
        model_filename:str,
        model_labels_filename:str,
        num_chunks:int=2,
        extension:str="npy"
    ):
        try:
            headers = {
                "Model-Filename": model_filename,
                "Model-Id":model_id,
                "Model-Labels-Filename": model_labels_filename,
                "Num-Chunks":str(num_chunks),
                "Extension":extension
            }
            response = R.post(f"{self.sknn_pqc_url}/train", headers=headers)
            response.raise_for_status()
            data = SknnTrainResponse(**response.json())
            return Ok(data)
        except Exception as e:
            return Err(e)

    def sknn_pqc_predict(self,
        model_id:str, 
        model_filename:str,
        model_labels_filename:str, 
        record_test_id:str,
        record_test_filename:str,
        encrypted_model_shape: str,
        num_chunks:int=2,
        extension:str="npy",
        encrypted_model_dtype:str ="float64"
        ):
        try:
            headers={
                "Extension": extension,
                "Model-Id":model_id,
                "Model-Filename": model_filename,
                "Model-Labels-Filename": model_labels_filename,
                "Records-Test-Id":record_test_id,
                "Records-Test-Filename":record_test_filename,
                "Num-Chunks":str(num_chunks),
                "Encrypted-Model-Shape": encrypted_model_shape,
                "Encrypted-Model-Dtype":encrypted_model_dtype
            }
            response = R.post(f"{self.sknn_pqc_url}/predict", headers=headers)
            response.raise_for_status()
            data = KnnPredictResponse(**response.json())
            return Ok(data)
        except Exception as e:
            return Err(e)

    def sknn_pqc(self,
        model_id:str, 
        model_filename:str,
        model_labels_filename:str, 
        record_test_id:str,
        record_test_filename:str,
        encrypted_model_shape: str,
        num_chunks:int=2,
        extension:str="npy",
        encrypted_model_dtype:str ="float64"
        ):
        try:
            sknn_train_result = self.sknn_pqc_train(
                model_id              = model_id,
                model_filename        = model_filename,
                model_labels_filename = model_labels_filename,
                extension             = extension,
                num_chunks            = num_chunks
            ) 
            if sknn_train_result.is_err:
                return sknn_train_result
            train_response = sknn_train_result.unwrap()

            predict_result = self.sknn_pqc_predict(
                model_id              = model_id,
                model_filename        = model_filename,
                model_labels_filename = model_labels_filename, 
                record_test_id        = record_test_id,
                record_test_filename  = record_test_filename,
                extension             = extension,
                num_chunks            = num_chunks,
                encrypted_model_shape = encrypted_model_shape,
                encrypted_model_dtype = encrypted_model_dtype
            )
            return Ok(predict_result)
        except Exception as e:
            return Err(e)

if __name__ == "__main__":
    rc = RoryClient(hostname="localhost",port=3000)
