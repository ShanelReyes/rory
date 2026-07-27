import requests as R
import time as T
from option import Result,Ok,Err
from roryclient.models import *

class RoryOrchestrator:
    def __init__(self, hostname:str="localhost",port:int = 6000,timeout:int= 120):
        self.uri                = "http://{}:{}".format(hostname,port)
        self.orchestrator_url     = "{}/orchestration".format(self.uri)
        self.timeout = timeout

    def orchestrate(self,policy:Policy):
        try:
            res = R.post(f"{self.orchestrator_url}",json=Policy.model_dump())
            res.raise_for_status()
        except Exception as e:
            return Err(e)
    def mark_task_as_completed(self, task_id: str) -> Result[dict, Exception]:
        """POST /orchestration/tasks/<task_id>/completed"""
        try:
            url = f"{self.orchestrator_url}/tasks/{task_id}/completed"
            res = R.post(url, timeout=self.timeout)
            res.raise_for_status()
            return Ok(res.json())
        except Exception as e:
            return Err(e)
    def register_completed_task(self, task:Dict[str,Any]) -> Result[dict, Exception]:
        """POST /orchestration/tasks/<task_id>/completed"""
        try:
            url = f"{self.orchestrator_url}/tasks/registration-completed"
            res = R.post(url, json=task,timeout=self.timeout)
            res.raise_for_status()
            return Ok(res.json())
        except Exception as e:
            return Err(e)
        
    
    def get_marked_as_completed_tasks(self) -> Result[dict, Exception]:
        """POST /orchestration/tasks/<task_id>/completed"""
        try:
            url = f"{self.orchestrator_url}/tasks/marked-completed"
            res = R.get(url, timeout=self.timeout)
            res.raise_for_status()
            return Ok(res.json())
        except Exception as e:
            return Err(e)
            # return {"error": str(e)}

    def get_tasks(self) -> Result[dict, Exception]:
        """GET /orchestration/tasks"""
        try:
            res = R.get(f"{self.orchestrator_url}/tasks", timeout=self.timeout)
            res.raise_for_status()
            return Ok(res.json())
        except Exception as e:
            return Err(e)
            # return {"error": str(e)}

    def get_task_details(self) -> Result[dict, Exception]:
        """GET /orchestration/tasks/q"""
        try:
            res = R.get(f"{self.orchestrator_url}/tasks/q", timeout=self.timeout)
            res.raise_for_status()
            return Ok(res.json())
        except Exception as e:
            return Err(e)

    def get_completed_tasks(self) -> Result[dict, Exception]:
        """GET /orchestration/tasks/completed"""
        try:
            res = R.get(f"{self.orchestrator_url}/tasks/completed", timeout=self.timeout)
            res.raise_for_status()
            return Ok(res.json())
        except Exception as e:
            return Err(e)

    def get_completed_task_by_id(self, task_id: str) -> Result[dict, Exception]:
        """GET /orchestration/tasks/<task_id>/completed"""
        try:
            res = R.get(f"{self.orchestrator_url}/tasks/{task_id}/completed", timeout=self.timeout)
            res.raise_for_status()
            return Ok(res.json())
        except Exception as e:
            return Err(e)

class RoryWorker(object):
    def __init__(self, hostname:str="localhost",port:int = 9000,timeout:int= 120):
        self.uri                = "http://{}:{}".format(hostname,port)
        self.distributed_url    = f"{self.uri}/distributed"

        self.timeout = timeout
    def stats(self):
        try:
            res = R.get(f"{self.distributed_url}/stats")
            res.raise_for_status()
            response_json = res.json()
            return Ok(response_json)

        except Exception as e:
            return Err(e)
    def encrypt(self,dto:EncryptDTO):
        try:
            res=  R.post(f"{self.distributed_url}/encrypt", json=dto.model_dump())
            res.raise_for_status()
            response_json = res.json()
            print(response_json)
            return Ok(response_json)
        except Exception as e:
            return Err(e)
    def kmeans(self, dto: KMeansDTO) -> Result[dict, Exception]:
        """POST /distributed/kmeans"""
        try:
            res = R.post(f"{self.distributed_url}/kmeans", json=dto.model_dump(), timeout=self.timeout)
            res.raise_for_status()
            return Ok(res.json())
        except Exception as e:
            return Err(e)

    def get_tasks(self) -> Result[dict, Exception]:
        """GET /distributed/tasks"""
        try:
            res = R.get(f"{self.distributed_url}/tasks", timeout=self.timeout)
            res.raise_for_status()
            return Ok(res.json())
        except Exception as e:
            return Err(e)

    def get_task_details(self) -> Result[dict, Exception]:
        """GET /distributed/tasks/q"""
        try:
            res = R.get(f"{self.distributed_url}/tasks/q", timeout=self.timeout)
            res.raise_for_status()
            return Ok(res.json())
        except Exception as e:
            return Err(e)

    def get_completed_tasks(self) -> Result[dict, Exception]:
        """GET /distributed/tasks/completed"""
        try:
            res = R.get(f"{self.distributed_url}/tasks/completed", timeout=self.timeout)
            res.raise_for_status()
            return Ok(res.json())
        except Exception as e:
            return Err(e)

    def get_completed_task_by_id(self, task_id: str) -> Result[dict, Exception]:
        """GET /distributed/tasks/<task_id>/completed"""
        try:
            url = f"{self.distributed_url}/tasks/{task_id}/completed"
            res = R.get(url, timeout=self.timeout)
            res.raise_for_status()
            return Ok(res.json())
        except Exception as e:
            return Err(e)       



class RoryClient(object):
    """The interface for interacting with the Rory Privacy-Preserving Data Mining platform.

    RoryClient provides a unified set of methods to access clustering, classification, 
    and distributed processing services. It simplifies communication with the backend by 
    handling endpoint orchestration, HTTP header management, and response validation through 
    predefined data models.

    The client supports multiple security paradigms, including:
    - Standard algorithms (K-Means, KNN, NNC).
    - Privacy-Preserving algorithms (Secure K-Means, Secure KNN).
    - Bouble-Blind variants (dbskmeans, dbsnnc).
    - Post-Quantum Cryptography (PQC) enabled methods for quantum-resistant security.

    Attributes:
        uri (str): The base connection string (e.g., http://localhost:9000).
        timeout (int): The maximum time in seconds to wait for a server response.
        clustering_url (str): Base endpoint for clustering services.
        classification_url (str): Base endpoint for classification services.
        distributed_url (str): Base endpoint for elastic task management and segmentation.
    """
    def __init__(self, hostname:str="localhost",port:int = 9000,timeout:int= 120):
        """Initializes the RoryClient with server connection details.

        Args:
            hostname (str, optional): The network address of the Rory server. Defaults to "localhost".
            port (int, optional): The port where the Rory service is listening. Defaults to 9000.
            timeout (int, optional): Connection and request timeout in seconds. Defaults to 120.
        """
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
        self.distributed_url    = f"{self.uri}/distributed"
        self.machine_learning_url = f"{self.uri}/machine-learning"
        self.timeout = timeout

    def segment(self,dto:SegmentDTO):
        """Initiates the segmentation of a dataset for distributed processing.

        Sends a POST request to the `/segment` endpoint of the distributed architecture. 
        This method is responsible for splitting a large dataset (identified by its bucket 
        and ball IDs) into smaller, manageable chunks based on the parameters specified 
        in the Data Transfer Object. This is a crucial pre-processing step for parallel 
        and elastic data mining operations.

        Args:
            dto (SegmentDTO): The Data Transfer Object containing the configuration required 
                to segment the dataset (such as bucket_id, ball_id, filename, and chunk size parameters).

        Returns:
            Result[SegementResponseDTO, Exception]: An `Ok` containing the parsed 
            `SegementResponseDTO` object on success, with attributes such as:
                - task_id (str): The unique identifier assigned to the segmentation task.
                - message (Any): Status message or additional task information.
                - ball_id (str): The identifier of the processed data object.
                - bucket_id (str): The identifier of the storage bucket.
            Returns an `Err` containing the raised Exception if the HTTP request, connection, 
            or data validation fails.
        """
        try:
            res=  R.post(f"{self.distributed_url}/segment", json=dto.model_dump())
            res.raise_for_status()
            response_json = res.json()
            return Ok(SegementResponseDTO.model_validate(response_json))
        except Exception as e:
            return Err(e)
        
    def get_tasks(self) -> Result[dict, Exception]:
        """Retrieves the current status and list of tasks from the distributed architecture.

        Sends a GET request to the `/tasks` endpoint of the distributed module. This 
        method is useful for monitoring the progress, state, or results of asynchronous 
        operations occurring within the elastic platform, such as dataset segmentation 
        or distributed processing jobs.

        Returns:
            Result[dict, Exception]: An `Ok` containing a dictionary with the task details 
            and statuses returned by the server on success.
            Returns an `Err` containing the raised Exception if the HTTP request, connection, 
            or JSON parsing fails.
        """
        try:
            res = R.get(f"{self.distributed_url}/tasks", timeout=self.timeout)
            res.raise_for_status()
            return Ok(res.json())
        except Exception as e:
            return Err(e)

    def get_task_details(self) -> Result[dict, Exception]:
        """Retrieves detailed information and status queues for distributed tasks.

        Sends a GET request to the `/tasks/q` endpoint of the distributed module. 
        Unlike the general `get_tasks` method which provides a top-level overview, 
        this method fetches more granular, in-depth details regarding the task queue, 
        execution states, or worker assignments within the elastic architecture.

        Returns:
            Result[dict, Exception]: An `Ok` containing a dictionary with the detailed 
            task queue information returned by the server on success.
            Returns an `Err` containing the raised Exception if the HTTP request, connection, 
            or JSON parsing fails.
        """
        try:
            res = R.get(f"{self.distributed_url}/tasks/q", timeout=self.timeout)
            res.raise_for_status()
            return Ok(res.json())
        except Exception as e:
            return Err(e)

    def get_completed_tasks(self) -> Result[dict, Exception]:
        """Retrieves a historical record of all successfully completed distributed tasks.

        Sends a GET request to the `/tasks/completed` endpoint of the distributed module. 
        This method allows users to fetch the final results, execution metrics, and 
        identifiers for tasks that have finished their lifecycle within the elastic 
        architecture, distinguishing them from pending or active processes.

        Returns:
            Result[dict, Exception]: An `Ok` containing a dictionary representing the 
            history of completed tasks and their associated metadata.
            Returns an `Err` containing the raised Exception if the HTTP request, 
            connection, or JSON parsing fails.
        """
        try:
            res = R.get(f"{self.distributed_url}/tasks/completed", timeout=self.timeout)
            res.raise_for_status()
            return Ok(res.json())
        except Exception as e:
            return Err(e)

    def get_completed_task_by_id(self, task_id: str) -> Result[dict, Exception]:
        """Retrieves the details and results of a specific completed distributed task.

        Sends a GET request to the `/tasks/<task_id>/completed` endpoint. This method 
        is used to fetch the final execution status, performance metrics, and 
        output data for a specific task identified by its unique ID, ensuring 
        traceability within the distributed architecture.

        Args:
            task_id (str): The unique identifier of the completed task to retrieve.

        Returns:
            Result[dict, Exception]: An `Ok` containing a dictionary with the specific 
            task's results and metadata on success.
            Returns an `Err` containing the raised Exception if the HTTP request, 
            connection, or task lookup fails.
        """
        try:
            url = f"{self.distributed_url}/tasks/{task_id}/completed"
            res = R.get(url, timeout=self.timeout)
            res.raise_for_status()
            return Ok(res.json())
        except Exception as e:
            return Err(e)

#### CLUSTERING

    def kmeans(self, 
            plaintext_matrix_id:str,
            plaintext_matrix_filename:str, 
            k:int = 2, 
            extension:str = "npy",
            num_chunks:int =2
            ):
        """Executes the standard K-Means clustering algorithm on the remote platform.

        Sends a POST request to the `/clustering/kmeans` endpoint using the specified 
        configuration parameters passed as HTTP headers.

        Args:
            plaintext_matrix_id (str): The unique identifier of the plaintext matrix.
            plaintext_matrix_filename (str): The filename of the plaintext matrix dataset.
            k (int, optional): The number of clusters to form. Defaults to 2.
            extension (str, optional): The file extension of the matrix dataset. Defaults to "npy".
            num_chunks (int, optional): The number of chunks to split the task into for the distributed architecture. Defaults to 2.

        Returns:
            Result[KmeansResponse, Exception]: An `Ok` containing the parsed `KmeansResponse` object on success, with attributes such as:
                - label_vector (list): The resulting label vector.
                - iterations (int): Number of iterations performed.
                - algorithm (str): Algorithm used.
                - worker_id (str): Identifier of the worker.
                - service_time_manager (float): Service time from the manager.
                - service_time_worker (float): Service time from the worker.
                - service_time_client (float): Service time from the client.
                - response_time_clustering (float): Clustering response time.
            Returns an `Err` containing the raised Exception if the HTTP request or validation fails.
        """

        try:
            headers = {
                "Extension": extension,
                "K": str(k),
                "Plaintext-Matrix-Filename": plaintext_matrix_filename,
                "Plaintext-Matrix-Id": plaintext_matrix_id,
                "Num-Chunks":str(num_chunks)
            }
            
            response = R.post(f"{self.kmeans_url}", headers = headers,timeout=self.timeout)
            response.raise_for_status()
            data = KmeansResponse.model_validate(response.json())

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
        """Executes the Secure K-Means (skmeans) clustering algorithm on the remote platform.

        Sends a POST request to the `/clustering/skmeans` endpoint. Task parameters and 
        matrix identifiers are securely transmitted via HTTP headers to interface with 
        the Privacy-Preserving Data Mining (PPDM) architecture.

        Args:
            plaintext_matrix_id (str): The unique identifier of the plaintext matrix.
            plaintext_matrix_filename (str): The filename of the plaintext matrix dataset.
            experiment_iteration (int, optional): The ID or iteration number of the current operation, useful for tracking research experiments. Defaults to 0.
            num_chunks (int, optional): The number of chunks into which the dataset is split for distributed elastic processing. Defaults to 2.
            k (int, optional): The number of clusters to form. Defaults to 2.
            max_iterations (int, optional): The maximum number of iterations allowed before the algorithm stops. Defaults to 5.
            extension (str, optional): The file extension of the matrix dataset. Defaults to "npy".

        Returns:
            Result[KmeansResponse, Exception]: An `Ok` containing the parsed `KmeansResponse` object on success, with attributes such as:
                - label_vector (list): The resulting label vector mapping data points to clusters.
                - iterations (int): The actual number of iterations performed.
                - algorithm (str): The specific algorithm used (e.g., 'skmeans').
                - worker_id (str): The unique identifier of the worker node that processed the task.
                - service_time_manager (float): The time taken by the manager node (in seconds).
                - service_time_worker (float): The time taken by the worker node (in seconds).
                - service_time_client (float): The time taken at the client level (in seconds).
                - response_time_clustering (float): The overall clustering response time.
            Returns an `Err` containing the raised Exception if the HTTP request, connection, or data validation fails.
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
            
            response = R.post(f"{self.skmeans_url}", headers=headers,timeout=self.timeout)
            response.raise_for_status()
            data = KmeansResponse.model_validate(response.json())
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
            sens: float = 0.00000000001
        ):
        """Executes the Double-Blind Secure K-Means (dbskmeans) clustering algorithm.

        Sends a POST request to the `/clustering/dbskmeans` endpoint. Task parameters, matrix 
        identifiers, and a sensitivity value (`sens`) are securely transmitted via HTTP headers 
        to interface with the privacy-preserving backend, ensuring robust data protection during 
        the mining process.

        Args:
            plaintext_matrix_id (str): The unique identifier of the plaintext matrix.
            plaintext_matrix_filename (str): The filename of the plaintext matrix dataset.
            k (int, optional): The number of clusters to form. Defaults to 2.
            experiment_iteration (int, optional): The ID or iteration number of the current operation, useful for tracking research experiments. Defaults to 0.
            num_chunks (int, optional): The number of chunks into which the dataset is split for distributed elastic processing. Defaults to 2.
            max_iterations (int, optional): The maximum number of iterations allowed before the algorithm stops. Defaults to 5.
            extension (str, optional): The file extension of the matrix dataset. Defaults to "npy".
            sens (float, optional): The sensitivity value used for calibrating privacy-preserving operations (e.g., differential privacy noise). Defaults to 0.00000000001.

        Returns:
            Result[KmeansResponse, Exception]: An `Ok` containing the parsed `KmeansResponse` object on success, with attributes such as:
                - label_vector (list): The resulting label vector mapping data points to clusters.
                - iterations (int): The actual number of iterations performed.
                - algorithm (str): The specific algorithm used (e.g., 'dbskmeans').
                - worker_id (str): The unique identifier of the worker node that processed the task.
                - service_time_manager (float): The time taken by the manager node (in seconds).
                - service_time_worker (float): The time taken by the worker node (in seconds).
                - service_time_client (float): The time taken at the client level (in seconds).
                - response_time_clustering (float): The overall clustering response time.
            Returns an `Err` containing the raised Exception if the HTTP request, connection, or data validation fails.
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
            
            response = R.post(f"{self.dbskmeans_url}", headers=headers,timeout=self.timeout)
            response.raise_for_status()
            data = KmeansResponse.model_validate(response.json())
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
            response = R.post(f"{self.skmeans_pqc_url}", headers=headers,timeout=self.timeout)
            response.raise_for_status()
            data = KmeansResponse.model_validate(response.json())
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
        """Executes the Post-Quantum Cryptography (PQC) enabled Secure K-Means clustering algorithm.

        Sends a POST request to the `/clustering/pqc/skmeans` endpoint. This method leverages 
        quantum-resistant cryptographic primitives to ensure data privacy during the clustering 
        process. Task parameters and matrix identifiers are securely transmitted via HTTP headers.

        Args:
            plaintext_matrix_id (str): The unique identifier of the plaintext matrix.
            plaintext_matrix_filename (str): The filename of the plaintext matrix dataset.
            k (int, optional): The number of clusters to form. Defaults to 2.
            experiment_iteration (int, optional): The ID or iteration number of the current operation, useful for tracking research experiments. Defaults to 0.
            num_chunks (int, optional): The number of chunks into which the dataset is split for distributed elastic processing. Defaults to 2.
            max_iterations (int, optional): The maximum number of iterations allowed before the algorithm stops. Defaults to 5.
            extension (str, optional): The file extension of the matrix dataset. Defaults to "npy".

        Returns:
            Result[KmeansResponse, Exception]: An `Ok` containing the parsed `KmeansResponse` object on success, with attributes such as:
                - label_vector (list): The resulting label vector mapping data points to clusters.
                - iterations (int): The actual number of iterations performed.
                - algorithm (str): The specific algorithm used (e.g., 'skmeans_pqc').
                - worker_id (str): The unique identifier of the worker node that processed the task.
                - service_time_manager (float): The time taken by the manager node (in seconds).
                - service_time_worker (float): The time taken by the worker node (in seconds).
                - service_time_client (float): The time taken at the client level (in seconds).
                - response_time_clustering (float): The overall clustering response time.
            Returns an `Err` containing the raised Exception if the HTTP request, connection, or data validation fails.
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
            response = R.post(f"{self.dbskmeans_pqc_url}", headers=headers,timeout=self.timeout)
            response.raise_for_status()
            data = KmeansResponse.model_validate(response.json())
            return Ok(data)
        except Exception as e:
            return Err(e)

    def nnc(self, 
        plaintext_matrix_id: str, 
        plaintext_matrix_filename: str, 
        extension: str = "npy", 
        threshold: float = 1.4
    ):
        """Executes the Nearest Neighbor Clustering (NNC) algorithm on the remote platform.

        Sends a POST request to the `/clustering/nnc` endpoint. Task parameters, including 
        matrix identifiers and the distance threshold, are transmitted via HTTP headers.

        Args:
            plaintext_matrix_id (str): The unique identifier of the plaintext matrix.
            plaintext_matrix_filename (str): The filename of the plaintext matrix dataset.
            extension (str, optional): The file extension of the matrix dataset. Defaults to "npy".
            threshold (float, optional): The distance threshold value used to determine cluster boundaries. Defaults to 1.4.

        Returns:
            Result[NncResponse, Exception]: An `Ok` containing the parsed `NncResponse` object on success, with attributes such as:
                - label_vector (list): The resulting label vector mapping data points to clusters.
                - iterations (int): The number of iterations performed.
                - algorithm (str): The specific algorithm used (e.g., 'nnc').
                - worker_id (str): The unique identifier of the worker node that processed the task.
                - service_time_manager (float): The time taken by the manager node (in seconds).
                - service_time_worker (float): The time taken by the worker node (in seconds).
                - service_time_client (float): The time taken at the client level (in seconds).
                - response_time_clustering (float): The overall clustering response time.
            Returns an `Err` containing the raised Exception if the HTTP request, connection, or data validation fails.
        """
        try:
            headers = {
                "Extension": extension,
                "Plaintext-Matrix-Filename": plaintext_matrix_filename,
                "Plaintext-Matrix-Id": plaintext_matrix_id,
                "Threshold": str(threshold),
            }
            response = R.post(f"{self.nnc_url}", headers=headers,timeout=self.timeout)
            response.raise_for_status()
            data = NncResponse.model_validate(response.json())
            return Ok(data)
        except Exception as e:
            return Err(e)
        
    def dbsnnc(self, 
        plaintext_matrix_id: str, 
        plaintext_matrix_filename: str, 
        extension: str = "npy", 
        threshold: float = 1.4,
        num_chunks:int = 2, 
        sens:float= 0.00000000001
        ):
        """Executes the Double-Blind Secure Nearest Neighbor Clustering (dbsnnc) algorithm.

        Sends a POST request to the `/clustering/dbsnnc` endpoint. Task parameters, including 
        matrix identifiers, distance threshold, dataset chunks, and a sensitivity value (`sens`), 
        are securely transmitted via HTTP headers to interface with the privacy-preserving backend.

        Args:
            plaintext_matrix_id (str): The unique identifier of the plaintext matrix.
            plaintext_matrix_filename (str): The filename of the plaintext matrix dataset.
            extension (str, optional): The file extension of the matrix dataset. Defaults to "npy".
            threshold (float, optional): The distance threshold value used to determine cluster boundaries. Defaults to 1.4.
            num_chunks (int, optional): The number of chunks into which the dataset is split for distributed elastic processing. Defaults to 2.
            sens (float, optional): The sensitivity value used for calibrating differential privacy operations. Defaults to 0.00000000001.

        Returns:
            Result[NncResponse, Exception]: An `Ok` containing the parsed `NncResponse` object on success, with attributes such as:
                - label_vector (list): The resulting label vector mapping data points to clusters.
                - iterations (int): The number of iterations performed.
                - algorithm (str): The specific algorithm used (e.g., 'dbsnnc').
                - worker_id (str): The unique identifier of the worker node that processed the task.
                - service_time_manager (float): The time taken by the manager node (in seconds).
                - service_time_worker (float): The time taken by the worker node (in seconds).
                - service_time_client (float): The time taken at the client level (in seconds).
                - response_time_clustering (float): The overall clustering response time.
            Returns an `Err` containing the raised Exception if the HTTP request, connection, or data validation fails.
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
            response = R.post(f"{self.dbsnnc_url}", headers=headers,timeout=self.timeout)
            response.raise_for_status()
            data = NncResponse.model_validate(response.json())
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
        """Initiates the training phase for the K-Nearest Neighbors (KNN) classification model.

        Sends a POST request to the `/classification/knn/train` endpoint. The method registers 
        the training dataset and its corresponding labels on the remote platform, preparing 
        the model for future prediction tasks. Parameters are securely transmitted via HTTP headers.

        Args:
            model_id (str): The unique identifier to assign to the trained model.
            model_filename (str): The filename of the dataset containing the training features.
            model_labels_filename (str): The filename containing the corresponding training labels.
            extension (str, optional): The file extension of the dataset files. Defaults to "npy".

        Returns:
            Result[KnnTrainResponse, Exception]: An `Ok` containing the parsed `KnnTrainResponse` object on success, with attributes such as:
                - response_time (float): The time taken to process the training request.
                - algorithm (str): The specific algorithm used.
                - model_labels_shape (List[int]): The dimensional shape of the loaded model labels.
            Returns an `Err` containing the raised Exception if the HTTP request, connection, or data validation fails.
        """
        try:
            headers = {
                "Model-Filename": model_filename,
                "Model-Id":model_id,
                "Model-Labels-Filename": model_labels_filename,
                "Extension":extension
            }
            response = R.post(f"{self.knn_url}/train", headers=headers,timeout=self.timeout)
            response.raise_for_status()
            data = KnnTrainResponse.model_validate(response.json())
            return Ok(data)
        except Exception as e:
            return Err(e)

    def knn_predict(self,
        model_id:str, 
        model_filename:str, 
        model_labels_filename:str,
        record_test_id:str,
        record_test_filename:str,
        model_labels_shape:str,
        extension:str="npy"         
        ):
        """Executes the prediction phase for the K-Nearest Neighbors (KNN) classification algorithm.

        Sends a POST request to the `/classification/knn/predict` endpoint. This method 
        evaluates a new set of test records against a previously registered training dataset 
        to determine their classifications. All configuration parameters and file identifiers 
        are transmitted via HTTP headers.

        Args:
            model_id (str): The unique identifier of the trained model to use.
            model_filename (str): The filename of the dataset containing the training features.
            model_labels_filename (str): The filename containing the corresponding training labels.
            record_test_id (str): The unique identifier for the set of test records to classify.
            record_test_filename (str): The filename of the dataset containing the test records.
            model_labels_shape (str): The dimensional shape of the model labels (e.g., as a string representation of a list/tuple).
            extension (str, optional): The file extension of the dataset files. Defaults to "npy".

        Returns:
            Result[KnnPredictResponse, Exception]: An `Ok` containing the parsed `KnnPredictResponse` object on success, with attributes such as:
                - label_vector (List[int]): The resulting predictions/labels for the test records.
                - algorithm (str): The specific algorithm used for prediction.
                - worker_id (str): The unique identifier of the worker node that processed the task.
                - service_time_manager (float): The time taken by the manager node (in seconds).
                - service_time_worker (float): The time taken by the worker node (in seconds).
                - service_time_client (float): The time taken at the client level (in seconds).
                - service_time_predict (float): The specific time taken to compute the predictions.
            Returns an `Err` containing the raised Exception if the HTTP request, connection, or data validation fails.
        """
        try:
            headers = {
                "Model-Filename": model_filename,
                "Model-Id":model_id,
                "Model-Labels-Filename": model_labels_filename,
                "Records-Test-Id":record_test_id,
                "Records-Test-Filename":record_test_filename,
                "Extension":extension,
                "Model-Labels-Shape":model_labels_shape
            }
            response = R.post(f"{self.knn_url}/predict", headers=headers,timeout=self.timeout)
            response.raise_for_status()
            data = KnnPredictResponse.model_validate(response.json())
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
        )->Result[KnnResponse, Exception]:
        """Executes the complete K-Nearest Neighbors (KNN) classification workflow.

        This method acts as a high-level wrapper that sequentially orchestrates both the training 
        and prediction phases. It first calls `knn_train` to register the training dataset and 
        labels on the platform. Upon success, it automatically extracts the model's label shape 
        and invokes `knn_predict` to classify the test records. Finally, it aggregates the 
        prediction results and timing metrics from both steps into a single, unified response.

        Args:
            model_id (str): The unique identifier to assign to the model.
            model_filename (str): The filename of the dataset containing the training features.
            model_labels_filename (str): The filename containing the corresponding training labels.
            record_test_id (str): The unique identifier for the set of test records to classify.
            record_test_filename (str): The filename of the dataset containing the test records.
            extension (str, optional): The file extension of the dataset files. Defaults to "npy".

        Returns:
            Result[KnnResponse, Exception]: An `Ok` containing the parsed `KnnResponse` object on success, with attributes such as:
                - algorithm (str): The algorithm name (e.g., 'KNN').
                - label_vector (List[int]): The resulting predictions/labels for the test records.
                - worker_id (str): The unique identifier of the worker node that processed the prediction task.
                - service_time_manager (float): The time taken by the manager node (in seconds).
                - service_time_worker (float): The time taken by the worker node (in seconds).
                - service_time_client (float): The total time recorded at the client level (in seconds).
                - service_time_predict (float): The specific time taken to compute the predictions.
                - service_time_train (float): The specific time taken to complete the training phase.
            Returns an `Err` containing the raised Exception if either the training or the prediction phase fails.
        """
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
                extension=extension,
                model_labels_shape=str(tuple(train_response.model_labels_shape))
            )

            if predict_result.is_err:
                return predict_result
            predict_response = predict_result.unwrap()

            return Ok(KnnResponse.model_validate(
                {

                "service_time_client":predict_response.service_time_client,
                "algorithm":"KNN",
                "label_vector":predict_response.label_vector,
                "service_time_manager":predict_response.service_time_manager,
                "service_time_predict":predict_response.service_time_predict,
                "service_time_worker":predict_response.service_time_worker,
                "worker_id":predict_response.worker_id,
                "service_time_train":train_response.response_time
                }
            )
            )
        except Exception as e:
            return Err(e)

    def sknn_train(self,
        model_id:str, 
        model_filename:str,
        model_labels_filename:str,
        num_chunks:int=2,
        extension:str="npy"
    ):
        """Initiates the training phase for the Secure K-Nearest Neighbors (SKNN) classification model.

        Sends a POST request to the secure `/classification/sknn/train` endpoint. This method 
        registers the training dataset and its corresponding labels on the remote platform. 
        Unlike the standard KNN, this secure version supports data partitioning (`num_chunks`) 
        to facilitate distributed, privacy-preserving processing across the elastic architecture.
        Parameters are securely transmitted via HTTP headers.

        Args:
            model_id (str): The unique identifier to assign to the trained secure model.
            model_filename (str): The filename of the dataset containing the training features.
            model_labels_filename (str): The filename containing the corresponding training labels.
            num_chunks (int, optional): The number of chunks into which the dataset is split for distributed elastic processing. Defaults to 2.
            extension (str, optional): The file extension of the dataset files. Defaults to "npy".

        Returns:
            Result[SknnTrainResponse, Exception]: An `Ok` containing the parsed `SknnTrainResponse` object on success, which includes the training metadata, label shape, and execution time.
            Returns an `Err` containing the raised Exception if the HTTP request, connection, or data validation fails.
        """
        try:
            headers = {
                "Model-Filename": model_filename,
                "Model-Id":model_id,
                "Model-Labels-Filename": model_labels_filename,
                "Num-Chunks":str(num_chunks),
                "Extension":extension
            }
            response = R.post(f"{self.sknn_url}/train", headers=headers,timeout=self.timeout)
            response.raise_for_status()
            data = SknnTrainResponse.model_validate(response.json())
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
        model_labels_shape:str,
        encrypted_model_dtype:str = "float64",
        num_chunks:int=2,
        extension:str="npy",
        ):
        """Executes the prediction phase for the Secure K-Nearest Neighbors (SKNN) algorithm.

        Sends a POST request to the secure `/classification/sknn/predict` endpoint. This method 
        evaluates a new set of test records against a previously trained, encrypted dataset. 
        It requires detailed metadata regarding the shape and data type of 
        the encrypted model to ensure the distributed elastic workers can properly reconstruct 
        and process the secure matrices. All parameters are transmitted via HTTP headers.

        Args:
            model_id (str): The unique identifier of the trained secure model.
            model_filename (str): The filename of the dataset containing the encrypted training features.
            model_labels_filename (str): The filename containing the corresponding training labels.
            record_test_id (str): The unique identifier for the set of test records to classify.
            record_test_filename (str): The filename of the dataset containing the test records.
            encrypted_model_shape (str): The dimensional shape of the encrypted model matrix (e.g., string representation of a tuple).
            model_labels_shape (str): The dimensional shape of the model labels.
            encrypted_model_dtype (str, optional): The data type of the encrypted model matrix. Defaults to "float64".
            num_chunks (int, optional): The number of chunks into which the dataset is split for distributed elastic processing. Defaults to 2.
            extension (str, optional): The file extension of the dataset files. Defaults to "npy".

        Returns:
            Result[KnnPredictResponse, Exception]: An `Ok` containing the parsed `KnnPredictResponse` object on success, with attributes such as:
                - label_vector (List[int]): The resulting predictions/labels for the test records.
                - algorithm (str): The specific algorithm used (e.g., 'SKNN').
                - worker_id (str): The unique identifier of the worker node that processed the task.
                - service_time_manager (float): The time taken by the manager node (in seconds).
                - service_time_worker (float): The time taken by the worker node (in seconds).
                - service_time_client (float): The time taken at the client level (in seconds).
                - service_time_predict (float): The specific time taken to compute the secure predictions.
            Returns an `Err` containing the raised Exception if the HTTP request, connection, or data validation fails.
        """
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
                "Encrypted-Model-Dtype":encrypted_model_dtype,
                "Model-Labels-Shape": model_labels_shape
            }
            response = R.post(f"{self.sknn_url}/predict", headers=headers,timeout=self.timeout)
            response.raise_for_status()
            data = KnnPredictResponse.model_validate(response.json())
            return Ok(data)
        except Exception as e:
            return Err(e) 

    def sknn(self,
        model_id:str, 
        model_filename:str, 
        model_labels_filename:str,
        record_test_id:str,
        record_test_filename:str,       
        num_chunks:int=2,
        extension:str="npy",
        )->Result[KnnResponse,Exception]:
        """Executes the complete Secure K-Nearest Neighbors (SKNN) classification workflow.

        This method acts as a high-level wrapper that sequentially orchestrates both the privacy-preserving 
        training and prediction phases. It first calls `sknn_train` to securely register, partition, and 
        encrypt the training dataset on the platform. Upon success, it automatically extracts the encrypted 
        model's metadata (shape, data type, and label shape) and invokes `sknn_predict` to securely classify 
        the test records. Finally, it aggregates the prediction results and timing metrics from both the 
        training and prediction steps into a single, unified response.

        Args:
            model_id (str): The unique identifier to assign to the trained secure model.
            model_filename (str): The filename of the dataset containing the training features.
            model_labels_filename (str): The filename containing the corresponding training labels.
            record_test_id (str): The unique identifier for the set of test records to classify.
            record_test_filename (str): The filename of the dataset containing the test records.
            num_chunks (int, optional): The number of chunks into which the datasets are split for distributed elastic processing. Defaults to 2.
            extension (str, optional): The file extension of the dataset files. Defaults to "npy".

        Returns:
            Result[KnnResponse, Exception]: An `Ok` containing the parsed `KnnResponse` object on success, with attributes such as:
                - algorithm (str): The algorithm name (e.g., 'SKNN').
                - label_vector (List[int]): The resulting predictions/labels for the test records.
                - worker_id (str): The unique identifier of the worker node that processed the prediction task.
                - service_time_manager (float): The time taken by the manager node (in seconds).
                - service_time_worker (float): The time taken by the worker node (in seconds).
                - service_time_client (float): The total time recorded at the client level (in seconds).
                - service_time_predict (float): The specific time taken to compute the secure predictions.
                - service_time_train (float): The specific time taken to complete the secure training phase.
            Returns an `Err` containing the raised Exception if either the secure training or the prediction phase fails.
        """
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
                encrypted_model_dtype = train_response.encrypted_model_dtype,
                model_labels_shape= str(tuple(train_response.model_labels_shape))
            )

            if predict_result.is_err:
                return predict_result
            predict_response = predict_result.unwrap()

            return Ok(KnnResponse.model_validate(
                {
                    "service_time_client":predict_response.service_time_client,
                    "algorithm":"SKNN",
                    "label_vector":predict_response.label_vector,
                    "service_time_manager":predict_response.service_time_manager,
                    "service_time_predict":predict_response.service_time_predict,
                    "service_time_worker":predict_response.service_time_worker,
                    "worker_id":predict_response.worker_id,
                    "service_time_train":train_response.response_time

                }
            ))
            # return predict_result
        except Exception as e:
            return Err(e)    

    def sknn_pqc_train(self,
        model_id:str, 
        model_filename:str,
        model_labels_filename:str,
        num_chunks:int=2,
        extension:str="npy"
    ):
        """Initiates the training phase for the Post-Quantum Cryptography (PQC) enabled Secure K-Nearest Neighbors model.

        Sends a POST request to the secure PQC `/classification/sknn_pqc/train` endpoint. 
        This method registers the training dataset and its corresponding labels on the remote platform. 
        It leverages post-quantum cryptographic primitives to ensure the dataset remains secure against 
        quantum-level threats. The data is partitioned (`num_chunks`) to facilitate distributed, 
        privacy-preserving processing. Parameters are securely transmitted via HTTP headers.

        Args:
            model_id (str): The unique identifier to assign to the trained PQC secure model.
            model_filename (str): The filename of the dataset containing the training features.
            model_labels_filename (str): The filename containing the corresponding training labels.
            num_chunks (int, optional): The number of chunks into which the dataset is split for distributed elastic processing. Defaults to 2.
            extension (str, optional): The file extension of the dataset files. Defaults to "npy".

        Returns:
            Result[SknnTrainResponse, Exception]: An `Ok` containing the parsed `SknnTrainResponse` object on success, which includes the training metadata, encrypted label shape, and execution time.
            Returns an `Err` containing the raised Exception if the HTTP request, connection, or data validation fails.
        """
        try:
            headers = {
                "Model-Filename": model_filename,
                "Model-Id":model_id,
                "Model-Labels-Filename": model_labels_filename,
                "Num-Chunks":str(num_chunks),
                "Extension":extension
            }
            response = R.post(f"{self.sknn_pqc_url}/train", headers=headers,timeout=self.timeout)
            response.raise_for_status()
            data = SknnTrainResponse.model_validate(response.json())
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
        )->Result[KnnPredictResponse, Exception]:
        """Executes the prediction phase for the Post-Quantum Cryptography (PQC) Secure K-Nearest Neighbors algorithm.

        Sends a POST request to the secure `/classification/sknn_pqc/predict` endpoint. This method 
        evaluates a new set of test records against a previously trained dataset that has been secured 
        using post-quantum cryptographic primitives. It requires specific metadata regarding the shape 
        and data type of the encrypted model to ensure the distributed elastic workers can properly 
        reconstruct and process the quantum-resistant matrices. All configuration parameters are 
        securely transmitted via HTTP headers.

        Args:
            model_id (str): The unique identifier of the trained PQC secure model.
            model_filename (str): The filename of the dataset containing the encrypted training features.
            model_labels_filename (str): The filename containing the corresponding training labels.
            record_test_id (str): The unique identifier for the set of test records to classify.
            record_test_filename (str): The filename of the dataset containing the test records.
            encrypted_model_shape (str): The dimensional shape of the encrypted model matrix (e.g., string representation of a tuple).
            num_chunks (int, optional): The number of chunks into which the dataset is split for distributed elastic processing. Defaults to 2.
            extension (str, optional): The file extension of the dataset files. Defaults to "npy".
            encrypted_model_dtype (str, optional): The data type of the encrypted model matrix. Defaults to "float64".

        Returns:
            Result[KnnPredictResponse, Exception]: An `Ok` containing the parsed `KnnPredictResponse` object on success, with attributes such as:
                - label_vector (List[int]): The resulting predictions/labels for the test records.
                - algorithm (str): The specific algorithm used (e.g., 'SKNN_PQC').
                - worker_id (str): The unique identifier of the worker node that processed the task.
                - service_time_manager (float): The time taken by the manager node (in seconds).
                - service_time_worker (float): The time taken by the worker node (in seconds).
                - service_time_client (float): The time taken at the client level (in seconds).
                - service_time_predict (float): The specific time taken to compute the secure PQC predictions.
            Returns an `Err` containing the raised Exception if the HTTP request, connection, or data validation fails.
        """
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
            response = R.post(f"{self.sknn_pqc_url}/predict", headers=headers,timeout=self.timeout)
            response.raise_for_status()
            data = KnnPredictResponse.model_validate(response.json())
            return Ok(data)
        except Exception as e:
            return Err(e)

    def sknn_pqc(self,
        model_id:str, 
        model_filename:str,
        model_labels_filename:str, 
        record_test_id:str,
        record_test_filename:str,
        num_chunks:int=2,
        extension:str="npy",
        )->Result[KnnResponse, Exception]:
        """Executes the complete Post-Quantum Cryptography (PQC) Secure K-Nearest Neighbors classification workflow.

        This method acts as a high-level wrapper that sequentially orchestrates both the privacy-preserving 
        training and prediction phases using quantum-resistant algorithms. It first calls `sknn_pqc_train` 
        to securely register, segmentation, and encrypt the training dataset on the platform. Upon success, 
        it automatically extracts the encrypted model's metadata (shape and data type) and invokes 
        `sknn_pqc_predict` to securely classify the test records. Finally, it aggregates the prediction 
        results and timing metrics from both steps into a single, unified response.

        Args:
            model_id (str): The unique identifier to assign to the trained PQC secure model.
            model_filename (str): The filename of the dataset containing the training features.
            model_labels_filename (str): The filename containing the corresponding training labels.
            record_test_id (str): The unique identifier for the set of test records to classify.
            record_test_filename (str): The filename of the dataset containing the test records.
            num_chunks (int, optional): The number of chunks into which the datasets are split for distributed elastic processing. Defaults to 2.
            extension (str, optional): The file extension of the dataset files. Defaults to "npy".

        Returns:
            Result[KnnResponse, Exception]: An `Ok` containing the parsed `KnnResponse` object on success, with attributes such as:
                - algorithm (str): The algorithm name (specifically 'SKNNPQC').
                - label_vector (List[int]): The resulting predictions/labels for the test records.
                - worker_id (str): The unique identifier of the worker node that processed the prediction task.
                - service_time_manager (float): The time taken by the manager node (in seconds).
                - service_time_worker (float): The time taken by the worker node (in seconds).
                - service_time_client (float): The total time recorded at the client level (in seconds).
                - service_time_predict (float): The specific time taken to compute the secure PQC predictions.
                - service_time_train (float): The specific time taken to complete the secure PQC training phase.
            Returns an `Err` containing the raised Exception if either the secure training or the prediction phase fails.
        """
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
                encrypted_model_shape = train_response.encrypted_model_shape,
                encrypted_model_dtype = train_response.encrypted_model_dtype
            )
            if predict_result.is_err:
                return predict_result
            predict_response = predict_result.unwrap()

            return Ok(KnnResponse.model_validate(
            {

                "service_time_client":predict_response.service_time_client,
                "algorithm":"SKNNPQC",
                "label_vector":predict_response.label_vector,
                "service_time_manager":predict_response.service_time_manager,
                "service_time_predict":predict_response.service_time_predict,
                "service_time_worker":predict_response.service_time_worker,
                "worker_id":predict_response.worker_id,
                "service_time_train":train_response.response_time
            }
            ))
            # return predict_result
        except Exception as e:
            return Err(e)

    def logistic_regression_train(self,
        plaintext_matrix_train_id: str,
        plaintext_matrix_train_filename: str,
        plaintext_label_vector_train_id: str,
        plaintext_label_vector_train_filename: str,
        extension: str = "npy",
        epochs: int = 1,
        learning_rate: float = 0.01,
    ) -> Result[LogisticRegressionTrainResponse, Exception]:
        try:
            headers = {
                "Experiment-Id": plaintext_matrix_train_id,
                "Plaintext-Matrix-Train-Id": plaintext_matrix_train_id,
                "Plaintext-Matrix-Train-Filename": plaintext_matrix_train_filename,
                "Plaintext-Label-Vector-Train-Id": plaintext_label_vector_train_id,
                "Plaintext-Label-Vector-Train-Filename": plaintext_label_vector_train_filename,
                "Extension": extension,
                "Epochs": str(epochs),
                "Learning-Rate": str(learning_rate),
            }
            response = R.post(f"{self.machine_learning_url}/logistic-regression/train", headers=headers, timeout=self.timeout)
            response.raise_for_status()
            data = LogisticRegressionTrainResponse.model_validate(response.json())
            return Ok(data)
        except Exception as e:
            return Err(e)

    def logistic_regression_predict(self,
        plaintext_matrix_train_id: str,
        plaintext_matrix_test_id: str,
        plaintext_matrix_test_filename: str,
        extension: str = "npy",
    ) -> Result[LogisticRegressionPredictResponse, Exception]:
        try:
            headers = {
                "Experiment-Id": plaintext_matrix_train_id,
                "Plaintext-Matrix-Train-Id": plaintext_matrix_train_id,
                "Plaintext-Matrix-Test-Id": plaintext_matrix_test_id,
                "Plaintext-Matrix-Test-Filename": plaintext_matrix_test_filename,
                "Extension": extension,
            }
            response = R.post(f"{self.machine_learning_url}/logistic-regression/predict", headers=headers, timeout=self.timeout)
            response.raise_for_status()
            data = LogisticRegressionPredictResponse.model_validate(response.json())
            return Ok(data)
        except Exception as e:
            return Err(e)

    def logistic_regression(self,
        plaintext_matrix_train_id: str,
        plaintext_matrix_train_filename: str,
        plaintext_label_vector_train_id: str,
        plaintext_label_vector_train_filename: str,
        plaintext_matrix_test_id: str,
        plaintext_matrix_test_filename: str,
        extension: str = "npy",
        epochs: int = 1,
        learning_rate: float = 0.01,
    ) -> Result[LogisticRegressionResponse, Exception]:
        try:
            train_result = self.logistic_regression_train(
                plaintext_matrix_train_id=plaintext_matrix_train_id,
                plaintext_matrix_train_filename=plaintext_matrix_train_filename,
                plaintext_label_vector_train_id=plaintext_label_vector_train_id,
                plaintext_label_vector_train_filename=plaintext_label_vector_train_filename,
                extension=extension,
                epochs=epochs,
                learning_rate=learning_rate,
            )
            if train_result.is_err:
                return train_result
            train_response = train_result.unwrap()

            predict_result = self.logistic_regression_predict(
                plaintext_matrix_train_id=plaintext_matrix_train_id,
                plaintext_matrix_test_id=plaintext_matrix_test_id,
                plaintext_matrix_test_filename=plaintext_matrix_test_filename,
                extension=extension,
            )
            if predict_result.is_err:
                return predict_result
            predict_response = predict_result.unwrap()

            return Ok(LogisticRegressionResponse.model_validate({
                "algorithm": "LOGISTIC_REGRESSION",
                "label_vector": predict_response.label_vector,
                "worker_id": predict_response.worker_id,
                "service_time_manager": predict_response.service_time_manager,
                "service_time_worker": predict_response.service_time_worker,
                "service_time_client": predict_response.service_time_client,
                "service_time_predict": predict_response.service_time_predict,
                "service_time_train": train_response.service_time_train,
            }))
        except Exception as e:
            return Err(e)

    def pplr_train(self,
        plaintext_matrix_train_id: str,
        plaintext_matrix_train_filename: str,
        plaintext_label_vector_train_id: str,
        plaintext_label_vector_train_filename: str,
        extension: str = "npy",
        epochs: int = 1,
        learning_rate: float = 0.01,
    ) -> Result[PPLRTrainResponse, Exception]:
        try:
            headers = {
                "Experiment-Id": plaintext_matrix_train_id,
                "Plaintext-Matrix-Train-Id": plaintext_matrix_train_id,
                "Plaintext-Matrix-Train-Filename": plaintext_matrix_train_filename,
                "Plaintext-Label-Vector-Train-Id": plaintext_label_vector_train_id,
                "Plaintext-Label-Vector-Train-Filename": plaintext_label_vector_train_filename,
                "Extension": extension,
                "Epochs": str(epochs),
                "Learning-Rate": str(learning_rate),
            }
            response = R.post(f"{self.machine_learning_url}/pplr/train", headers=headers, timeout=self.timeout)
            response.raise_for_status()
            data = PPLRTrainResponse.model_validate(response.json())
            return Ok(data)
        except Exception as e:
            return Err(e)

    def pplr_predict(self,
        plaintext_matrix_train_id: str,
        plaintext_matrix_test_id: str,
        plaintext_matrix_test_filename: str,
        extension: str = "npy",
    ) -> Result[PPLRPredictResponse, Exception]:
        try:
            headers = {
                "Experiment-Id": plaintext_matrix_train_id,
                "Plaintext-Matrix-Train-Id": plaintext_matrix_train_id,
                "Plaintext-Matrix-Test-Id": plaintext_matrix_test_id,
                "Plaintext-Matrix-Test-Filename": plaintext_matrix_test_filename,
                "Extension": extension,
            }
            response = R.post(f"{self.machine_learning_url}/pplr/predict", headers=headers, timeout=self.timeout)
            response.raise_for_status()
            data = PPLRPredictResponse.model_validate(response.json())
            return Ok(data)
        except Exception as e:
            return Err(e)

    def pplr(self,
        plaintext_matrix_train_id: str,
        plaintext_matrix_train_filename: str,
        plaintext_label_vector_train_id: str,
        plaintext_label_vector_train_filename: str,
        plaintext_matrix_test_id: str,
        plaintext_matrix_test_filename: str,
        extension: str = "npy",
        epochs: int = 1,
        learning_rate: float = 0.01,
    ) -> Result[PPLRResponse, Exception]:
        try:
            train_result = self.pplr_train(
                plaintext_matrix_train_id=plaintext_matrix_train_id,
                plaintext_matrix_train_filename=plaintext_matrix_train_filename,
                plaintext_label_vector_train_id=plaintext_label_vector_train_id,
                plaintext_label_vector_train_filename=plaintext_label_vector_train_filename,
                extension=extension,
                epochs=epochs,
                learning_rate=learning_rate,
            )
            if train_result.is_err:
                return train_result
            train_response = train_result.unwrap()

            predict_result = self.pplr_predict(
                plaintext_matrix_train_id=plaintext_matrix_train_id,
                plaintext_matrix_test_id=plaintext_matrix_test_id,
                plaintext_matrix_test_filename=plaintext_matrix_test_filename,
                extension=extension,
            )
            if predict_result.is_err:
                return predict_result
            predict_response = predict_result.unwrap()

            return Ok(PPLRResponse.model_validate({
                "algorithm": "PPLR",
                "label_vector": predict_response.label_vector,
                "worker_id": predict_response.worker_id,
                "epochs": train_response.epochs,
                "service_time_manager": predict_response.service_time_manager,
                "service_time_worker": predict_response.service_time_worker,
                "service_time_client": predict_response.service_time_client,
                "service_time_predict": predict_response.service_time_predict,
                "service_time_train": train_response.service_time_train,
            }))
        except Exception as e:
            return Err(e)


if __name__ == "__main__":
    rc = RoryClient(hostname="localhost",port=3000)
