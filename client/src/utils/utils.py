import numpy as np
import numpy.typing as npt
import os

MAX_RETRIES = int(os.environ.get("MAX_RETRIES","10"))
MAX_DELAY   = int(os.environ.get("MAX_DELAY","2"))
JITTER      = eval(os.environ.get("JITTER","(.1,.5)"))

class Utils:
    

    @staticmethod 
    def get_workers(num_chunks:int = 2):
        """
            Determine the optimal number of worker threads/processes based on CPU cores and desired chunk parallelism.

            This method calculates and returns the appropriate number of workers to efficiently process tasks in parallel,
            taking into account the available CPU cores and a user-specified maximum number of parallel chunks (`num_chunks`).

            The returned worker count ensures that system resources are efficiently utilized without oversubscription,
            maintaining optimal performance.

            Args:
                num_chunks (int, optional): The preferred maximum number of parallel chunks/tasks to execute concurrently.
                                            Defaults to 2.

            Returns:
                int: Optimal number of worker threads/processes calculated by considering:
                    - The number of CPU cores available on the host system.
                    - The specified maximum number of parallel chunks (`num_chunks`).

                    The final worker count returned will not exceed either:
                    - The number of available CPU cores.
                    - The specified `num_chunks`.

            Raises:
                ValueError: If `num_chunks` provided is less than 1.

            Examples:
                >>> Utils.get_workers(num_chunks=4)
                4  # assuming at least 4 cores available

                >>> Utils.get_workers(num_chunks=32)
                8  # assuming 8 CPU cores available, returns cores count as limit

            Notes:
                - Ensure this method is used when distributing tasks that benefit from parallel processing.
                - Excessive parallelism beyond CPU cores may degrade performance due to overhead.

        """
        cores = os.cpu_count()
        return cores if num_chunks > cores else num_chunks

    
    @staticmethod
    def verify_mean_error(old_matrix:npt.NDArray, new_matrix:npt.NDArray, min_error:float=0.15)->bool:
        mean_error = np.mean(np.abs((old_matrix - new_matrix) / old_matrix))
        return mean_error <= min_error

  



