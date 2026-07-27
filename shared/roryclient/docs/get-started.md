---
icon: material/rocket-launch
---

<!-- # Get started  -->
<!-- ## Getting Started with Rory -->

The **Rory Client** is a Python-based library designed to interact with the **PPDMaaS** (Privacy-Preserving Data Mining as a Service) platform. It provides a seamless way to perform clustering and classification tasks using both traditional and **Post-Quantum Secure (PQC)** methods.



## 1. Installation

Since the project manages dependencies with `poetry`, you can set up your environment by running the following commands in your terminal:

```bash
# Clone the repository
git clone <git@github.com:ShanelReyes/rory_client.git>
cd rory_client

# Install dependencies from the requirements file
pip install -r requirements.txt
```

## 2. Platform Deployment

Before using the client, you must ensure the Rory platform is running. You have two options depending on your infrastructure and needs:

### Option A: Local Deployment (For Developers)
This step is optional and should only be performed by users who wish to carry out local development or host their own instance of the platform.

### Permissions
First, grant execution permissions to the deployment script:

```bash
chmod +x deploy.sh

```

### Execution
Run the script to deploy the Rory platform services, including the Clustering, Classification, and Distributed modules:

```bash
./deploy.sh
```

> **Note:** This will deploy a minimal version of the Rory platform on your local machine.

### Option B: Remote Testing Service
If you do not meet the hardware requirements for a local deployment, or if you prefer to start testing immediately, you can point your client to the remote service.

* **Testing Endpoint**: `apix.tamps.cinvestav.mx/rory`

!!! warning "Volume Restriction"
    The remote testing route is **not** intended for processing large volumes of data. It is provided strictly for functional testing and small-scale demonstrations.

> **Note:** When using Option B, ensure your `RoryClient` initialization reflects the remote hostname instead of `localhost`.


## 3. Initializing the Client

To start using the platform, you need to initialize the `RoryClient` with your server's connection details. Depending on the deployment option you chose in the previous step, the configuration will vary:

### Connection Settings
```Python
    from roryclient.client import RoryClient

    # Option A: Connecting to a Local Deployment
    rc = RoryClient(hostname="localhost", port=3000)

    # Option B: Connecting to the Remote Testing Service
    # Note: Use the endpoint provided for small-scale tests
    rc_remote = RoryClient(hostname="apix.tamps.cinvestav.mx/rory", port=80)
```
### Parameter Details

By default, the client uses the following configuration:

- **Hostname**: `localhost` (for local development) or `apix.tamps.cinvestav.mx/rory` (for remote testing).
- **Port**: `3000` for local instances or the standard HTTP port (`80`) for the service.
- **Timeout**: `120` seconds. This extended time allows the elastic architecture to process complex data mining tasks without interrupting the connection.

> **Note:** Ensure that the `hostname` and `port` exactly match the environment where you executed the `deploy.sh` script or the remote service to avoid connection errors.


### 4. Running your first Secure Mining Task
Rory focuses on **Privacy-Preserving Data Mining (PPDM)**. The client uses a secure communication protocol where task parameters are sent via headers to interact with the backend.

#### Example: Secure K-Means (skmeans)
To run a Sk-Means task, you need to reference a dataset already identified by the platform. The client will handle the request to the `/clustering/skmeans` endpoint.


```Python
# define the dataset reference and parameters
matrix_id = "dataset_001"
filename = "research_data_v1"

# Execute Secure K-Means
# Parameters: matrix_id, filename, iteration, chunks, k, max_iterations
result = rc.skmeans(
    plaintext_matrix_id=matrix_id,
    plaintext_matrix_filename=filename,
    experiment_iteration=1,
    num_chunks=4,
    k=3,
    max_iterations=10
)

if result.is_ok():
    response = result.unwrap()
    print(f"Algorithm used: {response.algorithm}")
    print(f"Clusters (Label Vector): {response.label_vector}")
    print(f"Service Time (Worker): {response.service_time_worker}s")
else:
    print("Error in Secure Mining task:", result.unwrap_err())
```