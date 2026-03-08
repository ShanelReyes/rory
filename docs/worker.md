# Rory Worker

The **Rory Worker** is responsible for executing data mining tasks, such as secure clustering and classification, while interacting directly with the CSS (Cloud Storage Service) to retrieve and store encrypted data chunks.

## Key Responsibilities

* **Computational Execution:** Performs SKMeans, DBSKMeans, and SKNN algorithms on encrypted or plaintext data.
* **Manager Interaction:** Notifies availability status and receives workload instructions from the Manager.
* **CSS Chunk Management:** Uses an asynchronous client to handle data segmentation and persistence within the distributed storage layer.
* **Cryptographic Processing:** Implements Liu and CKKS schemes to ensure privacy-preserving computations during the mining process.

## Configuration & Environment

The Worker is configured via environment variables, typically loaded from `/rory/envs/.worker.env.`


### Node & Network Configuration
| Variable | Description | Default Value |
| :--- | :--- | :--- |
| `NODE_ID` | Unique identifier for the worker in the cluster. | `rory-worker-0` |
| `NODE_PORT` | Internal port where the Flask service listens. | `9000` |
| `NODE_INDEX` | Node index used for port offset calculations. | `0` |
| `NODE_IP_ADDR` | IP address or Hostname of the current node. | `NODE_ID` |
| `RORY_MANAGER_IP_ADDR` | IP address of the Manager node. | `localhost` |
| `RORY_MANAGER_PORT` | Communication port for the Manager. | `6000` |
| `SERVER_IP_ADDR` | Network interface for the app server. | `0.0.0.0` |
| `DISTANCE` | Distance metric for mining algorithms. | `MANHATHAN` |
| `MIN_ERROR` | Minimum error tolerance for convergence. | `0.015` |

### Cryptographic Parameters (Liu & CKKS)
| Variable | Description | Default Value |
| :--- | :--- | :--- |
| `CKKS_ROUND` | Indicates if rounding should be applied in CKKS operations. | `0` (False) |
| `CKKS_DECIMALS` | Number of decimal units to preserve during encryption. | `2` |
| `CTX_FILENAME` | Filename for the cryptographic context. | `ctx` |
| `PUBKEY_FILENAME` | Filename for the public key. | `pubkey` |
| `SECRET_KEY_FILENAME`| Filename for the secret key. | `secretkey` |
| `RELINKEY_FILENAME` | Filename for the relinearization key. | `relinkey` |

### Distributed Storage (Mictlanx / CSS)
| Variable | Description | Default Value |
| :--- | :--- | :--- |
| `MICTLANX_ROUTERS` | List of routers (host:port) for Mictlanx. | `mictlanx-router-0:localhost:60666` |
| `MICTLANX_BUCKET_ID` | Identifier for the storage bucket. | `rory` |
| `MICTLANX_CHUNK_SIZE` | Size of the data fragments/chunks. | `256kb` |
| `MICTLANX_TIMEOUT` | Timeout duration for storage requests. | `120` |
| `MICTLANX_MAX_WORKERS`| Maximum threads for the storage client. | `4` |
| `MICTLANX_MAX_RETRIES`| Maximum reconnection attempts to storage. | `10` |

### System Paths (File System)
| Variable | Description | Default Value |
| :--- | :--- | :--- |
| `SOURCE_PATH` | Input directory for raw data. | `/rory/source` |
| `SINK_PATH` | Output directory for processed results. | `/rory/sink` |
| `LOG_PATH` | Directory for system log files. | `/rory/log` |
| `KEYS_PATH` | Directory where security keys (.key) are stored. | `/rory/keys` |
| `ENV_FILE_PATH` | Path to the environment variables file. | `/rory/envs/.worker.env` |

---


## Local Usage

To run the **Rory Worker** in a local environment, follow these steps to ensure all cryptographic and storage dependencies are correctly configured.

### Prerequisites

Before starting the service, ensure you have the following:

* **Python 3.11+** installed.
* **Virtual Environment** tool (`venv` or `virtualenv`).
* **Mictlanx (CSS)**: The storage layer must be running (usually via Docker) as the Client attempts to initialize an `AsyncStorageClient` on startup.
* **Path Definition:** Set the environment variable for your client location:
```bash
export WORKER_PATH=/home/<user>/rory
```

### Environment Setup

1.  **Navigate to the worker directory**:

    ```bash
    cd rory/worker
    ```

2.  **Initialize the Virtual Environment**:

    ```bash
    python3 -m venv worker-env
    source worker-env/bin/activate
    ```

3.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

### Workspace & Directory Structure
The worker expects a specific filesystem organization for data persistence and logging. Create the following directories:

```bash
# Using default paths for node: rory-client-0
mkdir -p /rory/rory-worker-0/source
mkdir -p /rory/rory-worker-0/sink
mkdir -p /rory/rory-worker-0/log
mkdir -p /rory/keys
```

* **Relinearization Key:**  Move your CKKS context and relinearization key file into the `keys/` folder.

### Executing the Worker Service

1. **Initialize Mictlanx (CSS)**: Navigate to the Mictlanx directory and start the storage routers using Docker Compose:

    ```bash
    cd rory/mictlanx
    docker compose -f ./router-static.yml down
    docker compose -f ./router-static.yml up -d
    # Alternatively, you can use the provided startup script:
    # ./run.sh
    ```

2. **Run using Gunicorn:** Once the storage is ready, navigate back to the client source and start the service:

    ```bash
    cd ../worker/src
    gunicorn --chdir $WORKER_PATH/src --config $WORKER_PATH/src/gunicorn_config.py main:app
    # Alternatively, you can use the provided startup script:
    # ../run.sh
    ```

!!! warning "Service Dependencies & Startup Order"
    Although a Worker can be started independently, it is highly recommended to follow this startup sequence for proper cluster registration and data availability: 1. **Mictlanx / CSS**, 2. **Rory Manager**, 3. **Rory Worker**.


### Health Check & Verification

Once the service is running, verify the node's status and its identified role (Client) by performing a health check:

  ```bash
  curl -X GET http://localhost:9000/clustering/test
  ```

**Expected JSON Response:**
```json
{
  "component_type": "worker",
  "status": "ready"
}
```