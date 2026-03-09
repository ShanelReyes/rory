---
icon: lucide/cpu
---

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
* **Path Definition:** Set the environment variable for your worker location:
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
# Using default paths for node: rory-worker-0
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

2. **Run using Gunicorn:** Once the storage is ready, navigate back to the worker source and start the service:

    ```bash
    cd ../worker/src
    gunicorn --chdir $WORKER_PATH/src --config $WORKER_PATH/src/gunicorn_config.py main:app
    # Alternatively, you can use the provided startup script:
    # ../run.sh
    ```

!!! warning "Service Dependencies & Startup Order"
    Although a Worker can be started independently, it is highly recommended to follow this startup sequence for proper cluster registration and data availability: 1. **Mictlanx / CSS**, 2. **Rory Manager**, 3. **Rory Worker**.


### Health Check & Verification

Once the service is running, verify the node's status and its identified role (worker) by performing a health check:

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



## Docker Containerization

The system is designed to run in containerized environments using Docker, ensuring that the complex cryptographic dependencies remain consistent across different nodes in the distributed network.

### Dockerfile Architecture

The Worker includes a dedicated `Dockerfile` located in its root directory (`/worker/Dockerfile`). The container has been created using a multi-layer approach to improve performance and deployment. Below is a breakdown of the environment configuration:

* **Base Image**: python:3.11-bullseye (chosen for its stability and support for complex C++ extensions).
* **Working Directory**: /app
* **Dependency Management**: Uses a high-timeout installation (1800s) to accommodate the compilation of heavy cryptographic libraries.


### Manual Image Construction

To build the Worker image manually, you must navigate to the worker's directory. The build process uses the local context to package the source code and configuration:

```bash
# Navigate to the worker directory
cd /rory/worker

# Build the image using the local Dockerfile
docker build -t rory:worker -f Dockerfile .
```

`-t`: Defines the repository name and tag (e.g., `rory:worker`).

`-f`: Specifies the `Dockerfile` located within the current directory.

`.`: Sets the build context to the current folder to include the `src` and `requirements.txt`.

### Automated Build Script

For a standardized deployment, the `build.sh` script automates the process by targeting the worker folder structure dynamically.

```bash title="build.sh"
#!/bin/bash
readonly BASE_PATH=${1:-/rory}
readonly IMAGE=${2:-rory:worker}

# The script targets the worker component folder specifically
docker build -t ${IMAGE} ${BASE_PATH}/worker/
```

**Usage Instructions**

The script accepts two optional positional arguments to generalize the construction:

* `BASE_PATH`: The root directory where the Rory project is located.
* `IMAGE`: The full name and tag for the resulting Docker image.

```bash
# Build the worker image using default values (rory:worker)
./build.sh

# Build with a custom image name and project path
./build.sh /home/sreyes/rory shanelreyes/rory:worker-prod
```

### Orchestration with Docker Compose

The Worker node can be orchestrated using `docker-compose.yml` to manage network identities, volumes for keys/logs, and environment variables.

**Network Dependency**

The Worker requires the external mictlanx network to communicate with the CSS layer:
```bash
docker network create mictlanx
```

**Service Definition**

```yaml title="docker-compose.yml"
services:
  rory-worker-0:
    image: shanelreyes/rory:worker
    container_name: rory-worker-0
    hostname: rory-worker-0
    ports:
      - 3000:3000
    environment:
      - NODE_ID=rory-worker-0
      - RORY_MANAGER_IP_ADDR=rory-manager-0
    volumes:
      - /rory/rory-worker-0/source:/rory/source
      - /rory/rory-worker-0/keys:/rory/keys
      - /rory/rory-worker-0/log:/rory/log
    networks:
      - mictlanx
```

### Deployment Commands
The `deploy.sh` script simplifies the container lifecycle by managing environment files, build modes, and detached flags.


```bash title="deploy.sh"
#!/bin/bash
readonly COMPOSE_FILE=${4:-docker-compose.yml}
readonly ENV_FILE_PATH=${3:-.env.dev}
readonly BUILD_MODE=${1:-0}
readonly DETACHED_MODE=${2:-0}
readonly detached_flag=$([ "$DETACHED_MODE" -eq 1 ] && echo "-d" || echo "")

if [ "$BUILD_MODE" -eq 1 ]; then
    echo "Building Docker images..."
    docker compose --env-file ${ENV_FILE_PATH} -f ./${COMPOSE_FILE} up ${detached_flag} --build 
else
    docker compose --env-file ${ENV_FILE_PATH} -f ./${COMPOSE_FILE} up ${detached_flag}
fi
```

**Usage Instructions**
The script uses four positional arguments for deployment control:

* `BUILD_MODE ($1)`: Set to `1` to force image rebuilding, `0` to use existing images.
* `DETACHED_MODE ($2)`: Set to `1` to run in the background (detached), `0` for interactive mode.
* `ENV_FILE_PATH ($3)`: Path to the environment variables file (Default: `.env.dev`).
* `COMPOSE_FILE ($4)`: Filename of the target compose manifest (Default: `docker-compose.yml`).

```bash title="Examples"
# Standard deployment (Interactive, no build, using .env.dev)
./deploy.sh

# Rebuild images and run in detached mode
./deploy.sh 1 1

# Run in background using a specific production environment file
./deploy.sh 0 1 .env.prod
```


### Verification

Verify that the Worker image and container are correctly initialized in the Docker engine:

```bash
# Check worker image
docker images | grep rory:worker

# Check running worker container
docker ps | grep rory-worker-0
```