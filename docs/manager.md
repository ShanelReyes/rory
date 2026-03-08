
# Rory Manager

The **Rory Manager** acts as the central orchestrator of the system. It is responsible for managing the lifecycle of mining tasks, coordinating communication between Clients and Workers, and maintaining the global state of the distributed computing network.

## Key Responsibilities

* **Service Discovery & Registration**: Tracks active Worker nodes and manages their registration via prefix-based identification.
* **Dynamic Worker Replication**: Coordinates the deployment and monitoring of Worker nodes across the network using the CSS management layer.
* **Load Balancing**: Implements algorithms (Round Robin, Two Choices, Random) to distribute computational load across the Worker pool.
* **Parameters Distribution**: Ensures consistent Post-Quantum Cryptography (PQC) context across all orchestrated nodes.

## Configuration & Environment

The Manager is configured via environment variables, typically loaded from `/rory/envs/.manager.env`.

### Node & Network Configuration
| Variable | Description | Default Value |
|:---|:---|:---|
| `NODE_ID` | Unique identifier for the manager node. | `rory-manager-0` |
| `NODE_IP_ADDR` | IP address or hostname of the manager. | `NODE_ID` |
| `NODE_PORT` | Listening port for the orchestration service. | `6000` |
| `RORY_DEBUG` | Enables debug mode and .env loading. | `0` (False) |
| `LOAD_BALANCING` | Algorithm index (0: RoundRobin, 1: TwoChoices, 2: Random). | `0` |

### Worker Replication
Settings for the dynamic deployment of computational nodes.

| Variable | Description | Default Value |
|:---|:---|:---|
| `INIT_WORKERS` | Number of worker nodes to deploy on startup. | `0` |
| `WORKER_INIT_PORT`| Starting port for the worker pool. | `9000` |
| `NODE_PREFIX` | Prefix name for generated worker nodes. | `rory-worker-` |
| `DOCKER_IMAGE` | Full image name and tag for workers. | `shanelreyes/rory:worker` |
| `DOCKER_NETWORK_ID`| Target Docker network for the cluster. | `mictlanx` |
| `SWARM_NODES` | List of available Swarm node IDs. | `2,3,4,8` |
| `WORKER_TIMEOUT` | Max response time for worker operations. | `300` |

### Cryptographic & Algorithm Parameters
| Variable | Description | Default Value |
|:---|:---|:---|
| `FOLDER_KEYS` | Directory name for security keys. | `keys128` |
| `DISTANCE` | Distance metric (e.g., MANHATTAN). | `MANHATTAN` |
| `MIN_ERROR` | Convergence threshold for clustering. | `0.015` |
| `LIU_ROUND` | Rounding precision for Liu scheme. | `2` |
| `CKKS_DECIMALS` | Encoding precision for CKKS. | `2` |
| `CTX_FILENAME` | Filename for the CKKS context. | `ctx` |

### Mictlanx & Storage
| Variable | Description | Default Value |
|:---|:---|:---|
| `MICTLANX_SUMMONER_IP_ADDR`| IP for the Mictlanx Summoner service. | `localhost` |
| `MICTLANX_SUMMONER_MODE` | Deployment mode (docker/swarm). | `docker` |
| `MICTLANX_BUCKET_ID` | Target storage bucket. | `rory` |
| `MICTLANX_ROUTERS` | List of CSS routers. | `mictlanx-router-0:localhost:60666` |

---

## Local Usage

Follow these steps to deploy the **Rory Manager** in a local environment.

### Prerequisites

Before starting the service, ensure you have the following:

* **Python 3.11+**: installed.
* **Docker Engine**: Must be running if `INIT_WORKERS > 0`.
* **Mictlanx**: The replication service must be accessible at the configured IP.
* **Path Definition:** Set the environment variable for your manager location:
```bash
export MANAGER_PATH=/home/<user>/rory
```

### Environment Setup

1.  **Navigate to the manager directory**:

    ```bash
    cd rory/manager
    ```

2.  **Initialize the Virtual Environment**:

    ```bash
    python3 -m venv rory-env
    source rory-env/bin/activate
    ```

3.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

### Workspace & Directory Structure

Create the required directories for the manager's operation. These paths must match your `SOURCE_PATH`, `SINK_PATH`, and `LOG_PATH` settings.

```bash
mkdir -p /rory/source
mkdir -p /rory/sink
mkdir -p /rory/log
```

### Executing the Manager 

The Manager requires the storage and orchestration layer (CSS) to be active to successfully manage worker nodes.

1. **Initialize CSS (Storage & Management) Layer:** Navigate to the Mictlanx/CSS directory and start the necessary services:

    ```bash
    cd rory/mictlanx
    docker compose -f ./router-static.yml down
    docker compose -f ./router-static.yml up -d
    # Alternatively, you can use the provided startup script:
    # ./run.sh
    ```

2. **Run the Manager using Gunicorn:**

    ```bash
    cd ../manager/src
    gunicorn --chdir $MANAGER_PATH/src --config $MANAGER_PATH/src/gunicorn_config.py main:app
    # Alternatively, you can use the provided startup script:
    # ./run.sh
    ```

### Health Check & Verification

Confirm the Manager is ready to orchestrate the network by performing a health check:

  ```bash
  curl -X GET http://localhost:6000/clustering/test
  ```

**Expected JSON Response:**
```json
{
  "component_type": "manager",
  "status": "ready"
}
```