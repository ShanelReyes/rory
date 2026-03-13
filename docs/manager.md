---
icon: lucide/network
---

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



## Docker Containerization

The Manager component is containerized to ensure a stable orchestration environment. It handles the lifecycle of worker nodes and task distribution, requiring a consistent environment for the CSS management layer and cryptographic context synchronization.

### Dockerfile Architecture

The Manager includes a dedicated `Dockerfile` located in its root directory (`/manager/Dockerfile`). The container has been created using a multi-layer approach to improve performance and deployment. Below is a breakdown of the environment configuration:

* **Base Image**: python:3.11-bullseye (chosen for its stability and support for complex C++ extensions).
* **Working Directory**: /app
* **Dependency Management**: High-timeout installation (1800s) to ensure the Flask-based orchestration engine and its dependencies are correctly compiled.


### Manual Image Construction

To build the Manager image manually, navigate to its root directory and execute the build command:

```bash
# Navigate to the manager directory
cd /rory/manager

# Build the image using the local Dockerfile
docker build -t rory:manager -f Dockerfile .
```

`-t`: Defines the repository name and tag (e.g., `rory:manager`).

`-f`: Specifies the `Dockerfile` located within the current directory.

`.`: Sets the build context to the current folder to include the `src` and `requirements.txt`.

### Automated Build Script

The `build.sh` script for the Manager targets the `/manager/` directory and allows for flexible image tagging.

```bash title="build.sh"
#!/bin/bash
readonly BASE_PATH=${1:-/rory}
readonly IMAGE=${2:-rory:manager}

# The script targets the manager component folder specifically
docker build -t ${IMAGE} ${BASE_PATH}/manager/
```

**Usage Instructions**

The script accepts two optional positional arguments to generalize the construction:

* `BASE_PATH`: The root directory where the Rory project is located.
* `IMAGE`: The full name and tag for the resulting Docker image.

```bash
# Build the manager image using default values (rory:manager)
./build.sh

# Build with a custom image name and project path
./build.sh /home/sreyes/rory shanelreyes/rory:manager-prod
```

### Orchestration with Docker Compose

The manager node can be orchestrated using `docker-compose.yml` to manage network identities, volumes for keys/logs, and environment variables.

**Network Dependency**

The manager requires the external mictlanx network to communicate with the CSS layer:
```bash
docker network create mictlanx
```

**Service Definition**

```yaml title="docker-compose.yml"
services:
  rory-manager-0:
    image: shanelreyes/rory:manager
    container_name: rory-manager-0
    hostname: rory-manager-0
    ports:
      - 3000:3000
    environment:
      - NODE_ID=rory-manager-0
      - RORY_MANAGER_IP_ADDR=rory-manager-0
    volumes:
      - /rory/rory-manager-0/source:/rory/source
      - /rory/rory-manager-0/keys:/rory/keys
      - /rory/rory-manager-0/log:/rory/log
    networks:
      - mictlanx
```

### Deployment Commands
The `deploy.sh` script for the Manager handles environment variable loading and build flags to ensure the orchestrator starts with the correct configuration.  

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

Verify that the manager image and container are correctly initialized in the Docker engine:

```bash
# Check manager image
docker images | grep rory:manager

# Check running manager container
docker ps | grep rory-manager-0
```