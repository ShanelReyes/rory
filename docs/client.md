---
icon: lucide/monitor
---

# Rory Client

The **Rory Client** acts as the **Data Owner (DO)** and primary security authority within the **Rory** platform, a specialized system for **Post-Quantum Privacy-Preserving Data Mining as a Service (PPDMaaS)**. 

Its main mission is to ensure that confidential dataset never leave the local environment in plain text. Using a hybrid cryptographic approach, combining homomorphic encryption (Liu, CKKS) for secure arithmetic operations and order-preserving encryption (FDHOPE) for secure comparisons, the client enables distributed computing on remote nodes without compromising data privacy.

---

## Key Responsibilities

* **Data Preparation & Segmentation:** Reads local datasets in formats like CSV or NPY and prepares them for distributed processing.
* **Cryptographic Management:** Generates and maintains the private keys required for homomorphic operations.
* **Decryption Authority:** Participates in interactive protocols (e.g., SK-Means or SKNN) by decrypting intermediate results provided by Workers.
* **Cloud Integration:** Manages asynchronous uploads and downloads to the Cloud Storage System (CSS) using the **MictlanX** protocol for high resilience.

---

## Configuration & Environment

The **Rory Client** is configured primarily through environment variables. These can be managed via a `.env` file (located by default at `/rory/envs/.client.env`) as long as the `RORY_DEBUG` variable is enabled.

#### Node & Network Configuration
Defines the client's identity and its communication parameters with orchestrators and processing nodes.

| Variable | Description | Default Value |
|:---|:---|:---|
| `NODE_ID` | Unique identifier for the client node. | `rory-client-0` |
| `NODE_IP_ADDR` | IP address or hostname of the client node. | `NODE_ID` |
| `NODE_PORT` | Listening port for the client service. | `3000` |
| `RORY_MANAGER_IP_ADDR` | IP address of the Rory Manager. | `localhost` |
| `RORY_MANAGER_PORT` | Communication port for the Manager. | `6000` |
| `MAX_WORKERS` | Maximum threads for local encryption processes. | `2` |
| `MAX_ITERATIONS` | Maximum iteration limit for iterative algorithms (e.g., K-Means). | `10` |
| `WORKER_TIMEOUT` | Timeout (s) for waiting on Worker node responses. | `300` |
| `RORY_DEBUG` | Enables debug mode and .env file loading. | `0` (False) |

#### Cryptographic Parameters (Liu & CKKS)
Settings for Homomorphic Encryption (HE) and Post-Quantum Cryptography (PQC) schemes.

| Variable | Description | Default Value |
|:---|:---|:---|
| `LIU_SECURITY_LEVEL` | Security level for the Liu scheme (128, 192, 256). | `128` |
| `LIU_DECIMALS` | Decimal precision for mapping in the Liu scheme. | `6` |
| `LIU_ROUND` | Enables rounding for Liu scheme ciphertexts. | `0` (False) |
| `LIU_SEED` | Seed for secure random number generation. | `123` |
| `CKKS_DECIMALS` | Encoding precision for the CKKS scheme. | `2` |
| `CKKS_ROUND` | Enables rounding for CKKS computation results. | `0` (False) |
| `PUBKEY_FILENAME` | Filename for the CKKS public key. | `pubkey` |
| `SECRET_KEY_FILENAME` | Filename for the CKKS secret key. | `secretkey` |

#### Distributed Storage (Mictlanx / CSS)
Configuration for the persistence layer and communication with the Cloud Storage Service.

| Variable | Description | Default Value |
|:---|:---|:---|
| `NUM_CHUNKS` | Number of segments for data distribution in the CSS. | `4` |
| `MICTLANX_ROUTERS` | List of Mictlanx routers (id:host:port). | `mictlanx-router-0:localhost:60666` |
| `MICTLANX_BUCKET_ID` | Target bucket identifier for mining tasks. | `rory` |
| `MICTLANX_TIMEOUT` | Global timeout for storage operations. | `120` |
| `MICTLANX_MAX_RETRIES` | Maximum retry attempts for failed storage tasks. | `10` |
| `MICTLANX_PROTOCOL` | Communication protocol (http/https). | `https` |

#### System Paths (File System)
Directory organization for datasets, cryptographic keys, and logs.

| Variable | Description | Default Value |
|:---|:---|:---|
| `SOURCE_PATH` | Input directory for raw datasets. | `/rory/source` |
| `SINK_PATH` | Output directory for processed mining results. | `/rory/sink` |
| `LOG_PATH` | Directory for system and execution logs. | `/rory/log` |
| `KEYS_PATH` | Storage directory for PQC and Liu keys. | `/rory/keys` |
| `ENV_FILE_PATH` | Path to the .env configuration file. | `/rory/envs/.client.env` |

---

## Local Usage

To run the **Rory Client** in a local environment, follow these steps to ensure all cryptographic and storage dependencies are correctly configured.

### Prerequisites

Before starting the service, ensure you have the following:

* **Python 3.11+** installed.
* **Virtual Environment** tool (`venv` or `virtualenv`).
* **Mictlanx (CSS)**: The storage layer must be running (usually via Docker) as the Client attempts to initialize an `AsyncStorageClient` on startup.
* **Keys**: If using PQC (CKKS), valid context and key files must be present in your keys directory.
* **Path Definition:** Set the environment variable for your client location:
```bash
export CLIENT_PATH=/home/<user>/rory
```

### Environment Setup

1. **Clone the repository and enter the client directory**:

    ```bash
    git clone git@github.com:ShanelReyes/rory.git
    cd rory/client

    ```

2.  **Create and activate a Virtual Environment**:

    ```bash
    python3 -m venv rory-env
    source rory-env/bin/activate
    ```

3.  **Install required dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Initialize the Environment File**:

    Create the `.env` file in the path defined by `ENV_FILE_PATH` (default: `/rory/envs/.client.env`). Ensure `RORY_DEBUG=1` is set to allow the client to load these configurations on startup.

### Workspace & Directory Structure
The client expects a specific filesystem organization for data persistence and logging. Create the following directories:

```bash
# Using default paths for node: rory-client-0
mkdir -p /rory/rory-client-0/source
mkdir -p /rory/rory-client-0/sink
mkdir -p /rory/rory-client-0/log
mkdir -p /rory/keys
```

* **Datasets**: Place your target CSV files in the `source/` folder.
* **Encryption Keys**: Move your CKKS context and key files into the `keys/` folder.

### Executing the Client Service

The service uses Gunicorn to manage worker processes, which is essential for handling parallel encryption/decryption tasks efficiently.

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
    cd ../client/src
    gunicorn --chdir $CLIENT_PATH/src --config $CLIENT_PATH/src/gunicorn_config.py main:app
    # Alternatively, you can use the provided startup script:
    # ../run.sh
    ```

### Health Check & Verification

Once the service is running, verify the node's status and its identified role (Client) by performing a health check:

  ```bash
  curl -X GET http://localhost:3001/clustering/test
  ```

**Expected JSON Response:**
```json title="JSON Response"
{
  "component_type": "client",
  "status": "ready"
}
```


## Docker Containerization

The system is designed to run in containerized environments using Docker, ensuring that the complex cryptographic dependencies remain consistent across different nodes in the distributed network.

### Dockerfile Architecture

The Client includes a dedicated `Dockerfile` located in its root directory (`/client/Dockerfile`). The container has been created using a multi-layer approach to improve performance and deployment. Below is a breakdown of the environment configuration:

* **Base Image**: python:3.11-bullseye (chosen for its stability and support for complex C++ extensions).
* **Working Directory**: /app
* **Dependency Management**: Uses a high-timeout installation (1800s) to accommodate the compilation of heavy cryptographic libraries.


### Manual Image Construction

To build the Client image manually, you must navigate to the client's directory. The build process uses the local context to package the source code and configuration:

```bash
# Navigate to the client directory
cd /rory/client

# Build the image using the local Dockerfile
docker build -t rory:client -f Dockerfile .
```

`-t`: Defines the repository name and tag (e.g., `rory:client`).

`-f`: Specifies the `Dockerfile` located within the current directory.

`.`: Sets the build context to the current folder to include the `src` and `requirements.txt`.

### Automated Build Script

For a standardized deployment, the `build.sh` script automates the process by targeting the client folder structure dynamically.

```bash title="build.sh"
#!/bin/bash
readonly BASE_PATH=${1:-/rory}
readonly IMAGE=${2:-rory:client}

# The script targets the client component folder specifically
docker build -t ${IMAGE} ${BASE_PATH}/client/
```

**Usage Instructions**

The script accepts two optional positional arguments to generalize the construction:

* `BASE_PATH`: The root directory where the Rory project is located.
* `IMAGE`: The full name and tag for the resulting Docker image.

```bash
# Build the client image using default values (rory:client)
./build.sh

# Build with a custom image name and project path
./build.sh /home/sreyes/rory shanelreyes/rory:client-prod
```

### Orchestration with Docker Compose

The Client node can be orchestrated using `docker-compose.yml` to manage network identities, volumes for keys/logs, and environment variables.

**Network Dependency**

The Client requires the external mictlanx network to communicate with the CSS layer:
```bash
docker network create mictlanx
```

**Service Definition**

```yaml title="docker-compose.yml"
services:
  rory-client-0:
    image: shanelreyes/rory:client
    container_name: rory-client-0
    hostname: rory-client-0
    ports:
      - 3000:3000
    environment:
      - NODE_ID=rory-client-0
      - RORY_MANAGER_IP_ADDR=rory-manager-0
    volumes:
      - /rory/rory-client-0/source:/rory/source
      - /rory/rory-client-0/keys:/rory/keys
      - /rory/rory-client-0/log:/rory/log
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

Verify that the Client image and container are correctly initialized in the Docker engine:

```bash
# Check client image
docker images | grep rory:client

# Check running client container
docker ps | grep rory-client-0
```