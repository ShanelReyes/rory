
# Rory Client: Component Overview

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
    docker compose -f ./storage.yml down
    docker compose -f ./storage.yml up -d
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
```json
{
  "component_type": "client",
  "status": "ready"
}
```