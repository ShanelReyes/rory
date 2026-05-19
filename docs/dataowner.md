---
icon: lucide/flask-conical
---

# Rory Dataowner

The **Rory Dataowner** module is a batch experiment runner that automates the execution of data mining workloads across the Rory platform. Unlike the Client, Manager, and Worker components, it is **not a Flask web service** — it is a CLI script that processes a trace CSV file and submits experiments concurrently to the **Client** service via the `roryclient` library.

## Key Responsibilities

- **Trace Processing:** Reads a CSV file defining which algorithms to run, with what parameters, and how many iterations.
- **Concurrent Execution:** Uses a `ThreadPoolExecutor` to submit experiments to the Client service in parallel, with configurable inter-arrival delays.
- **Result Persistence:** Writes label vectors to `.npy` files in the sink directory and structured JSON logs for every experiment.
- **Retry Handling:** Automatically retries failed experiments up to `MAX_RETRIES` times.

## How It Works

1. The script loads a trace CSV and iterates over each row.
2. Each row is submitted `EXPERIMENT_ITERATION` times (default: 31) to a thread pool, with `INTERARRIVAL_TIME` seconds of delay between submissions.
3. The `ALGORITHM` column determines which Client endpoint is called (e.g., `KMEANS` → `client.kmeans()`).
4. Results (label vectors) are saved as `.npy` files in `SINK_PATH`.
5. Failed operations are collected and retried up to `MAX_RETRIES` times.

---

## Configuration & Environment

The Dataowner module is configured via environment variables, typically loaded from one of the per-algorithm `.env` files located in `dataowner/envs/`.

### Node & Network Configuration

| Variable | Description | Default Value |
|:---|:---|:---|
| `NODE_ID` | Unique identifier for the dataowner node. | `rory-dataowner-0` |
| `CLIENT_IP_ADDR` | IP address or hostname of the Client service. | `localhost` |
| `RORY_CLIENT_PORT` | Port of the Client service. | `3000` |
| `RORY_CLIENT_TIMEOUT` | HTTP timeout for Client requests (seconds). | `120` |

### Dataowner Parameters

| Variable | Description | Default Value |
|:---|:---|:---|
| `TRACE_ID` | Identifier for the trace file (used in naming). | `KMEANS` |
| `TRACE_EXTENSION` | File extension for the trace file. | `csv` |
| `TRACE_PATH` | Full path to the trace CSV. | `{SOURCE_PATH}/{TRACE_ID}.{TRACE_EXTENSION}` |
| `EXPERIMENT_ITERATION` | Number of times each trace record is repeated. | `31` |
| `MAX_RETRIES` | Maximum retry attempts for failed experiments. | `10` |
| `MAX_THREADS` | Number of concurrent threads in the pool. | `1` |
| `CLIENT_TIMEOUT` | Global timeout for experiment execution (seconds). | `300` |

### System Paths

| Variable | Description | Default Value |
|:---|:---|:---|
| `SOURCE_PATH` | Directory for input trace files. | `/rory/source` |
| `SINK_PATH` | Directory for output `.npy` label vectors. | `/rory/sink` |
| `LOG_PATH` | Directory for JSON log files. | `/rory/log` |
| `ENV_FILE_PATH` | Path to the `.env` configuration file. | `/home/sreyes/rory/dataowner/envs/.env-pplr` |

---

## Trace File Format

The trace is a CSV file where each row defines a single experiment configuration. The columns in the trace map directly to the HTTP headers sent to the Client service's algorithm endpoints.

### Essential Columns

| Column | Type | Description | Used By |
|:---|:---|:---|:---|
| `EXPERIMENT_ID` | int | Unique experiment identifier. | All |
| `ALGORITHM` | str | Algorithm to execute (e.g., `KMEANS`, `PPLR`). | All |
| `K` | int | Number of clusters. | Clustering |
| `DATASET_ID` | str | Unique dataset identifier in CSS. | All |
| `DATASET_FILENAME` | str | Local filename of the dataset. | All |
| `EXTENSION` | str | File extension (`csv`, `npy`). | All |
| `NUM_CHUNKS` | int | Number of chunks for matrix partitioning. | All |
| `INTERARRIVAL_TIME` | float | Sleep time between experiment submissions (seconds). | All |
| `MAX_ITERATIONS` | int | Maximum iterations for iterative algorithms. | SKMeans, DBSKMeans, PQC variants |
| `SENS` | float | Sensitivity parameter for FDHOPE-based protocols. | DBSKMeans, DBSKMeansPQC, DBSNNC |
| `THRESHOLD` | float | Distance threshold for NNC/DBSNNC. | NNC, DBSNNC |
| `MODEL_ID` | str | Model identifier. | Classification |
| `MODEL_FILENAME` | str | Model feature matrix filename. | Classification |
| `MODEL_LABELS_FILENAME` | str | Model labels filename. | Classification |
| `RECORD_TEST_ID` | str | Test records identifier. | Classification |
| `RECORD_TEST_FILENAME` | str | Test records filename. | Classification |
| `LABEL_VECTOR_TRAIN` | str | Label vector for training (ML). | LR, PPLR |
| `DATASET_TEST` | str | Test dataset identifier (ML). | LR, PPLR |
| `EPOCHS` | int | Training epochs. | LR, PPLR |
| `LEARNING_RATE` | float | Learning rate. | LR, PPLR |

### Supported Algorithms

| Algorithm | Task Category | Client Method |
|:---|:---|:---|
| `KMEANS` | Clustering | `client.kmeans()` |
| `SKMEANS` | Clustering | `client.skmeans()` |
| `DBSKMEANS` | Clustering | `client.dbskmeans()` |
| `SKMEANSPQC` | Clustering | `client.skmeans_pqc()` |
| `DBSKMEANSPQC` | Clustering | `client.dbskmeans_pqc()` |
| `NNC` | Clustering | `client.nnc()` |
| `DBSNNC` | Clustering | `client.dbsnnc()` |
| `KNN` | Classification | `client.knn()` |
| `SKNN` | Classification | `client.sknn()` |
| `SKNNPQC` | Classification | `client.sknn_pqc()` |
| `LOGISTICREGRESSION` | Machine Learning | `client.logistic_regression()` |
| `PPLR` | Machine Learning | `client.pplr()` |

### Example Trace

```csv title="trace.csv"
EXPERIMENT_ID,ALGORITHM,K,DATASET_ID,DATASET_FILENAME,EXTENSION,NUM_CHUNKS,INTERARRIVAL_TIME,MAX_ITERATIONS,SENS,THRESHOLD,MODEL_ID,MODEL_FILENAME,MODEL_LABELS_FILENAME,RECORD_TEST_ID,RECORD_TEST_FILENAME
1,KMEANS,3,kmeanstest01,dataset1_train,npy,2,1,3,0.00000000001,1.4,knnmodel01,classificationc0r10a5k20model,classificationc0r10a5k20modellabels,knndata01,classificationc0r10a5k20data
```

This trace will execute `KMEANS` with `K=3` on `dataset1_train.npy`, 31 times (default `EXPERIMENT_ITERATION`), with 1 second between each submission.

---

## Deployment

The Dataowner module is deployed via Docker Compose using the `deploy.sh` script. Before running, you must give execution permissions to the script and specify the `.env` file for the desired algorithm.

### Step 1: Grant Execution Permissions

```bash
cd rory/dataowner
chmod +x deploy.sh
```

### Step 2: Run with an Algorithm-Specific Env File

```bash
./deploy.sh ./envs/.env-kmeans
```

If no path is provided, the script defaults to `./envs/.env`.

### Available Environment Files

Each `.env` file in `dataowner/envs/` targets a specific algorithm with pre-configured timeouts and parameters:

| Env File | Algorithm | Task | Timeout |
|:---|:---|:---|:---|
| `.env-kmeans` | `KMEANS` | Clustering | 300s |
| `.env-skmeans` | `SKMEANS` | Clustering | 600s |
| `.env-dbskmeans` | `DBSKMEANS` | Clustering | 800s |
| `.env-skmeanspqc` | `SKMEANSPQC` | Clustering | 600s |
| `.env-dbskmeanspqc` | `DBSKMEANSPQC` | Clustering | 600s |
| `.env-nnc` | `NNC` | Clustering | 600s |
| `.env-dbsnnc` | `DBSNNC` | Clustering | 600s |
| `.env-knn` | `KNN` | Classification | 300s |
| `.env-sknn` | `SKNN` | Classification | 600s |
| `.env-sknnpqc` | `SKNNPQC` | Classification | 600s |
| `.env-lr` | `LOGISTICREGRESSION` | Machine Learning | 300s |
| `.env-pplr` | `PPLR` | Machine Learning | 300s |

---

## Docker Containerization

The Dataowner module is containerized to ensure consistent execution of experiment batches across environments.

### Dockerfile Architecture

The Dockerfile is located at `dataowner/Dockerfile`:

- **Base Image:** `python:3.10-slim`
- **Build Dependencies:** `g++`, `gcc`, `cmake`, `make`, `python3-dev`, `gfortran`, `libssl-dev`, `libgmp-dev`
- **Working Directory:** `/app`
- **Entrypoint:** `python -u /app/main.py` (unbuffered execution)

### Orchestration with Docker Compose

The service connects to the external `mictlanx` network and mounts host directories for trace input, result output, and logs.

**Network Dependency**

```bash
docker network create mictlanx
```

**Service Definition**

```yaml title="docker-compose.yml"
services:
  rory-dataowner-0:
    image: shanelreyes/rory:dataowner
    container_name: rory-dataowner-0
    hostname: rory-dataowner-0
    environment:
      - NODE_ID=rory-dataowner-0
      - TRACE_ID=KMEANS
      - CLIENT_IP_ADDR=rory-client-0
    volumes:
      - /rory/rory-dataowner-0/source:/rory/source
      - /rory/rory-dataowner-0/sink:/rory/sink
      - /rory/rory-dataowner-0/log:/rory/log
    networks:
      - mictlanx
```

### Production Deployment (Swarm / Elastic Server)

For Elastic Server or Docker Swarm deployments, use `docker-compose-server.yml` with `dataowner-elastic-server.yml`, which adds:

- Swarm deploy constraints (`node.labels.index==0`)
- Resource limits (5 CPUs, 10GB memory)
- Restart policy: `none` (run once and exit)

### Verification

Once the container runs, verify results are being produced:

```bash
# Check that the container started and processed the trace
docker ps -a | grep rory-dataowner-0

# Check output label vectors
ls /rory/rory-dataowner-0/sink/

# Check JSON logs
ls /rory/rory-dataowner-0/log/
```
