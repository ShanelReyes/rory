---
icon: lucide/rocket
---
<p align="center">
  <img src="/rory/images/logo_rory.svg" alt="Logo" width="50%">

</p>

# System Overview

Rory is a distributed and privacy-preserving data mining system designed to execute clustering and classification tasks over encrypted datasets. The architecture leverages Post-Quantum Cryptography (PQC) and Homomorphic Encryption (HE) to ensure data confidentiality throughout the entire mining lifecycle.

## Architecture

The system follows a decentralized orchestration model composed of four primary layers:

* **Orchestration Layer (Manager)**: Acts as the brain of the system, managing worker registration, load balancing, and task distribution.
* **Computational Layer (Workers)**: High-performance nodes that execute the data mining algorithms (SKMeans, DBSKMeans, SKNN) on encrypted data.
* **Storage Layer (CSS/Mictlanx):** A distributed and asynchronous storage service that handles data fragmentation and persistence.
* **Interaction Layer (Client)**: The entry point for users to upload datasets, configure mining parameters, and retrieve results.

## Quick Start

Follow these steps to deploy the complete ecosystem (Storage + Rory Nodes) using Docker.

1. **Initialize the Storage Layer (Mictlanx)**

    Navigate to the Mictlanx directory and start the CSS routers:

    ```bash
    cd rory/mictlanx
    ```

    **Troubleshooting (Connection Refused):** If you encounter the error `Failed to connect to localhost port 63666`, it means the default port is occupied or blocked. Resolve this by starting the service on an alternative port (e.g., 64666):

    ```bash
    ./run.sh .env.mictlanx.dev 64666
    ```

2. **Deploy Rory Ecosystem**

    Return to the project root and launch the Client, Manager, and Worker nodes:

    ```bash
    cd /Rory
    docker compose -f ./docker-compose.yml up --build
    ```

This command will build the images and orchestrate the three main nodes, linking them automatically to the Mictlanx storage network.


## Security & Privacy Model

Rory integrates cryptographic schemes to maintain privacy:

* **Liu Scheme**: A conventional homomorphic encryption scheme used for secure distance approximation.
* **CKKS**: A post-quantum homomorphic encryption scheme that allows performing arithmetic operations directly on encrypted floating-point numbers.
* **FDHOPE**: An order-preserving encryption (OPE) scheme used to maintain data sorting and comparison capabilities without decryption.


## General Workflow
The interaction between components follows a structured protocol to ensure efficiency and privacy. The following diagram illustrates the dynamic communication during a mining task:

<p align="center">
  <img src="/rory/images/interactions.svg" alt="Interactions" width="60%">
</p>


**Interaction Steps:**

* **Worker Request**: The `Client` sends a request to the `Manager` to assign an available `Worker` for a specific task.
* **Worker Assignment**: The `Manager` identifies a suitable `Worker` based on load-balancing algorithms and returns the `Worker ID/Address` to the `Client`.
* **Direct Communication**: The `Client` establishes a direct link with the assigned `Worker` to transmit encrypted task parameters or data references.
* **Iterative Processing**: For algorithms requiring multiple steps (like SKMeans), the `Worker` processes data and returns partial results to the `Client`. This loop repeats for `n` iterations until convergence or task completion.


## Technology Stack

* **Language**: Python 3.11+
* **Framework**: Flask (Microservices)
* **Containerization**: Docker & Docker Swarm
* **WSGI Server**: Gunicorn
* **Storage**: Mictlanx (Distributed CSS)
* **Crypto Libraries**: Pyfhel (CKKS implementation)