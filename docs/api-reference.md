---
icon: material/xml
---

# API Reference

Comprehensive technical documentation for the Rory platform, detailing the endpoints for the Client, Manager, and Worker components. This reference covers secure data mining protocols using homomorphic (Liu, CKKS) and order-preserving (FDHOPE) encryption schemes.

---

## Rory Client
The **Client** acts as the **Data Owner (DO)** and the primary **Decryption Authority**. It is responsible for local data preparation, cryptographic key management, and participating as a secure decryption authority in interactive privacy-preserving protocols.

### Route Summary

#### Clustering
| Method | Route | Function | Description |
|:---:|:---|:---|:---|
| `POST` | `/clustering/kmeans` | `kmeans` | Distributed K-Means on plaintext data. |
| `POST` | `/clustering/skmeans` | `skmeans` | Secure K-Means using the **Liu** scheme. |
| `POST` | `/clustering/dbskmeans` | `dbskmeans` | Double-blind Secure K-Means protocol. |
| `POST` | `/clustering/pqc_skmeans` | `pqc_skmeans` | Post-Quantum Secure K-Means (**CKKS**). |
| `POST` | `/clustering/nnc` | `nnc` | Nearest Neighbor Clustering execution. |
| `POST` | `/clustering/dbsnnc` | `dbsnnc` | Double-blind Nearest Neighbor Clustering. |
| `POST` | `/clustering/pqc_dbskmeans` | `pqc_dbskmeans` | Post-Quantum Double-blind K-Means. |

#### Classification
| Method | Route | Function | Description |
|:---:|:---|:---|:---|
| `POST` | `/classification/knn_train` | `knn_train` | Local training of the KNN model. |
| `POST` | `/classification/knn_predict` | `knn_predict` | Standard KNN prediction (Plaintext). |
| `POST` | `/classification/sknn_train` | `sknn_train` | Secure KNN weight training protocol. |
| `POST` | `/classification/sknn_predict` | `sknn_predict` | Secure KNN prediction (**FDHOPE**). |
| `POST` | `/classification/sknn_pqc_train` | `sknn_pqc_train` | Post-Quantum Secure KNN training. |
| `POST` | `/classification/sknn_pqc_predict` | `sknn_pqc_predict` | Post-Quantum Secure KNN prediction. |

### Clustering (Client-side)
Endpoints to initiate secure data grouping. These routes handle local encryption and the externalization of datasets to the CSS.
::: client.src.routes.clustering
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 4


### Classification (Client-side)
Endpoints for secure K-Nearest Neighbors (KNN) operations, managing the transformation between different encryption domains.
::: client.src.routes.classification
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 4

---

## Rory Manager
The **Manager** is the central **Orchestrator** of the PPDMaaS ecosystem. Its primary function is to manage the lifecycle of mining tasks, perform load balancing, and assign available Workers to specific experimental requests while maintaining an audit trail of service times.

### Route Summary
| Method | Route | Function | Description |
|:---:|:---|:---|:---|
| `POST` | `/workers/started` | `started` | Registers a new active Worker node. |
| `GET` | `/workers` | `getAll` | Retrieves metadata of all registered nodes. |
| `POST` | `/workers/deploy` | `deploy_worker` | Dynamically deploys a new Worker container. |

### Orchestration & Management
Endpoints for task status tracking, worker registration, and resource allocation.
::: manager.src.routes.workers
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 4

---

## Rory Worker
The **Worker** nodes represent the **Cloud Computing Power**. They perform high-performance data mining directly on encrypted data (ciphertext). They operate in a "zero-knowledge" state regarding the underlying plaintext, returning only encrypted intermediate or final results.

### Route Summary

#### Clustering
| Method | Route | Function | Description |
|:---:|:---|:---|:---|
| `GET/POST` | `/test` | `test` | Verifies node connectivity and role identification. |
| `POST` | `/kmeans` | `kmeans` | Standard K-Means execution on plaintext data. |
| `POST` | `/skmeans` | `skmeans` | Interactive Secure K-Means (Liu Scheme). |
| `POST` | `/dbskmeans` | `dbskmeans` | Interactive Double-Blind Secure K-Means. |
| `POST` | `/nnc` | `nnc` | Standard Nearest Neighbor Clustering. |
| `POST` | `/dbsnnc` | `dbsnnc` | Double-Blind Nearest Neighbor Clustering. |
| `POST` | `/pqc/skmeans` | `pqc_skmeans` | Post-Quantum Secure K-Means (**CKKS**). |
| `POST` | `/pqc/dbskmeans` | `pqc_dbskmeans` | Post-Quantum Double-Blind K-Means (**CKKS**). |

#### Classification
| Method | Route | Function | Description |
|:---:|:---|:---|:---|
| `POST` | `/knn/predict` | `knn_predict` | Standard KNN classification on plaintext data. |
| `POST` | `/sknn/predict` | `sknn_predict` | Interactive Secure KNN using the **FDHOPE** scheme. |
| `POST` | `/pqc/sknn/predict` | `sknn_pqc_predict` | Post-Quantum Secure KNN using the **CKKS** lattice-based scheme. |

### Clustering (Worker-side)
Execution logic for distributed K-Means, DBS K-Means, and NNC algorithms over encrypted matrices.
::: worker.src.routes.clustering
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 4

### Classification (Worker-side)
Endpoints for processing secure classification queries, utilizing FDHOPE for privacy-preserving distance comparisons.
::: worker.src.routes.classification
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 4