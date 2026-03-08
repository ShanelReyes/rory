---
icon: material/xml
---
# 📚 Rory Client API Reference

Technical documentation for all clustering and classification endpoints available in the Rory Client.

---
## 🛰️ Clustering Algorithms
*Endpoints for privacy-preserving data grouping.*


### K-Means (Plaintext)
`POST` /clustering/kmeans

> **Note:** All attributes listed below must be passed exclusively via **HTTP Headers**. The request body should remain empty.

::: client.src.routes.clustering.kmeans
    options:
      show_root_heading: false
      show_root_toc_entry: false

### SK-Means (Liu)
`POST` /clustering/skmeans
> **Note:** All attributes listed below must be passed exclusively via **HTTP Headers**. The request body should remain empty.

::: client.src.routes.clustering.skmeans
    options:
      show_root_heading: false
      show_root_toc_entry: false

### PQC SK-Means (CKKS)
`POST` /clustering/pqc/skmeans
> **Note:** All attributes listed below must be passed exclusively via **HTTP Headers**. The request body should remain empty.

::: client.src.routes.clustering.pqc_skmeans
    options:
      show_root_heading: false
      show_root_toc_entry: false

### DBK-Means
`POST` /clustering/dbskmeans
> **Note:** All attributes listed below must be passed exclusively via **HTTP Headers**. The request body should remain empty.

::: client.src.routes.clustering.dbskmeans
    options:
      show_root_heading: false
      show_root_toc_entry: false

### PQC DBSK-Means
`POST` /clustering/pqc/dbskmeans
> **Note:** All attributes listed below must be passed exclusively via **HTTP Headers**. The request body should remain empty.

::: client.src.routes.clustering.pqc_dbskmeans
    options:
      show_root_heading: false
      show_root_toc_entry: false

### NNC (Plaintext)
`POST` /clustering/nnc
> **Note:** All attributes listed below must be passed exclusively via **HTTP Headers**. The request body should remain empty.

::: client.src.routes.clustering.nnc
    options:
      show_root_heading: false
      show_root_toc_entry: false

### DBSNNC
`POST` /clustering/dbsnnc
> **Note:** All attributes listed below must be passed exclusively via **HTTP Headers**. The request body should remain empty.

::: client.src.routes.clustering.dbsnnc
    options:
      show_root_heading: false
      show_root_toc_entry: false

### Clustering Health
`GET/POST` /clustering/test

::: client.src.routes.clustering.test
    options:
      show_root_heading: false
      show_root_toc_entry: false

---

## 📊 Classification Algorithms
*Endpoints for secure K-Nearest Neighbors classification.*

### KNN Training (Plaintext)
`POST` /classification/knn/train
> **Note:** All attributes listed below must be passed exclusively via **HTTP Headers**. The request body should remain empty.

::: client.src.routes.classification.knn_train
    options:
      show_root_heading: false
      show_root_toc_entry: false

### KNN Prediction (Plaintext)
`POST` /classification/knn/predict
> **Note:** All attributes listed below must be passed exclusively via **HTTP Headers**. The request body should remain empty.

::: client.src.routes.classification.knn_predict
    options:
      show_root_heading: false
      show_root_toc_entry: false

### SKNN Training (Liu)
`POST` /classification/sknn/train
> **Note:** All attributes listed below must be passed exclusively via **HTTP Headers**. The request body should remain empty.

::: client.src.routes.classification.sknn_train
    options:
      show_root_heading: false
      show_root_toc_entry: false

### SKNN Prediction (Interactive)
`POST` /classification/sknn/predict
> **Note:** All attributes listed below must be passed exclusively via **HTTP Headers**. The request body should remain empty.

::: client.src.routes.classification.sknn_predict
    options:
      show_root_heading: false
      show_root_toc_entry: false

### PQC SKNN Training (CKKS)
`POST` /classification/pqc/sknn/train
> **Note:** All attributes listed below must be passed exclusively via **HTTP Headers**. The request body should remain empty.

::: client.src.routes.classification.sknn_pqc_train
    options:
      show_root_heading: false
      show_root_toc_entry: false

### PQC SKNN Prediction (Interactive)
`POST` /classification/pqc/sknn/predict 
> **Note:** All attributes listed below must be passed exclusively via **HTTP Headers**. The request body should remain empty.

::: client.src.routes.classification.sknn_pqc_predict
    options:
      show_root_heading: false
      show_root_toc_entry: false


### Classification Health
`GET/POST` /classification/test

::: client.src.routes.classification.test
    options:
      show_root_heading: false
      show_root_toc_entry: false