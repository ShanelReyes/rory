---
icon: lucide/home
---

# Introduction
<!-- ## Rory:Post-Quantum Privacy-Preserving Data Mining as a Service -->

<p align="center">
  <img src="/rory_client/images/logo_rory.svg" alt="Logo" width="50%">

  <!-- ![License](https://img.shields.io/badge/license-MIT-blue.svg) -->
  <!-- ![Python](https://img.shields.io/badge/python-3.9%2B-blue) -->
  <!-- ![Build](https://img.shields.io/badge/build-passing-brightgreen) -->
</p>



<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue">
  <img src="https://img.shields.io/badge/build-passing-brightgreen">


</p>

Rory is an elastic, cloud-native platform designed to facilitate secure, distributed data mining on encrypted data. It ensures data privacy using standard, differentially private, and post-quantum cryptographic (PQC) methods without requiring data decryption during the analysis process.

## ⭐ Key Features

* **Privacy-Preserving Data Mining (PPDM):** Execute algorithms like Secure K-Means and Secure KNN directly on encrypted datasets.
* **Post-Quantum Security:** Integrated support for PQC to future-proof data privacy against quantum computing threats.
* **Elastic Architecture:** Dynamic scale-in and scale-out of worker nodes based on real-time processing demand.
* **Distributed Processing:** Automated data segmentation, load balancing, and secure transport pipelines via a Cloud Storage System (CSS).

## ⚙️ Architecture

The platform maps traditional Data Owner (DO) and Service Provider (SP) entities into three core components:

* **Client:** Prepares, encrypts, and segments data for externalization.

* **Manager:** Orchestrates PPDM tasks, manages storage buckets, and handles load balancing.

* **Worker:** The execution engine that applies PPDM algorithms to encrypted chunks.

For a detailed breakdown of the system components and the elastic interaction model, please refer to the [Architecture Documentation](architecture.md).

<!-- ## 🚀 Quick Start

### Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/yourusername/rory.git](https://github.com/yourusername/rory.git)
cd rory
pip install -r requirements.txt
``` -->