# Basic Usage

## Installation

Rory Core uses [Poetry](https://python-poetry.org/) for dependency management.

```bash
poetry install                 # core dependencies
poetry install -E pqc          # with CKKS/PQC support
```

Requirements: Python >= 3.10

## Key Generation

Before using CKKS-based algorithms, generate encryption keys:

```bash
poetry run python3 scripts/keygen.py --scheme=CKKS --mode=default \
  --output-path=/rory/keys/keys128 --security-level=128 --decimals=2 \
  --enable-relinearize --enable-rotate
```

Keys are saved to the specified output path and loaded automatically by the
`Ckks` class.

## Quick Example: Plaintext Clustering

```python
import numpy as np
from rory.core.clustering.kmeans import kmeans

# Generate sample data
data = np.random.rand(100, 5).astype(np.float64)

# Run KMeans with k=3
result = kmeans(data, k=3)
print(result.label_vector)  # cluster assignments
```

## Quick Example: Secure KNN (CKKS)

```python
from rory.core.security.cryptosystem.pqc.ckks import Ckks

# Load CKKS client from pre-generated keys
ckks = Ckks.from_pyfhel(path="/rory/keys/keys128")

# Encrypt a vector
plaintext = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
ciphertext = ckks.encryptVector(plaintext)

# Decrypt
decrypted = ckks.decryptVector(ciphertext)
```

## Quick Example: Secure KMeans (CKKS)

```python
from rory.core.security.cryptosystem.pqc.ckks import Ckks
from rory.core.security.pqc.dataowner import DataOwner
from rory.core.clustering.secure.pqc.skmeans import Skmeans
from rory.core.utils.constants import Constants

# Load keys and prepare data
ckks = Ckks.from_pyfhel_client(path="/rory/keys/keys128")
owner = DataOwner(scheme=ckks)
data = np.random.rand(50, 4).astype(np.float64)

# Externalize data
result = owner.outsourcedData(
    data, threshold=0.01,
    algorithm=Constants.ClusteringAlgorithms.SKMEANS_PQC
)

# Run secure clustering
skmeans = Skmeans(he_object=ckks.he_object,
                  init_shiftmatrix=ckks.he_object.encryptFrac(np.zeros(4)))
labels = skmeans.fit(
    status=Constants.ClusteringStatus.START,
    k=3,
    encrypted_matrix=result.encrypted_matrix,
    UDM=result.UDM,
    num_attributes=4,
    Cent_j=[],
    iterations=10,
    n_iterations=0,
    scheme=ckks
)
```

## Testing

```bash
# Run all tests
poetry run pytest -v -s

# Run a single test file
poetry run pytest tests/test_ckks.py -v -s

# With coverage
poetry run coverage run -m pytest -v -s
```

Tests require pre-generated CKKS keys. See [Key Generation](#key-generation) above.
