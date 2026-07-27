---
icon: lucide/rocket
---

# rory-common

**rory-common** is the storage abstraction layer for the Rory encrypted-analytics platform.
It provides a single `StorageBackend` interface that handles segmentation, homomorphic
encryption, and chunked upload/download — all through [mictlanx](https://jub-ecosystem.github.io/mictlanx-client).

## Architecture

```
StorageBuilder  ──►  StorageBackend  (put / put_from_file / get)
                            │
                         Common      (static helpers: segment, encrypt, serialize, I/O)
                            │
                    mictlanx.AsyncClient   (network layer)
```

`StorageBackend` is the only surface callers should use.
`Common` is an internal helper class — methods are called by the backend automatically.

## Supported encryption schemes

| Scheme | `Scheme` value | Description | Status |
|---|---|---|---|
| CKKS | `Scheme.CKKS` | Approximate HE via Pyfhel — initialized-executor pipeline, fully abstracted | Stable |
| Liu | `Scheme.LIU` | Symmetric additive HE | Stable |
| FDHOPE | `Scheme.FDHOPE` | FDHoPE chunk encryption for caller-computed UDM matrices; reads return merged `ndarray` chunks | Stable |
| Paillier | `Scheme.PAILLIER` | Probabilistic additive HE | **Not implemented yet** |

!!! note "Deprecated Liu helpers"
    The legacy `Common` helpers `segment_and_encrypt_liu` and
    `segment_and_encrypt_liu_with_executor` are deprecated — they emit a `DeprecationWarning`
    and will be removed in rory-common 1.0.0. Use `StorageBackend.put` with `Scheme.LIU`
    instead.

## Quick start

```python
from mictlanx import AsyncClient
from rorycommon import StorageBuilder, StorageParams, Scheme, CkksParams, LiuParams, FdhopeParams
from rory.core.security.cryptosystem.pqc.ckks import Ckks
from rory.core.security.dataowner import DataOwner
import numpy as np

ckks   = Ckks.from_pyfhel(_round=True, decimals=2, path="/rory/keys")
client = AsyncClient(uri="mictlanx://...", client_id="my-app")
matrix = np.random.random((64, 64))
```

=== "Fluent builder"

    ```python
    backend = (
        StorageBuilder(storage_client=client, scheme=Scheme.CKKS)
        .with_ckks(ckks)
        .with_storage_params(StorageParams(num_chunks=4, timeout=300))
        .build()
    )
    ```

=== "Full constructor"

    ```python
    backend = StorageBuilder(
        storage_client = client,
        scheme      = Scheme.CKKS,
        ckks           = ckks,
        ckks_params    = CkksParams(
            keys_path          = "/rory/keys",
            ctx_filename       = "ctx",
            pubkey_filename    = "pubkey",
            secretkey_filename = "secretkey",
            relinkey_filename  = "relinkey",
            rotatekey_filename = "rotatekey",
            decimals           = 2,
            _round             = True,
        ),
    ).build()
    ```

```python
# Upload plaintext
result = await backend.put(bucket_id="rory", ball_id="model_v1", data=matrix)

# Upload encrypted matrix (2-D)
result = await backend.put(bucket_id="rory", ball_id="model_v1_enc", data=matrix, encrypt=True)

# Upload encrypted vector (1-D) — automatically detected from ndim
vector = np.random.random((64,))
result = await backend.put(bucket_id="rory", ball_id="vector_v1", data=vector, encrypt=True)

# Pass a file path directly — extension is inferred, delegates to put_from_file
result = await backend.put(bucket_id="rory", ball_id="model_v2", data="/rory/data/model.npy")
result = await backend.put(bucket_id="rory", ball_id="model_v2_enc", data="/rory/data/model.npy", encrypt=True)

# Overwrite an existing object — delete before upload
result = await backend.put(bucket_id="rory", ball_id="model_v1_enc", data=matrix, encrypt=True, delete=True)

# Download — mirror the same flags used in put
result = await backend.get(bucket_id="rory", ball_id="model_v1_enc", encrypt=True)
ciphertexts = result.unwrap().raw_value   # List[PyCtxt]
```


## Error handling

All operations return `Result[T, Exception]` from the `option` library.

```python
result = await backend.get(bucket_id="rory", ball_id="model_v1")
if result.is_err:
    raise result.unwrap_err()
value = result.unwrap()           # GetResult[np.ndarray]
matrix = value.raw_value          # np.ndarray
```

## Advanced usage

### Forking a backend for a different scheme

`as_builder()` snapshots every field of a running backend (client, params, key filenames,
CKKS context, etc.) into a fresh `StorageBuilder`. Override only what differs with the
fluent `.with_*()` methods, then call `.build()`.

This is the recommended way to run the same workload under multiple schemes without
re-wiring the shared infrastructure:

```python
from rorycommon import StorageBuilder, StorageParams, Scheme

# base CKKS backend
ckks_backend = (
    StorageBuilder(storage_client=client, scheme=Scheme.CKKS)
    .with_ckks(ckks)
    .with_storage_params(StorageParams(num_chunks=4))
    .build()
)

# fork into a Liu backend — client and params are inherited
liu_backend = (
    ckks_backend.as_builder()
    .with_scheme(Scheme.LIU)
    .with_liu_params(LiuParams(security_level=128, decimals=2, _round=True))
    .build()
)

result = await liu_backend.put(bucket_id="rory", ball_id="model_liu", data=matrix, encrypt=True)
```

The same pattern applies to FDHOPE: switch to `Scheme.FDHOPE`, provide
`FdhopeParams`, and pass a caller-computed UDM ndarray to `put(..., encrypt=True)`.
The caller-side `get_U` API uses `algorithm="DBSKMEANS"` (or another caller-side algorithm value), while the backend FDHOPE config uses
`FdhopeParams.scheme`.

## Generating CKKS keys

Keys must exist on disk before any encrypted `put` or `get` call.
Run `scripts/keygen.py` once per key set you need.

| Argument | Default | Description |
|---|---|---|
| `--output-path` | *(required)* | Directory to write key files into |
| `--mode` | `default` | CKKS parameter preset (`default`, `ml`, …) |
| `--security-level` | `128` | Security level in bits — `128`, `192`, or `256` |
| `--decimals` | `5` | Decimal precision preserved after encryption |
| `--round` | off | Enable rounding |
| `--enable-relinearize` | off | Generate relinearization keys |
| `--enable-rotate` | off | Generate rotation keys |

**Example — `ml` keys:**

```bash
python scripts/keygen.py \
  --output-path /rory/keys/ml \
  --mode ml \
  --security-level 128 \
  --decimals 2 \
  --round \
  --enable-relinearize \
  --enable-rotate
```

## Configuration

Copy `.env.test` to `.env` and fill in the values for your environment before running any code or tests.

### Mictlanx client

| Variable | Default | Purpose |
|---|---|---|
| `MICTLANX_URI` | `mictlanx://mictlanx-router-0@localhost:63666?api_version=4&protocol=http` | Storage node URI |
| `MICTLANX_CLIENT_ID` | `rory-common_mictlanx` | Client identifier sent to the router |
| `MICTLANX_BUCKET_ID` | `rory` | Default bucket name |
| `MICTLANX_TIMEOUT` | `3600` | Request timeout in seconds |
| `MICTLANX_API_VERSION` | `4` | Mictlanx API version |
| `MICTLANX_MAX_WORKERS` | `12` | Worker threads inside `AsyncClient` |
| `MICTLANX_OUTPUT_PATH` | `/rory/mictlanx` | Local output directory for mictlanx |
| `MICTLANX_PROTOCOL` | `http` | Transport protocol (`http` or `https`) |
| `MICTLANX_DEBUG` | `0` | Enable mictlanx debug output (`0`/`1`) |

### CKKS / key files

| Variable | Default | Purpose |
|---|---|---|
| `RORY_KEYS_PATH` | `/rory/keys/test2` | Directory containing CKKS key files |
| `RORY_COMMON_CTX_FILENAME` | `ctx` | CKKS context file name |
| `RORY_COMMON_PUBKEY_FILENAME` | `pubkey` | CKKS public key file name |
| `RORY_COMMON_SECRETKEY_FILENAME` | `secretkey` | CKKS secret key file name |
| `RORY_COMMON_RELINKEY_FILENAME` | `relinkey` | CKKS relinearization key file name |
| `RORY_COMMON_ROTATEKEY_FILENAME` | `rotatekey` | CKKS rotation key file name |
| `RORY_COMMON_SECURITY_LEVEL` | `128` | CKKS security level in bits |
| `RORY_COMMON_CKKS_DECIMALS` | `2` | Decimal precision for CKKS (used by test fixtures) |
| `RORY_COMMON_CKKS_SECURITY_LEVEL` | `128` | Security level for CKKS (used by test fixtures) |
| `RORY_CKKS_MODE` | `ml` | CKKS parameter preset (`default`, `ml`, …) |
| `RORY_ENABLE_ROTATE_KEY_GENERATION` | `1` | Generate rotation keys during test setup (`0`/`1`) |
| `RORY_ENABLE_REALINEARIZATION_KEY_GENERATION` | `1` | Generate relinearization keys during test setup (`0`/`1`) |

### Process pool / workers

| Variable | Default | Purpose |
|---|---|---|
| `RORY_MAX_WORKERS` | `2` | `ProcessPoolExecutor` size for encryption tasks |
| `RORY_SOURCE_PATH` | `/rory/source` | Root path for source data files |
| `RORY_COMMON_ENV_FILE_PATH` | `./.env` | `.env` file loaded automatically at import time |

### Logging

| Variable | Default | Purpose |
|---|---|---|
| `RORY_COMMON_LOG_DISABLED` | `1` | Disable rory-common logging entirely (`0`/`1`) |
| `RORY_COMMON_LOG_PATH` | `./.rory/log` | Log file directory |
| `RORY_COMMON_LOG_CONSOLE_HANDLER_LEVEL` | `INFO` | Console log level (`DEBUG`, `INFO`, `WARNING`, …) |
| `RORY_COMMON_LOG_FILE_HANDLER_LEVEL` | `DEBUG` | File log level |
| `RORY_COMMON_LOG_TO_FILE` | `0` | Write logs to a rotating file (`0`/`1`) |
| `RORY_COMMON_LOG_ERROR_TO_FILE` | `0` | Write error logs to a separate file (`0`/`1`) |
| `RORY_COMMON_LOG_INTERVAL` | `60` | Log rotation interval, in units of `RORY_COMMON_LOG_WHEN` |
| `RORY_COMMON_LOG_WHEN` | `m` | Rotation time unit (`s` seconds, `m` minutes, `h` hours, `d` days) |
| `RORY_COMMON_LOG_RICH` | `0` | Use Rich console formatter (`0`/`1`) |
| `RORY_COMMON_LOG_JSON_INDENT` | `0` | JSON log indentation (`0` = minified) |
| `RORY_COMMON_LOG_MICTLANX_PROPAGATE` | `1` | Mirror rory-common log settings into mictlanx (`0`/`1`) |

!!! note "Mictlanx log propagation"
    When `RORY_COMMON_LOG_MICTLANX_PROPAGATE=1`, rory-common automatically sets the following
    mictlanx variables via `os.environ.setdefault` (they can still be overridden individually in your `.env`):
    `MICTLANX_LOG_DISABLED`, `MICTLANX_LOG_LEVEL`, `MICTLANX_LOG_RICH`, `MICTLANX_LOG_TO_FILE`,
    `MICTLANX_LOG_ERROR_FILE`, `MICTLANX_LOG_ROTATION_WHEN`, `MICTLANX_LOG_ROTATION_INTERVAL`.

## Running tests

Integration tests require a running mictlanx instance and pre-generated CKKS keys.

```bash
# Copy and fill in the test environment file
cp .env.test.example .env.test

# Start a local mictlanx cluster
bash deploy_storage.sh

# Generate CKKS keys — see Generating CKKS keys above
python scripts/keygen.py --output-path /rory/keys --mode default

# Run tests
cd tests && pytest test_new_api.py test_storage_backend.py -v
```
