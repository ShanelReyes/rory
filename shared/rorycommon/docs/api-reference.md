---
icon: lucide/book-open
---

# API Reference

This reference is organized around the **public API first**. Start with
`StorageBackend`, `StorageBuilder`, and the result models you receive back from
`put()` and `get()`. The `Common` section is kept later on the page because it
documents internal helpers rather than the main user-facing surface.

## Public API map

| Area | What it covers |
|---|---|
| `StorageBackend` | The main runtime API for `put`, `put_from_file`, `get`, and backend cloning via `as_builder()` |
| Result models | `PutPlaintextResult`, `PutCiphertextResult`, `GetResult[T]`, and `SourceType` |
| Configuration | `StorageBuilder`, `StorageParams`, scheme-specific params, and `Scheme` |
| Internal helpers | `Common` methods used by the backend internally |


## Building and configuring a backend

### StorageBuilder

::: rorycommon.StorageBuilder
    options:
      show_root_heading: false 
      show_root_toc_entry: false 
      heading_level: 4
      members:
        - __init__
        - with_ckks
        - with_ckks_params
        - with_fdhope_params
        - with_liu_params
        - with_scheme
        - with_storage_params
        - build

### StorageParams


::: rorycommon.StorageParams
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 4

### CkksParams
::: rorycommon.CkksParams
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 4
### LiuParams
::: rorycommon.LiuParams
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 4
### FdhopeParams

::: rorycommon.FdhopeParams
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 4
### Scheme
::: rorycommon.Scheme
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 4

## StorageBackend

`StorageBackend` is the primary interface for storing and retrieving data.
Construct it with `StorageBuilder` and treat `put()` / `get()` as the main
entry points.

### Dispatch rules for `put()`

`put` and `get` accept boolean flags that control segmentation, encryption, and
pre-upload deletion. Always use the **same `encrypt` and `segment` flags** in
the matching `get()` call that you used in `put()`.

| `data` type | `encrypt` | `segment` | `data.ndim` | scheme | What happens |
|---|---|---|---|---|---|
| `str` (file path) | any | any | - | any | Extension derived from the path suffix; delegates to `put_from_file` |
| `List[int]` / `List[float]` | any | any | - | any | Auto-converted to a 1-D `float64` ndarray, then follows the ndarray rows below |
| `ndarray` | `False` | `False` | any | any | Single plaintext blob |
| `ndarray` | `False` | `True` | any | any | Split into `num_chunks` plaintext chunks |
| `ndarray` | `True` | - | 1 | CKKS | Encrypt the whole vector as one ciphertext chunk |
| `ndarray` | `True` | - | >=2 | CKKS | CKKS-encrypt chunked matrix data |
| `ndarray` | `True` | - | any | LIU | Liu-encrypt each chunk via an initialized process pool |
| `ndarray` | `True` | - | any | FDHOPE | Treat input as caller-computed UDM, FDHoPE-encrypt each chunk, then upload |

### What `get()` returns

`get()` always returns `Result[GetResult[T], Exception]`. The concrete type in
`GetResult.raw_value` depends on the storage path that was used:

| `encrypt` | `segment` | scheme | `GetResult.raw_value` |
|---|---|---|---|
| `True` | - | CKKS | `List[PyCtxt]` |
| `True` | - | LIU | `np.ndarray` |
| `True` | - | FDHOPE | `np.ndarray` |
| `False` | `True` | any | `np.ndarray` |
| `False` | `False` | any | `np.ndarray` |

<!-- ### FDHOPE contract

FDHOPE follows the same `StorageBuilder` / `StorageBackend` public pattern as
CKKS and LIU, with one important boundary: the backend does **not** compute
`get_U` for you. Callers must build the UDM first, then pass that ndarray to
`put(..., encrypt=True)` on a backend configured with `Scheme.FDHOPE`.

On the caller side, `get_U` uses its own `algorithm=...` parameter; that is
separate from `FdhopeParams.scheme`, which configures the backend's FDHOPE
chunk-encryption step.

For reads, `get(..., encrypt=True)` on an FDHOPE backend uses the generic
`get_and_merge(...)` path and returns a merged `ndarray`. It does not run a
separate FDHOPE reconstruction step or a CKKS-style ciphertext loader. -->

### Delete before put (`delete=True`)

Pass `delete=True` to `put()` or `put_from_file()` to remove any existing
object at `ball_id` before uploading. This is safe when the key does not yet
exist: the delete step completes and the upload continues normally.

### Methods
::: rorycommon.StorageBackend
    options:
      show_root_heading: false 
      show_root_toc_entry: false 
      heading_level: 4
      members:
        - put
        - put_from_file
        - get
        - as_builder

---

## Result models

All public read/write operations return `Result[T, Exception]` from the
`option` library. On success, `.unwrap()` yields one of the result models
below.

```python
result = await backend.get(bucket_id="rory", ball_id="model_v1")
if result.is_err:
    raise result.unwrap_err()

value = result.unwrap()     # GetResult[np.ndarray] or GetResult[List[PyCtxt]]
payload = value.raw_value
```

### PutPlaintextResult

Returned when an upload finishes **without encryption**. That includes:

1. Plain single-blob uploads
2. Plain segmented uploads
3. File-based plaintext uploads

<!-- | Field | Meaning |
|---|---|
| `path` | Source path when the upload came from disk. For in-memory uploads, current code paths leave this empty or unset. |
| `extension` | Source file extension such as `"npy"` or `"csv"`. Empty when the upload did not originate from a file. |
| `bucket_id` | Bucket the object was written into. |
| `ball_id` | Object key inside the bucket. |
| `tags` | Metadata tags sent with the upload. |
| `shape` | Shape of the uploaded payload. For normal ndarray uploads this is the original array shape. |
| `dtype` | NumPy dtype of the uploaded plaintext payload when available. |
| `read_time` | Time spent reading from disk before upload. Usually `0.0` for in-memory uploads. |
| `segment_time` | Time spent splitting plaintext data into chunks. `0.0` when no segmentation was needed. |
| `upload_time` | Time spent writing the payload to storage. | -->

Use this model when you want upload metadata but **no encryption timing**.

::: rorycommon.PutPlaintextResult
    options:
      show_root_heading: false 
      show_root_toc_entry: false 
      heading_level: 4


### PutCiphertextResult

Returned when an upload **encrypts data before writing**. It carries the same
core metadata as `PutPlaintextResult`, plus a separate encryption duration.

<!-- | Field | Meaning |
|---|---|
| `path` | Source path when the encrypted upload started from a file. |
| `extension` | Source file extension when applicable. |
| `bucket_id` | Bucket the ciphertext chunks were written into. |
| `ball_id` | Object key inside the bucket. |
| `tags` | Metadata tags sent with the upload. |
| `shape` | Shape of the plaintext input matrix or vector that was encrypted. |
| `dtype` | Dtype of the plaintext input before encryption. |
| `read_time` | Time spent loading the source data before encryption. Usually `0.0` for in-memory inputs. |
| `segment_time` | Time spent preparing chunks prior to or alongside encryption. |
| `encrypt_time` | Time spent running the encryption step itself. This is the main field that differentiates this model from `PutPlaintextResult`. |
| `upload_time` | Time spent uploading the encrypted chunks. | -->

Use this model when you need to inspect encryption overhead separately from
upload cost.

::: rorycommon.PutCiphertextResult
    options:
      show_root_heading: false 
      show_root_toc_entry: false 
      heading_level: 4

### GetResult[T]

Returned by read operations. `GetResult` wraps the retrieved payload and keeps
provenance metadata alongside it.

<!-- | Field | Meaning |
|---|---|
| `source` | Where the payload came from. In the current cloud retrieval APIs this is `SourceType.CLOUD`. |
| `raw_value` | The retrieved payload itself. This is polymorphic: `np.ndarray` for plaintext, LIU, FDHOPE, and segmented reads; `List[PyCtxt]` for CKKS encrypted reads. |
| `read_time` | Download duration when the specific retrieval path populates it. Lower-level helpers such as `Common.from_cloud_storage_to_matrix()` set it; `StorageBackend.get()` currently returns `None` here. | -->


::: rorycommon.GetResult
    options:
      show_root_heading: false 
      show_root_toc_entry: false 
      heading_level: 4


#### `raw_value` by retrieval path

| Retrieval path | `raw_value` shape/type |
|---|---|
| Plain single object | `np.ndarray` |
| Plain segmented object | `np.ndarray` |
| LIU encrypted object | `np.ndarray` |
| FDHOPE encrypted object | `np.ndarray` |
| CKKS encrypted object | `List[PyCtxt]` |

#### Conversion helpers

`GetResult` also exposes convenience helpers for callers that want an explicit
typed view of `raw_value`:

| Helper | Returns |
|---|---|
| `to_nd_array()` | `np.ndarray` or `None` |
| `to_list()` | Any list payload or `None` |
| `to_pyctxt_list()` | `List[PyCtxt]` or `None` |
| `to_bytes_list()` | `List[bytes]` or `None` |
| `to_float_list()` | `List[float]` or `None` |
| `to_int_list()` | `List[int]` or `None` |


### SourceType
::: rorycommon.SourceType
    options:
      show_root_heading: false 
      show_root_toc_entry: false 
      heading_level: 4

<!-- `SourceType` describes the provenance recorded in `GetResult.source`. -->

<!-- | Value | Meaning |
|---|---|
| `FILE` | Data originated from a file source. |
| `URL` | Data originated from a URL or HTTP source. |
| `DATABASE` | Data originated from a database source. |
| `IN_MEMORY` | Data originated from memory rather than an external source. |
| `CLOUD` | Data came from cloud storage. This is the value currently returned by the public cloud retrieval helpers in this library. |
| `OTHER` | Fallback for sources that do not fit the categories above. | -->

In practice, when you call `StorageBackend.get()` today, you should expect
`GetResult.source == SourceType.CLOUD`.


<!-- --- -->

---

## Internal helpers (`Common`)

!!! note
    `Common` is not part of the public API. `StorageBackend` calls these
    methods internally. This section is kept for contributors and advanced
    users who need to understand the lower-level implementation pieces.

### Plaintext

::: rorycommon.Common
    options:
      members:
        - from_matrix_to_cloud_storage
        - from_matrix_on_disk_to_cloud_storage
        - from_cloud_storage_to_matrix

### CKKS

::: rorycommon.Common
    options:
      members:
        - from_matrix_to_cloud_storage_ckks
        - from_matrix_on_disk_to_cloud_storage_ckks
        - from_vector_to_cloud_storage_ckks
        - from_vector_on_disk_to_cloud_storage_ckks
        - segment_and_encrypt_ckks_with_initialized_executor
        - encrypt_vector_ckks_with_initialized_executor

### Liu

The preferred path is `StorageBackend.put` with `Scheme.LIU`, which uses the
initialized-executor pipeline below. The `segment_and_encrypt_liu` and
`segment_and_encrypt_liu_with_executor` helpers are deprecated and will be
removed in **rory-common 1.0.0**.

::: rorycommon.Common
    options:
      members:
        - init_liu_worker_context
        - encrypt_chunk_liu_with_initialized_executor
        - segment_and_encrypt_liu_with_initialized_executor_timed
        - segment_and_encrypt_liu_timed
        - segment_and_encrypt_liu_with_executor_timed
        - segment_and_encrypt_liu
        - segment_and_encrypt_liu_with_executor

### FDHOPE

The preferred path is `StorageBackend.put` with `Scheme.FDHOPE`, using
caller-computed UDM input plus the initialized-executor FDHOPE pipeline below.
`StorageBackend.get(..., encrypt=True)` reads FDHOPE data back through the
generic `get_and_merge(...)` path.

::: rorycommon.Common
    options:
      members:
        - init_fdhope_worker_context
        - encrypt_chunk_fdhope_with_initialized_executor
        - segment_and_encrypt_fdhope_with_initialized_executor_timed

### Retrieval

::: rorycommon.Common
    options:
      members:
        - get_matrix_or_error
        - get_and_merge
        - get_pyctxt
        - get_by_chunk_index

### Low-level I/O

::: rorycommon.Common
    options:
      members:
        - put_ndarray
        - put_chunks
        - from_pyctxts_to_chunks

### Serialization

::: rorycommon.Common
    options:
      members:
        - serialize_matrix_with_pickle
        - deserialize_matrix_with_pickle
        - from_pyctxt_list_to_bytes
        - from_bytes_to_pyctxt_list
