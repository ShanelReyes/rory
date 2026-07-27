---
icon: lucide/table-properties
---

# Storage Compatibility

This page summarizes which input types `StorageBackend` accepts, which schemes handle
them, and what comes back from `get`.

Keep this page as the quick reference for:

- `np.ndarray` inputs
- `List[PyCtxt]`
- `Chunks`
- scheme-specific encrypted paths
- scheme-agnostic plaintext and merged-chunk paths

## Overview

| Input type | Typical shape / meaning | Compatible schemes | `put` behavior | `get` result |
|---|---|---|---|---|
| `np.ndarray` | 1-D numeric array (`int`/`float`) | Plaintext: any scheme; encrypted: CKKS, LIU | Plaintext single blob or segmented chunks; CKKS has a vector encrypt path; LIU uses generic encrypted ndarray path | Plaintext: `np.ndarray`; CKKS encrypted: `List[PyCtxt]`; LIU encrypted: merged `np.ndarray` |
| `np.ndarray` | 2-D numeric matrix (`int`/`float`) | Plaintext: any scheme; encrypted: CKKS, LIU, FDHOPE | Plaintext single blob or segmented chunks; CKKS matrix encrypt path; LIU generic encrypted ndarray path; FDHOPE encrypted path for caller-computed UDM matrices | Plaintext: `np.ndarray`; CKKS encrypted: `List[PyCtxt]`; LIU encrypted: merged `np.ndarray`; FDHOPE encrypted: merged `np.ndarray` |
| `List[PyCtxt]` | Pre-encrypted CKKS ciphertext list | CKKS only | Serialized to chunks and uploaded | `List[PyCtxt]` when retrieved through encrypted CKKS `get` |
| `Chunks` | Caller-prepared chunk stream | Scheme-agnostic manual path | Uploaded directly without scheme-specific encryption | Usually retrieved through merged-chunk flow as `np.ndarray` |

## CKKS

### Compatible types

- 1-D numeric `np.ndarray`
- 2-D numeric `np.ndarray`
- `List[PyCtxt]`
- plaintext `Chunks`

### CKKS behavior

| Input | `put` flags | Behavior | `get` result |
|---|---|---|---|
| 1-D `np.ndarray` | `encrypt=False, segment=False` | Store as a single plaintext blob | `np.ndarray` |
| 1-D `np.ndarray` | `encrypt=False, segment=True` | Split into plaintext chunks | merged `np.ndarray` |
| 1-D `np.ndarray` | `encrypt=True` | Use the CKKS vector encryption path | `List[PyCtxt]` |
| 2-D `np.ndarray` | `encrypt=False, segment=False` | Store as a single plaintext blob | `np.ndarray` |
| 2-D `np.ndarray` | `encrypt=False, segment=True` | Split into plaintext chunks | merged `np.ndarray` |
| 2-D `np.ndarray` | `encrypt=True` | Use the CKKS matrix encryption path | `List[PyCtxt]` |
| `List[PyCtxt]` | `encrypt=False` | Convert ciphertexts to chunks and upload | `List[PyCtxt]` |
| `Chunks` | `encrypt=False` | Upload chunks directly | merged `np.ndarray` in the usual chunk retrieval path |

### Notes

- `List[PyCtxt]` is a CKKS-only type.
- CKKS is the only scheme whose encrypted `get(..., encrypt=True)` returns
  `List[PyCtxt]` instead of a merged ndarray.

## LIU

### Compatible types

- 1-D numeric `np.ndarray`
- 2-D numeric `np.ndarray`
- plaintext `Chunks`

### LIU behavior

| Input | `put` flags | Behavior | `get` result |
|---|---|---|---|
| 1-D `np.ndarray` | `encrypt=False, segment=False` | Store as a single plaintext blob | `np.ndarray` |
| 1-D `np.ndarray` | `encrypt=False, segment=True` | Split into plaintext chunks | merged `np.ndarray` |
| 1-D `np.ndarray` | `encrypt=True` | Use the LIU encrypted ndarray path | merged `np.ndarray` |
| 2-D `np.ndarray` | `encrypt=False, segment=False` | Store as a single plaintext blob | `np.ndarray` |
| 2-D `np.ndarray` | `encrypt=False, segment=True` | Split into plaintext chunks | merged `np.ndarray` |
| 2-D `np.ndarray` | `encrypt=True` | Use the LIU encrypted ndarray path | merged `np.ndarray` |
| `Chunks` | `encrypt=False` | Upload chunks directly | merged `np.ndarray` in the usual chunk retrieval path |

### Notes

- LIU does not have a separate public vector type like `List[PyCtxt]`.
- Encrypted LIU reads use merged ndarray retrieval, not a ciphertext list loader.

## FDHOPE

### Compatible types

- caller-computed UDM `np.ndarray`
- plaintext `Chunks`

### FDHOPE behavior

| Input | `put` flags | Behavior | `get` result |
|---|---|---|---|
| `np.ndarray` | `encrypt=True` | Use the FDHOPE segment-and-encrypt upload path | merged `np.ndarray` |
|`np.ndarray` | `encrypt=False, segment=False` | Plaintext single-blob storage still follows the scheme-agnostic ndarray path | `np.ndarray` |
| `np.ndarray` | `encrypt=False, segment=True` | Plaintext chunked storage still follows the scheme-agnostic chunk path | merged `np.ndarray` |
| `Chunks` | `encrypt=False` | Upload chunks directly | merged `np.ndarray` in the usual chunk retrieval path |

### Notes

- FDHOPE encrypted `put` expects the caller to prepare the UDM first.
- The caller-side `get_U(...)` API uses its own `algorithm=...` parameter.
- The backend-side FDHOPE configuration uses `FdhopeParams.scheme`.
- FDHOPE encrypted `get(..., encrypt=True)` uses the generic `get_and_merge(...)`
  path and returns a merged `np.ndarray`.

## Scheme-agnostic paths

These combinations do not really depend on CKKS, LIU, or FDHOPE logic.

### Plaintext ndarray

If `data` is a numeric `np.ndarray` and `encrypt=False`:

- `segment=False` stores a single blob
- `segment=True` stores plaintext chunks

This applies regardless of the configured scheme.

### Direct `Chunks`

If `data` is already a `Chunks` object and `encrypt=False`:

- `put` uploads the chunks directly
- scheme-specific encryption logic is bypassed

This is the manual path for callers who already prepared chunked data.

### Generic merged retrieval

If the data was stored as merged chunks rather than CKKS ciphertext objects, the normal
read path is the merged ndarray flow:

- plaintext segmented ndarray → merged `np.ndarray`
- LIU encrypted ndarray → merged `np.ndarray`
- FDHOPE encrypted UDM → merged `np.ndarray`
- direct plaintext `Chunks` → merged `np.ndarray`

## Practical rules

1. Use plain `np.ndarray` when you want the backend to choose the normal plaintext or encrypted path.
2. Use `List[PyCtxt]` only when you already have CKKS ciphertexts.
3. Use `Chunks` only when you intentionally want the manual chunk-upload path.
4. Use FDHOPE only with caller-prepared UDM ndarrays for encrypted upload.
