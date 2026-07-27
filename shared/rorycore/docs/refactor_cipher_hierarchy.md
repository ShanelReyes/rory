# Refactor: Jerarquía de Clases Abstractas para Esquemas de Cifrado

## Objetivo
Crear jerarquía de clases abstractas para los 4 esquemas de cifrado y estandarizar sus interfaces.

---

## Fase 1: Renombrar `CipherschemeResult` → `CipherResult`

**Archivo**: `rory/core/interfaces/cipherscheme_result.py` → renombrar a `cipher_result.py`

```python
class CipherResult:
    def __init__(self, data):
        self.data = data
```

- Eliminar `time` y `operation_type`
- Actualizar todos los imports (grep en todo el proyecto)

---

## Fase 2: Crear `rory/core/security/cryptosystem/abstract.py`

```python
from abc import ABC, abstractmethod

class Cipher(ABC):
    @abstractmethod
    def generate_keys(self, *args, **kwargs): ...

    @abstractmethod
    def encrypt_scalar(self, plaintext) -> CipherResult: ...

    def encrypt_vector(self, plaintext_vector) -> CipherResult:  # default: itera encrypt_scalar

    def encrypt_matrix(self, plaintext_matrix) -> CipherResult:  # default: itera encrypt_vector

    @abstractmethod
    def decrypt_scalar(self, ciphertext) -> CipherResult: ...

    def decrypt_vector(self, ciphertext_vector) -> CipherResult: # default: itera decrypt_scalar

    def decrypt_matrix(self, ciphertext_matrix) -> CipherResult: # default: itera decrypt_vector

    def save_keys(self, path): raise NotImplementedError
    def load_keys(self, path): raise NotImplementedError


class HomomorphicCipher(Cipher):
    @abstractmethod
    def add(self, ciphertext_1, ciphertext_2): ...

    def subtract(self, ciphertext_1, ciphertext_2):
        return self.add(ciphertext_1, self.multiply_scalar(-1, ciphertext_2))

    @abstractmethod
    def multiply_scalar(self, scalar, ciphertext): ...


class PartiallyHomomorphicCipher(HomomorphicCipher):
    """Soporta add + multiply_scalar. NO multiply(c1,c2)."""
    pass


class FullyHomomorphicCipher(HomomorphicCipher):
    @abstractmethod
    def multiply(self, ciphertext_1, ciphertext_2): ...
```

- `decrypt_multiply` NO va en la ABC, es exclusivo de Liu
- `generate_keys()` no recibe `key` explícito; las llaves se almacenan internamente en la instancia
- `encrypt_*` / `decrypt_*` no reciben `key` explícito; usan las llaves internas

---

## Fase 3a: Refactorizar `Liu` → `FullyHomomorphicCipher`

**Archivo**: `rory/core/security/cryptosystem/liu.py`

**Cambios:**
- Hacer que herede de `FullyHomomorphicCipher`
- `encryptScalar` → `encrypt_scalar`, usa `self.sk` internamente
- `encryptVector` → `encrypt_vector`, usa `self.sk` internamente
- `encryptMatrix` → `encrypt_matrix`, usa `self.sk` internamente
- `decryptScalar` → `decrypt_scalar`, usa `self.sk` internamente
- `decryptVector` → `decrypt_vector`, usa `self.sk` internamente
- `decryptMatrix` → `decrypt_matrix`, usa `self.sk` internamente
- Convertir métodos estáticos a métodos de instancia:
  - `Liu.add(c1, c2)` → `self.add(ciphertext_1, ciphertext_2)`
  - `Liu.multiply(c1, c2)` → `self.multiply(ciphertext_1, ciphertext_2)`
  - `Liu.multiply_c(scalar, c)` → `self.multiply_scalar(scalar, ciphertext)`
  - `Liu.subtract(c1, c2)` → `self.subtract(ciphertext_1, ciphertext_2)` (heredado de HomomorphicCipher)
- Agregar `generate_keys()` (envuelve a `generate_secret_key`, almacena `self.sk`)
- `generate_secret_key` conservado como `@deprecated`
- `decryptMultiply` → `decrypt_multiply` se conserva como método de instancia (exclusivo Liu)
- Agregar `save_keys(path)` y `load_keys(path)`
- Retornar `CipherResult(data=...)` en lugar de raw ndarray / CipherschemeResult
- Renombrar variables: `ciphertext_1`, `ciphertext_2`, `plaintext_vector`, `ciphertext_vector`, `ciphertext_matrix`, etc.

---

## Fase 3b: Refactorizar `Ckks` → `FullyHomomorphicCipher`

**Archivo**: `rory/core/security/cryptosystem/pqc/ckks.py`

**Cambios:**
- Hacer que herede de `FullyHomomorphicCipher`
- Agregar `generate_keys()` (usa la lógica de `create_client` pero almacena `he_object` internamente)
- Conservar `create_client()` y `create_server()` como factory methods
- **Mover desde `Utils` a `Ckks`** como métodos de instancia:
  - `Utils.safe_add(HE, a, b)` → `ckks.add(ciphertext_1, ciphertext_2)`
  - `Utils.safe_sub(HE, a, b)` → `ckks.subtract(ciphertext_1, ciphertext_2)` (o usar el heredado)
  - `Utils.safe_multiply(HE, a, b, scale)` → `ckks.multiply(ciphertext_1, ciphertext_2)`
  - `Utils.mul_plain_scalar(HE, ct, scalar, ...)` → `ckks.multiply_scalar(scalar, ciphertext)`
  - `Utils.dot_cipher_garbage(...)` → `ckks.dot_product(...)` (método específico CKKS, no abstracto)
  - `Utils.try_rescale_next(...)` → `ckks._try_rescale_next(...)` (interno)
  - `Utils.relinearize_if_possible(...)` → `ckks._relinearize_if_possible(...)` (interno)
- `save_keys()` / `load_keys()` → unificar nombres con la interfaz
- Retornar `CipherResult` desde `encrypt_matrix`, `decrypt_matrix`, etc.

---

## Fase 3c: Refactorizar `Paillier` → `PartiallyHomomorphicCipher`

**Archivo**: `rory/core/security/cryptosystem/paillier.py`

**Cambios:**
- Hacer que herede de `PartiallyHomomorphicCipher`
- Convertir de completamente estático a **basado en instancias**:
  - `__init__(self, public_key=None, private_key=None)`
  - `generate_keys(security_level, ...)` → crea keypair y lo almacena en `self.public_key`, `self.private_key`
- Métodos de instancia:
  - `encrypt_scalar(plaintext)` → usa `self.public_key`
  - `encrypt_vector(plaintext_vector)` → usa `self.public_key`
  - `encrypt_matrix(plaintext_matrix)` → usa `self.public_key`
  - `decrypt_scalar(ciphertext)` → usa `self.private_key`
  - `decrypt_vector(ciphertext_vector)` → usa `self.private_key`
  - `decrypt_matrix(ciphertext_matrix)` → usa `self.private_key`
- Agregar `add(ciphertext_1, ciphertext_2)` → delegar a `EncryptedNumber.__add__`
- Agregar `multiply_scalar(scalar, ciphertext)` → delegar a `EncryptedNumber.__mul__`
- `save_keys(path)` / `load_keys(path)` → adaptar a instancia
- Retornar `CipherResult`
- Métodos estáticos antiguos: conservarlos como `@deprecated`

---

## Fase 3d: Refactorizar `FdHope` → `Cipher`

**Archivo**: `rory/core/security/cryptosystem/fdhope.py`

**Cambios:**
- Hacer que herede de `Cipher`
- Agregar `generate_keys(dataset, ...)` → envuelve `keygen`, almacena `self.messagespace`, `self.cipherspace`
- `encrypt_scalar(plaintext)` → usa `self.messagespace`/`self.cipherspace` internamente
- `encrypt_vector(plaintext_vector)` → usa espacios almacenados
- `encrypt_matrix(plaintext_matrix)` → usa espacios almacenados
- `encryptTensor` → `encrypt_tensor` conservado (método específico FDHOPE, no abstracto)
- `decrypt_scalar/vector/matrix` → `raise NotImplementedError` (OPE no descifra)
- Agregar `save_keys(path)` y `load_keys(path)`
- Retornar `CipherResult`
- Métodos estáticos antiguos: conservarlos como `@deprecated`

---

## Fase 4: Marcar `Utils` homomórfico como deprecated

**Archivo**: `rory/core/utils/utils.py`

- `safe_add`, `safe_sub`, `safe_multiply`, `mul_plain_scalar`, `dot_cipher_garbage` → decorar con deprecation warning
- La lógica real se mueve a `Ckks`; los wrappers en Utils pueden delegar a la nueva API de Ckks

---

## Fase 5: Actualizar callers

| Archivo | Cambio |
|---|---|
| `rory/core/security/dataowner.py` | Usar nueva API de Liu, Fdhope |
| `rory/core/security/pqc/dataowner.py` | Usar nueva API de Ckks, Fdhope |
| `rory/core/security/dataowner_paillier.py` | Usar instancia de Paillier |
| `rory/core/classification/secure/conventional/sknn.py` | `Liu.add()` → `liu.add()` (instancia) |
| `rory/core/classification/secure/pqc/pplr.py` | `Utils.safe_*` → `ckks.*` |
| `rory/core/clustering/secure/conventional/skmeans.py` | Adaptar a `CipherResult.data` |
| `rory/core/clustering/secure/conventional/dbskmeans.py` | Adaptar a `CipherResult.data` |
| `rory/core/clustering/secure/pqc/skmeans.py` | Adaptar a `CipherResult.data` |
| `rory/core/clustering/secure/pqc/dbskmeans.py` | Adaptar a `CipherResult.data` |
| `scripts/keygen.py` | Mantener `create_client` |
| `examples/*.py` | Actualizar |
| `tests/*.py` | Actualizar |

---

## Fase 6: Tests

Actualizar todos los tests para reflejar la nueva API.

---

## Resumen de estandarización

| Método | Liu | CKKS | Paillier | FDHOPE |
|---|---|---|---|---|
| `generate_keys()` | `generate_secret_key` → almacena `self.sk` | Nuevo, usa lógica de `create_client` | Nuevo, crea keypair y almacena en instancia | `keygen` → almacena `messagespace`/`cipherspace` |
| `encrypt_scalar(v)` | usa `self.sk` | encode + encrypt | usa `self.public_key` | usa `self.messagespace`/`cipherspace` |
| `decrypt_scalar(c)` | usa `self.sk` | decrypt | usa `self.private_key` | `NotImplementedError` |
| `add(c1, c2)` | OK | desde `Utils.safe_add` | desde `EncryptedNumber.__add__` | `NotImplementedError` |
| `multiply_scalar(s, c)` | era `multiply_c` | desde `Utils.mul_plain_scalar` | desde `EncryptedNumber.__mul__` | `NotImplementedError` |
| `multiply(c1, c2)` | OK | desde `Utils.safe_multiply` | `NotImplementedError` | `NotImplementedError` |
| `save_keys` / `load_keys` | Nuevo | OK | adaptar a instancia | Nuevo |
| Retorno | `CipherResult` | `CipherResult` | `CipherResult` | `CipherResult` |

---

## Notas adicionales

- **No borrar código existente**: métodos antiguos se conservan como `@deprecated`
- **Nombres en inglés**: `encrypt_scalar`, `decrypt_matrix`, `generate_keys`, etc.
- **Variables**: `ciphertext_1` en vez de `c1`, `ciphertext_2` en vez de `c2`
- **`decrypt_multiply`**: exclusivo de Liu, NO va en la ABC
- **`create_client()` / `create_server()` en CKKS**: se conservan, son factory methods con propósito distinto a `generate_keys()`
- **FDHOPE**: esquema OPE sin operaciones homomórficas ni descifrado
