# Security

## CKKS

CKKS (Cheon-Kim-Kim-Song) is a lattice-based fully homomorphic encryption scheme
suitable for approximate arithmetic on real numbers.

::: rory.core.security.cryptosystem.pqc.ckks.Ckks
    options:
      members:
        - create_client
        - create_server
        - from_pyfhel
        - from_pyfhel_client
        - from_pyfhel_server
        - encryptVector
        - encryptMatrix
        - decryptVector
        - decryptMatrix
        - encode_list
        - encrypt_list
        - post_process

::: rory.core.security.cryptosystem.pqc.ckks.CkksModes

## Liu

Liu's symmetric homomorphic encryption scheme supporting addition, subtraction,
and multiplication on encrypted data.

::: rory.core.security.cryptosystem.liu.Liu
    options:
      members:
        - __init__
        - encryptMatrix
        - encryptVector
        - encryptScalar
        - decryptMatrix
        - decryptVector
        - decryptScalar
        - add
        - multiply
        - subtract
        - multiply_c
        - decryptMultiply

## Paillier

Paillier partially homomorphic encryption scheme supporting addition of
encrypted values and scalar multiplication.

::: rory.core.security.cryptosystem.paillier.Paillier
    options:
      members:
        - generate_keypair
        - generate_keypair_by_sl
        - encryptMatrix
        - encryptVector
        - encryptScalar
        - decryptMatrix
        - decryptVector
        - decryptScalar
        - save_paillier_keys
        - load_paillier_keys

## FD-HOPE

Frequency Concealment and Distribution Order-Preserving Encryption scheme.

::: rory.core.security.cryptosystem.fdhope.Fdhope
    options:
      members:
        - keygen
        - encryptMatrix
        - encryptVector
        - encrypt
        - encryptTensor

## Data Owners

### Conventional (Liu-based)

::: rory.core.security.dataowner.DataOwner
    options:
      members:
        - algorithm
        - scheme
        - scheme_params
        - initialize
        - reseed
        - outsourcedData
        - calculate_UDM
        - calculate_DM
        - encrypt_U
        - encrypt_udm_chunks

### Paillier-based

::: rory.core.security.dataowner_paillier.DataOwner
    options:
      members:
        - generate_keys
        - from_keys
        - paillier_encrypt_matrix_chunk

### PQC (CKKS-based)

::: rory.core.security.pqc.dataowner.DataOwner
    options:
      members:
        - outsourcedData
        - calculate_UDM
        - encrypt_U
        - ckks_encrypt_matrix_chunk
