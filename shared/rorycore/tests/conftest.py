import os

import pytest
from dotenv import load_dotenv
from rory.core.security.cryptosystem.pqc.ckks import Ckks,CkksModes
from rory.core.security.dataowner_paillier import DataOwner as DataownerPaillier
import numpy as np
load_dotenv(".env.test")

RORY_KEYS_PATH = os.environ.get("RORY_KEYS_PATH", "/rory/keys/")
RORY_PALLIER_KEYS_PATH = os.environ.get("PAILLIER_KEYS_PATH", "/rory/pallier/keys/")
# RORY_SOURCE = os.environ.get("RORY_SOURCE", "/rory/source")
RORY_PATH = os.environ.get("RORY_PATH", "/rory")

@pytest.fixture(scope="module")
def setup_paths():
    output_path = os.environ.get("RORY_PATH", "/rory")
    paths = {
        "output_path": output_path,
        "keys_dir_path":RORY_KEYS_PATH,
        "keys_dir_path_128": f"{output_path}/keys/keys128",
        "keys_dir_path_192": f"{output_path}/keys/keys192",
        "keys_dir_path_256": f"{output_path}/keys/keys256",
        "source_path": f"{output_path}/source",
    }
    os.makedirs(paths["keys_dir_path_128"], exist_ok=True)
    os.makedirs(paths["keys_dir_path_192"], exist_ok=True)
    os.makedirs(paths["keys_dir_path_256"], exist_ok=True)
    os.makedirs(paths["source_path"], exist_ok=True)
    return paths

@pytest.fixture(scope="module")
def create_label_vector():
    # source_path = os.environ.get("", "/rory/source")
    labelvector_filename = "label_vector_train.npy"
    label_vector_train = np.random.randint(0, 2, size=10).astype(np.float32)
    np.save(os.path.join(RORY_PATH, labelvector_filename), label_vector_train)
    return label_vector_train, f"{RORY_PATH}/{labelvector_filename}"

@pytest.fixture(scope="module")
def create_datasets():
    dataset_train_filename = "dataset1_train.npy"
    dataset_test_filename = "dataset1_test.npy"
    dataset_train = np.random.rand(10, 10).astype(np.float32)
    dataset_test = np.random.rand(2, 10).astype(np.float32)
    np.save(os.path.join(RORY_PATH, dataset_train_filename), dataset_train)
    np.save(os.path.join(RORY_PATH, dataset_test_filename), dataset_test)
    return (dataset_train, f"{RORY_PATH}/{dataset_train_filename}"), (dataset_test, f"{RORY_PATH}/{dataset_test_filename}")


# 
@pytest.fixture(scope="module")
def pallier_dataowner():
    keys_file = "rory-phe-128"
    _ = DataownerPaillier(securitylevel=128)\
        .generate_keys(output_path=RORY_PALLIER_KEYS_PATH, filename=keys_file, save=True)

    return DataownerPaillier.from_keys(
        path=RORY_PALLIER_KEYS_PATH,
        filename=keys_file
    )


@pytest.fixture(scope="module")
def key_gen():

    Ckks.create_client(
        scheme             = os.environ.get("RORY_SCHEME", "CKKS"),
        mode               = CkksModes(os.environ.get("RORY_MODE", "ml")),
        security_level     = int(os.environ.get("RORY_SECURITY_LEVEL", "128")),
        _round             = bool(os.environ.get("RORY_ROUND", "True")),
        decimals           = int(os.environ.get("RORY_DECIMALS", "2")),
        output_path        = RORY_KEYS_PATH,
        save               = bool(os.environ.get("RORY_SAVE", "True")),
        enable_relinearize = bool(os.environ.get("RORY_ENABLE_RELINEARIZE", "True")),
        enable_rotate      = bool(os.environ.get("RORY_ENABLE_ROTATE", "True")),
    )

@pytest.fixture(scope="module")
def ckks_client(key_gen):

    # keys_path = os.environ.get("RORY_KEYS_PATH", "/rory/keys/keys128")
    
    return Ckks.from_pyfhel_client(
        path               = RORY_KEYS_PATH,
        ctx_filename       = Ckks._ctx_id,
        pubkey_filename    = Ckks._public_key_id,
        secretkey_filename = Ckks._secret_key_id,
        relinkey_filename  = Ckks._relin_key_id,
        rotatekey_filename = Ckks._rotate_key_id,
    )


@pytest.fixture(scope="module")
def liu_scheme():
    from rory.core.security.cryptosystem.liu import Liu

    seed_raw = os.environ.get("LIU_SEED", "None")
    seed = None if seed_raw in ("None", "") else int(seed_raw)
    return Liu(
        _round=True,
        decimals=int(os.environ.get("LIU_DECIMALS", "6")),
        secure_random=False,
        seed=seed,
        use_np_random=False,
        security_level=int(os.environ.get("LIU_SECURITY_LEVEL", "128")),
    )


@pytest.fixture(scope="module")
def secret_key(liu_scheme):
    return liu_scheme.generate_secret_key()
