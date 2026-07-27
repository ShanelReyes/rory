import pytest
import numpy as np
from rory.core.security.cryptosystem.pqc.ckks import Ckks, CkksModes

# --- Configuración ---
# PATH_SINK = "/rory/sink"
# --- Configuración Global ---
SCHEME = "CKKS"
MODE = CkksModes.ML
SECURITY_LEVEL = 128
# SCALE = Ckks.SECURITY_LEVELS[MODE][SECURITY_LEVEL]["scale"]
ROUND_VAL = True
DECIMALS = 6
SAVE = True
ENABLE_RELINEARIZE = True
ENABLE_ROTATION = True
GREEN = "\033[92m"
RESET = "\033[0m"

# @pytest.fixture(scope="module")
# def setup_paths():
#     keys_path =  os.environ.get("RORY_KEYS_PATH","/rory")
#     paths = {
#         "output_path": keys_path,
#         "keys_dir_path": f"{keys_path}/keys",
#     }
#     os.makedirs(paths["keys_dir_path"], exist_ok=True)
#     return paths


# --- Tests ---
# @pytest.mark.skip("Key generation")
def test_01_generar_y_guardar_llaves(setup_paths):
    ckks = Ckks.create_client(
        scheme=SCHEME,
        mode=MODE,
        security_level=SECURITY_LEVEL,
        _round=ROUND_VAL,
        decimals=DECIMALS,
        output_path=setup_paths["keys_dir_path"],
        save=SAVE,
        enable_relinearize=ENABLE_RELINEARIZE,
        enable_rotate=  ENABLE_ROTATION,
    )
    assert ckks is not None
    print(f"Completado {SECURITY_LEVEL}")

@pytest.mark.skip("Key generation")
def test_02_cargar_y_validar_llaves(setup_paths):
    """
    Lee las llaves generadas en el test anterior y verifica que funcionen.
    """

    # Cargamos usando el método estático de tu clase
    Ckks.from_pyfhel_client(
        path=setup_paths["keys_dir_path"],
        ctx_filename=Ckks._ctx_id,
        pubkey_filename=Ckks._public_key_id,
        secretkey_filename=Ckks._secret_key_id,
        relinkey_filename=Ckks._relin_key_id,
        rotatekey_filename=Ckks._rotate_key_id,
    )

    print(f"Completado {SECURITY_LEVEL}")



@pytest.mark.skip("Key generation")
def test_03_cargar_y_validar_llaves(setup_paths):
    """
    Lee las llaves generadas en el test anterior y verifica que funcionen.
    """

    # Cargamos usando el método estático de tu clase
    Ckks.from_pyfhel_server(
        path=setup_paths["keys_dir_path"],
        ctx_filename=Ckks._ctx_id,
        pubkey_filename=Ckks._public_key_id,
        # secretkey_filename=Ckks._secret_key_id,
        relinkey_filename=Ckks._relin_key_id,
        rotatekey_filename=Ckks._rotate_key_id,
    )

    print(f"Completado {SECURITY_LEVEL}")

@pytest.mark.skip("Key generation")
def test_02_leer_y_validar_llaves(setup_paths):
    """Lee las llaves desde /rory/sink y cifra/descifra algo para probarlas"""
    print(f"\n2. Leyendo llaves desde {setup_paths['keys_dir_path']} para validación...")

    # Cargamos usando la función from_pyfhel de tu clase
    ckks_cargado = Ckks.from_pyfhel_client(
        path=setup_paths["keys_dir_path"],
        ctx_filename=Ckks._ctx_id,
        pubkey_filename=Ckks._public_key_id,
        secretkey_filename=Ckks._secret_key_id,
        relinkey_filename=Ckks._relin_key_id,
        rotatekey_filename=Ckks._rotate_key_id,
    )

    #assert ckks_cargado.he_object.context_established(), "El contexto no se cargó correctamente"
    print(f"{GREEN}[OK]{RESET} Contexto y llaves cargados en memoria.")

    # Prueba de fuego: Cifrar y Descifrar
    dato_original = np.array([42.0])
    print(f"Probando operación con dato: {dato_original[0]}")
    
    cifrado = ckks_cargado.encryptVector(plaintext_vector=dato_original)
    descifrado = ckks_cargado.decryptVector(ciphertext_vector=cifrado)

    # Validamos que el resultado sea el mismo (con pequeño margen de error por CKKS)
    assert np.isclose(dato_original[0], descifrado[0], atol=0.01)
    print(f"{GREEN}[OK]{RESET} Descifrado exitoso: {descifrado[0]:.4f}")
    print(f"{GREEN}[EXITO]{RESET} Las llaves son válidas y funcionales.")