import time
import numpy as np
import os
from rory.core.security.cryptosystem.pqc.ckks import Ckks, CkksModes
from rory.core.classification.secure.pqc.pplr import PPLR




def test_experimentation(setup_paths, create_label_vector, create_datasets):
    mode                   = CkksModes.ML
    security_level         = 128
    source_dir             = setup_paths["source_path"]
    (dataset_train, dataset_train_path), (dataset_test, dataset_test_path) = create_datasets
    labelvector_train, labelvector_path = create_label_vector
    # keys_path              = setup_paths["keys_dir_path_128"]
    # dataset_train_filename = "dataset1_train.npy"
    # dataset_test_filename  = "dataset1_test.npy"
    # labelvector_filename   = "label_vector_train.npy"
    epochs                 = 2
    learning_rate          = 0.1

    pplr_result = pplr_completed(
        source_path            = source_dir,
        keys_path              = setup_paths["keys_dir_path"],
        mode                   = mode,
        security_level         = security_level,
        dataset_train_filename = dataset_train_path,
        dataset_test_filename  = dataset_test_path,
        labelvector_filename   = labelvector_path,
        epochs                 = epochs,
        learning_rate          = learning_rate
    )

    print(pplr_result)
    # print(lr_result)

def test_pplr_synthetic_epochs_gt_1(setup_paths,create_label_vector, create_datasets):
    mode                   = CkksModes.ML
    security_level         = 128
    learning_rate          = 0.1
    epochs                 = 2

    n_samples = 10
    n_features = 2
    rng = np.random.RandomState(42)
    rng.randn(n_samples, n_features).astype(np.float64)
    rng.binomial(1, 0.5, size=n_samples).astype(np.float64)

    source_dir = setup_paths["source_path"]
    (_, labelvector_path) = create_label_vector
    (_, dataset_train_path), (_, dataset_test_path) = create_datasets
    # train_path = os.path.join(source_dir, "pplr_synth_train.npy")
    # label_path = os.path.join(source_dir, "pplr_synth_label.npy")
    # test_path  = os.path.join(source_dir, "pplr_synth_test.npy")
    # np.save(train_path, X[:8])
    # np.save(label_path, y[:8])
    # np.save(test_path, X[8:])

    result = pplr_completed(
        source_path            = source_dir,
        keys_path              = setup_paths["keys_dir_path"],
        dataset_train_filename = dataset_train_path,
        dataset_test_filename  = dataset_test_path,
        labelvector_filename   = labelvector_path,
        epochs                 = epochs,
        learning_rate          = learning_rate,
        mode                   = mode,
        security_level         = security_level
    )

    assert result is not None
    assert "label_vector" in result
    assert len(result["label_vector"]) == 2
    assert all(v in (0, 1) for v in result["label_vector"])
    print(f"Epochs={epochs} completed. Service time: {result['service_time']:.2f}s, "
          f"Labels: {result['label_vector']}")

def pplr_completed(source_path, keys_path, dataset_train_filename, dataset_test_filename, labelvector_filename, epochs, learning_rate, mode, security_level):
    start_time = time.time()
    ckks = Ckks.from_pyfhel_client(
        path               = keys_path,
        ctx_filename       = Ckks._ctx_id,
        pubkey_filename    = Ckks._public_key_id,
        secretkey_filename = Ckks._secret_key_id,
        relinkey_filename  = Ckks._relin_key_id,
        rotatekey_filename = Ckks._rotate_key_id,
    )

    plain_dataset_train     = np.load(os.path.join(source_path, dataset_train_filename))
    plain_labelvector_train = np.load(os.path.join(source_path, labelvector_filename))
    plain_dataset_test      = np.load(os.path.join(source_path, dataset_test_filename))
        
    scale                       = ckks.SECURITY_LEVELS[mode.value][security_level]["scale"]
    n_features                  = plain_dataset_train.shape[1]
    n_samples                   = plain_dataset_train.shape[0]
    encrypted_matrix_train      = ckks.encryptMatrix(plain_dataset_train)
    encrypted_labelvector_train = ckks.encryptMatrix(plain_labelvector_train)

    print(encrypted_matrix_train)
    print(encrypted_labelvector_train)

    # max_level = ckks.he_object.context.first_context_data().chain_index()
    # print(f"DEBUG: La profundidad real de mi llave es: {max_level}")

    # time.sleep(1000)  # Simulate some processing time
    
    weights_matrix           = np.zeros(n_features, dtype=np.float32)
    bias                     = np.array([0.0], dtype=np.float32)
    encrypted_weights_matrix = ckks.encryptVector(weights_matrix)
    encrypted_bias           = ckks.encryptVector(bias)

    start_train_time = time.time()
    
    current_epoch = 0
    total_requested_epochs = epochs

    while current_epoch < total_requested_epochs:
        fresh_encrypted_X = ckks.encryptMatrix(plain_dataset_train)
        fresh_encrypted_y = ckks.encryptMatrix(plain_labelvector_train)
        encrypted_weights_train, encrypted_bias_train = PPLR.fit(
            scheme            = ckks,
            learning_rate     = learning_rate,
            encrypted_weights = encrypted_weights_matrix,
            encrypted_bias    = encrypted_bias,
            encrypted_matrix       = fresh_encrypted_X,
            encrypted_labelvector       = fresh_encrypted_y,
            n_features        = n_features,
            scale             = scale,
            n_samples         = n_samples
        )

        weight_plain                   = ckks.decryptVector(encrypted_weights_train)
        bias_plain                     = ckks.decryptVector(encrypted_bias_train)
        
        weight_without_noise           = Ckks.post_process(weight_plain)
        bias_without_noise             = Ckks.post_process(bias_plain)
        encrypted_weight_without_noise = ckks.encryptVector(weight_without_noise)
        encrypted_bias_without_noise   = ckks.encryptVector(bias_without_noise)
        encrypted_weights_matrix       = encrypted_weight_without_noise
        encrypted_bias                 = encrypted_bias_without_noise

        current_epoch += 1
        print(f"Epoch {current_epoch}/{total_requested_epochs} completed and refreshed.")

    end_train_time = time.time() - start_train_time
 
    encrypted_matrix_test          = ckks.encryptMatrix(plain_dataset_test)
    start_predict_time = time.time()
    encrypted_predictions = PPLR.predict(
        scheme            = ckks,
        encrypted_matrix  = encrypted_matrix_test, 
        encrypted_weights = encrypted_weight_without_noise, 
        encrypted_bias    = encrypted_bias_without_noise,
        scale             = scale,
        n_features        = n_features
    )
    
    decrypted_predictions = [ckks.decryptVector(p)[0] for p in encrypted_predictions]
    label_vector = [1 if v >= 0.5 else 0 for v in decrypted_predictions]
    end_predict_time = time.time() - start_predict_time

    end_time = time.time() - start_time
    return {
        "label_vector":label_vector,
        "service_time":end_time,
        "training_time":end_train_time,
        "predict_time":end_predict_time
    }