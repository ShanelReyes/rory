import pytest
from mictlanx.v4.client import Client
from mictlanx.v4.interfaces import Router
from roryclient.client import RoryClient
import time as T
client = RoryClient(hostname="localhost",port=3001)

@pytest.mark.skip("Kmeans algorithm")
def test_kmeans():
    k = 2
    plaintext_matrix_filename =  "clusteringc0r10a5k20"
    plaintext_matrix_id       = "kmeans1"
    extension                 = "npy"
    result = client.kmeans(
        k                         = k,
        plaintext_matrix_filename = plaintext_matrix_filename,
        plaintext_matrix_id       = plaintext_matrix_id,
        extension                 = extension
    )
    if result.is_ok:
        response = result.unwrap()
        print("KMEANS result", response.label_vector)
    else:
        print(result)

@pytest.mark.skip("Skmeans algorithm")
def test_skmeans():
    k = 2
    plaintext_matrix_filename =  "clusteringc0r10a5k20"
    plaintext_matrix_id       = "skmeans1"
    extension                 = "npy"
    num_chunks                = 2
    max_iterations            = 5
    result = client.skmeans(
        k                         = k,
        plaintext_matrix_filename = plaintext_matrix_filename,
        plaintext_matrix_id       = plaintext_matrix_id,
        extension                 = extension,
        num_chunks                = num_chunks,
        max_iterations            = max_iterations
    )
    if result.is_ok:
        response = result.unwrap()
        print("SKMEANS result", response.label_vector)
    else:
        print(result)

@pytest.mark.skip("Dbskmeans algorithm")
def test_dbskmeans():
    k = 2
    plaintext_matrix_filename =  "clusteringc0r10a5k20"
    plaintext_matrix_id       = "dbskmeans1"
    extension                 = "npy"
    num_chunks                = 2
    max_iterations            = 5
    sens                      = "0.00000000001"
    result = client.dbskmeans(
        k                         = k,
        plaintext_matrix_filename = plaintext_matrix_filename,
        plaintext_matrix_id       = plaintext_matrix_id,
        extension                 = extension,
        num_chunks                = num_chunks,
        max_iterations            = max_iterations,
        sens                      = sens
    )
    if result.is_ok:
        response = result.unwrap()
        print("DBSKMEANS result", response.label_vector)
    else:
        print(result)

@pytest.mark.skip("skmeans pqc algorithm")
def test_skmeans_pqc():
    k = 2
    plaintext_matrix_filename =  "clusteringc0r10a5k20"
    plaintext_matrix_id       = "skmeanspqc1"
    extension                 = "npy"
    num_chunks                = 2
    max_iterations            = 5
    experiment_iteration      = 0
    result = client.skmeans_pqc(
        k                         = k,
        plaintext_matrix_filename = plaintext_matrix_filename,
        plaintext_matrix_id       = plaintext_matrix_id,
        extension                 = extension,
        experiment_iteration      = experiment_iteration,
        num_chunks                = num_chunks,
        max_iterations            = max_iterations,
    )
    if result.is_ok:
        response = result.unwrap()
        print("SKMEANS PQC result", response.label_vector)
    else:
        print(result)
    

@pytest.mark.skip("Dbskmeans pqc algorithm")
def test_dbskmeans_pqc():
    k = 2
    plaintext_matrix_filename =  "clusteringc0r10a5k20"
    plaintext_matrix_id       = "dbskmeanspqc1"
    extension                 = "npy"
    num_chunks                = 2
    max_iterations            = 5
    sens                      = "0.00000000001"
    experiment_iteration      = 0
    result = client.dbskmeans_pqc(
        k                         = k,
        plaintext_matrix_filename = plaintext_matrix_filename,
        plaintext_matrix_id       = plaintext_matrix_id,
        extension                 = extension,
        experiment_iteration      = experiment_iteration,
        num_chunks                = num_chunks,
        max_iterations            = max_iterations,
        sens                      = sens
    )
    if result.is_ok:
        response = result.unwrap()
        print("DBSKMEANS PQC result", response.label_vector)
    else:
        print(result)


#### NNC
@pytest.mark.skip("nnc algorithm")
def test_nnc():
    plaintext_matrix_filename =  "clusteringc0r10a5k20"
    plaintext_matrix_id       = "nnc1"
    extension                 = "npy"
    threshold                 = "1.4"
    result = client.nnc(
        plaintext_matrix_filename = plaintext_matrix_filename,
        plaintext_matrix_id       = plaintext_matrix_id,
        threshold                 = threshold,
        extension                 = extension
    )
    if result.is_ok:
        response = result.unwrap()
        print("NNC result", response.label_vector)
    else:
        print(result)

    # print("NNC result", result.label_vector)

@pytest.mark.skip("dbsnnc algorithm")
def test_dbsnnc():
    plaintext_matrix_filename =  "clusteringc0r10a5k20"
    plaintext_matrix_id       = "nnc1"
    extension                 = "npy"
    threshold                 = "1.4"
    num_chunks                = 2
    sens                      = "0.00000000001"
    result = client.dbsnnc(
        plaintext_matrix_filename = plaintext_matrix_filename,
        plaintext_matrix_id       = plaintext_matrix_id,
        threshold                 = threshold,
        extension                 = extension,
        num_chunks                = num_chunks,
        sens                      = sens
    )
    if result.is_ok:
        response = result.unwrap()
        print("DBSNNC result", response.label_vector)
    else:
        print(result)
    # print("DBSNNC result", result.label_vector)


#### Classification

@pytest.mark.skip("KNN TRAIN")
def test_knn_train():
    model_id              = "knn"
    model_filename        = "classificationc0r10a5k20model"
    model_labels_filename = "classificationc0r10a5k20modellabels"
    extension             = "npy"
    result = client.knn_train(
        model_id              = model_id,
        model_filename        = model_filename,
        model_labels_filename = model_labels_filename,
        extension             = extension
    )
    if result.is_ok:
        response = result.unwrap()
        print("KNN TRAIN result", response)
    else:
        print(result)
    # print("KNN TRAIN RESULT",result)

@pytest.mark.skip("KNN PREDICT")
def test_knn_predict():
    model_id              = "knn"
    model_filename        = "classificationc0r10a5k20model"
    model_labels_filename = "classificationc0r10a5k20modellabels"
    record_test_id        = "knn1"
    record_test_filename  = "classificationc0r10a5k20data"
    extension             = "npy"
    result = client.knn_predict(
        model_id              = model_id,
        model_filename        = model_filename,
        model_labels_filename = model_labels_filename,
        record_test_filename  = record_test_filename,
        record_test_id        = record_test_id,
        extension             = extension
    )
    if result.is_ok:
        response = result.unwrap()
        print("KNN PREDICT result", response.label_vector)
    else:
        print(result)
    # print("KNN PREDICT RESULT",result)

@pytest.mark.skip("KNN COMPLETED")
def test_knn():
    model_id              = "knn1a"
    model_filename        = "classificationc0r10a5k20model"
    model_labels_filename = "classificationc0r10a5k20modellabels"
    record_test_id        = "knn1a"
    record_test_filename  = "classificationc0r10a5k20data"
    extension             = "npy"
    result = client.knn(
        model_id              = model_id,
        model_filename        = model_filename,
        model_labels_filename = model_labels_filename,
        record_test_filename  = record_test_filename,
        record_test_id        = record_test_id,
        extension             = extension
    )
    if result.is_ok:
        response = result.unwrap()
        print("KNN RESULT",response)
    assert result.is_ok

@pytest.mark.skip("SKNN TRAIN")
def test_sknn_train():
    model_id              = "sknn1b"
    model_filename        = "classificationc0r10a5k20model"
    model_labels_filename = "classificationc0r10a5k20modellabels"
    extension             = "npy"
    num_chunks            = 2
    result = client.sknn_train(
        model_id              = model_id,
        model_filename        = model_filename,
        model_labels_filename = model_labels_filename,
        num_chunks            = num_chunks,
        extension             = extension
    )
    if result.is_ok:
        response = result.unwrap()
        print("SKNN TRAIN result", response)
    else:
        print(result)

    # print("SKNN TRAIN RESULT",result)

@pytest.mark.skip("SKNN PREDICT")
def test_sknn_predict():
    model_id              = "sknn1b"
    model_filename        = "classificationc0r10a5k20model"
    model_labels_filename = "classificationc0r10a5k20modellabels"
    record_test_id        = "sknn1b"
    record_test_filename  = "classificationc0r10a5k20data"
    extension             = "npy"
    encrypted_model_shape = "(8,5,3)"
    encrypted_model_dtype = "float32"
    num_chunks            = 2
    result = client.sknn_predict(
        model_id              = model_id,
        model_filename        = model_filename,
        model_labels_filename = model_labels_filename,
        record_test_filename  = record_test_filename,
        record_test_id        = record_test_id,
        extension             = extension,
        num_chunks            = num_chunks,
        encrypted_model_shape = encrypted_model_shape,
        encrypted_model_dtype = encrypted_model_dtype
    )
    if result.is_ok:
        response = result.unwrap()
        print("SKNN PREDICT result", response.label_vector)
    else:
        print(result)
    # print("SKNN PREDICT RESULT",result)

# @pytest.mark.skip("SKNN COMPLETED")
def test_sknn():
    model_id              = "sknn1a"
    model_filename        = "classificationc0r10a5k20model"
    model_labels_filename = "classificationc0r10a5k20modellabels"
    record_test_id        = "xx"
    record_test_filename  = "classificationc0r10a5k20data"
    extension             = "npy"
    # encrypted_model_shape = "(8,5)"
    # encrypted_model_dtype = "float32"
    num_chunks            = 2
    result = client.sknn(
        model_id              = model_id,
        model_filename        = model_filename,
        model_labels_filename = model_labels_filename,
        record_test_filename  = record_test_filename,
        record_test_id        = record_test_id,
        extension             = extension,
        num_chunks            = num_chunks,
        # encrypted_model_shape = encrypted_model_shape,
        # encrypted_model_dtype = encrypted_model_dtype
    )
    if result.is_ok:
        response = result.unwrap()
        print("SKNN RESULT",response)
    assert result.is_ok


@pytest.mark.skip("SKNN PQC TRAIN")
def test_sknn_pqc_train():
    model_id              = "sknnpqc1a"
    model_filename        = "classificationc0r10a5k20model"
    model_labels_filename = "classificationc0r10a5k20modellabels"
    extension             = "npy"
    num_chunks            = 2
    result = client.sknn_pqc_train(
        model_id              = model_id,
        model_filename        = model_filename,
        model_labels_filename = model_labels_filename,
        num_chunks            = num_chunks,
        extension             = extension
    )
    if result.is_ok:
        response = result.unwrap()
        print("SKNN PQC TRAIN RESULT",response)
    assert result.is_ok

    # print("SKNN PQC TRAIN RESULT",result)

@pytest.mark.skip("SKNN PQC PREDICT")
def test_sknn_pqc_predict():
    model_id              = "sknnpqc1a"
    model_filename        = "classificationc0r10a5k20model"
    model_labels_filename = "classificationc0r10a5k20modellabels"
    record_test_id        = "sknn1pqc1a"
    record_test_filename  = "classificationc0r10a5k20data"
    extension             = "npy"
    encrypted_model_shape = "(8,5)"
    encrypted_model_dtype = "float32"
    num_chunks            = 2
    result = client.sknn_pqc_predict(
        model_id              = model_id,
        model_filename        = model_filename,
        model_labels_filename = model_labels_filename,
        record_test_filename  = record_test_filename,
        record_test_id        = record_test_id,
        extension             = extension,
        num_chunks            = num_chunks,
        encrypted_model_shape = encrypted_model_shape,
        encrypted_model_dtype = encrypted_model_dtype
    )
    if result.is_ok:
        response = result.unwrap()
        print("SKNN PQC PREDICT RESULT",response.label_vector)
    assert result.is_ok

    # print("SKNN PQC PREDICT RESULT",result)

@pytest.mark.skip("SKNN PQC COMPLETED")
def test_sknn_pqc():
    model_id              = "sknnpqc1aa"
    model_filename        = "classificationc0r10a5k20model"
    model_labels_filename = "classificationc0r10a5k20modellabels"
    record_test_id        = "sknn1pqc1aa"
    record_test_filename  = "classificationc0r10a5k20data"
    extension             = "npy"
    encrypted_model_shape = "(8,5)"
    encrypted_model_dtype = "float32"
    num_chunks            = 2
    result = client.sknn_pqc(
        model_id              = model_id,
        model_filename        = model_filename,
        model_labels_filename = model_labels_filename,
        record_test_filename  = record_test_filename,
        record_test_id        = record_test_id,
        extension             = extension,
        num_chunks            = num_chunks,
        encrypted_model_shape = encrypted_model_shape,
        encrypted_model_dtype = encrypted_model_dtype
    )
    if result.is_ok:
        response = result.unwrap()
        print("SKNN RESULT",response)
    assert result.is_ok


@pytest.mark.skip("")
def test_download():
    client = Client(
        client_id="rory_client_mictlanx",
        bucket_id="rory",
        log_output_path="/rory/log",
        routers=[Router(
            router_id="mictlanx-router-0",
            protocol="http",
            ip_addr="localhost",
            port=60666
        )]
    )
    t1 = T.time()
    res = client.get_with_retry(key="encryptedsknnmodel1bbbbb", chunk_size="50MB")
    print("RES", res)
    print("RESPONSE_TIME", T.time()-t1)