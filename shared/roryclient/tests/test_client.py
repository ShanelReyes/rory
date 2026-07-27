import pytest
from roryclient.client import RoryClient,RoryOrchestrator
import time as T
from roryclient.models import SegmentDTO
client = RoryClient(hostname="localhost",port=3001)
orchestrator = RoryOrchestrator(hostname="localhost", port=6000)

def test_mark_tasks_as_completed():
    result = orchestrator.mark_task_as_completed(task_id="1")
def test_get_mark_tasks_as_completed():
    result = orchestrator.get_marked_as_completed_tasks()
    print(result)
@pytest.mark.skip("")
def test_get_tasks():
    result = orchestrator.get_tasks()
    print(result)

@pytest.mark.skip("")
def test_get_task_details():
    result = orchestrator.get_task_details()
    print(result)

@pytest.mark.skip("")
def test_get_completed_tasks():
    result = orchestrator.get_completed_tasks()
    print(result)


@pytest.mark.skip("")
def test_segment():
    res = client.segment(
        dto = SegmentDTO(
            ball_id        = "bxx",
            bucket_id      = "rory",
            filename       = "X.npy",
            max_attempts   = 10,
            max_backoff    = 5,
            row_chunk_size = 100,
            tags           = {},
            timeout        = 120
        )
    )
    assert res.is_ok
    print(res)

@pytest.mark.skip("Kmeans algorithm")
def test_kmeans():
    k = 2
    plaintext_matrix_filename =  "clusteringc0r10a5k20"
    plaintext_matrix_id       = "kmeansx"
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
    sens                      = 0.00000000001
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

@pytest.mark.skip("dbsnnc algorithm")
def test_dbsnnc():
    plaintext_matrix_filename =  "clusteringc0r10a5k20"
    plaintext_matrix_id       = "nnc1"
    extension                 = "npy"
    threshold                 = 1.4
    num_chunks                = 2
    sens                      = 0.00000000001
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

@pytest.mark.skip("skmeans pqc algorithm")
def test_skmeans_pqc():
    k = 2
    plaintext_matrix_filename =  "clusteringc0r10a5k20"
    plaintext_matrix_id       = "skmeanspqc1"
    extension                 = "npy"
    num_chunks                = 2
    max_iterations            = 5
    experiment_iteration      = 0
    n = 10
    success_count = 0
    t1 = T.time()
    for i in range(n):
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
            success_count+=1
        else:
            print(result)
    print(f"SUCCESS={success_count} time={T.time() - t1}")
    

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
        print("KNN result",response.label_vector)
    else:
        print(result)


@pytest.mark.skip("SKNN COMPLETED")
def test_sknn():
    model_id              = "sknn1a"
    model_filename        = "classificationc0r10a5k20model"
    model_labels_filename = "classificationc0r10a5k20modellabels"
    record_test_id        = "xx"
    record_test_filename  = "classificationc0r10a5k20data"
    extension             = "npy"
    num_chunks            = 2
    result = client.sknn(
        model_id              = model_id,
        model_filename        = model_filename,
        model_labels_filename = model_labels_filename,
        record_test_filename  = record_test_filename,
        record_test_id        = record_test_id,
        extension             = extension,
        num_chunks            = num_chunks,
    )
    if result.is_ok:
        response = result.unwrap()
        print("SKNN result",response.label_vector)
    else:
        print(result)


@pytest.mark.skip("SKNN PQC COMPLETED")
def test_sknn_pqc():
    model_id              = "sknnpqc1aa"
    model_filename        = "classificationc0r10a5k20model"
    model_labels_filename = "classificationc0r10a5k20modellabels"
    record_test_id        = "sknn1pqc1aa"
    record_test_filename  = "classificationc0r10a5k20data"
    extension             = "npy"
    num_chunks            = 2
    result = client.sknn_pqc(
        model_id              = model_id,
        model_filename        = model_filename,
        model_labels_filename = model_labels_filename,
        record_test_filename  = record_test_filename,
        record_test_id        = record_test_id,
        extension             = extension,
        num_chunks            = num_chunks,
    )
    if result.is_ok:
        response = result.unwrap()
        print("SKNN PQC result",response.label_vector)
    else:
        print(result)


# ================================================
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


@pytest.mark.skip("Logistic Regression Train")
def test_logistic_regression_train():
    result = client.logistic_regression_train(
        plaintext_matrix_train_id="logreg1",
        plaintext_matrix_train_filename="classificationc0r10a5k20model",
        plaintext_label_vector_train_id="logreg1",
        plaintext_label_vector_train_filename="classificationc0r10a5k20modellabels",
        extension="npy",
        epochs=1,
        learning_rate=0.01,
    )
    assert result.is_ok


@pytest.mark.skip("Logistic Regression Predict")
def test_logistic_regression_predict():
    result = client.logistic_regression_predict(
        plaintext_matrix_train_id="logreg1",
        plaintext_matrix_test_id="logreg1",
        plaintext_matrix_test_filename="classificationc0r10a5k20data",
        extension="npy",
    )
    assert result.is_ok


@pytest.mark.skip("Logistic Regression Complete")
def test_logistic_regression():
    result = client.logistic_regression(
        plaintext_matrix_train_id="logreg1",
        plaintext_matrix_train_filename="classificationc0r10a5k20model",
        plaintext_label_vector_train_id="logreg1",
        plaintext_label_vector_train_filename="classificationc0r10a5k20modellabels",
        plaintext_matrix_test_id="logreg1",
        plaintext_matrix_test_filename="classificationc0r10a5k20data",
        extension="npy",
        epochs=1,
        learning_rate=0.01,
    )
    assert result.is_ok


@pytest.mark.skip("PPLR Train")
def test_pplr_train():
    result = client.pplr_train(
        plaintext_matrix_train_id="pplr1",
        plaintext_matrix_train_filename="classificationc0r10a5k20model",
        plaintext_label_vector_train_id="pplr1",
        plaintext_label_vector_train_filename="classificationc0r10a5k20modellabels",
        extension="npy",
        epochs=1,
        learning_rate=0.01,
    )
    assert result.is_ok


@pytest.mark.skip("PPLR Predict")
def test_pplr_predict():
    result = client.pplr_predict(
        plaintext_matrix_train_id="pplr1",
        plaintext_matrix_test_id="pplr1",
        plaintext_matrix_test_filename="classificationc0r10a5k20data",
        extension="npy",
    )
    assert result.is_ok


@pytest.mark.skip("PPLR Complete")
def test_pplr():
    result = client.pplr(
        plaintext_matrix_train_id="pplr1",
        plaintext_matrix_train_filename="classificationc0r10a5k20model",
        plaintext_label_vector_train_id="pplr1",
        plaintext_label_vector_train_filename="classificationc0r10a5k20modellabels",
        plaintext_matrix_test_id="pplr1",
        plaintext_matrix_test_filename="classificationc0r10a5k20data",
        extension="npy",
        epochs=1,
        learning_rate=0.01,
    )
    assert result.is_ok
