from rory.core.utils.constants import Constants


def test_clustering_status():
    assert Constants.ClusteringStatus.COMPLETED == 0
    assert Constants.ClusteringStatus.START == 1
    assert Constants.ClusteringStatus.WORK_IN_PROGRESS == 2


def test_clustering_algorithms():
    assert Constants.ClusteringAlgorithms.SKMEANS == "SKMEANS"
    assert Constants.ClusteringAlgorithms.DBSKMEANS == "DBSKMEANS"
    assert Constants.ClusteringAlgorithms.KMEANS == "KMEANS"
    assert Constants.ClusteringAlgorithms.DBSNNC == "DBSNNC"
    assert Constants.ClusteringAlgorithms.NNC == "NNC"
    assert Constants.ClusteringAlgorithms.SKMEANS_PQC == "SKMEANS_PQC"
    assert Constants.ClusteringAlgorithms.DBSKMEANS_PQC == "DBSKMEANS_PQC"


def test_classification_algorithms():
    assert Constants.ClassificationAlgorithms.KNN_TRAIN == "KNN_TRAIN"
    assert Constants.ClassificationAlgorithms.KNN_PREDICT == "KNN_PREDICT"
    assert Constants.ClassificationAlgorithms.SKNN_TRAIN == "SKNN_TRAIN"
    assert Constants.ClassificationAlgorithms.SKNN_PREDICT == "SKNN_PREDICT"
    assert Constants.ClassificationAlgorithms.SKNN_PQC_TRAIN == "SKNN_PQC_TRAIN"
    assert Constants.ClassificationAlgorithms.SKNN_PQC_PREDICT == "SKNN_PQC_PREDICT"


def test_machine_learning_algorithms():
    assert Constants.MachineLearningAlgorithms.LOGISTIC_REGRESSION_TRAIN == "LOGISTIC_REGRESSION_TRAIN"
    assert Constants.MachineLearningAlgorithms.LOGISTIC_REGRESSION_PREDICT == "LOGISTIC_REGRESSION_PREDICT"
    assert Constants.MachineLearningAlgorithms.PPLR_TRAIN == "PPLR_TRAIN"
    assert Constants.MachineLearningAlgorithms.PPLR_PREDICT == "PPLR_PREDICT"
