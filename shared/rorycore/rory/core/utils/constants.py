
class Constants:
	"""Namespace holding constant identifiers for clustering statuses,
	clustering algorithms, classification algorithms, and machine learning
	algorithms used throughout the Rory TPDM framework.
	"""

	class ClusteringStatus:
		COMPLETED        = 0
		START            = 1
		WORK_IN_PROGRESS = 2

	class ClusteringAlgorithms:
		SKMEANS       = "SKMEANS"
		DBSKMEANS     = "DBSKMEANS"
		KMEANS        = "KMEANS"
		DBSNNC        = "DBSNNC"
		DBSNNC_PQC    = "DBSNNC_PQC"
		NNC           = "NNC"
		SKMEANS_PQC   = "SKMEANS_PQC"
		DBSKMEANS_PQC = "DBSKMEANS_PQC"

	class ClassificationAlgorithms:
		KNN_TRAIN        = "KNN_TRAIN"
		KNN_PREDICT      = "KNN_PREDICT"
		SKNN_TRAIN       = "SKNN_TRAIN"
		SKNN_PREDICT     = "SKNN_PREDICT"
		SKNN_PQC_TRAIN   = "SKNN_PQC_TRAIN"
		SKNN_PQC_PREDICT = "SKNN_PQC_PREDICT"

	class MachineLearningAlgorithms:
		LOGISTIC_REGRESSION_TRAIN   = "LOGISTIC_REGRESSION_TRAIN"
		LOGISTIC_REGRESSION_PREDICT = "LOGISTIC_REGRESSION_PREDICT"
		PPLR_TRAIN                  = "PPLR_TRAIN"
		PPLR_PREDICT                = "PPLR_PREDICT"
