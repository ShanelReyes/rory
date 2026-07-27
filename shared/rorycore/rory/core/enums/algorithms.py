from enum import Enum


class Algorithm(Enum):
	NONE                = "NONE"
	SKMEANS             = "SKMEANS"
	DBSKMEANS           = "DBSKMEANS"
	DBSNNC              = "DBSNNC"
	NNC                 = "NNC"
	SKMEANS_PQC         = "SKMEANS_PQC"
	DBSKMEANS_PQC       = "DBSKMEANS_PQC"
	DBSNNC_PQC          = "DBSNNC_PQC"
	KMEANS              = "KMEANS"
	KNN                 = "KNN"
	SKNN                = "SKNN"
	SKNN_PQC            = "SKNN_PQC"
	LOGISTIC_REGRESSION = "LOGISTIC_REGRESSION"
	PPLR                = "PPLR"
