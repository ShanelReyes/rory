from enum import Enum
from rory.core.enums.algorithms import Algorithm


class RecipeStep(Enum):
	ENCRYPT_DATASET     = "encrypt_dataset"
	GENERATE_UDM        = "generate_udm"
	GENERATE_DM         = "generate_dm"
	FDHOPE_KEYGEN       = "fdhope_keygen"
	ENCRYPT_U           = "encrypt_u"
	GENERATE_THRESHOLD  = "generate_threshold"
	ENCRYPT_THRESHOLD   = "encrypt_threshold"
	INIT_WEIGHTS        = "init_weights"
	INIT_BIAS           = "init_bias"
	ENCRYPT_WEIGHTS     = "encrypt_weights"
	ENCRYPT_BIAS        = "encrypt_bias"
	ENCRYPT_LABELS      = "encrypt_labels"


ALGORITHM_RECIPES: dict[Algorithm, list[RecipeStep]] = {
	Algorithm.SKMEANS:               [RecipeStep.ENCRYPT_DATASET, RecipeStep.GENERATE_UDM],
	Algorithm.DBSKMEANS:             [RecipeStep.ENCRYPT_DATASET, RecipeStep.GENERATE_UDM, RecipeStep.FDHOPE_KEYGEN, RecipeStep.ENCRYPT_U],
	Algorithm.DBSNNC:                [RecipeStep.ENCRYPT_DATASET, RecipeStep.GENERATE_DM, RecipeStep.FDHOPE_KEYGEN, RecipeStep.ENCRYPT_U, RecipeStep.GENERATE_THRESHOLD, RecipeStep.ENCRYPT_THRESHOLD],
	Algorithm.NNC:                   [RecipeStep.GENERATE_DM],
	Algorithm.SKMEANS_PQC:           [RecipeStep.ENCRYPT_DATASET, RecipeStep.GENERATE_UDM],
	Algorithm.DBSKMEANS_PQC:         [RecipeStep.ENCRYPT_DATASET, RecipeStep.GENERATE_UDM, RecipeStep.FDHOPE_KEYGEN, RecipeStep.ENCRYPT_U],
	Algorithm.DBSNNC_PQC:            [RecipeStep.ENCRYPT_DATASET, RecipeStep.GENERATE_DM, RecipeStep.FDHOPE_KEYGEN, RecipeStep.ENCRYPT_U, RecipeStep.GENERATE_THRESHOLD, RecipeStep.ENCRYPT_THRESHOLD],
	Algorithm.SKNN:                  [RecipeStep.ENCRYPT_DATASET],
	Algorithm.SKNN_PQC:              [RecipeStep.ENCRYPT_DATASET],
	Algorithm.PPLR:                  [RecipeStep.ENCRYPT_DATASET, RecipeStep.ENCRYPT_LABELS, RecipeStep.INIT_WEIGHTS, RecipeStep.INIT_BIAS, RecipeStep.ENCRYPT_WEIGHTS, RecipeStep.ENCRYPT_BIAS],
	Algorithm.KMEANS:                [],
	Algorithm.KNN:                   [],
	Algorithm.LOGISTIC_REGRESSION:   [],
}
