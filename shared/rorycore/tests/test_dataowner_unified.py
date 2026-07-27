import pickle
from unittest.mock import patch

import pytest
import numpy as np
from rory.core.enums import Algorithm, Scheme
from rory.core.security.dataowner import DataOwner
from rory.core.security.scheme_params import (
    LiuParams, CkksParams, PaillierParams,
    LiuAndFdhopeParams, CkksAndFdhopeParams,
)
from rory.core.security.recipe import ALGORITHM_RECIPES, RecipeStep
from rory.core.interfaces.outsourced_result import OutsourcedDataResult


PLAINTEXT_3X2 = np.array([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
], dtype=np.float64)

PLAINTEXT_5X2 = np.array([
    [1.0, 2.0],
    [10.0, 11.0],
    [1.5, 2.5],
    [10.5, 11.5],
    [20.0, 21.0],
], dtype=np.float64)


class TestAlgorithmEnum:
    def test_all_values(self):
        assert Algorithm.NONE.value == "NONE"
        assert Algorithm.SKMEANS.value == "SKMEANS"
        assert Algorithm.KMEANS.value == "KMEANS"
        assert Algorithm.PPLR.value == "PPLR"

    def test_none_in_enum(self):
        assert Algorithm.NONE is not None


class TestSchemeEnum:
    def test_all_values(self):
        assert Scheme.LIU.value == "liu"
        assert Scheme.CKKS.value == "ckks"
        assert Scheme.PAILLIER.value == "paillier"
        assert Scheme.LIU_AND_FDHOPE.value == "liu_and_fdhope"
        assert Scheme.CKKS_AND_FDHOPE.value == "ckks_and_fdhope"
        assert Scheme.NONE.value == "none"


class TestSchemeParams:
    def test_liu_params_create_scheme(self):
        params = LiuParams(security_level=128)
        liu = params.create_scheme()
        assert liu is not None
        assert liu.sk is not None
        assert liu.m >= 1

    def test_ckks_params_create_scheme(self):
        params = CkksParams(security_level=128)
        ckks = params.create_scheme()
        assert ckks is not None
        assert ckks.n_features > 0

    def test_ckks_params_forwards_key_filenames(self):
        loaded_ckks = object()
        with patch(
            "rory.core.security.scheme_params.Ckks.from_pyfhel",
            return_value=loaded_ckks,
        ) as from_pyfhel:
            result = CkksParams(
                decimals=6,
                keys_path="/keys",
                relinkey_filename="custom-relinkey",
                rotatekey_filename="custom-rotatekey",
            ).create_scheme()

        assert result is loaded_ckks
        from_pyfhel.assert_called_once_with(
            _round=False,
            decimals=6,
            path="/keys",
            relinkey_filename="custom-relinkey",
            rotatekey_filename="custom-rotatekey",
        )

    def test_ckks_params_key_filenames_are_optional(self):
        loaded_ckks = object()
        with patch(
            "rory.core.security.scheme_params.Ckks.from_pyfhel",
            return_value=loaded_ckks,
        ) as from_pyfhel:
            result = CkksParams(keys_path="/keys").create_scheme()

        assert result is loaded_ckks
        from_pyfhel.assert_called_once_with(
            _round=False,
            decimals=2,
            path="/keys",
            relinkey_filename="",
            rotatekey_filename="",
        )

    def test_paillier_params_create_scheme(self):
        params = PaillierParams(security_level=128)
        paillier = params.create_scheme()
        assert paillier is not None
        assert paillier.public_key is not None

    def test_liu_and_fdhope_params_create_scheme(self):
        params = LiuAndFdhopeParams(security_level=128)
        result = params.create_scheme()
        assert isinstance(result, tuple)
        liu, fdhope = result
        assert liu is not None
        assert fdhope is not None
        assert liu.sk is not None

    def test_ckks_and_fdhope_params_create_scheme(self):
        params = CkksAndFdhopeParams(security_level=128)
        result = params.create_scheme()
        assert isinstance(result, tuple)
        ckks, fdhope = result
        assert ckks is not None
        assert fdhope is not None

    def test_liu_params_security_level(self):
        liu_128 = LiuParams(security_level=128).create_scheme()
        liu_192 = LiuParams(security_level=192).create_scheme()
        assert liu_128.m != liu_192.m


class TestRecipe:
    def test_skmeans_recipe(self):
        recipe = ALGORITHM_RECIPES[Algorithm.SKMEANS]
        assert RecipeStep.ENCRYPT_DATASET in recipe
        assert RecipeStep.GENERATE_UDM in recipe

    def test_dbsnnc_recipe(self):
        recipe = ALGORITHM_RECIPES[Algorithm.DBSNNC]
        assert RecipeStep.FDHOPE_KEYGEN in recipe
        assert RecipeStep.ENCRYPT_THRESHOLD in recipe

    def test_kmeans_empty_recipe(self):
        assert ALGORITHM_RECIPES[Algorithm.KMEANS] == []

    def test_pplr_recipe(self):
        recipe = ALGORITHM_RECIPES[Algorithm.PPLR]
        assert RecipeStep.INIT_WEIGHTS in recipe
        assert RecipeStep.INIT_BIAS in recipe
        assert RecipeStep.ENCRYPT_LABELS in recipe


class TestValidation:
    def test_invalid_combination_raises(self):
        with pytest.raises(ValueError):
            DataOwner.with_algorithm(Algorithm.KMEANS) \
                .with_scheme(Scheme.CKKS) \
                .with_scheme_params(CkksParams()) \
                .build()

    def test_missing_algorithm_and_scheme_raises(self):
        builder = DataOwner.with_algorithm(None)
        builder._algorithm = None
        with pytest.raises(ValueError):
            builder.build()

    def test_skmeans_with_liu_is_valid(self):
        do = DataOwner.with_algorithm(Algorithm.SKMEANS) \
            .with_scheme(Scheme.LIU) \
            .with_scheme_params(LiuParams(security_level=128)) \
            .build()
        result = do.outsourcedData(PLAINTEXT_3X2)
        assert isinstance(result, OutsourcedDataResult)


class TestOutsourcedDataPlaintext:
    def test_kmeans(self):
        do = DataOwner.with_algorithm(Algorithm.KMEANS) \
            .with_scheme(Scheme.NONE) \
            .with_scheme_params(None) \
            .build()
        result = do.outsourcedData(PLAINTEXT_3X2)
        assert isinstance(result, OutsourcedDataResult)
        assert result.num_attributes == 2
        assert len(result.encrypted_matrix) == 0

    def test_knn(self):
        do = DataOwner.with_algorithm(Algorithm.KNN) \
            .with_scheme(Scheme.NONE) \
            .with_scheme_params(None) \
            .build()
        result = do.outsourcedData(PLAINTEXT_3X2)
        assert result.num_attributes == 2

    def test_logistic_regression(self):
        do = DataOwner.with_algorithm(Algorithm.LOGISTIC_REGRESSION) \
            .with_scheme(Scheme.NONE) \
            .with_scheme_params(None) \
            .build()
        result = do.outsourcedData(PLAINTEXT_3X2)
        assert result.num_attributes == 2

    def test_nnc(self):
        do = DataOwner.with_algorithm(Algorithm.NNC) \
            .with_scheme(Scheme.NONE) \
            .with_scheme_params(None) \
            .build()
        result = do.outsourcedData(PLAINTEXT_3X2)
        assert result.UDM.shape == (3, 3)


class TestOutsourcedDataLiu:
    def test_skmeans(self):
        do = DataOwner.with_algorithm(Algorithm.SKMEANS) \
            .with_scheme(Scheme.LIU) \
            .with_scheme_params(LiuParams(security_level=128)) \
            .build()
        result = do.outsourcedData(PLAINTEXT_3X2)
        assert len(result.encrypted_matrix) == 3
        assert result.UDM.shape == (3, 3, 2)
        assert do.primary_scheme is not None
        assert do.primary_scheme.sk is not None

    def test_skmeans_defaults(self):
        do = DataOwner.with_algorithm(Algorithm.SKMEANS) \
            .with_scheme(Scheme.LIU) \
            .with_scheme_params(None) \
            .build()
        result = do.outsourcedData(PLAINTEXT_3X2)
        assert len(result.encrypted_matrix) == 3
        assert do.primary_scheme is not None

    def test_scheme_only_encrypt(self):
        do = DataOwner.with_algorithm(Algorithm.NONE) \
            .with_scheme(Scheme.LIU) \
            .with_scheme_params(LiuParams(security_level=128)) \
            .build()
        result = do.outsourcedData(PLAINTEXT_3X2)
        assert len(result.encrypted_matrix) == 3


class TestOutsourcedDataLiuAndFdhope:
    
    def test_dbskmeans(self):

        do = DataOwner.with_algorithm(Algorithm.DBSKMEANS) \
            .with_scheme(Scheme.LIU_AND_FDHOPE) \
            .with_scheme_params(LiuAndFdhopeParams(security_level=128)) \
            .build()
        
        result = do.outsourcedData(PLAINTEXT_5X2)
        # print("result",result)

        assert len(result.encrypted_matrix) == 5
        assert result.UDM.shape == (5, 5, 2)
        assert len(result.messageIntervals) > 0
        assert len(result.cypherIntervals) > 0
        assert do.primary_scheme is not None

    def test_dbsnnc(self):
        do = DataOwner.with_algorithm(Algorithm.DBSNNC) \
            .with_scheme(Scheme.LIU_AND_FDHOPE) \
            .with_scheme_params(LiuAndFdhopeParams(security_level=128, fdhope_seed=42)) \
            .build()
        result = do.outsourcedData(PLAINTEXT_5X2, threshold=5.0)
        assert result.UDM.shape == (5, 5)
        assert len(result.messageIntervals) > 0
        assert result.encrypted_threshold != 0


class TestOutsourcedDataCkks:
    def test_skmeans_pqc(self):
        do = DataOwner.with_algorithm(Algorithm.SKMEANS_PQC) \
            .with_scheme(Scheme.CKKS) \
            .with_scheme_params(CkksParams(security_level=128)) \
            .build()
        result = do.outsourcedData(PLAINTEXT_3X2)
        assert len(result.encrypted_matrix) == 3
        assert result.UDM.shape == (3, 3, 2)
        assert do.primary_scheme is not None

    def test_scheme_only_encrypt_ckks(self):
        do = DataOwner.with_algorithm(Algorithm.NONE) \
            .with_scheme(Scheme.CKKS) \
            .with_scheme_params(CkksParams(security_level=128)) \
            .build()
        result = do.outsourcedData(PLAINTEXT_3X2)
        assert len(result.encrypted_matrix) == 3


class TestOutsourcedDataCkksAndFdhope:
    def test_dbskmeans_pqc(self):
        do = DataOwner.with_algorithm(Algorithm.DBSKMEANS_PQC) \
            .with_scheme(Scheme.CKKS_AND_FDHOPE) \
            .with_scheme_params(CkksAndFdhopeParams(security_level=128)) \
            .build()
        result = do.outsourcedData(PLAINTEXT_3X2)
        assert len(result.encrypted_matrix) == 3
        assert len(result.messageIntervals) > 0

    def test_dbsnnc_pqc(self):
        do = DataOwner.with_algorithm(Algorithm.DBSNNC_PQC) \
            .with_scheme(Scheme.CKKS_AND_FDHOPE) \
            .with_scheme_params(CkksAndFdhopeParams(security_level=128, fdhope_seed=42)) \
            .build()
        result = do.outsourcedData(PLAINTEXT_3X2, threshold=5.0)
        assert result.UDM.shape == (3, 3)
        assert result.encrypted_threshold != 0


class TestOutsourcedDataSknn:
    def test_sknn_liu(self):
        do = DataOwner.with_algorithm(Algorithm.SKNN) \
            .with_scheme(Scheme.LIU) \
            .with_scheme_params(LiuParams(security_level=128)) \
            .build()
        result = do.outsourcedData(PLAINTEXT_3X2)
        assert len(result.encrypted_matrix) == 3
        assert result.UDM.shape == (0,)

    def test_sknn_ckks(self):
        do = DataOwner.with_algorithm(Algorithm.SKNN) \
            .with_scheme(Scheme.CKKS) \
            .with_scheme_params(CkksParams(security_level=128)) \
            .build()
        result = do.outsourcedData(PLAINTEXT_3X2)
        assert len(result.encrypted_matrix) == 3

    def test_sknn_pqc(self):
        do = DataOwner.with_algorithm(Algorithm.SKNN_PQC) \
            .with_scheme(Scheme.CKKS) \
            .with_scheme_params(CkksParams(security_level=128)) \
            .build()
        result = do.outsourcedData(PLAINTEXT_3X2)
        assert len(result.encrypted_matrix) == 3


class TestOutsourcedDataPplr:
    def test_pplr(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        y = np.array([0.0, 1.0], dtype=np.float64)
        do = DataOwner.with_algorithm(Algorithm.PPLR) \
            .with_scheme(Scheme.CKKS) \
            .with_scheme_params(CkksParams(security_level=128)) \
            .build()
        result = do.outsourcedData(
            plaintext_matrix=X, label_vector=y, n_features=2,
        )
        assert len(result.encrypted_matrix) == 2
        assert result.encrypted_weights is not None
        assert result.encrypted_bias is not None
        assert result.encrypted_labels is not None


class TestPrimaryScheme:
    def test_public_configuration_and_initialize(self):
        params = LiuParams(security_level=128, seed=7, use_np_random=True)
        do = DataOwner.with_scheme(Scheme.LIU) \
            .with_scheme_params(params) \
            .build()
        assert do.algorithm == Algorithm.NONE
        assert do.scheme == Scheme.LIU
        assert do.scheme_params is params
        assert do.primary_scheme is None
        assert do.initialize().initialize() is do
        assert do.primary_scheme is not None

    def test_liu_worker_copies_share_key_not_random_stream(self):
        do = DataOwner.with_scheme(Scheme.LIU) \
            .with_scheme_params(LiuParams(seed=7, use_np_random=True)) \
            .build() \
            .initialize()
        serialized = pickle.dumps(do)
        worker_a = pickle.loads(serialized).reseed()
        worker_b = pickle.loads(serialized).reseed()
        result_a = worker_a.outsourcedData(PLAINTEXT_3X2).encrypted_matrix
        result_b = worker_b.outsourcedData(PLAINTEXT_3X2).encrypted_matrix
        assert worker_a.primary_scheme.sk == worker_b.primary_scheme.sk
        assert not np.array_equal(result_a, result_b)
        np.testing.assert_allclose(
            worker_a.primary_scheme.decrypt_matrix(result_b).data,
            PLAINTEXT_3X2,
        )

    def test_scheme_only_vector_encryption(self):
        vector = np.array([1.0, 2.0, 3.0])
        do = DataOwner.with_scheme(Scheme.LIU) \
            .with_scheme_params(LiuParams(seed=7)) \
            .build()
        result = do.outsourcedData(vector)
        assert result.num_attributes == len(vector)
        assert result.encrypted_matrix.shape == (3, 3)

    def test_primary_scheme_available_after_outsourcedData(self):
        do = DataOwner.with_algorithm(Algorithm.SKMEANS) \
            .with_scheme(Scheme.LIU) \
            .with_scheme_params(LiuParams(security_level=128)) \
            .build()
        assert do.primary_scheme is None
        do.outsourcedData(PLAINTEXT_3X2)
        assert do.primary_scheme is not None
        assert do.primary_scheme.sk is not None

    def test_primary_scheme_not_recreated(self):
        do = DataOwner.with_algorithm(Algorithm.SKMEANS) \
            .with_scheme(Scheme.LIU) \
            .with_scheme_params(LiuParams(security_level=128)) \
            .build()
        do.outsourcedData(PLAINTEXT_3X2)
        first = do.primary_scheme
        do.outsourcedData(PLAINTEXT_3X2)
        assert do.primary_scheme is first


class TestFluentApi:
    def test_fluent_chain(self):
        builder = DataOwner.with_algorithm(Algorithm.SKMEANS)
        builder.with_scheme(Scheme.LIU)
        builder.with_scheme_params(LiuParams(security_level=128))
        do = builder.build()
        result = do.outsourcedData(PLAINTEXT_3X2)
        assert isinstance(result, OutsourcedDataResult)

    def test_fluent_chained(self):
        do = DataOwner.with_algorithm(Algorithm.SKMEANS) \
            .with_scheme(Scheme.LIU) \
            .with_scheme_params(LiuParams(security_level=128)) \
            .build()
        result = do.outsourcedData(PLAINTEXT_3X2)
        assert isinstance(result, OutsourcedDataResult)


class TestClientResult:
    def test_default_fields(self):
        cr = OutsourcedDataResult()
        assert cr.num_attributes == 0
        assert len(cr.encrypted_matrix) == 0
        assert len(cr.UDM) == 0

    def test_pplr_fields(self):
        cr = OutsourcedDataResult(
            encrypted_weights=np.array([1.0]),
            encrypted_bias=np.array([0.5]),
            encrypted_labels=np.array([0.0, 1.0]),
        )
        assert len(cr.encrypted_weights) == 1
        assert len(cr.encrypted_bias) == 1
        assert len(cr.encrypted_labels) == 2
