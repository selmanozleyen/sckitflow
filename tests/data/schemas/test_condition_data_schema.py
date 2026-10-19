from collections.abc import Collection

import pytest
from anndata import AnnData

from sckitflow.data.containers import MixedTypeData
from sckitflow.data.schemas import ConditionDataSchema

from ..shared import verify_categorical_data, verify_mixin  # noqa

inval_key: str = "invalid_key"


class TestConditionDataSchema:
    @staticmethod
    def _is_valid_args(
        conditions: dict[str, Collection[str]],
        categorical_reps: dict[str, str],
    ) -> bool:
        if conditions is None and categorical_reps is None:
            return True
        elif conditions is not None and categorical_reps is not None:
            return set(conditions.keys()) == set(categorical_reps.keys())
        else:
            return False

    @pytest.mark.parametrize(
        "conditions",
        [
            None,
            {
                "drug": ["drugA", "drugB"],
            },
            {
                "drug": ["drugA", "drugB"],
                "ko": ["koA", "koB"],
            },
        ],
    )
    @pytest.mark.parametrize(
        "categorical_reps",
        [
            None,
            {"drug": "drug"},
            {"drug": "drug", "ko": "ko"},
            {inval_key: "drug", "ko": "ko"},
        ],
    )
    @pytest.mark.parametrize(
        "continuous_conditions",
        [
            None,
            ["drugA_time", "drugA_dose"],
        ],
    )
    def test_init(
        self,
        conditions: dict[str, Collection[str]],
        categorical_reps: dict[str, str],
        continuous_conditions: Collection[str] | None,
    ) -> None:
        # failure at initialization
        is_valid_args = self._is_valid_args(conditions, categorical_reps)
        if (
            (conditions is not None and categorical_reps is None)
            or (conditions is None and categorical_reps is not None)
            or not is_valid_args
        ):
            with pytest.raises(ValueError):
                # with pytest.raises(ValueError, match="The following reference columns are missing"):
                schema = ConditionDataSchema(
                    categorical_conditions=conditions,
                    categorical_reps=categorical_reps,
                    continuous_conditions=continuous_conditions,
                )
            return None
        # init schema
        schema = ConditionDataSchema(
            categorical_conditions=conditions,
            categorical_reps=categorical_reps,
            continuous_conditions=continuous_conditions,
        )
        # check attributes
        if continuous_conditions is None:
            assert schema.allows_grouping
        else:
            assert not schema.allows_grouping
        assert schema.categorical_conditions is not None
        if conditions is None:
            assert len(schema.categorical_conditions) == 0
        assert schema.categorical_reps is not None
        if categorical_reps is None:
            assert len(schema.categorical_reps) == 0

    @pytest.mark.parametrize(
        "conditions",
        [
            None,
            {
                "drug": ["drugA", "drugB"],
                "ko": ["koA", "koB"],
            },
        ],
    )
    @pytest.mark.parametrize(
        "categorical_reps",
        [
            None,
            {"drug": "drug"},
            {"drug": "drug", "ko": "ko"},
            {"drug": inval_key, "ko": "ko"},
        ],
    )
    @pytest.mark.parametrize(
        "continuous_conditions",
        [
            None,
            ["drugA_time", "drugA_dose"],
            [inval_key],
        ],
    )
    def test_get_data(
        self,
        adata: AnnData,
        uns_keys_to_nunique_prefix_and_dim: dict[str, tuple[int, str, int]],
        obsm_keys_to_dim: dict[str, int],
        conditions: dict[str, Collection[str]] | None,
        categorical_reps: dict[str, str] | None,
        continuous_conditions: Collection[str] | None,
    ) -> None:
        is_valid_args = self._is_valid_args(conditions, categorical_reps)
        if not is_valid_args:
            pytest.skip()
        schema = ConditionDataSchema(
            categorical_conditions=conditions,
            categorical_reps=categorical_reps,
            continuous_conditions=continuous_conditions,
        )
        if continuous_conditions is not None and inval_key in continuous_conditions:
            with pytest.raises(KeyError, match=r"Key .* not found in adata"):
                data = schema.get_data(adata)
            return None
        if conditions is not None and inval_key in [e for val in conditions.values() for e in val]:
            with pytest.raises(KeyError, match=r"Key .* not found in adata"):
                data = schema.get_data(adata)
            return None
        if categorical_reps is not None and inval_key in categorical_reps.values():
            with pytest.raises(KeyError, match=r"Key .* not found in adata"):
                data = schema.get_data(adata)
            return None

        # get data
        data = schema.get_data(adata)
        if conditions is None and continuous_conditions is None:
            assert data is None
            return
        else:
            assert isinstance(data, MixedTypeData)

        # test condition reps
        if conditions is not None:
            expected_df_cols = [col for val in conditions.values() for col in val]
            verify_categorical_data(
                data.categorical_covariates,
                expected_df_cols,
                uns_keys_to_nunique_prefix_and_dim,
                categorical_reps=categorical_reps,
            )
        else:
            assert data.categorical_covariates is None

        # test condition covariates
        N = len(adata)
        verify_mixin(data.continuous_covariates, N, obsm_keys_to_dim, continuous_conditions)
