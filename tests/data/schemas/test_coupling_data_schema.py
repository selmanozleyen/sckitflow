import numpy as np
import pytest
from anndata import AnnData

from sckitflow.data.containers import CouplingData, StateData
from sckitflow.data.schemas import CouplingDataSchema

invalid_key = "invalid_key"


class TestCouplingDataSchema:
    @staticmethod
    def _verify_coupling_data(
        data: CouplingData,
        expected_lin: np.ndarray,
        expected_quad: np.ndarray | None,
    ) -> None:
        assert isinstance(data, CouplingData)
        assert np.array_equal(data.state_lin.X, expected_lin)
        if expected_quad is None:
            assert data.state_quad is None
        else:
            assert isinstance(data.state_quad, StateData)
            assert np.array_equal(data.state_quad.X, expected_quad)

    @pytest.mark.parametrize("rep_lin", [None, "X_repr", invalid_key])
    @pytest.mark.parametrize("source_rep_quad", [None, "X_src", invalid_key])
    @pytest.mark.parametrize("target_rep_quad", [None, "X_tgt", invalid_key])
    def test_init(
        self,
        rep_lin: str | None,
        source_rep_quad: str | None,
        target_rep_quad: str | None,
    ) -> None:
        schema = CouplingDataSchema(
            rep_lin=rep_lin,
            source_rep_quad=source_rep_quad,
            target_rep_quad=target_rep_quad,
        )
        assert isinstance(schema, CouplingDataSchema)
        # Properties expose the values passed at init verbatim.
        assert schema.rep_lin == rep_lin
        assert schema.source_rep_quad == source_rep_quad
        assert schema.target_rep_quad == target_rep_quad

    @pytest.mark.parametrize("rep_lin", [None, "X_repr", invalid_key])
    @pytest.mark.parametrize("source_rep_quad", [None, "X_src", invalid_key])
    @pytest.mark.parametrize("target_rep_quad", [None, "X_tgt", invalid_key])
    def test_verify_schema(
        self,
        adata: AnnData,
        rep_lin: str | None,
        source_rep_quad: str | None,
        target_rep_quad: str | None,
    ) -> None:
        schema = CouplingDataSchema(
            rep_lin=rep_lin,
            source_rep_quad=source_rep_quad,
            target_rep_quad=target_rep_quad,
        )
        has_invalid_key = invalid_key in (rep_lin, source_rep_quad, target_rep_quad)
        if has_invalid_key:
            with pytest.raises(KeyError):
                schema._verify_schema(adata)
        else:
            # Should not raise for any valid/None combination.
            schema._verify_schema(adata)

    @pytest.mark.parametrize("rep_lin", [None, "X_repr"])
    @pytest.mark.parametrize("source_rep_quad", [None, "X_src"])
    @pytest.mark.parametrize("target_rep_quad", [None, "X_tgt"])
    def test_get_data(
        self,
        adata: AnnData,
        rep_lin: str | None,
        source_rep_quad: str | None,
        target_rep_quad: str | None,
    ) -> None:
        schema = CouplingDataSchema(
            rep_lin=rep_lin,
            source_rep_quad=source_rep_quad,
            target_rep_quad=target_rep_quad,
        )

        # Expected arrays: mirror the extraction the schema is supposed to do.
        expected_lin = schema._extract_array(adata, repr=rep_lin)
        expected_src_quad = schema._extract_array(adata, repr=source_rep_quad) if source_rep_quad is not None else None
        expected_tgt_quad = schema._extract_array(adata, repr=target_rep_quad) if target_rep_quad is not None else None

        source_coupling, target_coupling = schema._get_data(adata)

        # Both source and target share the same linear term.
        self._verify_coupling_data(source_coupling, expected_lin, expected_src_quad)
        self._verify_coupling_data(target_coupling, expected_lin, expected_tgt_quad)
        source_coupling.assert_same_spatial_dims(target_coupling)
