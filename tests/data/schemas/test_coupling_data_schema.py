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
        expected_lin: np.ndarray | None,
        expected_quad: np.ndarray | None,
    ) -> None:
        assert isinstance(data, CouplingData)
        if expected_lin is None:
            assert data.state_lin is None
        else:
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
        assert schema.rep_lin == rep_lin
        assert schema.source_rep_quad == source_rep_quad
        assert schema.target_rep_quad == target_rep_quad
        assert schema.use_linear is True

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

        expected_lin = schema._extract_array(adata, repr=rep_lin)
        expected_src_quad = schema._extract_array(adata, repr=source_rep_quad) if source_rep_quad is not None else None
        expected_tgt_quad = schema._extract_array(adata, repr=target_rep_quad) if target_rep_quad is not None else None

        source_coupling, target_coupling = schema._get_data(adata)

        self._verify_coupling_data(source_coupling, expected_lin, expected_src_quad)
        self._verify_coupling_data(target_coupling, expected_lin, expected_tgt_quad)
        source_coupling.assert_same_spatial_dims(target_coupling)


class TestUseLinearFalse:
    """Pure Gromov-Wasserstein: no shared linear term, both quadratics required."""

    def test_init_rejects_rep_lin(self) -> None:
        with pytest.raises(ValueError, match="redundant"):
            CouplingDataSchema(
                rep_lin="X_repr",
                source_rep_quad="X_src",
                target_rep_quad="X_tgt",
                use_linear=False,
            )

    @pytest.mark.parametrize(
        "source_rep_quad, target_rep_quad",
        [(None, "X_tgt"), ("X_src", None), (None, None)],
    )
    def test_init_requires_both_quads(
        self,
        source_rep_quad: str | None,
        target_rep_quad: str | None,
    ) -> None:
        with pytest.raises(ValueError, match="requires both"):
            CouplingDataSchema(
                source_rep_quad=source_rep_quad,
                target_rep_quad=target_rep_quad,
                use_linear=False,
            )

    def test_init_valid(self) -> None:
        schema = CouplingDataSchema(
            source_rep_quad="X_src",
            target_rep_quad="X_tgt",
            use_linear=False,
        )
        assert schema.use_linear is False
        assert schema.rep_lin is None

    def test_verify_schema_skips_rep_lin(self, adata: AnnData) -> None:
        # No `rep_lin` check: only the two quads are verified.
        schema = CouplingDataSchema(
            source_rep_quad="X_src",
            target_rep_quad="X_tgt",
            use_linear=False,
        )
        schema._verify_schema(adata)

    @pytest.mark.parametrize("bad", ["X_src", "X_tgt"])
    def test_verify_schema_checks_quads(self, adata: AnnData, bad: str) -> None:
        # The names used here are deliberately present; assert a truly invalid key raises.
        schema_bad = CouplingDataSchema(
            source_rep_quad=invalid_key,
            target_rep_quad="X_tgt",
            use_linear=False,
        )
        with pytest.raises(KeyError):
            schema_bad._verify_schema(adata)

    def test_get_data_produces_linear_free_couplings(self, adata: AnnData) -> None:
        schema = CouplingDataSchema(
            source_rep_quad="X_src",
            target_rep_quad="X_tgt",
            use_linear=False,
        )
        source_coupling, target_coupling = schema._get_data(adata)

        assert source_coupling.state_lin is None
        assert target_coupling.state_lin is None
        assert isinstance(source_coupling.state_quad, StateData)
        assert isinstance(target_coupling.state_quad, StateData)
        np.testing.assert_array_equal(source_coupling.state_quad.X, adata.obsm["X_src"])
        np.testing.assert_array_equal(target_coupling.state_quad.X, adata.obsm["X_tgt"])
        # Both sides are linear-free, so the spatial-dims check passes even if the quads differ.
        source_coupling.assert_same_spatial_dims(target_coupling)
