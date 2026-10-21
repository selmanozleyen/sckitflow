import pytest

from sckitflow.data.schemas import CouplingDataSchema

invalid_key = "invalid_key"


class TestCouplingDataSchema:
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
