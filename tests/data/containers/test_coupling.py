import numpy as np
import pytest
from anndata import AnnData

from sckitflow.data.containers import CouplingData, StateData


class TestCouplingData:
    def test_init_lin_only(self, adata: AnnData) -> None:
        state_lin = StateData(adata.X)
        coupling = CouplingData(state_lin=state_lin)

        assert isinstance(coupling, CouplingData)
        assert coupling.state_lin is state_lin
        assert coupling.state_quad is None

    def test_init_lin_and_quad(self, adata: AnnData) -> None:
        state_lin = StateData(adata.X)
        state_quad = StateData(adata.X)

        coupling = CouplingData(state_lin=state_lin, state_quad=state_quad)

        assert coupling.state_lin is state_lin
        assert coupling.state_quad is state_quad

    def test_init_quad_only(self, adata: AnnData) -> None:
        state_quad = StateData(adata.X)
        coupling = CouplingData(state_quad=state_quad)

        assert coupling.state_lin is None
        assert coupling.state_quad is state_quad

    def test_init_both_none_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one of"):
            CouplingData()

    def test_post_init_mismatched_n_obs(self, adata: AnnData) -> None:
        X = adata.X
        state_lin = StateData(X)
        state_quad = StateData(X[:-1])

        with pytest.raises(ValueError):
            CouplingData(state_lin=state_lin, state_quad=state_quad)

    def test_post_init_matching_n_obs(self, adata: AnnData) -> None:
        X = adata.X
        # Should not raise when n_obs match.
        CouplingData(state_lin=StateData(X), state_quad=StateData(X))

    def test_post_init_only_lin(self, adata: AnnData) -> None:
        # No state_quad: length check is skipped.
        CouplingData(state_lin=StateData(adata.X))

    def test_post_init_only_quad(self, adata: AnnData) -> None:
        # No state_lin: length check is skipped.
        CouplingData(state_quad=StateData(adata.X))

    def test_len_lin_only(self, adata: AnnData) -> None:
        assert len(CouplingData(state_lin=StateData(adata.X))) == adata.n_obs

    def test_len_lin_and_quad(self, adata: AnnData) -> None:
        coupling = CouplingData(
            state_lin=StateData(adata.X),
            state_quad=StateData(adata.X),
        )
        assert len(coupling) == adata.n_obs

    def test_len_quad_only(self, adata: AnnData) -> None:
        assert len(CouplingData(state_quad=StateData(adata.X))) == adata.n_obs

    @pytest.mark.parametrize("idxs", [slice(0, 5), np.array([0, 2, 4])])
    def test_getitem_with_quad(self, adata: AnnData, idxs) -> None:
        state_lin = StateData(adata.X)
        state_quad = StateData(adata.X)
        coupling = CouplingData(state_lin=state_lin, state_quad=state_quad)

        subset = coupling[idxs]

        assert isinstance(subset, CouplingData)
        assert subset.state_lin is not state_lin
        assert subset.state_quad is not state_quad
        assert subset.state_lin.X.shape[1] == state_lin.X.shape[1]
        assert subset.state_quad.X.shape[1] == state_quad.X.shape[1]
        np.testing.assert_array_equal(subset.state_lin.X, state_lin.X[idxs])
        np.testing.assert_array_equal(subset.state_quad.X, state_quad.X[idxs])

    @pytest.mark.parametrize("idxs", [slice(0, 5), np.array([0, 2, 4])])
    def test_getitem_no_quad(self, adata: AnnData, idxs) -> None:
        state_lin = StateData(adata.X)
        coupling = CouplingData(state_lin=state_lin)

        subset = coupling[idxs]

        assert isinstance(subset, CouplingData)
        assert subset.state_quad is None
        np.testing.assert_array_equal(subset.state_lin.X, state_lin.X[idxs])

    @pytest.mark.parametrize("idxs", [slice(0, 5), np.array([0, 2, 4])])
    def test_getitem_no_lin(self, adata: AnnData, idxs) -> None:
        state_quad = StateData(adata.X)
        coupling = CouplingData(state_quad=state_quad)

        subset = coupling[idxs]

        assert isinstance(subset, CouplingData)
        assert subset.state_lin is None
        np.testing.assert_array_equal(subset.state_quad.X, state_quad.X[idxs])

    def test_assert_same_spatial_dims_valid(self, adata: AnnData) -> None:
        state_data = StateData(adata.X)
        c1 = CouplingData(state_lin=state_data)
        c2 = CouplingData(state_lin=state_data)

        c1.assert_same_spatial_dims(c2)

    def test_assert_same_spatial_dims_invalid(self, adata: AnnData) -> None:
        n_dims = adata.X.shape[1]
        c1 = CouplingData(state_lin=StateData(adata.X[:, : n_dims // 2]))
        c2 = CouplingData(state_lin=StateData(adata.X))

        with pytest.raises(ValueError, match="same number of spatial dimensions"):
            c1.assert_same_spatial_dims(c2)

    def test_assert_same_spatial_dims_both_lin_free(self, adata: AnnData) -> None:
        """Pure GW on both sides: quadratic widths may differ, and the check passes."""
        n_dims = adata.X.shape[1]
        c1 = CouplingData(state_quad=StateData(adata.X[:, : n_dims // 2]))
        c2 = CouplingData(state_quad=StateData(adata.X))

        c1.assert_same_spatial_dims(c2)

    def test_assert_same_spatial_dims_one_lin_free(self, adata: AnnData) -> None:
        c1 = CouplingData(state_quad=StateData(adata.X))
        c2 = CouplingData(state_lin=StateData(adata.X))

        with pytest.raises(ValueError, match="disagree on whether a linear term"):
            c1.assert_same_spatial_dims(c2)

    def test_repr_with_quad(self, adata: AnnData) -> None:
        coupling = CouplingData(
            state_lin=StateData(adata.X),
            state_quad=StateData(adata.X),
        )

        rep = repr(coupling)

        assert "CouplingData" in rep
        assert "linear" in rep
        assert "quadratic" in rep
        assert f"n_obs={adata.n_obs}" in rep

    def test_repr_no_quad(self, adata: AnnData) -> None:
        coupling = CouplingData(state_lin=StateData(adata.X))

        rep = repr(coupling)

        assert "CouplingData" in rep
        assert "quadratic=None" in rep
        assert f"n_obs={adata.n_obs}" in rep

    def test_repr_no_lin(self, adata: AnnData) -> None:
        coupling = CouplingData(state_quad=StateData(adata.X))

        rep = repr(coupling)

        assert "CouplingData" in rep
        assert "linear=None" in rep
        assert "quadratic(spatial_dims=" in rep
        assert f"n_obs={adata.n_obs}" in rep
