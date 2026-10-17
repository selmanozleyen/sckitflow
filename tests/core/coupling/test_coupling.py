import inspect

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from scfit.registry import parse

pytest.importorskip("ot")

from sckitflow.core.coupling import (
    CouplingConfig,
    independent_coupling,
    ot_linear_coupling,
    ot_quadratic_coupling,
)


@pytest.fixture
def small_tensors():
    source = torch.rand(5, 3)
    target = torch.rand(6, 3)
    return source, target


# ===============================================================
# Tests for independent_coupling
# ===============================================================
def test_independent_coupling_returns_indices(small_tensors):
    source, target = small_tensors
    src_idx, tgt_idx = independent_coupling(source, target, rng=np.random.default_rng(0))

    # Basic type and shape checks
    assert isinstance(src_idx, np.ndarray)
    assert isinstance(tgt_idx, np.ndarray)
    assert src_idx.ndim == 0 or src_idx.ndim == 1
    assert tgt_idx.ndim == 0 or tgt_idx.ndim == 1
    assert len(src_idx) == len(tgt_idx)
    assert (src_idx >= 0).all() and (src_idx < len(source)).all()
    assert (tgt_idx >= 0).all() and (tgt_idx < len(target)).all()


def test_independent_coupling_reproducibility(small_tensors):
    """Ensure deterministic behavior if the generator is seeded."""
    source, target = small_tensors
    idx1 = independent_coupling(source, target, rng=np.random.default_rng(42))
    idx2 = independent_coupling(source, target, rng=np.random.default_rng(42))
    assert all(np.allclose(a, b) for a, b in zip(idx1, idx2, strict=False))


# ===============================================================
# Tests for ot_linear_coupling
# ===============================================================
@pytest.mark.parametrize("method", ["exact", "sinkhorn"])
def test_ot_linear_coupling_basic(method, small_tensors):
    source, target = small_tensors
    src_idx, tgt_idx = ot_linear_coupling(source, target, method=method, rng=np.random.default_rng(0))

    assert isinstance(src_idx, np.ndarray)
    assert isinstance(tgt_idx, np.ndarray)
    assert len(src_idx) == len(source)
    assert len(tgt_idx) == len(source)
    assert (src_idx >= 0).all() and (src_idx < len(source)).all()
    assert (tgt_idx >= 0).all() and (tgt_idx < len(target)).all()


def test_ot_linear_coupling_invalid_method_raises(small_tensors):
    source, target = small_tensors
    with pytest.raises(ValueError):
        ot_linear_coupling(source, target, method="invalid", reg=5e-1, rng=np.random.default_rng(0))


@pytest.mark.parametrize("scale_cost", ["mean", "max", "median", 1.0])
def test_ot_linear_coupling_scale_modes(scale_cost, small_tensors):
    source, target = small_tensors
    src_idx, tgt_idx = ot_linear_coupling(
        source, target, scale_cost=scale_cost, method="sinkhorn", rng=np.random.default_rng(0)
    )

    dim_out = min(len(src_idx), len(tgt_idx))
    assert isinstance(src_idx, np.ndarray)
    assert len(src_idx) == dim_out
    assert isinstance(tgt_idx, np.ndarray)
    assert len(tgt_idx) == dim_out


# ===============================================================
# Tests for ot_quadratic_coupling
# ===============================================================
@pytest.fixture
def quadratic_inputs():
    torch.manual_seed(0)
    src_xx = torch.rand(4, 3)
    tgt_yy = torch.rand(5, 3)
    src_xy = torch.rand(4, 3)
    tgt_xy = torch.rand(5, 3)
    return src_xx, tgt_yy, src_xy, tgt_xy


@pytest.mark.parametrize("method", ["entropic_gromov_wasserstein", "entropic_fused_gromov_wasserstein"])
def test_ot_quadratic_coupling_basic(quadratic_inputs, method):
    src_xx, tgt_yy, src_xy, tgt_xy = quadratic_inputs
    src_idx, tgt_idx = ot_quadratic_coupling(
        source_quad=src_xx,
        target_quad=tgt_yy,
        source_lin=src_xy,
        target_lin=tgt_xy,
        method=method,
        rng=np.random.default_rng(0),
    )

    dim_out = min(len(src_xx), len(tgt_yy))
    assert isinstance(src_idx, np.ndarray)
    assert isinstance(tgt_idx, np.ndarray)
    assert isinstance(src_idx, np.ndarray)
    assert len(src_idx) == dim_out
    assert isinstance(tgt_idx, np.ndarray)
    assert len(tgt_idx) == dim_out
    assert (src_idx >= 0).all() and (tgt_idx >= 0).all()


@pytest.mark.xfail(
    reason="https://github.com/theislab/sckitflow/issues/145 - the optional linear terms are converted before the None check",
    strict=True,
)
def test_ot_quadratic_coupling_without_xy_couplings(quadratic_inputs):
    src_xx, tgt_yy, *_ = quadratic_inputs
    src_idx, tgt_idx = ot_quadratic_coupling(
        source_quad=src_xx,
        target_quad=tgt_yy,
        method="entropic_gromov_wasserstein",
        rng=np.random.default_rng(0),
    )

    dim_out = min(len(src_xx), len(tgt_yy))
    assert isinstance(src_idx, np.ndarray)
    assert len(src_idx) == dim_out
    assert isinstance(tgt_idx, np.ndarray)
    assert len(tgt_idx) == dim_out


def test_ot_quadratic_coupling_invalid_method_raises(quadratic_inputs):
    src_xx, tgt_yy, *_ = quadratic_inputs
    with pytest.raises(ValueError):
        ot_quadratic_coupling(
            source_quad=src_xx, target_quad=tgt_yy, method="invalid_method", rng=np.random.default_rng(0)
        )


# ===============================================================
# Tests for the coupling configs
# ===============================================================
def _all_subclasses(cls):
    return [c for sub in cls.__subclasses__() for c in (sub, *_all_subclasses(sub))]


@pytest.mark.parametrize(
    "spec",
    [
        {"type": "coupling.independent"},
        {"type": "coupling.pot.exact", "num_iter_max": 50},
        {"type": "coupling.pot.sinkhorn", "reg": 0.1},
        {"type": "coupling.pot.partial", "m": 0.5, "num_iter_max": 50},
        {"type": "coupling.pot.unbalanced", "reg_m": 2.0},
        {"type": "coupling.pot.gromov_wasserstein", "max_iter": 5},
        {"type": "coupling.pot.fused_gromov_wasserstein", "alpha": 0.3, "max_iter": 5},
    ],
)
def test_coupling_config_builds_a_match_fn(spec, small_tensors):
    source, target = small_tensors
    match_fn = parse({**spec, "version": 1}).build()
    src_idx, tgt_idx = match_fn(
        source_lin=source, target_lin=target, source_quad=source, target_quad=target, rng=np.random.default_rng(0)
    )
    assert len(src_idx) == len(tgt_idx) == len(source)


@pytest.mark.parametrize(
    "spec",
    [
        {"type": "coupling.pot.sinkhorn", "regg": 0.1},
        {"type": "coupling.pot.exact", "reg": 0.1},
        {"type": "coupling.pot.sinkhorn", "m": 0.5},
        {"type": "coupling.pot.gromov_wasserstein", "alpha": 0.3},
    ],
)
def test_coupling_config_rejects_settings_its_solver_ignores(spec):
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        parse({**spec, "version": 1})


@pytest.mark.parametrize("config", _all_subclasses(CouplingConfig), ids=lambda c: c.__name__)
def test_coupling_config_defaults_match_its_fn(config):
    params = inspect.signature(config.fn).parameters
    assert {name: field.default for name, field in config.model_fields.items()} == {
        name: params[name].default for name in config.model_fields
    }
