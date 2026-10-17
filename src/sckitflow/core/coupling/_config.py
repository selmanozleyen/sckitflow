"""Portable configs for the couplings a `Matcher` pairs a batch with.

Each config only binds its fields to a coupling function, which delegates the solve to POT. Which
solvers exist, what they accept and the couplings they return therefore depend on the installed
POT version as well.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import ClassVar

from scfit.registry import Component, component

from sckitflow.core._types import LinCouplingMethod, MatchFn, MatchFnOut, QuadCouplingMethod, ScaleMethod
from sckitflow.core.coupling._coupling import independent_coupling, ot_linear_coupling, ot_quadratic_coupling

__all__ = ["CouplingConfig", "IndependentCouplingConfig", "OTLinearCouplingConfig", "OTQuadraticCouplingConfig"]


class CouplingConfig(Component):
    """Family base for the couplings. ``build`` binds every field to ``fn`` as a keyword argument."""

    fn: ClassVar[Callable[..., MatchFnOut]]

    def build(self) -> MatchFn:
        return partial(type(self).fn, **dict(self))


@component("coupling.independent")
class IndependentCouplingConfig(CouplingConfig):
    """Random pairs."""

    fn = independent_coupling


@component("coupling.ot_linear")
class OTLinearCouplingConfig(CouplingConfig):
    """Linear optimal transport, see `ot_linear_coupling` for the fields."""

    fn = ot_linear_coupling

    method: LinCouplingMethod = None
    reg: float = 0.5
    reg_m: float = 1.0
    scale_cost: ScaleMethod = "mean"
    m: float | None = None
    num_iter_max: int | None = None
    stop_thr: float | None = None


@component("coupling.ot_quadratic")
class OTQuadraticCouplingConfig(CouplingConfig):
    """(Fused) Gromov-Wasserstein, see `ot_quadratic_coupling` for the fields."""

    fn = ot_quadratic_coupling

    method: QuadCouplingMethod = None
    scale_cost: ScaleMethod = 1.0
    epsilon: float = 1.0
    alpha: float = 0.5
    max_iter: int = 1000
    tol: float = 1e-9
