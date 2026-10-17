"""Portable configs for the couplings a `Matcher` pairs a batch with."""

from __future__ import annotations

from functools import partial

from scfit.registry import Component, component

from sckitflow.core._types import LinCouplingMethod, MatchFn, QuadCouplingMethod, ScaleMethod
from sckitflow.core.coupling._coupling import independent_coupling, ot_linear_coupling, ot_quadratic_coupling

__all__ = ["CouplingConfig", "IndependentCouplingConfig", "OTLinearCouplingConfig", "OTQuadraticCouplingConfig"]


class CouplingConfig(Component):
    """Family base for the couplings."""

    def build(self) -> MatchFn:
        raise NotImplementedError


@component("coupling.independent")
class IndependentCouplingConfig(CouplingConfig):
    """Random pairs."""

    def build(self) -> MatchFn:
        return independent_coupling


@component("coupling.ot_linear")
class OTLinearCouplingConfig(CouplingConfig):
    """Linear optimal transport between the source and target reps. See `ot_linear_coupling`."""

    method: LinCouplingMethod = None
    reg: float = 0.5
    reg_m: float = 1.0
    scale_cost: ScaleMethod = "mean"
    m: float | None = None
    num_iter_max: int | None = None
    stop_thr: float | None = None

    def build(self) -> MatchFn:
        return partial(
            ot_linear_coupling,
            method=self.method,
            reg=self.reg,
            reg_m=self.reg_m,
            scale_cost=self.scale_cost,
            m=self.m,
            num_iter_max=self.num_iter_max,
            stop_thr=self.stop_thr,
        )


@component("coupling.ot_quadratic")
class OTQuadraticCouplingConfig(CouplingConfig):
    """(Fused) Gromov-Wasserstein between the source and target reps. See `ot_quadratic_coupling`."""

    method: QuadCouplingMethod = None
    scale_cost: ScaleMethod = 1.0
    epsilon: float = 1.0
    alpha: float = 0.5
    max_iter: int = 1000
    tol: float = 1e-9

    def build(self) -> MatchFn:
        return partial(
            ot_quadratic_coupling,
            method=self.method,
            scale_cost=self.scale_cost,
            epsilon=self.epsilon,
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol,
        )
