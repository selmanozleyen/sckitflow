"""Portable configs for the couplings a `Matcher` pairs a batch with.

Each config only binds its fields to a coupling function, which delegates the solve to POT. Which
solvers exist, what they accept and the couplings they return therefore depend on the installed
POT version as well.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import ClassVar, Self

from pydantic import model_validator
from scfit.registry import Component, component

from sckitflow.core._types import LinCouplingMethod, MatchFn, MatchFnOut, QuadCouplingMethod, ScaleMethod
from sckitflow.core.coupling._coupling import independent_coupling, ot_linear_coupling, ot_quadratic_coupling

__all__ = ["CouplingConfig", "IndependentCouplingConfig", "OTLinearCouplingConfig", "OTQuadraticCouplingConfig"]


class CouplingConfig(Component):
    """Family base for the couplings. ``build`` binds every field to ``fn`` as a keyword argument.

    Field defaults must equal ``fn``'s, which the tests check.
    """

    fn: ClassVar[Callable[..., MatchFnOut]]
    only_for: ClassVar[dict[str, frozenset[str]]] = {}
    """Field -> the ``method`` values ``fn`` reads it for. Changing it from its default for any other method is
    rejected. Saved specs write every field, so the default itself has to pass."""

    @model_validator(mode="after")
    def _only_for_its_methods(self) -> Self:
        method = getattr(self, "method", None)
        for name, methods in sorted(self.only_for.items()):
            if method not in methods and getattr(self, name) != type(self).model_fields[name].default:
                msg = f"`{name}` is only read by method {sorted(methods)}, not {method!r}."
                raise ValueError(msg)
        return self

    def build(self) -> MatchFn:
        return partial(type(self).fn, **dict(self))


@component("coupling.independent")
class IndependentCouplingConfig(CouplingConfig):
    """Random pairs."""

    fn = independent_coupling


_ENTROPIC = frozenset({"sinkhorn", "partial", "unbalanced"})


@component("coupling.ot_linear")
class OTLinearCouplingConfig(CouplingConfig):
    """Linear optimal transport, see `ot_linear_coupling` for the fields."""

    fn = ot_linear_coupling
    only_for = {
        "reg": _ENTROPIC,
        "stop_thr": _ENTROPIC,
        "reg_m": frozenset({"unbalanced"}),
        "m": frozenset({"partial"}),
    }

    method: LinCouplingMethod = "sinkhorn"
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
    only_for = {"alpha": frozenset({"entropic_fused_gromov_wasserstein"})}

    method: QuadCouplingMethod = "entropic_gromov_wasserstein"
    scale_cost: ScaleMethod = 1.0
    epsilon: float = 1.0
    alpha: float = 0.5
    max_iter: int = 1000
    tol: float = 1e-9
