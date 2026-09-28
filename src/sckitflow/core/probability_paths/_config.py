"""Portable configs for the probability paths.

A path is a `sigma` and, at most, a seeded generator -- so it is portable, and
does not belong behind :func:`scfit.registry.register_live`. ``prng`` is the one
runtime piece: the config carries ``seed`` and builds the generator.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from scfit.registry import Component

from sckitflow._components import config_fields
from sckitflow.core.probability_paths._probability_paths import (
    BaseProbabilityPath,
    LinearDiracProbabilityPath,
    LinearGaussianProbabilityPath,
    SchrodingerBridgeProbabilityPath,
    VariancePreservingDiracProbabilityPath,
)

__all__ = [
    "ProbabilityPathConfig",
    "LinearDiracConfig",
    "LinearGaussianConfig",
    "SchrodingerBridgeConfig",
    "VariancePreservingDiracConfig",
]


# No `type_id`: the family base stays unregistered so it can be the `expected`
# family in `ProbabilityPathConfig.from_spec(spec)`.
@dataclass(frozen=True)
class ProbabilityPathConfig(Component):
    """Family base for the probability paths.

    :param sigma: The path's noise scale.
    """

    sigma: float = 0.0

    def build(self, context: object = None) -> BaseProbabilityPath:
        raise NotImplementedError


@dataclass(frozen=True)
class _SeededPathConfig(ProbabilityPathConfig):
    """A path whose sampling is stochastic, so a seed is meaningful.

    The deterministic paths deliberately do not offer one -- they warn and
    discard a generator, so exposing `seed` there would be a knob that does
    nothing.

    :param seed: Seeds the path's generator; `None` leaves it unseeded.
    """

    seed: int | None = None

    def _prng(self) -> torch.Generator | None:
        return None if self.seed is None else torch.Generator().manual_seed(self.seed)


@dataclass(frozen=True)
class LinearDiracConfig(ProbabilityPathConfig, type_id="probability_path.linear_dirac", version=1):
    """Straight-line interpolation to a Dirac target. Deterministic."""

    def build(self, context: object = None) -> LinearDiracProbabilityPath:
        return LinearDiracProbabilityPath(**config_fields(self))


@dataclass(frozen=True)
class LinearGaussianConfig(_SeededPathConfig, type_id="probability_path.linear_gaussian", version=1):
    """Straight-line interpolation with Gaussian noise."""

    def build(self, context: object = None) -> LinearGaussianProbabilityPath:
        return LinearGaussianProbabilityPath(**config_fields(self, exclude=("seed",)), prng=self._prng())


@dataclass(frozen=True)
class SchrodingerBridgeConfig(_SeededPathConfig, type_id="probability_path.schrodinger_bridge", version=1):
    """Schrödinger-bridge path.

    :param eps: The bridge's entropic regularization.
    """

    eps: float = 1e-3

    def build(self, context: object = None) -> SchrodingerBridgeProbabilityPath:
        return SchrodingerBridgeProbabilityPath(**config_fields(self, exclude=("seed",)), prng=self._prng())


@dataclass(frozen=True)
class VariancePreservingDiracConfig(
    ProbabilityPathConfig, type_id="probability_path.variance_preserving_dirac", version=1
):
    """Variance-preserving path to a Dirac target. Deterministic."""

    def build(self, context: object = None) -> VariancePreservingDiracProbabilityPath:
        return VariancePreservingDiracProbabilityPath(**config_fields(self))
