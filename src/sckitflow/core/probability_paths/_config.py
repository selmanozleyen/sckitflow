"""Portable configs for the probability paths.

A path is a `sigma` and, at most, a seeded generator, so it is portable. ``prng``
is the one runtime piece: the config carries ``seed`` and builds the generator.
"""

from __future__ import annotations

import torch
from scfit.registry import Component, component

from sckitflow.core.probability_paths._probability_paths import (
    BaseProbabilityPath,
    LinearDiracProbabilityPath,
    LinearGaussianProbabilityPath,
    SchrodingerBridgeProbabilityPath,
    VariancePreservingDiracProbabilityPath,
)

__all__ = [
    "ProbabilityPathConfig",
    "LinearDiracProbabilityPathConfig",
    "LinearGaussianProbabilityPathConfig",
    "SchrodingerBridgeProbabilityPathConfig",
    "VariancePreservingDiracProbabilityPathConfig",
]


@component()
class ProbabilityPathConfig(Component):
    """Family base for the probability paths.

    :param sigma: The path's noise scale.
    """

    sigma: float = 0.0

    def build(self) -> BaseProbabilityPath:
        raise NotImplementedError


@component()
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


@component("probability_path.linear_dirac", builds=LinearDiracProbabilityPath)
class LinearDiracProbabilityPathConfig(ProbabilityPathConfig):
    """Straight-line interpolation to a Dirac target. Deterministic."""

    def build(self) -> LinearDiracProbabilityPath:
        return LinearDiracProbabilityPath(sigma=self.sigma)


@component("probability_path.linear_gaussian", builds=LinearGaussianProbabilityPath)
class LinearGaussianProbabilityPathConfig(_SeededPathConfig):
    """Straight-line interpolation with Gaussian noise."""

    def build(self) -> LinearGaussianProbabilityPath:
        return LinearGaussianProbabilityPath(sigma=self.sigma, prng=self._prng())


@component("probability_path.schrodinger_bridge", builds=SchrodingerBridgeProbabilityPath)
class SchrodingerBridgeProbabilityPathConfig(_SeededPathConfig):
    """Schrödinger-bridge path.

    :param eps: The bridge's entropic regularization.
    """

    eps: float = 1e-3

    def build(self) -> SchrodingerBridgeProbabilityPath:
        return SchrodingerBridgeProbabilityPath(sigma=self.sigma, prng=self._prng(), eps=self.eps)


@component("probability_path.variance_preserving_dirac", builds=VariancePreservingDiracProbabilityPath)
class VariancePreservingDiracProbabilityPathConfig(ProbabilityPathConfig):
    """Variance-preserving path to a Dirac target. Deterministic."""

    def build(self) -> VariancePreservingDiracProbabilityPath:
        return VariancePreservingDiracProbabilityPath(sigma=self.sigma)
