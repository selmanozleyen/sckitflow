"""Portable configs for the probability paths.

A path is only its parameters. A stochastic path draws its noise from the
``generator`` passed to each :meth:`~BaseProbabilityPath.compute_xt` call, so no
config holds a seed.
"""

from __future__ import annotations

from scfit.registry import Builds, Component, component

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
class ProbabilityPathConfig[T: BaseProbabilityPath](Builds[T], Component):
    """Family base for the probability paths; ``T`` is the path it builds.

    :param sigma: The path's noise scale.
    """

    sigma: float = 0.0

    def build(self) -> T:
        raise NotImplementedError


@component("probability_path.linear_dirac")
class LinearDiracProbabilityPathConfig(ProbabilityPathConfig[LinearDiracProbabilityPath]):
    """Straight-line interpolation to a Dirac target. Deterministic."""

    def build(self) -> LinearDiracProbabilityPath:
        return LinearDiracProbabilityPath(sigma=self.sigma)


@component("probability_path.linear_gaussian")
class LinearGaussianProbabilityPathConfig(ProbabilityPathConfig[LinearGaussianProbabilityPath]):
    """Straight-line interpolation with Gaussian noise."""

    def build(self) -> LinearGaussianProbabilityPath:
        return LinearGaussianProbabilityPath(sigma=self.sigma)


@component("probability_path.schrodinger_bridge")
class SchrodingerBridgeProbabilityPathConfig(ProbabilityPathConfig[SchrodingerBridgeProbabilityPath]):
    """Schrödinger-bridge path.

    :param eps: The bridge's entropic regularization.
    """

    eps: float = 1e-3

    def build(self) -> SchrodingerBridgeProbabilityPath:
        return SchrodingerBridgeProbabilityPath(sigma=self.sigma, eps=self.eps)


@component("probability_path.variance_preserving_dirac")
class VariancePreservingDiracProbabilityPathConfig(ProbabilityPathConfig[VariancePreservingDiracProbabilityPath]):
    """Variance-preserving path to a Dirac target. Deterministic."""

    def build(self) -> VariancePreservingDiracProbabilityPath:
        return VariancePreservingDiracProbabilityPath(sigma=self.sigma)
