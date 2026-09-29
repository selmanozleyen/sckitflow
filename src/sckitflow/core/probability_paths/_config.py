"""Portable configs for the probability paths.

A path is only its parameters. A stochastic path draws its noise from the
``generator`` passed to each :meth:`~BaseProbabilityPath.compute_xt` call, so no
config holds a seed.
"""

from __future__ import annotations

from dataclasses import dataclass

from scfit.registry import Component

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

    def build(self, context: None = None) -> BaseProbabilityPath:
        raise NotImplementedError


@dataclass(frozen=True)
class LinearDiracConfig(ProbabilityPathConfig, type_id="probability_path.linear_dirac", version=1):
    """Straight-line interpolation to a Dirac target. Deterministic."""

    def build(self, context: None = None) -> LinearDiracProbabilityPath:
        return LinearDiracProbabilityPath(sigma=self.sigma)


@dataclass(frozen=True)
class LinearGaussianConfig(ProbabilityPathConfig, type_id="probability_path.linear_gaussian", version=1):
    """Straight-line interpolation with Gaussian noise."""

    def build(self, context: None = None) -> LinearGaussianProbabilityPath:
        return LinearGaussianProbabilityPath(sigma=self.sigma)


@dataclass(frozen=True)
class SchrodingerBridgeConfig(ProbabilityPathConfig, type_id="probability_path.schrodinger_bridge", version=1):
    """Schrödinger-bridge path.

    :param eps: The bridge's entropic regularization.
    """

    eps: float = 1e-3

    def build(self, context: None = None) -> SchrodingerBridgeProbabilityPath:
        return SchrodingerBridgeProbabilityPath(sigma=self.sigma, eps=self.eps)


@dataclass(frozen=True)
class VariancePreservingDiracConfig(
    ProbabilityPathConfig, type_id="probability_path.variance_preserving_dirac", version=1
):
    """Variance-preserving path to a Dirac target. Deterministic."""

    def build(self, context: None = None) -> VariancePreservingDiracProbabilityPath:
        return VariancePreservingDiracProbabilityPath(sigma=self.sigma)
