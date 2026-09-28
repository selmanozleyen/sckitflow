"""Portable configs for the training and inference methods.

A config holds the portable parameters; ``build(module)`` makes the runtime
method around the neural module. Anything live on a config is marked with
:func:`scfit.registry.register_live`, so it still builds but refuses to
serialize.
"""

from __future__ import annotations

from dataclasses import field
from typing import Any

import torch
from scfit.registry import Component, component, register_live

from sckitflow.core._types import SamplerFn
from sckitflow.core.methods._base import SupportsInference, SupportsTraining
from sckitflow.core.methods.inference._ode import ODEInference
from sckitflow.core.methods.training._cfm import CFMTraining
from sckitflow.core.probability_paths._config import ProbabilityPathConfig
from sckitflow.core.probability_paths._probability_paths import BaseProbabilityPath

__all__ = [
    "TrainingMethodConfig",
    "InferenceMethodConfig",
    "CFMConfig",
    "ODEConfig",
]

# Runtime-only types: a config may hold one to build and train with, but asking
# that config for a portable spec raises instead of dropping it on the floor.
# Genuinely runtime-only: a module's learned weights belong in a `state_dict`,
# not a spec, and a `torch.Generator` has no portable form (seed it via config).
register_live(torch.nn.Module)
register_live(torch.Generator)
# A live path still builds and trains; prefer `ProbabilityPathConfig`, which serializes.
register_live(BaseProbabilityPath)


class TrainingMethodConfig(Component):
    """Family base for anything that configures a training method."""

    def build(self, module: torch.nn.Module) -> SupportsTraining:
        """The training method around ``module``."""
        raise NotImplementedError


class InferenceMethodConfig(Component):
    """Family base for anything that configures an inference method."""

    def build(self, module: torch.nn.Module) -> SupportsInference:
        """The inference method around ``module``."""
        raise NotImplementedError


@component()
class _FlowConfig(Component):
    """The flow parameters every flow method shares.

    ``None`` for a sampler or path means "the method's own default", which is
    portable. A custom instance or callable is live, so a config holding one
    builds fine and refuses to serialize.
    """

    probability_path: ProbabilityPathConfig | BaseProbabilityPath | None = None
    """A `ProbabilityPathConfig` (portable) or a live path (builds, will not serialize)."""
    time_sampler: SamplerFn | None = None
    noise_sampler: SamplerFn | None = None
    generate_from_noise: bool = False

    def _flow_kwargs(self) -> dict[str, Any]:
        return {
            "probability_path": (
                self.probability_path.build()
                if isinstance(self.probability_path, ProbabilityPathConfig)
                else self.probability_path
            ),
            "time_sampler": self.time_sampler,
            "noise_sampler": self.noise_sampler,
            "generate_from_noise": self.generate_from_noise,
        }


@component("training_method.cfm")
class CFMConfig(_FlowConfig, TrainingMethodConfig):
    """Conditional Flow Matching training."""

    def build(self, module: torch.nn.Module) -> CFMTraining:
        return CFMTraining(module=module, **self._flow_kwargs())


@component("inference_method.ode")
class ODEConfig(_FlowConfig, InferenceMethodConfig):
    """ODE inference over a trained velocity field."""

    solver_kwargs: dict[str, Any] = field(default_factory=dict)
    return_trajectory: bool = False
    n_steps: int = 100
    n_samples: int | None = None

    def build(self, module: torch.nn.Module) -> ODEInference:
        return ODEInference(
            module=module,
            solver_kwargs=dict(self.solver_kwargs),
            return_trajectory=self.return_trajectory,
            n_steps=self.n_steps,
            n_samples=self.n_samples,
            **self._flow_kwargs(),
        )
