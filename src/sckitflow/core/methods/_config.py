"""Portable configs for the training and inference methods.

Built on :mod:`scfit.registry`, which is scfit's shared foundation for exactly
this -- a ``type_id`` slug in the class header auto-registers a config, and
:func:`scfit.registry.to_spec` turns it into a portable
``{type, version, config}`` dict. Using it rather than a private registry means
a sckitflow spec is readable by anything else in the ecosystem built on scfit.

The split it gives us:

* a **config** holds the portable parameters and nothing else;
* ``build(context)`` makes the runtime method, taking the non-portable pieces --
  the neural module above all -- from the ``context``;
* anything live that ends up *on* a config is marked with
  :func:`scfit.registry.register_live`, so constructing and training with it
  works while :func:`to_spec` raises :class:`~scfit.registry.PortabilityError`
  rather than silently dropping it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from scfit.registry import Component, register_live

from sckitflow.core._types import SamplerFn
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


# No `type_id`: these stay unregistered so they can be the `expected` family in
# `Family.from_spec(spec)`, which then rejects a spec of the wrong family.
class TrainingMethodConfig(Component):
    """Family base for anything that configures a training method."""


class InferenceMethodConfig(Component):
    """Family base for anything that configures an inference method."""


@dataclass
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


@dataclass
class CFMConfig(_FlowConfig, TrainingMethodConfig, type_id="training_method.cfm", version=1):
    """Conditional Flow Matching training."""

    def build(self, context: torch.nn.Module) -> CFMTraining:
        """:param context: The neural module to train."""
        return CFMTraining(module=context, **self._flow_kwargs())


@dataclass
class ODEConfig(_FlowConfig, InferenceMethodConfig, type_id="inference_method.ode", version=1):
    """ODE inference over a trained velocity field."""

    solver_kwargs: dict[str, Any] = field(default_factory=dict)
    return_trajectory: bool = False
    n_steps: int = 100
    n_samples: int | None = None

    def build(self, context: torch.nn.Module) -> ODEInference:
        """:param context: The neural module to integrate."""
        return ODEInference(
            module=context,
            solver_kwargs=dict(self.solver_kwargs),
            return_trajectory=self.return_trajectory,
            n_steps=self.n_steps,
            n_samples=self.n_samples,
            **self._flow_kwargs(),
        )
