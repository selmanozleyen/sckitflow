"""Portable configs for the samplers a flow method draws time and noise with.

Each config binds ``fn`` — a callable with signature
``(shape, *, generator, device, dtype) -> torch.Tensor`` — as a ``ClassVar``,
so the class itself is what ``scfit``'s ``Component`` serializes. ``build``
returns ``fn`` directly when there are no fields, and ``partial(fn, **fields)``
when a subclass adds options. This mirrors :class:`sckitflow.core.coupling.CouplingConfig`
and is what lets ``save_run`` write ``specs.json`` — a live callable on the
spec is a ``PortabilityError`` waiting to happen.

Subclasses can be added freely. Two common patterns:

* Override ``fn`` on a config that takes no fields: ``UniformTimeSamplerConfig``.
* Add fields and let ``build`` bake them into the returned ``partial``:
  e.g. a ``BetaTimeSamplerConfig(alpha=2.0, beta=5.0)`` whose ``fn`` reads
  ``alpha``/``beta`` as keywords. ``save_run`` writes the fields; the method
  that consumes the config resolves to a ``partial`` at build time.

The two defaults below match the previous raw-callable fallbacks
(``torch.rand`` / ``torch.randn``), so callers that leave a sampler unset
see no behavioral change.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import ClassVar

import torch
from scfit.registry import Component, component

__all__ = [
    "SamplerConfig",
    "UniformTimeSamplerConfig",
    "IsotropicGaussianNoiseSamplerConfig",
]


class SamplerConfig(Component):
    """Family base for the samplers. ``build`` returns the callable ``fn``, optionally partialed.

    Subclasses that add fields bake them into the returned callable, exactly
    the way :class:`sckitflow.core.coupling.CouplingConfig` subclasses bake
    POT options into ``match_fn``. A spec written by ``save_run`` therefore
    carries only scalars and strings, never a Python function.
    """

    fn: ClassVar[Callable[..., torch.Tensor]]

    def build(self) -> Callable[..., torch.Tensor]:
        kwargs = dict(self)
        if not kwargs:
            return type(self).fn
        return partial(type(self).fn, **kwargs)


def _uniform(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.types.Device = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Uniform samples in ``[0, 1)``, drawn from ``generator``."""
    return torch.rand(shape, generator=generator, device=device, dtype=dtype)


def _standard_normal(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.types.Device = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Standard normal samples, drawn from ``generator``."""
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)


@component("time_sampler.uniform")
class UniformTimeSamplerConfig(SamplerConfig):
    """Uniform time samples in ``[0, 1)``."""

    fn = _uniform


@component("noise_sampler.isotropic_gaussian")
class IsotropicGaussianNoiseSamplerConfig(SamplerConfig):
    """Standard normal noise samples — the CFM source-distribution default."""

    fn = _standard_normal
