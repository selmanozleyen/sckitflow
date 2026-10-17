from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from pydantic import field_validator
from scfit.registry import Component, component

__all__ = ["OptimizerConfig"]


# TODO: remove this config. Name + untyped kwargs is a stopgap; decide how a run spec should carry the optimizer
# (e.g. Lightning's `configure_optimizers`, or a typed per-optimizer component) before more code depends on it.
@component("optimizer.torch", builds=torch.optim.Optimizer)
class OptimizerConfig(Component):
    """A ``torch.optim`` optimizer by class name, e.g. ``OptimizerConfig(name="Adam", kwargs={"lr": 1e-4})``."""

    name: str = "Adam"
    """A class in ``torch.optim``."""
    kwargs: dict[str, Any] = {}
    """Arguments of that class. Its own signature checks them when :meth:`build` runs. In YAML write
    ``1.0e-4``: PyYAML reads ``1e-4`` as a string."""

    @field_validator("name")
    @classmethod
    def _is_optimizer(cls, name: str) -> str:
        cls_ = getattr(torch.optim, name, None)
        if not isinstance(cls_, type) or not issubclass(cls_, torch.optim.Optimizer) or cls_ is torch.optim.Optimizer:
            raise ValueError(f"torch.optim has no optimizer {name!r}.")
        return name

    def build(self, params: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
        return getattr(torch.optim, self.name)(params, **self.kwargs)
