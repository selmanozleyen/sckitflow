from __future__ import annotations

import inspect
from collections.abc import Iterable
from typing import Any

import torch
from pydantic import ConfigDict, ValidationInfo, create_model, field_validator
from scfit.registry import Component, component

__all__ = ["OptimizerConfig"]


@component("optimizer.torch", builds=torch.optim.Optimizer)
class OptimizerConfig(Component):
    """A ``torch.optim`` optimizer by class name, e.g. ``OptimizerConfig(name="Adam", kwargs={"lr": 1e-4})``."""

    name: str = "Adam"
    """A class in ``torch.optim``."""
    kwargs: dict[str, Any] = {}
    """Arguments of that class, validated against its own typed signature, so ``lr: 1e-4`` (a string in YAML)
    becomes a float and a misspelled key is rejected."""

    @field_validator("name")
    @classmethod
    def _is_optimizer(cls, name: str) -> str:
        cls_ = getattr(torch.optim, name, None)
        if not isinstance(cls_, type) or not issubclass(cls_, torch.optim.Optimizer) or cls_ is torch.optim.Optimizer:
            raise ValueError(f"torch.optim has no optimizer {name!r}.")
        return name

    @field_validator("kwargs")
    @classmethod
    def _match_signature(cls, kwargs: dict[str, Any], info: ValidationInfo) -> dict[str, Any]:
        if "name" not in info.data:  # the name already failed
            return kwargs
        params = inspect.signature(getattr(torch.optim, info.data["name"])).parameters
        fields: Any = {
            k: (Any if p.annotation is p.empty else p.annotation, p.default) for k, p in params.items() if k != "params"
        }
        signature = create_model(
            info.data["name"], __config__=ConfigDict(extra="forbid", arbitrary_types_allowed=True), **fields
        )
        return signature(**kwargs).model_dump(exclude_unset=True)

    def build(self, params: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
        return getattr(torch.optim, self.name)(params, **self.kwargs)
