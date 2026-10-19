"""Portable configs for the neural modules.

A module config holds the architecture only; the sizes the data fixes (state and condition dims) are
filled in by :meth:`ModuleConfig.build` from the data module's :class:`~sckitflow.data.DataDimensions`.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from pydantic import field_validator
from scfit.registry import Component, component

from sckitflow._types import ConditioningLayersId, TimeFeaturesId
from sckitflow.core.nn._modules import MLP
from sckitflow.core.nn._vf import MLPVelocity
from sckitflow.data._dims import DataDimensions

__all__ = ["ModuleConfig", "MLPConfig", "MLPVelocityConfig"]


# ---------------------------------------------------------------------------
# Activation resolution
#
# Serializability requires the config to store activations as strings ("silu"),
# not as classes (torch.nn.SiLU), because classes aren't JSON-serializable and
# `save_run` dumps configs to `specs.json`. `build()` resolves the strings
# back to classes before they reach `MLP.__init__`, where a class is required.
#
# Two directions, both handled below:
#   * `field_validator`  — normalizes classes → strings on construction
#   * `_resolve_...`     — resolves strings → classes at build time
# ---------------------------------------------------------------------------
def _resolve_nn_module_class(
    name: str,
    *,
    base_cls: type = torch.nn.Module,
) -> type[torch.nn.Module]:
    """Map a string name (``"SiLU"``, ``"silu"``, ``"ReLU"``, …) to its ``torch.nn`` class.

    Tries an exact match first, then a case-insensitive scan over ``torch.nn``
    attributes. Raises ``ValueError`` if no matching subclass exists — a bad
    name fails at config build, before the model is constructed.

    :param name: The class name (case-insensitive).
    :param base_cls: The class to look up. Defaults to :class:`torch.nn.Module`;
        pass e.g. :class:`torch.optim.Optimizer` for the analogous optimizer case.
    """
    # Exact match first: "SiLU" -> torch.nn.SiLU
    cls = getattr(torch.nn, name, None)
    if cls is not None and isinstance(cls, type) and issubclass(cls, base_cls) and cls is not base_cls:
        return cls

    # Case-insensitive fallback: "silu" -> torch.nn.SiLU
    for attr in dir(torch.nn):
        obj = getattr(torch.nn, attr)
        if isinstance(obj, type) and issubclass(obj, base_cls) and obj is not base_cls and attr.lower() == name.lower():
            return obj

    raise ValueError(
        f"torch.nn has no {base_cls.__name__} subclass named {name!r}. "
        f"Try e.g. 'SiLU', 'ReLU', 'GELU', 'LeakyReLU', 'Tanh'."
    )


_ACTIVATION_KEYS: tuple[str, ...] = ("activation_cls", "final_activation_cls")


def _normalize_activation_to_string(
    value: str | type[torch.nn.Module] | None,
) -> str | None:
    """Store an activation as a string, whether given a class or a string.

    ``torch.nn.SiLU`` → ``"SiLU"``; ``"silu"`` → ``"silu"``; ``None`` → ``None``.
    """
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, type) and issubclass(value, torch.nn.Module):
        return value.__name__
    raise TypeError(f"activation must be a str or a torch.nn.Module subclass, got {type(value).__name__}")


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


class ModuleConfig(Component):
    """Family base for anything that configures a neural module."""

    def build(self, dims: DataDimensions) -> torch.nn.Module:
        """The module, sized for ``dims``."""
        raise NotImplementedError


@component("mlp", builds=MLP)
class MLPConfig(ModuleConfig):
    hidden_dims: tuple[int, ...] | None = None
    activation_cls: str | None = None
    final_activation_cls: str | None = None
    use_batchnorm: bool = False
    batchnorm_eps: float = 1e-3
    batchnorm_momentum: float = 1e-2
    batchnorm_affine: bool = True
    batchnorm_track_running_stats: bool = True
    use_layernorm: bool = False
    layernorm_eps: float = 1e-5
    layernorm_elementwise_affine: bool = False
    layernorm_bias: bool = False
    dropout_p: float = 0.0
    dropout_inplace: bool = False
    activation_cls_kwargs: dict[str, Any] | None = None
    final_activation_cls_kwargs: dict[str, Any] | None = None
    bias: bool = True

    @field_validator("activation_cls", "final_activation_cls", mode="before")
    @classmethod
    def _coerce_activation_to_string(
        cls,
        value: str | type[torch.nn.Module] | None,
    ) -> str | None:
        """Accept either a class or a string; store as string for JSON-serializability."""
        return _normalize_activation_to_string(value)

    def mlp_kwargs(self) -> dict[str, Any]:
        """The keyword arguments of :class:`MLP`, with activation names resolved to classes."""
        kwargs = dict(self)
        for key in _ACTIVATION_KEYS:
            if isinstance(kwargs[key], str):
                kwargs[key] = _resolve_nn_module_class(kwargs[key])
        return kwargs

    def build(self, input_dim: int, output_dim: int) -> MLP:
        return MLP(input_dim, output_dim, **self.mlp_kwargs())


@component("velocity_field.mlp", builds=MLPVelocity)
class MLPVelocityConfig(ModuleConfig):
    """The JSON-able arguments of :class:`MLPVelocity`; see it for their meaning.

    ``state_dim`` and each condition encoder's ``input_dim`` come from the data. The callable
    arguments (``time_features_fn``, ``conditioning_fn``) have no spec, build the module by hand for those.
    """

    encode_state: bool = True
    encode_time: bool = True
    time_features_id: TimeFeaturesId | None = None
    num_time_features: int | None = None
    max_period: int | None = None
    state_encoder_output_dim: int | None = None
    time_encoder_output_dim: int | None = None
    state_encoder_mlp_kwargs: MLPConfig | None = None
    time_encoder_mlp_kwargs: MLPConfig | None = None
    vf_decoder_mlp_kwargs: MLPConfig | None = None
    conditioning_id: ConditioningLayersId | None = None
    conditioning_kwargs: dict[str, Any] | None = None
    condition_encoder_input_layers: dict[str, dict[str, Any]] | None = None
    """One layers dict per condition and group key, without ``input_dim``: the data sets it."""
    condition_encoder_output_dim: int | None = None
    condition_encoder_pooling_mode: Literal["mean", "sum", "attention-token", "attention-seed"] = "mean"
    condition_encoder_pooling_kwargs: dict[str, Any] | None = None
    condition_encoder_pooling_proj_dim: int | None = None
    condition_encoder_pooling_proj_bias: bool = True
    condition_encoder_covariates_not_pooled: tuple[str, ...] | None = None
    condition_encoder_output_layers_kwargs: MLPConfig | None = None
    source_encoder_mlp_kwargs: MLPConfig | None = None
    source_encoder_output_dim: int | None = None

    @field_validator("condition_encoder_input_layers")
    @classmethod
    def _no_input_dim(cls, layers: dict[str, dict[str, Any]] | None) -> dict[str, dict[str, Any]] | None:
        set_by_hand = [key for key, layer in (layers or {}).items() if "input_dim" in layer]
        if set_by_hand:
            raise ValueError(f"`input_dim` comes from the data; drop it from {set_by_hand}.")
        return layers

    def build(self, dims: DataDimensions) -> MLPVelocity:
        kwargs = {k: v.mlp_kwargs() if isinstance(v, MLPConfig) else v for k, v in dict(self).items()}

        # Fill in per-condition-encoder `input_dim` from the data dimensions.
        if self.condition_encoder_input_layers is not None:
            input_dims = {
                **(dims.condition_reps_dims or {}),
                **(dims.condition_continuous_dims or {}),
                **(dims.groups_reps_dims or {}),
            }
            kwargs["condition_encoder_input_layers"] = {
                key: {**layers, "input_dim": input_dims[key]}
                for key, layers in self.condition_encoder_input_layers.items()
            }
        return MLPVelocity(dims.state_dim, **kwargs)
