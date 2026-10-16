from typing import Any

import numpy as np
import torch

from sckitflow.core._types import SamplerFn, StepData, TensorMixin
from sckitflow.core._utils import to_torch_tensor
from sckitflow.data import mixins
from sckitflow.data._dims import DataDimensions
from sckitflow.data.containers import CategoricalData, MixedTypeData

__all__ = [
    "batchmixin_to_torch",
    "get_tensor_dict_from_data",
    "subscript_step_data",
    "expand_conditioning",
    "prepare_latent_train",
    "prepare_latent_inference",
    "TorchMixedTypeData",
]


def batchmixin_to_torch(
    data: mixins.BatchMixin | dict[str, np.ndarray | torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    """Converts the elements of a mixin to torch tensors."""
    if data is None:
        return {}
    data_dict = data.mapping if isinstance(data, mixins.BatchMixin) else data
    if not isinstance(data_dict, dict):
        raise ValueError(f"Data dictionary of the wrong type, expected `dict` got {type(data_dict)}.")
    return {k: to_torch_tensor(v) for k, v in data_dict.items()}


def get_tensor_dict_from_data(
    data: MixedTypeData | CategoricalData | dict[str, np.ndarray | torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    """Extracts torch tensors from the distribution data."""
    if data is None:
        return {}
    data_dict = data.extract_reps() if isinstance(data, MixedTypeData | CategoricalData) else data
    return batchmixin_to_torch(data_dict)


def _index_obj(value: Any, idx: Any | None) -> Any:
    """Row-index one step-data field by ``idx``.

    Handles the three field shapes that appear in a batch: ``None`` (passthrough), a
    plain ``dict[str, tensor]`` (each value is indexed; ``None`` values pass through), and any
    row-indexable object -- a ``torch.Tensor`` / array or a container that implements
    ``__getitem__``. A ``None`` index is a no-op.
    """
    if value is None or idx is None:
        return value
    if isinstance(value, dict):
        return {k: (None if v is None else v[idx]) for k, v in value.items()}
    return value[idx]


def subscript_step_data(
    step_data: StepData,
    src_idxs: Any | None = None,
    tgt_idxs: Any | None = None,
) -> StepData:
    """Row-slice a batch: the source side by ``src_idxs``, the target side by ``tgt_idxs``.

    Applies a coupling/matching permutation (or subselection) to a batch. Each side is sliced
    independently; a ``None`` index leaves that side untouched and a ``None`` field stays ``None``.
    The target side includes ``target_response_data``, so target covariates carried for output
    reconstruction remain aligned with ``target_state``. Replaces ``BaseMethod._safe_subscript_obj``.
    """
    source_fields = (
        "source_state",
        "source_coupling_lin",
        "source_coupling_quad",
        "source_condition_data",
        "source_group_data",
    )
    target_fields = (
        "target_state",
        "target_coupling_lin",
        "target_coupling_quad",
        "target_condition_data",
        "target_group_data",
        "target_response_data",
    )

    if src_idxs is not None or tgt_idxs is not None:
        unassigned = set(step_data) - {*source_fields, *target_fields}
        if unassigned:
            raise ValueError(
                f"cannot row-slice a batch carrying unaligned fields: {sorted(unassigned)}. Every "
                "per-observation field must belong to a side, or a matching permutation leaves it "
                "misaligned with the state it describes."
            )

    updates = {field: _index_obj(step_data.get(field), tgt_idxs) for field in target_fields}
    updates.update({field: _index_obj(step_data.get(field), src_idxs) for field in source_fields})
    return {**step_data, **updates}


def expand_conditioning(
    latent: torch.Tensor,
    condition_dict: dict[str, torch.Tensor],
    source: torch.Tensor | None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
    """
    Replicate condition_dict and source along the leading dimensions of latent

    Assumes original condition_dict values and source have shape (batch_size, ...)
    and latent has shape (n_samples, batch_size, dim) or (batch_size, dim).
    Returns expanded copies.
    """
    if latent.dim() <= 2:  # no extra sample dimension
        return condition_dict, source

    n_samples = latent.shape[0]
    expanded_cond = {}
    for k, v in condition_dict.items():
        # v shape: (batch_size, ...)
        expanded_cond[k] = v.unsqueeze(0).expand(n_samples, *v.shape)
    expanded_source = None
    if source is not None:
        expanded_source = source.unsqueeze(0).expand(n_samples, *source.shape)
    return expanded_cond, expanded_source


def prepare_latent_train(
    source: torch.Tensor | None,
    target: torch.Tensor,
    noise_sampler: SamplerFn,
    dims: DataDimensions,
    generate_from_noise: bool = False,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Called from ``compute_loss``. Always returns a single noise per batch element.

    The latent shape is ``(B, dims.state_dim)``, where ``B = target.shape[0]``. ``target``
    provides device and dtype, so the latent lands where the batch already is rather than
    on whatever the sampler defaults to. ``dims.state_dim`` is the authoritative width,
    so a module built for a state width different from ``target.shape[-1]`` (e.g. under
    the condition-space view) still gets a correctly shaped latent.
    """
    if source is not None and not generate_from_noise:
        return source

    shape = (target.shape[0], dims.state_dim)
    return noise_sampler(shape, generator=generator, device=target.device, dtype=target.dtype)


def prepare_latent_inference(
    source: torch.Tensor | None,
    condition_dict: dict[str, torch.Tensor],
    noise_sampler: SamplerFn,
    dims: DataDimensions,
    n_samples: int | None = None,
    generate_from_noise: bool = False,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Called from ``predict``. The target state is not required.

    Three resolutions, in order:

    1. **Paired, not generating** -- ``source`` is present and ``generate_from_noise`` is
       ``False``: it *is* the latent, returned unchanged.
    2. **Paired, generating** -- ``source`` fixes the batch size, and device/dtype.
    3. **Unpaired** -- no ``source``: batch size and device/dtype come from the first tensor
       in ``condition_dict``.

    In every generating case the state dimension comes from ``dims.state_dim``, so the
    latent matches what the velocity field was built for even when there is no target
    template in the batch.

    :param source: Source state from the batch, or ``None`` when predicting unpaired.
    :param condition_dict: The batch's conditioning tensors, used for batch size / device /
        dtype when ``source`` is absent.
    :param noise_sampler: Draws noise of a given shape on a given device / dtype.
    :param dims: The method's shape spec. Its ``state_dim`` is the state width.
    :param n_samples: Samples per batch element; ``None`` yields a single ``(B, D)`` draw.
    :param generate_from_noise: Ignore ``source`` and draw noise even when it is present.
    :param generator: Passed through to ``noise_sampler``.
    :raises ValueError: If the batch carries neither ``source`` nor a conditioning tensor.
    """
    if source is not None and not generate_from_noise:
        return source

    if source is not None:
        reference = source
        batch_size = source.shape[0]
    elif condition_dict:
        reference = next(iter(condition_dict.values()))
        batch_size = reference.shape[0]
    else:
        raise ValueError(
            "cannot size the latent: the batch carries neither `source` nor a conditioning tensor. "
            "At least one is needed to know the batch size and device."
        )

    shape = (batch_size, dims.state_dim) if n_samples is None else (n_samples, batch_size, dims.state_dim)
    return noise_sampler(shape, generator=generator, device=reference.device, dtype=reference.dtype)


class TorchMixedTypeData(MixedTypeData):
    """"""  # noqa

    def extract_reps(self) -> TensorMixin:
        """Extracts the representations for the underlying data."""
        if self.categorical_covariates is not None:
            cat_reps = self.categorical_covariates.extract_reps()
        else:
            cat_reps = TensorMixin({})

        if self.continuous_covariates is not None:
            return TensorMixin(
                {
                    **self.continuous_covariates.mapping,
                    **cat_reps.mapping,
                }
            )
        return cat_reps

    @classmethod
    def from_mixed_type_data(
        cls, key: str, x_cond: torch.Tensor, base_container: MixedTypeData | None = None
    ) -> "TorchMixedTypeData":
        """Constructor from input data and optional base container.

        Instaties the torch mixed data from a base container, while overriding the continuous covariates at the given key.

        :param key: The contiuous covariates key to override.
        :type key: class: `str`

        :param x_cond: The data for the continuous covariates to override with.
        :type x_cond: class: `torch.Tensor`

        :param base_container: Optional base container from which to retain the other condition covariates.
            Defaults to `None`, in which case the container will be created from scratch.
        :type base_container: class: `MixedTypeData | None`
        """
        if base_container is not None:
            categorical_covariates = base_container.categorical_covariates
            continuous_covariates = base_container.continuous_covariates
        else:
            categorical_covariates = None
            continuous_covariates = None

        if continuous_covariates is not None:
            mapping = continuous_covariates.mapping
            mapping[key] = x_cond
        else:
            mapping = {key: x_cond}

        mapping = {k: to_torch_tensor(v) for k, v in mapping.items()}
        mixin = TensorMixin(mapping)

        return cls(categorical_covariates=categorical_covariates, continuous_covariates=mixin)
