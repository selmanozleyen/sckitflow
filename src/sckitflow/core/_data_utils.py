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

    Handles the three field shapes that appear in a batch: ``None`` (passthrough),
    a plain ``dict[str, tensor]`` (each value is indexed; ``None`` values pass
    through), and any row-indexable object -- a ``torch.Tensor`` / array or a
    container that implements ``__getitem__``. A ``None`` index is a no-op.
    A tensor with a leading axis of 1 (a group's encoding, the same for every row)
    is left as it is: there is nothing to reorder.
    """
    if value is None or idx is None:
        return value
    if isinstance(value, dict):
        return {k: _index_obj(v, idx) for k, v in value.items()}
    if isinstance(value, torch.Tensor) and value.shape[0] == 1:
        return value
    return value[idx]


def subscript_step_data(
    step_data: StepData,
    src_idxs: Any | None = None,
    tgt_idxs: Any | None = None,
) -> StepData:
    """Row-slice a batch: the source side by ``src_idxs``, the target side by ``tgt_idxs``.

    Applies a coupling/matching permutation (or subselection) to a batch. Each side
    is sliced independently; a ``None`` index leaves that side untouched and a
    ``None`` field stays ``None``. The target side includes ``target_response_data``,
    so target covariates carried for output reconstruction remain aligned with
    ``target_state``. Replaces ``BaseMethod._safe_subscript_obj``.
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

    # A field on neither side would keep its row order while the rest of the batch
    # is permuted -- wrong rows, no error. Checked only when something is actually
    # permuted: with both `None`, nothing moves, so nothing can fall out of alignment.
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
    """Replicate ``condition_dict`` and ``source`` along the leading dimension of ``latent``.

    ``condition_dict`` values have shape ``(1 or batch_size, ...)``, ``source`` ``(batch_size, ...)`` and
    ``latent`` ``(n_samples, batch_size, dim)`` or ``(batch_size, dim)``. Returns expanded views.
    """
    if latent.dim() <= 2:  # no extra sample dimension
        return condition_dict, source

    n_samples = latent.shape[0]
    expanded_cond = {k: v.unsqueeze(0).expand(n_samples, *v.shape) for k, v in condition_dict.items()}
    expanded_source = None if source is None else source.unsqueeze(0).expand(n_samples, *source.shape)
    return expanded_cond, expanded_source


def _batch_size_from(
    source: torch.Tensor | None,
    condition_dict: dict[str, torch.Tensor],
    target: torch.Tensor | None = None,
) -> int | None:
    """How many rows a batch has, or ``None`` when nothing in it says.

    A source has one row per observation and sets the count. Without one, an unconditional batch has no
    rows to follow (the standalone-sample case, inference only), and a conditional batch has one row per
    target observation: the longest leading axis among ``target`` and the conditioning tensors, where a
    group's categorical encoding is a single row broadcast to all of them.
    """
    if source is not None:
        return source.shape[0]
    if len(condition_dict) == 0:
        return None
    return max(t.shape[0] for t in (target, *condition_dict.values()) if t is not None)


def prepare_latent_train(
    source: torch.Tensor | None,
    noise_sampler: SamplerFn,
    dims: DataDimensions,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    generate_from_noise: bool = False,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draws the training latent, ``(batch_size, dims.state_dim)``.

    ``source`` present with ``generate_from_noise=False`` short-circuits: it *is* the latent, returned
    unchanged. Device and dtype come from the caller (pass ``method.device`` / ``method.dtype``).
    """
    if source is not None and not generate_from_noise:
        return source
    return noise_sampler((batch_size, dims.state_dim), generator=generator, device=device, dtype=dtype)


def prepare_latent_inference(
    source: torch.Tensor | None,
    condition_dict: dict[str, torch.Tensor],
    noise_sampler: SamplerFn,
    dims: DataDimensions,
    *,
    device: torch.device,
    dtype: torch.dtype,
    n_samples: int | None = None,
    generate_from_noise: bool = False,
    generator: torch.Generator,
    target: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sizes and draws the inference latent.

    Three axes decide the shape -- ``source`` present, conditioning present,
    ``n_samples`` present -- plus ``generate_from_noise``, which only matters
    when a source is present:

    ========  ============  ===========  ======  =========================
    source    conditioning  n_samples    noise   result
    ========  ============  ===========  ======  =========================
    yes       any           any          off     ``source`` itself
    yes       any           yes          on      ``(n_samples, B, D)``
    yes       any           no           on      ``(B, D)``
    no        yes           yes          any     ``(n_samples, B, D)``
    no        yes           no           any     ``(B, D)``
    no        no            yes          any     ``(n_samples, D)``
    no        no            no           any     ``(1, D)``
    ========  ============  ===========  ======  =========================

    ``B`` is the batch's row count, the longest leading axis among ``source``, ``target`` and the
    conditioning tensors (see :func:`_batch_size_from`). ``D`` is ``dims.state_dim``. Device and dtype come
    from the caller; the batch is never asked for them.
    """
    # Short-circuit: paired and not generating -- the source is the latent.
    if source is not None and not generate_from_noise:
        return source

    batch_size = _batch_size_from(source, condition_dict, target)

    # Standalone samples: no batch axis, draw `n_samples` (or one).
    if batch_size is None:
        n = 1 if n_samples is None else n_samples
        return noise_sampler((n, dims.state_dim), generator=generator, device=device, dtype=dtype)

    # Batch present: one draw per observation, or `n_samples` per observation.
    shape = (batch_size, dims.state_dim) if n_samples is None else (n_samples, batch_size, dims.state_dim)
    return noise_sampler(shape, generator=generator, device=device, dtype=dtype)


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
