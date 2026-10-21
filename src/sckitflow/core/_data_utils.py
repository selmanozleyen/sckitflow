from typing import Any

import numpy as np
import torch

from sckitflow.core._types import SamplerFn, StepData
from sckitflow.core._utils import to_torch_tensor
from sckitflow.data import mixins
from sckitflow.data._dims import DataDimensions
from sckitflow.data.containers import CategoricalData

__all__ = [
    "batchmixin_to_torch",
    "get_tensor_dict_from_data",
    "subscript_step_data",
    "expand_conditioning",
    "prepare_latent_train",
    "prepare_latent_inference",
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
    data: CategoricalData | dict[str, np.ndarray | torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    """Extracts torch tensors from the distribution data."""
    if data is None:
        return {}
    data_dict = data.extract_reps() if isinstance(data, CategoricalData) else data
    return batchmixin_to_torch(data_dict)


def _index_obj(value: Any, idx: Any | None) -> Any:
    """Row-index one step-data field by ``idx``.

    Handles the three field shapes that appear in a batch: ``None`` (passthrough),
    a plain ``dict[str, tensor]`` (each value is indexed; ``None`` values pass
    through), and any row-indexable object -- a ``torch.Tensor`` / array or a
    container that implements ``__getitem__``. A ``None`` index is a no-op.
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

    Applies a coupling/matching permutation (or subselection) to a batch. Each side
    is sliced independently; a ``None`` index leaves that side untouched and a
    ``None`` field stays ``None``. The target side includes ``target_response_data``,
    so target covariates carried for output reconstruction remain aligned with
    ``target_state``. ``target_group_data`` describes the whole batch, one row, so it is
    not sliced; ``n_obs`` becomes the target's new row count.
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
        "target_response_data",
    )
    batch_fields = ("target_group_data", "n_obs")

    # A field on neither side would keep its row order while the rest of the batch
    # is permuted -- wrong rows, no error. Checked only when something is actually
    # permuted: with both `None`, nothing moves, so nothing can fall out of alignment.
    if src_idxs is not None or tgt_idxs is not None:
        unassigned = set(step_data) - {*source_fields, *target_fields, *batch_fields}
        if unassigned:
            raise ValueError(
                f"cannot row-slice a batch carrying unaligned fields: {sorted(unassigned)}. Every "
                "per-observation field must belong to a side, or a matching permutation leaves it "
                "misaligned with the state it describes."
            )

    updates = {field: _index_obj(step_data.get(field), tgt_idxs) for field in target_fields}
    updates.update({field: _index_obj(step_data.get(field), src_idxs) for field in source_fields})
    if tgt_idxs is not None:
        updates["n_obs"] = len(tgt_idxs)
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
    n_obs: int | None = None,
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

    ``B`` is the batch's row count: ``source``'s rows when present, else ``n_obs``. An unconditional batch
    without a source has no rows to follow and draws standalone samples. ``D`` is ``dims.state_dim``. Device and dtype come
    from the caller; the batch is never asked for them.
    """
    # Short-circuit: paired and not generating -- the source is the latent.
    if source is not None and not generate_from_noise:
        return source

    if source is not None:
        batch_size = source.shape[0]
    elif len(condition_dict) == 0:
        batch_size = None
    elif n_obs is None:
        raise ValueError(
            "Cannot size the latent: the batch has conditions but no source state, and its group encodings are "
            "one row each, so the number of observations to generate has to come from `step_data['n_obs']`, "
            "which is None. Batches from sckitflow's loaders set it; set it when building a `StepData` by hand."
        )
    else:
        batch_size = n_obs

    # Standalone samples: no batch axis, draw `n_samples` (or one).
    if batch_size is None:
        n = 1 if n_samples is None else n_samples
        return noise_sampler((n, dims.state_dim), generator=generator, device=device, dtype=dtype)

    # Batch present: one draw per observation, or `n_samples` per observation.
    shape = (batch_size, dims.state_dim) if n_samples is None else (n_samples, batch_size, dims.state_dim)
    return noise_sampler(shape, generator=generator, device=device, dtype=dtype)
