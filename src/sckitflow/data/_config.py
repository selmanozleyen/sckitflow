"""Portable config for the data side of a run.

Built on :mod:`scfit.registry`, the same foundation the group encoders already
use (:class:`~sckitflow.data._group_encoders.GroupEncoder`). Every field here is
JSON-serializable, so the schema travels as a spec rather than as a pickled
:class:`DataManager` -- which is what lets a saved run survive our own classes
being renamed, and keeps a checkpoint small.

``build(adata)`` returns the ready :class:`~sckitflow.data.FlowDataModule`; the
``AnnData`` is the context, never part of the spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from anndata import AnnData
from scfit.registry import Component, register_live

from sckitflow.data._datamodule import FlowDataModule
from sckitflow.data._group_encoders import GroupEncoder
from sckitflow.data.splitters._base import Splitter
from sckitflow.data.splitters._config import SplitterConfig

__all__ = ["FlowDataConfig"]

# A live `Splitter` instance still builds and trains; it just has no spec, so a
# config holding one refuses to serialize. Prefer `SplitterConfig`, which does.
register_live(Splitter)


@dataclass
class FlowDataConfig(Component, type_id="data_module.flow", version=1):
    """The schema and streaming options a run reads its batches with.

    Mirrors :class:`~sckitflow.data.DataManagerKwargs` plus the loader knobs
    :class:`~sckitflow.data.FlowDataModule` takes. Two fields need care:

    * ``matched_keys`` is a ``{source key: target key}`` mapping over tuples,
      which JSON cannot key on -- give it as a list of ``[source, target]``
      pairs and it is converted on the way in.
    * ``splitter`` is a live object; prefer ``split_by`` (a column name), which
      is portable.
    """

    # --- what each observation is ---
    sample_rep: str | None = None
    conditions: dict[str, tuple[str, ...]] | None = None
    conditions_reps: dict[str, str] | None = None
    conditions_covariates: tuple[str, ...] | None = None
    condition_state_key: str | None = None
    groups: tuple[str, ...] | None = None
    groups_reps: dict[str, str] | None = None
    groups_encoding: dict[str, dict[str, Any]] | None = None
    """Per-group encoder, each as a `GroupEncoder` spec -- ``{"g": OneHot().to_spec()}``.

    Specs rather than `GroupEncoder` instances because `scfit.registry` only tags
    a `Component` held in a *direct* field; one inside a container loses its
    ``type_id`` and cannot be parsed back. The bare ``"label"`` / ``"one-hot"``
    shorthands `DataManager` accepts are not offered here either -- cattrs
    cannot read a ``dict | Literal`` union -- so write ``Label().to_spec()``.
    """
    target_categorical_covs_dict: dict[str, Literal["label", "one-hot", "functional"]] | None = None
    target_continuous_covs: tuple[str, ...] | None = None

    # --- which observations flow into which ---
    control_values_dict: dict[str, str] | None = None
    matched_pairs: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] | None = None
    """``[(source key, target key), ...]`` -- the portable form of ``matched_keys``."""

    # --- which observations are held out ---
    split_by: str | None = None
    splitter: SplitterConfig | Splitter | None = None
    """A `SplitterConfig` (portable) or a live `Splitter` (builds, will not serialize)."""

    # --- incomparable spaces ---
    n_shared_dims: int | None = None
    source_rep: str | None = None

    # --- streaming ---
    train_split: str = "train"
    n_train_steps: int = 100_000
    batch_size: int = 128
    dtype: str = "float32"
    """Name of a ``torch`` dtype, e.g. ``"float32"``. A `torch.dtype` is not JSON."""
    loader_kwargs: dict[str, Any] = field(default_factory=dict)

    def build(self, context: AnnData) -> FlowDataModule:
        """Fits the schema on ``context`` and returns the data module.

        :param context: The `AnnData` to derive dimensionalities from and stream.
        """
        schema = {
            "sample_rep": self.sample_rep,
            "conditions": self.conditions,
            "conditions_reps": self.conditions_reps,
            "conditions_covariates": self.conditions_covariates,
            "condition_state_key": self.condition_state_key,
            "groups": self.groups,
            "groups_reps": self.groups_reps,
            "groups_encoding": (
                {key: GroupEncoder.from_spec(spec) for key, spec in self.groups_encoding.items()}
                if self.groups_encoding
                else None
            ),
            "target_categorical_covs_dict": self.target_categorical_covs_dict,
            "target_continuous_covs": self.target_continuous_covs,
            "control_values_dict": self.control_values_dict,
            "matched_keys": dict(self.matched_pairs) if self.matched_pairs else None,
            "split_by": self.split_by,
            "splitter": (self.splitter.build() if isinstance(self.splitter, SplitterConfig) else self.splitter),
            "n_shared_dims": self.n_shared_dims,
            "source_rep": self.source_rep,
        }
        return FlowDataModule.from_adata(
            context,
            train_split=self.train_split,
            n_train_steps=self.n_train_steps,
            batch_size=self.batch_size,
            dtype=getattr(torch, self.dtype),
            loader_kwargs=self.loader_kwargs or None,
            **{k: v for k, v in schema.items() if v is not None},
        )
