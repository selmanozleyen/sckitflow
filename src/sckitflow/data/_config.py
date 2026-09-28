"""Portable config for the data side of a run.

Every field is JSON-serializable, so the schema travels as a spec rather than a
pickled :class:`DataManager`. ``build(adata, rng=...)`` returns the ready
:class:`~sckitflow.data.FlowDataModule`; the ``AnnData`` is never in the spec.
"""

from __future__ import annotations

from dataclasses import field
from typing import Any, Literal

import numpy as np
import torch
from anndata import AnnData
from scfit.registry import Component, component

from sckitflow.data._datamodule import FlowDataModule
from sckitflow.data._group_encoders import GroupEncoderConfig
from sckitflow.data.splitters._base import Splitter

__all__ = ["FlowDataModuleConfig"]


@component("data_module.flow", builds=FlowDataModule)
class FlowDataModuleConfig(Component):
    """The schema and streaming options a run reads its batches with.

    Mirrors :class:`~sckitflow.data.DataManagerKwargs` plus the loader knobs
    :class:`~sckitflow.data.FlowDataModule` takes. ``matched_keys`` is a
    mapping over tuples, which JSON cannot key on, so it is given here as
    ``matched_pairs``, a list of ``(source, target)`` pairs.

    The splitter is not a field: it has its own seed, so the run builds it
    beside this config and passes it to :meth:`build`.
    """

    # --- what each observation is ---
    sample_rep: str | None = None
    conditions: dict[str, tuple[str, ...]] | None = None
    conditions_reps: dict[str, str] | None = None
    conditions_covariates: tuple[str, ...] | None = None
    condition_state_key: str | None = None
    groups: tuple[str, ...] | None = None
    groups_reps: dict[str, str] | None = None
    groups_encoding: dict[str, GroupEncoderConfig] | None = None
    """Per-group encoder, e.g. ``{"g": OneHotEncoderConfig()}``. Write ``LabelEncoderConfig()`` rather than the ``"label"`` shorthand."""
    target_categorical_covs_dict: dict[str, Literal["label", "one-hot", "functional"]] | None = None
    target_continuous_covs: tuple[str, ...] | None = None

    # --- which observations flow into which ---
    control_values_dict: dict[str, str] | None = None
    matched_pairs: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] | None = None
    """``[(source key, target key), ...]``, the portable form of ``matched_keys``."""

    # --- which observations are held out ---
    split_by: str | None = None

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
    """Forwarded to every loader. No ``seed``: the schedule is drawn from ``rng``."""

    def build(self, adata: AnnData, *, rng: np.random.Generator, splitter: Splitter | None = None) -> FlowDataModule:
        """Fits the schema on ``adata`` and returns the data module.

        :param adata: The `AnnData` to derive dimensionalities from and stream.
        :param rng: The loaders' sampling schedule is seeded from it.
        :param splitter: Applied to ``adata`` before streaming; exclusive with ``split_by``.
        """
        if "seed" in self.loader_kwargs:
            raise ValueError("loader_kwargs must not set `seed`; the schedule is drawn from `rng`.")
        schema = {
            "sample_rep": self.sample_rep,
            "conditions": self.conditions,
            "conditions_reps": self.conditions_reps,
            "conditions_covariates": self.conditions_covariates,
            "condition_state_key": self.condition_state_key,
            "groups": self.groups,
            "groups_reps": self.groups_reps,
            "groups_encoding": self.groups_encoding,
            "target_categorical_covs_dict": self.target_categorical_covs_dict,
            "target_continuous_covs": self.target_continuous_covs,
            "control_values_dict": self.control_values_dict,
            "matched_keys": dict(self.matched_pairs) if self.matched_pairs else None,
            "split_by": self.split_by,
            "splitter": splitter,
            "n_shared_dims": self.n_shared_dims,
            "source_rep": self.source_rep,
        }
        return FlowDataModule.from_adata(
            adata,
            train_split=self.train_split,
            n_train_steps=self.n_train_steps,
            batch_size=self.batch_size,
            dtype=getattr(torch, self.dtype),
            loader_kwargs={**self.loader_kwargs, "seed": int(rng.integers(2**63))},
            **{k: v for k, v in schema.items() if v is not None},
        )
