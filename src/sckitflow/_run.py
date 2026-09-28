"""Saving and loading a run as its spec plus weights.

A run is a :class:`RunSpec`, the data schema, the methods and the two seeds, plus the learned
parameters. ``specs.json`` holds ``RunSpec.to_spec()``, ``weights.pt`` a plain ``state_dict``.
Nothing is pickled, so a saved run survives our own classes being renamed.

The run's seeds live on the spec and nowhere else: :func:`run_rngs` derives every rng from them.
The neural module is supplied on load:

.. code-block:: python

    spec = RunSpec(seed=0, data=FlowDataConfig(...), training=CFMConfig())
    save_run("run", spec, module)

    dmod, plan = load_run("run", adata, module=MLPVelocity(...))
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch
from scfit.registry import Component, component

from sckitflow.core.methods._base import InferenceMethodConfig, TrainingMethodConfig
from sckitflow.data._config import FlowDataConfig
from sckitflow.data.splitters._base import SplitterConfig
from sckitflow.trainer._plan import TrainingPlan

if TYPE_CHECKING:
    from anndata import AnnData

    from sckitflow.data._datamodule import FlowDataModule

__all__ = ["RunRngs", "RunSpec", "run_rngs", "save_run", "load_run"]

SPECS_NAME = "specs.json"
WEIGHTS_NAME = "weights.pt"


class RunRngs(NamedTuple):
    """Every rng of a run, derived from its two seeds."""

    split: np.random.Generator
    loader: np.random.Generator


def run_rngs(*, seed: int, split_seed: int) -> RunRngs:
    """The run's rngs. The split has its own seed, so retraining with another ``seed`` keeps the split."""
    # Append new streams at the end: `spawn` is append-only, so existing ones keep their numbers.
    (loader,) = np.random.default_rng(seed).spawn(1)
    return RunRngs(split=np.random.default_rng(split_seed), loader=loader)


@component("run")
class RunSpec(Component):
    """Everything portable about a run: what ``specs.json`` holds."""

    data: FlowDataConfig
    training: TrainingMethodConfig
    inference: InferenceMethodConfig | None = None
    splitter: SplitterConfig | None = None
    """Derives the split. ``None`` reads it from ``data.split_by`` instead."""
    seed: int = 0
    """Seeds everything but the split, see :func:`run_rngs`."""
    split_seed: int = 0

    def build(
        self, adata: AnnData, module: torch.nn.Module, *, optimizer: torch.optim.Optimizer | None = None
    ) -> tuple[FlowDataModule, TrainingPlan]:
        """The data module over ``adata`` and a plan over ``module``."""
        rngs = run_rngs(seed=self.seed, split_seed=self.split_seed)
        splitter = self.splitter.build(rng=rngs.split) if self.splitter is not None else None
        datamodule = self.data.build(adata, rng=rngs.loader, splitter=splitter)
        plan = TrainingPlan(
            self.training.build(module),
            optimizer,
            inference_method=self.inference.build(module) if self.inference is not None else None,
            val_names=datamodule.val_names,
        )
        return datamodule, plan


def save_run(path: str | Path, spec: RunSpec, module: torch.nn.Module, *, allow_overwrite: bool = False) -> None:
    """Writes ``spec`` and the weights of ``module`` to the directory ``path``.

    :raises FileExistsError: If files exist and `allow_overwrite` is `False`.
    :raises scfit.registry.PortabilityError: If the spec holds a live object, before anything is written.
    """
    out = Path(path)
    specs_path, weights_path = out / SPECS_NAME, out / WEIGHTS_NAME
    if not allow_overwrite:
        for existing in (specs_path, weights_path):
            if existing.exists():
                raise FileExistsError(f"{existing} already exists. Use allow_overwrite=True.")
    document = json.dumps(spec.to_spec(), indent=2)  # fails on a live object before any file is touched
    out.mkdir(parents=True, exist_ok=True)
    specs_path.write_text(document)
    torch.save(module.state_dict(), weights_path)


def load_run(
    path: str | Path,
    adata: AnnData,
    module: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | None = "cpu",
) -> tuple[FlowDataModule, TrainingPlan]:
    """Rebuilds a run from ``path``: the data module, and a plan over ``module``.

    :param adata: The data to attach; the schema comes from the saved spec, not from this.
    :param module: A freshly built module of the right shape; its weights are loaded from ``weights.pt``.
    :param optimizer: Optional; omit for a run you only mean to predict with.
    """
    src = Path(path)
    spec = RunSpec.from_spec(json.loads((src / SPECS_NAME).read_text()))
    module.load_state_dict(torch.load(src / WEIGHTS_NAME, map_location=map_location))
    return spec.build(adata, module, optimizer=optimizer)
