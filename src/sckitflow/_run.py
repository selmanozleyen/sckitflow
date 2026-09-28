"""Saving and loading a run as config plus weights.

A run is three things -- the data schema, the methods, and the learned
parameters -- and each is stored as what it is: the first two as portable
`scfit.registry` specs in ``specs.json``, the third as a plain
``state_dict`` in ``weights.pt``. Nothing is pickled, so a saved run survives
our own classes being renamed, and the specs are readable without loading
torch at all.

The run's seeds live here and nowhere else: :func:`run_rngs` derives every rng
from them, and the configs hold none. The neural module is supplied on load:

.. code-block:: python

    save_run("run", seed=0, data_config=data_cfg, method_configs={"training": tcfg}, module=module)

    dmod, plan = load_run("run", adata, module=MLPVelocity(...))
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch
from scfit.registry import Component

from sckitflow.core.methods._config import InferenceMethodConfig, TrainingMethodConfig
from sckitflow.data._config import FlowDataConfig
from sckitflow.data.splitters._config import SplitterConfig
from sckitflow.trainer._plan import TrainingPlan

if TYPE_CHECKING:
    from anndata import AnnData

    from sckitflow.data._datamodule import FlowDataModule

__all__ = ["RunRngs", "run_rngs", "save_run", "load_run"]

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


def save_run(
    path: str | Path,
    *,
    seed: int,
    split_seed: int = 0,
    splitter_config: SplitterConfig | None = None,
    data_config: FlowDataConfig,
    method_configs: Mapping[str, Component],
    module: torch.nn.Module,
    allow_overwrite: bool = False,
) -> None:
    """Writes a run to ``path`` as a directory of specs plus weights.

    :param path: Directory to write into; created if absent.
    :param seed: Seeds everything but the split, see :func:`run_rngs`.
    :param split_seed: Seeds the split.
    :param splitter_config: The splitter, if the split is derived rather than read from ``split_by``.
    :param data_config: The schema and streaming options.
    :param method_configs: ``{name: Component}``, e.g.
        ``{"training": CFMConfig(), "inference": ODEConfig(n_steps=50)}``.
        ``"training"`` and ``"inference"`` are the names :func:`load_run` reads.
    :param module: The trained module; its ``state_dict`` is what gets stored.
    :param allow_overwrite: Whether to replace existing files.
    :raises FileExistsError: If files exist and `allow_overwrite` is `False`.
    :raises scfit.registry.PortabilityError: If a config holds a live object,
        raised before anything is written.
    """
    out = Path(path)
    specs_path, weights_path = out / SPECS_NAME, out / WEIGHTS_NAME
    if not allow_overwrite:
        for existing in (specs_path, weights_path):
            if existing.exists():
                raise FileExistsError(f"{existing} already exists. Use allow_overwrite=True.")

    # Build the whole document first: a config holding a live object must fail
    # before any file is touched.
    document = {
        "format_version": 1,
        "seeds": {"seed": seed, "split_seed": split_seed},
        "splitter": splitter_config.to_spec() if splitter_config is not None else None,
        "data": data_config.to_spec(),
        "methods": {name: cfg.to_spec() for name, cfg in method_configs.items()},
    }

    out.mkdir(parents=True, exist_ok=True)
    specs_path.write_text(json.dumps(document, indent=2))
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

    :param path: Directory written by :func:`save_run`.
    :param adata: The data to attach; the schema comes from the saved spec, so
        it is *not* re-derived from this.
    :param module: A freshly built module of the right shape, e.g.
        ``MLPVelocity(dmod.data_dims.state_dim)``. Its weights are overwritten
        from ``weights.pt``.
    :param optimizer: Optional; omit for a run you only mean to predict with.
    :param map_location: Forwarded to `torch.load`.
    :return: ``(datamodule, plan)``.
    """
    src = Path(path)
    document = json.loads((src / SPECS_NAME).read_text())

    rngs = run_rngs(**document["seeds"])
    splitter_spec = document["splitter"]
    splitter = SplitterConfig.from_spec(splitter_spec).build(rng=rngs.split) if splitter_spec else None
    datamodule = FlowDataConfig.from_spec(document["data"]).build(adata, rng=rngs.loader, splitter=splitter)
    module.load_state_dict(torch.load(src / WEIGHTS_NAME, map_location=map_location))

    methods = document["methods"]
    if "training" not in methods:
        raise KeyError(f"{src / SPECS_NAME} has no 'training' method spec; found {sorted(methods)}.")
    inference = methods.get("inference")
    plan = TrainingPlan(
        TrainingMethodConfig.from_spec(methods["training"]).build(module),
        optimizer,
        inference_method=InferenceMethodConfig.from_spec(inference).build(module) if inference else None,
        val_names=datamodule.val_names,
    )
    return datamodule, plan
