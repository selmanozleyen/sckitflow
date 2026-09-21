"""Saving and loading a run as config plus weights.

A run is three things -- the data schema, the methods, and the learned
parameters -- and each is stored as what it is: the first two as portable
`scfit.registry` specs in ``specs.json``, the third as a plain
``state_dict`` in ``weights.pt``. Nothing is pickled, so a saved run survives
our own classes being renamed, and the specs are readable without loading
torch at all.

The neural module is supplied by the caller on load, the same way every
`Component.build` takes its runtime dependency as ``context``:

.. code-block:: python

    save_run("run", data_config=data_cfg, method_configs={"training": tcfg}, module=module)

    dmod, plan = load_run("run", adata, module=MLPVelocity(...))
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from scfit.registry import Component, parse

from sckitflow.data._config import FlowDataConfig
from sckitflow.trainer._plan import TrainingPlan

if TYPE_CHECKING:
    from anndata import AnnData

    from sckitflow.data._datamodule import FlowDataModule

__all__ = ["save_run", "load_run"]

SPECS_NAME = "specs.json"
WEIGHTS_NAME = "weights.pt"


def save_run(
    path: str | Path,
    *,
    data_config: FlowDataConfig,
    method_configs: Mapping[str, Component],
    module: torch.nn.Module,
    allow_overwrite: bool = False,
) -> None:
    """Writes a run to ``path`` as a directory of specs plus weights.

    :param path: Directory to write into; created if absent.
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

    datamodule = parse(document["data"]).build(adata)
    module.load_state_dict(torch.load(src / WEIGHTS_NAME, map_location=map_location))

    methods: dict[str, Any] = {name: parse(spec).build(module) for name, spec in document["methods"].items()}
    try:
        training_method = methods["training"]
    except KeyError as e:
        raise KeyError(f"{src / SPECS_NAME} has no 'training' method spec; found {sorted(methods)}.") from e

    plan = TrainingPlan(
        training_method,
        optimizer,
        inference_method=methods.get("inference"),
        val_names=datamodule.val_names,
    )
    return datamodule, plan
