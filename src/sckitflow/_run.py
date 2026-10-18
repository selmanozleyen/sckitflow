"""Saving and loading a run as its spec plus weights.

A run is a :class:`RunConfig`, the data schema, the methods and the three seeds, plus the learned
parameters, saved in the Hugging Face Hub layout of :mod:`scfit.pretrained`: ``config.json`` holds
``RunConfig.to_spec()`` and ``module/model.safetensors`` the module's ``state_dict``. Nothing is pickled.
Share a saved run with ``huggingface_hub.HfApi().upload_folder(repo_id=..., folder_path=path)`` and load it
back by its repo id. A run reads back under the package versions that saved it.

The run's seeds live on the spec and nowhere else, one per consumer of randomness.
With a ``module`` config the spec is the whole model; without one the module is supplied on load:

.. code-block:: python

    spec = RunConfig(data=FlowDataModuleConfig(...), training=CFMTrainingConfig(), module=MLPVelocityConfig())
    datamodule, plan = spec.build(adata)
    save_run("run", spec, plan.module)

    dmod, plan = load_run("run", adata)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch
from scfit.pretrained import load_pretrained, save_pretrained
from scfit.registry import Component, component

from sckitflow.core.methods._base import InferenceMethodConfig, TrainingMethodConfig
from sckitflow.core.nn._config import ModuleConfig
from sckitflow.data._config import FlowDataModuleConfig
from sckitflow.data.splitters._base import SplitterConfig
from sckitflow.trainer._optim import OptimizerConfig
from sckitflow.trainer._plan import TrainingPlan

if TYPE_CHECKING:
    from anndata import AnnData

    from sckitflow.data._datamodule import FlowDataModule

__all__ = ["Run", "RunConfig", "save_run", "load_run"]


class Run(NamedTuple):
    """A built run: its data module and its plan."""

    datamodule: FlowDataModule
    plan: TrainingPlan


@component("run", builds=Run)
class RunConfig(Component):
    """Everything portable about a run: what ``config.json`` holds."""

    data: FlowDataModuleConfig
    training: TrainingMethodConfig
    inference: InferenceMethodConfig | None = None
    splitter: SplitterConfig | None = None
    """Derives the split. ``None`` reads it from ``data.split_by`` instead."""
    loader_seed: int = 0
    """Seeds the loaders' sampling schedule."""
    splitter_seed: int = 0
    """Seeds the split, so retraining with another ``loader_seed`` keeps it."""
    method_seed: int = 0
    """Seeds every time, noise and coupling draw of training, validation and prediction."""
    module: ModuleConfig | None = None
    """Builds the module from the data's dims. ``None`` means the module is passed to :meth:`build`."""
    optimizer: OptimizerConfig | None = None
    """Builds the optimizer over the module. ``None`` means it is passed to :meth:`build`, or omitted to only predict."""

    def build(
        self,
        adata: AnnData,
        module: torch.nn.Module | None = None,
        *,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> Run:
        """The data module over ``adata`` and a plan over ``module``, or over the module ``self.module`` builds.

        :param module: Overrides ``self.module``; required when it is ``None``.
        :param optimizer: Overrides ``self.optimizer``.
        """
        splitter = (
            self.splitter.build(rng=np.random.default_rng(self.splitter_seed)) if self.splitter is not None else None
        )
        datamodule = self.data.build(adata, rng=np.random.default_rng(self.loader_seed), splitter=splitter)
        dims = datamodule.data_dims
        if module is None:
            if self.module is None:
                raise ValueError("pass a `module`, or set `module` on the config.")
            module = self.module.build(dims)
        if optimizer is None and self.optimizer is not None:
            optimizer = self.optimizer.build(module.parameters())
        plan = TrainingPlan(
            self.training.build(module, dims=dims),
            optimizer,
            inference_method=self.inference.build(module, dims=dims) if self.inference is not None else None,
            val_names=datamodule.val_names,
            seed=self.method_seed,
        )
        return Run(datamodule, plan)


def save_run(path: str | Path, spec: RunConfig, module: torch.nn.Module, *, allow_overwrite: bool = False) -> None:
    """Writes ``spec`` and the weights of ``module`` to the folder ``path``.

    :raises FileExistsError: If ``path`` holds a run and `allow_overwrite` is `False`.
    :raises scfit.registry.PortabilityError: If the spec holds a live object, before anything is written.
    """
    save_pretrained(path, spec, {"module": module.state_dict()}, overwrite=allow_overwrite)


def load_run(
    path_or_repo_id: str | Path,
    adata: AnnData,
    module: torch.nn.Module | None = None,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    revision: str | None = None,
) -> Run:
    """Rebuilds a run from a folder or a Hub repo id: the data module, and a plan over ``module``.

    :param adata: The data to attach; the schema comes from the saved spec, not from this.
    :param module: A freshly built module of the right shape; ``None`` builds it from the spec's ``module``.
    :param optimizer: Overrides the spec's ``optimizer``.
    :param revision: The Hub revision to load, for a repo id.
    """
    spec, states = load_pretrained(path_or_repo_id, revision=revision)
    if not isinstance(spec, RunConfig):
        raise TypeError(f"{path_or_repo_id} holds a {type(spec).__name__}, not a run.")
    if "module" not in states:
        raise ValueError(f"{path_or_repo_id} holds no module weights (no module/model.safetensors).")
    run = spec.build(adata, module, optimizer=optimizer)
    run.plan.module.load_state_dict(states["module"])
    return run
