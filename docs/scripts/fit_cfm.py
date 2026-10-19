"""Train a conditional flow matching run from ``docs/configs/fit_cfm.yaml``.

    python docs/scripts/fit_cfm.py [hydra overrides, e.g. run.training.coupling.type=coupling.pot.sinkhorn]

Needs ``hydra-core`` and ``wandb`` (the ``scripts`` extra). ``config.run`` is a portable
:class:`~sckitflow.RunConfig` spec, built as is; the other sections configure the Lightning
loggers, callbacks and trainer around it. After training, the best checkpoint's weights are saved
with the spec via :func:`~sckitflow.save_run`, so :func:`~sckitflow.load_run` rebuilds the run.
Returns the last validation energy distance (``inf`` if validation never ran), for sweeps.
"""

import logging
from pathlib import Path
from typing import Any, cast

import anndata as ad
import hydra
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    EarlyStopping,
    GradientAccumulationScheduler,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf

from sckitflow import RunConfig, save_run
from sckitflow.core.metrics import EnergyDistance, MaximumMeanDiscrepancy

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="fit_cfm", version_base=None)
def main(config: DictConfig) -> float:
    """Trains ``config.run`` and returns its last validation energy distance."""
    cfg = cast(dict[str, Any], OmegaConf.to_container(config, resolve=True))
    run_dir, allow_overwrite = Path(cfg["save"]["run_dir"]), cfg["save"]["allow_overwrite"]
    if (run_dir / "specs.json").exists() and not allow_overwrite:  # fail before training, not after
        raise FileExistsError(f"{run_dir} already holds a run; set save.allow_overwrite=true.")

    spec = RunConfig.from_spec(cfg["run"])
    seed_everything(spec.method_seed, workers=True)
    paths = cfg["paths"]
    adata = ad.read_h5ad(paths["target_adata"])
    control_adata = ad.read_h5ad(paths["control_adata"]) if paths["control_adata"] else None
    metrics = {
        "edist": EnergyDistance(**cfg["metrics"]["edist"]),
        "mmd": MaximumMeanDiscrepancy(**cfg["metrics"]["mmd"]),
    }
    datamodule, plan = spec.build(adata, control_adata=control_adata, metrics=metrics)

    callbacks = cfg["callbacks"]
    checkpoint = ModelCheckpoint(**callbacks["model_checkpoint"])
    trainer = Trainer(
        **cfg["trainer"],
        logger=[WandbLogger(**cfg["loggers"]["wandb"]), CSVLogger(**cfg["loggers"]["csv"])],
        callbacks=[
            EarlyStopping(**callbacks["early_stopping"]),
            GradientAccumulationScheduler(**callbacks["grad_accumulation"]),
            LearningRateMonitor(**callbacks["lr_monitor"]),
            checkpoint,
        ],
    )
    trainer.fit(plan, datamodule=datamodule, ckpt_path=cfg["ckpt_path"])

    if checkpoint.best_model_path:  # save the best weights, not the last epoch's
        plan.load_state_dict(
            torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)["state_dict"]
        )
    save_run(run_dir, spec, plan.module, allow_overwrite=allow_overwrite)
    score = float(trainer.callback_metrics.get(f"{datamodule.val_names[0]}/edist", float("inf")))
    logger.info(f"Saved the run to {run_dir}; validation energy distance {score:.4f}.")
    return score


if __name__ == "__main__":
    main()
