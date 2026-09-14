"""The `lightning.pytorch` module that runs a sckitflow training method.

Everything a training run needs -- the loop, the backward pass, optimizer
stepping, the progress bar, device placement, validation cadence, logging,
checkpointing -- comes from Lightning. This file only says what one step is.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import lightning.pytorch as pl
import torch

from sckitflow.core._types import StepData
from sckitflow.core.methods._base import SupportsInference, SupportsTraining

__all__ = ["TrainingPlan"]


class TrainingPlan(pl.LightningModule):
    """Runs a `SupportsTraining` method, and optionally scores a `SupportsInference` one.

    Hand it to a ``lightning.Trainer``:

    .. code-block:: python

        plan = TrainingPlan(CFMTraining(module=module), torch.optim.Adam(module.parameters()))
        pl.Trainer(max_steps=1000).fit(plan, train_loader)

    :param training_method: The method whose `compute_loss` is one training step.
    :param optimizer: An already-built optimizer over the method's module.
        Taken rather than configured, so there is no optimizer factory here.
    :param lr_scheduler: Optional scheduler for that optimizer.
    :param lr_scheduler_interval: ``"step"`` or ``"epoch"``; how often Lightning
        steps the scheduler.
    :param inference_method: The method used for validation. When `None`, or
        when no metrics are given, validation produces nothing.
    :param metrics: ``{name: torchmetrics.Metric}`` scored on every validation
        batch. Registered as submodules, so Lightning moves and resets them.
    :param val_names: Names of the validation sets, positionally matching the
        ``val_dataloaders`` handed to ``lightning.Trainer``. Lightning
        identifies those by index; metrics are logged under a name.
    :param pred_transform: Optional callable applied to predictions before scoring.
    :param target_transform: Optional callable applied to targets before scoring.
    :param predict_kwargs: Forwarded to the inference method's ``predict``.
    """

    def __init__(
        self,
        training_method: SupportsTraining,
        optimizer: torch.optim.Optimizer,
        *,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        lr_scheduler_interval: str = "step",
        inference_method: SupportsInference | None = None,
        metrics: Mapping[str, torch.nn.Module] | None = None,
        val_names: Sequence[str] = ("val",),
        pred_transform: Callable[[Any], Any] | None = None,
        target_transform: Callable[[Any], Any] | None = None,
        predict_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.training_method = training_method
        self.inference_method = inference_method
        # Registers the method's parameters with Lightning, which is what makes
        # optimizer wiring, device placement and train/eval mode work.
        self.module = training_method.module
        # `ModuleDict` so the metrics ride along to the accelerator.
        self.metrics = torch.nn.ModuleDict(dict(metrics)) if metrics else None

        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._lr_scheduler_interval = lr_scheduler_interval
        self._val_names = list(val_names)
        self._pred_transform = pred_transform
        self._target_transform = target_transform
        self._predict_kwargs = {} if predict_kwargs is None else predict_kwargs

    def training_step(self, batch: StepData, batch_idx: int) -> torch.Tensor:
        loss, metrics = self.training_method.compute_loss(batch)
        self.log_dict(metrics, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch: StepData, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Scores one validation batch. Metrics aggregate until the epoch ends."""
        if self.inference_method is None or self.metrics is None:
            return

        preds = self.inference_method.predict(batch, **self._predict_kwargs)
        preds = getattr(preds, "X", preds)
        targets = batch["target_state"]
        if self._pred_transform is not None:
            preds = self._pred_transform(preds)
        if self._target_transform is not None:
            targets = self._target_transform(targets)

        for metric in self.metrics.values():
            metric.update(torch.as_tensor(preds), torch.as_tensor(targets))

    def on_validation_epoch_end(self) -> None:
        if self.metrics is None or self.trainer.sanity_checking:
            return
        val_name = self._val_names[0] if self._val_names else "val"
        for name, metric in self.metrics.items():
            self.log(f"{val_name}/{name}", metric.compute())
            metric.reset()

    def configure_optimizers(self) -> Any:
        if self._lr_scheduler is None:
            return self._optimizer
        return {
            "optimizer": self._optimizer,
            "lr_scheduler": {"scheduler": self._lr_scheduler, "interval": self._lr_scheduler_interval},
        }
