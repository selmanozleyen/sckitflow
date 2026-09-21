from __future__ import annotations

import io
import json
import logging
import tarfile
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

import cloudpickle
import torch
from anndata import AnnData
from tqdm import tqdm

from sckitflow._predict import prediction_record, predictions_to_adata
from sckitflow._serialization import name_for_saving
from sckitflow.core._types import PredictionData
from sckitflow.core.methods._base import SupportsInference, SupportsTraining
from sckitflow.data._datamodule import FlowDataModule
from sckitflow.data._dims import DataDimensions
from sckitflow.data._manager import DataManager
from sckitflow.trainer._plan import TrainingPlan

if TYPE_CHECKING:
    pass

__all__ = ["Model"]


class Model:
    """Methods plus a data module: predicts, and serializes the whole run.

    Training is not here -- it is a :class:`~sckitflow.trainer.TrainingPlan` and
    a ``lightning.Trainer``, which :meth:`plan` builds for you:

    .. code-block:: python

        dmod = FlowDataModule.from_adata(adata, conditions=..., groups=...)
        module = MLPVelocity(dmod.data_dims.state_dim)
        model = Model(dmod, CFMTraining(module=module), ODEInference(module=module))
        pl.Trainer(max_steps=1000).fit(model.plan(optimizer), datamodule=dmod)
        model.save("run.tar.gz")
    """

    def __init__(
        self,
        datamodule: FlowDataModule,
        training_method: SupportsTraining,
        inference_method: SupportsInference | None = None,
    ) -> None:
        """Initialize a model from a data module and the methods to run.

        The model takes methods, it does not build them: construct them
        yourself (they own the module and its configuration) and hand them over.
        The module is read back off the training method. To match `StepData`
        before the loss, wrap the training method in a
        :class:`~sckitflow.core.methods._base.MatchedTrainingMethod` before
        passing it -- the model itself knows nothing about matching.

        :param datamodule: The fitted :class:`~sckitflow.data.FlowDataModule`
            holding the schema and the streaming loaders.
        :type datamodule: class: `FlowDataModule`

        :param training_method: An object satisfying `SupportsTraining`.
        :type training_method: class: `SupportsTraining`

        :param inference_method: An object satisfying `SupportsInference`, or
            `None` for a training-only model -- `Model.predict` and validation
            then raise. May be the very same object as `training_method` when
            one class implements both contracts; they are kept as separate
            arguments so a pair that does not is equally natural.
        :type inference_method: class: `SupportsInference | None`
        """
        self._datamodule = datamodule

        # ----- Take the methods as given -----
        self._training_method: SupportsTraining = training_method
        self._inference_method: SupportsInference | None = inference_method

        # The module to optimize, save and move between devices. Both contracts
        # expose it, so there is nothing for the model to build or reconcile.
        self._module: torch.nn.Module = training_method.module

    # The schema lives in the data module; read it through, never copy it, so
    # reattaching or reloading the data module cannot leave the model stale.
    @property
    def _dm(self) -> DataManager:
        return self._datamodule.dm

    @property
    def _data_dims(self) -> DataDimensions:
        return self._datamodule.data_dims

    def plan(
        self,
        optimizer: torch.optim.Optimizer | None = None,
        *,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        metrics: Mapping[str, torch.nn.Module] | None = None,
        val_predict_kwargs: dict[str, Any] | None = None,
    ) -> TrainingPlan:
        """Wraps this model's methods in a `TrainingPlan` for ``Trainer.fit``.

        :param optimizer: An already-built optimizer over this model's module,
            e.g. ``torch.optim.Adam(model.module.parameters(), lr=1e-4)``.
        :param lr_scheduler: Optional scheduler for that optimizer.
        :param metrics: ``{name: torchmetrics.Metric}`` scored during validation.
        :param val_predict_kwargs: Forwarded to the inference method's ``predict``.
        """
        return TrainingPlan(
            self._training_method,
            optimizer,
            lr_scheduler=lr_scheduler,
            inference_method=self._inference_method,
            metrics=metrics,
            val_names=self._datamodule.val_names,
            predict_kwargs=val_predict_kwargs,
        )

    @overload
    def predict(
        self,
        adata: AnnData,
        *,
        return_raw: Literal[False] = ...,
        **kwargs,
    ) -> AnnData:
        pass

    @overload
    def predict(
        self,
        adata: AnnData,
        *,
        return_raw: Literal[True],
        **kwargs,
    ) -> tuple[AnnData, PredictionData]:
        pass

    def to_device(self, device: str) -> None:
        """Move the underlying PyTorch module to the specified device.

        Optimizer state is Lightning's to place -- it is re-homed on the next
        ``fit``, so there is nothing to walk here.
        """
        self._module.to(device)

    @torch.inference_mode()
    def predict(
        self,
        adata: AnnData,
        *,
        inference_method: SupportsInference | None = None,
        return_raw: bool = False,
        max_per_group: int | None = None,
        require_target_state: bool = True,
        control_values_dict: dict[str, str] | None = None,
        matched_keys: Mapping[tuple, tuple] | None = None,
        control_adata: AnnData | None = None,
        predict_kwargs: dict[str, Any] | None = None,
    ) -> AnnData | tuple[AnnData, PredictionData]:
        """Generate flow predictions, one deterministic pass per group via :class:`EvalLoader`.

        :param adata: The input adata containing the metadata for prediction.
        :param inference_method: Overrides the model's inference method for this call only.
        :param return_raw: If ``True``, also return the raw concatenated ``PredictionData``.
        :param max_per_group: Per-group cap on observations.
        :param require_target_state: Whether ``adata`` must carry a target state representation.
        :param control_values_dict: Optional mapping from each condition level to its control value.
        :param matched_keys: Optional ``{source group key: target group key}`` pairs for fixed matching.
        :param control_adata: Optional separate control (source) pool.
        :param predict_kwargs: Forwarded to the inference method's ``predict``.
        :return: An AnnData with predictions, or a tuple ``(AnnData, PredictionData)`` if ``return_raw``.
        """
        if inference_method is None:
            inference_method = self._inference_method
        if inference_method is None:
            raise ValueError("this model has no inference method: pass one to `Model(...)` or to `predict(...)`.")

        loader = self._datamodule.set_predict_data(
            adata,
            max_per_group=max_per_group,
            require_target_state=require_target_state,
            control_values_dict=control_values_dict,
            matched_keys=matched_keys,
            control_adata=control_adata,
        ).predict_dataloader()
        predict_kwargs = {} if predict_kwargs is None else predict_kwargs

        # A plain pass, not `Trainer.predict`: predicting needs no loop machinery,
        # and requiring a `lightning.Trainer` just to call this would be ceremony.
        # For multi-device prediction or a `BasePredictionWriter`, use the Lightning
        # path instead -- `trainer.predict(model.plan(), datamodule=model.datamodule)`.
        was_training = inference_method.module.training
        inference_method.module.eval()
        records = []
        try:
            for step_data, leaf in tqdm(loader, total=len(loader), desc="Predicting"):
                preds = inference_method.predict(step_data, **predict_kwargs)
                records.append(prediction_record(loader, step_data, leaf, preds))
        finally:
            inference_method.module.train(was_training)

        return predictions_to_adata(self._data_dims, records, return_raw=return_raw)

    def manifest(self) -> dict[str, Any]:
        """Which classes this model is built from, by registered name.

        Recording names rather than pickled class references is what lets a saved
        run survive those classes being renamed or moved. Anything can be
        *constructed* and trained; only registered classes can be written down,
        so this raises rather than producing an artifact nothing can read back.

        :raises TypeError: If a method or the module is not registered.
        """
        entries = {
            "training_method": name_for_saving(self._training_method, kind="method"),
            "module": name_for_saving(self._module, kind="module"),
        }
        if self._inference_method is not None:
            entries["inference_method"] = name_for_saving(self._inference_method, kind="method")
        return entries

    def save(self, filepath: str, allow_overwrite: bool = False) -> None:
        """Save the entire model (including registered data) to a tarball.

        Writes a ``manifest.json`` of registered class names beside the pickle,
        so the archive says what it holds without being unpickled -- and so an
        unregistered class is refused here rather than at load time.
        """
        path = Path(filepath)
        if path.exists() and not allow_overwrite:
            raise FileExistsError(f"{filepath} already exists. Use allow_overwrite=True.")
        elif path.exists() and allow_overwrite:
            path.unlink()

        # before writing anything: refuse a model that cannot be named
        manifest = json.dumps(self.manifest(), indent=2).encode()

        self._module.cpu()

        with tarfile.open(filepath, "w:gz") as tar:
            info = tarfile.TarInfo(name="manifest.json")
            info.size = len(manifest)
            tar.addfile(info, io.BytesIO(manifest))
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
                cloudpickle.dump(self, tmp)
                tmp.flush()
                tar.add(tmp.name, arcname="model.pkl")
            Path(tmp.name).unlink()

        logging.info(f"Model saved to {filepath} (moved to CPU).")

    @classmethod
    def load(
        cls,
        filepath: str,
        adata: AnnData | None = None,
        map_location: str | None = None,
        **attach_kwargs,
    ) -> Model:
        """Load a saved model from a tarball."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"{filepath} not found.")

        with tempfile.TemporaryDirectory() as tmpdir:
            extract_dir = Path(tmpdir).resolve()
            with tarfile.open(filepath, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.issym() or member.islnk():
                        raise ValueError(f"Refusing to extract link from archive: {member.name}")

                    member_path = (extract_dir / member.name).resolve()
                    try:
                        member_path.relative_to(extract_dir)
                    except ValueError as e:
                        raise ValueError(
                            f"Refusing to extract archive member outside target directory: {member.name}"
                        ) from e

                    tar.extract(member, tmpdir)

            with open(Path(tmpdir) / "model.pkl", "rb") as f:
                model = cloudpickle.load(f)

        if adata is not None:
            # Attach, not re-fit: the saved schema is the one the model trained
            # with, so rebuilding it from `adata` could silently change it.
            model._datamodule.attach(adata, **attach_kwargs)

        if map_location is not None:
            model.to_device(map_location)

        return model

    @property
    def datamodule(self) -> FlowDataModule:
        """The data module holding the schema and the streaming loaders."""
        return self._datamodule

    @property
    def dm(self) -> DataManager:
        """Returns the data manager associated to the current instance."""
        return self._dm

    @property
    def is_paired_setting(self) -> bool:
        """Whether the data was registered in a paired setting."""
        return self._dm.control_values_dict is not None or self._dm.matched_keys is not None

    @property
    def module(self) -> torch.nn.Module:
        """Returns the underlying module."""
        return self._module

    @property
    def training_method(self) -> SupportsTraining:
        """The underlying training method (may be a `MatchedTrainingMethod` wrapper)."""
        return self._training_method

    @property
    def inference_method(self) -> SupportsInference | None:
        """The underlying inference method."""
        return self._inference_method

    @property
    def condition_state_key(self) -> str | None:
        """Return the key used to extract the state from the condition."""
        return self._dm.condition_state_key
