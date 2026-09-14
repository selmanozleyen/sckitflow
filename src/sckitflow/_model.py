from __future__ import annotations

import logging
import tarfile
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

import cloudpickle
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from tqdm import tqdm

from sckitflow._types import PredictionData
from sckitflow.core._types import StepData
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
    def _dims(self) -> DataDimensions:
        return self._datamodule.data_dims

    def plan(
        self,
        optimizer: torch.optim.Optimizer,
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
    def _predict_empty(
        self,
        return_raw: Literal[False],
    ) -> AnnData:
        pass

    @overload
    def _predict_empty(
        self,
        return_raw: Literal[True],
    ) -> tuple[AnnData, None]:
        pass

    @overload
    def _aggregate_nodes_pred(
        self,
        all_preds: list[PredictionData],
        all_obs: list[pd.DataFrame],
        all_obsm: dict[str, list[np.ndarray]],
        return_raw: Literal[False],
    ) -> AnnData:
        pass

    @overload
    def _aggregate_nodes_pred(
        self,
        all_preds: list[PredictionData],
        all_obs: list[pd.DataFrame],
        all_obsm: dict[str, list[np.ndarray]],
        return_raw: Literal[True],
    ) -> tuple[AnnData, PredictionData]:
        pass

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

    def _predict_empty(self, return_raw: bool) -> AnnData | tuple[AnnData, None]:
        """Returns empty anndata for prediction."""
        empty_adata = AnnData(
            X=np.empty((0, len(self._dims.feature_names))),
            var=pd.DataFrame(index=self._dims.feature_names),
        )
        return empty_adata if not return_raw else (empty_adata, None)

    def _pred_obs_from_leaf(self, group_cols: tuple[str, ...], leaf: tuple, pred_obj: PredictionData) -> pd.DataFrame:
        """Rebuild a group's obs rows from its ``leaf`` (the ``group_by`` value tuple), one per predicted observation."""
        n_pred_obs = pred_obj.X.shape[0] if getattr(pred_obj, "X", None) is not None else 1
        return pd.DataFrame({col: np.repeat(val, n_pred_obs) for col, val in zip(group_cols, leaf, strict=True)})

    def _get_pred_traj(self, pred_obj: PredictionData) -> np.ndarray | None:
        if pred_obj.traj is None:
            return None

        n_obs = pred_obj.X.shape[0]
        traj_np = self._to_numpy(pred_obj.traj)

        if traj_np.ndim == 2 and traj_np.shape[0] == n_obs:
            return traj_np
        elif traj_np.ndim == 3 and traj_np.shape[1] == n_obs:
            return np.transpose(traj_np, (1, 0, 2))
        elif traj_np.ndim == 4 and traj_np.shape[2] == n_obs:
            return np.transpose(traj_np, (2, 0, 1, 3))
        else:
            raise ValueError(
                "Trajectory array has incompatible shape for AnnData.obsm: "
                f"got {traj_np.shape}, expected first dimension to equal "
                f"n_obs ({n_obs}) or, for 3D trajectories, second "
                "dimension to equal n_obs so it can be transposed from "
                "(n_time_steps, n_obs, n_features) to "
                "(n_obs, n_time_steps, n_features)."
            )

    def _get_pred_raw_samples(self, pred_obj: PredictionData) -> np.ndarray | None:
        raw_samples = getattr(pred_obj, "raw_samples", None)
        if raw_samples is None:
            return None

        X = getattr(pred_obj, "X", None)
        if X is None:
            raise ValueError("Prediction object should have the .X attribute.")
        n_obs = X.shape[0]

        samples_np = self._to_numpy(raw_samples)
        if samples_np.ndim == 2 and samples_np.shape[0] == n_obs:
            return samples_np
        elif samples_np.ndim == 3 and samples_np.shape[1] == n_obs:
            return np.transpose(samples_np, (1, 0, 2))
        else:
            raise ValueError(
                "Samples array has incompatible shape for AnnData.obsm: "
                f"got {samples_np.shape}, expected data of shape "
                f"(n_obs, n_features) or (n_samples, n_obs, n_features)"
            )

    def _get_pred_obsm_dict(
        self, step_data: StepData, pred_obj: PredictionData, cont_keys: tuple[str, ...]
    ) -> dict[str, np.ndarray]:
        obsm_dict: dict[str, np.ndarray] = {}
        traj = self._get_pred_traj(pred_obj)
        if traj is not None:
            obsm_dict["trajectory"] = traj
        raw_samples = self._get_pred_raw_samples(pred_obj)
        if raw_samples is not None:
            obsm_dict["raw_samples"] = raw_samples

        condition = step_data["target_condition_data"] or {}
        response = step_data["target_response_data"] or {}
        for key in cont_keys:
            if key in condition:
                obsm_dict[key] = self._to_numpy(condition[key])
            elif key in response:
                obsm_dict[key] = self._to_numpy(response[key])
        return obsm_dict

    def _aggregate_nodes_pred(
        self,
        all_preds: list[PredictionData],
        all_obs: list[pd.DataFrame],
        all_obsm: dict[str, list[np.ndarray]],
        return_raw: bool = False,
    ) -> AnnData | tuple[AnnData, PredictionData]:
        merged_pred = type(all_preds[0]).concatenate(all_preds)

        X_np = self._to_numpy(merged_pred.X)

        obs_final = pd.concat(all_obs, axis=0, ignore_index=True)
        obs_final.index = obs_final.index.astype(str)

        obsm_final = {k: np.concatenate(v, axis=0) for k, v in all_obsm.items()}

        pred_adata = AnnData(
            X=X_np, obs=obs_final, var=pd.DataFrame(index=self._dims.feature_names), obsm=obsm_final
        )

        if return_raw:
            return pred_adata, merged_pred

        return pred_adata

    def _to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert a torch tensor (or array-like) to a numpy array."""
        if tensor is None:
            return None
        import torch

        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.array(tensor)

    def to_device(self, device: str) -> None:
        """Move the underlying PyTorch module to the specified device.

        Optimizer state is Lightning's to place -- it is re-homed on the next
        ``fit``, so there is nothing to walk here.
        """
        self._module.to(device)

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

        inference_method.module.train(False)
        param = next(inference_method.module.parameters(), None)
        module_dtype = torch.float32 if param is None else param.dtype
        module_device = "cpu" if param is None else param.device
        predict_kwargs = {} if predict_kwargs is None else predict_kwargs

        eval_loader = self._dm.get_eval_loader(
            adata,
            max_per_group=max_per_group,
            require_target_state=require_target_state,
            control_values_dict=control_values_dict,
            matched_keys=matched_keys,
            control_adata=control_adata,
            to=None,
            # the module is the reference: build the batch where it already lives
            dtype=module_dtype,
            device=str(module_device),
        )

        if len(eval_loader) == 0:
            return self._predict_empty(return_raw)

        all_preds = []
        all_obs = []
        all_obsm = defaultdict(list)
        group_cols = eval_loader.group_cols
        cont_keys = (*eval_loader.cond_cont_keys, *eval_loader.resp_keys)

        for step_data, leaf in tqdm(eval_loader, total=len(eval_loader), desc="Predicting"):
            pred_obj = inference_method.predict(step_data, **predict_kwargs)
            all_preds.append(pred_obj)

            all_obs.append(self._pred_obs_from_leaf(group_cols, leaf, pred_obj))

            node_obsm_dict = self._get_pred_obsm_dict(step_data, pred_obj, cont_keys)
            for key, val in node_obsm_dict.items():
                all_obsm[key].append(val)

        return self._aggregate_nodes_pred(all_preds, all_obs, all_obsm, return_raw=return_raw)

    def save(self, filepath: str, allow_overwrite: bool = False) -> None:
        """Save the entire model (including registered data) to a tarball."""
        path = Path(filepath)
        if path.exists() and not allow_overwrite:
            raise FileExistsError(f"{filepath} already exists. Use allow_overwrite=True.")
        elif path.exists() and allow_overwrite:
            path.unlink()

        self._module.cpu()

        with tarfile.open(filepath, "w:gz") as tar:
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
