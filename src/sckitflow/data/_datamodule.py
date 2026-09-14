"""The `lightning.pytorch` data module: schema plus an ``AnnData`` becomes loaders."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Unpack

import cloudpickle
import lightning.pytorch as pl
import torch
from anndata import AnnData

from sckitflow.data._dims import DataDimensions
from sckitflow.data._manager import DataManager, DataManagerKwargs

if TYPE_CHECKING:
    from sckitflow.data._loader import Loader, LoaderKwargs

__all__ = ["FlowDataModule"]


class FlowDataModule(pl.LightningDataModule):
    """Streams ``StepData`` for training and validation, one loader per split.

    The data half of a run: it owns the schema (:class:`DataManager`) and the
    dimensionalities derived from it, and turns them plus an ``AnnData`` into
    loaders. Training itself is a :class:`~sckitflow.trainer.TrainingPlan` and a
    ``lightning.Trainer``; neither needs to know how a batch was assembled.

    .. code-block:: python

        dmod = FlowDataModule.from_adata(adata, conditions=..., groups=..., split_by="split")
        module = MLPVelocity(dmod.data_dims.state_dim)
        plan = TrainingPlan(CFMTraining(module=module), torch.optim.Adam(module.parameters()))
        pl.Trainer(max_steps=1000).fit(plan, datamodule=dmod)

    The schema rides along in the Lightning checkpoint: `state_dict` is written
    into it by ``Trainer.save_checkpoint`` and read back by
    :meth:`load_from_checkpoint`, so weights and schema travel in one file.

    :param dm: The fitted data manager describing the data schema.
    :param data_dims: The dimensionalities derived from the registration data.
    :param adata: The annotated data object to stream. May be attached later
        with :meth:`attach`, which is what reloading does.
    :param control_adata: Optional separate control (source) pool, shared by
        every split.
    :param train_split: The split value whose loader drives optimization.
    :param n_train_steps: Batches the training loader yields per epoch; the
        ``Trainer`` decides how many are actually consumed.
    :param batch_size: Observations per streamed batch.
    :param dtype: The dtype every streamed tensor is built with. Must match the
        module's, or the forward pass fails on a dtype mismatch -- coercing the
        batch downstream would only hide that. Defaults to ``torch.float32``,
        torch's own default for module parameters.
    :param loader_kwargs: Options forwarded to every streaming loader. A
        ``device`` here is honoured but not needed: batches are built on CPU and
        Lightning moves them to the accelerator.
    """

    def __init__(
        self,
        dm: DataManager,
        data_dims: DataDimensions,
        adata: AnnData | None = None,
        *,
        control_adata: AnnData | None = None,
        train_split: str = "train",
        n_train_steps: int = 100_000,
        batch_size: int = 128,
        dtype: torch.dtype = torch.float32,
        loader_kwargs: LoaderKwargs | None = None,
    ) -> None:
        super().__init__()
        self._dtype = dtype
        self._dm = dm
        self._data_dims = data_dims
        self._adata = adata
        self._control_adata = control_adata
        self._train_split = train_split
        self._n_train_steps = n_train_steps
        self._batch_size = batch_size
        self._loader_kwargs = dict(loader_kwargs or {})
        self._loaders: dict[str, Loader] | None = None

    @classmethod
    def from_adata(
        cls,
        adata: AnnData,
        *,
        control_adata: AnnData | None = None,
        train_split: str = "train",
        n_train_steps: int = 100_000,
        batch_size: int = 128,
        dtype: torch.dtype = torch.float32,
        loader_kwargs: LoaderKwargs | None = None,
        **dm_kwargs: Unpack[DataManagerKwargs],
    ) -> FlowDataModule:
        """Fits the schema on ``adata`` and derives the data dimensionalities.

        Any preprocessing of the state representation must already have been
        applied by the caller.
        """
        dm = DataManager(**dm_kwargs)
        return cls(
            dm,
            dm.get_data_dimensionalities(adata),
            adata,
            control_adata=control_adata,
            train_split=train_split,
            n_train_steps=n_train_steps,
            batch_size=batch_size,
            dtype=dtype,
            loader_kwargs=loader_kwargs,
        )

    def attach(self, adata: AnnData, *, control_adata: AnnData | None = None) -> FlowDataModule:
        """Points this schema at an ``AnnData``, e.g. after reloading. Returns self."""
        self._adata = adata
        if control_adata is not None:
            self._control_adata = control_adata
        self._loaders = None
        return self

    # ---------------- Lightning hooks ----------------
    def setup(self, stage: str | None = None) -> None:
        """Builds one loader per split. Called by Lightning before fitting."""
        if self._loaders is not None:
            return
        if self._adata is None:
            raise ValueError("no `adata` attached: build with `from_adata` or call `attach(adata)`.")

        kwargs = dict(self._loader_kwargs)
        kwargs.setdefault("to", None)
        kwargs.setdefault("batch_size", self._batch_size)
        # Build the batch in the module's dtype. Without this the loader emits the
        # AnnData's native dtype (often float64) against a float32 module.
        kwargs.setdefault("dtype", self._dtype)
        self._loaders = self._dm.get_dataloaders(self._adata, control_adata=self._control_adata, **kwargs)
        if self._train_split not in self._loaders:
            raise KeyError(
                f"train split {self._train_split!r} has no loader; available splits: {list(self._loaders)}. "
                "(A split with only control groups produces no loader.)"
            )

    def train_dataloader(self) -> Loader:
        self.setup()
        return self._loaders[self._train_split].set_n_iters(self._n_train_steps)

    def val_dataloader(self) -> list[Loader]:
        """One loader per non-training split, positionally matching :attr:`val_names`."""
        self.setup()
        return [loader for split, loader in self._loaders.items() if split != self._train_split]

    # ---------------- Serialization ----------------
    def state_dict(self) -> dict[str, Any]:
        """The schema, for the Lightning checkpoint.

        The ``AnnData`` is deliberately left out -- a checkpoint should not carry
        the dataset. Reattach it with :meth:`attach` after loading.
        """
        return {
            "dm": cloudpickle.dumps(self._dm),
            "data_dims": cloudpickle.dumps(self._data_dims),
            "train_split": self._train_split,
            "n_train_steps": self._n_train_steps,
            "batch_size": self._batch_size,
            "dtype": self._dtype,
            "loader_kwargs": self._loader_kwargs,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._dm = cloudpickle.loads(state_dict["dm"])
        self._data_dims = cloudpickle.loads(state_dict["data_dims"])
        self._train_split = state_dict["train_split"]
        self._n_train_steps = state_dict["n_train_steps"]
        self._batch_size = state_dict["batch_size"]
        self._dtype = state_dict["dtype"]
        self._loader_kwargs = state_dict["loader_kwargs"]
        self._loaders = None

    @classmethod
    def load_from_checkpoint(cls, filepath: str, adata: AnnData | None = None) -> FlowDataModule:
        """Rebuilds the data module from the schema stored in a Lightning checkpoint."""
        import torch

        ckpt = torch.load(filepath, weights_only=False, map_location="cpu")
        # Lightning files the data module state under its class `__qualname__`.
        try:
            state = ckpt[cls.__qualname__]
        except KeyError as e:
            raise KeyError(
                f"{filepath} carries no {cls.__qualname__} state; it was saved by a `Trainer` "
                "that was not given `datamodule=`."
            ) from e

        dmod = cls(cloudpickle.loads(state["dm"]), cloudpickle.loads(state["data_dims"]))
        dmod.load_state_dict(state)
        return dmod if adata is None else dmod.attach(adata)

    # ---------------- Accessors ----------------
    @property
    def dm(self) -> DataManager:
        """The fitted data manager."""
        return self._dm

    @property
    def data_dims(self) -> DataDimensions:
        """The data dimensionalities derived from the registration data."""
        return self._data_dims

    @property
    def adata(self) -> AnnData | None:
        """The attached annotated data object, if any."""
        return self._adata

    @property
    def val_names(self) -> list[str]:
        """Names of the validation splits, in `val_dataloader` order."""
        self.setup()
        return [split for split in self._loaders if split != self._train_split]
