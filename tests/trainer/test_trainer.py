# tests/trainer/test_trainer.py
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from sckitflow.core.methods._opt import OptimizationManager
from sckitflow.trainer._callbacks import ComputationalCallback, LoggingCallback
from sckitflow.trainer._trainer import Trainer

from sckitflow.core.methods._base import (
    BaseInferenceProtocol,
    BaseTrainingProtocol,
    MatchedTrainingMethod,
    ProtocolSpecs,
    SupportsInference,
    SupportsProtocol,
    SupportsTraining,
)
from sckitflow.core.nn._modules import BaseModule


# -----------------------------------------------------------------------------
# Dummy module
# -----------------------------------------------------------------------------
class DummyModule(BaseModule):
    """Minimal module with parameters, enough to satisfy `ProtocolSpecs.__init__`."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def _make_modules(self, data_dims=None, *args, **kwargs):
        # BaseModule may call this during setup; keep it a no-op.
        pass

    def forward(self, t, x, condition_dict=None, source=None):
        return self.linear(x)


class DummyPredictionData:
    """Minimal stand-in for the real ``PredictionData``; the Trainer reads ``.X``."""

    def __init__(self, X, traj=None, raw_samples=None):
        self.X = X
        self.traj = traj
        self.raw_samples = raw_samples


class DummyStepData(dict):
    """Minimal `StepData` stand-in.

    Always carries the four coupling keys (defaulting to `None`), matching the
    contract `Matcher.match` relies on. Any extra key can be passed as
    a keyword argument.
    """

    def __init__(self, **kwargs):
        super().__init__(
            source_coupling_lin=None,
            source_coupling_quad=None,
            target_coupling_lin=None,
            target_coupling_quad=None,
        )
        self.update(kwargs)


# -----------------------------------------------------------------------------
# Dummy methods. Both are constructed with a shared `ProtocolSpecs`, matching
# the holder-based design introduced with the `FlowSpecs` refactor.
# -----------------------------------------------------------------------------
class DummyTrainingProtocol(BaseTrainingProtocol):
    """Concrete training method: a constant loss and metric dict."""

    def compute_loss(self, step_data, *args, **kwargs):
        return 0.5, {"loss": 0.5, "accuracy": 0.8}


class DummyInferenceProtocol(BaseInferenceProtocol):
    """Concrete inference method: returns a `PredictionData` with `.X`."""

    def predict(self, step_data, *args, **kwargs):
        rng = np.random.default_rng(0)
        return DummyPredictionData(rng.standard_normal((10, 5)), traj=None, raw_samples=None)


# Module-level so `cloudpickle` can serialize it (not needed for these tests, but
# harmless and consistent with how `match_fn` would be used in production).
def dummy_match_fn(source_lin=None, target_lin=None, source_quad=None, target_quad=None):
    """No-op matcher: returns no indices so `Matcher.match` short-circuits."""
    return None, None


# -----------------------------------------------------------------------------
# Dummy optimizer manager
# -----------------------------------------------------------------------------
class DummyOptManager(OptimizationManager):
    def __init__(self):
        super().__init__(None, None, None)

    def step(self, loss):
        pass


# -----------------------------------------------------------------------------
# Dummy loaders
# -----------------------------------------------------------------------------
class DummyTrainLoader:
    """Finite, re-iterable loader yielding `n` `DummyStepData` batches.

    Uses `DummyStepData` (a dict-like) rather than a bare `Mock` so that
    methods which subscript `step_data` — e.g. `MatchedTrainingMethod`
    routing through `Matcher.match` — work end-to-end.
    """

    def __init__(self, n=2):
        self._n = n

    def __iter__(self):
        return iter([DummyStepData() for _ in range(self._n)])


class DummyValLoader:
    """Yields two `MagicMock` batches (subscriptable, so `batch["target_state"]` works)."""

    def __iter__(self):
        return iter([MagicMock(), MagicMock()])


# -----------------------------------------------------------------------------
# Recording callbacks
# -----------------------------------------------------------------------------
class RecordingCallback(LoggingCallback):
    """Logging callback; `TrainingCallbacks` forwards train hooks here."""

    def __init__(self):
        self.train_begin_calls = []
        self.train_step_calls = []
        self.valid_step_calls = []
        self.train_end_calls = []

    def on_train_begin(self, trainer, **kwargs):
        self.train_begin_calls.append((trainer, kwargs))

    def on_train_step(self, trainer, step, logs, **kwargs):
        self.train_step_calls.append((trainer, step, logs, kwargs))

    def on_valid_step(self, trainer, step, val_id, predictions_dict, **kwargs):
        self.valid_step_calls.append((trainer, step, val_id, predictions_dict, kwargs))

    def on_train_end(self, trainer, **kwargs):
        self.train_end_calls.append((trainer, kwargs))


class RecordingComputationalCallback(ComputationalCallback):
    """Computational callback; receives the raw predictions dict and returns metrics."""

    def __init__(self):
        self.valid_step_calls = []

    def on_valid_step(self, trainer, step, val_id, predictions_dict, **kwargs):
        self.valid_step_calls.append((trainer, step, val_id, predictions_dict, kwargs))
        return {"dummy_metric": 1.0}


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def module():
    return DummyModule()


@pytest.fixture
def specs(module):
    """Shared `ProtocolSpecs` for the dummy methods.

    The base methods no longer inherit from `ProtocolSpecs`; they hold one
    instance and delegate the storage surface to it. Both fixtures below reuse
    the same instance, mirroring how `Model` wires them up.
    """
    return ProtocolSpecs(module, device_id="cpu")


@pytest.fixture
def training_method(specs):
    return DummyTrainingProtocol(specs)


@pytest.fixture
def inference_method(specs):
    return DummyInferenceProtocol(specs)


@pytest.fixture
def opt_manager():
    return DummyOptManager()


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
class TestTrainer:
    # ---- Construction -----------------------------------------------------
    def test_init(self, training_method, inference_method, opt_manager):
        callbacks = [RecordingCallback()]
        trainer = Trainer(
            training_method,
            opt_manager,
            inference_method=inference_method,
            callbacks=callbacks,
        )

        assert trainer.training_method is training_method
        assert trainer.inference_method is inference_method
        assert trainer.opt_manager is opt_manager
        assert len(trainer._callbacks) == 1
        assert trainer.train_logs_raw == []
        assert trainer.val_logs_raw == {}
        assert trainer.current_step == 0

    def test_init_without_inference_method(self, training_method, opt_manager):
        """Inference method is optional; validation is skipped when absent."""
        trainer = Trainer(training_method, opt_manager)
        assert trainer.inference_method is None

    def test_init_rejects_bad_callbacks(self, training_method, opt_manager):
        with pytest.raises(TypeError, match="callbacks"):
            Trainer(training_method, opt_manager, callbacks=42)

    # ---- Log appenders ----------------------------------------------------
    def test_append_train_log(self, training_method, opt_manager):
        trainer = Trainer(training_method, opt_manager)
        trainer._append_train_log({"loss": 0.5})
        assert len(trainer.train_logs_raw) == 1
        assert trainer.train_logs_raw[0]["loss"] == 0.5

    def test_append_val_log_new_key(self, training_method, opt_manager):
        trainer = Trainer(training_method, opt_manager)
        trainer._append_val_log("val1", {"metric": 0.5})
        assert "val1" in trainer.val_logs_raw
        assert trainer.val_logs_raw["val1"][0]["metric"] == 0.5

    def test_append_val_log_existing_key(self, training_method, opt_manager):
        trainer = Trainer(training_method, opt_manager)
        trainer._append_val_log("val1", {"metric": 0.5})
        trainer._append_val_log("val1", {"metric": 0.8})
        assert len(trainer.val_logs_raw["val1"]) == 2

    # ---- Validation pass --------------------------------------------------
    def test_run_val_on_loader(self, training_method, inference_method, opt_manager):
        callback = RecordingCallback()
        metric_cb = RecordingComputationalCallback()
        trainer = Trainer(
            training_method,
            opt_manager,
            inference_method=inference_method,
            callbacks=[metric_cb, callback],
        )
        trainer._current_step = 5

        trainer._run_val_on_loader(DummyValLoader(), "test_val")

        # The val log holds the metrics the callbacks computed, tagged with the step.
        assert "test_val" in trainer.val_logs_raw
        assert len(trainer.val_logs_raw["test_val"]) == 1
        log_entry = trainer.val_logs_raw["test_val"][0]
        assert log_entry["dummy_metric"] == 1.0
        assert log_entry["step"] == 5

        # The raw predictions/targets reach the computational callback, one entry per node.
        assert len(metric_cb.valid_step_calls) == 1
        predictions_dict = metric_cb.valid_step_calls[0][3]
        assert len(predictions_dict) == 2
        assert all(set(v) == {"predictions", "targets"} for v in predictions_dict.values())

        # The logging callback is also notified.
        assert len(callback.valid_step_calls) == 1
        assert callback.valid_step_calls[0][1] == 5
        assert callback.valid_step_calls[0][2] == "test_val"

    def test_run_val_on_loader_no_inference_method(self, training_method, opt_manager):
        """When no inference method is set, validation is a no-op."""
        callback = RecordingCallback()
        trainer = Trainer(training_method, opt_manager, callbacks=[callback])
        trainer._run_val_on_loader(DummyValLoader(), "test_val")
        assert trainer.val_logs_raw == {}
        assert callback.valid_step_calls == []

    # ---- Log DataFrame conversion ----------------------------------------
    def test_get_train_logs_df_empty(self, training_method, opt_manager):
        trainer = Trainer(training_method, opt_manager)
        df = trainer.get_train_logs_df()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_get_train_logs_df_with_data(self, training_method, opt_manager):
        trainer = Trainer(training_method, opt_manager)
        trainer._append_train_log({"loss": 0.5, "step": 0})
        trainer._append_train_log({"loss": 0.3, "step": 1})

        df = trainer.get_train_logs_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "loss" in df.columns
        # `step` becomes the index, so the columns are metrics only.
        assert df.index.name == "step"
        assert "step" not in df.columns
        assert list(df.index) == [0, 1]
        assert list(df["loss"]) == [0.5, 0.3]

    def test_get_val_logs_df_single(self, training_method, opt_manager):
        trainer = Trainer(training_method, opt_manager)
        trainer._append_val_log("val1", {"metric": 0.5})

        df = trainer.get_val_logs_df("val1")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["metric"] == 0.5

    def test_get_val_logs_df_missing(self, training_method, opt_manager):
        trainer = Trainer(training_method, opt_manager)
        df = trainer.get_val_logs_df("nonexistent")
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_get_val_logs_df_all(self, training_method, opt_manager):
        trainer = Trainer(training_method, opt_manager)
        trainer._append_val_log("val1", {"metric": 0.5})
        trainer._append_val_log("val2", {"metric": 0.8})

        result = trainer.get_val_logs_df()
        assert isinstance(result, dict)
        assert set(result.keys()) == {"val1", "val2"}
        assert isinstance(result["val1"], pd.DataFrame)
        assert len(result["val1"]) == 1

    # ---- Training loop ----------------------------------------------------
    @patch("sckitflow.trainer._trainer.tqdm")
    def test_train_calls_callbacks(self, mock_tqdm, training_method, opt_manager):
        mock_pbar = MagicMock()
        mock_pbar.__iter__.return_value = range(3)
        mock_tqdm.return_value = mock_pbar

        callback = RecordingCallback()
        trainer = Trainer(training_method, opt_manager, callbacks=[callback])
        trainer.train(DummyTrainLoader())

        assert len(callback.train_begin_calls) == 1
        assert len(callback.train_step_calls) == 3
        assert len(callback.train_end_calls) == 1

    @patch("sckitflow.trainer._trainer.tqdm")
    def test_train_with_validation(self, mock_tqdm, training_method, inference_method, opt_manager):
        mock_tqdm.side_effect = lambda steps: steps

        callback = RecordingCallback()
        trainer = Trainer(
            training_method,
            opt_manager,
            inference_method=inference_method,
            callbacks=[callback],
        )
        # 5 steps -> validate at 2, 4.
        trainer.train(
            DummyTrainLoader(5),
            val_loaders={"val1": DummyValLoader()},
            valid_freq=2,
        )

        assert [call[1] for call in callback.valid_step_calls] == [2, 4]

    @patch("sckitflow.trainer._trainer.tqdm")
    def test_train_without_inference_method_skips_validation(self, mock_tqdm, training_method, opt_manager):
        """Even with val_loaders, no inference method means no validation metrics."""
        mock_tqdm.side_effect = lambda steps: steps

        callback = RecordingCallback()
        trainer = Trainer(training_method, opt_manager, callbacks=[callback])
        trainer.train(
            DummyTrainLoader(3),
            val_loaders={"val1": DummyValLoader()},
            valid_freq=1,
        )
        # No metrics from a val run: the log stays empty.
        assert trainer.val_logs_raw == {}
        assert callback.valid_step_calls == []

    @patch("sckitflow.trainer._trainer.tqdm")
    def test_train_continues_from_current_step(self, mock_tqdm, training_method, inference_method, opt_manager):
        mock_tqdm.side_effect = lambda steps: steps

        callback = RecordingCallback()
        trainer = Trainer(
            training_method,
            opt_manager,
            inference_method=inference_method,
            callbacks=[RecordingComputationalCallback(), callback],
        )
        loader = DummyTrainLoader(3)  # 3 steps per call; two calls -> 6
        val_loaders = {"val1": DummyValLoader()}

        trainer.train(loader, val_loaders=val_loaders, valid_freq=2)
        trainer.train(loader, val_loaders=val_loaders, valid_freq=2)

        assert trainer.current_step == 6
        assert [call[1] for call in callback.train_step_calls] == [1, 2, 3, 4, 5, 6]
        assert list(trainer.get_val_logs_df("val1").index) == [2, 4, 6]

    # ---- Properties -------------------------------------------------------
    def test_properties(self, training_method, inference_method, opt_manager):
        trainer = Trainer(
            training_method,
            opt_manager,
            inference_method=inference_method,
        )
        assert trainer.training_method is training_method
        assert trainer.inference_method is inference_method
        assert trainer.opt_manager is opt_manager
        assert trainer.train_logs_raw == []
        assert trainer.val_logs_raw == {}

    # ---- Storage delegation ----------------------------------------------
    def test_training_method_delegates_storage_to_specs(self, training_method, module):
        """The training method exposes the shared specs' storage surface."""
        assert training_method.specs.module is module
        assert training_method.module is module
        assert training_method.device_id == "cpu"
        assert training_method.dtype == torch.float32

    def test_inference_method_delegates_storage_to_specs(self, inference_method, module):
        """The inference method exposes the shared specs' storage surface."""
        assert inference_method.specs.module is module
        assert inference_method.module is module
        assert inference_method.device_id == "cpu"
        assert inference_method.dtype == torch.float32

    def test_train_and_inference_methods_share_specs(self, training_method, inference_method):
        """The training and inference methods are wired to the same specs instance."""
        assert training_method.specs is inference_method.specs
        assert training_method.module is inference_method.module


class TestTrainerStructuralContracts:
    """`Trainer` accepts anything satisfying the structural method shapes."""

    def test_trainer_accepts_matched_training_method(self, specs, opt_manager):
        """The structural refactor: a `MatchedTrainingMethod` is a valid training method.

        `MatchedTrainingMethod` is *not* a subclass of `BaseTrainingProtocol` — they
        are siblings under `_AbstractTrainingProtocol` — so this only works because the
        `Trainer` parameter is typed `SupportsTraining`.
        """
        inner = DummyTrainingProtocol(specs)
        matched = MatchedTrainingMethod(inner, match_fn=dummy_match_fn)
        # Precondition: not a nominal subclass.
        assert not isinstance(matched, BaseTrainingProtocol)
        # The check `SupportsTraining` is what makes it acceptable.
        assert isinstance(matched, SupportsTraining)

        trainer = Trainer(matched, opt_manager)
        assert trainer.training_method is matched

    def test_matched_protocol_forwards_storage_to_specs(self, specs, opt_manager):
        """The matched wrapper exposes the shared specs' storage surface."""
        inner = DummyTrainingProtocol(specs)
        matched = MatchedTrainingMethod(inner, match_fn=dummy_match_fn)
        trainer = Trainer(matched, opt_manager)

        assert trainer.training_method.module is specs.module
        assert trainer.training_method.device_id == "cpu"
        assert trainer.training_method.dtype == torch.float32

    def test_training_method_property_satisfies_structural_contract(self, training_method, opt_manager):
        trainer = Trainer(training_method, opt_manager)
        assert isinstance(trainer.training_method, SupportsTraining)
        assert isinstance(trainer.training_method, SupportsProtocol)

    def test_inference_method_property_satisfies_structural_contract(
        self, training_method, inference_method, opt_manager
    ):
        trainer = Trainer(training_method, opt_manager, inference_method=inference_method)
        assert isinstance(trainer.inference_method, SupportsInference)
        assert isinstance(trainer.inference_method, SupportsProtocol)

    def test_training_method_does_not_satisfy_inference_contract(self, training_method, opt_manager):
        """A training method has no `predict`, so it is not usable as an inference method."""
        trainer = Trainer(training_method, opt_manager)
        assert not isinstance(trainer.training_method, SupportsInference)

    def test_inference_method_does_not_satisfy_training_contract(self, training_method, inference_method, opt_manager):
        """An inference method has no `compute_loss`, so it is not usable as a training method."""
        trainer = Trainer(training_method, opt_manager, inference_method=inference_method)
        assert not isinstance(trainer.inference_method, SupportsTraining)

    @patch("sckitflow.trainer._trainer.tqdm")
    def test_matched_protocol_train_loop(self, mock_tqdm, specs, opt_manager):
        """The training loop runs end-to-end with a matched training method."""
        mock_tqdm.side_effect = lambda steps: steps

        inner = DummyTrainingProtocol(specs)
        matched = MatchedTrainingMethod(inner, match_fn=dummy_match_fn)
        trainer = Trainer(matched, opt_manager)

        trainer.train(DummyTrainLoader(3))

        assert trainer.current_step == 3
        assert len(trainer.train_logs_raw) == 3
