"""Tests for sckitflow._predict: chunking, reassembly, and the helpers that connect them."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("anndata")

import sckitflow._predict as predict_module
from sckitflow._predict import (
    _chunks_for,
    _concat_chunks,
    _get_pred_obsm_dict,
    _get_pred_raw_samples,
    _get_pred_traj,
    _group_size,
    _iter_chunks,
    _obs_dim_for_field,
    _pred_obs_from_leaf,
    _predict_empty,
    _slice_step_data,
    _to_numpy,
    predict_adata,
    prediction_record,
    predictions_to_adata,
)

# ============================================================================
# Fakes
# ============================================================================


@dataclasses.dataclass(frozen=True)
class FakePredictionData:
    """A drop-in for sckitflow's PredictionData with the same three fields used here."""

    X: torch.Tensor
    raw_samples: torch.Tensor | None = None
    traj: torch.Tensor | None = None


def make_step_data(n_obs: int, n_features: int = 4, n_cond: int = 2, with_source: bool = True) -> dict:
    state = torch.arange(n_obs * n_features, dtype=torch.float32).reshape(n_obs, n_features)
    return {
        "target_state": state,
        "source_state": state + 1 if with_source else None,
        "target_condition_data": {"cond": torch.arange(n_obs * n_cond, dtype=torch.float32).reshape(n_obs, n_cond)},
        "target_response_data": {},
    }


class FakeLoader:
    def __init__(
        self,
        items,
        cond_cont_keys: tuple[str, ...] = (),
        resp_keys: tuple[str, ...] = (),
        group_cols: tuple[str, ...] = (),
    ):
        self.items = items
        self.cond_cont_keys = cond_cont_keys
        self.resp_keys = resp_keys
        self.group_cols = group_cols

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)


class FakeInference:
    """Records every predict() call and returns a simple PredictionData."""

    def __init__(self, n_features: int = 4, with_traj: bool = False, n_samples: int = 1):
        self.module = torch.nn.Linear(1, 1)
        self.module.train(False)
        self.n_features = n_features
        self.with_traj = with_traj
        self.n_samples = n_samples
        self.calls: list[dict] = []

    def predict(self, step_data, generator, **kwargs):
        n_obs = step_data["target_state"].shape[0]
        self.calls.append({"n_obs": n_obs})
        X = step_data["target_state"].clone()
        raw_samples = None
        if self.n_samples > 1:
            raw_samples = X.unsqueeze(0).expand(self.n_samples, -1, -1).clone()
        traj = X.unsqueeze(1) if self.with_traj else None
        return FakePredictionData(X=X, raw_samples=raw_samples, traj=traj)


class ReiterableLoader:
    """Loader that returns a fresh iterator on every `iter()`.

    Mimics the real `get_eval_loader` -- `progress_total=True` relies on this
    to walk the loader once for the shape-only pre-count and then again for
    the real pass.
    """

    def __init__(self, items, cond_cont_keys=(), resp_keys=(), group_cols=()):
        self.items = items
        self.cond_cont_keys = cond_cont_keys
        self.resp_keys = resp_keys
        self.group_cols = group_cols

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)


class OneShotLoader(Iterator):
    """Single-use loader that is *also* a full loader (carries the column names).

    Models the contract the pre-count must respect: ``iter(loader) is loader``,
    so a count walk would consume it. Subclasses ``Iterator`` so the source's
    ``isinstance(loader, Iterator)`` guard detects it, while still exposing
    ``group_cols`` / ``cond_cont_keys`` / ``resp_keys`` for ``prediction_record``.
    """

    def __init__(self, items, cond_cont_keys=(), resp_keys=(), group_cols=()):
        self._iter = iter(items)
        self.cond_cont_keys = cond_cont_keys
        self.resp_keys = resp_keys
        self.group_cols = group_cols

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)


def fake_datamodule(loader, feature_names, *, per_call=False):
    """``per_call=True``: each `get_eval_loader()` call returns a fresh loader.

    Needed for `progress_total=True`, where the pre-count walks one loader and
    the real pass walks a second one. With ``per_call=False``, both passes see
    the *same* loader object; if it's a one-shot iterator, the real pass would
    see nothing.
    """
    dm = MagicMock()
    dm.dtype = torch.float32
    dm.data_dims = MagicMock(feature_names=list(feature_names))
    if per_call:
        dm.dm.get_eval_loader.side_effect = lambda *a, **kw: loader
    else:
        dm.dm.get_eval_loader.return_value = loader
    return dm


# ============================================================================
# _to_numpy
# ============================================================================


class TestToNumpy:
    def test_tensor(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        out = _to_numpy(t)
        np.testing.assert_array_equal(out, np.array([1.0, 2.0, 3.0]))

    def test_none(self):
        assert _to_numpy(None) is None

    def test_array_like(self):
        out = _to_numpy([1, 2, 3])
        np.testing.assert_array_equal(out, np.array([1, 2, 3]))


# ============================================================================
# _predict_empty
# ============================================================================


class TestPredictEmpty:
    def test_shape_and_var(self):
        dims = MagicMock(feature_names=["a", "b", "c"])
        adata = _predict_empty(dims, return_raw=False)
        assert adata.shape == (0, 3)
        assert list(adata.var_names) == ["a", "b", "c"]

    def test_return_raw(self):
        dims = MagicMock(feature_names=["a"])
        adata, raw = _predict_empty(dims, return_raw=True)
        assert adata.shape == (0, 1)
        assert raw is None


# ============================================================================
# _pred_obs_from_leaf
# ============================================================================


class TestPredObsFromLeaf:
    def test_single_row(self):
        pred = FakePredictionData(X=torch.zeros(4, 3))
        df = _pred_obs_from_leaf(("g",), ("ctrl",), pred)
        assert len(df) == 4
        assert list(df["g"]) == ["ctrl"] * 4

    def test_no_group_cols(self):
        pred = FakePredictionData(X=torch.zeros(2, 3))
        df = _pred_obs_from_leaf((), (), pred)
        assert len(df) == 2
        assert list(df.columns) == []

    def test_multiple_group_cols(self):
        pred = FakePredictionData(X=torch.zeros(2, 3))
        df = _pred_obs_from_leaf(("a", "b"), ("x", "y"), pred)
        assert list(df["a"]) == ["x", "x"]
        assert list(df["b"]) == ["y", "y"]


# ============================================================================
# _get_pred_traj / _get_pred_raw_samples
# ============================================================================


class TestGetPredTraj:
    def test_none(self):
        assert _get_pred_traj(FakePredictionData(X=torch.zeros(3, 2))) is None

    def test_2d_obs_first(self):
        pred = FakePredictionData(X=torch.zeros(3, 2), traj=torch.zeros(3, 5))
        out = _get_pred_traj(pred)
        assert out.shape == (3, 5)

    def test_3d_time_first(self):
        pred = FakePredictionData(X=torch.zeros(3, 2), traj=torch.zeros(5, 3, 2))
        out = _get_pred_traj(pred)
        assert out.shape == (3, 5, 2)

    def test_4d_iter_time_obs(self):
        pred = FakePredictionData(X=torch.zeros(3, 2), traj=torch.zeros(4, 5, 3, 2))
        out = _get_pred_traj(pred)
        assert out.shape == (3, 4, 5, 2)

    def test_incompatible_shape_raises(self):
        pred = FakePredictionData(X=torch.zeros(3, 2), traj=torch.zeros(7, 8, 9))
        with pytest.raises(ValueError, match="incompatible shape"):
            _get_pred_traj(pred)


class TestGetPredRawSamples:
    def test_none(self):
        pred = FakePredictionData(X=torch.zeros(3, 2))
        assert _get_pred_raw_samples(pred) is None

    def test_2d(self):
        pred = FakePredictionData(X=torch.zeros(3, 2), raw_samples=torch.zeros(3, 2))
        out = _get_pred_raw_samples(pred)
        assert out.shape == (3, 2)

    def test_3d_samples_first(self):
        # (n_samples, n_obs, n_features) -> transposed to (n_obs, n_samples, n_features)
        pred = FakePredictionData(X=torch.zeros(3, 2), raw_samples=torch.zeros(10, 3, 2))
        out = _get_pred_raw_samples(pred)
        assert out.shape == (3, 10, 2)

    def test_incompatible_shape_raises(self):
        pred = FakePredictionData(X=torch.zeros(3, 2), raw_samples=torch.zeros(7, 8, 9))
        with pytest.raises(ValueError, match="incompatible shape"):
            _get_pred_raw_samples(pred)


# ============================================================================
# _get_pred_obsm_dict
# ============================================================================


class TestGetPredObsmDict:
    def test_empty(self):
        pred = FakePredictionData(X=torch.zeros(2, 3))
        out = _get_pred_obsm_dict(
            {"target_condition_data": None, "target_response_data": None},
            pred,
            cont_keys=(),
        )
        assert out == {}

    def test_traj_and_samples(self):
        # raw_samples must be (n_samples, n_obs, n_features):
        # 5 samples, 2 obs, 3 features -> transposes to (2, 5, 3).
        pred = FakePredictionData(
            X=torch.zeros(2, 3),
            traj=torch.zeros(2, 4),
            raw_samples=torch.zeros(5, 2, 3),
        )
        out = _get_pred_obsm_dict(
            {"target_condition_data": None, "target_response_data": None},
            pred,
            cont_keys=(),
        )
        assert "trajectory" in out
        assert "raw_samples" in out
        assert out["raw_samples"].shape == (2, 5, 3)

    def test_cont_keys(self):
        pred = FakePredictionData(X=torch.zeros(2, 3))
        step_data = {
            "target_condition_data": {"x": torch.ones(2, 4)},
            "target_response_data": {"y": torch.zeros(2, 5)},
        }
        out = _get_pred_obsm_dict(step_data, pred, cont_keys=("x", "y", "missing"))
        assert set(out.keys()) == {"x", "y"}
        assert out["x"].shape == (2, 4)
        assert out["y"].shape == (2, 5)


# ============================================================================
# Chunking helpers
# ============================================================================


class TestGroupSize:
    def test_from_target_state(self):
        assert _group_size(make_step_data(7)) == 7

    def test_from_source_only(self):
        sd = {"target_state": None, "source_state": torch.zeros(11, 2)}
        assert _group_size(sd) == 11

    def test_from_condition_data(self):
        sd = {
            "target_state": None,
            "source_state": None,
            "target_condition_data": {"c": torch.zeros(5, 2)},
        }
        assert _group_size(sd) == 5

    def test_missing(self):
        with pytest.raises(ValueError, match="Cannot infer"):
            _group_size({})


class TestSliceStepData:
    def test_slices_tensors_on_dim0(self):
        sd = {
            "target_state": torch.arange(10).reshape(5, 2).float(),
            "source_state": torch.arange(10).reshape(5, 2).float() + 100,
        }
        out = _slice_step_data(sd, 1, 3)
        assert out["target_state"].shape == (2, 2)
        assert torch.equal(out["target_state"], sd["target_state"][1:3])
        assert torch.equal(out["source_state"], sd["source_state"][1:3])

    def test_passes_none_through(self):
        sd = {"target_state": None, "x": torch.zeros(5, 2)}
        out = _slice_step_data(sd, 0, 2)
        assert out["target_state"] is None
        assert out["x"].shape == (2, 2)

    def test_recurses_into_dicts(self):
        sd = {"target_condition_data": {"c": torch.arange(10).reshape(5, 2).float()}}
        out = _slice_step_data(sd, 2, 4)
        assert out["target_condition_data"]["c"].shape == (2, 2)

    def test_passes_non_tensor_through(self):
        sd = {"meta": "hello", "n": 42}
        out = _slice_step_data(sd, 0, 1)
        assert out["meta"] == "hello"
        assert out["n"] == 42

    def test_does_not_mutate_original(self):
        sd = {"x": torch.zeros(5, 2)}
        original_id = id(sd["x"])
        _slice_step_data(sd, 0, 2)
        assert id(sd["x"]) == original_id
        assert sd["x"].shape == (5, 2)


class TestObsDimForField:
    @pytest.mark.parametrize(
        "field,ndim,expected",
        [
            ("X", 2, 0),
            ("X", 3, 0),
            ("raw_samples", 2, 0),
            ("raw_samples", 3, 1),
            ("traj", 2, 0),
            ("traj", 3, 1),
            ("traj", 4, 2),
            ("unknown_field", 2, 0),
        ],
    )
    def test_mapping(self, field, ndim, expected):
        assert _obs_dim_for_field(field, ndim) == expected


class TestChunksFor:
    def test_none_chunk_size(self):
        assert _chunks_for(50, None) == [(0, 50)]

    def test_group_smaller_than_chunk(self):
        assert _chunks_for(20, 100) == [(0, 20)]

    def test_group_equal_to_chunk(self):
        assert _chunks_for(100, 100) == [(0, 100)]

    def test_exact_split(self):
        assert _chunks_for(300, 100) == [(0, 100), (100, 200), (200, 300)]

    def test_uneven_split(self):
        assert _chunks_for(250, 100) == [(0, 100), (100, 200), (200, 250)]

    def test_zero_obs(self):
        assert _chunks_for(0, 100) == [(0, 0)]


class TestIterChunks:
    def test_single_group_no_chunking(self):
        step_data = make_step_data(7)
        loader = FakeLoader([(step_data, ("g",))])
        chunks = list(_iter_chunks(loader, chunk_size=None))
        assert len(chunks) == 1
        g, c, nc, sd, start, end, leaf = chunks[0]
        assert (g, c, nc, start, end, leaf) == (0, 0, 1, 0, 7, ("g",))
        # no chunking: step_data is not copied
        assert sd is step_data

    def test_single_group_chunked(self):
        step_data = make_step_data(250)
        loader = FakeLoader([(step_data, ("g",))])
        chunks = list(_iter_chunks(loader, chunk_size=100))
        assert [(c, s, e) for _, c, _, _, s, e, _ in chunks] == [
            (0, 0, 100),
            (1, 100, 200),
            (2, 200, 250),
        ]
        assert all(nc == 3 for _, _, nc, _, _, _, _ in chunks)

    def test_multiple_groups(self):
        loader = FakeLoader(
            [
                (make_step_data(250), ("g1",)),
                (make_step_data(120), ("g2",)),
            ]
        )
        chunks = list(_iter_chunks(loader, chunk_size=100))
        # group 0: 3 chunks; group 1: 2 chunks
        assert [g for g, *_ in chunks] == [0, 0, 0, 1, 1]
        assert [(s, e) for _, _, _, _, s, e, _ in chunks] == [
            (0, 100),
            (100, 200),
            (200, 250),
            (0, 100),
            (100, 120),
        ]

    def test_lazy(self):
        """Only one group's step_data is materialised at a time."""
        loader = FakeLoader([(make_step_data(10), ("a",)), (make_step_data(10), ("b",))])
        it = _iter_chunks(loader, chunk_size=None)
        first = next(it)
        # group 0 fully consumed before group 1 is touched
        assert first[0] == 0
        assert first[6] == ("a",)
        second = next(it)
        assert second[0] == 1
        assert second[6] == ("b",)


class TestConcatChunks:
    def test_single_chunk_returns_input(self):
        a = FakePredictionData(X=torch.zeros(3, 4))
        result = _concat_chunks([a], [3])
        assert result is a

    def test_X_concats_on_dim0(self):
        a = FakePredictionData(X=torch.zeros(3, 4))
        b = FakePredictionData(X=torch.ones(2, 4))
        result = _concat_chunks([a, b], [3, 2])
        assert result.X.shape == (5, 4)
        assert torch.all(result.X[:3] == 0)
        assert torch.all(result.X[3:] == 1)

    def test_raw_samples_2d(self):
        a = FakePredictionData(X=torch.zeros(3, 4), raw_samples=torch.zeros(3, 4))
        b = FakePredictionData(X=torch.zeros(2, 4), raw_samples=torch.ones(2, 4))
        result = _concat_chunks([a, b], [3, 2])
        assert result.raw_samples.shape == (5, 4)

    def test_raw_samples_3d_samples_first(self):
        a = FakePredictionData(X=torch.zeros(3, 4), raw_samples=torch.zeros(10, 3, 4))
        b = FakePredictionData(X=torch.zeros(2, 4), raw_samples=torch.ones(10, 2, 4))
        result = _concat_chunks([a, b], [3, 2])
        assert result.raw_samples.shape == (10, 5, 4)

    def test_traj_2d(self):
        a = FakePredictionData(X=torch.zeros(3, 4), traj=torch.zeros(3, 5))
        b = FakePredictionData(X=torch.zeros(2, 4), traj=torch.ones(2, 5))
        result = _concat_chunks([a, b], [3, 2])
        assert result.traj.shape == (5, 5)

    def test_traj_3d_time_first(self):
        a = FakePredictionData(X=torch.zeros(3, 4), traj=torch.zeros(5, 3, 4))
        b = FakePredictionData(X=torch.zeros(2, 4), traj=torch.ones(5, 2, 4))
        result = _concat_chunks([a, b], [3, 2])
        assert result.traj.shape == (5, 5, 4)

    def test_traj_4d_iter_time_obs(self):
        a = FakePredictionData(X=torch.zeros(3, 4), traj=torch.zeros(2, 5, 3, 4))
        b = FakePredictionData(X=torch.zeros(2, 4), traj=torch.ones(2, 5, 2, 4))
        result = _concat_chunks([a, b], [3, 2])
        assert result.traj.shape == (2, 5, 5, 4)

    def test_three_chunks(self):
        a = FakePredictionData(X=torch.zeros(3, 4))
        b = FakePredictionData(X=torch.ones(2, 4))
        c = FakePredictionData(X=torch.full((1, 4), 2.0))
        result = _concat_chunks([a, b, c], [3, 2, 1])
        assert result.X.shape == (6, 4)

    def test_raises_on_size_mismatch(self):
        a = FakePredictionData(X=torch.zeros(3, 4))
        b = FakePredictionData(X=torch.zeros(2, 4))
        with pytest.raises(ValueError, match="expected obs dim"):
            _concat_chunks([a, b], [5, 2])

    def test_raises_on_num_preds_vs_sizes_mismatch(self):
        a = FakePredictionData(X=torch.zeros(3, 4))
        b = FakePredictionData(X=torch.zeros(2, 4))
        with pytest.raises(ValueError, match="2 predictions but 3 chunk sizes"):
            _concat_chunks([a, b], [3, 2, 1])

    def test_raises_on_partial_none(self):
        a = FakePredictionData(X=torch.zeros(3, 4), raw_samples=None)
        b = FakePredictionData(X=torch.zeros(2, 4), raw_samples=torch.zeros(2, 4))
        with pytest.raises(ValueError, match="present in some chunks"):
            _concat_chunks([a, b], [3, 2])


# ============================================================================
# prediction_record + predictions_to_adata
# ============================================================================


class TestPredictionRecord:
    def test_shape_of_record(self):
        step_data = make_step_data(3)
        pred = FakePredictionData(X=torch.zeros(3, 4))
        loader = FakeLoader(
            items=[],
            cond_cont_keys=("cond",),
            resp_keys=(),
            group_cols=("g",),
        )
        record = prediction_record(loader, step_data, ("ctrl",), pred)
        assert record["preds_cls"] is FakePredictionData
        assert set(record["preds_fields"].keys()) == {"X", "raw_samples", "traj"}
        assert list(record["obs"]["g"]) == ["ctrl"] * 3
        assert "cond" in record["obsm"]


class TestPredictionsToAdata:
    def test_empty_records(self):
        dims = MagicMock(feature_names=["a", "b"])
        adata = predictions_to_adata(dims, records=[])
        assert adata.shape == (0, 2)

    def test_single_record(self):
        dims = MagicMock(feature_names=["a", "b", "c", "d"])
        record = {
            "preds_cls": FakePredictionData,
            "preds_fields": {"X": torch.zeros(3, 4), "raw_samples": None, "traj": None},
            "obs": pd.DataFrame({"g": ["x"] * 3}, index=pd.RangeIndex(3)),
            "obsm": {},
        }
        with patch.object(
            predict_module,
            "concatenate_predictions",
            side_effect=lambda preds: preds[0],
        ):
            adata = predictions_to_adata(dims, records=[record])
        assert adata.shape == (3, 4)
        assert list(adata.var_names) == ["a", "b", "c", "d"]


# ============================================================================
# predict_adata: the main entry point
# ============================================================================


class TestPredictAdataNoChunking:
    def test_single_group_single_call(self):
        step_data = make_step_data(20)
        loader = FakeLoader([(step_data, ("g",))])
        dm = fake_datamodule(loader, ["f0", "f1", "f2", "f3"])
        inf = FakeInference(n_features=4)

        with patch.object(predict_module, "predictions_to_adata", return_value=MagicMock()):
            predict_adata(dm, inf, MagicMock(), chunk_size=None, progress_total=False)

        assert [c["n_obs"] for c in inf.calls] == [20]

    def test_multiple_groups(self):
        loader = FakeLoader(
            [
                (make_step_data(5), ("g1",)),
                (make_step_data(7), ("g2",)),
                (make_step_data(3), ("g3",)),
            ]
        )
        dm = fake_datamodule(loader, ["f0", "f1", "f2", "f3"])
        inf = FakeInference(n_features=4)

        with patch.object(predict_module, "predictions_to_adata", return_value=MagicMock()):
            predict_adata(dm, inf, MagicMock(), chunk_size=None, progress_total=False)

        assert [c["n_obs"] for c in inf.calls] == [5, 7, 3]


class TestPredictAdataChunking:
    def test_even_split(self):
        step_data = make_step_data(200)
        loader = FakeLoader([(step_data, ("g",))])
        dm = fake_datamodule(loader, ["f0", "f1", "f2", "f3"])
        inf = FakeInference(n_features=4)

        with patch.object(predict_module, "predictions_to_adata", return_value=MagicMock()):
            predict_adata(dm, inf, MagicMock(), chunk_size=100, progress_total=False)

        assert [c["n_obs"] for c in inf.calls] == [100, 100]

    def test_uneven_split(self):
        step_data = make_step_data(250)
        loader = FakeLoader([(step_data, ("g",))])
        dm = fake_datamodule(loader, ["f0", "f1", "f2", "f3"])
        inf = FakeInference(n_features=4)

        with patch.object(predict_module, "predictions_to_adata", return_value=MagicMock()):
            predict_adata(dm, inf, MagicMock(), chunk_size=100, progress_total=False)

        assert [c["n_obs"] for c in inf.calls] == [100, 100, 50]

    def test_chunk_size_equals_n_obs(self):
        step_data = make_step_data(50)
        loader = FakeLoader([(step_data, ("g",))])
        dm = fake_datamodule(loader, ["f0", "f1", "f2", "f3"])
        inf = FakeInference(n_features=4)

        with patch.object(predict_module, "predictions_to_adata", return_value=MagicMock()):
            predict_adata(dm, inf, MagicMock(), chunk_size=50, progress_total=False)

        assert [c["n_obs"] for c in inf.calls] == [50]

    def test_chunk_size_larger_than_n_obs(self):
        step_data = make_step_data(20)
        loader = FakeLoader([(step_data, ("g",))])
        dm = fake_datamodule(loader, ["f0", "f1", "f2", "f3"])
        inf = FakeInference(n_features=4)

        with patch.object(predict_module, "predictions_to_adata", return_value=MagicMock()):
            predict_adata(dm, inf, MagicMock(), chunk_size=1000, progress_total=False)

        assert [c["n_obs"] for c in inf.calls] == [20]

    def test_multiple_groups_each_chunked(self):
        loader = FakeLoader(
            [
                (make_step_data(250), ("g1",)),
                (make_step_data(120), ("g2",)),
            ]
        )
        dm = fake_datamodule(loader, ["f0", "f1", "f2", "f3"])
        inf = FakeInference(n_features=4)

        with patch.object(predict_module, "predictions_to_adata", return_value=MagicMock()):
            predict_adata(dm, inf, MagicMock(), chunk_size=100, progress_total=False)

        # group 1: 100, 100, 50; group 2: 100, 20
        assert [c["n_obs"] for c in inf.calls] == [100, 100, 50, 100, 20]

    def test_chunking_preserves_total_obs_via_record(self):
        """Every input row is accounted for exactly once in the reassembled prediction."""
        n_obs = 250
        step_data = make_step_data(n_obs)
        loader = FakeLoader([(step_data, ("g",))])
        dm = fake_datamodule(loader, ["f0", "f1", "f2", "f3"])
        inf = FakeInference(n_features=4)

        captured: list[dict] = []

        def fake_record(loader, step_data, leaf, preds):
            captured.append({"leaf": leaf, "X_shape": tuple(preds.X.shape)})
            return {
                "preds_cls": FakePredictionData,
                "preds_fields": {"X": preds.X},
                "obs": None,
                "obsm": {},
            }

        with (
            patch.object(predict_module, "prediction_record", side_effect=fake_record),
            patch.object(predict_module, "predictions_to_adata", return_value=MagicMock()),
        ):
            predict_adata(dm, inf, MagicMock(), chunk_size=100, progress_total=False)

        assert len(captured) == 1
        assert captured[0]["X_shape"] == (n_obs, 4)

    def test_chunked_prediction_values_reassemble(self):
        """The chunked output is the concatenation of the per-chunk X tensors, in order."""
        n_obs = 250
        step_data = make_step_data(n_obs)
        # Fill target_state with row index so we can verify order after reassembly.
        step_data["target_state"] = torch.arange(n_obs, dtype=torch.float32).unsqueeze(1).expand(n_obs, 4).clone()
        loader = FakeLoader([(step_data, ("g",))])
        dm = fake_datamodule(loader, ["f0", "f1", "f2", "f3"])
        inf = FakeInference(n_features=4)

        captured: list[torch.Tensor] = []

        def fake_record(loader, step_data, leaf, preds):
            captured.append(preds.X)
            return {
                "preds_cls": FakePredictionData,
                "preds_fields": {"X": preds.X},
                "obs": None,
                "obsm": {},
            }

        with (
            patch.object(predict_module, "prediction_record", side_effect=fake_record),
            patch.object(predict_module, "predictions_to_adata", return_value=MagicMock()),
        ):
            predict_adata(dm, inf, MagicMock(), chunk_size=100, progress_total=False)

        reassembled = captured[0]
        expected = torch.arange(n_obs, dtype=torch.float32).unsqueeze(1).expand(n_obs, 4)
        assert torch.allclose(reassembled, expected)

    def test_zero_chunk_size_raises(self):
        dm = fake_datamodule(FakeLoader([]), ["f0"])
        inf = FakeInference()
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            predict_adata(dm, inf, MagicMock(), chunk_size=0)

    def test_negative_chunk_size_raises(self):
        dm = fake_datamodule(FakeLoader([]), ["f0"])
        inf = FakeInference()
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            predict_adata(dm, inf, MagicMock(), chunk_size=-1)

    def test_empty_loader(self):
        loader = FakeLoader([])
        dm = fake_datamodule(loader, ["f0", "f1"])
        inf = FakeInference(n_features=2)

        with patch.object(predict_module, "predictions_to_adata", return_value=MagicMock()) as mock_pta:
            predict_adata(dm, inf, MagicMock(), chunk_size=100, progress_total=False)

        assert inf.calls == []
        mock_pta.assert_called_once()
        # empty records list
        assert mock_pta.call_args.args[1] == []

    def test_module_mode_restored(self):
        """`predict_adata` restores the original training mode of the module on exit."""
        loader = FakeLoader([(make_step_data(10), ("g",))])
        dm = fake_datamodule(loader, ["f0", "f1"])
        inf = FakeInference(n_features=2)
        inf.module.train(True)
        assert inf.module.training is True

        with patch.object(predict_module, "predictions_to_adata", return_value=MagicMock()):
            predict_adata(dm, inf, MagicMock(), chunk_size=None, progress_total=False)

        assert inf.module.training is True

    def test_module_mode_restored_on_exception(self):
        """If `predict` raises, the module mode is still restored before the error propagates."""
        loader = FakeLoader([(make_step_data(10), ("g",))])
        dm = fake_datamodule(loader, ["f0", "f1"])

        class RaisingInference(FakeInference):
            def predict(self, step_data, generator, **kwargs):
                raise RuntimeError("boom")

        inf = RaisingInference(n_features=2)
        inf.module.train(True)

        with pytest.raises(RuntimeError, match="boom"):
            predict_adata(dm, inf, MagicMock(), chunk_size=100, progress_total=False)

        assert inf.module.training is True


class TestPredictAdataProgress:
    def test_progress_total_precounts_chunks(self):
        """With progress_total=True and a re-iterable loader, both the count and the real pass run."""
        loader = ReiterableLoader([(make_step_data(250), ("g",))])
        dm = fake_datamodule(loader, ["f0", "f1", "f2", "f3"], per_call=True)
        inf = FakeInference(n_features=4)

        with patch.object(predict_module, "predictions_to_adata", return_value=MagicMock()):
            predict_adata(dm, inf, MagicMock(), chunk_size=100, progress_total=True)

        # Real pass: 250 obs split into 100/100/50
        assert [c["n_obs"] for c in inf.calls] == [100, 100, 50]

    def test_progress_total_false_still_runs(self):
        loader = FakeLoader([(make_step_data(250), ("g",))])
        dm = fake_datamodule(loader, ["f0", "f1", "f2", "f3"])
        inf = FakeInference(n_features=4)

        with patch.object(predict_module, "predictions_to_adata", return_value=MagicMock()):
            predict_adata(dm, inf, MagicMock(), chunk_size=100, progress_total=False)

        assert [c["n_obs"] for c in inf.calls] == [100, 100, 50]

    def test_progress_total_skips_precount_for_one_shot_loader(self):
        """A one-shot iterator can't be walked twice, so the pre-count is skipped and prediction still runs."""
        step_data = make_step_data(50)
        loader = OneShotLoader([(step_data, ("g",))])
        dm = fake_datamodule(loader, ["f0", "f1", "f2", "f3"])
        inf = FakeInference(n_features=4)

        with patch.object(predict_module, "predictions_to_adata", return_value=MagicMock()):
            predict_adata(dm, inf, MagicMock(), chunk_size=100, progress_total=True)

        assert [c["n_obs"] for c in inf.calls] == [50]
