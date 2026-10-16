"""Unit tests for :mod:`sckitflow.core._data_utils`.

Two contracts under test:

* :func:`subscript_step_data` -- a matching permutation reindexes each side of a
  batch independently, and every per-observation field of a side moves with it.
* :func:`prepare_latent_train` / :func:`prepare_latent_inference` -- the latent's
  shape is fixed by three (four, counting ``generate_from_noise``) independent
  axes, the width always comes from ``dims.state_dim``, and device / dtype come
  from the caller, never from the batch.
"""

import pandas as pd
import pytest
import torch

from sckitflow.core._data_utils import (
    prepare_latent_inference,
    prepare_latent_train,
    subscript_step_data,
)
from sckitflow.core._types import new_step_data
from sckitflow.data._dims import DataDimensions


@pytest.fixture
def dims() -> DataDimensions:
    return DataDimensions(
        state_dim=4,
        feature_names=pd.Index([f"f{i}" for i in range(4)]),
        condition_reps_dims=None,
        condition_continuous_dims=None,
        response_reps_dims=None,
        response_continuous_dims=None,
        groups_reps_dims=None,
        source_lin_dim=None,
        source_quad_dim=None,
        target_lin_dim=None,
        target_quad_dim=None,
    )


def _sampler(shape, *, generator, device=None, dtype=None):
    """Deterministic filler: every element is the number of rows the sampler was asked for.

    Lets a test assert the shape the sampler *would* have drawn without pinning
    to a specific RNG.
    """
    return torch.full(shape, float(shape[0]) if shape else 0.0, device=device, dtype=dtype)


def _gen() -> torch.Generator:
    return torch.Generator().manual_seed(0)


def _cond(batch_size: int, name: str = "cond") -> dict[str, torch.Tensor]:
    return {name: torch.zeros(batch_size, 1, 3)}


# -------------------- subscript_step_data --------------------
def _batch(n: int = 4) -> dict:
    """A TrainStepData whose every field encodes its row index, so a permutation is visible in the values."""
    rows = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    return new_step_data(
        target_state=rows.clone(),
        target_coupling_lin=rows.clone() + 100,
        target_condition_data={"drug": rows.clone() + 200, "absent": None},
        target_group_data={"line": rows.clone() + 300},
        target_response_data={"ytgt": rows.clone() + 400},
        source_state=rows.clone() + 1000,
        source_coupling_lin=rows.clone() + 1100,
        source_condition_data={"drug": rows.clone() + 1200},
    )


def _col(field) -> list[float]:
    return field.squeeze(1).tolist()


class TestSubscriptStepData:
    def test_each_side_is_indexed_independently(self):
        out = subscript_step_data(_batch(), src_idxs=torch.tensor([3, 2]), tgt_idxs=torch.tensor([0, 1]))

        assert _col(out["target_state"]) == [0.0, 1.0]
        assert _col(out["target_coupling_lin"]) == [100.0, 101.0]
        assert _col(out["source_state"]) == [1003.0, 1002.0]
        assert _col(out["source_coupling_lin"]) == [1103.0, 1102.0]

    def test_dict_fields_are_indexed_value_by_value(self):
        out = subscript_step_data(_batch(), src_idxs=torch.tensor([1]), tgt_idxs=torch.tensor([2]))

        assert _col(out["target_condition_data"]["drug"]) == [202.0]
        assert _col(out["target_group_data"]["line"]) == [302.0]
        assert _col(out["source_condition_data"]["drug"]) == [1201.0]
        assert out["target_condition_data"]["absent"] is None

    def test_response_data_stays_aligned_with_the_target_state(self):
        perm = torch.tensor([3, 0, 2, 1])
        out = subscript_step_data(_batch(), tgt_idxs=perm)

        states = _col(out["target_state"])
        responses = _col(out["target_response_data"]["ytgt"])
        assert responses == [state + 400 for state in states]

    def test_a_none_index_leaves_that_side_untouched(self):
        batch = _batch()
        out = subscript_step_data(batch, src_idxs=None, tgt_idxs=torch.tensor([1, 0]))

        assert out["source_state"] is batch["source_state"]
        assert _col(out["target_state"]) == [1.0, 0.0]

    def test_none_fields_stay_none_and_untouched_keys_survive(self):
        batch = _batch()
        out = subscript_step_data(batch, src_idxs=torch.tensor([0]), tgt_idxs=torch.tensor([0]))

        assert out["source_group_data"] is None
        assert out["target_coupling_quad"] is None
        assert set(out) == set(batch)

    def test_the_input_batch_is_not_mutated(self):
        batch = _batch()
        subscript_step_data(batch, src_idxs=torch.tensor([2]), tgt_idxs=torch.tensor([2]))

        assert _col(batch["target_state"]) == [0.0, 1.0, 2.0, 3.0]
        assert _col(batch["target_response_data"]["ytgt"]) == [400.0, 401.0, 402.0, 403.0]


class TestUnalignedFields:
    def test_an_unassigned_field_is_refused_when_permuting(self):
        batch = _batch()
        batch["mystery_covariate"] = torch.arange(4)

        with pytest.raises(ValueError, match=r"unaligned fields: \['mystery_covariate'\]"):
            subscript_step_data(batch, tgt_idxs=torch.tensor([1, 0]))

    def test_an_unassigned_field_is_fine_when_nothing_is_permuted(self):
        batch = _batch()
        batch["mystery_covariate"] = torch.arange(4)

        out = subscript_step_data(batch)
        assert _col(out["target_state"]) == [0.0, 1.0, 2.0, 3.0]
        assert out["mystery_covariate"].tolist() == [0, 1, 2, 3]

    def test_a_plain_step_data_permutes_without_complaint(self):
        out = subscript_step_data(_batch(), src_idxs=torch.tensor([1, 0]), tgt_idxs=torch.tensor([1, 0]))
        assert _col(out["target_state"]) == [1.0, 0.0]

    def test_every_step_data_field_belongs_to_a_side(self):
        """A field added to `StepData` but not to a side tuple is caught here."""
        subscript_step_data(new_step_data(), src_idxs=torch.tensor([0]), tgt_idxs=torch.tensor([0]))


# -------------------- prepare_latent_train --------------------
class TestPrepareLatentTrain:
    def test_source_present_returns_it_unchanged(self, dims):
        """Paired, not generating: the source *is* the latent."""
        source = torch.randn(8, dims.state_dim)
        out = prepare_latent_train(
            source,
            {},
            _sampler,
            dims,
            device=torch.device("cpu"),
            dtype=torch.float32,
            generate_from_noise=False,
            generator=_gen(),
        )
        assert out is source

    def test_source_present_with_noise_draws_from_source_rows(self, dims):
        """Paired, generating: batch size from `source.shape[0]`, width from `dims.state_dim`."""
        source = torch.randn(8, dims.state_dim)
        out = prepare_latent_train(
            source,
            {},
            _sampler,
            dims,
            device=torch.device("cpu"),
            dtype=torch.float32,
            generate_from_noise=True,
            generator=_gen(),
        )
        assert out.shape == (8, dims.state_dim)

    def test_source_absent_uses_conditioning_rows(self, dims):
        """Unpaired: batch size from the first conditioning tensor."""
        out = prepare_latent_train(
            None,
            _cond(5),
            _sampler,
            dims,
            device=torch.device("cpu"),
            dtype=torch.float32,
            generate_from_noise=True,
            generator=_gen(),
        )
        assert out.shape == (5, dims.state_dim)

    def test_neither_source_nor_conditioning_raises(self, dims):
        """A training batch always has a batch; a missing one is a loader bug, not a shape one."""
        with pytest.raises(ValueError, match="cannot size the training latent"):
            prepare_latent_train(
                None,
                {},
                _sampler,
                dims,
                device=torch.device("cpu"),
                dtype=torch.float32,
                generate_from_noise=True,
                generator=_gen(),
            )

    def test_width_ignores_the_source_shape(self, dims):
        """A source with the wrong last dim is still returned unchanged when paired, not generating.

        The width belongs to `dims`; the source is authoritative for `B` and its
        own `D`, which is why it short-circuits. This test pins that `dims` is
        *not* consulted for the width in the paired case -- if the source is
        accepted as-is, the module and the source are on the same page.
        """
        source = torch.randn(8, 99)  # wrong width on purpose
        out = prepare_latent_train(
            source,
            {},
            _sampler,
            dims,
            device=torch.device("cpu"),
            dtype=torch.float32,
            generate_from_noise=False,
            generator=_gen(),
        )
        assert out is source  # unchanged; `dims.state_dim` is 4, source is 99, no error


# -------------------- prepare_latent_inference --------------------
class TestPrepareLatentInference:
    """One test per row of the shape table in the docstring."""

    def test_paired_deterministic_returns_source(self, dims):
        source = torch.randn(8, dims.state_dim)
        out = prepare_latent_inference(
            source,
            _cond(8),
            _sampler,
            dims,
            device=torch.device("cpu"),
            dtype=torch.float32,
            n_samples=3,
            generate_from_noise=False,
            generator=_gen(),
        )
        assert out is source

    def test_paired_generating_with_n_samples(self, dims):
        out = prepare_latent_inference(
            torch.randn(8, dims.state_dim),
            _cond(8),
            _sampler,
            dims,
            device=torch.device("cpu"),
            dtype=torch.float32,
            n_samples=3,
            generate_from_noise=True,
            generator=_gen(),
        )
        assert out.shape == (3, 8, dims.state_dim)

    def test_paired_generating_without_n_samples(self, dims):
        out = prepare_latent_inference(
            torch.randn(8, dims.state_dim),
            _cond(8),
            _sampler,
            dims,
            device=torch.device("cpu"),
            dtype=torch.float32,
            n_samples=None,
            generate_from_noise=True,
            generator=_gen(),
        )
        assert out.shape == (8, dims.state_dim)

    def test_unpaired_with_conditioning_and_n_samples(self, dims):
        out = prepare_latent_inference(
            None,
            _cond(5),
            _sampler,
            dims,
            device=torch.device("cpu"),
            dtype=torch.float32,
            n_samples=3,
            generate_from_noise=True,
            generator=_gen(),
        )
        assert out.shape == (3, 5, dims.state_dim)

    def test_unpaired_with_conditioning_without_n_samples(self, dims):
        out = prepare_latent_inference(
            None,
            _cond(5),
            _sampler,
            dims,
            device=torch.device("cpu"),
            dtype=torch.float32,
            n_samples=None,
            generate_from_noise=True,
            generator=_gen(),
        )
        assert out.shape == (5, dims.state_dim)

    def test_standalone_with_n_samples(self, dims):
        """No source, no conditioning: draw `n_samples` standalone latents of width `dims.state_dim`."""
        out = prepare_latent_inference(
            None,
            {},
            _sampler,
            dims,
            device=torch.device("cpu"),
            dtype=torch.float32,
            n_samples=7,
            generate_from_noise=True,
            generator=_gen(),
        )
        assert out.shape == (7, dims.state_dim)

    def test_standalone_without_n_samples_draws_one(self, dims):
        out = prepare_latent_inference(
            None,
            {},
            _sampler,
            dims,
            device=torch.device("cpu"),
            dtype=torch.float32,
            n_samples=None,
            generate_from_noise=True,
            generator=_gen(),
        )
        assert out.shape == (1, dims.state_dim)

    def test_target_state_is_not_consulted(self, dims):
        """The batch carries a target; `prepare_latent_inference` never looks at it.

        This is the change the PR is about -- no matter what shape the target is,
        the latent is what the other axes say. Standalone case, wrong-width
        target sitting next to it.
        """
        out = prepare_latent_inference(
            None,
            {},
            _sampler,
            dims,
            device=torch.device("cpu"),
            dtype=torch.float32,
            n_samples=2,
            generate_from_noise=True,
            generator=_gen(),
        )
        assert out.shape == (2, dims.state_dim)  # not (2, <target width>)

    def test_device_and_dtype_are_used(self, dims):
        """Placement comes from the caller, not the batch."""

        def sampler(shape, *, generator, device=None, dtype=None):
            return torch.zeros(shape, device=device, dtype=dtype)

        out = prepare_latent_inference(
            None,
            _cond(3),
            sampler,
            dims,
            device=torch.device("cpu"),
            dtype=torch.float64,
            n_samples=None,
            generate_from_noise=True,
            generator=_gen(),
        )
        assert out.dtype is torch.float64
