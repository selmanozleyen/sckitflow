"""The readout and potential components a surrogate's spec can name."""

import torch

from sckitflow.core._types import PredictionData
from sckitflow.core.surrogate import (
    MeanSamplesReadoutConfig,
    PotentialConfig,
    ReadoutConfig,
    SquaredErrorPotentialConfig,
)


def test_mean_samples_averages_every_axis_before_the_conditions():
    samples = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    out = MeanSamplesReadoutConfig().build()(PredictionData(X=None, traj=None, raw_samples=samples))
    torch.testing.assert_close(out, samples.mean(dim=(0, 1)))


def test_squared_error_sums_over_response_dimensions():
    y, ystar = torch.tensor([[1.0, 2.0], [0.0, 0.0]]), torch.tensor([1.0, 0.0])
    torch.testing.assert_close(SquaredErrorPotentialConfig().build()(y, ystar), torch.tensor([4.0, 1.0]))


def test_specs_round_trip_through_their_family():
    assert ReadoutConfig.from_spec(MeanSamplesReadoutConfig().to_spec()) == MeanSamplesReadoutConfig()
    assert PotentialConfig.from_spec(SquaredErrorPotentialConfig().to_spec()) == SquaredErrorPotentialConfig()
