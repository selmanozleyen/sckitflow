"""Readouts and potentials used by more than one surrogate, as components so a surrogate's spec can name them."""

from __future__ import annotations

from collections.abc import Callable

import torch
from scfit.registry import Component, component

from sckitflow.core._types import PredictionData

__all__ = [
    "Readout",
    "Potential",
    "ReadoutConfig",
    "PotentialConfig",
    "MeanSamplesReadoutConfig",
    "SquaredErrorPotentialConfig",
]

Readout = Callable[[PredictionData], torch.Tensor]
"""A prediction -> ``(N, D)``, one response per condition."""

Potential = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
"""``(y, ystar)``, shapes ``(N, D')`` and ``(D',)`` -> ``(N,)``, one potential per condition."""


def _mean_samples(pred_data: PredictionData) -> torch.Tensor:
    samples = pred_data.raw_samples
    return samples.reshape(-1, *samples.shape[-2:]).mean(0)


def _squared_error(y: torch.Tensor, ystar: torch.Tensor) -> torch.Tensor:
    return ((y - ystar) ** 2).sum(-1)


class ReadoutConfig(Component):
    """Family base for the readouts."""

    def build(self) -> Readout:
        raise NotImplementedError


class PotentialConfig(Component):
    """Family base for the potentials."""

    def build(self) -> Potential:
        raise NotImplementedError


@component("surrogate.readout.mean_samples")
class MeanSamplesReadoutConfig(ReadoutConfig):
    """The mean of ``raw_samples`` over every axis before the conditions': the Monte Carlo mean response."""

    def build(self) -> Readout:
        return _mean_samples


@component("surrogate.potential.squared_error")
class SquaredErrorPotentialConfig(PotentialConfig):
    """The squared error, summed over the response dimensions."""

    def build(self) -> Potential:
        return _squared_error
