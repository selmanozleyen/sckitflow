"""The one place that turns a seed into generators."""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["generators"]

# Stream keys: each stage draws from its own generators, never from another stage's.
TRAIN, VALIDATE, PREDICT = 0, 1, 2


def generators(seed: int, *key: int, device: torch.types.Device = None) -> torch.Generator:
    """A fresh torch generator for the stream ``key`` of ``seed``, e.g. ``(stage, step)``.

    The same ``(seed, *key)`` always gives the same draws, so a resumed run repeats its streams without
    saved RNG state, and no global generator is read or written.
    """
    (torch_seed,) = np.random.SeedSequence(seed, spawn_key=key).generate_state(1, np.uint64)
    return torch.Generator(device=device).manual_seed(int(torch_seed))
