"""Shared fixtures for method tests."""

import pandas as pd
import pytest
import torch

from sckitflow.data._dims import DataDimensions


@pytest.fixture
def module() -> torch.nn.Module:
    """A module sized to match `dims.state_dim` (4)."""
    return torch.nn.Linear(4, 4)


@pytest.fixture
def dims() -> DataDimensions:
    """A fully-populated `DataDimensions` for tests.

    Values are arbitrary but distinct, so a test that reads the wrong field
    fails on the assert rather than silently getting a plausible-looking shape.
    """
    return DataDimensions(
        state_dim=4,
        feature_names=pd.Index([f"f{i}" for i in range(4)]),
        condition_reps_dims={"cond": 3},
        condition_continuous_dims={"Xcond": 6},
        response_reps_dims=None,
        response_continuous_dims=None,
        groups_reps_dims={"group": 2},
        source_lin_dim=5,
        source_quad_dim=3,
        target_lin_dim=5,
        target_quad_dim=4,
    )
