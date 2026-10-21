from dataclasses import dataclass

import pandas as pd

__all__ = ["DataDimensions"]


@dataclass(frozen=True)
class DataDimensions:
    state_dim: int
    feature_names: pd.Index
    condition_reps_dims: dict[str, int] | None
    condition_continuous_dims: dict[str, int] | None
    response_reps_dims: dict[str, int] | None
    response_continuous_dims: dict[str, int] | None
    groups_reps_dims: dict[str, int] | None
    source_lin_dim: int | None
    source_quad_dim: int | None
    target_lin_dim: int | None
    target_quad_dim: int | None
    source_state_dim: int | None = None
    """The width of the sources' own state rep (``DataManagerConfig.source_sample_rep``). ``None``: the sources share
    the targets' state, ``state_dim``."""
