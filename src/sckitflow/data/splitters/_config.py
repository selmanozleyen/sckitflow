"""Portable configs for the splitters.

A splitter config holds no seed: ``build`` takes the ``rng`` the run derives
from its split seed.
"""

from __future__ import annotations

import numpy as np
from scfit.registry import Component, component

from sckitflow.data.splitters._base import Splitter
from sckitflow.data.splitters._combination import CombinationSplitter

__all__ = ["SplitterConfig", "CombinationSplitterConfig"]


class SplitterConfig(Component):
    """Family base for the splitters."""

    def build(self, *, rng: np.random.Generator) -> Splitter:
        """The splitter, drawing its hold-out choice from ``rng``."""
        raise NotImplementedError


@component("splitter.combination")
class CombinationSplitterConfig(SplitterConfig):
    """Holds out whole condition combinations. See :class:`CombinationSplitter`."""

    group_keys: tuple[str, ...] = ()
    always_train_keys: tuple[str, ...] = ()
    control_key: str | None = None
    control_value: str = "control"
    test_fraction: float = 0.2
    split_key: str = "split"
    train_label: str = "train"
    test_label: str = "test"
    control_label: str = "control"

    def build(self, *, rng: np.random.Generator) -> CombinationSplitter:
        return CombinationSplitter(
            group_keys=self.group_keys,
            always_train_keys=self.always_train_keys,
            control_key=self.control_key,
            control_value=self.control_value,
            test_fraction=self.test_fraction,
            rng=rng,
            split_key=self.split_key,
            train_label=self.train_label,
            test_label=self.test_label,
            control_label=self.control_label,
        )
