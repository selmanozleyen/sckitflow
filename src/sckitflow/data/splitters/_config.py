"""Portable configs for the splitters.

Every field a splitter takes is a string, number or sequence of strings, so a
splitter is fully portable and does not belong behind
:func:`scfit.registry.register_live`.
"""

from __future__ import annotations

from dataclasses import dataclass

from scfit.registry import Component

from sckitflow.data.splitters._base import Splitter
from sckitflow.data.splitters._combination import CombinationSplitter

__all__ = ["SplitterConfig", "CombinationSplitterConfig"]


# No `type_id`: the family base stays unregistered so it can be the `expected`
# family in `SplitterConfig.from_spec(spec)`.
class SplitterConfig(Component):
    """Family base for the splitters."""

    def build(self, context: object = None) -> Splitter:
        raise NotImplementedError


@dataclass(frozen=True)
class CombinationSplitterConfig(SplitterConfig, type_id="splitter.combination", version=1):
    """Holds out whole condition combinations. See :class:`CombinationSplitter`."""

    group_keys: tuple[str, ...] = ()
    always_train_keys: tuple[str, ...] = ()
    control_key: str | None = None
    control_value: str = "control"
    test_fraction: float = 0.2
    seed: int = 0
    split_key: str = "split"
    train_label: str = "train"
    test_label: str = "test"
    control_label: str = "control"

    def build(self, context: object = None) -> CombinationSplitter:
        return CombinationSplitter(
            group_keys=self.group_keys,
            always_train_keys=self.always_train_keys,
            control_key=self.control_key,
            control_value=self.control_value,
            test_fraction=self.test_fraction,
            seed=self.seed,
            split_key=self.split_key,
            train_label=self.train_label,
            test_label=self.test_label,
            control_label=self.control_label,
        )
