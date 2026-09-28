"""Two helpers every `scfit.registry.Component` config in sckitflow builds with.

Neither is specific to sckitflow. Every package on scfit writes the same
field-forwarding builds and resolves the same ``Config | Live`` unions, so both
are candidates for `scfit.registry` itself.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import fields
from typing import Any

from scfit.registry import Component

__all__ = ["config_fields", "built"]


def config_fields(config: Component, *, exclude: Collection[str] = ()) -> dict[str, Any]:
    """A config's fields as constructor keyword arguments.

    Shallow on purpose: `dataclasses.asdict` would recurse into a nested config
    and flatten it into a plain dict, losing what it is.

    :param config: A dataclass config.
    :param exclude: Field names to leave out.
    """
    return {f.name: getattr(config, f.name) for f in fields(config) if f.name not in exclude}


def built(value: Any, context: Any = None) -> Any:
    """Builds a config into its runtime object, and passes a live object through.

    Resolves the ``Config | Live`` union that scfit's escape hatch allows on a
    field: scfit parses such a field, but leaves building it to the caller.

    :param value: A `Component`, a live instance, or `None`.
    :param context: Forwarded to ``value.build`` when `value` is a config.
    """
    return value.build(context) if isinstance(value, Component) else value
