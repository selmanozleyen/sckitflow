"""Name-to-class lookups, consulted only when saving or loading a run.

The asymmetry is deliberate:

* **Construction takes anything.** :class:`~sckitflow.Model` and
  :class:`~sckitflow.trainer.TrainingPlan` accept any object satisfying
  `SupportsTraining` / `SupportsInference`, and never look in these maps. A
  one-off class defined in a notebook trains exactly like a built-in one.
* **Saving needs a name.** A config records *which* class to rebuild as a plain
  string, so a checkpoint survives that class being renamed or moved -- unlike a
  pickled class reference, which does not. A name only exists if the class is
  registered, so an unregistered class trains fine and fails loudly at save
  time rather than producing an artifact nothing can read back.

Register your own to make it saveable:

.. code-block:: python

    @register_method
    class MyTraining(AbstractFlowMethod): ...
"""

from __future__ import annotations

__all__ = [
    "METHOD_CLASSES",
    "MODULE_CLASSES",
    "register_method",
    "register_module",
    "resolve_method",
    "resolve_module",
    "name_for_saving",
]

# name -> class. Written by the decorators, read only by save/load.
METHOD_CLASSES: dict[str, type] = {}
MODULE_CLASSES: dict[str, type] = {}


def _register[C: type](where: dict[str, type], cls: C, kind: str) -> C:
    name = cls.__name__
    known = where.get(name)
    if known is not None and known is not cls:
        raise ValueError(
            f"a different {kind} is already registered as {name!r} "
            f"({known.__module__}.{known.__qualname__}); names must be unique to resolve."
        )
    where[name] = cls
    return cls


def register_method[C: type](cls: C) -> C:
    """Makes a training or inference method saveable by name. Returns `cls`."""
    return _register(METHOD_CLASSES, cls, "method")


def register_module[C: type](cls: C) -> C:
    """Makes a neural module saveable by name. Returns `cls`."""
    return _register(MODULE_CLASSES, cls, "module")


def _resolve(where: dict[str, type], name: str, kind: str) -> type:
    try:
        return where[name]
    except KeyError as e:
        raise KeyError(
            f"no {kind} registered as {name!r}; import the module that defines it, "
            f"or decorate it with `@register_{kind}`. Known: {sorted(where)}."
        ) from e


def resolve_method(name: str) -> type:
    """The method class registered under `name`."""
    return _resolve(METHOD_CLASSES, name, "method")


def resolve_module(name: str) -> type:
    """The neural module class registered under `name`."""
    return _resolve(MODULE_CLASSES, name, "module")


def name_for_saving(obj: object, *, kind: str) -> str:
    """The registered name of `obj`'s class, to record in a config.

    This is the gate: anything can be constructed and trained, but only a
    registered class can be written down.

    :param obj: The method or module instance about to be saved.
    :param kind: ``"method"`` or ``"module"``, selecting the map to look in.
    :raises TypeError: If `obj`'s class is not registered.
    """
    where = METHOD_CLASSES if kind == "method" else MODULE_CLASSES
    cls = type(obj)
    if where.get(cls.__name__) is cls:
        return cls.__name__
    raise TypeError(
        f"cannot save {cls.__module__}.{cls.__qualname__}: it is not a registered {kind}. "
        f"Constructing and training with it is fine -- saving is not, because a config records "
        f"the class by name and there is no name for it. Decorate it with `@register_{kind}`."
    )
