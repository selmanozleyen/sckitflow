"""Compact TTY-aware logging for sckitflow pipelines.

One entry point, `get_logger`, that returns a stdlib logger configured with:

  * color when attached to a TTY, plain when piped
  * ``NO_COLOR`` / ``FORCE_COLOR`` env vars respected (https://no-color.org/)
  * optional file handler alongside the console handler
  * exception tracebacks on their own lines (no ``%`` pollution of the log)
  * multiline messages indented so config dumps stay readable
  * idempotent setup (safe to call twice; uses a marker, not ``logger.handlers``)

Level resolution order (first match wins):

  1. explicit ``level=`` argument
  2. ``config.run.log_level`` (only if ``config`` is passed)
  3. ``$SCKITFLOW_LOG_LEVEL`` env var
  4. ``"INFO"``
"""

from __future__ import annotations

import logging
import os
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from omegaconf import DictConfig

__all__ = ["get_logger"]


# --- ANSI codes ---------------------------------------------------------------

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"

_LEVEL_COLOR = {
    logging.DEBUG: _DIM,
    logging.INFO: _CYAN,
    logging.WARNING: _YELLOW,
    logging.ERROR: _RED,
    logging.CRITICAL: _BOLD + _RED,
}

# Marks a logger as configured by *this* module, so re-entry is a no-op even
# if another framework has attached handlers of its own.
_MARKER_ATTR = "_sckitflow_configured"


# --- helpers ------------------------------------------------------------------


def _env_flag(name: str) -> bool | None:
    """Return True/False for a boolean env var; ``None`` if unset."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    if raw == "":
        return True  # e.g. NO_COLOR= means "on" per no-color.org
    return raw.lower() not in ("0", "false", "no", "off")


def _should_color(stream: TextIO) -> bool:
    if _env_flag("NO_COLOR") is True:
        return False
    if _env_flag("FORCE_COLOR") is True:
        return True
    return hasattr(stream, "isatty") and stream.isatty()


def _parse_level(level: str | int) -> int:
    """Parse a level specification to an int, falling back to INFO."""
    if isinstance(level, int):
        return level
    named = logging.getLevelNamesMapping().get(str(level).upper())
    return named if named is not None else logging.INFO


# --- formatter ----------------------------------------------------------------


class _Formatter(logging.Formatter):
    """Compact formatter with TTY-aware color, multiline, and exception support."""

    def __init__(self, use_color: bool) -> None:
        super().__init__(datefmt="[%H:%M:%S]")
        self._use_color = use_color

    def _prefix(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record)  # uses self.datefmt
        lvl = f"{record.levelname:<8}"
        if not self._use_color:
            return f"{ts} {lvl}"
        color = _LEVEL_COLOR.get(record.levelno, "")
        return f"{_DIM}{ts}{_RESET} {color}{lvl}{_RESET}"

    def format(self, record: logging.LogRecord) -> str:
        prefix = self._prefix(record)
        message = record.getMessage()
        if "\n" in message:
            indent = " " * (len(prefix) + 1)
            message = message.replace("\n", "\n" + indent)
        line = f"{prefix} {message}"
        if record.exc_info:
            tb = "".join(traceback.format_exception(*record.exc_info)).rstrip()
            if self._use_color:
                tb = f"{_RED}{tb}{_RESET}"
            line = f"{line}\n{tb}"
        return line


# --- public API ---------------------------------------------------------------


def get_logger(
    name: str = "sckitflow",
    config: DictConfig | None = None,
    *,
    level: str | int | None = None,
    log_file: str | Path | None = None,
    force_color: bool | None = None,
) -> logging.Logger:
    """Initialise a stdlib logger for a sckitflow pipeline.

    Idempotent: calling twice with the same ``name`` returns the same logger
    without adding a second handler. Level resolution is first-wins:

    1. ``level=`` argument
    2. ``config.run.log_level`` (only if ``config`` is passed)
    3. ``$SCKITFLOW_LOG_LEVEL`` env var
    4. ``"INFO"``

    :param name: Logger name. Defaults to ``"sckitflow"``; pass ``__name__``
        for a per-module logger.
    :param config: Optional Hydra config; only ``run.log_level`` is read.
    :param level: Explicit level override (``"DEBUG"``, ``"INFO"``, ``10``, …).
    :param log_file: Optional path; when set, an additional ``FileHandler``
        is attached that writes the same records (color disabled) to disk.
    :param force_color: Force color on/off; ``None`` auto-detects from
        ``NO_COLOR`` / ``FORCE_COLOR`` / TTY.
    """
    logger = logging.getLogger(name)
    if getattr(logger, _MARKER_ATTR, False):
        return logger

    # ---- resolve level -------------------------------------------------------
    if level is None:
        if config is not None:
            run_cfg = config.get("run")
            if run_cfg is not None:
                level = run_cfg.get("log_level")
        if level is None:
            level = os.environ.get("SCKITFLOW_LOG_LEVEL", "INFO")
    logger.setLevel(_parse_level(level))

    # ---- console handler -----------------------------------------------------
    console = logging.StreamHandler()  # attaches to sys.stderr at construction
    if force_color is None:
        use_color = _should_color(console.stream)
    else:
        use_color = force_color
    console.setFormatter(_Formatter(use_color=use_color))
    logger.addHandler(console)

    # ---- optional file handler ----------------------------------------------
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, mode="a", encoding="utf-8")
        fh.setFormatter(_Formatter(use_color=False))
        logger.addHandler(fh)

    logger.propagate = False
    setattr(logger, _MARKER_ATTR, True)
    return logger
