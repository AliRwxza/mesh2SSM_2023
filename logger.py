"""
Logger Module
=============
Centralised, Kaggle-aware logging for mesh2SSM.

Design goals
------------
* **Dual output**: every message goes to stdout (visible in the live Kaggle
  notebook cell) AND to a rotating log file on disk (readable after a crashed
  commit via the Kaggle output viewer).
* **Silent-crash detection**: installs a top-level ``sys.excepthook`` and
  ``faulthandler`` so that OOM kills, segfaults, and unhandled Python
  exceptions all leave a trace in the log file.
* **CUDA diagnostics**: GPU memory snapshots attached to every log record so
  you can see exactly which step ran out of memory.
* **Concise API**: one import, one ``get_logger()`` call, then use the returned
  logger exactly like the standard ``logging.Logger``.

Usage (in any module)
---------------------
    from logger import get_logger
    log = get_logger(__name__)

    log.info("Starting training")
    log.debug("batch shape: %s", data.shape)
    log.error("Something went wrong", exc_info=True)

    # Context manager - logs entry/exit + wall-clock time
    with log.section("DataLoader init"):
        loader = DataLoader(...)

    # One-liner tensor/memory snapshot (no-op if CUDA unavailable)
    log.gpu_snapshot("after forward pass")
"""

from __future__ import annotations

import datetime
import faulthandler
import logging
import os
import platform
import sys
import textwrap
import time
import traceback
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Public knobs
# ---------------------------------------------------------------------------

#: Directory where log files are written.  Override before the first
#: ``get_logger()`` call if you want a different location.
LOG_DIR: str = os.environ.get("MESH2SSM_LOG_DIR", "logs")

#: Base name of the rotating log file.
LOG_FILE: str = "mesh2ssm.log"

#: Maximum size of a single log file (bytes) before rotation.
LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB

#: How many rotated backup files to keep.
LOG_BACKUP_COUNT: int = 5

#: Root logger level.  Set to ``logging.DEBUG`` for maximum verbosity.
LOG_LEVEL: int = logging.DEBUG

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

_CONFIGURED: bool = False          # True after _configure() has been called
_FILE_HANDLER: Optional[RotatingFileHandler] = None
_LOG_PATH: Optional[Path] = None


# ---------------------------------------------------------------------------
# Custom formatter - adds UTC timestamp + GPU memory
# ---------------------------------------------------------------------------

class _KaggleFormatter(logging.Formatter):
    """Compact formatter suited for Kaggle's scrollable cell output."""

    _FMT = "[{asctime}] [{levelname:^8s}] [{name}] {message}"
    _DATEFMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self, include_gpu: bool = False) -> None:
        super().__init__(fmt=self._FMT, datefmt=self._DATEFMT, style="{")
        self._include_gpu = include_gpu and torch.cuda.is_available()

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        base = super().format(record)
        if self._include_gpu:
            try:
                alloc = torch.cuda.memory_allocated() / 1024 ** 2
                reserved = torch.cuda.memory_reserved() / 1024 ** 2
                base += f"  [GPU {alloc:.0f}/{reserved:.0f} MB alloc/reserved]"
            except Exception:  # pragma: no cover
                pass
        return base


# ---------------------------------------------------------------------------
# Extended Logger class
# ---------------------------------------------------------------------------

class _Mesh2SSMLogger(logging.Logger):
    """
    Thin subclass that adds ``section()`` and ``gpu_snapshot()`` helpers.
    Drop-in replacement for ``logging.Logger``.
    """

    # ------------------------------------------------------------------
    # section() - context manager
    # ------------------------------------------------------------------

    @contextmanager
    def section(self, name: str, level: int = logging.INFO):
        """
        Context manager that logs entry and exit of a named block, including
        the wall-clock elapsed time.

        Example::

            with log.section("Loading dataset"):
                dataset = MeshesWithFaces(...)
        """
        self.log(level, ">>> BEGIN  %s", name)
        t0 = time.perf_counter()
        try:
            yield
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            self.exception(
                "!!! FAILED %s  (%.2fs)  ->  %s: %s",
                name, elapsed, type(exc).__name__, exc,
            )
            raise
        else:
            elapsed = time.perf_counter() - t0
            self.log(level, "<<< END    %s  (%.2fs)", name, elapsed)

    # ------------------------------------------------------------------
    # gpu_snapshot() - memory report
    # ------------------------------------------------------------------

    def gpu_snapshot(self, tag: str = "", level: int = logging.DEBUG) -> None:
        """
        Logs current GPU memory usage.  No-op when CUDA is not available.

        Args:
            tag:   Short label prepended to the log line (e.g. "after forward").
            level: ``logging`` level to emit at.
        """
        if not torch.cuda.is_available():
            return
        try:
            alloc    = torch.cuda.memory_allocated()  / 1024 ** 2
            reserved = torch.cuda.memory_reserved()   / 1024 ** 2
            maxalloc = torch.cuda.max_memory_allocated() / 1024 ** 2
            dev      = torch.cuda.current_device()
            name     = torch.cuda.get_device_name(dev)
            prefix   = f"[GPU:{dev} {name}]"
            if tag:
                prefix += f" [{tag}]"
            self.log(
                level,
                "%s  alloc=%.1f MB  reserved=%.1f MB  peak=%.1f MB",
                prefix, alloc, reserved, maxalloc,
            )
        except Exception as exc:  # pragma: no cover
            self.debug("gpu_snapshot failed: %s", exc)


# ---------------------------------------------------------------------------
# One-time setup
# ---------------------------------------------------------------------------

def _configure() -> None:
    """
    Configure the root logger exactly once:
    1. Rotating file handler  -> ``LOG_DIR/LOG_FILE``
    2. Stream handler         -> stdout (Kaggle cell output)
    3. faulthandler           -> low-level crash traces (segfaults, SIGKILL)
    4. sys.excepthook         -> unhandled Python exceptions
    """
    global _CONFIGURED, _FILE_HANDLER, _LOG_PATH

    if _CONFIGURED:
        return

    # ------------------------------------------------------------------
    # 1. Directory and file
    # ------------------------------------------------------------------
    log_dir = Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    _LOG_PATH = log_dir / LOG_FILE

    # ------------------------------------------------------------------
    # 2. Root logger
    # ------------------------------------------------------------------
    # Register our custom Logger class BEFORE any logger is created
    logging.setLoggerClass(_Mesh2SSMLogger)

    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)

    # Guard against duplicate handlers if _configure() is somehow called
    # multiple times in the same process
    root.handlers.clear()

    # File handler - GPU memory included for post-mortem analysis
    _FILE_HANDLER = RotatingFileHandler(
        _LOG_PATH,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    _FILE_HANDLER.setLevel(LOG_LEVEL)
    _FILE_HANDLER.setFormatter(_KaggleFormatter(include_gpu=True))
    root.addHandler(_FILE_HANDLER)

    # Stream handler - no GPU suffix to keep notebook output cleaner
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(LOG_LEVEL)
    stream_handler.setFormatter(_KaggleFormatter(include_gpu=False))
    root.addHandler(stream_handler)

    # ------------------------------------------------------------------
    # 3. Emit startup banner
    # ------------------------------------------------------------------
    _startup_banner(logging.getLogger("logger"))

    # ------------------------------------------------------------------
    # 4. faulthandler - writes C-level tracebacks to the log file on crash
    # ------------------------------------------------------------------
    try:
        faulthandler.enable(file=_FILE_HANDLER.stream, all_threads=True)
    except Exception as exc:  # pragma: no cover
        logging.getLogger("logger").warning(
            "faulthandler could not be enabled: %s", exc
        )

    # ------------------------------------------------------------------
    # 5. sys.excepthook - catches unhandled Python exceptions
    # ------------------------------------------------------------------
    _orig_excepthook = sys.excepthook

    def _excepthook(exc_type, exc_value, exc_tb):
        _flush_on_crash(exc_type, exc_value, exc_tb)
        _orig_excepthook(exc_type, exc_value, exc_tb)

    sys.excepthook = _excepthook

    _CONFIGURED = True


def _startup_banner(log: logging.Logger) -> None:
    """Emits a structured startup block with environment information."""
    sep = "=" * 72
    lines = [
        sep,
        f"  mesh2SSM  -  run started at {datetime.datetime.utcnow().isoformat()} UTC",
        f"  Python   : {sys.version.split()[0]}  ({platform.python_implementation()})",
        f"  Platform : {platform.platform()}",
        f"  PID      : {os.getpid()}",
        f"  Log file : {_LOG_PATH}",
    ]

    # CUDA info
    if torch.cuda.is_available():
        lines += [
            f"  CUDA     : {torch.version.cuda}  /  torch {torch.__version__}",
            f"  GPU(s)   : {torch.cuda.device_count()}",
        ]
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            lines.append(
                f"             [{i}] {props.name}  "
                f"total={props.total_memory / 1024**3:.1f} GB"
            )
    else:
        lines.append(f"  CUDA     : not available  /  torch {torch.__version__}")

    lines.append(sep)
    for line in lines:
        log.info(line)


def _flush_on_crash(
    exc_type: type,
    exc_value: BaseException,
    exc_tb,
) -> None:
    """
    Called by sys.excepthook on any unhandled exception.
    Writes a full traceback + GPU state to the log file so it survives a
    crashed Kaggle commit.
    """
    log = logging.getLogger("CRASH")
    tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    sep = "!" * 72
    log.critical(sep)
    log.critical("UNHANDLED EXCEPTION  ->  %s: %s", exc_type.__name__, exc_value)
    log.critical("Full traceback:\n%s", textwrap.indent(tb_str, "    "))

    if torch.cuda.is_available():
        try:
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                alloc    = torch.cuda.memory_allocated()  / 1024 ** 2
                reserved = torch.cuda.memory_reserved()   / 1024 ** 2
                maxalloc = torch.cuda.max_memory_allocated() / 1024 ** 2
                log.critical(
                    "  GPU[%d] alloc=%.1f MB  reserved=%.1f MB  peak=%.1f MB",
                    i, alloc, reserved, maxalloc,
                )
        except Exception:  # pragma: no cover
            pass

    log.critical(sep)

    # Force-flush to disk so the data survives even if the process is killed
    if _FILE_HANDLER is not None:
        try:
            _FILE_HANDLER.flush()
        except Exception:  # pragma: no cover
            pass


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def get_logger(name: str = "mesh2ssm") -> _Mesh2SSMLogger:
    """
    Returns a fully-configured ``_Mesh2SSMLogger`` for the given module name.

    Call once per module at import time::

        from logger import get_logger
        log = get_logger(__name__)

    The first call also performs one-time global configuration (file handler,
    faulthandler, excepthook).

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A ``_Mesh2SSMLogger`` instance (subclass of ``logging.Logger``).
    """
    _configure()
    logger = logging.getLogger(name)
    # Ensure the object is actually our subclass even if the standard library
    # cached an instance before setLoggerClass() was called.
    if not isinstance(logger, _Mesh2SSMLogger):
        logger.__class__ = _Mesh2SSMLogger
    return logger  # type: ignore[return-value]


def get_log_path() -> Optional[Path]:
    """
    Returns the absolute path of the current log file, or ``None`` if logging
    has not been initialised yet.
    """
    return _LOG_PATH
