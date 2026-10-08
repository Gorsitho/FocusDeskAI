"""
This file sets up the technical log of FocusDesk AI.
Its main job is to save messages, warnings and errors to a file, even when
there is no console window.

Why this is needed: in the .exe built with PyInstaller (``console=False``),
``sys.stdout`` and ``sys.stderr`` are None. Some libraries (Ultralytics, tqdm)
print text, and without a place to print they fail. In the past this silently
turned off phone detection.

This file:

* writes a log file that rotates (older parts are kept as backups) to
  %LOCALAPPDATA%\\FocusDeskAI\\FocusDeskAI.log,
* replaces missing Python output streams with streams that write to this log,
* logs errors that nobody caught (main thread and other threads) with the full traceback,
* sends hard crashes and C/C++ library output (MediaPipe) to
  FocusDeskAI_native.log in the same folder.

It uses app/config/paths.py to find the log folder. app/main.py calls it first.
"""

import faulthandler
import io
import logging
import os
import platform
import sys
import threading
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.config.paths import is_frozen, log_file_path

LOG_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-8s [%(threadName)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_BYTES = 2_000_000
BACKUP_COUNT = 3

logger = logging.getLogger("focusdesk")
_native_file = None  # kept open for faulthandler / redirected native stderr


class StreamToLogger(io.TextIOBase):
    """File-like object that forwards complete lines to a logger.

    Carriage-return progress bars (tqdm) only log their final state per line.
    """

    def __init__(self, target: logging.Logger, level: int):
        super().__init__()
        self._logger = target
        self._level = level
        self._buffer = ""

    @property
    def encoding(self) -> str:
        return "utf-8"

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.split("\r")[-1].rstrip()
            if line:
                self._logger.log(self._level, line)
        return len(text)

    def flush(self) -> None:
        # Partial lines (progress bars in flight) are kept until their newline arrives.
        pass


def _is_our_handler(handler: logging.Handler) -> bool:
    return getattr(handler, "_focusdesk", False)


def setup_logging(log_path: Path | None = None, level: int = logging.INFO, console: bool | None = None) -> Path:
    """Configure logging once at startup; safe to call again (handlers are replaced)."""
    path = Path(log_path) if log_path is not None else log_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Give library code a place to print to before anything else runs.
    if sys.stdout is None:
        sys.stdout = StreamToLogger(logging.getLogger("stdout"), logging.INFO)
    if sys.stderr is None:
        sys.stderr = StreamToLogger(logging.getLogger("stderr"), logging.WARNING)

    root = logging.getLogger()
    root.setLevel(level)
    for handler in [h for h in root.handlers if _is_our_handler(h)]:
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    file_handler = RotatingFileHandler(path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler._focusdesk = True
    root.addHandler(file_handler)

    if console is None:
        console = not isinstance(sys.stderr, StreamToLogger)
    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(formatter)
        console_handler._focusdesk = True
        root.addHandler(console_handler)

    logging.captureWarnings(True)
    sys.excepthook = log_uncaught_exception
    threading.excepthook = _log_thread_exception
    _setup_native_output(path.with_name("FocusDeskAI_native.log"))
    return path


def log_uncaught_exception(exc_type, exc_value, exc_traceback) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))


def _log_thread_exception(args: threading.ExceptHookArgs) -> None:
    if args.exc_type is SystemExit:
        return
    name = args.thread.name if args.thread is not None else "?"
    logger.critical("Unhandled exception in thread %s", name,
                    exc_info=(args.exc_type, args.exc_value, args.exc_traceback))


def _native_stderr_missing() -> bool:
    try:
        os.fstat(2)
        return False
    except OSError:
        return True


def _setup_native_output(native_path: Path) -> None:
    """Route hard crashes and C/C++ stderr output to a file."""
    global _native_file
    try:
        if _native_file is None:
            _native_file = open(native_path, "w", encoding="utf-8", buffering=1)
            _native_file.write(
                f"FocusDesk AI native log, started {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z}\n"
                "Native crashes and C/C++ library output (MediaPipe, TensorFlow Lite).\n"
                "Note: on Windows, 'Windows fatal exception' entries also appear for exceptions that\n"
                "Windows handled internally (e.g. accessibility/automation tools). If FocusDeskAI.log\n"
                "continues normally after that time, the app did not crash.\n\n"
            )
        if _native_stderr_missing():
            # No console: point the C-level stdout/stderr at the native log, so MediaPipe
            # and TensorFlow Lite messages are kept instead of being lost.
            os.dup2(_native_file.fileno(), 1)
            os.dup2(_native_file.fileno(), 2)
        faulthandler.enable(file=_native_file, all_threads=True)
    except OSError:
        logger.warning("Native crash logging unavailable", exc_info=True)


def log_startup() -> None:
    """Write basic facts about this run (process id, Python version, paths) to the log."""
    logger.info("=" * 72)
    logger.info("FocusDesk AI starting (pid %d)", os.getpid())
    logger.info("Python %s on %s", sys.version.split()[0], platform.platform())
    logger.info("Executable: %s (frozen=%s)", sys.executable, is_frozen())
    logger.info("Log file: %s", log_file_path())
