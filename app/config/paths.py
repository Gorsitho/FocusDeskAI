"""Writable per-user locations (logs, session logs, downloaded models).

The install directory of a PyInstaller build may be read-only, so everything the
app writes lives under %LOCALAPPDATA%\\FocusDeskAI (or a platform equivalent).
"""

import os
import sys
import tempfile
from pathlib import Path

APP_DIR_NAME = "FocusDeskAI"


def app_data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_STATE_HOME")
    if not base:
        base = str(Path.home() / ("AppData/Local" if sys.platform == "win32" else ".local/state"))
    return Path(base) / APP_DIR_NAME


def _writable(directory: Path) -> Path:
    """Create the directory, falling back to the temp folder if that is impossible."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".write_test"
        probe.write_bytes(b"")
        probe.unlink()
        return directory
    except OSError:
        fallback = Path(tempfile.gettempdir()) / APP_DIR_NAME / directory.name
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def log_file_path() -> Path:
    return _writable(app_data_dir()) / "FocusDeskAI.log"


def sessions_dir() -> Path:
    return _writable(app_data_dir() / "sessions")


def user_models_dir() -> Path:
    return _writable(app_data_dir() / "models")


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))
