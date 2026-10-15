"""Where VektorFIL keeps its files, both when run from source and when installed.

* The log file goes to the user's application-data folder (always writable, also when
  the program is installed under Program Files):
    Windows: %LOCALAPPDATA%\\VektorFIL\\vektorfil.log
    Linux/macOS: ~/.local/share/VektorFIL/vektorfil.log
* Bundled resources (the user guide, licence texts, icon) live next to the program in a
  compiled build, or in the source tree when run with Python.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "VektorFIL"


def is_frozen() -> bool:
    """True in a compiled (Nuitka / PyInstaller) build."""
    return "__compiled__" in globals() or getattr(sys, "frozen", False)


def resource_root() -> Path:
    """Folder that holds docs/, LICENSE, THIRD_PARTY_LICENSES.txt and assets/."""
    if getattr(sys, "_MEIPASS", None):          # PyInstaller: bundled data lives here
        return Path(sys._MEIPASS)
    if is_frozen():
        return Path(sys.argv[0]).resolve().parent
    return Path(__file__).resolve().parent.parent


def resource(*parts) -> Path:
    """Path of a bundled resource, e.g. resource('docs', 'index.html')."""
    return resource_root().joinpath(*parts)


def data_dir() -> Path:
    """Per-user writable folder for logs and settings (created on demand)."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    d = base / APP_NAME
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        d = Path.home()
    return d


def log_file() -> Path:
    """The error log written by the crash guard."""
    return data_dir() / "vektorfil.log"
