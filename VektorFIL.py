"""Launcher used by the Nuitka build (tools/build_windows.py) -> VektorFIL.exe."""
import sys

from vektorfil.gui.filter_window import main

if __name__ == "__main__":
    sys.exit(main())
