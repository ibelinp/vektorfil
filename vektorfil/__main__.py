"""Entry point: ``python -m vektorfil`` (and the compiled VektorFIL.exe) starts the filter
designer; ``--selftest`` runs a quick check without opening a window."""
import sys

from vektorfil.gui.filter_window import main

if __name__ == "__main__":
    sys.exit(main())
