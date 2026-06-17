#!/usr/bin/env python3
"""Fetch HOMER's data + artifacts from the Zenodo archive (CLI).

    python scripts/fetch_data.py                 # reproduce bundle (default)
    python scripts/fetch_data.py --tier raw      # full raw inputs (for a rebuild)
    python scripts/fetch_data.py --tier all      # both
    python scripts/fetch_data.py --check         # report what's present

This is a thin wrapper around ``homer.data.fetch``; the same logic also runs
automatically (with a prompt) when a library call needs data that isn't present.
See DATA.md for what each tier contains.
"""
import sys
from pathlib import Path

# Work before `pip install -e .` too: put src/ on the path if needed.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from homer.data.fetch import main  # noqa: E402

if __name__ == "__main__":
    main()
