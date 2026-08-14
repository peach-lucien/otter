#!/usr/bin/env python3
"""Fetch OTTER's data + artifacts from the Zenodo archive (CLI).

    python scripts/fetch_data.py                 # reproduce bundle (default)
    python scripts/fetch_data.py --tier raw      # full raw inputs (for a rebuild)
    python scripts/fetch_data.py --tier all      # both
    python scripts/fetch_data.py --check         # report what's present

This is a thin wrapper around ``otter.data.fetch``; the same logic also runs
automatically (with a prompt) when a library call needs data that isn't present.
See DATA.md for what each tier contains.
"""
import sys
from pathlib import Path

# Work before `pip install -e .` too: put src/ on the path if needed.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from otter.data.fetch import main  # noqa: E402

# The coupling load_pi() returns. Checked after fetching so that an incomplete or stale archive
# reports itself here, rather than surfacing as a bare FileNotFoundError in the first cell of a
# notebook with no indication that the archive, rather than the code, is at fault.
CANONICAL = Path(__file__).resolve().parents[1] / "outputs" / "coupling" / "pi_canonical.npy"

if __name__ == "__main__":
    main()
    if not CANONICAL.exists():
        print(
            f"\nWARNING: {CANONICAL.relative_to(CANONICAL.parents[2])} is not present after "
            f"fetching.\n"
            "load_pi() returns this file, so the notebooks will not run. Check that the archive "
            "in data_manifest.json is v1.3.0 or later.",
            file=sys.stderr,
        )
