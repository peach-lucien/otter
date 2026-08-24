"""Command-line entry point for refitting the canonical OTTER coupling."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _pipeline_dir() -> Path:
    """Locate the repository's pipeline directory from an editable install."""
    here = Path(__file__).resolve()
    for parent in (here.parents[2], here.parents[3], here.parents[4]):
        candidate = parent / "pipeline"
        if candidate.exists():
            return candidate
    raise RuntimeError("Could not locate pipeline/ relative to the otter package")


def solve_canonical() -> None:
    """Refit the canonical coupling, forwarding command-line arguments."""
    script = _pipeline_dir() / "run_recommended_model.py"
    command = [sys.executable, str(script), *sys.argv[1:]]
    raise SystemExit(subprocess.run(command).returncode)
