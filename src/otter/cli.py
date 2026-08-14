"""CLI entry points exposed via ``[project.scripts]`` in pyproject.toml.

After ``pip install -e .``:

    otter-solve [--config fc_plus_SC] [--multistart]
    otter-evaluate [--recompute] [--skip 05c_null_distributions.py]
    otter-artefacts

These wrappers delegate to the corresponding numbered script in
``pipeline/``. Running the pipeline scripts directly with
``PYTHONPATH=src python pipeline/...`` is equivalent.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _pipeline_dir() -> Path:
    """Locate otter/pipeline/. Works whether installed editable or sourced."""
    # Walk up from this file until we find pipeline/
    here = Path(__file__).resolve()
    for parent in (here.parents[2], here.parents[3], here.parents[4]):
        cand = parent / "pipeline"
        if cand.exists():
            return cand
    raise RuntimeError("Could not locate otter/pipeline/ relative to otter package")


def _run(script: str, *extra: str) -> int:
    """Run a pipeline script forwarding sys.argv extras."""
    path = _pipeline_dir() / script
    if not path.exists():
        print(f"Pipeline script not found: {path}", file=sys.stderr)
        return 2
    cmd = [sys.executable, str(path), *extra, *sys.argv[1:]]
    return subprocess.run(cmd).returncode


def solve_production() -> None:
    """Entry point for ``otter-solve``."""
    sys.exit(_run("04_solve_production.py"))


def evaluate() -> None:
    """Entry point for ``otter-evaluate``."""
    sys.exit(_run("05_evaluate.py"))


def build_artefacts() -> None:
    """Entry point for ``otter-artefacts``."""
    sys.exit(_run("07_build_artefacts.py"))
