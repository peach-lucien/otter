"""Fit the canonical OTTER coupling.

This numbered entry point forwards to pipeline/run_recommended_model.py.
"""
from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).with_name("run_recommended_model.py")),
        run_name="__main__",
    )
