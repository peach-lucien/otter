"""Pipeline step 03 — orchestrator: run all three cost-building substeps.

Equivalent to running 03a → 03b → 03c in order. Builds the full set of cost
matrices used by every downstream model:

    outputs/anndata/full_costs.npz    (FC + xyz + SC + gene + cross-species M)

Usage:
    python pipeline/03_build_costs.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent

STEPS = [
    "03a_build_full_costs.py",
    "03b_build_spatial_costs.py",
    "03c_build_multimodal_costs.py",
]


def main():
    for step in STEPS:
        path = PIPELINE / step
        if not path.exists():
            print(f"[03 orchestrator] skipping missing {step}")
            continue
        print(f"\n{'=' * 60}\n[03 orchestrator] running {step}\n{'=' * 60}")
        result = subprocess.run([sys.executable, str(path)],
                                cwd=PIPELINE.parent)
        if result.returncode != 0:
            sys.exit(f"[03 orchestrator] {step} failed (exit {result.returncode})")
    print("\n[03 orchestrator] all cost-build substeps completed successfully")


if __name__ == "__main__":
    main()
