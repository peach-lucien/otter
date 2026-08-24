"""Run the low-level diagnostic scripts listed in STEPS.

Current manuscript analyses are organised under experiments/ and are not orchestrated by this helper."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent

STEPS = [
    "05a_anchor_cv.py",
    "05b_fc_translation.py",
    "05c_null_distributions.py",
    "05d_full_space_eval.py",
    "05f_beauchamp_validation.py",
    "05j_region_level_eval.py",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true",
                    help="forward --recompute to each substep")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="substep filenames to skip (e.g. --skip 05c_null_distributions.py)")
    args = ap.parse_args()

    for step in STEPS:
        if step in args.skip:
            print(f"[05 orchestrator] skipping {step} (--skip)")
            continue
        path = PIPELINE / step
        if not path.exists():
            print(f"[05 orchestrator] missing {step}, skipping")
            continue
        cmd = [sys.executable, str(path)]
        if args.recompute:
            cmd.append("--recompute")
        print(f"\n{'=' * 60}\n[05 orchestrator] running {step}\n{'=' * 60}")
        result = subprocess.run(cmd, cwd=PIPELINE.parent)
        if result.returncode != 0:
            sys.exit(f"[05 orchestrator] {step} failed (exit {result.returncode})")
    print("\n[05 orchestrator] all evaluation substeps completed")


if __name__ == "__main__":
    main()
