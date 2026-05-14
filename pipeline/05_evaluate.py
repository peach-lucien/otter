"""Pipeline step 05 — evaluate the production model end-to-end.

Runs (in order):
    05a_anchor_cv.py            leave-one-network-out CV across all 13 configs
    05b_fc_translation.py       FC-translation Pearson r per production config
    05c_null_distributions.py   random_pi + permuted_anchor null trials
    05d_full_space_eval.py      full-space top-K + mean rank
    05f_beauchamp_validation.py external validation against Beauchamp 2022
    05j_region_level_eval.py    region-level top-K (Beauchamp-22 candidate set)

Each substep is resumable — already-cached cells are skipped. To force a full
recompute, pass --recompute (forwards to each substep).

Usage:
    python pipeline/05_evaluate.py
    python pipeline/05_evaluate.py --recompute   # blow caches and rerun all
    python pipeline/05_evaluate.py --skip 05c_null_distributions.py
"""
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
