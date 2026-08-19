#!/usr/bin/env python3
"""Minimax and regret summary of the held-out three-config comparison.

The claim under test is that connectivity, spatial structure and curation cover each other's failures.
The evidence is that no single configuration wins everywhere, and that the combination has the
best worst case and the smallest gap to whichever configuration was best for a given region.
This script computes that summary.

It refits nothing. Every displacement it uses is already in
``outputs/logs/out_a1b_loro.json``, which holds, for each of the 19 Beauchamp homology pairs,
the displacement of the routed centroid from the expected human homologue under three
configurations, with that region's own curation withheld and the model refitted:

    both        connectivity + anchor-warped space, at production settings
    xyz_only    space only (alpha = 0)
    conn_only   connectivity only, no cross-species feature cost

Definitions. Regret for a region under a configuration is that configuration's displacement
minus the smallest displacement any configuration achieved for that region, so the best
configuration for a region has zero regret there. A configuration's mean regret is the average
over the 19 regions, and is the quantity reported. `n_best` counts the regions where a
configuration is strictly best.

Provenance. This script neither loads nor fits a coupling, so neither ``provenance()`` nor
``refit_provenance()`` describes it. It records the path and sha256 of the log it read instead.
The source log carries no coupling stamp of its own, so the chain terminates there rather than
at a coupling.

    cd otter && python3 experiments/section2_supervision/06_regret.py --check   # compare only
    cd otter && python3 experiments/section2_supervision/06_regret.py           # write the log
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "outputs" / "logs" / "out_a1b_loro.json"
OUT = ROOT / "outputs" / "logs" / "out_g2_regret.json"

CONFIGS = ["both", "xyz_only", "conn_only"]
FAR_MM = 25.0            # chance displacement for the benchmark regions
VERY_FAR_MM = 40.0
BIG_REGRET_MM = 10.0

WHAT = ("Minimax and regret summary of the held-out three-config comparison. Derived from "
        "out_a1b_loro.json by experiments/section2_supervision/06_regret.py; no refitting. "
        "Regret is a configuration's displacement for a region minus the best displacement any "
        "configuration achieved for that region. mean_regret and n_best are the reported values.")


def is_region(value) -> bool:
    """A region entry is a mapping carrying all three configurations.

    Identifying regions by shape rather than by name keeps this producer working when the source
    log gains provenance keys, which sit at the top level beside the regions.
    """
    return isinstance(value, dict) and all(c in value for c in CONFIGS)


def build(src: dict) -> dict:
    regions = [k for k, v in src.items() if is_region(v)]
    if not regions:
        raise SystemExit("no region entries found in the source log")
    disp = [[src[r][c]["cdist_mm"] for c in CONFIGS] for r in regions]
    auroc = [[src[r][c]["auroc"] for c in CONFIGS] for r in regions]

    summary = {}
    for j, config in enumerate(CONFIGS):
        column = [row[j] for row in disp]
        regret = [row[j] - min(row) for row in disp]
        summary[config] = {
            "mean": sum(column) / len(column),
            "worst": max(column),
            "n_gt25": sum(v > FAR_MM for v in column),
            "n_gt40": sum(v > VERY_FAR_MM for v in column),
            "mean_regret": sum(regret) / len(regret),
            "worst_regret": max(regret),
            "n_regret_gt10": sum(v > BIG_REGRET_MM for v in regret),
            "n_best": sum(row.index(min(row)) == j for row in disp),
        }
    return {"_what": WHAT, "configs": CONFIGS, "regions": regions,
            "disp": disp, "auroc": auroc, "summary": summary}


# A mean over 19 values computed by this script and by an independent implementation can differ
# in the last bit because the two summed in a different order. The reported values are quoted to
# one decimal place, so a relative tolerance of 1e-9 is far tighter than anything readable.
FLOAT_RTOL = 1e-9


def differences(a, b, path="") -> list[str]:
    """Every leaf where two nested structures disagree beyond FLOAT_RTOL."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: present in only one")
            else:
                out += differences(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} against {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += differences(x, y, f"{path}[{i}]")
        return out
    if isinstance(a, float) and isinstance(b, float):
        if a == b or math.isclose(a, b, rel_tol=FLOAT_RTOL, abs_tol=0.0):
            return []
        return [f"{path}: {a!r} against {b!r}"]
    return [] if a == b else [f"{path}: {a!r} against {b!r}"]


def max_rel_drift(a, b) -> float:
    """Largest relative difference between two matching structures, for reporting."""
    worst = 0.0
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) & set(b):
            worst = max(worst, max_rel_drift(a[k], b[k]))
    elif isinstance(a, list) and isinstance(b, list):
        for x, y in zip(a, b):
            worst = max(worst, max_rel_drift(x, y))
    elif isinstance(a, float) and isinstance(b, float) and a != 0.0:
        worst = abs(a - b) / abs(a)
    return worst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="compare against the committed log without writing")
    args = ap.parse_args()

    if not SRC.exists():
        print(f"missing input: {SRC.relative_to(ROOT)}", file=sys.stderr)
        return 1

    raw = SRC.read_bytes()
    built = build(json.loads(raw))
    built["source_log"] = str(SRC.relative_to(ROOT))
    built["source_sha256"] = hashlib.sha256(raw).hexdigest()

    s = built["summary"]
    print(f"  {'config':10s} {'mean':>7s} {'worst':>7s} {'>25mm':>6s} "
          f"{'regret':>7s} {'best':>5s}")
    for config in CONFIGS:
        print(f"  {config:10s} {s[config]['mean']:7.2f} {s[config]['worst']:7.2f} "
              f"{s[config]['n_gt25']:6d} {s[config]['mean_regret']:7.2f} "
              f"{s[config]['n_best']:5d}")

    if OUT.exists():
        committed = json.loads(OUT.read_text())
        # source_log/source_sha256 are fields this producer adds that the committed log does not
        # carry, so they are excluded from the comparison rather than counted as a difference.
        derived = ("_what", "source_log", "source_sha256")
        mine = {k: v for k, v in built.items() if k not in derived}
        theirs = {k: v for k, v in committed.items() if k not in derived}
        diffs = differences(theirs, mine)
        if diffs:
            print(f"\nDOES NOT REPRODUCE the committed log, {len(diffs)} difference(s):",
                  file=sys.stderr)
            for d in diffs[:20]:
                print(f"  {d}", file=sys.stderr)
            if len(diffs) > 20:
                print(f"  ... and {len(diffs) - 20} more", file=sys.stderr)
            print("\nNot written; the producer disagrees with the committed log.", file=sys.stderr)
            return 1
        drift = max_rel_drift(theirs, mine)
        print(f"\nreproduces {OUT.relative_to(ROOT)}; "
              f"largest relative difference {drift:.2e} (summation order)")

    if not args.check:
        OUT.write_text(json.dumps(built, indent=2) + "\n")
        print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
