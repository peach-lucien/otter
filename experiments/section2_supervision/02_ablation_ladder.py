#!/usr/bin/env python3
"""Decompose the coupling by ablating its cost terms.

Fits the model six times, adding one source of information at a time, and scores each arm against
the 19 held-out Beauchamp region correspondences. The relational term on the two connectomes is
scored alone, then extended by the anchor-warped spatial scaffold, then the Garin point anchors, then
the curated region packs, which gives the production coupling. Two further arms drop the connectivity
term entirely, to separate what connectivity contributes from what position contributes. Results
section 2 and Figure 2a; Extended Data Fig. 3 uses the alpha=0 arms.

Each arm's coupling is written to outputs/coupling/pi_ladder_<arm>.npy, because
03_downstream_by_arm.py and 04_gradient_components.py score the same arms on downstream measures and
must use the identical couplings rather than refitting them.

Writes outputs/logs/out_a1_ladder.json.

    conda activate retune
    cd otter && python3 experiments/section2_supervision/02_ablation_ladder.py

Six fits. Expect this to take a while; --arms restricts it to a subset for a partial rerun.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]                       # .../otter
sys.path.insert(0, str(ROOT / "src"))

from otter.repro import (anchor_warped_xyz, beauchamp_scorer, fit_coupling,   # noqa: E402
                         load_inputs, refit_provenance, stamp)

OUT = ROOT / "outputs" / "logs" / "out_a1_ladder.json"
COUPLINGS = ROOT / "outputs" / "coupling"
METRICS = ("auroc", "top1", "mass_in_region", "centroid_disp_mm")

# The rungs, in the order Figure 2a plots them. Names are the keys used by the downstream scripts.
ARMS: dict[str, dict] = {
    "1_connectivity_only":         dict(alpha=0.5, xyz_weight=0.0,  garin=False, packs=False),
    "2_+spatial":                  dict(alpha=0.5, xyz_weight=0.25, garin=False, packs=False),
    "3_+anchors":                  dict(alpha=0.5, xyz_weight=0.25, garin=True,  packs=False),
    "4_+packs_CANONICAL":          dict(alpha=0.5, xyz_weight=0.25, garin=True,  packs=True),
    "5_NOCONN_spatial_only":       dict(alpha=0.0, xyz_weight=0.25, garin=False, packs=False),
    "6_NOCONN_spatial+anch+packs": dict(alpha=0.0, xyz_weight=0.25, garin=True,  packs=True),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", choices=sorted(ARMS), default=sorted(ARMS),
                    help="subset of arms to fit; the log merges with any previous run")
    ap.add_argument("--force", action="store_true",
                    help="refit an arm even if its coupling is already on disk")
    args = ap.parse_args()

    M, H, costs, packs = load_inputs()
    M_xyz = anchor_warped_xyz(M, H)
    BB = beauchamp_scorer()
    pairs, reg_cents, reg_masks, h_xyz, brain_c, _ = BB.build(M, H)

    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    COUPLINGS.mkdir(parents=True, exist_ok=True)

    for arm in args.arms:
        config = ARMS[arm]
        target = COUPLINGS / f"pi_ladder_{arm}.npy"
        start = time.time()

        if target.exists() and not args.force:
            pi = np.load(target).astype(np.float64)
            note = "loaded"
        else:
            pi = fit_coupling(M, H, costs, packs, M_xyz, **config)
            np.save(target, pi.astype(np.float32))
            note = "fitted"

        agg = BB.score_all(pi, pairs, reg_cents, reg_masks, h_xyz, brain_c)["aggregate"]
        out[arm] = {**{m: float(agg[m]) for m in METRICS},
                    "cfg": config, "secs": round(time.time() - start),
                    "coupling": target.name,
                    **refit_provenance(pi, recipe=config)}
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(out, indent=2))

        print(f"{arm:30s} AUROC={agg['auroc']:.3f} top1={agg['top1']:.3f} "
              f"mass={agg['mass_in_region']:.3f} disp={agg['centroid_disp_mm']:.1f}mm  "
              f"({note}, {time.time() - start:.0f}s)", flush=True)

    # Region-level AUROC falls slightly across the last two rungs while parcel-exact recovery rises.
    # Several packs subdivide a Beauchamp region into sub-targets outside its broad validation ball,
    # so the anatomy sharpens while the benchmark metric stays coarse. Figure 2a's caption says this.
    if {"3_+anchors", "4_+packs_CANONICAL"} <= set(out):
        d_auroc = out["4_+packs_CANONICAL"]["auroc"] - out["3_+anchors"]["auroc"]
        d_top1 = out["4_+packs_CANONICAL"]["top1"] - out["3_+anchors"]["top1"]
        print(f"\npacks rung: AUROC {d_auroc:+.3f}, top-1 {d_top1:+.3f}")

    print(f"\nwrote {OUT.relative_to(ROOT)}")
    print(f"arm couplings in {COUPLINGS.relative_to(ROOT)}/pi_ladder_*.npy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
