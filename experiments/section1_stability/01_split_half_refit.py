#!/usr/bin/env python3
"""Refit the coupling on each half of the resting-state cohorts.

Tests whether the coupling depends on the particular subjects it was fitted on. Each species cohort
is split in two, connectivity is averaged within each half, and the full model is refitted on each
half in turn. Recovery on the held-out Beauchamp benchmark and agreement between the two couplings
are reported. Results section 1.

Structural connectivity, the spatial warp and the curation are held fixed across arms, so the only
thing that varies is which subjects contributed to functional connectivity.

Requires outputs/splithalf/*.npz from pipeline/02b_build_splithalf_fc.py.
Writes outputs/logs/out_a2_splithalf.json.

    conda activate retune
    cd homer && python3 experiments/section1_stability/01_split_half_refit.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]                       # .../homer
sys.path.insert(0, str(ROOT / "src"))

from homer.costs.normalisation import normalise_cost                          # noqa: E402
from homer.costs.relational import correlation_distance                       # noqa: E402
from homer.repro import (anchor_warped_xyz, beauchamp_scorer, fit_coupling,   # noqa: E402
                         load_canonical, load_inputs, refit_provenance, stamp)

SPLITHALF = ROOT / "outputs" / "splithalf"
OUT = ROOT / "outputs" / "logs" / "out_a2_splithalf.json"
METRICS = ("auroc", "top1", "mass_in_region", "centroid_disp_mm")


def fc_cost(fc: np.ndarray) -> np.ndarray:
    """Connectivity matrix to the correlation-distance cost the model expects."""
    return normalise_cost(correlation_distance(np.nan_to_num(fc.astype(np.float64))), scheme="max")


def agreement(p: np.ndarray, q: np.ndarray) -> dict:
    """How far two couplings agree, by top partner and entrywise."""
    return {"argmax_match": float((p.argmax(1) == q.argmax(1)).mean()),
            "entrywise_r": float(np.corrcoef(p.ravel(), q.ravel())[0, 1])}


def main() -> int:
    for species in ("human", "mouse"):
        if not (SPLITHALF / f"{species}_splithalf.npz").exists():
            print(f"missing {SPLITHALF / f'{species}_splithalf.npz'}.\n"
                  f"Run pipeline/02b_build_splithalf_fc.py first.", file=sys.stderr)
            return 1

    M, H, costs, packs = load_inputs()
    M_xyz = anchor_warped_xyz(M, H)
    BB = beauchamp_scorer()
    pairs, reg_cents, reg_masks, h_xyz, brain_c, _ = BB.build(M, H)

    human = np.load(SPLITHALF / "human_splithalf.npz")
    mouse = np.load(SPLITHALF / "mouse_splithalf.npz")

    out: dict = {"n_subj": {"human": int(human["n_subj"]), "mouse": int(mouse["n_subj"])}}

    # How reproducible the connectivity itself is, before any coupling is fitted. This is the
    # ceiling against which the coupling's own stability should be read.
    for species, half in (("human", human), ("mouse", mouse)):
        both = np.isfinite(half["A"]) & np.isfinite(half["B"])
        out[f"{species}_fc_halfhalf_r"] = float(np.corrcoef(half["A"][both], half["B"][both])[0, 1])
        print(f"{species} FC half-half r = {out[f'{species}_fc_halfhalf_r']:.4f}")

    couplings = {}
    for half in ("A", "B"):
        start = time.time()
        couplings[half] = fit_coupling(M, H, costs, packs, M_xyz,
                                       Cm_FC=fc_cost(mouse[half]), Ch_FC=fc_cost(human[half]))
        agg = BB.score_all(couplings[half], pairs, reg_cents, reg_masks, h_xyz, brain_c)["aggregate"]
        out[f"half_{half}"] = {k: float(agg[k]) for k in METRICS}
        print(f"half {half}: AUROC={agg['auroc']:.3f} top1={agg['top1']:.3f} "
              f"mass={agg['mass_in_region']:.3f} disp={agg['centroid_disp_mm']:.1f}mm "
              f"({time.time() - start:.0f}s)", flush=True)

    pi_canonical, _ = load_canonical()
    out["A_vs_B"] = agreement(couplings["A"], couplings["B"])
    out["A_vs_canonical"] = agreement(couplings["A"], pi_canonical)
    out["B_vs_canonical"] = agreement(couplings["B"], pi_canonical)
    agg = BB.score_all(pi_canonical, pairs, reg_cents, reg_masks, h_xyz, brain_c)["aggregate"]
    out["canonical"] = {k: float(agg[k]) for k in METRICS}

    for label in ("A_vs_B", "A_vs_canonical", "B_vs_canonical"):
        print(f"{label:18s} argmax {out[label]['argmax_match']:.3f}  "
              f"entrywise r {out[label]['entrywise_r']:.4f}")

    # Both arms are refits rather than loads, so the stamp records the recipe and the measured
    # distance from the release rather than borrowing its sha.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(stamp(out, **refit_provenance(couplings["A"])), indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
