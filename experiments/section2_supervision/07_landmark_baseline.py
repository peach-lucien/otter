#!/usr/bin/env python3
"""The simplest model that could work, as a floor for the ablation ladder.

Every arm of the ablation ladder solves a transport problem. Even the alpha = 0 arm,
which carries no connectivity, still solves an entropic semi-relaxed transport problem with a
spatial cost and curated anchors, so the ladder cannot answer whether the transport machinery is
needed at all.

This baseline does. It takes the 42 bilateral Garin landmark pairs, fits the same
thin-plate-spline warp used by the canonical recipe, pushes the mouse
parcel centroids onto the human brain, and assigns each mouse parcel to the nearest human parcel.
No connectivity, no region packs, no transport, no free parameters.

Scoring. Top-1, mass-in-region and centroid displacement are read from the hard assignment and are
directly comparable to the ladder. AUROC is read from the negated warped distance rather than from
the hard assignment, because AUROC ranks all 2,094 human parcels and a one-hot row would be
penalised for its shape rather than for its accuracy. The negated distance is the whole of what
this baseline knows, and it introduces no tuning constant.

Two arms are scored:

    full        all 42 landmarks, against the six ladder arms in out_a1_ladder.json
    heldout     each region's own Garin landmark withheld from the warp before refitting it,
                against the three configurations in out_a1b_loro.json

The held-out arm is the informative one. Connectivity is expected to matter most for regions whose
cross-species position is misleading, so this baseline should fail hardest in the tectum and the
hippocampal subfields.

Writes outputs/logs/out_b2_landmark_baseline.json.

    conda activate retune
    cd otter && python3 experiments/section2_supervision/07_landmark_baseline.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]                       # .../otter
sys.path.insert(0, str(ROOT / "src"))

from otter.repro import beauchamp_scorer, load_inputs, warped_mouse_xyz   # noqa: E402

OUT = ROOT / "outputs" / "logs" / "out_b2_landmark_baseline.json"
LADDER = ROOT / "outputs" / "logs" / "out_a1_ladder.json"
HELDOUT = ROOT / "outputs" / "logs" / "out_a1b_loro.json"

WHAT = ("Landmark-only baseline for the ablation ladder. The 42 bilateral Garin anchors define a "
        "thin-plate-spline warp; mouse parcel centroids are pushed through it and assigned to the "
        "nearest human parcel. No connectivity, no region packs, no transport. Top-1, "
        "mass_in_region and centroid_disp_mm come from the hard assignment. AUROC comes from the "
        "negated warped distance, because a one-hot row would be penalised by a ranking metric "
        "for its shape rather than its accuracy. Produced by "
        "experiments/section2_supervision/07_landmark_baseline.py.")


def assign(mouse_warped: np.ndarray, human_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest human parcel for every mouse parcel. Returns (hard coupling, distance matrix mm)."""
    d = np.sqrt(((mouse_warped[:, None, :] - human_xyz[None, :, :]) ** 2).sum(-1))
    pi = np.zeros(d.shape, dtype=np.float64)
    pi[np.arange(d.shape[0]), d.argmin(1)] = 1.0
    return pi, d


def score(pi, d, m_mask, h_mask, h_xyz) -> dict:
    """The ladder's four metrics, with AUROC taken from the distance rather than the assignment."""
    hidx = np.where(h_mask)[0]
    true_c = h_xyz[hidx].mean(0)
    block = pi[m_mask]
    tot = block.sum(0)
    s = tot.sum()
    proximity = -d[m_mask].mean(0)          # graded score: closer is better
    out = {
        "auroc": float(roc_auc_score(h_mask, proximity)),
        "rand_mm": float(np.linalg.norm(h_xyz - true_c[None, :], axis=1).mean()),
    }
    if s <= 0:
        return {**out, "centroid_disp_mm": float("nan"), "mass_in_region": 0.0, "top1": 0.0}
    totn = tot / s
    pred_c = (totn[:, None] * h_xyz).sum(0)
    return {**out,
            "centroid_disp_mm": float(np.linalg.norm(pred_c - true_c)),
            "mass_in_region": float(tot[hidx].sum() / s),
            "top1": float(np.isin(block.argmax(1), hidx).mean())}


def aggregate(per_region: dict, weights: dict) -> dict:
    """Parcel-weighted means, matching how beauchamp_battery aggregates the ladder."""
    keys = list(per_region)
    w = np.array([weights[k] for k in keys], dtype=float)
    return {m: float(np.average([per_region[k][m] for k in keys], weights=w))
            for m in ("auroc", "top1", "mass_in_region", "centroid_disp_mm")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="score without writing")
    args = ap.parse_args()

    M, H, _, _ = load_inputs()
    BB = beauchamp_scorer()
    pairs, _, _, h_xyz, _, _ = BB.build(M, H)
    apid_m = pd.to_numeric(M.var["anchor_pair_id"], errors="coerce").to_numpy(
        dtype="float64", na_value=np.nan)
    garin_M = M.var["garin_anchor"].fillna(False).to_numpy().astype(bool)

    # ---- full arm: every landmark present -------------------------------------------------
    pi, d = assign(warped_mouse_xyz(M, H), h_xyz)
    n_landmarks = int((garin_M & np.isfinite(apid_m)).sum())
    full, weights = {}, {}
    for key, (m_mask, h_mask) in pairs.items():
        full[key] = score(pi, d, m_mask, h_mask, h_xyz)
        weights[key] = int(m_mask.sum())
    full_agg = aggregate(full, weights)
    print(f"full arm ({n_landmarks} mouse landmark parcels): "
          f"AUROC={full_agg['auroc']:.3f} top1={full_agg['top1']:.3f} "
          f"disp={full_agg['centroid_disp_mm']:.1f}mm")

    # ---- held-out arm: the region's own landmark withheld from the warp ---------------------
    heldout = {}
    print("\nheld-out, each region's own Garin landmark removed before refitting the warp:")
    for key, (m_mask, h_mask) in pairs.items():
        finite = np.isfinite(apid_m)
        drop = set(apid_m[garin_M & m_mask & finite].astype(int).tolist())
        pi_ho, d_ho = assign(warped_mouse_xyz(M, H, drop_pairs=drop), h_xyz)
        heldout[key] = {**score(pi_ho, d_ho, m_mask, h_mask, h_xyz),
                        "dropped_pair_ids": sorted(drop)}
        print(f"  {key.split(' -> ')[0][:26]:26s} disp={heldout[key]['centroid_disp_mm']:6.1f}mm"
              f"  (dropped {sorted(drop) or 'nothing'})", flush=True)
    heldout_agg = aggregate({k: {m: v[m] for m in
                                 ("auroc", "top1", "mass_in_region", "centroid_disp_mm")}
                             for k, v in heldout.items()}, weights)

    payload = {
        "_what": WHAT,
        "_recipe": {"warp": "thin_plate_spline RBFInterpolator, smoothing 1e-3",
                    "landmarks": "bilateral Garin anchor pairs matched across species",
                    "n_mouse_landmark_parcels": n_landmarks,
                    "assignment": "nearest human parcel to the warped mouse centroid",
                    "coupling_used": None,
                    "transport_solved": False},
        "full": {"per_region": full, "aggregate": full_agg},
        "heldout": {"per_region": heldout, "aggregate": heldout_agg},
    }

    # Comparators, so the panel and the text read from one file. Recorded by sha, because these
    # are the numbers the baseline is being set against.
    for label, path in (("ladder", LADDER), ("heldout_three_config", HELDOUT)):
        if path.exists():
            raw = path.read_bytes()
            payload.setdefault("_compared_against", {})[label] = {
                "log": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(raw).hexdigest()}

    if LADDER.exists():
        ladder = json.loads(LADDER.read_text())
        print("\nagainst the ladder, parcel-weighted:")
        print(f"  {'arm':32s} {'AUROC':>7s} {'top-1':>7s} {'disp mm':>8s}")
        for arm in sorted(k for k in ladder if isinstance(ladder[k], dict)
                          and "auroc" in ladder[k]):
            a = ladder[arm]
            print(f"  {arm:32s} {a['auroc']:7.3f} {a['top1']:7.3f} "
                  f"{a['centroid_disp_mm']:8.1f}")
        print(f"  {'0_landmark_only_BASELINE':32s} {full_agg['auroc']:7.3f} "
              f"{full_agg['top1']:7.3f} {full_agg['centroid_disp_mm']:8.1f}")

    if HELDOUT.exists():
        ho = json.loads(HELDOUT.read_text())
        regions = [k for k, v in ho.items()
                   if isinstance(v, dict) and all(c in v for c in ("both", "xyz_only"))]
        both = float(np.mean([ho[r]["both"]["cdist_mm"] for r in regions]))
        print(f"\nheld out, unweighted mean displacement over {len(regions)} regions:")
        print(f"  full model (both)        {both:6.2f} mm")
        print(f"  landmark-only baseline   "
              f"{np.mean([heldout[r]['centroid_disp_mm'] for r in regions]):6.2f} mm")

    if not args.check:
        OUT.write_text(json.dumps(payload, indent=2, default=float) + "\n")
        print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
