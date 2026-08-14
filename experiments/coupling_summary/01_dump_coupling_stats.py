
"""Persist every headline coupling statistic so the method's own numbers are checkable.

The coupling claims are recomputed directly from the canonical pi and written to a JSON, so
every number traces to an output file.

Run: cd otter && PYTHONPATH=src python experiments/coupling_summary/01_dump_coupling_stats.py
Writes outputs/logs/coupling_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from otter.data import load_cached, load_pi, pi_provenance  # noqa: E402
from otter.data.anchors import get_anchor_index              # noqa: E402

OUT = ROOT / "outputs/logs/coupling_summary.json"
SEED = 0


# The trust map read here must be the canonical one. Stamping the output with the
# canonical pi while reading a superseded trust map reports tier numbers that do not
# correspond to the coupling named in the log.
def main():
    pi = load_pi()
    M, _ = load_cached("mouse", cache_dir=ROOT / "outputs/anndata")
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    n_m, n_h = pi.shape
    mouse_xyz = M.var[["x", "y", "z"]].to_numpy(float)
    human_xyz = H.var[["x", "y", "z"]].to_numpy(float)

    # --- sharpness of the coupling ------------------------------------------------
    row = pi / np.maximum(pi.sum(1, keepdims=True), 1e-300)     # row-normalised
    top = row.max(1)
    out = {
        "pi_shape": [int(n_m), int(n_h)],
        "top_target_probability": {
            "median": float(np.median(top)),
            "mean": float(top.mean()),
            "fraction_above_0.5": float((top > 0.5).mean()),
        },
    }

    # --- self-correspondence: not computed here ------------------------------------
    # The class-diagonal mean uses a coarse-region assignment and is written to
    # outputs/logs/fig1_coupling_matrix.json (0.262). A nearest-anchor assignment gives a
    # different quantity, so it is not duplicated here.

    # --- spatial accuracy: mouse parcel distance vs routed human centroid distance --
    w = pi / np.maximum(pi.sum(1, keepdims=True), 1e-300)
    routed = w @ human_xyz                                       # routed human centroid
    rng = np.random.default_rng(SEED)
    d_mouse = pdist(mouse_xyz)                                   # all 1,864 parcels, exact
    r_spatial = float(pearsonr(d_mouse, pdist(routed))[0])

    # permuted-coupling null: shuffle pi's rows, recompute
    null = []
    for _ in range(50):
        rn = w[rng.permutation(n_m)] @ human_xyz
        null.append(float(pearsonr(d_mouse, pdist(rn))[0]))
    out["spatial_accuracy"] = {
        "pearson_r": r_spatial,
        "n_parcels": int(n_m),
        "permuted_null_mean": float(np.mean(null)),
        "permuted_null_sd": float(np.std(null)),
    }

    # --- evidence tiers + per-tier recovery ----------------------------------------
    # These live only in an .npz, which no JSON-based check can read, so they are re-emitted here.
    tiers = json.loads((ROOT / "outputs/logs/evidence_tiers_v2.json").read_text())
    n_tier = tiers["n"]
    out["evidence_tiers_percent"] = {
        k: round(100 * v / n_tier, 1) for k, v in tiers["tiers"].items()}
    out["evidence_tiers_percent"]["_n"] = n_tier

    npz = np.load(ROOT / "outputs/coupling/trust_multisource_canonical.npz", allow_pickle=True)
    tier, top1, auroc = npz["evidence_tier"], npz["region_top1"], npz["region_auroc"]
    out["per_tier_recovery"] = {
        t: {"n": int((tier == t).sum()),
            "percent_of_brain": round(100 * float((tier == t).mean()), 1),
            "top1": round(float(np.nanmean(top1[tier == t])), 3),
            "auroc": round(float(np.nanmean(auroc[tier == t])), 3)}
        for t in np.unique(tier)}
    validated = np.isin(tier, ["anchored_and_validated", "validated_only"])
    out["per_tier_recovery"]["_validated_tiers_percent_of_brain"] = round(
        100 * float(validated.mean()), 1)

    out.update(pi_provenance())   # which coupling produced these numbers
    OUT.write_text(json.dumps(out, indent=2))
    print(f"pi {n_m} x {n_h}")
    print(f"  top-target prob: median {out['top_target_probability']['median']:.2f}, "
          f"{out['top_target_probability']['fraction_above_0.5']:.1%} of parcels > 0.5")
    print("  self-mass: see outputs/logs/fig1_coupling_matrix.json (0.262)")
    print(f"  spatial accuracy r = {r_spatial:.2f} (permuted null "
          f"{out['spatial_accuracy']['permuted_null_mean']:+.2f})")
    print(f"  tiers %: {out['evidence_tiers_percent']}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
