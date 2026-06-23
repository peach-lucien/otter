"""Compare uniform / volume / stability marginals via held-out region CV.

By default, MultimodalFGW uses ``p = 1/n_m`` for every mouse parcel
(uniform). This experiment tests two non-uniform alternatives:

  - **volume**: p_i ∝ # voxels in parcel i  (anatomically natural, bigger
    parcels carry more brain content)
  - **stability**: p_i ∝ bootstrap-stability of row i in the production π
    (more reliable parcels carry more weight)

For each marginal, run leave-one-region-out CV across the 15 atlas-derived
region anchors (S7) and compare top-1 / mean rank.

Usage:
    PYTHONPATH=src python experiments/marginal_weighting/01_compare_marginals.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from homer.data import load_cached                                       # noqa: E402
from homer.data.atlas_regions import build_garin_region_anchors_from_atlases  # noqa: E402
from homer.models import MultimodalFGW                                   # noqa: E402

ANN  = ROOT / "outputs" / "anndata"
COUP = ROOT / "outputs" / "coupling"
LOG  = ROOT / "outputs" / "logs"


def evaluate_region_recovery(pi, mset, hset):
    h_set = set(hset)
    pi_block = pi[mset, :]
    argmax = pi_block.argmax(axis=1)
    in_top1 = np.array([int(int(am) in h_set) for am in argmax])
    pi_to_h = pi[mset][:, hset]
    best = np.asarray(hset)[pi_to_h.argmax(axis=1)]
    ranks = np.array([int(np.where(np.argsort(-pi[mset[i]]) == best[i])[0][0]) + 1
                       for i in range(len(mset))])
    return {"top1": float(in_top1.mean()), "mean_rank": float(ranks.mean()),
            "n_mouse": len(mset), "n_human": len(hset)}


def normalize_to_unit(x: np.ndarray) -> np.ndarray:
    """Sum to 1, all positive."""
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, 1e-9, None)
    return x / x.sum()


def run_held_out_cv(M, H, costs, entries, p, label: str, cache_path: Path):
    """Run held-out region CV using marginal p; cache per-pair results."""
    state = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    section = state.setdefault(label, {})
    n = 0
    for held in entries:
        key = str(held.pair_id)
        if key in section: continue
        visible = [e for e in entries if e.pair_id != held.pair_id]
        t = time.time()
        m = MultimodalFGW(use_sc=True, sc_weight=0.3, fc_weight=0.7,
                           epsilon=5e-3, xyz_weight=0.5, lam_anchor=1.0)
        m.fit(M, H, Cm_SC=costs["Cm_SC"], Ch_SC=costs["Ch_SC"],
              region_anchors=visible, p=p)
        metrics = evaluate_region_recovery(m.pi, held.mouse_indices, held.human_indices)
        metrics["elapsed_s"] = round(time.time()-t, 1)
        section[key] = metrics
        cache_path.write_text(json.dumps(state, indent=2, default=float))
        print(f"  {label:>10s} pid={held.pair_id} top1={metrics['top1']:.0%} "
              f"({metrics['elapsed_s']}s)", flush=True)
        n += 1
    return state[label]


def aggregate(section, regions):
    valid = [(r, section[str(e.pair_id)]) for e in regions if str(e.pair_id) in section]
    if not valid: return None
    n_total = sum(v[1]["n_mouse"] for v in valid)
    top1 = sum(v[1]["top1"] * v[1]["n_mouse"] for v in valid) / n_total
    rank = sum(v[1]["mean_rank"] * v[1]["n_mouse"] for v in valid) / n_total
    return {"top1": top1, "mean_rank": rank, "n_pairs": len(valid), "n_parcels": n_total}


def main():
    M, _ = load_cached("mouse", cache_dir=ANN)
    H, _ = load_cached("human", cache_dir=ANN)
    costs = np.load(ANN / "full_costs.npz")

    print("Building region anchors...")
    entries = build_garin_region_anchors_from_atlases(M.var, H.var)

    n_m = len(M.var)
    p_uniform = np.full(n_m, 1.0 / n_m)

    # Volume-weighted: load parcel sizes
    volumes_path = ANN / "mouse_voxel_counts.npy"
    if not volumes_path.exists():
        raise FileNotFoundError(f"Run the parcel-volume extractor first; need {volumes_path}")
    sizes_m = np.load(volumes_path).astype(np.float64)
    p_volume = normalize_to_unit(sizes_m)

    # Stability-weighted: load bootstrap stability
    boot = np.load(COUP / "bootstrap_aggregate_fc_plus_SC.npz")
    p_stability = normalize_to_unit(boot["per_row_stability"].astype(np.float64))

    print(f"\nMarginal stats:")
    print(f"  uniform:    1/{n_m} = {p_uniform[0]:.6e}")
    print(f"  volume:     range [{p_volume.min():.2e}, {p_volume.max():.2e}], "
          f"effective n_eff = {1.0 / (p_volume**2).sum():.0f}")
    print(f"  stability:  range [{p_stability.min():.2e}, {p_stability.max():.2e}], "
          f"effective n_eff = {1.0 / (p_stability**2).sum():.0f}")

    cache_path = LOG / "marginal_weighting_cv.json"
    print(f"\n=== Running held-out CV for each marginal ===")
    for label, p in [("uniform", p_uniform),
                      ("volume", p_volume),
                      ("stability", p_stability)]:
        print(f"\n[{label}]")
        run_held_out_cv(M, H, costs, entries, p, label, cache_path)

    # Comparison
    state = json.loads(cache_path.read_text())
    print(f"\n=== Aggregate held-out region CV (n=15 pairs, FC+SC) ===")
    print(f"  {'config':>12s}  {'top-1':>7s}  {'mean rank':>9s}")
    for label in ["uniform", "volume", "stability"]:
        agg = aggregate(state[label], entries)
        if agg:
            print(f"  {label:>12s}  {agg['top1']:>7.1%}  {agg['mean_rank']:>9.0f}/2094")

    # Per-region detail
    print(f"\n=== Per-region (uniform / volume / stability top-1) ===")
    print(f"  {'pid':>4s} {'n_m':>4s}  {'uniform':>8s} {'volume':>8s} {'stability':>10s}  label")
    for e in entries:
        row = [state[L].get(str(e.pair_id), {"top1": float("nan")})["top1"]
               for L in ["uniform", "volume", "stability"]]
        n_m = state["uniform"].get(str(e.pair_id), {"n_mouse": 0})["n_mouse"]
        print(f"  {e.pair_id:>4d} {n_m:>4d}  "
              f"{row[0]:>8.0%} {row[1]:>8.0%} {row[2]:>10.0%}  {e.label[:55]}")


if __name__ == "__main__":
    main()
