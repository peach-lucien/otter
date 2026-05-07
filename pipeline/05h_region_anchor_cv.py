"""Pipeline 05h — held-out region-anchor cross-validation.

The honest evaluation for region anchors. For each region anchor we test:
hold it out (don't pass it to the solver as a constraint), solve with the
remaining anchors, then score recovery: do mouse parcels in the held-out
region's mouse-set argmax to parcels in the held-out region's human-set?

If the answer is "yes" without supervision → the model has captured the
homology from FC/SC structure (real generalisation).
If the answer is "no" without supervision → the supervision is what was
giving us the appearance of recovery (which is fine — we built region
anchors with that purpose — but the headline claim should be qualified).

This avoids the circularity exposed in S4: declaring + testing the same
region gives 100% by construction. Held-out CV gives the honest number.

Usage:
    PYTHONPATH=src python pipeline/05h_region_anchor_cv.py
    PYTHONPATH=src python pipeline/05h_region_anchor_cv.py --base-config keep-points
        # also keep the 21 Garin point anchors visible during training
        # (recommended; otherwise the held-out region has no nearby anchors)

Output: outputs/logs/region_anchor_cv.json with per-pid hold-out metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from homer.data import load_cached                                      # noqa: E402
from homer.data.atlas_regions import build_garin_region_anchors_from_atlases  # noqa: E402
from homer.data.region_anchors import RegionAnchorEntry                 # noqa: E402
from homer.models import MultimodalFGW                                   # noqa: E402

ANN  = ROOT / "outputs" / "anndata"
COUP = ROOT / "outputs" / "coupling"
LOG  = ROOT / "outputs" / "logs"; LOG.mkdir(parents=True, exist_ok=True)


def evaluate_region_recovery(
    pi: np.ndarray, mouse_set: list[int], human_set: list[int], n_h: int
) -> dict:
    """For each mouse parcel in the region's mouse_set, does its argmax fall
    in the region's human_set? Return aggregate metrics."""
    if not mouse_set or not human_set:
        return {"top1": float("nan"), "top5": float("nan"), "top10": float("nan"),
                 "mean_rank": float("nan"), "n_mouse": 0, "n_human": 0}
    h_set = set(human_set)
    pi_block = pi[mouse_set, :]
    argmax_h = pi_block.argmax(axis=1)
    in_h_top1 = np.array([int(int(am) in h_set) for am in argmax_h])
    top5_lists = np.argsort(-pi_block, axis=1)[:, :5]
    top10_lists = np.argsort(-pi_block, axis=1)[:, :10]
    in_h_top5 = np.array([int(any(int(t) in h_set for t in tops)) for tops in top5_lists])
    in_h_top10 = np.array([int(any(int(t) in h_set for t in tops)) for tops in top10_lists])
    # Rank of best in-region human parcel
    pi_to_h = pi[mouse_set][:, human_set]
    best_within = np.asarray(human_set)[pi_to_h.argmax(axis=1)]
    ranks = np.array([
        int(np.where(np.argsort(-pi[mouse_set[i]]) == best_within[i])[0][0]) + 1
        for i in range(len(mouse_set))
    ])
    return {
        "top1":      float(in_h_top1.mean()),
        "top5":      float(in_h_top5.mean()),
        "top10":     float(in_h_top10.mean()),
        "mean_rank": float(ranks.mean()),
        "n_mouse":   len(mouse_set),
        "n_human":   len(human_set),
    }


def main(args):
    M, _ = load_cached("mouse", cache_dir=ANN)
    H, _ = load_cached("human", cache_dir=ANN)
    costs = np.load(ANN / "full_costs.npz")
    n_h = pi_n_h = len(H.var)

    print(f"Building region anchors from atlases...")
    all_entries = build_garin_region_anchors_from_atlases(M.var, H.var)
    if not all_entries:
        print("No region anchors built — can't run hold-out CV. Exiting.")
        sys.exit(1)
    print(f"Have {len(all_entries)} region anchors to hold out one at a time.")

    # Cache: outputs/logs/region_anchor_cv.json
    cache_path = LOG / "region_anchor_cv.json"
    state = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    # Solve once with all region anchors visible (full-train baseline)
    if "_full_train" not in state or args.recompute:
        print(f"\n[full-train] solving with all {len(all_entries)} region anchors visible...")
        t = time.time()
        m_full = MultimodalFGW(use_sc=True, sc_weight=0.3, fc_weight=0.7,
                                epsilon=5e-3, xyz_weight=0.5, lam_anchor=1.0)
        m_full.fit(M, H, Cm_SC=costs["Cm_SC"], Ch_SC=costs["Ch_SC"],
                   region_anchors=all_entries)
        full_results = {}
        for e in all_entries:
            full_results[str(e.pair_id)] = evaluate_region_recovery(
                m_full.pi, e.mouse_indices, e.human_indices, n_h)
        state["_full_train"] = {
            "elapsed_s": round(time.time()-t, 1),
            "loss": float(m_full.fit_info_.loss),
            "per_region": full_results,
        }
        cache_path.write_text(json.dumps(state, indent=2, default=float))
        print(f"  full-train elapsed: {state['_full_train']['elapsed_s']}s")
    else:
        print(f"\n[full-train] cached")

    # Hold-out CV: for each region, solve without it, evaluate it
    print(f"\n[held-out CV] solving {len(all_entries)} times, holding out each region once...")
    held_out = state.setdefault("_held_out", {})
    for h_idx, held in enumerate(all_entries):
        key = str(held.pair_id)
        if key in held_out and not args.recompute:
            r = held_out[key]
            print(f"  {h_idx+1}/{len(all_entries)} pid={held.pair_id} {held.label[:40]} cached: "
                  f"top1={r['top1']:.0%}")
            continue
        t = time.time()
        visible = [e for e in all_entries if e.pair_id != held.pair_id]
        m = MultimodalFGW(use_sc=True, sc_weight=0.3, fc_weight=0.7,
                           epsilon=5e-3, xyz_weight=0.5, lam_anchor=1.0)
        m.fit(M, H, Cm_SC=costs["Cm_SC"], Ch_SC=costs["Ch_SC"],
               region_anchors=visible)
        metrics = evaluate_region_recovery(
            m.pi, held.mouse_indices, held.human_indices, n_h)
        metrics["elapsed_s"] = round(time.time()-t, 1)
        metrics["label"] = held.label[:80]
        held_out[key] = metrics
        cache_path.write_text(json.dumps(state, indent=2, default=float))
        print(f"  {h_idx+1}/{len(all_entries)} pid={held.pair_id} {held.label[:40]:40s} "
              f"top1={metrics['top1']:.0%} top5={metrics['top5']:.0%} top10={metrics['top10']:.0%} "
              f"rank={metrics['mean_rank']:.0f}/{n_h}  ({metrics['elapsed_s']}s)", flush=True)

    # Aggregate (held-out)
    print(f"\n=== Aggregate held-out region-anchor CV ===")
    print(f"{'pair_id':>8s}  {'label':<60s}  {'top1':>6s} {'top5':>6s} {'top10':>6s} {'rank':>8s}  {'held-in':>8s}")
    print('-'*120)
    held_top1, held_top5, held_top10 = [], [], []
    held_w = []
    full = state["_full_train"]["per_region"]
    for e in all_entries:
        key = str(e.pair_id)
        ho = held_out[key]
        fi = full[key]
        if not np.isnan(ho["top1"]):
            held_top1.append(ho["top1"]); held_top5.append(ho["top5"]); held_top10.append(ho["top10"])
            held_w.append(ho["n_mouse"])
        print(f"  {e.pair_id:>8d}  {e.label[:58]:<60s}  "
              f"{ho['top1']:>6.0%} {ho['top5']:>6.0%} {ho['top10']:>6.0%} "
              f"{ho['mean_rank']:>8.0f}  {fi['top1']:>8.0%}")
    if held_top1:
        w = np.asarray(held_w, dtype=float); wsum = w.sum()
        print(f"\nWeighted aggregate (held-out, n_pairs={len(held_top1)}, n_parcels={int(wsum)}):")
        print(f"  top1  = {sum(w*held_top1)/wsum:.1%}")
        print(f"  top5  = {sum(w*held_top5)/wsum:.1%}")
        print(f"  top10 = {sum(w*held_top10)/wsum:.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true", help="recompute even if cached")
    main(ap.parse_args())
