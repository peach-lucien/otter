"""Sweep over the soft-region-anchor penalty (``lam_outside``).

Tests whether **soft** region anchors (mild penalty outside the region instead
of a hard 0/1 wall) give better held-out region recovery than the current
hard formulation. The hard version forces the optimizer to satisfy the
region constraint exactly, which is harmful when atlas regions overlap or
the Garin anchor is mis-placed.

Reports per-config held-out top-1, top-5, top-10, and mean rank, so the soft
penalty can be picked by the metric appropriate to a soft probabilistic
mapping (top-K, not top-1).

Sweep values: {0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}.
1.0 reproduces the hard behaviour and serves as the baseline.

Usage:
    PYTHONPATH=src python experiments/ablations/soft_region_anchors.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from otter.data import load_cached                                      # noqa: E402
from otter.data.atlas_regions import build_garin_region_anchors_from_atlases  # noqa: E402
from otter.models import MultimodalFGW                                   # noqa: E402

ANN = ROOT / "outputs" / "anndata"
COUP = ROOT / "outputs" / "coupling"
LOG = ROOT / "outputs" / "logs"; LOG.mkdir(parents=True, exist_ok=True)


SWEEP_VALUES = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]


def evaluate_region_recovery_topk(pi, mset, hset):
    """Top-1, top-5, top-10, mean rank in (mset → hset)."""
    h_set = set(hset)
    pi_block = pi[mset, :]
    top10 = np.argsort(-pi_block, axis=1)[:, :10]
    top5 = top10[:, :5]
    in_top1 = np.array([int(int(t[0]) in h_set) for t in top10])
    in_top5 = np.array([int(any(int(t) in h_set for t in row)) for row in top5])
    in_top10 = np.array([int(any(int(t) in h_set for t in row)) for row in top10])
    pi_to_h = pi[mset][:, hset]
    best = np.asarray(hset)[pi_to_h.argmax(axis=1)]
    ranks = np.array([
        int(np.where(np.argsort(-pi[mset[i]]) == best[i])[0][0]) + 1
        for i in range(len(mset))
    ])
    return {
        "top1":      float(in_top1.mean()),
        "top5":      float(in_top5.mean()),
        "top10":     float(in_top10.mean()),
        "mean_rank": float(ranks.mean()),
        "n_mouse":   len(mset),
        "n_human":   len(hset),
    }


def run_sweep(M, H, costs, entries, lam_out: float, cache_path: Path):
    """Run held-out region CV at a given lam_outside; cache per-pid."""
    state = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    key = f"lam_out_{lam_out:.2f}"
    section = state.setdefault(key, {})
    for held in entries:
        pkey = str(held.pair_id)
        if pkey in section: continue
        visible = [e for e in entries if e.pair_id != held.pair_id]
        t = time.time()
        m = MultimodalFGW(use_sc=True, sc_weight=0.3, fc_weight=0.7,
                           epsilon=5e-3, xyz_weight=0.5, lam_anchor=1.0)
        m.fit(M, H, Cm_SC=costs["Cm_SC"], Ch_SC=costs["Ch_SC"],
              region_anchors=visible, region_lam_outside=lam_out)
        metrics = evaluate_region_recovery_topk(m.pi, held.mouse_indices, held.human_indices)
        metrics["elapsed_s"] = round(time.time() - t, 1)
        section[pkey] = metrics
        cache_path.write_text(json.dumps(state, indent=2, default=float))
        print(f"  lam_out={lam_out:.2f} pid={held.pair_id} "
              f"top1={metrics['top1']:.0%} top5={metrics['top5']:.0%} "
              f"top10={metrics['top10']:.0%}  ({metrics['elapsed_s']}s)", flush=True)
    return state[key]


def aggregate(section):
    if not section: return None
    n = sum(v["n_mouse"] for v in section.values())
    return {
        "top1":  sum(v["top1"]  * v["n_mouse"] for v in section.values()) / n,
        "top5":  sum(v["top5"]  * v["n_mouse"] for v in section.values()) / n,
        "top10": sum(v["top10"] * v["n_mouse"] for v in section.values()) / n,
        "mean_rank": sum(v["mean_rank"] * v["n_mouse"] for v in section.values()) / n,
        "n_parcels": n,
        "n_pairs":   len(section),
    }


def main():
    M, _ = load_cached("mouse", cache_dir=ANN)
    H, _ = load_cached("human", cache_dir=ANN)
    costs = np.load(ANN / "full_costs.npz")
    print("Building region anchors...")
    entries = build_garin_region_anchors_from_atlases(M.var, H.var)

    cache_path = LOG / "soft_region_anchors_sweep.json"
    for lam_out in SWEEP_VALUES:
        print(f"\n[lam_outside = {lam_out:.2f}]")
        run_sweep(M, H, costs, entries, lam_out, cache_path)

    state = json.loads(cache_path.read_text())
    print(f"\n=== Aggregate held-out region CV across lam_outside sweep ===")
    print(f"  {'lam_out':>8s}  {'top-1':>7s}  {'top-5':>7s}  {'top-10':>7s}  {'mean rank':>9s}")
    print('-' * 65)
    for lam_out in SWEEP_VALUES:
        section = state.get(f"lam_out_{lam_out:.2f}", {})
        agg = aggregate(section)
        if agg is None: continue
        tag = "  (hard, baseline)" if lam_out == 1.0 else ""
        print(f"  {lam_out:>8.2f}  {agg['top1']:>7.1%}  {agg['top5']:>7.1%}  "
              f"{agg['top10']:>7.1%}  {agg['mean_rank']:>9.0f}{tag}")


if __name__ == "__main__":
    main()
