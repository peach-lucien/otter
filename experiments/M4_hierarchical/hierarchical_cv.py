"""M4 evaluation: leave-one-network-out CV using hierarchical FGW.

Compares hierarchical (per-network sub-FGW) vs flat (single 1864×2094 FGW)
on the same CV folds + the FC translation quality metric.

Saves:
    outputs/logs/hierarchical_cv.json    per-network CV metrics for hierarchical
    outputs/logs/fc_translation.json     adds 'hierarchical_fc_only' entry
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import ot

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from homer.data import load_cached                                              # noqa: E402
from homer.data.anchors import (                                                 # noqa: E402
    get_anchor_index, held_out_metrics_graded,
)
from homer.data.networks import PAIRID_TO_NETWORK, NETWORKS                       # noqa: E402
from homer.models.hierarchical import hierarchical_semirelaxed_fgw              # noqa: E402
from homer.eval.translation import fc_translation_quality                       # noqa: E402

ANN = ROOT / "outputs" / "anndata"
LOG = ROOT / "outputs" / "logs"; LOG.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore")


def main(args):
    H, _ = load_cached("human", cache_dir=ANN)
    M_, _ = load_cached("mouse", cache_dir=ANN)
    idx_h = get_anchor_index(H.var); idx_m = get_anchor_index(M_.var)
    n_m, n_h = M_.uns["n_nodes"], H.uns["n_nodes"]

    d = np.load(ANN / "full_costs.npz")
    Cm = d["Cm"].astype(np.float32); Ch = d["Ch"].astype(np.float32)
    M_xyz = d["M_xyz"].astype(np.float32)

    net_to_pairs = {n: [] for n in NETWORKS}
    for pid, name in PAIRID_TO_NETWORK.items():
        net_to_pairs[name].append(pid)

    cache = LOG / "hierarchical_cv.json"
    state = json.loads(cache.read_text()) if cache.exists() else {}

    nets_to_run = (args.networks.split(",") if args.networks else NETWORKS)

    if args.mode == "cv":
        # Leave-one-network-out CV
        for net_name in nets_to_run:
            if net_name in state and not args.recompute:
                r = state[net_name]
                print(f"  {net_name:15s} cached: top1={r['top1']:.0%} pair={r['pair_id']:.0%}")
                continue
            held = sorted(net_to_pairs[net_name])
            visible = sorted([pid for pid in PAIRID_TO_NETWORK if pid not in held])
            t = time.time()
            pi, info = hierarchical_semirelaxed_fgw(
                Cm, Ch, M_xyz, idx_m, idx_h, M_.var, H.var,
                visible_pair_ids=visible,
                alpha=0.5, epsilon=5e-3, lam_anchor=1.0, xyz_w=0.5,
                max_iter=25, tol=1e-5, verbose=False,
            )
            elapsed = time.time() - t

            pi_anchor = pi[np.ix_(idx_m.pos, idx_h.pos)].astype(np.float64)
            graded = held_out_metrics_graded(pi_anchor, idx_m, idx_h, held, var_h=H.var)
            n_visible_in_net = info["per_network"].get(net_name, {}).get("n_anchors_visible", 0)
            n_total_in_net   = info["per_network"].get(net_name, {}).get("n_anchors_total", 0)

            state[net_name] = {
                "n_anchors_held":  graded["n"],
                "n_pair_ids_held": len(held),
                "top1":            graded["top1"],
                "top5":            graded["top5"],
                "pair_id":         graded["pair_id"],
                "hemi":            graded["hemisphere"],
                "mean_rank":       graded["mean_rank"],
                "mean_xyz_dist":   graded.get("mean_xyz_dist", float("nan")),
                "n_visible_anchors_in_held_net":  n_visible_in_net,
                "n_total_anchors_in_held_net":    n_total_in_net,
                "elapsed":         round(elapsed, 1),
            }
            cache.write_text(json.dumps(state, indent=2, default=float))
            print(f"  {net_name:15s} (n={graded['n']}): "
                  f"top1={graded['top1']:.0%} top5={graded['top5']:.0%} "
                  f"pair={graded['pair_id']:.0%} hemi={graded['hemisphere']:.0%}  "
                  f"rank={graded['mean_rank']:.1f}/{graded.get('max_rank_possible', graded['n'])} "
                  f"xyz_d={graded.get('mean_xyz_dist', float('nan')):.3f}  "
                  f"(visible_in_net={n_visible_in_net})  ({elapsed:.1f}s)",
                  flush=True)

        # Aggregate
        if all(n in state for n in NETWORKS):
            weights = np.array([state[n]["n_anchors_held"] for n in NETWORKS])
            top1 = np.array([state[n]["top1"] for n in NETWORKS])
            pair = np.array([state[n]["pair_id"] for n in NETWORKS])
            hemi = np.array([state[n]["hemi"] for n in NETWORKS])
            top5 = np.array([state[n]["top5"] for n in NETWORKS])
            xd = np.array([state[n]["mean_xyz_dist"] for n in NETWORKS])
            wt = weights.sum()
            print(f"\nWEIGHTED: top1={(top1*weights).sum()/wt:.1%} "
                  f"top5={(top5*weights).sum()/wt:.1%} "
                  f"pair={(pair*weights).sum()/wt:.1%} "
                  f"hemi={(hemi*weights).sum()/wt:.1%}  "
                  f"xyz_d={(xd*weights).sum()/wt:.3f}")

    elif args.mode == "production":
        # Full anchor supervision (no held-out) → for FC translation comparison
        all_pids = sorted(PAIRID_TO_NETWORK.keys())
        t = time.time()
        pi, info = hierarchical_semirelaxed_fgw(
            Cm, Ch, M_xyz, idx_m, idx_h, M_.var, H.var,
            visible_pair_ids=all_pids,
            alpha=0.5, epsilon=5e-3, lam_anchor=1.0, xyz_w=0.5,
            max_iter=25, tol=1e-5, verbose=True,
        )
        print(f"\nfull production solve: {time.time()-t:.1f}s")

        # Save π
        np.save(ROOT / "outputs" / "coupling" / "pi_hierarchical.npy", pi.astype(np.float32))
        print(f"saved pi_hierarchical.npy")

        # FC translation quality
        from homer.data.networks import assign_networks
        net_h = assign_networks(H.var, idx_h)
        m = fc_translation_quality(
            pi.astype(np.float64), M_.uns["fc_mean"].astype(np.float64),
            H.uns["fc_mean"].astype(np.float64), network_labels_h=net_h,
        )
        print(f"\nFC translation Pearson r:")
        print(f"  overall    = {m['pearson_r_overall']:.3f}")
        print(f"  within-net = {m.get('pearson_r_within_net', float('nan')):.3f}")
        print(f"  cross-net  = {m.get('pearson_r_cross_net',  float('nan')):.3f}")
        print(f"  n_human_kept = {m['n_human_nodes_kept']}/{n_h}")

        # Update fc_translation.json with this entry
        fc_path = LOG / "fc_translation.json"
        existing = json.loads(fc_path.read_text()) if fc_path.exists() else {}
        existing["hierarchical_fc_only"] = m
        fc_path.write_text(json.dumps(existing, indent=2, default=float))
        print(f"\nadded to → {fc_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cv", "production"], default="cv")
    ap.add_argument("--networks", default=None)
    ap.add_argument("--recompute", action="store_true")
    main(ap.parse_args())
