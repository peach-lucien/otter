"""Pipeline 05e — Knox-leaf-level SC vs. Allen summary-structure SC LONO comparison.

Tests whether replacing the Allen summary-structure SC fingerprints for the
22 cortical Garin anchors with Knox 2019 leaf-level SC fingerprints improves
held-out anchor recovery in MultimodalFGW.

The original Knox comparison was run with an unnormalized Cm_SC_knox (range
[0, 1.32]) which silently over-weighted SC by ~30% relative to the Allen SC
(range [0, 1]). This script regenerates the comparison using the properly
normalized cost matrices so the only thing that changes between configs is the
SC content, not its scale.

Output: outputs/logs/knox_vs_standard_sc.json (per-network, both configs).
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
from homer.data.anchors import get_anchor_index                         # noqa: E402
from homer.data.networks import NETWORKS, PAIRID_TO_NETWORK             # noqa: E402
from homer.eval.full_space_metrics import full_space_metrics            # noqa: E402
from homer.models import MultimodalFGW                                  # noqa: E402

ANN  = ROOT / "outputs" / "anndata"
LOG  = ROOT / "outputs" / "logs"; LOG.mkdir(parents=True, exist_ok=True)


CONFIGS = {
    "fc_plus_SC_allen":   "Cm_SC",        # production: Allen summary-structure SC
    "fc_plus_SC_knox":    "Cm_SC_knox",   # comparative: Knox leaf-level SC for 22 cortical anchors
}


def main(args):
    H, _ = load_cached("human", cache_dir=ANN)
    M, _ = load_cached("mouse", cache_dir=ANN)
    idx_m = get_anchor_index(M.var); idx_h = get_anchor_index(H.var)
    costs = np.load(ANN / "full_costs.npz")

    # Sanity: confirm both SC matrices are on the same scale before we use them
    for k in CONFIGS.values():
        assert k in costs.files, f"missing {k} in full_costs.npz — run 06_knox_sc.py"
        a = costs[k]
        assert a.min() >= 0 and a.max() <= 1.001, (
            f"{k} not on [0,1] scale (range=[{a.min():.4f}, {a.max():.4f}]); "
            f"re-run pipeline/00_external/06_knox_sc.py after normalisation fix."
        )
    print(f"Cm_SC      range=[{costs['Cm_SC'].min():.4f}, {costs['Cm_SC'].max():.4f}], "
          f"mean={costs['Cm_SC'].mean():.4f}")
    print(f"Cm_SC_knox range=[{costs['Cm_SC_knox'].min():.4f}, {costs['Cm_SC_knox'].max():.4f}], "
          f"mean={costs['Cm_SC_knox'].mean():.4f}")

    # Confirm Knox actually differs from Allen on cortical anchors
    diff = np.abs(costs["Cm_SC_knox"] - costs["Cm_SC"]).mean()
    print(f"|Cm_SC_knox - Cm_SC|.mean() = {diff:.4f} "
          f"(should be > 0 — Knox replaces ~22 cortical fingerprints)")

    net_to_pairs = {n: [] for n in NETWORKS}
    for pid, name in PAIRID_TO_NETWORK.items():
        net_to_pairs[name].append(pid)

    cache_path = LOG / "knox_vs_standard_sc.json"
    state = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    cfg_names = (args.configs.split(",") if args.configs else list(CONFIGS.keys()))
    nets      = (args.networks.split(",") if args.networks else NETWORKS)

    for cfg_name in cfg_names:
        sc_key = CONFIGS[cfg_name]
        results = state.get(cfg_name, {})
        print(f"\n=== {cfg_name} (Cm_SC={sc_key}) ===")
        for net_name in nets:
            if net_name in results and not args.recompute:
                r = results[net_name]
                print(f"  {net_name:15s} cached: full_top1={r['full_top1']:.0%}  "
                      f"rank={r['mean_rank_full']:.0f}/{r['n_h_total']}")
                continue
            held = sorted(net_to_pairs[net_name])
            t = time.time()
            m = MultimodalFGW(use_sc=True, sc_weight=0.3, fc_weight=0.7,
                              epsilon=5e-3, xyz_weight=0.5)
            m.fit(M, H, holdout_pair_ids=held,
                  Cm_SC=costs[sc_key], Ch_SC=costs["Ch_SC"])
            metrics = full_space_metrics(m.pi, idx_m, idx_h, held, var_h=H.var)
            metrics["elapsed_s"] = round(time.time() - t, 1)
            metrics["sc_key"] = sc_key
            results[net_name] = metrics
            state[cfg_name] = results
            cache_path.write_text(json.dumps(state, indent=2, default=float))
            print(f"  {net_name:15s} (n={metrics['n']}): "
                  f"full_top1={metrics['full_top1']:.0%}  "
                  f"top5={metrics['full_top5']:.0%}  "
                  f"mean_rank={metrics['mean_rank_full']:.0f}/{metrics['n_h_total']}  "
                  f"argmax_is_anchor={metrics['frac_argmax_is_anchor']:.0%}  "
                  f"({metrics['elapsed_s']}s)", flush=True)

    # --- Side-by-side summary on the 4 cortical-heavy networks --------
    cortical_nets = ["visual", "sensorimotor", "salience", "frontoparietal"]
    print("\n=== Cortical-network LONO summary (where Knox actually differs) ===")
    print(f"{'network':16s} {'config':25s} {'full_top1':>10s} "
          f"{'mean_rank':>10s} {'argmax_is_anchor':>18s}")
    for net in cortical_nets:
        for cfg_name in cfg_names:
            r = state.get(cfg_name, {}).get(net)
            if r is None: continue
            print(f"{net:16s} {cfg_name:25s} {r['full_top1']:10.0%} "
                  f"{r['mean_rank_full']:10.1f} {r['frac_argmax_is_anchor']:18.0%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs",  default=None, help="comma-separated subset")
    ap.add_argument("--networks", default=None,
                    help="comma-separated subset (default: all 11)")
    ap.add_argument("--recompute", action="store_true",
                    help="recompute even if cached")
    main(ap.parse_args())
