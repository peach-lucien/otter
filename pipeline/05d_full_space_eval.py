"""Pipeline 05d, full-space LONO recovery metrics.

Companion to ``05a_anchor_cv.py``. The 05a script reports recovery using the
held-out anchor sub-block as the search space (restricted top-1 ~81%); this
script reports the SAME folds but using the full ``n_h`` human node space
(typically 0–5% top-1 because the model lands on a non-anchor grid node near
the correct anchor rather than the anchor itself).

Both numbers matter and answer different questions; this script is the
"per-voxel mapping" metric your colleague asked for.

Resumable: outputs/logs/full_space_eval.json caches per-(config, network).

Usage:
    python pipeline/05d_full_space_eval.py
    python pipeline/05d_full_space_eval.py --configs fc_plus_SC
    python pipeline/05d_full_space_eval.py --networks visual,brainstem
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
from otter.data import load_cached                                      # noqa: E402
from otter.data.anchors import get_anchor_index                         # noqa: E402
from otter.data.networks import NETWORKS, PAIRID_TO_NETWORK             # noqa: E402
from otter.eval.full_space_metrics import full_space_metrics            # noqa: E402
from otter.models import MultimodalFGW, SupervisedFGW                   # noqa: E402

ANN  = ROOT / "outputs" / "anndata"
LOG  = ROOT / "outputs" / "logs"; LOG.mkdir(parents=True, exist_ok=True)


CONFIGS = {
    "fc_only":    dict(model_cls=SupervisedFGW,
                        kwargs=dict(epsilon=5e-3, xyz_weight=0.5),
                        needs_sc=False),
    "fc_plus_SC": dict(model_cls=MultimodalFGW,
                        kwargs=dict(use_sc=True, sc_weight=0.3, fc_weight=0.7,
                                    epsilon=5e-3, xyz_weight=0.5),
                        needs_sc=True),
}


def main(args):
    H, _ = load_cached("human", cache_dir=ANN)
    M, _ = load_cached("mouse", cache_dir=ANN)
    idx_m = get_anchor_index(M.var); idx_h = get_anchor_index(H.var)
    costs = np.load(ANN / "full_costs.npz")

    net_to_pairs = {n: [] for n in NETWORKS}
    for pid, name in PAIRID_TO_NETWORK.items():
        net_to_pairs[name].append(pid)

    cache_path = LOG / "full_space_eval.json"
    state = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    cfg_names = (args.configs.split(",") if args.configs else list(CONFIGS.keys()))
    nets      = (args.networks.split(",") if args.networks else NETWORKS)

    for cfg_name in cfg_names:
        cfg = CONFIGS[cfg_name]
        cls = cfg["model_cls"]
        results = state.get(cfg_name, {})
        print(f"\n=== {cfg_name} ===")
        for net_name in nets:
            if net_name in results and not args.recompute:
                r = results[net_name]
                print(f"  {net_name:15s} cached: full_top1={r['full_top1']:.0%}  "
                      f"rank={r['mean_rank_full']:.0f}/{r['n_h_total']}")
                continue
            held = sorted(net_to_pairs[net_name])
            t = time.time()
            m = cls(**cfg["kwargs"])
            fit_kwargs = ({"Cm_SC": costs["Cm_SC"], "Ch_SC": costs["Ch_SC"]}
                          if cfg["needs_sc"] else {})
            m.fit(M, H, holdout_pair_ids=held, **fit_kwargs)
            metrics = full_space_metrics(m.pi, idx_m, idx_h, held, var_h=H.var)
            metrics["elapsed_s"] = round(time.time() - t, 1)
            results[net_name] = metrics
            state[cfg_name] = results
            cache_path.write_text(json.dumps(state, indent=2, default=float))
            print(f"  {net_name:15s} (n={metrics['n']}): "
                  f"full_top1={metrics['full_top1']:.0%}  "
                  f"top5={metrics['full_top5']:.0%}  "
                  f"mean_rank={metrics['mean_rank_full']:.0f}/{metrics['n_h_total']}  "
                  f"argmax_is_anchor={metrics['frac_argmax_is_anchor']:.0%}  "
                  f"in_neighborhood={metrics.get('frac_in_neighborhood', float('nan')):.0%}  "
                  f"({metrics['elapsed_s']}s)", flush=True)

        # Aggregate weighted
        if all(n in results for n in NETWORKS):
            w = np.array([results[n]["n"] for n in NETWORKS], dtype=float)
            wt = w.sum()
            for k in ("full_top1", "full_top5", "mean_rank_full",
                      "frac_argmax_is_anchor", "frac_in_neighborhood",
                      "mean_xyz_dist_full", "mean_mass_on_correct_anchor"):
                if k not in results[NETWORKS[0]]: continue
                vals = np.array([results[n].get(k, np.nan) for n in NETWORKS])
                agg = float(np.nansum(vals * w) / wt)
                print(f"  ▶ weighted {k:30s} = {agg:.3f}"
                      + (f"  ({agg:.0%})" if "top" in k or "frac" in k else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs",  default=None)
    ap.add_argument("--networks", default=None)
    ap.add_argument("--recompute", action="store_true")
    main(ap.parse_args())
