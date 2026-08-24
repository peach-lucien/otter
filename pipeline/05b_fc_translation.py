"""Evaluate functional-connectivity push-forward for the configurations defined in this script."""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ot

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from otter.data import load_cached                                              # noqa: E402
from otter.data.anchors  import get_anchor_index                                  # noqa: E402
from otter.data.networks import assign_networks, network_mismatch_mask            # noqa: E402
from otter.eval.translation import (                                             # noqa: E402
    fc_translation_quality, random_pi_baseline, uniform_pi_baseline,
)

ANN = ROOT / "outputs" / "anndata"
PI  = ROOT / "outputs" / "coupling"
LOG = ROOT / "outputs" / "logs"; LOG.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "outputs" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore")


# Same configs (subset) as multimodal_cv but solved with FULL anchor supervision
CONFIGS: dict[str, dict] = {
    "baseline_fc_only": {
        "relational": {"FC": 1.0},
        "M":          {"xyz": 0.5, "network": 0.0, "gene": 0.0},
    },
    "fc_plus_SC": {
        "relational": {"FC": 0.7, "SC": 0.3},
        "M":          {"xyz": 0.5, "network": 0.0, "gene": 0.0},
    },
    "fc_plus_xyz_gw": {
        "relational": {"FC": 0.75, "xyz": 0.25},
        "M":          {"xyz": 0.5, "network": 0.0, "gene": 0.0},
    },
    "fc_plus_network_mask": {
        "relational": {"FC": 0.75, "xyz": 0.25},
        "M":          {"xyz": 0.5, "network": 0.10, "gene": 0.0},
    },
}


def build_cost(rel_weights: dict, costs: dict) -> tuple[np.ndarray, np.ndarray]:
    Cm = np.zeros_like(costs["FC_m"], dtype=np.float64)
    Ch = np.zeros_like(costs["FC_h"], dtype=np.float64)
    for k, w in rel_weights.items():
        if w == 0: continue
        Cm += w * costs[f"{k}_m"].astype(np.float64)
        Ch += w * costs[f"{k}_h"].astype(np.float64)
    return Cm, Ch


def build_M_full(M_weights: dict, costs: dict, idx_m, idx_h, net_mask, lam=1.0):
    """Full anchor supervision, every anchor visible."""
    M = np.zeros_like(costs["M_xyz"], dtype=np.float64)
    if M_weights.get("xyz", 0): M += M_weights["xyz"] * costs["M_xyz"].astype(np.float64)
    if M_weights.get("network", 0) > 0:
        M += M_weights["network"] * net_mask.astype(np.float64)
    for k, mp in enumerate(idx_m.pos):
        M[mp, :] = lam; M[mp, idx_h.pos[k]] = 0.0
    for k, hp in enumerate(idx_h.pos):
        col_mask = M[:, hp] < lam
        M[col_mask, hp] = lam
        M[idx_m.pos[k], hp] = 0.0
    return M


def main(args):
    H, _ = load_cached("human", cache_dir=ANN)
    M_, _ = load_cached("mouse", cache_dir=ANN)
    idx_h = get_anchor_index(H.var); idx_m = get_anchor_index(M_.var)
    n_m, n_h = M_.uns["n_nodes"], H.uns["n_nodes"]
    p = np.full(n_m, 1.0 / n_m)

    fc_mouse = M_.uns["fc_mean"].astype(np.float64)
    fc_human = H.uns["fc_mean"].astype(np.float64)
    print(f"FC matrices: mouse {fc_mouse.shape}, human {fc_human.shape}")

    d = np.load(ANN / "full_costs.npz")
    costs = {
        "FC_m": d["Cm"], "FC_h": d["Ch"],
        "xyz_m": d["Cm_xyz"], "xyz_h": d["Ch_xyz"],
        "SC_m":  d["Cm_SC"],  "SC_h":  d["Ch_SC"],
        "M_xyz": d["M_xyz"], "M_gene": d["M_gene"],
    }
    net_h = assign_networks(H.var, idx_h)
    net_mask = network_mismatch_mask(assign_networks(M_.var, idx_m), net_h)

    config_names = args.configs.split(",") if args.configs else list(CONFIGS.keys())
    results: dict[str, dict] = {}

    for cfg_name in config_names:
        cfg = CONFIGS[cfg_name]
        print(f"\n=== {cfg_name} ===")
        Cm, Ch = build_cost(cfg["relational"], costs)
        M = build_M_full(cfg["M"], costs, idx_m, idx_h, net_mask)

        t = time.time()
        pi, log = ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein(
            M=M, C1=Cm, C2=Ch, p=p, alpha=0.5, epsilon=5e-3,
            max_iter=25, tol=1e-5, log=True,
        )
        print(f"  solved in {time.time()-t:.1f}s; fgw_dist={log.get('srfgw_dist', '?')}")

        t = time.time()
        q = pi.sum(axis=0)
        m = fc_translation_quality(pi, fc_mouse, fc_human, network_labels_h=net_h)
        print(f"  FC translation Pearson r:  overall={m['pearson_r_overall']:.3f}")
        if "pearson_r_within_net" in m:
            print(f"     within-net (n={m.get('n_within_net', 0)}): {m['pearson_r_within_net']:.3f}")
            print(f"     cross-net  (n={m.get('n_cross_net', 0)}): {m['pearson_r_cross_net']:.3f}")
        print(f"  n_human_nodes_kept: {m['n_human_nodes_kept']}/{n_h} (mass > 1e-6)")
        print(f"  metric in {time.time()-t:.1f}s")

        # Save π for downstream use
        np.save(PI / f"pi_{cfg_name}.npy", pi.astype(np.float32))
        m["q_marginal_min"] = float(q.min())
        m["q_marginal_max"] = float(q.max())
        m["fgw_dist"]       = float(log.get("srfgw_dist", log.get("fgw_dist", -1)))
        results[cfg_name] = m

    # --- Baselines ---
    if args.baselines:
        print("\n=== baselines ===")
        # uniform π marginals: use the marginal q from baseline_fc_only as
        # the "natural" target marginal (all configs use the same p)
        if "baseline_fc_only" in results:
            print("computing uniform π baseline...")
            q_natural = np.full(n_h, 1.0 / n_h)
            results["_baseline_uniform"] = uniform_pi_baseline(
                fc_mouse, fc_human, p=p, q=q_natural, network_labels_h=net_h
            )
            print(f"  uniform: r = {results['_baseline_uniform']['pearson_r_overall']:.3f}")

            print("computing random π baseline (20 trials)...")
            results["_baseline_random"] = random_pi_baseline(
                fc_mouse, fc_human, p=p, q=q_natural, n_trials=args.n_random_trials,
                network_labels_h=net_h,
            )
            print(f"  random: r = {results['_baseline_random']['pearson_r_mean']:.3f} "
                  f"± {results['_baseline_random']['pearson_r_std']:.3f}")

    # Merge with existing results so partial runs accumulate
    out = LOG / "fc_translation.json"
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing.update(results)
    out.write_text(json.dumps(existing, indent=2, default=float))
    results = existing       # use combined dict for the figure
    print(f"\nsaved → {out}")

    # --- Figure ---
    cfg_results = {k: v for k, v in results.items() if not k.startswith("_")}
    if cfg_results:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        labels = list(cfg_results.keys())
        overall = [cfg_results[k]["pearson_r_overall"] for k in labels]
        within  = [cfg_results[k].get("pearson_r_within_net", np.nan) for k in labels]
        cross   = [cfg_results[k].get("pearson_r_cross_net",  np.nan) for k in labels]
        x = np.arange(len(labels))
        w = 0.27
        ax.bar(x - w, overall, w, label="overall",     color="C0")
        ax.bar(x,     within,  w, label="within-net",  color="C2")
        ax.bar(x + w, cross,   w, label="cross-net",   color="C3")
        for i, v in enumerate(overall):
            ax.text(i - w, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
        # baseline lines
        if "_baseline_random" in results:
            r_rand = results["_baseline_random"]["pearson_r_mean"]
            ax.axhline(r_rand, color="grey", linestyle="--", linewidth=0.8,
                       label=f"random π (mean = {r_rand:.3f})")
        if "_baseline_uniform" in results:
            r_unif = results["_baseline_uniform"]["pearson_r_overall"]
            ax.axhline(r_unif, color="grey", linestyle=":", linewidth=0.8,
                       label=f"uniform π = {r_unif:.3f}")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("Pearson r between predicted and actual human FC")
        ax.set_title("E1: FC translation quality\n"
                     "Higher = π better translates mouse FC structure into human FC structure")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_ylim(min(0, min(overall)) - 0.05, max(overall) + 0.10)
        fig.tight_layout()
        fig.savefig(FIG / "10_fc_translation.png", dpi=140, bbox_inches="tight")
        print(f"saved figure → {FIG / '10_fc_translation.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=None, help="comma-sep subset of CONFIGS")
    ap.add_argument("--baselines", action="store_true", default=True,
                    help="include random + uniform π baselines (default on)")
    ap.add_argument("--n-random-trials", type=int, default=10)
    main(ap.parse_args())
