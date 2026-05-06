"""Iterative anchor expansion / soft co-clustering CV (ROADMAP item M9 / "B").

Workflow per fold:
  1. Solve FGW with VISIBLE anchors only (lam=1.0 hard supervision).
  2. From π, identify the most confident non-anchor mouse rows
     (concentration = n_m * max(pi_row) ∈ [0, 1]).
  3. Add the top-K rows as SOFT anchors at lam=lam_soft.
  4. Re-solve. Repeat for n_iter rounds.
  5. Score the final π on the held-out anchors with the standard graded helper.

The soft anchors are derived from the model's own predictions, not from
ground truth, so this is CV-fair. We're testing whether the model can
bootstrap itself: after one solve, does it know enough about the easy nodes
that locking them in helps the hard nodes?

Saves results into outputs/logs/iterative_cv.json keyed by config_name +
"__iter{n}_topk{K}_lam{lam_soft}".
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
from homer.data import load_cached                                # noqa: E402
from homer.data.anchors import (                                   # noqa: E402
    get_anchor_index, held_out_metrics_graded,
)
from homer.data.networks import (                                  # noqa: E402
    PAIRID_TO_NETWORK, NETWORKS, assign_networks, network_mismatch_mask,
)

ANN = ROOT / "outputs" / "anndata"
LOG = ROOT / "outputs" / "logs"; LOG.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore")


# Reuse a couple of configs from multimodal_cv (subset that we want to iterate)
CONFIGS = {
    "baseline_fc_only": {
        "relational": {"FC": 1.0},
        "M":          {"xyz": 0.5},
    },
    "fc_plus_SC": {
        "relational": {"FC": 0.7, "SC": 0.3},
        "M":          {"xyz": 0.5},
    },
}


def build_cost(rel_weights, costs):
    Cm = np.zeros_like(costs["FC_m"], dtype=np.float64)
    Ch = np.zeros_like(costs["FC_h"], dtype=np.float64)
    for k, w in rel_weights.items():
        if w == 0: continue
        Cm += w * costs[f"{k}_m"].astype(np.float64)
        Ch += w * costs[f"{k}_h"].astype(np.float64)
    return Cm, Ch


def build_M_with_soft(M_weights, costs, idx_m, idx_h, visible_pair_ids,
                       *, lam_anchor=1.0, soft_pairs=None, lam_soft=0.3):
    """Like build_M in multimodal_cv but also accepts soft_pairs.

    soft_pairs: optional list of (mouse_pos, human_pos) tuples — these are
    added as low-λ anchors. They CANNOT overlap with the rows already
    occupied by hard (visible) anchors (we skip those).
    """
    M = np.zeros_like(costs["M_xyz"], dtype=np.float64)
    if M_weights.get("xyz", 0):
        M += M_weights["xyz"] * costs["M_xyz"].astype(np.float64)

    # Hard anchor supervision (visible)
    hard_mouse_rows = set()
    visible = set(int(p) for p in visible_pair_ids)
    for k, mp in enumerate(idx_m.pos):
        if int(idx_m.pair_ids[k]) in visible:
            M[mp, :] = lam_anchor
            M[mp, idx_h.pos[k]] = 0.0
            hard_mouse_rows.add(int(mp))
    for k, hp in enumerate(idx_h.pos):
        if int(idx_h.pair_ids[k]) in visible:
            mp_correct = idx_m.pos[k]
            col_mask = M[:, hp] < lam_anchor
            M[col_mask, hp] = lam_anchor
            M[mp_correct, hp] = 0.0

    # Soft anchors (low-λ, never overwriting a hard row)
    if soft_pairs is not None:
        for mp, hp in soft_pairs:
            mp = int(mp); hp = int(hp)
            if mp in hard_mouse_rows:
                continue
            # Penalise the entire row by lam_soft, then make the chosen target free
            M[mp, :] = np.maximum(M[mp, :], lam_soft)
            M[mp, hp] = 0.0
    return M


def confidence_per_row(pi, n_m):
    """Concentration of a row of π. Rows sum to 1/n_m for semirelaxed FGW.

    confidence ∈ [1/n_h, 1]: 1 if the row is one-hot at a single column,
    1/n_h if it's uniform over n_h columns.
    """
    row_max = pi.max(axis=1)
    row_sum = pi.sum(axis=1).clip(min=1e-12)
    return row_max / row_sum                # ∈ (0, 1]


def pick_soft_pairs(pi, idx_m, conf_thresh, top_k=None,
                     forbid_human_pos=None):
    """Pick (mouse_pos, human_pos) pairs from π for soft anchoring.

    Skips rows whose mouse_pos is already an anchor (regardless of visibility).
    Also skips human columns that are already used by visible anchors
    (those are passed in via `forbid_human_pos`, used to avoid locking in a
    soft anchor that would compete with the visible hard anchor).

    If top_k given, take the top_k by confidence (after threshold filter);
    otherwise take all above conf_thresh.
    """
    n_m, n_h = pi.shape
    conf = confidence_per_row(pi, n_m)
    argmax_h = pi.argmax(axis=1)

    # Exclude rows that ARE the anchor positions (we don't want to "soft anchor"
    # a hard-anchor mouse node; that's redundant)
    anchor_mouse_pos = set(int(p) for p in idx_m.pos)
    forbid_h = set(int(p) for p in (forbid_human_pos or []))

    pairs = []
    for i in range(n_m):
        if i in anchor_mouse_pos:
            continue
        h = int(argmax_h[i])
        if h in forbid_h:
            continue
        if conf[i] < conf_thresh:
            continue
        pairs.append((i, h, float(conf[i])))

    # Sort by confidence desc, optionally truncate
    pairs.sort(key=lambda t: -t[2])
    if top_k is not None:
        pairs = pairs[:top_k]
    return [(p[0], p[1]) for p in pairs], [p[2] for p in pairs]


def solve_fgw(M, Cm, Ch, p, alpha=0.5, eps=5e-3, max_iter=25, tol=1e-5):
    pi, _ = ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein(
        M=M, C1=Cm, C2=Ch, p=p, alpha=alpha, epsilon=eps,
        max_iter=max_iter, tol=tol, log=True,
    )
    return pi


def main(args):
    H, _ = load_cached("human", cache_dir=ANN)
    M_, _ = load_cached("mouse", cache_dir=ANN)
    idx_h = get_anchor_index(H.var); idx_m = get_anchor_index(M_.var)
    n_m = M_.uns["n_nodes"]; n_h = H.uns["n_nodes"]
    p = np.full(n_m, 1.0 / n_m)

    d = np.load(ANN / "full_costs.npz")
    costs = {
        "FC_m":  d["Cm"], "FC_h":  d["Ch"],
        "xyz_m": d["Cm_xyz"], "xyz_h": d["Ch_xyz"],
        "SC_m":  d["Cm_SC"],  "SC_h":  d["Ch_SC"],
        "M_xyz": d["M_xyz"],
    }
    net_to_pairs = {n: [] for n in NETWORKS}
    for pid, name in PAIRID_TO_NETWORK.items():
        net_to_pairs[name].append(pid)

    cfg_name   = args.config
    cfg        = CONFIGS[cfg_name]
    Cm, Ch     = build_cost(cfg["relational"], costs)

    cache_path = LOG / "iterative_cv.json"
    state = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    cfg_key = (
        f"{cfg_name}__iter{args.n_iter}"
        f"_topk{args.top_k if args.top_k else 'inf'}"
        f"_thr{args.conf_thresh:.2f}"
        f"_lam{args.lam_soft:.2f}"
    )
    if args.cache_suffix:
        cfg_key += args.cache_suffix
    print(f"\n=== {cfg_key} ===")
    print(f"  relational={cfg['relational']}, M={cfg['M']}")
    print(f"  n_iter={args.n_iter}, top_k={args.top_k}, "
          f"conf_thresh={args.conf_thresh}, lam_soft={args.lam_soft}")

    nets_to_run = (args.networks.split(",") if args.networks else NETWORKS)
    results = state.get(cfg_key, {})

    for net_name in nets_to_run:
        if net_name in results and not args.recompute:
            r = results[net_name]
            print(f"  {net_name:15s} cached: pair={r['pair_id']:.0%} top1={r['top1']:.0%}")
            continue
        held = sorted(net_to_pairs[net_name])
        visible = sorted([pid for pid in PAIRID_TO_NETWORK if pid not in held])
        forbid_h = [hp for k, hp in enumerate(idx_h.pos)
                     if int(idx_h.pair_ids[k]) in set(visible)]

        t0 = time.time()
        soft_pairs = None
        per_iter = []
        pi = None
        for it in range(args.n_iter + 1):                       # +1 = the initial solve
            M = build_M_with_soft(
                cfg["M"], costs, idx_m, idx_h, visible,
                lam_anchor=1.0,
                soft_pairs=soft_pairs, lam_soft=args.lam_soft,
            )
            pi = solve_fgw(M, Cm, Ch, p)
            pi_anchor = pi[np.ix_(idx_m.pos, idx_h.pos)]
            graded = held_out_metrics_graded(pi_anchor, idx_m, idx_h, held, var_h=H.var)
            n_soft = 0 if soft_pairs is None else len(soft_pairs)
            per_iter.append({
                "iter": it,
                "n_soft": n_soft,
                "top1":   graded["top1"],
                "top5":   graded["top5"],
                "pair":   graded["pair_id"],
                "hemi":   graded["hemisphere"],
                "rank":   graded["mean_rank"],
                "xyz_d":  graded.get("mean_xyz_dist", float("nan")),
            })
            if it < args.n_iter:
                # Build next set of soft anchors from the just-solved π
                soft_pairs, _ = pick_soft_pairs(
                    pi, idx_m,
                    conf_thresh=args.conf_thresh,
                    top_k=args.top_k,
                    forbid_human_pos=forbid_h,
                )
        elapsed = time.time() - t0
        # Final iteration's metrics are the ones we report
        final = per_iter[-1]
        results[net_name] = {
            "n_anchors_held":  graded["n"],
            "n_pair_ids_held": len(held),
            "top1":            final["top1"],
            "top5":            final["top5"],
            "pair_id":         final["pair"],
            "hemi":            final["hemi"],
            "mean_rank":       final["rank"],
            "mean_xyz_dist":   final["xyz_d"],
            "n_soft_final":    final["n_soft"],
            "per_iter":        per_iter,
            "elapsed":         round(elapsed, 1),
        }
        state[cfg_key] = results
        cache_path.write_text(json.dumps(state, indent=2, default=float))
        # Print iteration progression in one line
        prog = " → ".join(
            f"it{p['iter']}({p['n_soft']:>4d}s):top1={p['top1']:.0%},pair={p['pair']:.0%}"
            for p in per_iter
        )
        print(f"  {net_name:15s} {prog}  ({elapsed:.1f}s)", flush=True)

    if all(n in results for n in NETWORKS):
        weights = np.array([results[n]["n_anchors_held"] for n in NETWORKS])
        top1 = np.array([results[n]["top1"] for n in NETWORKS])
        pair = np.array([results[n]["pair_id"] for n in NETWORKS])
        hemi = np.array([results[n]["hemi"] for n in NETWORKS])
        wt = weights.sum()
        print(f"  ▶ weighted: top1={(top1*weights).sum()/wt:.1%} "
              f"pair={(pair*weights).sum()/wt:.1%} hemi={(hemi*weights).sum()/wt:.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",   default="fc_plus_SC", choices=list(CONFIGS.keys()))
    ap.add_argument("--n-iter",   type=int, default=2,
                    help="number of REFINEMENT iterations after the initial solve")
    ap.add_argument("--top-k",    type=int, default=200,
                    help="cap soft anchors at this many highest-confidence pairs (0 = no cap)")
    ap.add_argument("--conf-thresh", type=float, default=0.50,
                    help="only nodes with row-max-fraction ≥ thresh become soft anchors")
    ap.add_argument("--lam-soft", type=float, default=0.30,
                    help="penalty for off-diagonal soft anchor entries (vs lam_anchor=1.0)")
    ap.add_argument("--networks", default=None, help="comma-sep subset of NETWORKS")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--cache-suffix", default="")
    args = ap.parse_args()
    if args.top_k == 0:
        args.top_k = None
    main(args)
