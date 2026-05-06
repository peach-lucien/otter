"""E3: Null distributions for held-out CV metrics.

Two null baselines per evaluation:

  1. RANDOM π — sample π from uniform[0, 1], project to satisfy the mouse
     marginal (semirelaxed: source marginal fixed, target free). No FGW solve.
     Tells us "what would these metrics be by pure chance?"

  2. PERMUTED-ANCHOR — shuffle which mouse anchor pair maps to which human
     anchor pair, then re-solve semirelaxed FGW with the same M, C_FC, etc.
     Tests whether anchor supervision is doing real work, or whether
     connectivity + xyz alone would give the same numbers.

For each null, run N trials (default 50), evaluate per-network top-1, top-5,
mean_rank, mean_xyz_dist on held-out anchors (using the standard
leave-one-network-out CV setup), and report the null distribution.

Outputs:
    outputs/logs/null_distributions.json
    outputs/figures/12_null_distributions.png

Resumable: trials cached per (null_type, trial_seed, network).
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
from homer.data.anchors import (                                                  # noqa: E402
    AnchorIndex, get_anchor_index, held_out_metrics_graded,
)
from homer.data.networks import (                                                 # noqa: E402
    PAIRID_TO_NETWORK, NETWORKS, assign_networks, network_mismatch_mask,
)

ANN = ROOT / "outputs" / "anndata"
LOG = ROOT / "outputs" / "logs"; LOG.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore")


def _build_M_visible(M_xyz, idx_m, idx_h, visible_pair_ids,
                     *, lam=1.0, xyz_w=0.5):
    """Standard production M build (FC-only baseline config)."""
    M = (xyz_w * M_xyz).astype(np.float64)
    visible = set(int(p) for p in visible_pair_ids)
    for k, mp in enumerate(idx_m.pos):
        if int(idx_m.pair_ids[k]) in visible:
            M[mp, :] = lam; M[mp, idx_h.pos[k]] = 0.0
    for k, hp in enumerate(idx_h.pos):
        if int(idx_h.pair_ids[k]) in visible:
            mp_correct = idx_m.pos[k]
            M[M[:, hp] < lam, hp] = lam
            M[mp_correct, hp] = 0.0
    return M


def _build_M_permuted_anchors(M_xyz, idx_m, idx_h, visible_pair_ids,
                               permutation, *, lam=1.0, xyz_w=0.5):
    """Same as _build_M_visible but with anchor pairings shuffled.

    `permutation` is a length-len(idx_h.pos) array of *positional* indices into
    the anchor list. Mouse anchor at sorted position k is paired with human
    anchor at sorted position permutation[k].
    """
    M = (xyz_w * M_xyz).astype(np.float64)
    visible = set(int(p) for p in visible_pair_ids)
    for k, mp in enumerate(idx_m.pos):
        if int(idx_m.pair_ids[k]) in visible:
            permuted_h_idx = permutation[k]
            hp_correct = idx_h.pos[permuted_h_idx]
            M[mp, :] = lam
            M[mp, hp_correct] = 0.0
    for k_perm, hp in enumerate(idx_h.pos):
        # which mouse anchor was permuted onto this human anchor?
        # k_perm-th human position is the target for mouse anchor with permutation[k_orig] == k_perm
        # find k_orig
        try:
            k_orig = int(np.where(permutation == k_perm)[0][0])
        except IndexError:
            continue
        if int(idx_m.pair_ids[k_orig]) in visible:
            mp_correct = idx_m.pos[k_orig]
            M[M[:, hp] < lam, hp] = lam
            M[mp_correct, hp] = 0.0
    return M


def _eval_pi(pi, idx_m, idx_h, held_out_pair_ids, var_h):
    """Restrict π to the anchor block and compute graded metrics on held-out."""
    pi_anchor = pi[np.ix_(idx_m.pos, idx_h.pos)]
    return held_out_metrics_graded(pi_anchor, idx_m, idx_h, held_out_pair_ids,
                                    var_h=var_h)


def _random_pi(n_m, n_h, p, seed):
    """Random doubly-stochastic-ish matrix satisfying mouse marginal."""
    rng = np.random.default_rng(seed)
    A = rng.uniform(0.5, 1.5, size=(n_m, n_h))
    # Satisfy mouse marginal exactly; human marginal floats (semirelaxed style)
    A = A * (p / A.sum(axis=1).clip(min=1e-12))[:, None]
    return A


def main(args):
    H, _ = load_cached("human", cache_dir=ANN)
    M_, _ = load_cached("mouse", cache_dir=ANN)
    idx_h = get_anchor_index(H.var); idx_m = get_anchor_index(M_.var)
    n_m, n_h = M_.uns["n_nodes"], H.uns["n_nodes"]
    p = np.full(n_m, 1.0 / n_m)

    d = np.load(ANN / "full_costs.npz")
    Cm = d["Cm"].astype(np.float64); Ch = d["Ch"].astype(np.float64)
    M_xyz = d["M_xyz"].astype(np.float64)

    net_to_pairs = {n: [] for n in NETWORKS}
    for pid, name in PAIRID_TO_NETWORK.items():
        net_to_pairs[name].append(pid)

    cache_path = LOG / "null_distributions.json"
    state = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    nets_to_run = (args.networks.split(",") if args.networks else NETWORKS)

    # ==================== RANDOM π NULL =====================================
    if args.null in ("random", "both"):
        print("\n=== RANDOM π NULL ===")
        random_results = state.setdefault("random_pi", {})
        for net_name in nets_to_run:
            r = random_results.setdefault(net_name, [])
            existing = len(r)
            need = args.n_trials - existing
            if need <= 0:
                print(f"  {net_name:15s} cached: {existing} trials")
                continue
            held = sorted(net_to_pairs[net_name])
            for trial in range(existing, args.n_trials):
                pi = _random_pi(n_m, n_h, p, seed=trial * 100 + hash(net_name) % 100)
                m = _eval_pi(pi, idx_m, idx_h, held, var_h=H.var)
                r.append({
                    "trial": trial,
                    "top1": m["top1"], "top5": m["top5"],
                    "pair_id": m["pair_id"], "hemisphere": m["hemisphere"],
                    "mean_rank": m["mean_rank"],
                    "mean_xyz_dist": m.get("mean_xyz_dist", float("nan")),
                })
            cache_path.write_text(json.dumps(state, indent=2, default=float))
            top1s = [t["top1"] for t in r]
            top5s = [t["top5"] for t in r]
            print(f"  {net_name:15s} ({len(r)} trials): "
                  f"top1={np.mean(top1s):.0%}±{np.std(top1s):.0%}  "
                  f"top5={np.mean(top5s):.0%}±{np.std(top5s):.0%}", flush=True)

    # ==================== PERMUTED-ANCHOR NULL ==============================
    if args.null in ("permute", "both"):
        print("\n=== PERMUTED-ANCHOR NULL ===")
        # Default config = baseline_fc_only (FC GW + xyz in M + anchor supervision)
        perm_results = state.setdefault("permuted_anchors", {})
        for net_name in nets_to_run:
            r = perm_results.setdefault(net_name, [])
            existing = len(r)
            need = args.n_perm_trials - existing
            if need <= 0:
                print(f"  {net_name:15s} cached: {existing} trials")
                continue
            held = sorted(net_to_pairs[net_name])
            visible = sorted([pid for pid in PAIRID_TO_NETWORK if pid not in held])
            for trial in range(existing, args.n_perm_trials):
                rng = np.random.default_rng(trial * 1000 + hash(net_name) % 1000)
                permutation = rng.permutation(len(idx_m.pos))
                M = _build_M_permuted_anchors(M_xyz, idx_m, idx_h, visible, permutation)
                t = time.time()
                pi, log = ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein(
                    M=M, C1=Cm, C2=Ch, p=p, alpha=0.5, epsilon=5e-3,
                    max_iter=25, tol=1e-5, log=True,
                )
                m = _eval_pi(pi, idx_m, idx_h, held, var_h=H.var)
                r.append({
                    "trial": trial,
                    "top1": m["top1"], "top5": m["top5"],
                    "pair_id": m["pair_id"], "hemisphere": m["hemisphere"],
                    "mean_rank": m["mean_rank"],
                    "mean_xyz_dist": m.get("mean_xyz_dist", float("nan")),
                    "elapsed": round(time.time() - t, 1),
                })
            cache_path.write_text(json.dumps(state, indent=2, default=float))
            top1s = [t["top1"] for t in r]
            print(f"  {net_name:15s} ({len(r)} trials): "
                  f"top1={np.mean(top1s):.0%}±{np.std(top1s):.0%}  "
                  f"({np.mean([t['elapsed'] for t in r]):.1f}s/trial)", flush=True)

    print(f"\nsaved → {cache_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--null", choices=["random", "permute", "both"],
                    default="random",
                    help="which null to run; 'random' is fast (no FGW), "
                         "'permute' takes ~5s per trial")
    ap.add_argument("--n-trials",      type=int, default=50, help="random π trials")
    ap.add_argument("--n-perm-trials", type=int, default=20, help="permuted-anchor trials")
    ap.add_argument("--networks", default=None, help="comma-sep subset of NETWORKS")
    main(ap.parse_args())
