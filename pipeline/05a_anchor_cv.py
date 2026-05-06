"""Multi-modal FGW CV: ablate combinations of (FC, SC, gene-coexpr) GW terms
and (xyz, network-mask, gene-cosine) M terms.

For each named config:
  1. Build combined relational cost C_m, C_h as a weighted sum across modalities.
  2. Build combined M as a weighted sum + anchor supervision (per fold).
  3. Run leave-one-network-out semirelaxed FGW.
  4. Report per-modality contribution.

Result is a comparison table written to outputs/logs/multimodal_cv.json.

Each fold takes ~5 s, 11 folds × N configs → ~1 min × N. Resumable per
(config × network) cell.
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
from homer.models import entropic_semirelaxed_fgw_multistart        # noqa: E402

ANN = ROOT / "outputs" / "anndata"
LOG = ROOT / "outputs" / "logs"; LOG.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Named configs to try. Each is (relational_weights, m_weights).
# relational_weights: dict modality → weight in the combined C_m, C_h
# m_weights:          dict modality → weight in the combined M
#                     (anchor_lam is always 1.0 — that's the supervision strength)
# ---------------------------------------------------------------------------
CONFIGS: dict[str, dict] = {
    "baseline_fc_only": {
        "relational": {"FC": 1.0},
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
    "fc_plus_SC": {
        "relational": {"FC": 0.7, "xyz": 0.0, "SC": 0.3},
        "M":          {"xyz": 0.5, "network": 0.0, "gene": 0.0},
    },
    "fc_plus_gene_GW": {
        "relational": {"FC": 0.7, "xyz": 0.0, "gene": 0.3},
        "M":          {"xyz": 0.5, "network": 0.0, "gene": 0.0},
    },
    "fc_plus_M_gene": {
        "relational": {"FC": 1.0},
        "M":          {"xyz": 0.5, "network": 0.0, "gene": 0.5},
    },
    "fc_plus_SC_plus_M_gene": {
        "relational": {"FC": 0.7, "SC": 0.3},
        "M":          {"xyz": 0.5, "network": 0.0, "gene": 0.5},
    },
    "all_modalities": {
        "relational": {"FC": 0.4, "xyz": 0.2, "SC": 0.2, "gene": 0.2},
        "M":          {"xyz": 0.5, "network": 0.10, "gene": 0.5},
    },
    # === selective M_gene (gene cost only where both species have data) ===
    "fc_plus_selective_M_gene": {
        "relational": {"FC": 1.0},
        "M":          {"xyz": 0.5, "network": 0.0, "gene": 0.5, "gene_selective": True},
    },
    "fc_plus_SC_plus_selective_M_gene": {
        "relational": {"FC": 0.7, "SC": 0.3},
        "M":          {"xyz": 0.5, "network": 0.0, "gene": 0.5, "gene_selective": True},
    },
    "all_modalities_selective": {
        "relational": {"FC": 0.4, "xyz": 0.2, "SC": 0.2, "gene": 0.2},
        "M":          {"xyz": 0.5, "network": 0.10, "gene": 0.5, "gene_selective": True},
    },
    # === Anchor-relationship features in M (item A) ===
    "fc_plus_M_anchor": {
        "relational": {"FC": 1.0},
        "M":          {"xyz": 0.5, "network": 0.0, "gene": 0.0, "anchor_feat": 0.5},
    },
    "fc_plus_SC_plus_M_anchor": {
        "relational": {"FC": 0.7, "SC": 0.3},
        "M":          {"xyz": 0.5, "network": 0.0, "gene": 0.0, "anchor_feat": 0.5},
    },
}


def build_cost(rel_weights: dict, costs: dict) -> tuple[np.ndarray, np.ndarray]:
    """Combine relational cost matrices for both species into one each."""
    Cm = np.zeros_like(costs["FC_m"], dtype=np.float64)
    Ch = np.zeros_like(costs["FC_h"], dtype=np.float64)
    for k, w in rel_weights.items():
        if w == 0: continue
        Cm += w * costs[f"{k}_m"].astype(np.float64)
        Ch += w * costs[f"{k}_h"].astype(np.float64)
    return Cm, Ch


def _anchor_M_visible_only(fc_m, fc_h, idx_m, idx_h, visible_pair_ids, eps=1e-6):
    """Compute M_anchor restricted to ONLY the visible anchors (CV-fair).

    For each fold, we recompute the anchor-relationship features using just the
    anchors whose cross-species pairing is visible to the model. This removes
    the leak where pre-computed M_anchor would secretly include held-out
    anchors' positions and their assumed cross-species correspondence.
    """
    visible = set(int(p) for p in visible_pair_ids)
    visible_local = [k for k, pid in enumerate(idx_m.pair_ids) if int(pid) in visible]
    if not visible_local:                                    # no visible anchors
        return np.zeros((fc_m.shape[0], fc_h.shape[0]), dtype=np.float64)
    pos_m = np.asarray(idx_m.pos)[visible_local]
    pos_h = np.asarray(idx_h.pos)[visible_local]            # same order — anchor pairing
    af_m = fc_m[:, pos_m].astype(np.float64)                # (n_m, k)
    af_h = fc_h[:, pos_h].astype(np.float64)                # (n_h, k)
    af_m = (af_m - af_m.mean(0, keepdims=True)) / af_m.std(0, keepdims=True).clip(min=eps)
    af_h = (af_h - af_h.mean(0, keepdims=True)) / af_h.std(0, keepdims=True).clip(min=eps)
    af_m = af_m / np.linalg.norm(af_m, axis=1, keepdims=True).clip(min=eps)
    af_h = af_h / np.linalg.norm(af_h, axis=1, keepdims=True).clip(min=eps)
    d = (1.0 - af_m @ af_h.T).clip(0.0, 2.0).astype(np.float64)
    return d / max(float(d.max()), 1e-9)


def build_M(M_weights: dict, costs: dict, idx_m, idx_h, visible_pair_ids,
            net_mask: np.ndarray, *, lam_anchor: float = 1.0,
            fc_m=None, fc_h=None) -> np.ndarray:
    """Build the cross-species M with combined modalities + anchor supervision.

    M_weights['gene_selective'] (bool) — if True, M_gene is multiplied by the
    M_gene_valid coverage mask, so it contributes 0 wherever either species
    lacks ortholog data (vs. the default which adds a max-cost penalty there).
    """
    M = np.zeros_like(costs["M_xyz"], dtype=np.float64)
    if M_weights.get("xyz", 0):
        M += M_weights["xyz"] * costs["M_xyz"].astype(np.float64)
    if M_weights.get("gene", 0):
        gene_term = costs["M_gene"].astype(np.float64)
        if M_weights.get("gene_selective", False):
            valid = costs["M_gene_valid"].astype(bool)
            mean_valid = float(gene_term[valid].mean()) if valid.any() else 0.5
            gene_term = np.where(valid, gene_term - mean_valid, 0.0)
        M += M_weights["gene"] * gene_term
    if M_weights.get("anchor_feat", 0):
        # Anchor-relationship cost — recomputed per-fold using only VISIBLE
        # anchors so we don't leak held-out anchor identities into the cost.
        if fc_m is None or fc_h is None:
            raise ValueError("anchor_feat requires fc_m, fc_h passed to build_M")
        M_anchor_fold = _anchor_M_visible_only(fc_m, fc_h, idx_m, idx_h, visible_pair_ids)
        M += M_weights["anchor_feat"] * M_anchor_fold
    if M_weights.get("network", 0) > 0:
        M += M_weights["network"] * net_mask.astype(np.float64)

    # Anchor supervision (forbidden cells)
    visible = set(int(p) for p in visible_pair_ids)
    for k, mp in enumerate(idx_m.pos):
        if int(idx_m.pair_ids[k]) in visible:
            M[mp, :] = lam_anchor
            M[mp, idx_h.pos[k]] = 0.0
    for k, hp in enumerate(idx_h.pos):
        if int(idx_h.pair_ids[k]) in visible:
            mp_correct = idx_m.pos[k]
            col_mask = M[:, hp] < lam_anchor
            M[col_mask, hp] = lam_anchor
            M[mp_correct, hp] = 0.0
    return M


def main(args):
    H, _ = load_cached("human", cache_dir=ANN)
    M_, _ = load_cached("mouse", cache_dir=ANN)
    idx_h = get_anchor_index(H.var); idx_m = get_anchor_index(M_.var)
    n_m = M_.uns["n_nodes"]; n_h = H.uns["n_nodes"]
    p = np.full(n_m, 1.0 / n_m)

    d = np.load(ANN / "full_costs.npz")
    costs = {
        "FC_m":         d["Cm"],
        "FC_h":         d["Ch"],
        "xyz_m":        d["Cm_xyz"],
        "xyz_h":        d["Ch_xyz"],
        "SC_m":         d["Cm_SC"],
        "SC_h":         d["Ch_SC"],
        "gene_m":       d["Cm_gene"],
        "gene_h":       d["Ch_gene"],
        "M_xyz":        d["M_xyz"],
        "M_gene":       d["M_gene"],
        "M_gene_valid": d["M_gene_valid"],
        "M_anchor":     d["M_anchor"],
    }

    net_m = assign_networks(M_.var, idx_m); net_h = assign_networks(H.var, idx_h)
    net_mask = network_mismatch_mask(net_m, net_h)

    # Build network → pair_id list for leave-one-network-out CV
    net_to_pairs = {n: [] for n in NETWORKS}
    for pid, name in PAIRID_TO_NETWORK.items():
        net_to_pairs[name].append(pid)

    config_names = args.configs.split(",") if args.configs else list(CONFIGS.keys())

    cache_path = LOG / "multimodal_cv.json"
    state = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    nets_to_run = (args.networks.split(",") if args.networks else NETWORKS)

    for cfg_name in config_names:
        cfg = CONFIGS[cfg_name]
        Cm, Ch = build_cost(cfg["relational"], costs)
        cfg_key = cfg_name + args.cache_suffix
        tag = f" [multistart×{args.n_restarts + 2}]" if args.n_restarts > 0 else ""
        print(f"\n=== {cfg_key}{tag} ===")
        print(f"  relational: {cfg['relational']}")
        print(f"  M weights : {cfg['M']}")

        results = state.get(cfg_key, {})
        for net_name in nets_to_run:
            cached = results.get(net_name)
            # Recompute if cached but lacks the new graded fields
            if cached and "mean_rank" in cached and not args.recompute:
                print(f"  {net_name:15s} cached: pair={cached['pair_id']:.0%} rank={cached['mean_rank']:.1f}")
                continue
            held = sorted(net_to_pairs[net_name])
            visible = sorted([p for p in PAIRID_TO_NETWORK if p not in held])

            t = time.time()
            M = build_M(cfg["M"], costs, idx_m, idx_h, visible, net_mask,
                        fc_m=M_.uns["fc_mean"], fc_h=H.uns["fc_mean"])
            if args.n_restarts > 0:
                # Multistart: anchors VISIBLE in this fold inform the warm init
                vis_mask = np.array([int(p) in set(visible) for p in idx_m.pair_ids])
                pi, ms_info = entropic_semirelaxed_fgw_multistart(
                    M=M, C1=Cm, C2=Ch, p=p,
                    alpha=0.5, epsilon=5e-3, max_iter=25, tol=1e-5,
                    n_random_inits=args.n_restarts,
                    anchor_warm=(idx_m.pos[vis_mask], idx_h.pos[vis_mask])
                                 if vis_mask.any() else None,
                )
            else:
                pi, log = ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein(
                    M=M, C1=Cm, C2=Ch, p=p, alpha=0.5, epsilon=5e-3,
                    max_iter=25, tol=1e-5, log=True,
                )
                ms_info = None
            elapsed = time.time() - t
            # Restrict π to the anchor sub-block, then call the graded helper
            pi_anchor = pi[np.ix_(idx_m.pos, idx_h.pos)]   # (42, 42)
            graded = held_out_metrics_graded(pi_anchor, idx_m, idx_h, held, var_h=H.var)
            results[net_name] = {
                "n_anchors_held": graded["n"],
                "n_pair_ids_held": len(held),
                "top1":            graded["top1"],
                "top5":            graded["top5"],
                "pair_id":         graded["pair_id"],
                "hemi":            graded["hemisphere"],
                "mean_rank":       graded["mean_rank"],
                "median_rank":     graded["median_rank"],
                "max_rank":        graded.get("max_rank_possible", graded["n"]),
                "mean_xyz_dist":   graded.get("mean_xyz_dist", float("nan")),
                "median_xyz_dist": graded.get("median_xyz_dist", float("nan")),
                "elapsed":         round(elapsed, 1),
            }
            if ms_info is not None:
                results[net_name]["multistart"] = ms_info
            state[cfg_key] = results
            cache_path.write_text(json.dumps(state, indent=2, default=float))
            print(f"  {net_name:15s} (n={graded['n']}): "
                  f"top1={graded['top1']:.0%} top5={graded['top5']:.0%} "
                  f"pair={graded['pair_id']:.0%} hemi={graded['hemisphere']:.0%}  "
                  f"rank={graded['mean_rank']:.1f}/{graded.get('max_rank_possible', graded['n'])} "
                  f"xyz_d={graded.get('mean_xyz_dist', float('nan')):.3f}  ({elapsed:.1f}s)",
                  flush=True)

        # Aggregate weighted by n_held
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
    ap.add_argument("--configs",  default=None, help="comma-sep subset of CONFIGS")
    ap.add_argument("--networks", default=None, help="comma-sep subset of NETWORKS")
    ap.add_argument("--recompute", action="store_true",
                    help="ignore cache and recompute all (config, network) cells")
    ap.add_argument("--n-restarts", type=int, default=0,
                    help="if >0, use multistart FGW with this many random inits "
                         "(plus default uniform + anchor-warm)")
    ap.add_argument("--cache-suffix", default="",
                    help="append a suffix to the JSON cache key — useful when "
                         "comparing single-shot vs multistart (e.g. '_ms5')")
    main(ap.parse_args())
