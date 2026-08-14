"""Pipeline 05j, region-level evaluation of π.

The parcel-level Beauchamp validation (05f) asks "is the right human *parcel*
in the top-K of π[m, :]?". This script asks the region-level analogue:

    Given a mouse region M (set of parcels), which *human region* does the
    model predict, out of a candidate set of named human regions?

π is a soft probabilistic mapping that spreads mass across multiple human
parcels in a region, so the region-level question matches its structure.

Pipeline
--------
1. Load π and the Beauchamp 22 mouse↔human region pairs (same masks as 05f).
2. Use the 22 human regions as the candidate set (Beauchamp-22; chance ~4.5%).
3. For each pair: aggregate π[M, :], rank candidates, report top-K, fold
   enrichment, mass on true region.
4. Run column-permuted null (preserves total mass, shuffles where it lands)
   and source-permuted null (scores H_true against another mouse region's π_M).
5. Save per-pair + aggregate to outputs/logs/region_level_eval.json.

Usage:
    PYTHONPATH=src python pipeline/05j_region_level_eval.py
    PYTHONPATH=src python pipeline/05j_region_level_eval.py --pi-file pi_fc_plus_SC_with_atlas_regions.npy
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from otter.data import load_cached                                   # noqa: E402
from otter.data.atlas_regions import (                                # noqa: E402
    assign_atlas_labels, ATLAS_PATHS,
)
from otter.eval.region_level import (                                 # noqa: E402
    evaluate_region_level,
    column_permuted_null,
    source_permuted_null,
)

# Reuse mask-building infrastructure from 05f
sys.path.insert(0, str(ROOT / "pipeline"))
from importlib import import_module                                   # noqa: E402
_beau = import_module("05f_beauchamp_validation")
BEAUCHAMP_PAIRS      = _beau.BEAUCHAMP_PAIRS
HUMAN_REGION_MNI     = _beau.HUMAN_REGION_MNI
parse_dsurqe_tree    = _beau.parse_dsurqe_tree
assign_dsurqe_labels = _beau.assign_dsurqe_labels
assign_human_region_membership = _beau.assign_human_region_membership

# Hippocampal + medulla pairs are "novel" w.r.t. our supervision (no Garin anchor).
# All other Beauchamp pairs overlap with our 21 Garin anchors.
HIPPOCAMPAL_OR_MISSING = {
    "Subiculum", "Field CA1", "Field CA2", "Field CA3", "Dentate gyrus",
    "Medulla",
}


ANN  = ROOT / "outputs" / "anndata"
LOG  = ROOT / "outputs" / "logs"; LOG.mkdir(parents=True, exist_ok=True)
EXT  = ROOT / "data_external" / "MouseHumanTranscriptomicSimilarity"
COUP = ROOT / "outputs" / "coupling"


def build_jubrain_candidate_masks(
    H_var, min_parcels: int = 3,
) -> dict[str, np.ndarray]:
    """Build candidate masks from JuBrain-184 atlas labels.

    Returns one mask per non-zero JuBrain ID with at least ``min_parcels``
    members. Names follow the convention ``"jubrain_{id}"``. JuBrain covers
    ~60% of human parcels; the remaining parcels (mostly cortical regions
    not in JuBrain's cyto-architectonic vocabulary) are unassigned.
    """
    jubrain_ids = assign_atlas_labels(H_var, "jubrain_184",
                                       ROOT / ATLAS_PATHS["jubrain_184"])
    out = {}
    for jid in np.unique(jubrain_ids):
        if jid == 0:
            continue
        mask = jubrain_ids == jid
        if mask.sum() < min_parcels:
            continue
        out[f"jubrain_{int(jid)}"] = mask
    return out


def map_beauchamp_to_jubrain(
    beauchamp_masks: dict[str, np.ndarray],
    jubrain_masks: dict[str, np.ndarray],
) -> dict[str, str]:
    """For each Beauchamp human region, find the best-matching JuBrain region
    by Jaccard overlap. Used to relabel pairs so eval can target the
    JuBrain-named region directly.

    Returns ``{beauchamp_name: jubrain_name_or_None}``. If no JuBrain region
    has overlap >= 0.1, the Beauchamp target stays separate (we'll union it
    into the candidate set).
    """
    out = {}
    for bname, bmask in beauchamp_masks.items():
        if bmask is None or bmask.sum() == 0:
            out[bname] = None
            continue
        best_jname, best_jacc = None, 0.0
        for jname, jmask in jubrain_masks.items():
            inter = int((bmask & jmask).sum())
            union = int((bmask | jmask).sum())
            jacc = inter / union if union > 0 else 0
            if jacc > best_jacc:
                best_jname, best_jacc = jname, jacc
        out[bname] = best_jname if best_jacc >= 0.1 else None
    return out


def main(args):
    print(f"Loading π from {args.pi_file}")
    pi = np.load(COUP / args.pi_file).astype(np.float64)
    print(f"  shape={pi.shape}")

    M, _ = load_cached("mouse", cache_dir=ANN)
    H, _ = load_cached("human", cache_dir=ANN)
    assert pi.shape == (len(M.var), len(H.var))

    # ---- Mouse-side region masks (same as 05f)
    name_to_dsurqe_labels = parse_dsurqe_tree(EXT / "AMBA/data/DSURQE_tree.json")
    parcel_dsurqe = assign_dsurqe_labels(
        M, EXT / "AMBA/data/imaging/DSURQE_CCFv3_labels_200um.mnc",
    )
    mouse_masks: dict[str, np.ndarray] = {}
    for mname, _ in BEAUCHAMP_PAIRS:
        if mname not in name_to_dsurqe_labels:
            continue
        m_lbl_set = name_to_dsurqe_labels[mname]
        mouse_masks[mname] = np.isin(parcel_dsurqe, list(m_lbl_set))

    # ---- Human-side region masks
    h_membership = assign_human_region_membership(H, HUMAN_REGION_MNI)
    beauchamp_masks = {
        hname: mask for hname, mask in h_membership.items() if mask is not None
    }

    if args.candidate_set == "beauchamp22":
        candidate_masks = beauchamp_masks
        pairs_to_run = BEAUCHAMP_PAIRS
        print(f"\nCandidate set: Beauchamp-22 = {len(candidate_masks)} human regions")
    elif args.candidate_set == "jubrain_union":
        jubrain_masks = build_jubrain_candidate_masks(H.var)
        print(f"\nLoaded JuBrain candidate masks: {len(jubrain_masks)} regions "
              f"(>= 3 parcels each)")
        # Map Beauchamp targets to nearest JuBrain region by Jaccard overlap.
        # For Beauchamp targets with no good JuBrain match, keep the
        # hand-curated Beauchamp mask as an extra candidate.
        beauchamp_to_jubrain = map_beauchamp_to_jubrain(beauchamp_masks, jubrain_masks)
        candidate_masks = dict(jubrain_masks)
        renamed_pairs = []
        for m_name, b_h_name in BEAUCHAMP_PAIRS:
            mapped = beauchamp_to_jubrain.get(b_h_name)
            if mapped is not None:
                renamed_pairs.append((m_name, mapped))
            elif b_h_name in beauchamp_masks:
                # Keep Beauchamp's hand-curated target as an extra candidate
                candidate_masks[b_h_name] = beauchamp_masks[b_h_name]
                renamed_pairs.append((m_name, b_h_name))
            else:
                renamed_pairs.append((m_name, b_h_name))   # will get skipped
        pairs_to_run = renamed_pairs
        print(f"Candidate set: JuBrain ∪ Beauchamp-extras = "
              f"{len(candidate_masks)} regions")
        # Print the Beauchamp → JuBrain mapping
        print("\nBeauchamp → JuBrain mapping (by Jaccard overlap):")
        for bname, jname in beauchamp_to_jubrain.items():
            tag = jname if jname else "(no JuBrain match, kept as Beauchamp mask)"
            print(f"  {bname:25s} -> {tag}")
    else:
        raise ValueError(f"unknown --candidate-set {args.candidate_set!r}")

    if args.candidate_set == "beauchamp22":
        for hname, mask in candidate_masks.items():
            print(f"  {hname:25s} {int(mask.sum()):4d} parcels")

    anchor_overlap = {m: (m not in HIPPOCAMPAL_OR_MISSING) for m, _ in BEAUCHAMP_PAIRS}

    # ---- Region-level top-K
    print(f"\n=== Region-level top-K (candidate set: {args.candidate_set}) ===")
    res = evaluate_region_level(
        pi, pairs_to_run, mouse_masks, candidate_masks,
        k_list=(1, 3, 5),
        anchor_overlap=anchor_overlap,
    )
    res["candidate_set"] = args.candidate_set

    # Pretty-print per-pair
    print(f"\n{'pair':52s} {'n_m':>4s} {'n_h':>4s} {'rank':>5s} "
          f"{'mass':>7s} {'fold':>6s} {'q1':>4s} {'q3':>4s} {'q5':>4s} "
          f"{'cov':>5s} {'anchor?':>8s}")
    print("-" * 120)
    for p in res["per_pair"]:
        print(f"  {p['pair']:50s} {p['n_mouse_parcels']:>4d} {p['n_human_parcels']:>4d} "
              f"{p['rank']:>5d} {p['score_true']:>7.4f} {p['fold_enrichment']:>6.2f} "
              f"{'Y' if p['qualified_top_k_hits'].get('1') else '-':>4s} "
              f"{'Y' if p['qualified_top_k_hits'].get('3') else '-':>4s} "
              f"{'Y' if p['qualified_top_k_hits'].get('5') else '-':>4s} "
              f"{p['total_mass_in_candidates']:>5.2f} "
              f"{'YES' if p['is_anchor_overlapping'] else 'no':>8s}")
    if res["skipped"]:
        print(f"\nSkipped: {len(res['skipped'])}")
        for s in res["skipped"]:
            print(f"  - {s}")

    agg = res["aggregate"]
    print("\n--- Aggregate (weighted by n_mouse_parcels) ---")
    print(f"  n_pairs_evaluated       = {res['n_pairs_evaluated']}")
    print(f"  n_candidates            = {res['n_candidates']}")
    print(f"  top-1 (rank)            = {agg['top_k'][1]:.1%}    top-3 = {agg['top_k'][3]:.1%}    top-5 = {agg['top_k'][5]:.1%}")
    print(f"  qualified top-1         = {agg['qualified_top_k'][1]:.1%}    top-3 = {agg['qualified_top_k'][3]:.1%}    top-5 = {agg['qualified_top_k'][5]:.1%}")
    print(f"     (qualified = rank ≤ k AND fold_enrichment ≥ 1.0)")
    print(f"  mean rank               = {agg['mean_rank']:.2f}  (best=1, worst={res['n_candidates']})")
    print(f"  median rank             = {agg['median_rank']:.1f}")
    print(f"  mean fold enrichment    = {agg['mean_fold_enrichment']:.2f}x")
    print(f"  median fold enrichment  = {agg['median_fold_enrichment']:.2f}x")
    print(f"  mean total candidate-set mass coverage = {agg['mean_total_mass_in_candidates']:.1%}")

    for label in ("anchor_overlapping", "novel"):
        if label in res:
            d = res[label]
            print(f"\n  [{label:20s}] n_pairs={d['n_pairs']} n_parcels={d['n_parcels']}")
            print(f"      rank-only:  top-1 = {d['top_k'][1]:.1%}, top-3 = {d['top_k'][3]:.1%}, "
                  f"top-5 = {d['top_k'][5]:.1%}")
            print(f"      qualified:  top-1 = {d['qualified_top_k'][1]:.1%}, top-3 = {d['qualified_top_k'][3]:.1%}, "
                  f"top-5 = {d['qualified_top_k'][5]:.1%}")
            print(f"      mean rank = {d['mean_rank']:.2f}, "
                  f"mean fold = {d['mean_fold_enrichment']:.2f}x")

    # ---- Nulls
    if not args.skip_nulls:
        print(f"\n=== Column-permuted null (n_trials={args.n_null_trials}) ===")
        null_col = column_permuted_null(
            pi, pairs_to_run, mouse_masks, candidate_masks,
            k_list=(1, 3, 5), n_trials=args.n_null_trials,
        )
        print(f"  null top-1 = {null_col['null_topk_mean'][1]:.1%} ± {null_col['null_topk_std'][1]:.1%}")
        print(f"  null top-3 = {null_col['null_topk_mean'][3]:.1%} ± {null_col['null_topk_std'][3]:.1%}")
        print(f"  null top-5 = {null_col['null_topk_mean'][5]:.1%} ± {null_col['null_topk_std'][5]:.1%}")
        print(f"  null fold  = {null_col['null_fold_mean']:.2f}x ± {null_col['null_fold_std']:.2f}")

        for k in (1, 3, 5):
            z = (agg["top_k"][k] - null_col["null_topk_mean"][k]) / max(null_col["null_topk_std"][k], 1e-9)
            print(f"  z top-{k}  = {z:+.1f}")
        z_fold = (agg["mean_fold_enrichment"] - null_col["null_fold_mean"]) / max(null_col["null_fold_std"], 1e-9)
        print(f"  z fold    = {z_fold:+.1f}")

        print(f"\n=== Source-permuted null (n_trials={args.n_null_trials}) ===")
        null_src = source_permuted_null(
            pi, pairs_to_run, mouse_masks, candidate_masks,
            k_list=(1, 3, 5), n_trials=args.n_null_trials,
        )
        print(f"  null top-1 = {null_src['null_topk_mean'][1]:.1%} ± {null_src['null_topk_std'][1]:.1%}")
        print(f"  null top-3 = {null_src['null_topk_mean'][3]:.1%} ± {null_src['null_topk_std'][3]:.1%}")
        print(f"  null top-5 = {null_src['null_topk_mean'][5]:.1%} ± {null_src['null_topk_std'][5]:.1%}")
        print(f"  null fold  = {null_src['null_fold_mean']:.2f}x ± {null_src['null_fold_std']:.2f}")
        for k in (1, 3, 5):
            z = (agg["top_k"][k] - null_src["null_topk_mean"][k]) / max(null_src["null_topk_std"][k], 1e-9)
            print(f"  z top-{k}  = {z:+.1f}")
        z_fold = (agg["mean_fold_enrichment"] - null_src["null_fold_mean"]) / max(null_src["null_fold_std"], 1e-9)
        print(f"  z fold    = {z_fold:+.1f}")

        res["null_column_permuted"] = null_col
        res["null_source_permuted"] = null_src

    # ---- Save
    out_path = LOG / args.output
    out_path.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi-file", default="pi_canonical.npy",
                    help="filename in outputs/coupling/")
    ap.add_argument("--output", default="region_level_eval.json",
                    help="output filename in outputs/logs/")
    ap.add_argument("--candidate-set",
                    choices=("beauchamp22", "jubrain_union"),
                    default="beauchamp22",
                    help="Beauchamp-22 (21 evaluable) or JuBrain ∪ extras (~150)")
    ap.add_argument("--n-null-trials", type=int, default=100)
    ap.add_argument("--skip-nulls", action="store_true")
    main(ap.parse_args())
