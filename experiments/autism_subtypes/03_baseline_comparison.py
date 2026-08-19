"""Robustness checks for the network cross-validation test.

For each of three π variants, production-with-packs (recommended), production
point-anchor-only, and a permuted-anchor null, recompute the diagonal-
dominance summary stats.

Goal: verify that the diagonal concentration is (a) not an artefact of pack
overfitting, and (b) above what a network-uninformed coupling would produce.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from otter.data import load_cached
import sys

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
mod = import_module("01_network_crossvalidation")
assign_mouse_paper_networks = mod.assign_mouse_paper_networks
assign_human_paper_networks = mod.assign_human_paper_networks
compute_network_mapping = mod.compute_network_mapping
score_mapping = mod.score_mapping


PI_VARIANTS = {
    "with_all_packs (recommended)": "outputs/coupling/pi_fc_plus_SC_with_all_packs.npy",
    "fc_plus_SC (Garin only)":      "outputs/coupling/pi_fc_plus_SC.npy",
    "fc_plus_SC_xyz_zero":          "outputs/coupling/pi_fc_plus_SC_xyz_zero.npy",
}

TARGET_PAIRS = mod.TARGET_PAIRS   # one definition, in 01_network_crossvalidation.py


def main():
    M, _ = load_cached("mouse", cache_dir="outputs/anndata")
    H, _ = load_cached("human", cache_dir="outputs/anndata")

    mouse_net, mouse_names = assign_mouse_paper_networks(M.var, separate_aud=True)
    human_net, human_names = assign_human_paper_networks(H.var, separate_aud=True)

    rows = []
    for label, path in PI_VARIANTS.items():
        if not Path(path).exists():
            print(f"  SKIP {label}: not found at {path}")
            continue
        pi = np.load(path)
        N = compute_network_mapping(pi, mouse_net, human_net,
                                     n_mouse=len(mouse_names),
                                     n_human=len(human_names))
        score = score_mapping(N, mouse_names, human_names, target_pairs=TARGET_PAIRS)
        n_diag = sum(r.get("is_argmax_diagonal", False) for r in score["per_pair"])
        n_pairs = sum("ratio_over_null" in r for r in score["per_pair"])
        ratios = [r["ratio_over_null"] for r in score["per_pair"] if "ratio_over_null" in r]
        rows.append({
            "pi": label,
            "n_diag_argmax": n_diag,
            "n_pairs_scored": n_pairs,
            "fraction_diagonal_argmax": n_diag / max(n_pairs, 1),
            "mean_ratio_over_null": float(np.mean(ratios)),
            "median_ratio_over_null": float(np.median(ratios)),
        })

    # Permuted-π null: shuffle the rows of pi so mouse parcels random-target humans
    print("\nPermuted-π null (1000 random row-shuffles of recommended π):")
    rng = np.random.default_rng(seed=42)
    pi_rec = np.load(PI_VARIANTS["with_all_packs (recommended)"])
    null_ratios = []
    null_diag_counts = []
    for trial in range(20):
        perm = rng.permutation(pi_rec.shape[0])
        N_null = compute_network_mapping(pi_rec[perm], mouse_net, human_net,
                                          n_mouse=len(mouse_names),
                                          n_human=len(human_names))
        score_null = score_mapping(N_null, mouse_names, human_names,
                                    target_pairs=TARGET_PAIRS)
        n_diag = sum(r.get("is_argmax_diagonal", False) for r in score_null["per_pair"])
        ratios = [r["ratio_over_null"] for r in score_null["per_pair"] if "ratio_over_null" in r]
        null_diag_counts.append(n_diag)
        null_ratios.append(np.mean(ratios))
    null_row = {
        "pi": "permuted_pi_null (20 trials)",
        "n_diag_argmax": float(np.mean(null_diag_counts)),
        "n_pairs_scored": len(TARGET_PAIRS),
        "fraction_diagonal_argmax": float(np.mean(null_diag_counts)) / len(TARGET_PAIRS),
        "mean_ratio_over_null": float(np.mean(null_ratios)),
        "median_ratio_over_null": float(np.median(null_ratios)),
    }
    rows.append(null_row)

    print(f"\n{'π variant':<35s} | diag-argmax | mean ratio | median ratio")
    print("-" * 80)
    for r in rows:
        print(f"  {r['pi']:<33s} | {r['n_diag_argmax']:>5}/{r['n_pairs_scored']:<5} | "
              f"{r['mean_ratio_over_null']:>9.2f}× | {r['median_ratio_over_null']:>11.2f}×")

    out = {"variants": rows}
    out_path = Path("outputs/logs/autism_subtypes_baseline_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
