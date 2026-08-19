"""Visualise the Coletta-style cross-species RSN correspondence + coherence test."""
from __future__ import annotations
import json
import sys
from importlib import import_module
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "coletta_2020_cross_species_rsn"))
# One definition of the canonical pairs, in 01_correspondence_validation.py.
TARGET_PAIRS = set(import_module("01_correspondence_validation").TARGET_PAIRS)


def main():
    j = json.loads((ROOT / "outputs/logs/coletta_2020_cross_species_rsn.json").read_text())

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Panel 1: Labeled correspondence, ratio over null per pair
    ax = axes[0]
    A = j["sub_test_A_labeled_correspondence"]
    pairs = A["per_pair_scores"]
    names = [f"{p['mouse_net']} → {p['human_net']}" for p in pairs if 'ratio_over_null' in p]
    ratios = [p['ratio_over_null'] for p in pairs if 'ratio_over_null' in p]
    is_diag = [p['is_argmax_diagonal'] for p in pairs if 'is_argmax_diagonal' in p]
    colors = ["#2a9d8f" if d else "#e76f51" for d in is_diag]
    y = np.arange(len(names))
    ax.barh(y, ratios, color=colors)
    ax.axvline(1.0, color="black", linewidth=0.5, linestyle="--", label="null (1×)")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("OTTER mass / null expectation")
    n_diag = A["n_diagonal_argmax"]; n_pairs = A["n_pairs_scored"]
    ax.set_title(f"Sub-test A: Labeled correspondence\n"
                 f"{n_diag}/{n_pairs} pairs are diagonal-argmax")
    legend = [
        mpatches.Patch(facecolor="#2a9d8f", label="diagonal argmax (correct)"),
        mpatches.Patch(facecolor="#e76f51", label="argmax goes elsewhere"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=8)

    # Panel 2: ICA labels and best-match Yeo
    ax = axes[1]
    B = j["sub_test_B_ica_data_driven"]
    corr = B["correspondence"]
    ic_labels = [f"IC{c['ic']}\n({c['mouse_label']})" for c in corr]
    pred = [c["best_match_yeo7"] for c in corr]
    # For visual: just show as text annotations
    ax.set_xlim(0, 10); ax.set_ylim(-0.5, len(corr) - 0.5)
    for i, (lbl, p) in enumerate(zip(ic_labels, pred)):
        mouse_lbl = corr[i]['mouse_label']
        is_match = (mouse_lbl, p) in TARGET_PAIRS
        c = "#2a9d8f" if is_match else "#999999"
        ax.text(0.5, len(corr) - 1 - i, lbl, fontsize=9, va="center", ha="left", family="monospace")
        ax.text(5, len(corr) - 1 - i, "→", fontsize=12, va="center", ha="center")
        ax.text(6, len(corr) - 1 - i, p, fontsize=10, va="center", ha="left", color=c, weight="bold")
    ax.axis("off")
    ax.set_title("Sub-test B: ICA-derived data-driven\n"
                 "Mouse ICA component → predicted Yeo-7 argmax\n"
                 "(green = match anatomical label; weaker test because ICA components\n"
                 " do not cleanly partition into single anatomical networks)", fontsize=9)

    # Panel 3: Network coherence, bar of real vs null spread
    ax = axes[2]
    C = j["sub_test_C_network_coherence"]
    nets = [r["network"] for r in C["per_network"]]
    real = [r["real_spread_mm"] for r in C["per_network"]]
    nulls = [r["null_mean_mm"] for r in C["per_network"]]
    more_compact = [r["more_compact_than_null"] for r in C["per_network"]]
    y = np.arange(len(nets))
    ax.barh(y - 0.2, real, 0.4,
            color=["#2a9d8f" if c else "#e76f51" for c in more_compact],
            label="OTTER (real)")
    ax.barh(y + 0.2, nulls, 0.4, color="#aaaaaa", label="permuted-π null mean")
    ax.set_yticks(y); ax.set_yticklabels(nets, fontsize=9)
    ax.set_xlabel("Centroid spread (mm), smaller = more compact")
    ax.set_title(f"Sub-test C: Network coherence\n"
                 f"{C['n_networks_more_compact_than_null']}/{len(C['per_network'])} networks "
                 f"are more compact than null")
    ax.legend(loc="lower right", fontsize=8)

    plt.suptitle(
        "OTTER × Coletta 2020, cross-species RSN correspondence + coherence",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    out = ROOT / "outputs" / "figures" / "coletta_2020_cross_species_rsn.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
