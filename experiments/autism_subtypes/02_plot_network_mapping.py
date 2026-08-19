"""Visualise the mouse↔human network mapping matrix from 01_network_crossvalidation.

Produces a heatmap of the row-normalised mapping matrix, with the canonical
name-based pairs marked. Saved to outputs/figures/.
"""
from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parent))
# One definition of the canonical pairs, in 01_network_crossvalidation.py.
TARGET_PAIRS = import_module("01_network_crossvalidation").TARGET_PAIRS


def main():
    j = json.loads(Path("outputs/logs/autism_subtypes_network_crossval.json").read_text())
    mouse_names = j["mouse_networks"]
    human_names = j["human_networks"]
    N = np.array(j["row_normalised_matrix"])
    N_raw = np.array(j["mapping_matrix"])

    # Drop empty mouse-net rows for readability
    keep = N_raw.sum(axis=1) > 0
    mouse_names_k = [n for i, n in enumerate(mouse_names) if keep[i]]
    N_k = N[keep] * 100

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    im = ax.imshow(N_k, cmap="viridis", aspect="auto", vmin=0, vmax=60)
    ax.set_xticks(range(len(human_names)))
    ax.set_xticklabels(human_names, rotation=45, ha="right")
    ax.set_yticks(range(len(mouse_names_k)))
    ax.set_yticklabels(mouse_names_k)
    ax.set_xlabel("Human network (Schaefer-400 → Yeo-7 + Subcortical)")
    ax.set_ylabel("Mouse network (OTTER PAIRID_TO_NETWORK → paper-aligned)")
    ax.set_title("OTTER π mass aggregated by network\n"
                 "(each row = mouse network; sums to 100%)", fontsize=11)

    # Annotate cells
    for i in range(N_k.shape[0]):
        for k in range(N_k.shape[1]):
            v = N_k[i, k]
            if v >= 5:
                ax.text(k, i, f"{v:.0f}", ha="center", va="center",
                        color="white" if v < 30 else "black", fontsize=8)

    # Mark canonical pairs with red boxes. The same list the scoring uses, so the panel and
    # the reported count cannot disagree.
    for m, h in TARGET_PAIRS:
        if m in mouse_names_k and h in human_names:
            i = mouse_names_k.index(m)
            k = human_names.index(h)
            rect = mpatches.Rectangle((k - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor="red", linewidth=2.0)
            ax.add_patch(rect)
    ax.legend(handles=[mpatches.Patch(facecolor="none", edgecolor="red",
                                       label="canonical name-based pair")],
              loc="upper right", framealpha=0.9)

    plt.colorbar(im, ax=ax, label="row mass %")
    plt.tight_layout()
    out_path = Path("outputs/figures/autism_subtypes_network_mapping.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")

    # Also write a small diagonal-dominance bar plot
    pairs = j["target_pair_scores"]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    labels = [f"{p['mouse_net']}→{p['human_net']}" for p in pairs]
    ratios = [p.get("ratio_over_null", 0) for p in pairs]
    colors = ["#2a9d8f" if p.get("is_argmax_diagonal") else "#e76f51" for p in pairs]
    x = np.arange(len(labels))
    ax.bar(x, ratios, color=colors)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.7, label="null (uniform π)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("OTTER mass on target / null expectation")
    ax.set_title("Pagani 2026 canonical pairs: how much OTTER mass concentrates on the name-matched human network")
    legend = [
        mpatches.Patch(facecolor="#2a9d8f", label="argmax human-net = target"),
        mpatches.Patch(facecolor="#e76f51", label="argmax human-net ≠ target"),
    ]
    ax.legend(handles=legend, loc="upper right")
    plt.tight_layout()
    out_path2 = Path("outputs/figures/autism_subtypes_diagonal_dominance.png")
    plt.savefig(out_path2, dpi=150)
    print(f"Wrote {out_path2}")


if __name__ == "__main__":
    main()
