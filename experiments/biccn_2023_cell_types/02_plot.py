"""Visualise the BICCN cell-type marker cross-species result."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[2]


def main():
    j = json.loads((ROOT / "outputs/logs/biccn_2023_cell_types.json").read_text())
    results = j["per_marker"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    # Panel 1: per-marker r with null CI, sorted by r
    ax = axes[0]
    results_sorted = sorted(results, key=lambda r: r["pearson_r"], reverse=True)
    names = [r["marker"] for r in results_sorted]
    rs = [r["pearson_r"] for r in results_sorted]
    null_means = [r["null_mean"] for r in results_sorted]
    null_lo = [r["null_ci95"][0] for r in results_sorted]
    null_hi = [r["null_ci95"][1] for r in results_sorted]
    sig = [r["empirical_p"] < 0.05 for r in results_sorted]
    classes = [r["class"] for r in results_sorted]
    # Color by cell-type class
    class_color = {
        "interneuron": "#264653",
        "gabaergic_synth": "#2a9d8f",
        "glutamatergic": "#e9c46a",
        "astrocyte": "#f4a261",
        "oligodendrocyte": "#e76f51",
        "microglia": "#8338ec",
        "dopaminergic": "#ff006e",
        "serotonergic": "#3a86ff",
    }
    bar_colors = [class_color.get(c, "#999") for c in classes]
    edges = ["black" if s else "lightgray" for s in sig]
    edge_w = [1.5 if s else 0.5 for s in sig]
    y = np.arange(len(names))
    for i in range(len(names)):
        ax.barh(y[i], rs[i], color=bar_colors[i], edgecolor=edges[i], linewidth=edge_w[i])
        ax.plot([null_lo[i], null_hi[i]], [y[i], y[i]], color="black", linewidth=0.7, alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Pearson r (predicted vs observed cross-species)")
    n_sig = sum(sig)
    ax.set_title(f"Per-marker cross-species translation\n"
                 f"{n_sig}/{len(names)} markers empirical p < 0.05 (bold edge)")
    # Legend by class
    legend_handles = [mpatches.Patch(facecolor=class_color[c], edgecolor="black", label=c)
                       for c in sorted(class_color.keys())
                       if any(r["class"] == c for r in results)]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, ncol=2)

    # Panel 2: class-level summary + side-by-side vs Hodge
    ax = axes[1]
    class_summary = j["per_class"]
    classes_sorted = sorted(class_summary, key=lambda c: c["mean_pearson_r"], reverse=True)
    class_names = [c["class"] for c in classes_sorted]
    class_rs = [c["mean_pearson_r"] for c in classes_sorted]
    class_colors = [class_color.get(c, "#999") for c in class_names]
    y = np.arange(len(class_names))
    ax.barh(y, class_rs, color=class_colors, edgecolor="black", linewidth=0.7)
    for i, c in enumerate(classes_sorted):
        ax.text(class_rs[i] + 0.003, y[i], f"{c['n_significant']}/{c['n_markers']} sig",
                fontsize=8, va="center")
    # Add Hodge layer markers as comparison
    if "comparison" in j:
        ax.barh(len(class_names) + 0.5, j["comparison"]["hodge_mean_pearson_r"],
                color="#aaaaaa", edgecolor="black", linewidth=0.7)
        ax.text(j["comparison"]["hodge_mean_pearson_r"] + 0.003, len(class_names) + 0.5,
                f"{j['comparison']['hodge_n_significant']}/7 sig (Hodge 2019)",
                fontsize=8, va="center")
        ax.set_yticks(list(y) + [len(class_names) + 0.5])
        ax.set_yticklabels(class_names + ["layer markers\n(Hodge 2019)"], fontsize=9)
    else:
        ax.set_yticks(y); ax.set_yticklabels(class_names, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Mean Pearson r per cell-type class")
    biccn_mean = float(np.mean([r["pearson_r"] for r in results]))
    ax.set_title(f"Per-class summary + comparison to Hodge layer markers\n"
                 f"BICCN overall mean r = {biccn_mean:+.3f}")

    plt.suptitle(
        "HOMER × BICCN cell-type marker validation. HOMER preserves regionally-concentrated\n"
        "cell-type signals (glia, dopamine) but not broadly-distributed cortical class markers (interneurons)",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    out = ROOT / "outputs" / "figures" / "biccn_2023_cell_types.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
