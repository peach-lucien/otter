"""Visualise the Hodge 2019 layer-marker cross-species result."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[2]


def main():
    j1 = json.loads((ROOT / "outputs/logs/hodge_2019_layer_markers.json").read_text())
    j2 = json.loads((ROOT / "outputs/logs/hodge_2019_layer_markers_refined.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: per-marker (all 2094 parcels)
    ax = axes[0]
    markers = j1["markers"]
    names = [m["gene"] for m in markers]
    rs    = [m["pearson_r"] for m in markers]
    nulls = [m["null_mean"] for m in markers]
    ci_lo = [m["null_ci95"][0] for m in markers]
    ci_hi = [m["null_ci95"][1] for m in markers]
    sig   = [m["empirical_p"] < 0.05 for m in markers]

    x = np.arange(len(names))
    ax.bar(x - 0.2, rs, 0.4,
           color=["#2a9d8f" if s else "#e76f51" for s in sig],
           label="HOMER predicted vs observed (cortex+subcortex)")
    ax.bar(x + 0.2, nulls, 0.4, color="#aaaaaa",
           label="permuted-π null mean")
    # Add CI bars on null
    for i, (lo, hi) in enumerate(zip(ci_lo, ci_hi)):
        ax.plot([x[i] + 0.2, x[i] + 0.2], [lo, hi], color="black", linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Pearson r")
    ax.set_title(f"Per-marker cross-species translation (all 2,094 parcels)\n"
                 f"Only RORB is above the null band — single significant marker")
    ax.legend(loc="upper left", fontsize=8)

    # Right: layer-group composites (cortex only)
    ax = axes[1]
    groups = j2["per_layer_group"]
    g_names = [g["layer_group"] for g in groups]
    g_rs = [g["pearson_r"] for g in groups]
    g_nulls = [g["null_mean"] for g in groups]
    g_ci_lo = [g["null_ci95"][0] for g in groups]
    g_ci_hi = [g["null_ci95"][1] for g in groups]
    g_sig = [g["empirical_p"] < 0.05 for g in groups]

    x = np.arange(len(g_names))
    ax.bar(x - 0.2, g_rs, 0.4,
           color=["#2a9d8f" if s else "#e76f51" for s in g_sig],
           label="HOMER predicted vs observed (cortex only)")
    ax.bar(x + 0.2, g_nulls, 0.4, color="#aaaaaa",
           label="permuted-π null mean")
    for i, (lo, hi) in enumerate(zip(g_ci_lo, g_ci_hi)):
        ax.plot([x[i] + 0.2, x[i] + 0.2], [lo, hi], color="black", linewidth=1)
    # Show contrast
    contrast = j2["upper_minus_deep_contrast"]
    x_c = len(g_names)
    ax.bar(x_c - 0.2, contrast["pearson_r"], 0.4,
           color="#2a9d8f" if contrast["empirical_p"] < 0.05 else "#e76f51")
    ax.bar(x_c + 0.2, contrast["null_mean"], 0.4, color="#aaaaaa")
    ax.plot([x_c + 0.2, x_c + 0.2],
            [contrast["null_ci95"][0], contrast["null_ci95"][1]],
            color="black", linewidth=1)

    g_names_extended = g_names + ["upper − deep\ncontrast"]
    ax.set_xticks(list(x) + [x_c]); ax.set_xticklabels(g_names_extended, rotation=30, ha="right")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Pearson r")
    ax.set_title(f"Layer-group composites (cortical parcels only, n=1,768)\n"
                 f"L4 (RORB) is the only group where HOMER signal beats null")
    ax.legend(loc="upper left", fontsize=8)

    plt.suptitle("Hodge 2019 layer-marker cross-species test — HOMER works at area level, not at layer level",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = ROOT / "outputs" / "figures" / "hodge_2019_layer_markers.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
