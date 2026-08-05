"""Visualise the OTTER × Fulcher 2019 multimodal-gradient result (3 panels)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def _scatter(ax, x, y, color, xlabel, ylabel, title):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    ax.scatter(x, y, s=14, alpha=0.55, color=color, edgecolor="none")
    z = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, z[0] * xs + z[1], "k--", linewidth=0.9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9.5)


def main():
    j = json.loads((ROOT / "outputs/logs/fulcher_2019_gradient.json").read_text())
    pred_t1t2 = np.array(j["predicted_t1t2_region"])
    pred_cyto = np.array(j["predicted_cytoarch_region"])
    myelin = np.array(j["myelin_region"])
    grad = np.array(j["gradient_region"])
    territory = np.array(j["territory_mask"], dtype=bool)

    p1, p2, p3 = (j["panel1_t1t2_vs_myelin"], j["panel2_gradient_territory"],
                  j["panel3_cytoarch_vs_myelin"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # --- Panel 1: T1w:T2w → human myelin -----------------------------------
    _scatter(
        axes[0], myelin, pred_t1t2, "#2a9d8f",
        "Observed human T1w/T2w myelin (Schaefer-400 region)",
        "Predicted via OTTER π\n(mouse T1w:T2w translated)",
        f"1 · Mouse myelin hierarchy → human myelin\n"
        f"Pearson r = {p1['pearson_r']:+.3f}   ρ = {p1['spearman_r']:+.3f}   "
        f"p = {p1['pearson_p_analytical']:.1e}\n"
        f"n = {p1['n_regions']} regions   empirical p = "
        f"{p1['null']['empirical_p']:.3f} vs permuted-π null",
    )

    # --- Panel 2: routed territory on the principal gradient ---------------
    ax = axes[1]
    g_all = grad[np.isfinite(grad)]
    g_terr = grad[np.isfinite(grad) & territory]
    parts = ax.violinplot([g_all, g_terr], showmeans=True, showextrema=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(["#cccccc", "#e76f51"][i])
        pc.set_alpha(0.75)
    for key in ("cbars", "cmins", "cmaxes", "cmeans"):
        parts[key].set_color("#333333")
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"All human cortex\n(n={len(g_all)})",
                        f"Routed territory\n(n={len(g_terr)})"])
    ax.set_ylabel("Principal connectivity gradient\n(Margulies/Huntenburg, region mean)")
    ax.set_title(
        f"2 · Mouse isocortex maps onto a gradient-compressed slice\n"
        f"gradient SD  all cortex {p2['gradient_sd_all_cortex']:.4f}  →  "
        f"territory {p2['gradient_sd_routed_territory']:.4f}\n"
        f"compression ×{p2['compression_ratio']:.2f}   "
        f"(predicted-vs-gradient r = {p2['predicted_vs_gradient_r']:+.3f})",
        fontsize=9.5,
    )

    # --- Panel 3: cytoarchitecture → human myelin --------------------------
    _scatter(
        axes[2], myelin, pred_cyto, "#264653",
        "Observed human T1w/T2w myelin (Schaefer-400 region)",
        "Predicted via OTTER π\n(mouse cytoarchitecture translated)",
        f"3 · Independent modality, cytoarchitecture → human myelin\n"
        f"Pearson r = {p3['pearson_r']:+.3f}   ρ = {p3['spearman_r']:+.3f}   "
        f"p = {p3['pearson_p_analytical']:.1e}\n"
        f"n = {p3['n_regions']} regions   empirical p = "
        f"{p3['null']['empirical_p']:.3f} vs permuted-π null",
    )

    plt.suptitle(
        "OTTER × Fulcher 2019, π translates the mouse multimodal cortical "
        "hierarchy onto the human cortex",
        fontsize=12, y=1.04,
    )
    plt.tight_layout()
    out = ROOT / "outputs" / "figures" / "fulcher_2019_multimodal_gradient.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
