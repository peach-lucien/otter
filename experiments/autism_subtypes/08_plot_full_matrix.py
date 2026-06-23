"""Visualise Test 2c, full per-network-pair matrix translation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def main():
    j = json.loads(Path("outputs/logs/autism_subtypes_full_matrix.json").read_text())
    obs = np.array(j["delta_human_observed_flat"])
    pred = np.array(j["delta_human_predicted_flat"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: scatter, predicted vs observed per matrix element
    # Color by quadrant: agree on sign (green) vs disagree (red); zero-obs are gray
    same_sign = np.sign(pred) == np.sign(obs)
    nonzero = (np.abs(pred) > 1e-6) & (np.abs(obs) > 1e-6)
    colors = np.where(nonzero & same_sign, "#2a9d8f",
              np.where(nonzero & ~same_sign, "#e76f51", "#aaaaaa"))
    ax1.scatter(obs, pred, c=colors, s=70, alpha=0.85, edgecolor="black", linewidth=0.4)

    # Best-fit line
    z = np.polyfit(obs, pred, 1)
    xs = np.linspace(obs.min(), obs.max(), 50)
    ax1.plot(xs, z[0]*xs + z[1], color="black", linewidth=0.8, linestyle="--",
             label=f"linear fit (slope={z[0]:+.3f})")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.axvline(0, color="black", linewidth=0.5)
    ax1.set_xlabel("Observed Δ (Pagani Fig 4e, hyper − hypo)")
    ax1.set_ylabel("Predicted Δ via HOMER π")
    ax1.set_title(f"Full network-pair Δ-matrix translation\n"
                  f"Pearson r = {j['pearson_r']:+.3f}  (n=36, analytical p={j['pearson_p_analytical']:.4f}, "
                  f"empirical p={j['null']['pearson_empirical_p']:.3f})")
    legend = [
        mpatches.Patch(facecolor="#2a9d8f", label="same sign"),
        mpatches.Patch(facecolor="#e76f51", label="opposite sign"),
        mpatches.Patch(facecolor="#aaaaaa", label="zero in mouse or human"),
    ]
    ax1.legend(handles=legend + [ax1.get_legend_handles_labels()[0][0]], loc="upper left", fontsize=8)

    # Right: null distribution
    null_mean = j["null"]["pearson_mean"]
    null_ci = j["null"]["pearson_ci95"]
    ax2.axvspan(null_ci[0], null_ci[1], color="#cccccc", alpha=0.6, label="null 95% CI")
    ax2.axvline(null_mean, color="gray", linestyle="--", label=f"null mean {null_mean:+.2f}")
    ax2.axvline(j["pearson_r"], color="#e76f51", linewidth=2.5,
                label=f"observed r={j['pearson_r']:+.2f}")
    ax2.set_xlim(-1, 1); ax2.set_yticks([])
    ax2.set_xlabel("Pearson r (predicted vs observed, 36 elements)")
    ax2.set_title("Permuted-π null (200 trials)")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    out = Path("outputs/figures/autism_subtypes_full_matrix.png")
    plt.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
