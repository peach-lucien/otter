"""Visualise the HOMER × TransBrain 2025 methods-comparison."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def main():
    j = json.loads((ROOT / "outputs/logs/transbrain_2025_benchmark.json").read_text())
    a = j["homology_benchmark_cortex"]
    g = j["head_to_head_gradient"]
    au = j["head_to_head_autism"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # --- Panel 1: homology benchmark — centroid distance -------------------
    ax = axes[0]
    bars = ax.bar([0, 1], [a["centroid_dist_mm"], a["null_centroid_dist_mm"]],
                  color=["#2a9d8f", "#cccccc"], width=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["HOMER π", "permuted-π\nnull"])
    ax.set_ylabel("Distance: HOMER's predicted human centroid\n"
                  "→ literature homolog (mm)")
    for b, v in zip(bars, [a["centroid_dist_mm"], a["null_centroid_dist_mm"]]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}", ha="center",
                fontsize=9)
    ax.set_title(f"1 · Homology benchmark ({a['n_regions']} mouse regions)\n"
                 f"HOMER lands near the literature homolog (p < 0.001)\n"
                 f"top-3 {a['top3']:.0%} on {a['n_bn_regions']} BN regions "
                 f"(null {a['null_top3_mean']:.0%}, p = "
                 f"{a['null_top3_empirical_p']:.3f})", fontsize=9)

    # --- Panel 2: head-to-head bars ---------------------------------------
    ax = axes[1]
    labels = ["HOMER\nvs human", "TransBrain\nvs human", "HOMER vs\nTransBrain"]
    grad = [g["homer_vs_reference"], g["transbrain_vs_reference"],
            g["homer_vs_transbrain"]]
    x = np.arange(3)
    ax.bar(x, grad, width=0.55, color=["#264653", "#e76f51", "#8ab17d"])
    for i, v in enumerate(grad):
        ax.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("|Pearson r|  (Brainnetome regions)")
    ax.set_ylim(0, max(grad) * 1.25)
    ax.set_title("2 · Head-to-head — resting-fMRI gradient\n"
                 "both methods recover the human gradient;\n"
                 "TransBrain (region-level tool) scores higher", fontsize=9)

    # --- Panel 3: per-individual autism risk scores -----------------------
    ax = axes[2]
    rh = np.array(au["risk_homer"])
    rt = np.array(au["risk_transbrain"])
    ax.scatter(rt, rh, s=20, alpha=0.55, color="#2a9d8f", edgecolor="none")
    lim = max(np.abs(np.r_[rh, rt]).max(), 0.05) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.7)
    ax.axhline(0, color="black", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.4)
    ax.set_xlabel("TransBrain ASD risk score")
    ax.set_ylabel("HOMER ASD risk score")
    ax.set_title(f"3 · Head-to-head — Magel2 autism pattern\n"
                 f"per-individual risk scores, {au['n_individuals']} ASD subjects\n"
                 f"method concordance r = {au['risk_score_concordance']:+.2f} "
                 f"(noisy phenotype — methods diverge)", fontsize=9)

    plt.suptitle("HOMER × TransBrain 2025 — honest methods-landscape comparison "
                 "against a published sibling translator", fontsize=12, y=1.04)
    plt.tight_layout()
    out = ROOT / "outputs" / "figures" / "transbrain_2025_benchmark.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
