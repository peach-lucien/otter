"""Visualise the ABIDE per-subject OTTER-template result."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[3]


def main():
    j = json.loads((ROOT / "outputs/logs/autism_subtypes_abide.json").read_text())
    df = pd.read_csv(ROOT / "outputs/logs/abide_per_subject_scores.csv")
    df = df[df["valid"] == True]
    asd = df[df["DX_GROUP"] == 1]["otter_score"].dropna().values
    ctrl = df[df["DX_GROUP"] == 2]["otter_score"].dropna().values

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Left: violin/box of ASD vs CTRL scores
    ax = axes[0]
    bp = ax.boxplot([ctrl, asd], labels=[f"Control\n(n={len(ctrl)})", f"ASD\n(n={len(asd)})"],
                     showfliers=False, patch_artist=True)
    for box, c in zip(bp["boxes"], ["#2a9d8f", "#e76f51"]):
        box.set_facecolor(c); box.set_alpha(0.6)
    # Jittered points
    rng = np.random.default_rng(0)
    ax.scatter(1 + rng.normal(0, 0.04, len(ctrl)), ctrl,
               s=4, alpha=0.3, color="#2a9d8f")
    ax.scatter(2 + rng.normal(0, 0.04, len(asd)),  asd,
               s=4, alpha=0.3, color="#e76f51")
    ax.set_ylabel("OTTER cross-species template score\n(subject perturbation · template Δ)")
    ax.set_title(f"ASD vs Control\nMann-Whitney p = {j['mann_whitney_p']:.3f}, "
                 f"Cliff's δ = {j['cliffs_delta']:+.3f}")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")

    # Middle: KDE comparison
    ax = axes[1]
    kde_a = gaussian_kde(asd); kde_c = gaussian_kde(ctrl)
    xs = np.linspace(min(asd.min(), ctrl.min()), max(asd.max(), ctrl.max()), 300)
    ax.fill_between(xs, kde_c(xs), color="#2a9d8f", alpha=0.4, label=f"Control (n={len(ctrl)})")
    ax.fill_between(xs, kde_a(xs), color="#e76f51", alpha=0.4, label=f"ASD (n={len(asd)})")
    ax.set_xlabel("OTTER template score")
    ax.set_yticks([])
    ax.set_title("Density: distributions almost identical")
    ax.legend()
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--")

    # Right: GMM bimodality test on ASD only
    ax = axes[2]
    asd_density = kde_a(xs)
    ax.fill_between(xs, asd_density, color="#e76f51", alpha=0.5)
    ax.plot(xs, asd_density, color="#e76f51", linewidth=1.5)
    gmm = j["gmm_bimodality"]
    ax.set_title(f"ASD only, bimodality check\n"
                 f"Δ BIC (2-comp − 1-comp) = {gmm['delta_bic_2_minus_1']:+.1f} "
                 f"({'2-comp' if gmm['two_comp_preferred_bic'] else '1-comp'} preferred)")
    ax.set_xlabel("OTTER template score (ASD subjects)")
    ax.set_yticks([])

    plt.suptitle(
        f"PAGANI-B. OTTER cross-species template does NOT classify ASD at individual level\n"
        f"(n={j['n_valid']} valid subjects across 24 sites, AAL-116 parcellation)",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()

    out = ROOT / "outputs" / "figures" / "autism_subtypes_abide.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
