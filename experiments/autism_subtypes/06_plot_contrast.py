"""Visualise Test 2b result, predicted vs observed subtype contrast."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main():
    j = json.loads(Path("outputs/logs/autism_subtypes_contrast.json").read_text())
    nets = list(j["human_delta_observed"].keys())
    obs = np.array([j["human_delta_observed"][n] for n in nets])
    pred = np.array([j["human_delta_predicted"][n] for n in nets])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    # Left: bar comparison
    x = np.arange(len(nets))
    width = 0.4
    # Standardize each vector for visual comparison (Pearson is scale-invariant)
    obs_z = (obs - obs.mean()) / obs.std()
    pred_z = (pred - pred.mean()) / pred.std()
    ax1.bar(x - width/2, obs_z, width, label="observed (Pagani Fig 4e Δ)", color="#264653")
    ax1.bar(x + width/2, pred_z, width, label="predicted via HOMER π", color="#e76f51")
    ax1.set_xticks(x); ax1.set_xticklabels(nets, rotation=30, ha="right")
    ax1.set_ylabel("z-scored subtype contrast (hyper − hypo)")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_title(f"Subtype-contrast per network: predicted vs observed\n"
                  f"Pearson r = {j['pearson_r']:+.3f} (empirical p = {j['null']['pearson_one_sided_p']:.3f})")
    ax1.legend(loc="upper left")

    # Right: null distribution
    # Re-run to collect null distribution? It's already summarized, just show CI
    ax2.set_title("Permuted-π null distribution\n"
                  f"(200 row-shuffles, mean {j['null']['pearson_mean']:+.2f}, CI shaded)")
    ax2.set_xlabel("Pearson r (predicted Δ vs observed Δ)")
    ax2.set_ylabel("count")
    # We can't reconstruct the exact null without re-running, show CI band + real
    null_mean = j["null"]["pearson_mean"]
    null_ci = j["null"]["pearson_ci95"]
    ax2.axvspan(null_ci[0], null_ci[1], color="#cccccc", alpha=0.6, label="null 95% CI")
    ax2.axvline(null_mean, color="gray", linestyle="--", label="null mean")
    ax2.axvline(j["pearson_r"], color="#e76f51", linewidth=2.5,
                label=f"observed r={j['pearson_r']:+.2f}")
    ax2.set_xlim(-1, 1)
    ax2.legend(loc="upper right")
    ax2.set_yticks([])

    plt.tight_layout()
    out = Path("outputs/figures/autism_subtypes_contrast.png")
    plt.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
