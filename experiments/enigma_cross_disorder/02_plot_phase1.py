"""Phase 1 visualisation: cross-disorder correlation matrix + OTTER predicted spatial pattern."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def main():
    j = json.loads((ROOT / "outputs/logs/enigma_phase1_per_disorder.json").read_text())
    preds = np.load(ROOT / "outputs/coupling/per_disorder_predictions.npz")
    disorders = j["disorders"]
    corr_mat = np.array(j["correlation_matrix"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: cross-disorder correlation matrix
    ax = axes[0]
    im = ax.imshow(corr_mat, cmap="viridis", vmin=0.95, vmax=1.0)
    ax.set_xticks(range(len(disorders))); ax.set_xticklabels(disorders, rotation=45, ha="right")
    ax.set_yticks(range(len(disorders))); ax.set_yticklabels(disorders)
    for ii in range(len(disorders)):
        for jj in range(len(disorders)):
            ax.text(jj, ii, f"{corr_mat[ii,jj]:.3f}", ha="center", va="center",
                    color="white" if corr_mat[ii,jj] < 0.97 else "black", fontsize=9)
    plt.colorbar(im, ax=ax, label="Pearson r")
    mean_off = float(np.array(j['correlation_matrix'])[np.triu_indices(len(disorders), k=1)].mean())
    ax.set_title(f"OTTER per-disorder predicted human patterns (2,094 parcels)\n"
                 f"Mean off-diagonal r = {mean_off:+.3f}\n"
                 f"→ predictions are nearly identical across disorders")

    # Panel 2: distribution of predicted values per disorder
    ax = axes[1]
    colors = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51"]
    for i, d in enumerate(disorders):
        ax.hist(preds[d], bins=60, alpha=0.5, label=d, color=colors[i % len(colors)])
    ax.set_xlabel("OTTER-predicted per-parcel score")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of OTTER predictions per disorder\n"
                 f"(distributions overlap heavily, confirms predictions are similar)")
    ax.legend(fontsize=8)

    plt.suptitle(
        "Phase 1. OTTER per-disorder spatial predictions\n"
        "(autism / bipolar / schizophrenia / ADHD genes from MOESM4 + MOESM5)",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    out = ROOT / "outputs" / "figures" / "enigma_phase1_per_disorder.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
