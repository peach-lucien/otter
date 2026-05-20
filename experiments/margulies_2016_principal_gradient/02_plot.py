"""Visualise the Margulies/Huntenburg principal-gradient cross-species result."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def main():
    j = json.loads((ROOT / "outputs/logs/margulies_2016_gradient.json").read_text())
    mouse_grad   = np.array(j["mouse_gradient"])
    human_grad   = np.array(j["human_gradient"])
    pred_human   = np.array(j["predicted_human_gradient"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: mouse gradient distribution
    ax = axes[0]
    ax.hist(mouse_grad, bins=60, color="#264653", alpha=0.85)
    ax.set_xlabel("Mouse principal gradient (per-parcel)")
    ax.set_ylabel("Count")
    ax.set_title(f"Mouse FC principal gradient\nn={len(mouse_grad)} parcels")

    # Panel 2: scatter predicted vs observed human
    ax = axes[1]
    ax.scatter(human_grad, pred_human, s=4, alpha=0.4, color="#2a9d8f")
    z = np.polyfit(human_grad, pred_human, 1)
    xs = np.linspace(human_grad.min(), human_grad.max(), 50)
    ax.plot(xs, z[0]*xs + z[1], "k--", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.4); ax.axvline(0, color="black", linewidth=0.4)
    ax.set_xlabel("Observed human gradient (per-parcel)")
    ax.set_ylabel("Predicted human gradient via HOMER π")
    ax.set_title(f"HOMER preserves the principal axis\n"
                 f"Pearson r = {j['pearson_r']:+.3f}  "
                 f"Spearman ρ = {j['spearman_r']:+.3f}\n"
                 f"analytical p = {j['pearson_p_analytical']:.1e},  "
                 f"empirical p = {j['null']['empirical_p']:.3f}")

    # Panel 3: null distribution
    ax = axes[2]
    null_mean = j["null"]["abs_r_mean"]
    null_ci = j["null"]["abs_r_ci95"]
    # Show null as a CI band + observed as a vertical line
    xs = np.linspace(0, max(0.2, j["abs_pearson_r"] + 0.05), 200)
    ax.axvspan(null_ci[0], null_ci[1], color="#cccccc", alpha=0.6,
               label=f"null 95% CI: ({null_ci[0]:.3f}, {null_ci[1]:.3f})")
    ax.axvline(null_mean, color="gray", linestyle="--",
               label=f"null mean |r| = {null_mean:.3f}")
    ax.axvline(j["abs_pearson_r"], color="#e76f51", linewidth=2.5,
               label=f"observed |r| = {j['abs_pearson_r']:.3f}")
    ax.set_xlim(0, max(0.2, j["abs_pearson_r"] + 0.05))
    ax.set_yticks([])
    ax.set_xlabel("|Pearson r|")
    ax.set_title("Permuted-π null (200 trials)\n10× null mean, 3× upper 95% CI")
    ax.legend(loc="upper right", fontsize=8)

    plt.suptitle(
        "HOMER × Margulies/Huntenburg — π preserves the cross-species principal connectivity gradient",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    out = ROOT / "outputs" / "figures" / "margulies_2016_gradient.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
