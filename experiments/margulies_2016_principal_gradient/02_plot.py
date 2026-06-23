"""Visualise the HOMER × Margulies/Huntenburg principal-gradient result."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def main():
    j = json.loads((ROOT / "outputs/logs/margulies_2016_gradient.json").read_text())
    mouse_grad = np.array(j["mouse_gradient"])
    human_grad = np.array(j["human_gradient"])
    pred = np.array(j["predicted_human_gradient"])
    pl, rl = j["parcel_level"], j["region_level"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: mouse gradient distribution
    ax = axes[0]
    ax.hist(mouse_grad, bins=60, color="#264653", alpha=0.85)
    ax.set_xlabel("Mouse principal gradient (per-parcel)")
    ax.set_ylabel("Count")
    ax.set_title(f"Mouse FC principal gradient\nn = {len(mouse_grad)} parcels")

    # Panel 2: predicted vs observed human gradient
    ax = axes[1]
    m = np.isfinite(pred) & np.isfinite(human_grad)
    ax.scatter(human_grad[m], pred[m], s=5, alpha=0.4, color="#2a9d8f")
    z = np.polyfit(human_grad[m], pred[m], 1)
    xs = np.linspace(human_grad[m].min(), human_grad[m].max(), 50)
    ax.plot(xs, z[0] * xs + z[1], "k--", linewidth=0.8)
    ax.set_xlabel("Observed human gradient (per-parcel)")
    ax.set_ylabel("Predicted via HOMER π\n(transport-weighted routing)")
    ax.set_title(f"HOMER preserves the principal axis\n"
                 f"parcel |r| = {pl['abs_pearson_r']:.3f}  "
                 f"|ρ| = {pl['abs_spearman_r']:.3f}  (n = {pl['n']})\n"
                 f"region-level |r| = {rl['abs_pearson_r']:.3f}")

    # Panel 3: permuted-π null
    ax = axes[2]
    nm = j["null"]["abs_r_mean"]
    ci = j["null"]["abs_r_ci95"]
    obs = j["abs_pearson_r"]
    ax.axvspan(ci[0], ci[1], color="#cccccc", alpha=0.6,
               label=f"null 95% CI ({ci[0]:.3f}, {ci[1]:.3f})")
    ax.axvline(nm, color="gray", linestyle="--", label=f"null mean = {nm:.3f}")
    ax.axvline(obs, color="#e76f51", linewidth=2.5, label=f"observed |r| = {obs:.3f}")
    ax.set_xlim(0, max(0.5, obs + 0.05))
    ax.set_yticks([])
    ax.set_xlabel("|Pearson r|")
    ax.set_title(f"Permuted-π null (200 trials)\n"
                 f"{obs / max(nm, 1e-6):.0f}× null mean, "
                 f"empirical p = {j['null']['empirical_p']:.3f}")
    ax.legend(loc="upper right", fontsize=8)

    plt.suptitle("HOMER × Margulies/Huntenburg, π preserves the cross-species "
                 "principal connectivity gradient", fontsize=12, y=1.02)
    plt.tight_layout()
    out = ROOT / "outputs" / "figures" / "margulies_2016_gradient.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
