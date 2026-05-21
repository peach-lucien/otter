"""Visualise the HOMER × Buckner & Krienen tethering test (3 panels)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def main():
    j = json.loads((ROOT / "outputs/logs/buckner_krienen_2013_tethering.json").read_text())
    cov = np.array(j["coverage_per_parcel"])
    ent = np.array(j["entropy_per_parcel"])
    mye = np.array(j["myelin_per_parcel"])
    m = np.isfinite(cov) & np.isfinite(mye)
    cov, ent, mye = cov[m], ent[m], mye[m]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # --- Panel 1: coverage by sensorimotor→association decile -------------
    ax = axes[0]
    dec = j["decile_coverage"]
    colors = plt.cm.RdYlBu(np.linspace(0.1, 0.9, 10))
    ax.bar(range(1, 11), dec, color=colors)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("myelin decile   D1 = association  →  D10 = sensorimotor")
    ax.set_ylabel("HOMER coverage  (mean log₁₀ π column mass)")
    ax.set_title("1 · HOMER coverage collapses toward association cortex\n"
                 f"sensorimotor−association gap = "
                 f"{j['coverage_gap_log_units']:.1f} log units", fontsize=9.5)

    # --- Panel 2: coverage vs the sensorimotor→association axis -----------
    ax = axes[1]
    ax.scatter(mye, cov, s=10, alpha=0.4, color="#2a9d8f", edgecolor="none")
    a = j["association_tertile_coverage"]
    s = j["sensorimotor_tertile_coverage"]
    t = len(mye) // 3
    mo = np.sort(mye)
    ax.plot([mo[0], mo[t]], [a, a], color="#e76f51", lw=3,
            label=f"association tertile  ({a:.0f})")
    ax.plot([mo[-t], mo[-1]], [s, s], color="#264653", lw=3,
            label=f"sensorimotor tertile  ({s:.0f})")
    ax.set_xlabel("Human T1w/T2w myelin  (sensorimotor → association ←)")
    ax.set_ylabel("HOMER coverage  (log₁₀ π column mass)")
    ax.set_title(f"2 · Coverage vs the sensorimotor–association axis\n"
                 f"Spearman ρ = {j['spearman_coverage_vs_myelin']:+.3f}   "
                 f"Mann–Whitney p = {j['mannwhitney_p']:.0e}", fontsize=9.5)
    ax.legend(fontsize=8, loc="lower right")

    # --- Panel 3: entropy is flat (honest negative) ----------------------
    ax = axes[2]
    ax.scatter(mye, ent, s=10, alpha=0.4, color="#9c9c9c", edgecolor="none")
    z = np.polyfit(mye, ent, 1)
    xs = np.linspace(mye.min(), mye.max(), 50)
    ax.plot(xs, z[0] * xs + z[1], "k--", linewidth=0.9)
    ax.set_xlabel("Human T1w/T2w myelin")
    ax.set_ylabel("π column entropy (diffuseness of mouse origin)")
    ax.set_title(f"3 · Entropy is flat (ρ = {j['spearman_entropy_vs_myelin']:+.3f})\n"
                 f"it is the *amount* of coverage, not its diffuseness,\n"
                 f"that carries the tethering signal", fontsize=9.5)

    plt.suptitle("HOMER × Buckner & Krienen 2013 — π is sparsest over human "
                 "association cortex, as the tethering hypothesis predicts",
                 fontsize=12, y=1.04)
    plt.tight_layout()
    out = ROOT / "outputs" / "figures" / "buckner_krienen_2013_tethering.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
