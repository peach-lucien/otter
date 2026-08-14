"""Visualise the OTTER × Schaeffer 2020 MFC-divergence falsification test."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from otter.data import load_cached

COL = {"dlPFC": "#e76f51", "premotor": "#2a9d8f", "medial_PFC": "#264653",
       "mid_cingulate": "#8ab17d", "other": "#cccccc"}


def main():
    j = json.loads((ROOT / "outputs/logs/schaeffer_2020_mfc_divergence.json").read_text())
    rec = j["recommended_pi"]
    rois = j["rois"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.9))

    # --- Panel 1: enrichment per ROI (recommended π) -----------------------
    ax = axes[0]
    names = ["dlPFC", "premotor", "medial_PFC", "mid_cingulate"]
    enr = [rec["null"][n]["enrichment"] for n in names]
    mass = [rec["mass_fraction"][n] * 100 for n in names]
    bars = ax.bar(range(4), enr, color=[COL[n] for n in names], alpha=0.9)
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1,
               label="chance (permuted-π null)")
    for i, (b, m) in enumerate(zip(bars, mass)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.25,
                f"{m:.1f}%\nof mass", ha="center", fontsize=8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["dlPFC\n(BA9/46)", "premotor", "medial PFC", "mid-cingulate"])
    ax.set_ylabel("Enrichment of mouse-MFC mass\n(observed ÷ permuted-π null)")
    ax.set_title("1 · Where π routes mouse medial frontal cortex\n"
                 "dlPFC ×0.0 (avoided), premotor / medial / cingulate ×4–10",
                 fontsize=9.5)
    ax.legend(fontsize=8)

    # --- Panel 2: contrast, what the contested anchor does ----------------
    ax = axes[1]
    tags = ["baseline (Garin only)", "canonical", "+lateral_pfc pack"]
    mfc_dl = [j["contrast"][t]["mfc_to_dlpfc"] * 100 for t in tags]
    pl_dl = [j["contrast"][t]["pl_to_dlpfc"] * 100 for t in tags]
    x = np.arange(3)
    ax.bar(x - 0.2, mfc_dl, 0.4, color="#264653", label="all mouse MFC")
    ax.bar(x + 0.2, pl_dl, 0.4, color="#e76f51", label="mouse Prelimbic only")
    ax.set_xticks(x)
    ax.set_xticklabels(["baseline\n(Garin only)", "recommended\n(shipped π)",
                        "+lateral_pfc\n(contested anchor)"], fontsize=8.5)
    ax.set_ylabel("Mouse-frontal mass routed to human dlPFC (%)")
    ax.set_title("2 · The contested Prelimbic→dlPFC anchor is opt-in\n"
                 "only forcing that anchor puts mass on dlPFC, "
                 "Schaeffer 2020 argues it shouldn't", fontsize=9.5)
    ax.legend(fontsize=8)

    # --- Panel 3: argmax landing in MNI space ------------------------------
    ax = axes[2]
    H, _ = load_cached("human", cache_dir=str(ROOT / "outputs/anndata"))
    xyz = H.var[["x", "y", "z"]].to_numpy(dtype=float)
    ax.scatter(xyz[:, 0], xyz[:, 1], s=5, color="#e8e8e8", edgecolor="none")
    for n, info in rois.items():
        cx, cy, _ = info["center_mni"]
        r = info["radius_mm"]
        for sx in (-1, 1):
            ax.add_patch(Circle((sx * cx, cy), r, fill=True, alpha=0.18,
                                color=COL[n], zorder=2))
            ax.add_patch(Circle((sx * cx, cy), r, fill=False, linewidth=1.3,
                                color=COL[n], zorder=3))
    amax = np.array(rec["argmax_human_idx"])
    ax.scatter(xyz[amax, 0], xyz[amax, 1], s=46, color="#1d3557",
               edgecolor="white", linewidth=0.6, zorder=5,
               label=f"top-1 human partner of\nthe {len(amax)} mouse-MFC parcels")
    ax.set_xlabel("MNI x (mm)  ·  ←L   R→")
    ax.set_ylabel("MNI y (mm)  ·  posterior → anterior")
    ax.set_title("3 · Top-1 landing of mouse MFC, 0 of 39 in dlPFC\n"
                 "(red = dlPFC target; teal/green = Schaeffer-consistent)",
                 fontsize=9.5)
    ax.legend(fontsize=7.5, loc="lower center")
    ax.set_aspect("equal")

    plt.suptitle("OTTER × Schaeffer 2020, π routes mouse medial frontal cortex "
                 "away from human dlPFC, as the connectivity evidence predicts",
                 fontsize=12, y=1.04)
    plt.tight_layout()
    out = ROOT / "outputs" / "figures" / "schaeffer_2020_mfc_divergence.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
