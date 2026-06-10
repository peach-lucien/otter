"""Visualise the corrected Pagani 2026 subtype translation.

Reads outputs/logs/pagani_subtype_translation_corrected.json (written by
01_per_model_clustering.py) and produces a 3-panel figure:

  1. The 20 mouse models placed on the hyper↔hypo membership axis
     (leave-one-out correlation to each subtype signature), coloured by their
     verified subtype.
  2. The per-subtype mouse→human network prediction through π vs the observed
     human pattern (the cross-species translation).
  3. The 2×2 cross-species correlation matrix (subtype-specificity).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

HYPER_C = "#e76f51"
HYPO_C = "#2a9d8f"


def main():
    j = json.loads((ROOT / "outputs/logs/pagani_subtype_translation_corrected.json").read_text())
    models = j["models"]
    trans = j["translation"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))

    # ---- Panel 1: membership axis ----
    ax = axes[0]
    order = np.argsort([m["membership_score"] for m in models])
    ms = [models[i] for i in order]
    y = np.arange(len(ms))
    scores = [m["membership_score"] for m in ms]
    colors = [HYPER_C if m["subtype"] == "hyper" else HYPO_C for m in ms]
    edge = ["black" if m["predicted_side"] == m["subtype"] else "red" for m in ms]
    ax.barh(y, scores, color=colors, edgecolor=edge, linewidth=1.2)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([m["model"] for m in ms], fontsize=8)
    ax.set_xlabel("membership score  (r to hyper − r to hypo signature, leave-one-out)")
    ax.set_title(f"Per-model subtype membership\n"
                 f"colour = verified subtype · red edge = LOO disagreement "
                 f"({j['loo_consistency']})")
    ax.text(0.02, 0.98, "← hypo", transform=ax.transAxes, va="top", color=HYPO_C, fontweight="bold")
    ax.text(0.98, 0.98, "hyper →", transform=ax.transAxes, va="top", ha="right",
            color=HYPER_C, fontweight="bold")

    # ---- Panel 2: predicted vs observed human network pattern ----
    ax = axes[1]
    nets = trans["human_networks"]
    x = np.arange(len(nets))
    w = 0.2
    def vec(d):
        return [d[n] for n in nets]
    ax.bar(x - 1.5 * w, vec(trans["predicted"]["hyper"]), w, color=HYPER_C, label="pred hyper")
    ax.bar(x - 0.5 * w, vec(trans["observed"]["hyper"]), w, color=HYPER_C, alpha=0.45, label="obs hyper")
    ax.bar(x + 0.5 * w, vec(trans["predicted"]["hypo"]), w, color=HYPO_C, label="pred hypo")
    ax.bar(x + 1.5 * w, vec(trans["observed"]["hypo"]), w, color=HYPO_C, alpha=0.45, label="obs hypo")
    ax.set_xticks(x)
    ax.set_xticklabels(nets, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("network intensity (a.u.)")
    ax.set_title("Mouse subtype → human prediction via π\nvs observed human subtype pattern")
    ax.legend(fontsize=7, ncol=2)

    # ---- Panel 3: cross-species correlation matrix ----
    ax = axes[2]
    xc = trans["cross_correlation"]
    mat = np.array([[xc["pred_hyper__obs_hyper"], xc["pred_hyper__obs_hypo"]],
                    [xc["pred_hypo__obs_hyper"], xc["pred_hypo__obs_hypo"]]])
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["obs hyper", "obs hypo"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["pred hyper", "pred hypo"])
    for i in range(2):
        for k in range(2):
            ax.text(k, i, f"{mat[i, k]:+.2f}", ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="white" if abs(mat[i, k]) > 0.5 else "black")
    ax.set_title("Cross-species subtype specificity\n(diagonal should dominate)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")

    plt.suptitle(
        "HOMER × Pagani 2026 — corrected subtype translation "
        "(verified n=9 hyper / n=11 hypo; π routing, no 1,491-feature decode)",
        fontsize=13, y=1.03)
    plt.tight_layout()
    out = ROOT / "outputs" / "figures" / "pagani_subtype_translation_corrected.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
