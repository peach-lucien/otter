"""Visualise the per-mouse-model HOMER translation showcase."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[2]


def main():
    j = json.loads((ROOT / "outputs/logs/pagani_per_model_translation.json").read_text())
    models = j["models"]
    hyper_pred = np.array(j["homer_predictions"]["hyper_human_per_parcel"])
    hypo_pred = np.array(j["homer_predictions"]["hypo_human_per_parcel"])
    template_delta = hyper_pred - hypo_pred   # (2094,)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: PCA of 20 models in 1491-feature space
    ax = axes[0]
    for m in models:
        c = "#e76f51" if m["inferred"] == "hyper" else "#2a9d8f"
        marker = "o" if m["prior"] != "(unknown)" else "x"
        ax.scatter(m["pc1"], m["pc2"], c=c, s=80, alpha=0.8, marker=marker)
        ax.annotate(m["model"], (m["pc1"], m["pc2"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({j['pca_var']['pc1']*100:.0f}% var)")
    ax.set_ylabel(f"PC2 ({j['pca_var']['pc2']*100:.0f}% var)")
    ax.set_title("PCA of 20 mouse models\nin Pagani 1,491-feature space (KMeans k=2)")
    legend = [
        mpatches.Patch(facecolor="#e76f51", label="cluster A (inferred 'hyper')"),
        mpatches.Patch(facecolor="#2a9d8f", label="cluster B (inferred 'hypo')"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8)

    # Panel 2: Per-model HOMER hyper/hypo weight bar chart
    ax = axes[1]
    names = [m["model"] for m in models]
    hweights = [m["hyper_weight"] for m in models]
    # Sort by hyper weight
    order = np.argsort(hweights)[::-1]
    names_s = [names[i] for i in order]
    hweights_s = [hweights[i] for i in order]
    colors = ["#e76f51" if m["inferred"] == "hyper" else "#2a9d8f"
              for m in [models[i] for i in order]]
    y = np.arange(len(names_s))
    ax.barh(y, hweights_s, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0.5, color="black", linewidth=0.5, linestyle="--")
    ax.set_yticks(y); ax.set_yticklabels(names_s, fontsize=8)
    ax.set_xlabel("HOMER hyper-weight (1 = pure hyper template, 0 = pure hypo)")
    ax.set_title("Per-model HOMER soft membership\n(distance-weighted KMeans probability)")

    # Panel 3: HOMER hyper - hypo human-parcel template (the per-subtype δ pattern)
    ax = axes[2]
    # Show as histogram of per-parcel Δ values
    ax.hist(template_delta, bins=80, color="#264653", alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("HOMER-predicted human-parcel Δ (hyper template − hypo template)")
    ax.set_ylabel("count")
    ax.set_title(f"HOMER per-subtype Δ-prediction over 2,094 human parcels\n"
                 f"(from Test 2c — this is what each model would 'see' in human space)\n"
                 f"range [{template_delta.min():+.2f}, {template_delta.max():+.2f}]")

    plt.suptitle(
        "HOMER × Pagani 2026 — per-mouse-model translation showcase\n"
        "(exploratory: 1,491-feature decoding not possible without Pagani's atlas methods)",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    out = ROOT / "outputs" / "figures" / "pagani_per_model_translation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
