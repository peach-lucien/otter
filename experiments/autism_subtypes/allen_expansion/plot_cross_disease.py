"""Visualise the cross-disease specificity test."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[3]


def main():
    j = json.loads((ROOT / "outputs/logs/autism_subtypes_cross_disease.json").read_text())
    results = j["results"]

    # Filter to valid (>=10 genes overlapping)
    valid = [(c, v) for c, v in results.items() if not v.get("skipped")]
    valid.sort(key=lambda x: -x[1]["boot_mean"])

    fig, ax = plt.subplots(figsize=(10, 5))
    names = [c for c, _ in valid]
    means = [v["boot_mean"] for _, v in valid]
    ci_lo = [v["boot_ci95"][0] for _, v in valid]
    ci_hi = [v["boot_ci95"][1] for _, v in valid]
    ns = [v["n_overlap"] for _, v in valid]

    # Highlight autism
    colors = ["#e76f51" if "autism" in n else "#264653" for n in names]
    y = np.arange(len(names))
    ax.barh(y, means, xerr=[np.array(means)-np.array(ci_lo), np.array(ci_hi)-np.array(means)],
            color=colors, alpha=0.85, capsize=4, error_kw={"linewidth": 1.5})

    for i, (n, v) in enumerate(valid):
        ax.text(v["boot_mean"] + 0.005, i,
                f"n={ns[i]}", va="center", fontsize=9)

    ax.set_yticks(y); ax.set_yticklabels([n.replace("_", " ").title() for n in names])
    ax.set_xlabel("Bootstrap-mean Pearson r (predicted Δ vs observed human ASD Δ)\n"
                  "across 500 gene-resamples per condition")
    ax.set_xlim(0, 0.6)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_title(
        "OTTER cross-species translation produces equal correlation\n"
        "with the human ASD pattern for ALL brain-disorder gene sets, "
        "signal is NOT autism-specific",
        fontsize=11,
    )
    legend = [
        mpatches.Patch(facecolor="#e76f51", label="Autism (Pagani's claim)"),
        mpatches.Patch(facecolor="#264653", label="Other psych conditions (MOESM5)"),
    ]
    ax.legend(handles=legend, loc="lower right")

    # Note about excluded conditions
    skipped = [c for c, v in results.items() if v.get("skipped")]
    if skipped:
        note = (f"Skipped (too few OTTER-overlap genes): " + ", ".join(skipped) +
                "\n(Psoriasis = non-brain control; expected near-zero r had it been testable.)")
        plt.figtext(0.5, -0.05, note, ha="center", fontsize=8, style="italic")

    plt.tight_layout()
    out = ROOT / "outputs" / "figures" / "autism_subtypes_cross_disease.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
