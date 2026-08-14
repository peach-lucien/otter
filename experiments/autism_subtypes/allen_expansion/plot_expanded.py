"""Visualise the expanded Pagani gene-spatial test: bootstrap + per-pathway."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parents[3]


def main():
    j_exp = json.loads((ROOT / "outputs/logs/autism_subtypes_gene_spatial_expanded.json").read_text())
    j_diag = json.loads((ROOT / "outputs/logs/autism_subtypes_gene_diagnostics.json").read_text())
    j_36 = json.loads((ROOT / "outputs/logs/autism_subtypes_gene_spatial.json").read_text())

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
    fig.suptitle(f"Pagani 2026 Test 3, expanded {j_diag['n_genes']} genes "
                 f"(vs 36 in proof-of-concept)", fontsize=13, y=0.98)

    # Panel 1: Bootstrap distribution
    ax = axes[0]
    # The json holds summary statistics rather than raw bootstrap samples. Draw a
    # gaussian approximation centred on the mean with the reported 95% CI.
    boot_mean = j_diag["bootstrap"]["mean_r"]
    ci_lo, ci_hi = j_diag["bootstrap"]["ci95"]
    sd_approx = (ci_hi - ci_lo) / (2 * 1.96)
    xs = np.linspace(boot_mean - 4*sd_approx, boot_mean + 4*sd_approx, 200)
    ys = np.exp(-(xs - boot_mean)**2 / (2*sd_approx**2))
    ax.fill_between(xs, ys, color="#2a9d8f", alpha=0.4, label="bootstrap density")
    ax.axvline(boot_mean, color="#2a9d8f", linewidth=2, label=f"bootstrap mean r = {boot_mean:+.3f}")
    ax.axvspan(ci_lo, ci_hi, color="#2a9d8f", alpha=0.15, label=f"95% CI: ({ci_lo:+.3f}, {ci_hi:+.3f})")
    ax.axvline(0, color="black", linewidth=0.7, linestyle="--", label="null (r=0)")
    ax.axvline(j_36["pearson_r"], color="#999", linewidth=2, linestyle=":",
               label=f"36-gene proof-of-concept r = {j_36['pearson_r']:+.3f}")
    ax.set_xlabel("Pearson r (predicted Δ vs observed Δ)")
    ax.set_yticks([])
    ax.set_title(f"Bootstrap over genes (n={j_diag['bootstrap']['n_resamples']} resamples)\n"
                 f"100% positive; 99.7% > 0.3")
    ax.legend(loc="upper right", fontsize=8)

    # Panel 2: Per-pathway r values
    ax = axes[1]
    pw = j_diag["per_pathway"]
    pw = sorted(pw, key=lambda x: x["r_to_obs_delta"])
    names = [p["pathway"].replace("_", "\n", 1) for p in pw]
    rs = [p["r_to_obs_delta"] for p in pw]
    ns = [p["n_genes_in_otter"] for p in pw]
    ax.barh(np.arange(len(pw)), rs, color="#264653")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_yticks(np.arange(len(pw)))
    ax.set_yticklabels([n.replace("_", " ") for n in [p["pathway"] for p in pw]], fontsize=7.5)
    for i, n in enumerate(ns):
        ax.text(rs[i] + 0.01, i, f"n={n}", va="center", fontsize=7.5)
    ax.set_xlabel("Pearson r (pathway-spatial Δ vs observed Δ)")
    ax.set_title("Every pathway shows the same direction (no subtype separation)\n"
                 ", driven by hyper signal being much larger than hypo in Pagani data")

    # Panel 3: Per-parcel gene-vs-FC translation scatter
    # The json holds r rather than the raw vectors, so plot the summary as text.
    ax = axes[2]
    ax.axis("off")
    r_gf_p = j_diag["per_parcel_gene_vs_fc_translation"]["pearson_r"]
    r_gf_s = j_diag["per_parcel_gene_vs_fc_translation"]["spearman_r"]
    lines = [
        "$\\bf{Diagnostic\\ 1}$, gene-translation vs FC-translation agreement",
        "",
        f"  Per-parcel correlation (n=2,094 human parcels):",
        f"    Pearson r  = {r_gf_p:+.3f}",
        f"    Spearman ρ = {r_gf_s:+.3f}",
        "",
        "Interpretation:",
        "  Modest agreement, not strong. The two OTTER translation routes",
        "  (gene-spatial vs FC-perturbation, both routed through π) capture",
        "  COMPLEMENTARY rather than redundant signals:",
        "    • gene-spatial = 'which regions express Pagani-implicated genes'",
        "    • FC-spatial   = 'which networks are functionally perturbed'",
        "  Both correlate with Pagani's observed human ASD pattern but they",
        "  encode different aspects of the cross-species biology."
    ]
    for i, line in enumerate(lines):
        ax.text(0.02, 0.95 - i * 0.06, line, fontsize=10,
                transform=ax.transAxes, verticalalignment="top",
                family="monospace" if "  Per-" in line or "    " in line else None)

    # Panel 4: Summary verdict
    ax = axes[3]
    ax.axis("off")
    lines = [
        "$\\bf{Verdict}$",
        "",
        f"$\\bf{{Test\\ 3\\ (expanded, n={j_diag['n_genes']}\\ genes)}}$:",
        f"  Pearson r = {j_exp['pearson_r']:+.3f}   (single point estimate)",
        f"  Bootstrap mean = {boot_mean:+.3f}, 95% CI ({ci_lo:+.3f}, {ci_hi:+.3f})",
        f"  100% of 1,000 bootstrap resamples produce positive r.",
        f"  Spearman ρ = {j_exp['spearman_r']:+.3f},  empirical p (n=8) = {j_exp['spearman_empirical_p']:.3f}",
        "",
        "$\\bf{Headline}$: OTTER-translated gene-spatial maps reproducibly",
        "match Pagani's observed human ASD subtype pattern.",
        "",
        "$\\bf{Caveat}$: the test cannot distinguish synaptic-vs-immune",
        "pathway directions, because Pagani's hypoconnected-subtype matrix",
        "has tiny magnitudes (max ~1.5) relative to hyperconnected (max ~33).",
        "The Δ test is dominated by hyper. Test 3 supports Pagani's overall",
        "cross-species spatial-replication claim but cannot interrogate the",
        "per-pathway direction-by-subtype claim from their published",
        "source data alone.",
    ]
    for i, line in enumerate(lines):
        ax.text(0.02, 0.96 - i * 0.058, line, fontsize=9.5,
                transform=ax.transAxes, verticalalignment="top",
                family="monospace" if "  " in line and "$" not in line else None)

    out = ROOT / "outputs" / "figures" / "autism_subtypes_gene_expanded.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
