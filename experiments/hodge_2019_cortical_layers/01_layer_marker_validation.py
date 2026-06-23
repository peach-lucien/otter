"""Hodge et al. 2019 cortical-layer-marker cross-species validation.

[Hodge et al. 2019, Nature](https://www.nature.com/articles/s41586-019-1506-7)
"Conserved cell types with divergent features in human versus mouse cortex"
showed that the canonical layer-defining transcription factors maintain their
layer-specific spatial expression across mouse and human cortex:

  - CUX1, CUX2, SATB2 → upper cortical layers (L2/3)
  - RORB             → granular layer 4
  - FEZF2            → infragranular L5
  - TBR1, FOXP2      → deep layers (L6)

HOMER's claim: the coupling π maps mouse parcels to human parcels such that
their connectivity neighborhoods + spatial position match. If π is sensible,
then **mouse parcels with high CUX2 expression should map to human parcels
with high CUX2 expression**, and the same should hold for the other markers.

This is a Beauchamp-independent test: Hodge's data comes from Allen ISH (mouse)
and AHBA microarray (human), which are different platforms from Beauchamp's
Mouse-Human Transcriptomic Similarity dataset. So agreement here is
independent evidence about π's anatomical fidelity.

Procedure:
  1. For each Hodge marker, pull mouse per-parcel expression from HOMER's
     `mouse_genes.npy` (Allen ISH, 1864 parcels × 61 genes).
  2. Translate through π:  pred_human[h] = Σ_m π[m, h] · mouse_expr[m].
  3. Compare predicted human spatial pattern to observed AHBA per-parcel
     expression for the same marker (from `human_genes.npy`, 2094 × 15633).
  4. Score: Pearson r over 2094 human parcels.
  5. Compare against permuted-π null (200 trials, within-row permutation).

Headline statistic: mean cross-species Pearson r over the 7 markers, vs null
mean. If HOMER preserves layer geometry, we expect r ≈ +0.3 to +0.6 per marker.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from homer.data import load_cached

# Hodge et al. 2019 canonical cortical-layer markers (gene symbol, layer).
HODGE_MARKERS = [
    ("Cux1",  "L2/3 upper",  "upper"),
    ("Cux2",  "L2/3 upper",  "upper"),
    ("Satb2", "L2/3 upper",  "upper"),
    ("Rorb",  "L4 granular", "middle"),
    ("Fezf2", "L5 infragranular", "deep"),
    ("Tbr1",  "L6 deep",     "deep"),
    ("Foxp2", "L6 deep",     "deep"),
]


def _column_for_gene(gene_list: pd.DataFrame, gene_symbol: str) -> int | None:
    """Find the column index in the gene matrix for a given gene symbol (case-insensitive)."""
    match = gene_list[gene_list["gene_symbol"].str.upper() == gene_symbol.upper()]
    if len(match) == 0:
        return None
    return int(match.iloc[0].name)


def _zscore_safe(v: np.ndarray) -> np.ndarray:
    sd = v.std()
    if sd < 1e-9:
        return np.zeros_like(v)
    return (v - v.mean()) / sd


def main():
    print("=" * 80)
    print("Hodge 2019 cortical-layer marker cross-species validation")
    print("=" * 80)

    # ---- Load HOMER ----
    pi = np.load(ROOT / "outputs/coupling/pi_fc_plus_SC_with_all_packs.npy")
    print(f"π shape: {pi.shape}, total mass: {pi.sum():.4f}")

    # Mouse gene matrix + names
    mouse_expr  = np.load(ROOT / "data_external/mouse_genes.npy")    # (1864, 61)
    mouse_genes = pd.read_csv(ROOT / "data_external/mouse_gene_list.csv")
    # Replace NaN with column means
    col_mean = np.nanmean(mouse_expr, axis=0)
    nz = np.where(np.isnan(mouse_expr))
    mouse_expr = mouse_expr.copy()
    mouse_expr[nz] = np.take(col_mean, nz[1])
    print(f"Mouse Allen ISH: {mouse_expr.shape} parcels × genes ({len(mouse_genes)} symbols)")

    # Human gene matrix + names (the FULL AHBA matrix, not the aligned subset)
    human_expr  = np.load(ROOT / "data_external/human_genes.npy")    # (2094, 15633)
    human_genes = pd.read_csv(ROOT / "data_external/human_gene_list.csv")
    # AHBA also has NaNs (regions with no expression measurement)
    col_mean_h = np.nanmean(human_expr, axis=0)
    nz = np.where(np.isnan(human_expr))
    human_expr = human_expr.copy()
    human_expr[nz] = np.take(col_mean_h, nz[1])
    print(f"Human AHBA:      {human_expr.shape} parcels × genes ({len(human_genes)} symbols)")

    # ---- Per-marker test ----
    print("\n" + "=" * 80)
    print(f"{'Marker':<8s} | {'Layer':<22s} | {'r_pred_vs_obs':>14s} | "
          f"{'spearman':>10s} | {'null mean (CI)':>26s}")
    print("-" * 95)

    rng = np.random.default_rng(seed=42)
    n_trials = 200
    results = []
    for symbol, layer, _grp in HODGE_MARKERS:
        # Mouse column
        m_idx = int(mouse_genes[mouse_genes["gene_symbol"].str.lower() == symbol.lower()].iloc[0].name)
        m_vec = mouse_expr[:, m_idx]    # (1864,)
        # Human column (case-insensitive match against the FULL human gene list)
        h_match = human_genes[human_genes["gene_symbol"].str.upper() == symbol.upper()]
        if len(h_match) == 0:
            print(f"  {symbol:<8s} | (not in human AHBA, skipped)")
            continue
        h_idx = int(h_match.iloc[0].name)
        h_obs = human_expr[:, h_idx]   # (2094,)

        # Translate via π: pred_human[h] = Σ_m π[m, h] · z(m_expr[m])
        m_z = _zscore_safe(m_vec)
        pred_human = m_z @ pi          # (2094,)

        # Observed human z-score
        h_z = _zscore_safe(h_obs)

        # Correlation
        r_p, p_p = pearsonr(pred_human, h_z)
        r_s, _ = spearmanr(pred_human, h_z)

        # Permuted-π null (shuffle the rows of π, recompute prediction)
        null_rs = []
        for _ in range(n_trials):
            perm = rng.permutation(pi.shape[0])
            pi_n = pi[perm]
            pred_n = m_z @ pi_n
            r_n, _ = pearsonr(pred_n, h_z)
            null_rs.append(r_n)
        null_rs = np.array(null_rs)
        emp_p = float((null_rs >= r_p).mean())

        print(f"  {symbol:<8s} | {layer:<22s} | {r_p:>+14.3f} | "
              f"{r_s:>+10.3f} | {null_rs.mean():+.3f} "
              f"({np.percentile(null_rs, 2.5):+.3f}, {np.percentile(null_rs, 97.5):+.3f})  "
              f"emp p={emp_p:.3f}")

        results.append({
            "gene": symbol, "layer": layer,
            "pearson_r": float(r_p), "pearson_p_analytical": float(p_p),
            "spearman_r": float(r_s),
            "null_mean": float(null_rs.mean()),
            "null_ci95": [float(np.percentile(null_rs, 2.5)),
                           float(np.percentile(null_rs, 97.5))],
            "empirical_p": emp_p,
        })

    # Summary
    rs = [r["pearson_r"] for r in results]
    nulls = [r["null_mean"] for r in results]
    print(f"\n{'='*80}")
    print(f"Mean Pearson r across {len(results)} markers: {np.mean(rs):+.3f}")
    print(f"Mean null Pearson r:                          {np.mean(nulls):+.3f}")
    print(f"Markers with empirical p < 0.05:              "
          f"{sum(1 for r in results if r['empirical_p'] < 0.05)}/{len(results)}")

    # Save
    out = {
        "n_markers":          len(results),
        "markers":            results,
        "mean_pearson_r":     float(np.mean(rs)),
        "mean_null_pearson_r": float(np.mean(nulls)),
        "n_markers_p_below_0.05":
            int(sum(1 for r in results if r["empirical_p"] < 0.05)),
    }
    out_path = ROOT / "outputs" / "logs" / "hodge_2019_layer_markers.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
