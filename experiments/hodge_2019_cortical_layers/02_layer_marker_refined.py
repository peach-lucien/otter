"""Refined Hodge 2019 validation, restrict to cortex, combine layer-group markers.

Test 1 (single-marker, all 2094 parcels) gave null results. Two issues to address:
  (a) Subcortical parcels dilute the layer-marker signal (Hodge's findings are
      cortex-specific; subcortex has near-zero expression for these markers).
  (b) Single-marker noise (Allen ISH vs AHBA microarray cross-platform variance).

Refinement:
  - Restrict the test to cortical parcels only (parcels with a Schaefer-400
    label, which means they're cortex by construction).
  - For each layer group (upper L2/3, middle L4, deep L5-L6), combine markers
    into a single z-scored composite score before testing.

Hypothesis: if HOMER's π preserves cortical layer-marker geometry at the area
level (not the layer level, π has no layer awareness), then the area-level
spatial pattern of "where is upper-layer expression highest" should agree
between species after π translation.
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
sys.path.insert(0, str(ROOT / "experiments" / "autism_subtypes"))

from homer.data import load_cached, load_pi, pi_provenance
from homer.data.atlas_regions import (
    ATLAS_PATHS,
    assign_atlas_labels,
    assign_atlas_labels_with_hemisphere,
)


# Hodge 2019 layer groups
LAYER_GROUPS = {
    "upper (L2/3)":    ["Cux1", "Cux2", "Satb2"],
    "middle (L4)":     ["Rorb"],
    "deep (L5/6)":     ["Fezf2", "Tbr1", "Foxp2"],
}


def _zscore_safe(v):
    sd = v.std()
    if sd < 1e-9:
        return np.zeros_like(v)
    return (v - v.mean()) / sd


def main():
    print("=" * 80)
    print("Hodge 2019, refined (cortex-only, layer-group composites)")
    print("=" * 80)

    pi = load_pi()                      # canonical coupling (pi_canonical.npy)
    prov = pi_provenance()
    print(f"π file: {prov['pi_file']}  sha256 {prov['pi_sha256']}")
    H, _ = load_cached("human", cache_dir=str(ROOT / "outputs" / "anndata"))

    mouse_expr = np.load(ROOT / "data_external/mouse_genes.npy")
    mouse_genes = pd.read_csv(ROOT / "data_external/mouse_gene_list.csv")
    col_mean = np.nanmean(mouse_expr, axis=0)
    nz = np.where(np.isnan(mouse_expr))
    mouse_expr = mouse_expr.copy()
    mouse_expr[nz] = np.take(col_mean, nz[1])

    human_expr = np.load(ROOT / "data_external/human_genes.npy")
    human_genes = pd.read_csv(ROOT / "data_external/human_gene_list.csv")
    col_mean_h = np.nanmean(human_expr, axis=0)
    nz = np.where(np.isnan(human_expr))
    human_expr = human_expr.copy()
    human_expr[nz] = np.take(col_mean_h, nz[1])

    # ---- Identify cortical human parcels via Schaefer-400 ----
    print("\nAssigning human parcels to cortex (via Schaefer-400)...")
    schaefer_ids = assign_atlas_labels(H.var, "schaefer_400",
                                        str(ROOT / ATLAS_PATHS["schaefer_400"]))
    schaefer_ids = assign_atlas_labels_with_hemisphere(H.var, schaefer_ids)
    cortical_mask = schaefer_ids > 0   # has a Schaefer cortical label
    n_cortical = int(cortical_mask.sum())
    print(f"  cortical human parcels: {n_cortical}/{len(H.var)} "
          f"({100*n_cortical/len(H.var):.0f}%)")

    # ---- Per-layer-group composite test ----
    print(f"\n{'='*80}")
    print(f"{'Layer group':<18s} | {'genes':>20s} | {'r_pred_obs':>11s} | "
          f"{'spearman':>9s} | {'null mean (CI)':>26s} | {'emp p':>6s}")
    print("-" * 110)

    rng = np.random.default_rng(seed=42)
    n_trials = 500
    results = []
    for layer, markers in LAYER_GROUPS.items():
        # Compose mouse score per parcel: mean of z-scored marker expressions
        m_idxs = [int(mouse_genes[mouse_genes["gene_symbol"].str.lower() == m.lower()].iloc[0].name)
                  for m in markers if (mouse_genes["gene_symbol"].str.lower() == m.lower()).any()]
        if not m_idxs:
            continue
        m_composite = np.column_stack([_zscore_safe(mouse_expr[:, i]) for i in m_idxs]).mean(axis=1)

        # Compose human score per parcel
        h_idxs = []
        for m in markers:
            mm = human_genes[human_genes["gene_symbol"].str.upper() == m.upper()]
            if len(mm) > 0:
                h_idxs.append(int(mm.iloc[0].name))
        if not h_idxs:
            continue
        h_composite = np.column_stack([_zscore_safe(human_expr[:, i]) for i in h_idxs]).mean(axis=1)

        # Translate via π
        pred_full = m_composite @ pi    # (2094,)

        # Restrict to cortical
        pred = pred_full[cortical_mask]
        obs = h_composite[cortical_mask]

        r_p, p_p = pearsonr(pred, obs)
        r_s, _ = spearmanr(pred, obs)

        # Permuted-π null
        null_rs = []
        for _ in range(n_trials):
            perm = rng.permutation(pi.shape[0])
            pi_n = pi[perm]
            pred_n = (m_composite @ pi_n)[cortical_mask]
            r_n, _ = pearsonr(pred_n, obs)
            null_rs.append(r_n)
        null_rs = np.array(null_rs)
        emp_p = float((null_rs >= r_p).mean())

        gene_str = ",".join(markers)
        print(f"  {layer:<18s} | {gene_str:>20s} | {r_p:>+11.3f} | "
              f"{r_s:>+9.3f} | {null_rs.mean():+.3f} "
              f"({np.percentile(null_rs, 2.5):+.3f}, {np.percentile(null_rs, 97.5):+.3f}) | "
              f"{emp_p:.3f}")

        results.append({
            "layer_group":  layer,
            "markers":      markers,
            "n_cortical_parcels": int(cortical_mask.sum()),
            "pearson_r":    float(r_p),
            "pearson_p_analytical": float(p_p),
            "spearman_r":   float(r_s),
            "null_mean":    float(null_rs.mean()),
            "null_ci95":    [float(np.percentile(null_rs, 2.5)),
                             float(np.percentile(null_rs, 97.5))],
            "empirical_p":  emp_p,
        })

    # ---- Layer-discrimination test: upper-deep contrast ----
    # If π preserves layer geometry, the *contrast* (upper - deep) should
    # correlate even more strongly across species because absolute expression
    # noise cancels.
    print(f"\n{'='*80}")
    print("Contrast test: upper-layer score minus deep-layer score (per parcel)")
    print("-" * 110)

    upper_m_idxs = [int(mouse_genes[mouse_genes["gene_symbol"].str.lower() == m.lower()].iloc[0].name)
                    for m in ["Cux1", "Cux2", "Satb2"]]
    deep_m_idxs  = [int(mouse_genes[mouse_genes["gene_symbol"].str.lower() == m.lower()].iloc[0].name)
                    for m in ["Fezf2", "Tbr1", "Foxp2"]]
    upper_m = np.column_stack([_zscore_safe(mouse_expr[:, i]) for i in upper_m_idxs]).mean(axis=1)
    deep_m  = np.column_stack([_zscore_safe(mouse_expr[:, i]) for i in deep_m_idxs]).mean(axis=1)
    m_contrast = upper_m - deep_m

    upper_h_idxs = [int(human_genes[human_genes["gene_symbol"].str.upper() == m.upper()].iloc[0].name)
                    for m in ["CUX1", "CUX2", "SATB2"]]
    deep_h_idxs  = [int(human_genes[human_genes["gene_symbol"].str.upper() == m.upper()].iloc[0].name)
                    for m in ["FEZF2", "TBR1", "FOXP2"]]
    upper_h = np.column_stack([_zscore_safe(human_expr[:, i]) for i in upper_h_idxs]).mean(axis=1)
    deep_h  = np.column_stack([_zscore_safe(human_expr[:, i]) for i in deep_h_idxs]).mean(axis=1)
    h_contrast = upper_h - deep_h

    pred_contrast_full = m_contrast @ pi
    pred = pred_contrast_full[cortical_mask]
    obs  = h_contrast[cortical_mask]

    r_p, p_p = pearsonr(pred, obs)
    r_s, _ = spearmanr(pred, obs)

    null_rs = []
    for _ in range(n_trials):
        perm = rng.permutation(pi.shape[0])
        pi_n = pi[perm]
        pred_n = (m_contrast @ pi_n)[cortical_mask]
        r_n, _ = pearsonr(pred_n, obs)
        null_rs.append(r_n)
    null_rs = np.array(null_rs)
    emp_p = float((null_rs >= r_p).mean())
    print(f"  upper − deep contrast | {'CUX1+CUX2+SATB2 − FEZF2+TBR1+FOXP2':>20s} | "
          f"{r_p:>+11.3f} | {r_s:>+9.3f} | {null_rs.mean():+.3f} "
          f"({np.percentile(null_rs, 2.5):+.3f}, {np.percentile(null_rs, 97.5):+.3f}) | "
          f"{emp_p:.3f}")

    contrast_result = {
        "test": "upper − deep contrast (cortex only)",
        "pearson_r":            float(r_p),
        "pearson_p_analytical": float(p_p),
        "spearman_r":           float(r_s),
        "null_mean":            float(null_rs.mean()),
        "null_ci95":            [float(np.percentile(null_rs, 2.5)),
                                  float(np.percentile(null_rs, 97.5))],
        "empirical_p":          emp_p,
    }

    # ---- Lobe-aggregated test ----
    # Aggregate predicted and observed scores to anatomical lobes (rough Yeo7
    # network categories) to see if HOMER preserves the *broad* gradient even
    # if per-parcel comparison is noisy.
    print(f"\n{'='*80}")
    print("Lobe-aggregated test (mean score per Yeo7 network, cortex only)")
    print("-" * 110)

    from importlib import import_module
    nc = import_module("01_network_crossvalidation")
    human_net, human_paper_names = nc.assign_human_paper_networks(H.var, separate_aud=True)
    # Lobe-aggregate each marker score
    lobe_results = []
    for layer, markers in LAYER_GROUPS.items():
        m_idxs = [int(mouse_genes[mouse_genes["gene_symbol"].str.lower() == m.lower()].iloc[0].name)
                  for m in markers]
        m_composite = np.column_stack([_zscore_safe(mouse_expr[:, i]) for i in m_idxs]).mean(axis=1)
        h_idxs = [int(human_genes[human_genes["gene_symbol"].str.upper() == m.upper()].iloc[0].name)
                  for m in markers]
        h_composite = np.column_stack([_zscore_safe(human_expr[:, i]) for i in h_idxs]).mean(axis=1)
        pred_full = m_composite @ pi

        # Mean per Yeo7 network (cortical only). Some networks may have zero
        # cortical parcels, drop those to avoid NaN.
        pred_per_net = []
        obs_per_net = []
        kept_nets = []
        for i in range(len(human_paper_names)):
            mask_i = (human_net == i) & cortical_mask
            if mask_i.any():
                pred_per_net.append(pred_full[mask_i].mean())
                obs_per_net.append(h_composite[mask_i].mean())
                kept_nets.append(human_paper_names[i])
        pred_per_net = np.array(pred_per_net)
        obs_per_net = np.array(obs_per_net)
        r_p, p_p = pearsonr(pred_per_net, obs_per_net)
        r_s, _ = spearmanr(pred_per_net, obs_per_net)
        print(f"  {layer:<18s} | n_nets={len(pred_per_net):>2d} | "
              f"r_pearson={r_p:+.3f}  spearman={r_s:+.3f}  analytical p={p_p:.3f}")
        lobe_results.append({
            "layer_group": layer,
            "n_networks":  int(len(pred_per_net)),
            "pearson_r":   float(r_p),
            "pearson_p_analytical": float(p_p),
            "spearman_r":  float(r_s),
        })

    out = {
        **prov,
        "n_cortical_parcels": int(cortical_mask.sum()),
        "per_layer_group":    results,
        "upper_minus_deep_contrast": contrast_result,
        "lobe_aggregated":    lobe_results,
    }
    out_path = ROOT / "outputs" / "logs" / "hodge_2019_layer_markers_refined.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
