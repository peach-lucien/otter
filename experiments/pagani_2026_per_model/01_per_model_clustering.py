"""Per-mouse-model HOMER translation showcase.

Pagani 2026 publishes a 20-model × 1,491-feature matrix in MOESM6 'Figura 1c'.
Each row is one mouse autism model (Fmr1, Tsc2, Shank3, Mecp2, …) and the
1,491 columns are FC perturbation features.

We can't decode exactly what the 1,491 features represent without the paper's
methods section, and ~5 % of the cells appear to be Excel-formatting-corrupted
outliers (huge integers like 1,283,728 where small decimals should be). After
masking outliers, the remaining 95 % are correlation-like values [-1, 1] that
we can use for clustering.

Procedure:
  1. Load 20 × 1,491 matrix, mask |v| > 5 as NaN, impute column-mean.
  2. PCA (n=2) of the 20 models in 1,491-feature space.
  3. KMeans (k=2) to recover the hyper/hypo split that Pagani derives.
  4. Compare against the biological prior: Trem2/Btbr/Il6/Mecp2 = hyper;
     Tsc2/Shank3/Fmr1 = hypo (per Pagani Fig 1d).
  5. Take HOMER's per-subtype human-parcel prediction (from Test 2c output) as
     the "human-space template" each model maps to via its inferred subtype.
  6. Visualise where each model lands in HOMER-translated human-space.

What this gives us: a showcase of "which human ASD subtype does each mouse
model resemble in HOMER-translated space?". Caveat: per-model resolution is
degraded to subtype-resolution because we couldn't decode 1,491 → 1,864.
Honest exploratory result, not validated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "autism_subtypes"))

PAGANI_XLSX = ROOT / "data_external" / "pagani_2026" / "41593_2026_2287_MOESM6_ESM.xlsx"
if not PAGANI_XLSX.exists():
    PAGANI_XLSX = Path("/sessions/wizardly-admiring-tesla/mnt/uploads/41593_2026_2287_MOESM6_ESM.xlsx")


# Biological prior on subtype assignment (per Pagani Fig 1d).
# Hyperconnected: immune / late-onset / certain syndromic models
# Hypoconnected: synaptic / NMDA-glutamatergic / classical autism risk models
PAGANI_KNOWN_HYPER = {"Trem2", "Btbr", "Il6", "Mecp2", "Oxtr",
                      "16p11.2", "Sgsh", "Ube3a", "22q11.2"}
PAGANI_KNOWN_HYPO  = {"Fmr1", "Chd8", "Tsc2", "Cdkl5 [ko]", "Cdkl5 [ht]",
                      "Shank3", "En2", "Syn2", "Cntnap2",
                      "Nlgn3 [ko]", "Nlgn3-R451"}


def load_figura_1c():
    """Load the 20 × 1491 per-model matrix from MOESM6. Mask outliers."""
    wb = openpyxl.load_workbook(PAGANI_XLSX, data_only=True)
    ws = wb["Figura 1c"]
    labels, data = [], []
    for r in range(1, 21):
        row = ws[r]
        labels.append(row[0].value)
        vals = []
        for c in row[1:]:
            v = c.value
            if v is None:
                vals.append(np.nan)
                continue
            try:
                f = float(v)
                vals.append(f if abs(f) <= 5 else np.nan)
            except (TypeError, ValueError):
                vals.append(np.nan)
        # Pad or truncate to 1491
        if len(vals) < 1491:
            vals = vals + [np.nan] * (1491 - len(vals))
        data.append(vals[:1491])
    return np.array(data, dtype=np.float64), labels


def fill_nans_columnwise(X):
    col_mean = np.nanmean(X, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    inds = np.where(np.isnan(X))
    X = X.copy()
    X[inds] = np.take(col_mean, inds[1])
    return X


def main():
    print("=" * 80)
    print("Per-mouse-model HOMER translation showcase")
    print("=" * 80)

    # ---- Load and clean ----
    print("\nLoading Figura 1c (20 models × 1491 features)...")
    X_raw, labels = load_figura_1c()
    n_outliers = np.isnan(X_raw).sum()
    print(f"  raw shape: {X_raw.shape}")
    print(f"  outliers masked: {n_outliers} cells ({100*n_outliers/X_raw.size:.1f}%)")
    print(f"  model labels: {labels}")
    X = fill_nans_columnwise(X_raw)
    print(f"  after column-mean imputation: range "
          f"[{X.min():.2f}, {X.max():.2f}], mean {X.mean():.3f}")

    # ---- PCA + KMeans clustering in 1491-feature space ----
    print("\nPCA + KMeans (k=2) on 20 × 1491 matrix...")
    pca = PCA(n_components=2, random_state=0)
    pcs = pca.fit_transform(X)
    print(f"  PC1 explained variance: {pca.explained_variance_ratio_[0]*100:.1f}%")
    print(f"  PC2 explained variance: {pca.explained_variance_ratio_[1]*100:.1f}%")

    km = KMeans(n_clusters=2, n_init=20, random_state=0).fit(X)
    cluster_labels = km.labels_

    # Assign which cluster is hyper vs hypo using the biological prior
    cluster_0_models = {labels[i] for i in range(20) if cluster_labels[i] == 0}
    cluster_1_models = {labels[i] for i in range(20) if cluster_labels[i] == 1}
    # Cluster with more known-hyper models = hyper
    n_hyper_in_0 = len(cluster_0_models & PAGANI_KNOWN_HYPER)
    n_hyper_in_1 = len(cluster_1_models & PAGANI_KNOWN_HYPER)
    if n_hyper_in_1 > n_hyper_in_0:
        hyper_cluster, hypo_cluster = 1, 0
    else:
        hyper_cluster, hypo_cluster = 0, 1
    cluster_subtype = ["hyper" if c == hyper_cluster else "hypo" for c in cluster_labels]

    # Score against the prior
    print(f"\n{'Model':<14s} | {'inferred subtype':>16s} | {'prior subtype':>14s} | match?")
    print("-" * 65)
    n_match = 0
    n_known = 0
    rows = []
    for i, lbl in enumerate(labels):
        prior = "hyper" if lbl in PAGANI_KNOWN_HYPER else (
            "hypo" if lbl in PAGANI_KNOWN_HYPO else "(unknown)")
        inferred = cluster_subtype[i]
        match = "★" if prior == inferred else ("·" if prior == "(unknown)" else "✗")
        if prior in ("hyper", "hypo"):
            n_known += 1
            if prior == inferred:
                n_match += 1
        print(f"  {lbl:<14s} | {inferred:>16s} | {prior:>14s} | {match}")
        rows.append({"model": lbl, "inferred": inferred, "prior": prior,
                      "pc1": float(pcs[i, 0]), "pc2": float(pcs[i, 1])})
    print(f"\nClustering recovery: {n_match}/{n_known} known subtypes match the prior")

    # ---- HOMER per-subtype prediction ----
    # The Test 2c per-subtype prediction is in outputs/logs/autism_subtypes_full_matrix.json
    # The per-parcel prediction for hyper vs hypo can be reconstructed from
    # mouse-network Δ + π.
    print("\nHOMER per-subtype prediction (from Test 2c logic)...")
    from importlib import import_module
    st = import_module("04_subtype_translation")
    fm = import_module("07_full_matrix_translation")
    from homer.data import load_cached
    M, _ = load_cached("mouse", cache_dir=str(ROOT / "outputs/anndata"))
    H, _ = load_cached("human", cache_dir=str(ROOT / "outputs/anndata"))
    pi = np.load(ROOT / "outputs/coupling/pi_fc_plus_SC_with_all_packs.npy")

    data = st.load_pagani_subtype_matrices()
    mouse_pagani_net, mouse_pagani_names = fm.assign_mouse_pagani_networks(M.var)
    keep = mouse_pagani_net >= 0

    def _intensity(MM):
        Ma = np.abs(MM)
        return Ma.sum(axis=0) + Ma.sum(axis=1) - np.diag(Ma)

    def _per_subtype_prediction(M_subtype):
        intensity = _intensity(0.5 * (M_subtype + M_subtype.T))
        v = np.zeros(pi.shape[0])
        for i in range(len(mouse_pagani_names)):
            v[(mouse_pagani_net == i) & keep] = intensity[i]
        return v @ pi   # (2094,)

    hyper_human_pred = _per_subtype_prediction(data["mouse_hyper"])
    hypo_human_pred  = _per_subtype_prediction(data["mouse_hypo"])
    print(f"  hyper template: shape={hyper_human_pred.shape}, "
          f"range [{hyper_human_pred.min():.2f}, {hyper_human_pred.max():.2f}]")
    print(f"  hypo template:  shape={hypo_human_pred.shape}, "
          f"range [{hypo_human_pred.min():.2f}, {hypo_human_pred.max():.2f}]")

    # For each model, assign its HOMER-predicted human map = the hyper or hypo template
    # weighted by its cluster membership (soft via KMeans distance ratio)
    distances = km.transform(X)  # (20, 2) — distance to each centroid
    # Convert to soft membership: closer to hyper centroid = higher hyper-weight
    hyper_dist = distances[:, hyper_cluster]
    hypo_dist  = distances[:, hypo_cluster]
    # Softmax(inverse distance) for membership probability
    eps = 1e-6
    hyper_weight = (1.0 / (hyper_dist + eps)) / (1.0 / (hyper_dist + eps) + 1.0 / (hypo_dist + eps))
    hypo_weight = 1 - hyper_weight

    # Per-model human-space score = which subtype's HOMER prediction dominates
    print(f"\n{'Model':<14s} | {'inferred subtype':>16s} | "
          f"{'hyper-weight':>13s} | {'hypo-weight':>12s}")
    print("-" * 65)
    for i, lbl in enumerate(labels):
        rows[i]["hyper_weight"] = float(hyper_weight[i])
        rows[i]["hypo_weight"]  = float(hypo_weight[i])
        print(f"  {lbl:<14s} | {cluster_subtype[i]:>16s} | "
              f"{hyper_weight[i]:>13.3f} | {hypo_weight[i]:>12.3f}")

    # Save
    out = {
        "n_models":       int(len(labels)),
        "feature_outliers_masked_pct": float(100 * n_outliers / X_raw.size),
        "models":         rows,
        "n_match_known":  int(n_match),
        "n_known":        int(n_known),
        "pca_var": {
            "pc1": float(pca.explained_variance_ratio_[0]),
            "pc2": float(pca.explained_variance_ratio_[1]),
        },
        "homer_predictions": {
            "hyper_human_per_parcel": hyper_human_pred.tolist(),
            "hypo_human_per_parcel":  hypo_human_pred.tolist(),
        },
    }
    out_path = ROOT / "outputs" / "logs" / "pagani_per_model_translation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
