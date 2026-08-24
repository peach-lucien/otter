"""Compare OTTER per-disorder predictions to ENIGMA observed maps.

REQUIRES external data: ENIGMA per-region Cohen's d cortical-thickness effect
sizes per disorder. Easiest source is the ENIGMA Toolbox repository:

    git clone https://github.com/MICA-MNI/ENIGMA.git
    # then look in: enigmatoolbox/datasets/summary_statistics/

Each disorder ships a CSV like `cortical_thickness_<disorder>.csv` with
Cohen's d per Desikan-Killiany region. Place the CSVs in:

    otter/data_external/enigma/
        cortical_thickness_22q11.csv
        cortical_thickness_adhd.csv
        cortical_thickness_asd.csv
        cortical_thickness_bipolar.csv
        cortical_thickness_depression.csv
        cortical_thickness_ocd.csv
        cortical_thickness_schizophrenia.csv

This script then:
  1. Maps Desikan-Killiany regions to OTTER's 2,094-parcel atlas via
     nearest-MNI-centroid (Desikan-Killiany centroids in MNI mm are
     well-tabulated).
  2. Aggregates OTTER's per-disorder predicted patterns from 01_per_disorder_prediction.py to
     Desikan-Killiany region-level scores.
  3. Correlates OTTER-predicted region scores vs ENIGMA observed Cohen's d.
  4. Cross-disorder specificity check, whether OTTER's autism prediction
     correlates more with ENIGMA-autism than with ENIGMA-schizophrenia.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from otter.data import load_cached


# Desikan-Killiany MNI centroids (approximate, from Freesurfer template + DK
# atlas labels). Used to map DK regions to OTTER's 2,094 parcels via
# nearest-centroid matching. These are L/R hemisphere centroids in MNI mm.
# Source: aggregated from various Freesurfer-based atlases.
DK_REGIONS_MNI = {
    # (region_name, x_left, y_left, z_left)
    "bankssts":                (-58, -50, 4),
    "caudalanteriorcingulate": (-6, 30, 30),
    "caudalmiddlefrontal":     (-35, 30, 36),
    "cuneus":                  (-10, -80, 22),
    "entorhinal":              (-22, -8, -30),
    "fusiform":                (-35, -50, -18),
    "inferiorparietal":        (-46, -64, 34),
    "inferiortemporal":        (-52, -25, -22),
    "isthmuscingulate":        (-12, -38, 30),
    "lateraloccipital":        (-22, -90, 8),
    "lateralorbitofrontal":    (-22, 24, -22),
    "lingual":                 (-12, -68, -2),
    "medialorbitofrontal":     (-6, 50, -12),
    "middletemporal":          (-58, -32, -10),
    "parahippocampal":         (-22, -28, -18),
    "paracentral":             (-6, -28, 60),
    "parsopercularis":         (-50, 18, 18),
    "parsorbitalis":           (-46, 36, -12),
    "parstriangularis":        (-46, 28, 22),
    "pericalcarine":           (-12, -78, 6),
    "postcentral":             (-40, -25, 50),
    "posteriorcingulate":      (-6, -38, 38),
    "precentral":              (-40, -10, 50),
    "precuneus":               (-10, -56, 42),
    "rostralanteriorcingulate": (-6, 38, 20),
    "rostralmiddlefrontal":    (-32, 50, 18),
    "superiorfrontal":         (-12, 30, 50),
    "superiorparietal":        (-24, -64, 50),
    "superiortemporal":        (-56, -16, -2),
    "supramarginal":           (-54, -45, 28),
    "frontalpole":             (-10, 64, -8),
    "temporalpole":            (-38, 14, -32),
    "transversetemporal":      (-46, -22, 12),
    "insula":                  (-38, -2, 6),
}


def dk_to_otter_parcels(H_var, dk_regions):
    """Map each DK region to its nearest-MNI-centroid OTTER parcel set.
    Returns dict {dk_region: list of OTTER parcel indices}."""
    H_xyz = H_var[["x", "y", "z"]].to_numpy()
    out = {}
    for name, (x_l, y_l, z_l) in dk_regions.items():
        # Left + right hemisphere centroids
        centroids = np.array([[x_l, y_l, z_l], [-x_l, y_l, z_l]])
        # Find nearest OTTER parcels (one per hemisphere)
        d_l = np.linalg.norm(H_xyz - centroids[0], axis=1)
        d_r = np.linalg.norm(H_xyz - centroids[1], axis=1)
        # Take all parcels within 20mm of either centroid
        idx_l = np.where(d_l < 20)[0]
        idx_r = np.where(d_r < 20)[0]
        out[name] = sorted(set(idx_l.tolist() + idx_r.tolist()))
    return out


def aggregate_per_disorder_to_dk(predicted, dk_to_parcels):
    """Aggregate per-parcel OTTER prediction to per-DK-region scores."""
    out = {}
    for region, parcel_idxs in dk_to_parcels.items():
        if not parcel_idxs:
            out[region] = np.nan
            continue
        out[region] = float(predicted[parcel_idxs].mean())
    return out


def main():
    print("=" * 80)
    print("OTTER × ENIGMA cross-disorder comparison")
    print("=" * 80)

    # ---- Check ENIGMA data is present ----
    enigma_dir = ROOT / "data_external" / "enigma"
    if not enigma_dir.exists() or not list(enigma_dir.glob("cortical_thickness_*.csv")):
        print(f"\nERROR: ENIGMA cortical-thickness CSVs not found in {enigma_dir}")
        print(f"\nTo run this experiment:")
        print(f"  1. git clone https://github.com/MICA-MNI/ENIGMA.git /tmp/ENIGMA")
        print(f"  2. mkdir -p {enigma_dir}")
        print(f"  3. cp /tmp/ENIGMA/enigmatoolbox/datasets/summary_statistics/"
              f"cortical_thickness_*.csv {enigma_dir}/")
        print(f"  4. Re-run this script.")
        sys.exit(1)

    # ---- Load OTTER per-disorder predictions ----
    preds = np.load(ROOT / "outputs/coupling/per_disorder_predictions.npz")
    H, _ = load_cached("human", cache_dir=str(ROOT / "outputs/anndata"))
    print(f"\nOTTER per-disorder predictions: {list(preds.keys())}")

    # ---- Build DK → OTTER parcel mapping ----
    print(f"\nMapping {len(DK_REGIONS_MNI)} Desikan-Killiany regions to OTTER parcels...")
    dk_to_parcels = dk_to_otter_parcels(H.var, DK_REGIONS_MNI)
    print(f"  median OTTER parcels per DK region: "
          f"{int(np.median([len(v) for v in dk_to_parcels.values()]))}")

    # ---- Aggregate OTTER predictions per DK region ----
    otter_per_dk = {}
    for disorder in preds.keys():
        otter_per_dk[disorder] = aggregate_per_disorder_to_dk(preds[disorder], dk_to_parcels)

    # ---- Load ENIGMA CSVs and align region labels ----
    print(f"\nLoading ENIGMA cortical-thickness Cohen's d per disorder...")
    enigma_data = {}
    for csv in sorted(enigma_dir.glob("cortical_thickness_*.csv")):
        disorder = csv.stem.replace("cortical_thickness_", "")
        df = pd.read_csv(csv)
        # Heuristic: assume DK region names are in a 'Structure' column
        # or first column. ENIGMA CSVs typically have 'Structure' + d_icv etc.
        col_struct = next((c for c in df.columns
                            if "struct" in c.lower() or "region" in c.lower()), df.columns[0])
        col_d = next((c for c in df.columns
                       if c.lower().startswith("d_") or "cohen" in c.lower()), None)
        if col_d is None:
            print(f"  {disorder}: no Cohen's d column found, skipping")
            continue
        # Normalize region names
        d_dict = {}
        for _, row in df.iterrows():
            name = str(row[col_struct]).lower().strip().replace("lh_", "").replace("rh_", "")
            d_dict[name] = float(row[col_d]) if pd.notna(row[col_d]) else np.nan
        enigma_data[disorder] = d_dict
        print(f"  {disorder}: {len(d_dict)} regions loaded")

    # ---- Cross-disorder correlation: OTTER-predicted vs ENIGMA-observed ----
    print(f"\n{'='*80}")
    print(f"Cross-disorder correlation (OTTER-predicted vs ENIGMA-observed)")
    print(f"{'='*80}")
    otter_disorders = list(preds.keys())
    enigma_disorders = list(enigma_data.keys())

    corr_matrix = np.zeros((len(otter_disorders), len(enigma_disorders)))
    for i, h_d in enumerate(otter_disorders):
        for j, e_d in enumerate(enigma_disorders):
            # Align regions: which DK regions are in both?
            common = set(otter_per_dk[h_d].keys()) & set(enigma_data[e_d].keys())
            if len(common) < 10:
                corr_matrix[i, j] = np.nan
                continue
            hv = np.array([otter_per_dk[h_d][r] for r in sorted(common)])
            ev = np.array([enigma_data[e_d][r]  for r in sorted(common)])
            valid = np.isfinite(hv) & np.isfinite(ev)
            if valid.sum() < 10:
                corr_matrix[i, j] = np.nan
                continue
            r, _ = pearsonr(hv[valid], ev[valid])
            corr_matrix[i, j] = r

    print(f"\n{'':<22s}" + "".join(f"{d[:10]:>12s}" for d in enigma_disorders))
    for i, h_d in enumerate(otter_disorders):
        row = "".join(f"{corr_matrix[i,j]:>+12.3f}" if not np.isnan(corr_matrix[i,j])
                       else f"{'N/A':>12s}" for j in range(len(enigma_disorders)))
        print(f"  {h_d:<20s}{row}")

    # Specificity: diagonal vs off-diagonal
    diag_pairs = [(i, j) for i, h in enumerate(otter_disorders)
                  for j, e in enumerate(enigma_disorders)
                  if h.lower().replace("_", "") in e.lower().replace("_", "")
                  or e.lower().replace("_", "") in h.lower().replace("_", "")]
    diag_vals = [corr_matrix[i, j] for i, j in diag_pairs if not np.isnan(corr_matrix[i, j])]
    off_diag_vals = [corr_matrix[i, j] for i in range(len(otter_disorders))
                     for j in range(len(enigma_disorders))
                     if (i, j) not in diag_pairs and not np.isnan(corr_matrix[i, j])]
    print(f"\nDiagonal pairs (OTTER-X vs ENIGMA-X): mean r = {np.mean(diag_vals):+.3f}")
    print(f"Off-diagonal pairs (OTTER-X vs ENIGMA-Y): mean r = {np.mean(off_diag_vals):+.3f}")
    print(f"  → If OTTER's predictions are disorder-specific, diagonal should beat off-diagonal.")
    print(f"  → OTTER's per-disorder predictions are near-identical (r > 0.97), so")
    print(f"     diagonal ≈ off-diagonal is expected here.")

    out = {
        "otter_disorders":   otter_disorders,
        "enigma_disorders":  enigma_disorders,
        "correlation_matrix": corr_matrix.tolist(),
        "diagonal_mean":     float(np.mean(diag_vals)) if diag_vals else None,
        "off_diagonal_mean": float(np.mean(off_diag_vals)) if off_diag_vals else None,
        "dk_otter_aggregation": {d: {r: float(v) for r, v in otter_per_dk[d].items()
                                       if not np.isnan(v)}
                                  for d in otter_disorders},
    }
    out_path = ROOT / "outputs" / "logs" / "enigma_phase2_comparison.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
