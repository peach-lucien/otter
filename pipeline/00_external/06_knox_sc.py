"""Build a hybrid mouse SC matrix using Knox 2019 leaf-level cortical SC for
the cortical Garin anchors, falling back to existing summary-structure SC for
everything else.

The existing `data_external/mouse_sc.npy` is the Allen *summary-structure*
aggregation (~290 regions for 1864 parcels), which gives only 192 unique SC
fingerprints across the atlas. For example,
"L_Somatosensory cortex" and "L_Posterior parietal cortex" share an SC
fingerprint, and "L_Auditory cortex" and "R_Auditory cortex" share one too.

Knox 2019's voxel-level model evaluated at the *leaf* cortical structure level
gives 43 distinct cortical fingerprints. For each Garin anchor mapped to a
Knox cortical leaf set, we replace its SC fingerprint with a Knox-derived
average of the relevant leaves' rows.

Output: `data_external/mouse_sc_knox.npy`, same shape as `mouse_sc.npy`
(1864, 1864), but with cortical-anchor rows/columns recomputed using Knox
leaf-level data.

Hand-curated Garin → Knox mapping for the 11 cortical anchor pair_ids; the
remaining 10 pair_ids (subcortical + brainstem) are not in Knox's cortex-only
mask and keep their original SC vectors.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from homer.data import load_cached  # noqa: E402

EXT = ROOT / "data_external"


# Hand-curated map: Garin anchor region name → list of Knox cortical region
# abbreviations that together cover the Garin region.
GARIN_TO_KNOX = {
    "Medial prefrontal cortex (mPFC)":          ["ACAd", "ACAv", "PL", "ILA"],
    "Motor and premotor":                       ["MOp", "MOs"],
    "Somatosensory cortex":                     ["SSp-bfd", "SSp-tr", "SSp-ll", "SSp-ul",
                                                  "SSp-un", "SSp-n", "SSp-m", "SSs"],
    "Posterior parietal cortex":                ["VISa", "VISrl", "VISam", "VISpm"],
    "Visual striate cortex":                    ["VISp"],
    "Visual pre and extra striate cortex":      ["VISal", "VISl", "VISpl", "VISli", "VISpor"],
    "Auditory cortex (Superior temporal )":     ["AUDp", "AUDd", "AUDpo", "AUDv"],
    "Middle Temporal, Inferior temporal, Temporal pole (MIPT)":
                                                ["TEa", "PERI", "ECT"],
    "Insula and others in lateral sulcus":      ["AIp", "AIv", "AId", "GU", "VISC"],
    "Periarchicortex":                          ["RSPagl", "RSPd", "RSPv"],
    "Olfactory cortex":                         ["ORBl", "ORBm", "ORBvl"],
}


def main():
    # Load Knox 43-leaf cortical SC matrix (rows = source cortical leaves,
    # cols = target cortical leaves, ipsi only).
    knox = pd.read_csv(EXT / "knox_sc" / "normalized_connection_density_ipsi_ctx.csv",
                        index_col=0)
    print(f"Knox 43-leaf SC: {knox.shape}, sym? "
          f"{np.allclose(knox.values, knox.values.T, atol=1e-6)}")
    # Symmetrise for safety (Knox should already be symmetric since it's connection
    # density; tiny asymmetries can come from the regression model)
    knox_sym = 0.5 * (knox.values + knox.values.T)

    # Load mouse atlas
    M, _ = load_cached("mouse", cache_dir=ROOT / "outputs" / "anndata")
    sc_old = np.load(EXT / "mouse_sc.npy")
    n_parcels = sc_old.shape[0]
    print(f"Existing mouse_sc.npy: {sc_old.shape}")

    # Find positional indices of cortical anchors mapped to Knox
    named = M.var[M.var["garin_anchor"]].copy()
    named["clean"] = named["region"].apply(lambda s: re.sub(r"^[LR]_", "", s))
    cortical_pos: list[int] = []           # parcel index in mouse_sc array
    cortical_knox_set: list[list[str]] = [] # which Knox leaves to use for each
    cortical_region_name: list[str] = []
    for garin_name, knox_codes in GARIN_TO_KNOX.items():
        valid_codes = [c for c in knox_codes if c in knox.columns]
        if not valid_codes:
            print(f"  ⚠ no Knox columns for '{garin_name}', skipping")
            continue
        rows = named[named["clean"] == garin_name]
        if len(rows) == 0:
            continue
        for idx in rows.index:
            pos = M.var.index.get_loc(idx)
            cortical_pos.append(pos)
            cortical_knox_set.append(valid_codes)
            cortical_region_name.append(rows.loc[idx, "region"])

    n_cortical = len(cortical_pos)
    print(f"\nMapped {n_cortical} of 42 anchors to Knox cortical leaves "
          f"({n_cortical/42:.0%})")

    # Build per-anchor Knox fingerprint: average of the assigned leaf rows
    # (each row has 43 entries = SC strength to each Knox cortical region)
    anchor_knox_fp = np.zeros((n_cortical, knox_sym.shape[0]), dtype=np.float64)
    for i, codes in enumerate(cortical_knox_set):
        rows = [knox.loc[c].values for c in codes if c in knox.index]
        anchor_knox_fp[i] = np.mean(rows, axis=0) if rows else np.nan

    # Now compute pairwise correlation distance between these 22 anchors
    # using Knox fingerprints. Replace ONLY the within-cortical-anchor block
    # of the SC matrix; everything else keeps the existing values.
    # First: standardise per cortical anchor
    eps = 1e-9
    z = (anchor_knox_fp - anchor_knox_fp.mean(1, keepdims=True))
    z = z / z.std(1, keepdims=True).clip(min=eps)
    # log1p + standardise to match how sc_correlation_distance handles SC
    sc_row_for_dist = np.log1p(np.maximum(anchor_knox_fp, 0))
    sc_row_for_dist = (sc_row_for_dist - sc_row_for_dist.mean(1, keepdims=True))
    sc_row_for_dist = sc_row_for_dist / sc_row_for_dist.std(1, keepdims=True).clip(min=eps)
    knox_corr = (sc_row_for_dist @ sc_row_for_dist.T) / sc_row_for_dist.shape[1]
    knox_dist = (1.0 - knox_corr).clip(0.0, 2.0)

    # Build the new SC matrix as a copy of the old, then OVERWRITE the
    # cortical-anchor block with knox_dist (in *correlation-distance* form,
    # which matches what homer.costs.sc_correlation_distance produces from
    # an SC matrix).
    # The existing data_external/mouse_sc.npy is the RAW SC matrix (streamline
    # density), not the cost matrix. We need to produce a NEW raw SC matrix
    # whose log-corr-distance matches the Knox-derived signal for cortical
    # anchors. The cleanest way: directly produce a NEW per-parcel SC
    # fingerprint matrix and save that.
    #
    # Instead of trying to inject Knox-derived distances into the raw SC
    # matrix (would be inconsistent), we save the per-parcel feature vectors
    # in Knox 43-dim space for cortical anchors, alongside fallback rows from
    # existing SC for everything else (zero-padded).
    #
    # Final output: a (1864 × n_features) feature matrix that downstream code
    # turns into a 1864 × 1864 cost matrix via correlation_distance.

    # For each parcel, build a (43 + 1864) feature vector:
    #   first 43 entries = Knox fingerprint (zero for non-mapped parcels)
    #   next  1864 entries = existing SC row (signal for everything)
    # Then sc_correlation_distance(fingerprints) gives a 1864×1864 cost matrix
    # that uses Knox info for cortical anchors and existing SC for everything.
    fp = np.zeros((n_parcels, knox.shape[1] + n_parcels), dtype=np.float64)
    # Existing SC rows in the second half, every parcel gets this signal
    fp[:, knox.shape[1]:] = sc_old
    # Knox in the first half for the cortical anchors
    for i, pos in enumerate(cortical_pos):
        fp[pos, :knox.shape[1]] = anchor_knox_fp[i]

    # Save the parcel feature matrix (Knox + existing SC concatenated)
    out_path = EXT / "mouse_sc_knox_augmented.npy"
    np.save(out_path, fp.astype(np.float32))
    print(f"\nsaved augmented per-parcel SC features → {out_path}")
    print(f"  shape: {fp.shape}  (43 Knox cols + 1864 existing-SC cols)")
    print(f"  cortical anchors with Knox info: {n_cortical}")

    # Also save a dedicated COST matrix derived from this feature matrix,
    # using log1p-correlation (same recipe as sc_correlation_distance), then
    # apply the same `normalise_cost(scheme="max")` step the production
    # pipeline applies to Cm_SC (see pipeline/03c_build_multimodal_costs.py).
    # Without this, Cm_SC_knox lives in [0, ~1.32] while every other cost
    # matrix in the cache lives in [0, 1], that scale mismatch silently
    # over-weights SC by ~30% in any FGW solve that uses Cm_SC_knox.
    from homer.costs.normalisation import normalise_cost

    eps = 1e-6
    x = np.log1p(np.maximum(fp, 0))                # log-transform (positive only)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True).clip(min=eps)
    z = (x - mu) / sd
    r = (z @ z.T) / z.shape[1]
    d = 1.0 - r
    d = 0.5 * (d + d.T); np.fill_diagonal(d, 0.0)
    d = np.clip(d, 0.0, 2.0).astype(np.float64)
    d = normalise_cost(d, scheme="max").astype(np.float64)  # match Cm_SC scale
    print(f"  Cm_SC_knox cost matrix: shape={d.shape}, "
          f"range=[{d.min():.4f}, {d.max():.4f}], "
          f"off-diag mean={d[~np.eye(d.shape[0], dtype=bool)].mean():.4f}")

    # How many unique rows now? (compare like-for-like against the existing
    # *cost* matrix Cm_SC, NOT the raw streamline-density matrix mouse_sc.npy
    # which has only 192 unique rows. The cost-matrix version of the existing
    # SC has 454 unique rows after sc_correlation_distance + normalise_cost.)
    ann_path = ROOT / "outputs" / "anndata" / "full_costs.npz"
    existing = dict(np.load(ann_path))
    Cm_SC_existing = existing["Cm_SC"]
    unique_rows_knox = len(np.unique(np.round(d, 4), axis=0))
    unique_rows_existing = len(np.unique(np.round(Cm_SC_existing, 4), axis=0))
    print(f"  unique rows (Cm_SC_knox cost): {unique_rows_knox}")
    print(f"  unique rows (Cm_SC existing cost): {unique_rows_existing}")
    print(f"  resolution gain at the cost-matrix level: "
          f"{unique_rows_knox / unique_rows_existing:.2f}x")

    # Add to outputs/anndata/full_costs.npz under a NEW key, leaving Cm_SC alone
    existing["Cm_SC_knox"] = d.astype(np.float32)
    np.savez_compressed(ann_path, **existing)
    print(f"\nadded Cm_SC_knox to {ann_path}")


if __name__ == "__main__":
    main()
