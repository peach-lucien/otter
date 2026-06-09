# Warp-based rebuild: HOMER coordinate inputs under Paul's nonlinear DSURQE→CCFv3 warp

> **DEPRECATED / archival (as of v2).** The warp-driven coordinate rebuild
> implemented here has been superseded by the **v2 mouse package**
> (`corrs_mouse_v2.mat`), which now ships the pre-warped CCFv3 voxel
> indices (`ns_center_ix` for 25 µm, `AS_ix` for 200 µm) directly inside
> the .mat file. Production v2 paths consume those fields via the v2
> loader — see `../01b_mouse_sc_v2.py` and `../02c_mouse_genes_v2.py`.
> The scripts in this directory are kept for historical reference and to
> document how the v2 voxel indices were derived; they are **not** part of
> the current build path and should not be re-run as part of normal
> reproduction.

Replaces HOMER's heuristic 48-permutation + centroid-translation transform
(produced by `00c_align_mouse_to_ccf.py`) with Paul's elastix-derived
rigid + affine + B-spline nonlinear warp from `data_crossspecies/warpfields/`.

The warpfield itself has been validated end-to-end (see chat log
[2026-06-02]): templates bit-identical to Allen NRRD source, conventions
match Paul's description, striatum round-trip preserves 99.9997% of voxel
labels, all 1864 HOMER parcel centres map into the brain mask.

This directory contains five scripts that consume the warpfield to
regenerate every coordinate-dependent HOMER input we can build *without*
new RS data from Paul.

## Scripts (run in order)

| # | Script | Output | Notes |
|---|--------|--------|-------|
| 01 | `01_parcel_ccf_labels.py` | `data_external/_warp_rebuild/parcel_ccfv3_labels.{csv,json}` | Per-parcel Allen CCFv3 acronym at warped centre + majority vote across the parcel's voxel set. 1774/1864 (95.2%) get a positive Allen label at the centre; 1863/1864 (99.9%) under majority vote. |
| 02 | `02_compare_heuristic_vs_warp.py` | `data_external/_warp_rebuild/heuristic_vs_warp.csv` | OLD-vs-NEW per-parcel comparison: mean displacement **4.26 mm**, median **4.25 mm**, max **10.28 mm**; summary-structure agreement **25.8%** among valid parcels. Top transitions in the report. |
| 03 | `03_new_node_struct_idx.py` | `data_external/_warp_rebuild/node_struct_idx_warped.json` | Drop-in replacement for `mouse_sc_meta.json['node_struct_idx']`. Re-index the production summary-level SC with this to get a corrected per-parcel SC matrix — no re-download of Allen unionise data needed. |
| 04 | `04_warped_voxel_sets.py` | `data_external/_warp_rebuild/parcel_warped_voxels_{25,200}um.npz` | Per-parcel CCFv3 voxel index sets at 25 µm (annotation lookup) and 200 µm (ISH sampling), stored as concatenated arrays + offsets. |
| 05 | `05_rebuild_mouse_genes.py` | `data_external/mouse_genes_warped.npy`, `mouse_gene_list_warped.csv`, `mouse_genes_warped_meta.json` | Resamples all 61 Allen ISH energy volumes at the new voxel sets. **All 61 genes kept**; 36 reused from the Pagani cache, 25 freshly downloaded. Per-parcel cosine similarity vs OLD: mean 0.868, median 0.904, min 0.032; **3.3% of parcels have cosine < 0.5** — the tail where the transform change matters most. |

## What's still needed

1. **New FC matrix from Paul's RS-data extraction.** Only this requires
   Paul. Everything above is computable from the warpfield alone.
2. **Refit π** with the corrected gene matrix (and corrected FC once
   available) — should be done as one cascade so before/after numbers
   are clean.
3. **Re-run the validation suite** (Beauchamp, Margulies, BICCN, Hodge,
   ENIGMA, Fulcher, autism, TransBrain, …) on the refitted π. Hold
   until both FC and gene matrix are corrected.

## Files produced and their consumers

```
data_external/_warp_rebuild/
├── parcel_ccfv3_labels.csv            <- adds 'ccf_acronym' column to M.var
├── parcel_ccfv3_labels.json           <- provenance
├── heuristic_vs_warp.csv              <- diagnostic, for the writeup
├── node_struct_idx_warped.json        <- replaces mouse_sc_meta.json node_struct_idx
├── parcel_warped_voxels_25um.npz      <- input for future per-voxel queries
├── parcel_warped_voxels_200um.npz     <- input for 05_rebuild_mouse_genes.py
├── parcel_warped_voxels.json          <- metadata
└── gene_vector_cosine_old_vs_new.npy  <- (1864,) cosine per parcel

data_external/
├── mouse_genes_warped.npy             <- replacement for mouse_genes.npy
├── mouse_gene_list_warped.csv         <- same gene list, kept-only flag
└── mouse_genes_warped_meta.json
```

## Headline numbers

- **HOMER's rsmask grid = Paul's RS func grid**, verified bit-identical
  (origin -6.27, -8.19, -4.20; RAS+; 62×94×47 at 200 µm). The
  "1-voxel offset" we worried about was a MATLAB 1-based vs NIfTI
  0-based readout artifact.
- **OLD heuristic transform displaces parcel centres by 4.3 mm on
  average vs Paul's nonlinear warp** (max 10.3 mm). 78.7% of parcels
  off by >2 mm; 38.5% off by >5 mm.
- **Summary-structure agreement between OLD and NEW = 25.8%** among
  the 1421 parcels where both methods assign a summary structure.
  Top disagreements: fiber tracts ↔ CP, SUB ↔ CA1, MOp ↔ MOs, CA3 ↔
  fiber tracts, CUL ↔ IC, RSPv ↔ RSPd. These are mostly
  *neighbouring-region* confusions — gross alignment is OK, fine
  alignment is what the heuristic was getting wrong.
- **Gene matrix is mostly robust to the transform change** at the
  per-parcel level (median cosine 0.904 between OLD and NEW vectors),
  but a 3.3% tail of parcels see substantial shifts in gene
  fingerprint. Those will be the parcels whose anchor-pack labels and
  cross-species coupling are most likely to move on refit.

## Provenance

- Source warpfield: `data_crossspecies/warpfields/warpfield2SS.nii.gz`
  (Paul's elastix rigid + affine + B-spline, NS=CCFv3 25 µm,
  SS=DSURQE 70 µm).
- Source annotation: `data_crossspecies/_orig_ccfv3_2017/annotation_25_fixed.nii.gz`
  (uint32, PIR-correct affine; bit-identical data to canonical
  Allen NRRD).
- Source rsmask: `data_crossspecies/_mouse_mask/rsmask.nii` (200 µm
  62×94×47, RAS+, same world frame as Paul's SS DSURQE grid).
- Allen ISH cache reused from
  `experiments/autism_subtypes/allen_expansion/pagani_ish_cache/` (36
  of 61 genes were already present).
