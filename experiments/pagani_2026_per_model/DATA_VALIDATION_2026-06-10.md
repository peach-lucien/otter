# Pagani data ingest & validation — 2026-06-10

Validation of the files Silvia (Gozzi lab) shared, now in `data_crossspecies/pagani/`.
Goal: confirm the new data is what we think it is, and determine whether it
unblocks the per-model translation — **before** building on it.

This pass went deeper after a first round was (rightly) challenged: I checked the
full file inventory, all six template/mask NIfTIs, every sheet of MOESM6
(including the long-format source data), the published Methods, and ran reduction
tests. Findings below supersede the first-round note.

## TL;DR

- **Fig 1c CSV, subtype maps, region masks, templates — all check out.**
- **Per-model hyper/hypo labels are NOT a gap** — they're fully recoverable from
  the CSV row order (verified against the paper). The old experiment's prior was
  *inverted*.
- **The 1,491-feature → voxel mapping cannot be robustly reconstructed** from any
  delivered or published file. But that mapping is the *wrong thing to chase* —
  see "the real per-model ask" below.

## What each file is, and whether it's usable

### 1. `sorted_etiology_by_feature_matrix.csv` (Fig 1c) ✅
- 20 models × 1,491 features; row labels/order identical to MOESM6 `Figura 1c`.
- It is the **de-corrupted** MOESM6: the 1,441 Excel-mangled cells are fixed; on
  the ~28k clean overlapping cells the two agree to 1e-5. Use this, not MOESM6.
- Per the paper, each value is a **voxelwise weighted-degree-centrality**
  (global connectivity) difference, mutant − WT (Liska 2015 metric).

### 2. `rsfMRI-templates-main/` (6 NIfTIs) ✅ present
- Functional EPI grid, 100×100×18 (rodent-inflated voxels ≈ 0.23×0.23×0.6 mm).
- Voxel counts: full mask 11,129; **wo-cerebellum 10,111**; ventricles 81;
  wo-cerebellum-and-ventricles `_ag` 10,032 (values −1/0/+1 = L/R hemisphere split).
- **None of the six files contains 1,491 voxels.**

### 3. Fig 1d occurrence maps ✅
- `cluster1_…_pos` = hyperconnectivity, `cluster2_…_neg` = hypoconnectivity.
- Integer occurrence counts 0–5. **Allen CCFv3** grid (456×320×528), same family
  as HOMER's `annotation_25_fixed.nii.gz`. Route through π directly.
- These are **group/cluster-level**, not per-model.

### 4. `Region_masks/` ✅
- 13 binary conserved-region masks (incl. caudoputamen = striatum), all on the
  same Allen CCFv3 grid as the occurrence maps.

## The 1,491-feature question (the one I was challenged on)

**Can we robustly map the 10,111-voxel mask → 1,491 features? No.** Evidence:

- The paper (Methods + main text) says Fig 1c is **hierarchical clustering of the
  per-model voxelwise degree-centrality maps**. So the 1,491 columns are
  voxelwise values on a *downsampled* analysis grid.
- **No delivered or published file carries the feature-index → voxel key.** I
  checked: the 6 templates (none = 1,491), and every MOESM6 sheet. `Figure1a` is
  the long-format source (29,821 rows = 1 header + 20×1,491) but holds only
  `(etiology, value)` pairs — **no coordinate, no voxel index**.
- Block-downsampling the wo-cerebellum mask lands *near* 1,491 only at arbitrary
  anisotropic factors (4×2×1 → 1,482; 2×4×1 → 1,507) — neither exact nor
  principled. The retained-voxel set and the column ordering (the matrix is
  *sorted* by the dendrogram, not by voxel index) are unrecoverable.

**Conclusion:** inverting 10,111 → 1,491 is lossy and cannot be done robustly
from what exists. Crucially, it's also unnecessary (next section).

## Per-model subtype labels — recovered, not missing

The CSV is *sorted* by the clustering, and the split falls exactly on row order:

| rows | mean global-conn (avg) | subtype |
|---|---|---|
| 1–9: Fmr1, Chd8, Il6, Tsc2, Trem2, Btbr, Cdkl5[ko], Mecp2, Cdkl5[ht] | **+0.19** | **hyperconnectivity (n=9)** |
| 10–20: Shank3, En2, Syn2, Cntnap2, Nlgn3[ko], Nlgn3-R451, Oxtr, 16p11.2, Sgsh, Ube3a, 22q11.2 | **−0.26** | **hypoconnectivity (n=11)** |

This reproduces the paper's n=9 / n=11 split and every named example
(hyper: Cdkl5[ko], Fmr1, Chd8, Tsc2, Il6; hypo: En2, Shank3, 22q11.2, 16p11.2,
Ube3A, Sgsh). **⚠️ The old `01_per_model_clustering.py` prior is inverted** — it
labels Fmr1/Chd8/Tsc2 as *hypo* and 16p11.2/Sgsh/Ube3a as *hyper*; the truth is
the reverse. That prior must be replaced with this row-order assignment.

## Bottom line

- **Subtype-level translation: fully unblocked.** We have correct per-model
  hyper/hypo labels *and* the Allen-space occurrence maps. No decode needed.
- **Per-model voxelwise translation: still needs one specific thing — but it is
  NOT a "1,491 lookup."** The right object is the **20 per-model
  weighted-degree-centrality maps as NIfTIs in functional (or Allen) space** —
  the full-resolution Fig 1a/b maps. Those register to HOMER's mouse atlas and
  route through π directly. The lab's pipeline produces these natively, so it's a
  clean ask. Trying to reconstruct them from the 1,491-feature CSV is the wrong
  path (lossy, unrecoverable ordering).

## Suggested next step

Build the subtype-level translation now (correct labels in hand), and ask Silvia
for the 20 per-model degree-centrality NIfTIs (not a feature-index key) to enable
per-model translation later.

---
Source: Pagani, Gozzi et al., "Autism subtypes identified using cross-species
functional connectivity analyses," *Nature Neuroscience* (2026),
doi:10.1038/s41593-026-02287-z.
