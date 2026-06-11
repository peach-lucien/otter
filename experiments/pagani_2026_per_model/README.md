# Pagani 2026 subtype translation through HOMER's π

How HOMER's π translates Pagani's autism connectivity **subtypes** from mouse to
human, with each of the 20 mouse models placed on the hyper↔hypo axis. Frames a
real biological question: *does the human ASD subtype HOMER predicts from a mouse
subtype match the human subtype Pagani actually observed?*

> **Rewritten 2026-06-10** after the Gozzi lab shared the clean data. The earlier
> version PCA/KMeans-clustered the 20 models in the (Excel-corrupted) 1,491-feature
> space and labelled subtypes from a biological prior that was **inverted**. Both
> problems are now fixed. Background: [`DATA_VALIDATION_2026-06-10.md`](DATA_VALIDATION_2026-06-10.md).

## What it does now

1. **Loads the clean Fig 1c matrix** (`sorted_etiology_by_feature_matrix.csv`,
   20 models × 1,491 voxelwise weighted-degree-centrality features) — no outlier
   masking needed.
2. **Derives the subtype split from the data, and verifies it.** The CSV is sorted
   by Pagani's hierarchical clustering: rows 1–9 = hyperconnectivity (n=9), rows
   10–20 = hypoconnectivity (n=11). We assert this against mean global
   connectivity sign (hyper > 0, hypo < 0) rather than guessing.
3. **Leave-one-out per-model membership.** Each model is correlated to the mean
   hyper and hypo feature signature (excluding itself), placing all 20 on the
   hyper↔hypo axis. 17/20 fall on their own side; the 3 exceptions are the
   near-zero-polarization models (Trem2, Btbr, Syn2-class).
4. **Subtype translation through π.** Each subtype's mouse network signature
   (Pagani ED Fig 1) is routed through π to human-parcel space and aggregated to
   human networks, then compared to the observed human subtype pattern (Fig 4e).
   **Result (recommended π `pi_fc_plus_SC_with_all_packs.npy`): hyper is
   subtype-specific, hypo is not.** Predicted-hyper correlates with observed-hyper
   (r = +0.351) better than with observed-hypo (−0.254). Predicted-hypo does **not**
   match observed-hypo (−0.133) and leans toward observed-hyper (+0.205). So at
   network resolution HOMER recovers the *hyper* subtype cross-species but not the
   *hypo* subtype — consistent with the Direction 1 spatial result and the recurring
   hypo-side weakness.

   > **Note (2026-06-11 audit):** an earlier run of this experiment used the *base*
   > coupling `pi_fc_plus_SC.npy` and reported both subtypes as specific
   > (hyper +0.52, hypo +0.25). That was an artifact of the wrong π — the base
   > point-anchor coupling *before* the region-anchor packs. Under the recommended
   > `_with_all_packs` coupling the hypo subtype is **not** specific. See
   > `_audit/FINDINGS_LOG.md` F-001/F-005.

This deliberately does **not** depend on decoding the 1,491 features to voxels
(not robustly possible — see the validation note); the subtype network signatures
come from Pagani's own published matrices.

## What it's NOT

- Not a *per-voxel* per-model translation — that needs the 20 per-model
  degree-centrality NIfTIs (see "remaining" below), not the 1,491-feature CSV.
- The π routing is at network resolution (Pagani's 9 mouse / 8 human networks).

## Direction 1 result — parcel-resolution spatial routing (occurrence maps)

`03_spatial_subtype_routing.py` drives the mouse side from the actual Fig 1d
occurrence maps (aggregated to the 13 conserved regions via a verified Allen
region-name bridge; 1,052/1,864 parcels matched) instead of the 9-network
matrices. Finding, reported honestly:

- **Hyper subtype translates cross-species** (pred-hyper ↔ obs-hyper r=+0.78 vs
  obs-hypo −0.57).
- **Hypo subtype does NOT** (pred-hypo ↔ obs-hypo −0.50; it actually leans toward
  obs-hyper +0.69).

(Values under the recommended `_with_all_packs` coupling; the base-π run gave the
weaker +0.37/−0.25.)

This is consistent with the recurring "hypo is the weak side" theme in the
autism_subtypes tests, and has a clear methodological cause: the occurrence maps
are **unsigned consistency counts** (0–5), not signed connectivity Δ, so they
can't carry the hypo subtype's direction. The signed per-model degree-centrality
NIfTIs (requested in `email_draft_per_model_nifti.md`) are what a clean
parcel-resolution hypo translation needs. (Under the recommended π, the signed
*network* matrices in `01_per_model_clustering.py` also recover **hyper but not
hypo** — so the hypo-side weakness is consistent across both routings, not a
quirk of the occurrence maps alone.)

## Files

| File | What |
|---|---|
| `01_per_model_clustering.py` | Clean Fig 1c → verified subtype split → LOO membership → π subtype translation (signed networks; hyper specific, hypo not, under recommended π) |
| `03_spatial_subtype_routing.py` | Occurrence-map → conserved-region → π routing (parcel resolution; hyper specific, hypo not) |
| `02_plot.py` | 3-panel figure for `01` |
| `DATA_VALIDATION_2026-06-10.md` | Ingest/validation of the shared data package |
| `email_draft_per_model_nifti.md` | Request to Silvia for the 20 signed per-model maps |
| `README.md` | This file |

## Reproduce

```bash
PYTHONPATH=src python experiments/pagani_2026_per_model/01_per_model_clustering.py
PYTHONPATH=src python experiments/pagani_2026_per_model/02_plot.py
```

Outputs:
- `outputs/logs/pagani_subtype_translation_corrected.json`
- `outputs/figures/pagani_subtype_translation_corrected.png`

## Showcase notebook

[`notebooks/09_pagani_per_model_translation.ipynb`](../../notebooks/09_pagani_per_model_translation.ipynb) — interactive walkthrough of the corrected analysis (verified subtype split, LOO membership, π subtype-specificity, and the Direction 1 spatial-routing limit). Rewritten 2026-06-10; supersedes the earlier exploratory PCA/KMeans version.

## To make this a real validation, we would need

> **Status update 2026-06-10** — Gozzi lab shared a data package (now in
> `data_crossspecies/pagani/`). Full ingest/validation in
> [`DATA_VALIDATION_2026-06-10.md`](DATA_VALIDATION_2026-06-10.md). Net effect:
> the **subtype-level** route is now unblocked, but the **per-model 1,491-feature**
> route is *not* — the decode below was wrong.

1. ~~**Pagani's exact 1,491-feature definition** — DECODED as per-voxel global connectivity in `chd8_functional_template_mask_wo_cerebellum.nii.gz`~~ — **FALSIFIED (2026-06-10)**. The shared mask contains **10,111** voxels, not 1,491, and the `rsfMRI-global-local-connectivity` repo confirms the maps are voxelwise (≈10,111 values). The 1,491 Fig 1c columns are a *reduced/parcel-level* summary whose index→location mapping is still unknown. The earlier ~700 µm / "0.335 mm³ per voxel" reasoning was unfounded.

2. ~~**The mask file itself**~~ — **RECEIVED** (`rsfMRI-templates-main/`, functional grid 100×100×18 @ 2.3×2.3×6 mm). Necessary but, per (1), **not sufficient** to decode the 1,491 features.

3. **The 20 per-model voxelwise degree-centrality maps as NIfTIs** *(the real per-model blocker — NOT a "1,491 key")* — the full-resolution Fig 1a/b maps in functional or Allen space, one per model. These register to HOMER's mouse atlas and route through π directly. The 1,491-feature CSV cannot substitute: it is a downsampled, dendrogram-sorted reduction with no published feature-index → voxel key, so it can't be inverted robustly (see `DATA_VALIDATION_2026-06-10.md`).

4. **Per-model human ASD FC** — ABIDE doesn't disaggregate by genetic subtype, but ENIGMA-ASD or SFARI's per-cohort data might.

**Newly available — subtype-level translation (no decode needed):** the Fig 1d
occurrence maps (`cluster1_…_pos` = hyper, `cluster2_…_neg` = hypo) are in Allen
CCFv3 space and route directly through π. Moreover the **per-model hyper/hypo
labels are recoverable** straight from the CSV row order — rows 1–9 (mean global
connectivity +0.19) = hyperconnectivity (n=9), rows 10–20 (−0.26) =
hypoconnectivity (n=11), matching the paper. **⚠️ This is inverted relative to the
`PAGANI_KNOWN_HYPER/HYPO` prior currently hard-coded in
`01_per_model_clustering.py`** (which wrongly lists Fmr1/Chd8/Tsc2 as hypo) — that
prior must be replaced.
