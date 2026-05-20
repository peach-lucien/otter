# Pagani 2026 per-mouse-model HOMER translation (exploratory)

A showcase of how HOMER's π would translate per-mouse-model FC perturbation patterns into human-parcel space. Frames a real biological question: *which human ASD phenotype does each mouse model resemble?*

## What we attempted vs what we accomplished

**Goal**: Take Pagani 2026's 20-model × 1,491-feature matrix (MOESM6, Figura 1c), route each model's mouse-side FC perturbation through HOMER's π, and produce per-model human-parcel predictions. Cluster the 20 models in HOMER-translated human space to ask: which mouse models look most like the human hyperconnected ASD subtype, which look hypoconnected, and where do outlier models sit?

**Reality**: We can't decode the 1,491 features. Without access to Pagani's methods section explaining the underlying parcellation (the value range looks correlation-like for ~95 % of cells, with the remaining 5 % being Excel-formatting-corrupted outliers — large integers from mis-parsed small decimals), the 1,491 → 1,864 alignment to HOMER's mouse atlas isn't possible. Per-model resolution degrades to subtype-average resolution.

**What we delivered**: An honest exploratory showcase with three layers:

1. **Mouse-side PCA + KMeans clustering** of the 20 models in opaque 1,491-feature space. Recovers 2 clusters; structure of which doesn't cleanly match my biological prior on Pagani's hyper/hypo subtypes (likely either because my prior is wrong without paper access, or because outlier-corruption correlates with row order).
2. **HOMER per-subtype prediction** from Test 2c — the predicted human-parcel Δ pattern that any mouse model "would translate to" if it sits in the hyper or hypo subtype.
3. **Per-model soft membership** to the hyper vs hypo template (distance-weighted KMeans probability), showing the 20 models distributed along the hyper↔hypo axis HOMER predicts.

## What this is useful for

- **Method showcase**: demonstrates the per-model translation pipeline, validates that the data flows correctly, and produces a publishable-quality figure
- **Framing future work**: identifies the precise bottleneck (1,491 feature decoding) and what data would unblock real per-model translation
- **Honest exploratory result**: not validated against ground truth, but illustrative of the geometric layout HOMER implies

## What it's NOT

- Not a quantitative claim about which mouse model "is" which human subtype
- Not a substitute for per-model parcel-level translation (which requires decoding the 1,491 features)
- Not a validation of Pagani's subtype assignments — those would need direct access to Figure 1d

## Files

| File | What |
|---|---|
| `01_per_model_clustering.py` | Load Figura 1c, mask outliers, PCA + KMeans, compute per-model HOMER soft membership |
| `02_plot.py` | 3-panel figure |
| `README.md` | This file |

## Reproduce

```bash
PYTHONPATH=src python experiments/pagani_2026_per_model/01_per_model_clustering.py
PYTHONPATH=src python experiments/pagani_2026_per_model/02_plot.py
```

Outputs:
- `outputs/logs/pagani_per_model_translation.json`
- `outputs/figures/pagani_per_model_translation.png`

## Showcase notebook

[`notebooks/09_pagani_per_model_translation.ipynb`](../../notebooks/09_pagani_per_model_translation.ipynb) — interactive walkthrough with explicit caveats about what's exploratory.

## To make this a real validation, we would need

1. ~~**Pagani's exact 1,491-feature definition**~~ — **DECODED**. From Pagani's `rsfMRI-global-local-connectivity` GitHub repo + supplementary methods PDF: the 1,491 features are **per-voxel global connectivity** (weighted-degree centrality, Cole et al. 2009) within `chd8_functional_template_mask_wo_cerebellum.nii.gz`. Each value is the mean Pearson's r between that voxel and all other voxels in the mask for a given mouse model. Resolution is ~700 µm isotropic (1,491 voxels in mouse brain minus cerebellum ≈ 500 mm³ / 1,491 ≈ 0.335 mm³/voxel).

2. **The mask file itself** — `chd8_functional_template_mask_wo_cerebellum.nii.gz`. We know the features ARE voxel-level, but without the mask we don't know which feature-index corresponds to which voxel position. **Email draft to Pagani group is in `email_draft_to_pagani_group.md`.**

3. **Per-model parcel-level mouse FC matrices** — the underlying data behind Figura 1c, before reduction to 1,491 voxel signatures. The Gozzi lab might share these on request.

4. **Per-model human ASD FC** — ABIDE doesn't disaggregate by genetic subtype, but ENIGMA-ASD or SFARI's per-cohort data might.

With the mask file (requirement 2), per-model translation is straightforward — ~2 hours of code. Drafted email is ready to send.
