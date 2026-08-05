# BICCN cell-type marker cross-species validation

We asked whether OTTER's π preserves cell-type-defining marker spatial patterns across mouse and human, using markers aligned to BICCN's cross-species cell-type atlases ([Yao 2023, Nature](https://www.nature.com/articles/s41586-023-06812-z); [Siletti 2023, Science](https://www.science.org/doi/10.1126/science.add7046)).

## Why this experiment

BICCN's atlases establish cross-species conservation of cell types with their defining markers (Pvalb interneurons, Sst interneurons, Vip interneurons, Camk2a glutamatergic neurons, Gfap astrocytes, Th dopaminergic neurons, etc.). This is a parallel test to Hodge 2019 layer-marker validation, but tests *cell-type-defining* markers (mostly area-level distributions) rather than *cortical-layer* markers (within-area structure).

**Hypothesis**: OTTER should preserve cell-type markers that have area-specific spatial distributions (interneuron class preferences across cortex, glia density variation, subcortical neuromodulator localisation).

## Result

We routed each mouse marker-expression map through π and correlated the predicted human map with the observed AHBA map. 23 cell-type markers were tested across 7 classes. 13 of 23 translate at empirical p < 0.05, with a mean Pearson r of +0.089 across all markers. The class breakdown:

| Class | n markers | n significant | Mean r |
|---|---:|---:|---:|
| Glutamatergic | 4 | 4/4 | +0.193 |
| Interneuron | 7 | 4/7 | +0.107 |
| GABA synthesis | 1 | 1/1 | +0.092 |
| Dopaminergic | 4 | 2/4 | +0.075 |
| Astrocyte | 2 | 1/2 | +0.037 |
| Microglia | 1 | 0/1 | +0.034 |
| Oligodendrocyte | 4 | 1/4 | +0.007 |

Per-marker (significant, descending):
- Drd1 +0.227, Slc17a7 +0.221, Vip +0.201, Calb2 +0.199, Pvalb +0.198, Slc17a6 +0.196, Camk2a +0.182, Grin1 +0.174 (all p ≈ 0)
- Gad1 +0.092, Drd2 +0.079, Reln +0.069, Plp1 +0.063, Gfap +0.051 (p ≤ 0.025)

Not significant: Th, Slc6a3, Aqp4, Olig2, Sox10, Mbp, Sst, Calb1, Lhx6, Cx3cr1.

## Interpretation

Cell-type marker spatial patterns translate across species at moderate strength. The glutamatergic markers are the strongest class (4/4, mean +0.193), followed by interneuron markers (4/7, mean +0.107). Glial and oligodendrocyte markers are mostly not significant. The magnitude (top markers around +0.2, mean +0.089) is well below the region-level tests (Beauchamp, Margulies gradient ~0.4), so these are a real but modest cross-species signal, consistent with the markers carrying area-level expression structure that π can route.

## How this compares to the Hodge result

| Test | Spatial structure | Translation result |
|---|---|---|
| Beauchamp 2022 | 22 specific region pairs | 57 % top-1, AUROC 0.90 |
| Margulies/Huntenburg | Brain-wide gradient | \|r\| = 0.54, spin p = 0.004 (survives) |
| BICCN cell-type markers | Cell-class spatial distributions | 13/23 significant, mean r +0.089 |
| Hodge layer markers | Cortical layer markers | 6/7 significant, mean r +0.23 |

Both cell-type and layer markers translate at moderate strength (mean r ≈ 0.1), below the region-level signal. The Schaefer-400 parcellation does not separate layers within an area, so the layer-marker result reflects the area-level distribution of those genes rather than within-area lamination.

## Files

| File | What |
|---|---|
| `01_cell_type_validation.py` | Per-marker test + per-class aggregation + comparison to Hodge |
| `02_plot.py` | 2-panel figure (per-marker r with null CI; per-class summary + Hodge comparison) |
| `README.md` | This file |

## Reproduce

```bash
PYTHONPATH=src python experiments/biccn_2023_cell_types/01_cell_type_validation.py
PYTHONPATH=src python experiments/biccn_2023_cell_types/02_plot.py
```

Outputs:
- `outputs/logs/biccn_2023_cell_types.json`
- `outputs/figures/biccn_2023_cell_types.png`


## Future work, full BICCN atlas data

This test uses OTTER's curated 61-gene panel + AHBA microarray as proxies for cell types. A full integration would:
1. Pull Yao 2023's per-CCFv3-region cell-type abundance tables (~5,000 cell types × ~700 regions, tens of GB)
2. Pull Siletti 2023's per-dissection-region cell-type abundance tables (~3,000 cell types × ~100 regions)
3. Map both to OTTER's parcellations via centroid alignment
4. Test cross-species correlations for actual transcriptomically-defined cell types, not gene-marker proxies

This would extend from ~25 cell-type-marker proxies to thousands of actual cell types, closer to BICCN's intended use case. Tractable but a multi-day data engineering project.

## Contrast reframe (2026-06-19), `03_contrast_reframe.py`

Per-gene smooth-map correlations are weak (mean r=0.089) and share cortical spatial
autocorrelation. We therefore tested cell-class contrasts instead (magnitude-cancelling,
like the Pagani contrast), against the fair translation-spin null:
- **Excitatory − inhibitory** (Glut − interneuron): r=+0.262, **spin p=0.001, survives.**
- Neuronal − glial: r=+0.049, spin p=0.58 (n.s.); glia aren't network-organised.
- Dopaminergic subcortical hotspot: top-decile overlap 17/124, hypergeometric p=0.10
  (marginal); full-map spin n.s.

**The E/I axis is a specific cross-species result** that the per-marker test
missed. OTTER preserves where excitatory versus inhibitory neurons dominate, beyond
spatial smoothness. Log: `outputs/logs/biccn_contrast_reframe.json`.

A higher-resolution upgrade (real per-region cell-type *abundance* from Yao 2023 +
Siletti 2023, not marker proxies) is scaffolded in `04_abundance_composition.py`
needs the two cell atlases + `abc_atlas_access`/`cellxgene-census`, so run locally
(not feasible in the disk-limited sandbox).
