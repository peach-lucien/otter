# BICCN cell-type marker cross-species validation

Tests whether HOMER's π preserves cell-type-defining marker spatial patterns across mouse and human, using markers aligned to BICCN's cross-species cell-type atlases ([Yao 2023, Nature](https://www.nature.com/articles/s41586-023-06812-z); [Siletti 2023, Science](https://www.science.org/doi/10.1126/science.add7046)).

## Why this experiment

BICCN's atlases establish cross-species conservation of cell types with their defining markers (Pvalb interneurons, Sst interneurons, Vip interneurons, Camk2a glutamatergic neurons, Gfap astrocytes, Th dopaminergic neurons, etc.). This is a parallel test to Hodge 2019 layer-marker validation — but tests *cell-type-defining* markers (mostly area-level distributions) rather than *cortical-layer* markers (within-area structure).

**Hypothesis**: HOMER should preserve cell-type markers that have area-specific spatial distributions (interneuron class preferences across cortex, glia density variation, subcortical neuromodulator localisation) — even though it failed for layer markers (which are within-area).

## Result

23 cell-type markers tested across 8 classes. Mean Pearson r over all markers = +0.010 (similar to Hodge's −0.001), but the class-level breakdown reveals a clear pattern:

| Class | n markers | n significant | Mean r |
|---|---:|---:|---:|
| **Astrocyte** | 2 | **2/2** | **+0.067** |
| **Dopaminergic** | 4 | 2/4 | +0.038 |
| **Oligodendrocyte** | 4 | 2/4 | +0.012 |
| Microglia | 1 | 0/1 | +0.012 |
| Glutamatergic | 4 | 0/4 | +0.008 |
| GABA synthesis | 1 | 0/1 | −0.013 |
| **Interneuron** | 7 | 1/7 | **−0.019** |

Strongest individual hits:
- **Th (tyrosine hydroxylase, dopaminergic)**: r = +0.105, empirical p < 0.001
- **Aqp4 (astrocyte)**: r = +0.080, empirical p < 0.001
- **Slc6a3 (DAT, dopaminergic)**: r = +0.061, empirical p < 0.001
- **Plp1 (oligodendrocyte)**: r = +0.058, empirical p < 0.001
- **Gfap (astrocyte)**: r = +0.054, empirical p = 0.035
- **Olig2 (oligodendrocyte TF)**: r = +0.052, empirical p = 0.010

Null markers: Pvalb, Sst, Vip, Calb1, Reln, Lhx6, Camk2a, Slc17a7, Slc17a6, Grin NMDA-receptors, Gad1+Gad2, Drd1, Drd2, Cx3cr1.

## Interpretation

**HOMER preserves regionally-concentrated cell-type signals but not broadly-distributed cortical class markers.**

The successful markers all have strong *anatomical-region-specific* spatial patterns:
- Th and Slc6a3 are highly concentrated in midbrain (substantia nigra, VTA) — HOMER's subcortical anchors bind these strongly
- Gfap and Aqp4 (astrocytes) vary systematically across cortical areas (thickness/composition differences between sensorimotor vs association cortex)
- Plp1, Olig2 (oligodendrocytes) concentrate in white-matter-rich regions

The failed markers are *broadly distributed* across cortex with subtle within-cortex variations:
- Pvalb, Sst, Vip interneurons are present throughout cortex with weak area gradients
- Camk2a glutamatergic neurons are pan-cortical
- Drd1, Drd2 dopamine receptors are broadly distributed across striatum + cortex

## How this complements the Hodge result

Together with Hodge 2019 (layer markers), this completes the resolution-boundary story:

| Test | Spatial structure | Translation result |
|---|---|---|
| Beauchamp 2022 | 22 specific region pairs | 37 % top-1, 100 % qualified top-3 |
| Pagani Test 2c | Network-pair Δ | r = +0.527, p = 0.0009 |
| Margulies/Huntenburg | Brain-wide gradient | r = +0.144, p = 4e-11 |
| **BICCN Th/Aqp4** | **Region-concentrated cell types** | **r = +0.05 to +0.10 per marker, several p < 0.001** |
| Hodge layer markers | Within-area lamination | Null (only RORB significant) |
| **BICCN interneurons** | **Broadly-cortical cell classes** | **Null** |

The cleanest reading: HOMER preserves **regional / area-level** cross-species signal. Where signal is regionally concentrated (Th in midbrain, Pagani's network-pair Δ at the network level, the principal gradient as a region-ordering), HOMER captures it. Where signal is broadly distributed across cortex with only subtle within-area variation (Pvalb interneurons, cortical layer markers), HOMER doesn't have the resolution to translate it.

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

## Showcase notebook

See [`notebooks/10_biccn_cell_type_markers.ipynb`](../../notebooks/10_biccn_cell_type_markers.ipynb) for an interactive walkthrough.

## Future work — full BICCN atlas data

This test uses HOMER's curated 61-gene panel + AHBA microarray as proxies for cell types. A full integration would:
1. Pull Yao 2023's per-CCFv3-region cell-type abundance tables (~5,000 cell types × ~700 regions, tens of GB)
2. Pull Siletti 2023's per-dissection-region cell-type abundance tables (~3,000 cell types × ~100 regions)
3. Map both to HOMER's parcellations via centroid alignment
4. Test cross-species correlations for actual transcriptomically-defined cell types, not gene-marker proxies

This would extend from ~25 cell-type-marker proxies to thousands of actual cell types — closer to BICCN's intended use case. Tractable but a multi-day data engineering project.
