# Hodge 2019 cortical-layer-marker validation

Tests whether HOMER's π preserves cortical-layer-marker spatial patterns across mouse and human, using the canonical layer-defining transcription factors from [Hodge et al. 2019, *Nature*](https://www.nature.com/articles/s41586-019-1506-7).

## Why this experiment

[Hodge 2019](https://www.nature.com/articles/s41586-019-1506-7) ("Conserved cell types with divergent features in human versus mouse cortex") showed that canonical layer-defining transcription factors maintain their layer-specific spatial expression across mouse and human cortex. We use these markers as an independent quantitative test of HOMER's anatomical fidelity:

- **CUX1, CUX2, SATB2** → upper layers (L2/3)
- **RORB** → granular L4
- **FEZF2** → infragranular L5
- **TBR1, FOXP2** → deep layers (L6)

This is a Beauchamp-independent test — Hodge's data comes from Allen ISH (mouse) and AHBA microarray (human), which are different platforms from Beauchamp 2022's transcriptomic-similarity dataset. Agreement here is genuinely independent evidence.

## Result

Six of the seven layer markers translate at parcel resolution. The three upper-layer markers (CUX1, CUX2, SATB2) at +0.083, +0.176 and +0.189; RORB (L4) at +0.090; FEZF2 (L5) at +0.168; and one L6 marker at +0.108. The other L6 marker is not significant (r = +0.019, p = 0.25). Mean across the seven is +0.119, against a permuted-π null near zero.

## What this means

The Schaefer-400 parcellation does not separate layers within an area — each mouse parcel mixes the cortical layers — so this test measures the area-level spatial distribution of the layer-marker genes, not within-area lamination. At that level six of seven markers carry cross-species signal that π routes, at a strength (mean +0.119) comparable to the BICCN cell-type markers and below the region-level tests (~0.4). It does not show that HOMER resolves laminar geometry; the parcellation has no information at that scale.

## Files

| File | What |
|---|---|
| `01_layer_marker_validation.py` | Per-marker cross-species translation (all 2,094 parcels) + permuted-π null |
| `02_layer_marker_refined.py` | Cortex-only + layer-group composites + upper−deep contrast + lobe-aggregated test |
| `03_plot.py` | Bar plot of per-marker + per-layer-group results vs null |
| `README.md` | This file |

## Reproduce

```bash
PYTHONPATH=src python experiments/hodge_2019_cortical_layers/01_layer_marker_validation.py
PYTHONPATH=src python experiments/hodge_2019_cortical_layers/02_layer_marker_refined.py
PYTHONPATH=src python experiments/hodge_2019_cortical_layers/03_plot.py
```

Outputs:
- `outputs/logs/hodge_2019_layer_markers.json`
- `outputs/logs/hodge_2019_layer_markers_refined.json`
- `outputs/figures/hodge_2019_layer_markers.png`

## Showcase notebook

See [`notebooks/06_hodge_2019_layer_markers.ipynb`](../../notebooks/06_hodge_2019_layer_markers.ipynb) for an interactive walkthrough of the analysis.

## Areal-type reframe (2026-06-19) — `03_areal_type_reframe.py`

Schaefer-400 can't resolve layers, so a laminar test is impossible by construction.
We recast it as the question it *can* answer — does π preserve cortical AREAL TYPE
(the supragranular↔infragranular / eulaminate↔agranular axis)? Cortex-only, fair
spin null:
- supragranular − infragranular: r=−0.02, spin p=0.71 (n.s.)
- granular L4 − infragranular: r=+0.09, spin p=0.055 (marginal, n.s.)
- supragranular − granular: r=−0.04, spin p=0.52 (n.s.)

So the layer-marker *gene* contrasts do not recover areal type through π. The
cytoarchitectural axis DOES survive when measured **structurally** — that is the
Fulcher 2019 T1w:T2w + Goulas-type result (spin p=0.021/0.010). Conclusion: Hodge is
genuinely capped (laminar impossible at this parcellation, gene-contrast areal-type
n.s.); cite Fulcher for cross-species cytoarchitecture, and keep Hodge as an honest
area-level-only result. Log: `outputs/logs/hodge_areal_type_reframe.json`.
