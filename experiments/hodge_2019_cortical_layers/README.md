# Hodge 2019 cortical-layer-marker validation

Tests whether HOMER's π preserves cortical-layer-marker spatial patterns across mouse and human, using the canonical layer-defining transcription factors from [Hodge et al. 2019, *Nature*](https://www.nature.com/articles/s41586-019-1506-7).

## Why this experiment

[Hodge 2019](https://www.nature.com/articles/s41586-019-1506-7) ("Conserved cell types with divergent features in human versus mouse cortex") showed that canonical layer-defining transcription factors maintain their layer-specific spatial expression across mouse and human cortex. We use these markers as an independent quantitative test of HOMER's anatomical fidelity:

- **CUX1, CUX2, SATB2** → upper layers (L2/3)
- **RORB** → granular L4
- **FEZF2** → infragranular L5
- **TBR1, FOXP2** → deep layers (L6)

This is a Beauchamp-independent test — Hodge's data comes from Allen ISH (mouse) and AHBA microarray (human), which are different platforms from Beauchamp 2022's transcriptomic-similarity dataset. Agreement here is genuinely independent evidence.

## Result — informative null

| Test | Pearson r | Significant? |
|---|---:|:---:|
| Per-marker, all 2,094 parcels | 6 of 7 near zero, RORB +0.07 | only RORB (emp p=0.002) |
| Per-layer-group, cortex only (1,768 parcels) | L4 +0.07, others near zero | only L4 (emp p=0.002) |
| Upper − deep contrast (cortex only) | r=−0.04 | not significant |
| Lobe-aggregated (n=8 Yeo nets) | all positive (r=+0.11 to +0.23) | underpowered (n=8) |

**HOMER does NOT preserve fine-grained cortical-layer-marker spatial patterns at parcel resolution.** Only RORB (L4) translates significantly — and that works because L4 has area-specific spatial structure (concentrated in primary sensory cortices V1/S1/A1) which HOMER's anchors strongly bind, not because of laminar conservation.

## What this tells us about HOMER's resolution

HOMER's anchors are **anatomical-area-level** (mouse mPFC ↔ human mPFC), not **layer-level**. At parcel resolution (~10-30 voxels per mouse parcel, mixing all 6 cortical layers), the layer-specific marker signal is averaged out on both sides, and π has no information that would preserve laminar geometry across species.

This establishes a useful resolution boundary: HOMER works at the area / network level, but cannot translate fine-grained laminar gene expression. If you need layer-level translation, you'd need either a finer parcellation that preserves laminar boundaries or anchor packs specifying layer-level correspondences.

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
