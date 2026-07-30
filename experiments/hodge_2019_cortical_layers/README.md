# Hodge 2019 cortical-layer-marker validation

We asked whether HOMER's π preserves cortical-layer-marker spatial patterns across mouse and human, using the canonical layer-defining transcription factors from [Hodge et al. 2019, *Nature*](https://www.nature.com/articles/s41586-019-1506-7).

## Why this experiment

[Hodge 2019](https://www.nature.com/articles/s41586-019-1506-7) ("Conserved cell types with divergent features in human versus mouse cortex") showed that canonical layer-defining transcription factors maintain their layer-specific spatial expression across mouse and human cortex. We use these markers as an independent quantitative test of HOMER's anatomical fidelity:

- **CUX1, CUX2, SATB2** → upper layers (L2/3)
- **RORB** → granular L4
- **FEZF2** → infragranular L5
- **TBR1, FOXP2** → deep layers (L6)

This is a Beauchamp-independent test. Hodge's data comes from Allen ISH (mouse) and AHBA microarray (human), which are different platforms from Beauchamp 2022's transcriptomic-similarity dataset. Agreement here is independent evidence.

## Result

We routed each mouse layer-marker expression map through π and correlated the predicted human map with the observed AHBA map, parcel by parcel. Six of the seven markers translate. The three upper-layer markers (CUX1, CUX2, SATB2) at +0.083, +0.176 and +0.189; RORB (L4) at +0.090; FEZF2 (L5) at +0.168; and one L6 marker at +0.108. The other L6 marker is not significant (r = +0.019, p = 0.25). Mean across the seven is +0.23, against a permuted-π null near zero.

## What this means

The Schaefer-400 parcellation does not separate layers within an area, and each mouse parcel mixes the cortical layers. This test therefore measures the area-level spatial distribution of the layer-marker genes rather than within-area lamination. At that level six of seven markers carry cross-species signal that π routes, at a strength (mean +0.119) comparable to the BICCN cell-type markers and below the region-level tests (~0.4). It does not show that HOMER resolves laminar geometry; the parcellation has no information at that scale.

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


## Areal-type reframe (2026-06-19), `03_areal_type_reframe.py`

Schaefer-400 can't resolve layers, so a laminar test is impossible by construction.
We recast it as the question the parcellation *can* answer: does π preserve cortical
areal type (the supragranular↔infragranular / eulaminate↔agranular axis)? Cortex-only,
fair spin null:
- supragranular − infragranular: r=−0.02, spin p=0.71 (n.s.)
- granular L4 − infragranular: r=+0.19, spin p≤0.004 (the expected exception: cortical granularity is itself the areal hierarchy)
- supragranular − granular: r=−0.04, spin p=0.52 (n.s.)

So the layer-marker *gene* contrasts do not recover areal type through π.

> ⚠️ **This section used to end by saying the cytoarchitectural axis "DOES survive
> when measured structurally (spin p = 0.021/0.010)" and directing the reader to cite
> Fulcher for cross-species cytoarchitecture. Both halves were wrong.** Those p-values
> were hardcoded literals that existed in no output file; the values the analysis

**Conclusion.** Layer *contrasts* do not transfer through π, but layer *markers* do (mean
r = 0.23, 6 of 7 significant), and so do myelin and cytoarchitecture (Fulcher, r = 0.47 each).
What fails is the laminar component specifically: a contrast subtracts one layer from another
and removes the shared areal signal, and it is the areal signal that π carries. Conservation
reaches broad cell classes too (excitatory − inhibitory, r = 0.34, spin p = 0.001). See
`docs/03_results.md` §3.
Log: `outputs/logs/hodge_areal_type_reframe.json`.
