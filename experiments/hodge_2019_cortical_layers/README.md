# Hodge 2019 cortical layer-marker expression

This experiment tests whether mouse cortical layer-marker expression patterns
retain their broad areal organisation after translation through OTTER. Marker
definitions follow [Hodge et al. (2019)](https://doi.org/10.1038/s41586-019-1506-7).

Mouse expression is taken from Allen in situ hybridisation volumes and human
expression from the Allen Human Brain Atlas. The comparison is cross-modal and
was not used to fit the coupling. Because the parcellations do not resolve
layers within a cortical area, the analysis concerns the areal distribution of
layer-marker expression, not laminar geometry.

## Scripts

| File | Purpose |
|---|---|
| `01_layer_marker_validation.py` | Translates individual marker maps and evaluates correspondence with the human maps. |
| `02_layer_marker_refined.py` | Evaluates cortex-restricted marker groups and contrasts. |
| `03_areal_type_reframe.py` | Tests three prespecified marker contrasts using the cortical translation-spin null. |
| `04_marker_like_for_like.py` | Rescores individual markers with the same cortical mask and null used for the contrasts. |
| `03_plot.py` | Plots the individual-marker and grouped-marker results. |

## Inputs and outputs

The scripts use the canonical coupling, cached mouse and human parcellations,
and the gene-expression matrices and gene lists in `data_external/`. These
inputs are included in the Zenodo reproduce bundle described in the repository
README.

Outputs are written to:

- `outputs/logs/hodge_2019_layer_markers.json`
- `outputs/logs/hodge_2019_layer_markers_refined.json`
- `outputs/logs/hodge_areal_type_reframe.json`
- `outputs/logs/hodge_markers_like_for_like.json`
- `outputs/figures/hodge_2019_layer_markers.png`

## Run

From the repository root:

```bash
PYTHONPATH=src python experiments/hodge_2019_cortical_layers/01_layer_marker_validation.py
PYTHONPATH=src python experiments/hodge_2019_cortical_layers/02_layer_marker_refined.py
PYTHONPATH=src python experiments/hodge_2019_cortical_layers/03_areal_type_reframe.py
PYTHONPATH=src python experiments/hodge_2019_cortical_layers/04_marker_like_for_like.py
PYTHONPATH=src python experiments/hodge_2019_cortical_layers/03_plot.py
```
