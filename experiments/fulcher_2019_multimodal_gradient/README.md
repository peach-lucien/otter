# Transfer of cortical hierarchy maps

This analysis routes two mouse cortical maps from [Fulcher et al. (2019)](https://doi.org/10.1073/pnas.1814144116) through the released coupling:

- mouse T1w:T2w, compared like-for-like with the human HCP T1w:T2w map;
- mouse cytoarchitectural type, compared with human T1w:T2w as a proxy for their shared cortical hierarchy.

Neither mouse map is used to fit the coupling. The cytoarchitecture comparison is cross-modal and should not be described as recovery of a like-for-like human counterpart. Spatial inference for both transfers uses the translation-spin null.

## Files

| File | Purpose |
|---|---|
| `01_gradient_validation.py` | Route both maps and calculate Schaefer-400 summaries |
| `02_plot.py` | Generate the three-panel summary figure |

Input provenance is documented in `data_external/fulcher_2019_gradients/SOURCES.md`.

Run from the repository root:

```bash
PYTHONPATH=src python experiments/margulies_2016_principal_gradient/01_gradient_validation.py
PYTHONPATH=src python experiments/fulcher_2019_multimodal_gradient/01_gradient_validation.py
PYTHONPATH=src python experiments/fulcher_2019_multimodal_gradient/02_plot.py
```

Outputs:

- `outputs/logs/fulcher_2019_gradient.json`
- `outputs/figures/fulcher_2019_multimodal_gradient.png`
