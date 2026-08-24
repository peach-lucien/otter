# Rodent medial-frontal correspondence

This analysis compares OTTER's routing of mouse medial frontal cortex with the
cross-species connectivity findings of
[Schaeffer et al. (2020)](https://doi.org/10.1073/pnas.2003181117). That study
reported that rodent medial frontal connectivity resembles primate medial
frontal and premotor territories more closely than dorsolateral prefrontal
cortex.

Mouse ACAd, ACAv, PL and ILA parcels are routed through the coupling and their
mass is summarized over bilateral human medial prefrontal, mid-cingulate,
premotor and dorsolateral prefrontal regions. A row-permutation null evaluates
regional enrichment. The analysis also reports a sensitivity comparison across
the canonical coupling and predefined alternative anchor configurations.

Schaeffer et al. studied rat, marmoset and human data, whereas OTTER maps mouse
to human. This analysis therefore tests consistency with the published
directional finding; it is not a reanalysis of the original connectivity data
or a direct species-matched benchmark.

## Files

| File | Purpose |
|---|---|
| `01_mfc_divergence.py` | Routes mouse medial-frontal parcels, computes regional mass and the permutation null, and compares coupling configurations. |
| `02_plot.py` | Plots the regional routing and coupling comparison. |

## Inputs and outputs

The analysis requires cached human parcellation data, mouse structure metadata,
the canonical coupling and the alternative couplings named in
`01_mfc_divergence.py`. These data are supplied by the Zenodo reproduce bundle.

Outputs are written to:

- `outputs/logs/schaeffer_2020_mfc_divergence.json`
- `outputs/figures/schaeffer_2020_mfc_divergence.png`

## Run

From the repository root:

```bash
PYTHONPATH=src python experiments/schaeffer_2020_mfc_divergence/01_mfc_divergence.py
PYTHONPATH=src python experiments/schaeffer_2020_mfc_divergence/02_plot.py
```
