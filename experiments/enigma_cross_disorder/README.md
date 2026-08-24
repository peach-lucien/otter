# Exploratory cross-disorder analyses

This directory contains optional analyses that route disorder-associated mouse gene sets to human space and compare the resulting maps across disorders and with ENIGMA cortical-thickness maps.

## Scripts

| Script | Purpose |
|---|---|
| `01_per_disorder_prediction.py` | Generate parcel-level predictions and their cross-disorder correlation matrix |
| `02_plot_phase1.py` | Plot the prediction summaries |
| `03_enigma_comparison.py` | Compare predictions with ENIGMA Desikan-Killiany cortical-thickness effects |
| `04_disorder_unique.py` | Repeat routing with pairwise non-overlapping gene sets |
| `05_transdiagnostic.py` | Compare the mean prediction with transdiagnostic ENIGMA maps using a spatial null |

Scripts 03 and 05 require ENIGMA Toolbox summary statistics in `data_external/enigma/`. See their module documentation for the expected filenames and columns.

Run from the repository root:

```bash
PYTHONPATH=src python experiments/enigma_cross_disorder/01_per_disorder_prediction.py
PYTHONPATH=src python experiments/enigma_cross_disorder/02_plot_phase1.py
PYTHONPATH=src python experiments/enigma_cross_disorder/03_enigma_comparison.py
PYTHONPATH=src python experiments/enigma_cross_disorder/04_disorder_unique.py
PYTHONPATH=src python experiments/enigma_cross_disorder/05_transdiagnostic.py
```

Outputs are written to `outputs/logs/`, `outputs/figures/` and `outputs/coupling/per_disorder_predictions.npz`.
