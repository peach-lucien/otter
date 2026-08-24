# Cell-class marker-expression transfer

These analyses route mouse marker-expression maps through the released coupling and compare them with human Allen Human Brain Atlas maps. Class definitions follow cross-species cell-type atlases from [Yao et al. (2023)](https://doi.org/10.1038/s41586-023-06812-z) and [Siletti et al. (2023)](https://doi.org/10.1126/science.add7046).

The maps are averages or contrasts derived from curated marker-gene panels. They are **marker-expression proxies, not measurements of cell abundance or composition**.

## Primary analyses

| Script | Purpose | Output |
|---|---|---|
| `03_contrast_reframe.py` | Cell-class contrasts and dopaminergic hotspot | `outputs/logs/biccn_contrast_reframe.json` |
| `05_composition_from_markers.py` | Five cell-class marker-expression maps | `outputs/logs/biccn_composition_from_markers.json` |
| `06_cortex_restricted.py` | Cortex-restricted sensitivity analysis | `outputs/logs/biccn_cortex_restricted.json` |
| `07_donor_tissue_restricted.py` | Restriction to human parcels containing donor tissue, with translation-spin tests | `outputs/logs/gene_maps_donor_tissue_restricted.json` |

Run from the repository root:

```bash
PYTHONPATH=src python experiments/biccn_2023_cell_types/03_contrast_reframe.py
PYTHONPATH=src python experiments/biccn_2023_cell_types/05_composition_from_markers.py
PYTHONPATH=src python experiments/biccn_2023_cell_types/06_cortex_restricted.py
PYTHONPATH=src python experiments/biccn_2023_cell_types/07_donor_tissue_restricted.py
```

`07_donor_tissue_restricted.py` also evaluates laminar marker contrasts. It requires the raw, unimputed expression matrices in `data_external/`.

## Additional utilities

- `01_cell_type_validation.py` and `02_plot.py` provide per-gene summaries.
- `00_fetch_abundance.py` and `04_abundance_composition.py` support optional analyses using external cell-atlas abundance data.

The analyses use Allen mouse ISH energy volumes and human expression processed through abagen. Donor-restricted results should be used when assessing expression coverage.
