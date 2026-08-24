# Cross-species resting-state network correspondence

This directory evaluates whether network-level organisation is retained after translation, with reference to the mouse resting-state networks described by [Coletta et al. (2020)](https://doi.org/10.1126/sciadv.abb7187).

The primary analysis labels the Garin homology classes by canonical rodent resting-state system and assigns remaining mouse parcels by their nearest Garin anchor. These mouse network labels therefore reuse OTTER's anatomical supervision and are an **internal consistency test**, not an independent decomposition of mouse resting-state data. The script also includes exploratory ICA and spatial-coherence summaries.

Run from the repository root:

```bash
PYTHONPATH=src python experiments/coletta_2020_cross_species_rsn/01_correspondence_validation.py
PYTHONPATH=src python experiments/coletta_2020_cross_species_rsn/02_plot.py
```

Outputs:

- `outputs/logs/coletta_2020_cross_species_rsn.json`
- `outputs/figures/coletta_2020_cross_species_rsn.png`
