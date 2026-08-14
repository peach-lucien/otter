# OTTER notebooks

Eight notebooks: two that introduce the coupling, and six that reproduce the project's analyses.
Each of the six recomputes what can be derived in a cell and compares it against the values this
project reports. Results that need a model re-fit or a third-party package are read from
`outputs/logs/` after their coupling provenance is checked.

```
python scripts/fetch_data.py      # coupling, parcel tables, reference volumes, external maps
```

`03_coupling` and `07_coverage` derive every reported value without reading `outputs/logs/`.

## Introductory notebooks

| notebook | what it covers |
|---|---|
| [`01_quickstart.ipynb`](01_quickstart.ipynb) | π, a single-region query, and the human distribution for a mouse region. |
| [`02_methodology.ipynb`](02_methodology.ipynb) | How π is fitted, and how ε and the spatial weight are selected. |

## The six analyses

| notebook | what it derives |
|---|---|
| [`03_coupling.ipynb`](03_coupling.ipynb) | Concentration, mass on the homology diagonal, topographic accuracy. |
| [`04_cost_terms_and_supervision.ipynb`](04_cost_terms_and_supervision.ipynb) | Beauchamp scoring, recomputed here. The ablation and leave-one-out arms are re-fits. |
| [`05_map_transfer.ipynb`](05_map_transfer.ipynb) | Myelin translation from the raw maps; network correspondence. |
| [`06_vs_transbrain.ipynb`](06_vs_transbrain.ipynb) | Every benchmark score, recomputed from the per-region distributions. |
| [`07_coverage.ipynb`](07_coverage.ipynb) | Reconstruction-coverage from π and the FC matrices; no stored statistics. |
| [`08_disease.ipynb`](08_disease.ipynb) | The translation and its 1,000-permutation null. |

## The organising claim

π appears to carry areal position on the cortical hierarchy, along with the properties that vary
across that axis. `05_map_transfer` establishes this, microstructure included, which translates
at r = 0.47, and marks the boundary; properties varying through the cortical depth do not come
across. `06_vs_transbrain` compares that against a transcriptomic translator. `07_coverage`
locates where the mouse cannot rebuild human connectivity, and finds a network-shaped territory
tracking cortical expansion. `08_disease` turns the coupling into a prediction about a human
brain.

## Conventions

`load_pi()` decides which coupling is canonical, so a filename should not be hardcoded. A re-run
says nothing about which input was used. `pi_provenance()` returns the file and its sha256, which
is the value to verify against.

`verified_log()` declines a log whose coupling does not match. Where a notebook has to read a
re-fit result it checks the recorded sha first, and reports the case of a log that carries no
provenance.

Several quantities are reported under two definitions, and the two values answer different
questions:

- parcel-weighted held-out AUROC (0.90 → 0.73) against unweighted (0.93 → 0.67);
- round-trip r at region level over the matched 52 regions (0.97) against raw parcels (0.91);
- the ContB coverage gap on whole cortex (−0.63) against the molecular-control subset (−0.69).

## Results that cannot be derived in a cell

An ablation study re-fits the model once per condition, and leave-one-region-out re-fits it 41
times. Those are hours of compute rather than derivations. `04_cost_terms_and_supervision` and
`06_vs_transbrain` name the producing script and expose a `RUN_REFITS` flag; with it off, the
logs are provenance-checked before use.

`06_vs_transbrain`'s TransBrain comparison also needs the third-party `transbrain` package.
Without it the scores still recompute from cached distributions, though the distributions
themselves cannot be rebuilt. The other four need nothing but the fetched data.

## Material not distributed

Per-dataset exploration notebooks and their executed copies are not distributed. The analyses are
in `experiments/`. The figure-rendering scripts are not distributed with this repository.
