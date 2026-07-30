# HOMER notebooks

Eight notebooks: two that introduce the coupling, and one per figure of the paper. Each figure
notebook recomputes that figure's results from the data and compares them against the numbers
printed in the manuscript, rather than reading them from a results file.

```
python scripts/fetch_data.py      # coupling, parcel tables, reference volumes, external maps
```

Nothing under `outputs/logs/` is required for the derivations.

## Start here

| notebook | what it covers |
|---|---|
| [`01_quickstart.ipynb`](01_quickstart.ipynb) | Load π, query it, read a human distribution for a mouse region. |
| [`02_methodology.ipynb`](02_methodology.ipynb) | How π is fitted, and how ε and the spatial weight are selected. |

## One per figure

| notebook | figure | what it derives |
|---|---|---|
| [`fig1_coupling.ipynb`](fig1_coupling.ipynb) | 1 | Concentration, mass on the homology diagonal, topographic fidelity. |
| [`fig2_what_carries_homology.ipynb`](fig2_what_carries_homology.ipynb) | 2 | Beauchamp scoring, live. The ablation and leave-one-out arms are re-fits. |
| [`fig3_what_transfers.ipynb`](fig3_what_transfers.ipynb) | 3 | Myelin translation from the raw maps; network correspondence. |
| [`fig4_vs_transbrain.ipynb`](fig4_vs_transbrain.ipynb) | 4 | Every benchmark score, recomputed from the per-region distributions. |
| [`fig5_coverage.ipynb`](fig5_coverage.ipynb) | 5 | Reconstruction-coverage from π and the FC matrices; no stored statistics. |
| [`fig6_disease.ipynb`](fig6_disease.ipynb) | 6 | The translation and its 1,000-permutation null. |

## The organising claim

π appears to carry areal position on the cortical hierarchy, along with the properties that vary
across that axis. Figure 3 establishes this, microstructure included, which translates at r =
0.47, and marks the boundary: properties varying through the cortical depth do not come across.
Figure 4 compares that against a transcriptomic translator. Figure 5 asks where the mouse cannot
rebuild human connectivity at all, and finds a network-shaped territory tracking cortical
expansion. Figure 6 turns the coupling into a prediction about a human brain.

## Three conventions

`load_pi()` decides which coupling is canonical, so avoid hardcoding a filename. The July 2026
re-analysis was reported clean once before anyone noticed that several scripts had been re-run
without being repointed; a re-run says nothing about which input was used. `pi_provenance()`
returns the file and its sha256, which is the thing to verify against.

`verified_log()` declines a log whose coupling does not match. A filename ending `_canonical` is
not evidence of much. Where a notebook has to read a re-fit result it checks the recorded sha
first, and says so when a log carries no provenance at all.

Match the definition before doubting a number. Most near-misses in the audit were not wrong
numbers but two right numbers answering different questions, sitting side by side:

- parcel-weighted held-out AUROC (0.90 → 0.73) against unweighted (0.93 → 0.67);
- round-trip r at region level over the matched 52 regions (0.97) against raw parcels (0.91);
- the ContB coverage gap on whole cortex (−0.63) against the molecular-control subset (−0.69).

The script that builds a panel owns that panel's numbers. If a value looks wrong, find that
script before recomputing it another way.

## What cannot be derived in a cell

An ablation study re-fits the model once per condition, and leave-one-region-out re-fits it 41
times. Those are hours of compute rather than derivations. Figures 2 and 4 name the producing
script and expose a `RUN_REFITS` flag; with it off, the logs are provenance-checked before use.

Figure 4's TransBrain comparison also needs the third-party `transbrain` package. Without it the
scores still recompute from cached distributions, though the distributions themselves cannot be
rebuilt. Figures 1, 3, 5 and 6 need nothing but the fetched data.

## Where the rest went

Per-dataset exploration notebooks and the executed copies under `.scratch/` were removed in the
2026-07 cleanup, since several contained claims this rebuild exists to correct. The analyses that
survived into the paper are in `experiments/`, and each published panel is built by one script
under `manuscript/figures/figN/`.
