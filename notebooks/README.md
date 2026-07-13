# HOMER notebooks

Nine notebooks. Six of them correspond one-to-one with the six figures of the paper, and each
**produces that figure's results and its panels**, so the notebook is the place to look when you want
to know where a number came from.

| notebook | what it establishes |
|---|---|
| [`01_quickstart.ipynb`](01_quickstart.ipynb) | Query π. Get a human distribution for a mouse region, with its confidence grade. |
| [`02_methodology.ipynb`](02_methodology.ipynb) | How the coupling is built: data → cost matrices → the semi-relaxed FGW solve. |
| [`fig1_coupling.ipynb`](fig1_coupling.ipynb) | **Fig 1 + ED1.** π is sharp, homology-respecting, topographically faithful and confidence-graded. |
| [`fig2_what_carries_homology.ipynb`](fig2_what_carries_homology.ipynb) | **Fig 2 + ED2.** Connectivity and space carry *which region*; curation carries *which parcel*. |
| [`fig3_what_transfers.ipynb`](fig3_what_transfers.ipynb) | **Fig 3 + ED3.** Connectional organisation transfers through π; microstructure does not. |
| [`fig4_vs_transbrain.ipynb`](fig4_vs_transbrain.ipynb) | **Fig 4 + ED4.** Each method wins on the modality it encodes. |
| [`fig5_coverage.ipynb`](fig5_coverage.ipynb) | **Fig 5 + ED5.** Where π has no support, the deficit is connectional rather than molecular. |
| [`fig6_disease.ipynb`](fig6_disease.ipynb) | **Fig 6 + ED6.** Which component of a human disorder a mouse can reach. |
| [`discussion_pagani.ipynb`](discussion_pagani.ipynb) | The Discussion's worked application: re-subtyping autism mouse models. |

`archive/` holds the previous notebook set, kept for provenance. Nothing in it should be run: several
of those notebooks contain the errors this rebuild exists to correct.

## The organising principle

**π is a connectional correspondence: connectional organisation transfers through it; microstructure
does not.** Figs 3 to 6 are the tests of that sentence, and it sets the order of the notebooks. Fig 3
establishes it. Fig 4 applies it to a competing method. Fig 5 pushes it to the limit by asking what
happens where π has *no* mass. Fig 6 turns that limit into a prediction about human disease.

## How the notebooks are written

Each notebook is close to standalone. Between them they import only four things from the package
(`load_pi`, `load_cached`, `spin_null`, `coarse_region`) and inline everything else, so a notebook can
be read top-to-bottom without chasing helpers through `src/`.

The exception is the handful of experiments that re-fit the FGW model dozens of times: the 41-unit
leave-one-region-out and the 125-cell weight scan. Those are hours of compute, so the notebook reads
their persisted results and names the script that produced them.

**Two rules run through every notebook.** Both come out of a July 2026 audit that found a dispiriting
number of errors, and both exist to keep those errors out.

### 1. Never type a statistic into prose or a figure title; read it from a JSON

Every headline number is either read from the canonical log that `manuscript/results_section.md` cites,
or **recomputed in the notebook and asserted against that log**. If a notebook's arithmetic drifts from
the manuscript, it fails loudly instead of quietly disagreeing.

The concern is not hypothetical. Two spin p-values in §3 (`0.021`, `0.010`) were **hardcoded literals**
in a figure script and existed in no output file. The values the analysis produces are `0.11` and
`0.10`, which reverses the conclusion.

`homer/tools/check_manuscript_numbers.py` enforces the same rule across the manuscript: every numeral in
`results_section.md` must match a value in a JSON *that its own section cites*, at the written precision.

### 2. Match the definition before you change a number

Most of the audit's near-misses were not wrong numbers. They were **two right numbers that answer
different questions**, placed side by side. In this repo alone:

- `centroid_disp_mm` (17 mm) versus `expected_disp_mm` (34 mm): both real, a factor of two apart.
- Parcel-count-**weighted** held-out AUROC (0.73) versus unweighted (0.72).
- Coverage as a mass-normalised **mean** versus a **sum**: the mean gives ρ = +0.64 and the sum gives
  ρ = +0.05, which is the difference between a result and a null.

**The script that builds a panel owns that panel's numbers.** If a value looks wrong, find the script
that produced it before recomputing it a different way.

## What this rebuild corrects

The old notebooks did not merely go stale; several taught the wrong thing.

- **`07_margulies_huntenburg_gradient`** took the **first** non-trivial eigenvector of the FC Laplacian as
  the principal gradient. In this data that eigenvector is an **anterior–posterior spatial axis**; the
  unimodal→transmodal hierarchy is the **second** component. Routing an A–P spatial axis and then testing
  it against a *spatial-autocorrelation-preserving* null is close to tautological, and it produced a
  confident false negative ("the gradient does not translate") that the paper believed for months. The
  gradient does translate: |r| = 0.54, spin p = 0.004. `fig3_what_transfers.ipynb` **selects** the
  component against an external reference and never hard-codes an index.
- **`11_enigma_cross_disorder`** summed coverage instead of taking the mass-normalised mean, and reported
  a null.
- **`14_transbrain_2025_benchmark`** scored HOMER's round-trip on 52 mouse regions and TransBrain's on 68,
  16 of them mean-filled. The two numbers were not comparable.
- **`05_pagani_2026_validation`** had the hyper/hypo subtype labels **inverted**.

Every one of those produced clean, coherent, publishable-looking output. Plausible output is no evidence
that an analysis is correct, which is why the two rules above are in place.

## Running them

The notebooks need the data bundle (Zenodo, DOI 10.5281/zenodo.20746024). Fetch it first:

```bash
cd homer && python scripts/fetch_data.py
```

Then run any notebook from this directory. `load_pi()` defaults to
`pi_fc_plus_SC_with_all_packs.npy`, the recommended coupling. It does not default to
`pi_fc_plus_SC.npy`, which is a different matrix and gives different answers.

Notebook 01 uses `ipywidgets` for interactivity:

```bash
pip install ipywidgets
```
