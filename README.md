# HOMER

**Hom**ology **E**stimation across species via **R**egional optimal transport.

A Python package for learning soft cross-species correspondences between
mouse and human brain parcels using **Fused Gromov–Wasserstein optimal
transport** anchored on a small set of curated homologue pairs.

The result is a **soft coupling matrix π** of shape (1864 mouse nodes,
2094 human nodes) that you can query interactively, use as a translation
prior for downstream tasks, or extend with new anchors.

---

## What it does

Given two brain atlases in different species (mouse CCFv3 + human MNI152),
each with parcellated functional connectivity (FC) and structural
connectivity (SC) data, plus a small set of anatomically-curated
**anchor pairs** (here 21 bilateral pairs from the Garin atlas), HOMER
returns a probability distribution over human parcels for every mouse
parcel — its best estimate of the cross-species homologue.

```
mouse parcel ──► [HOMER (FGW + anchors + multi-modal cost)] ──► distribution over human parcels
```

## What works (and what doesn't)

We validated against [Beauchamp et al. 2022 (eLife)](https://elifesciences.org/articles/79418)'s
22 published mouse↔human region pairs. The picture splits cleanly:

| Where | Top-1 vs Beauchamp | Enrichment vs chance |
|---|---:|---:|
| **In supervised regions** (15 pairs, 927 mouse parcels) | 12% | **11.8×** |
| In novel regions with no anchor (4 hippocampal pairs) | 0% | 0× |
| After **adding 4 hippocampal supplementary anchors** | 7-9% on 3/4 pairs | **24.4×** |

Convergent negatives confirm the bottleneck is anchor density, not the
solver: we tested FUGW (different OT formulation) and Knox 2019 leaf-level
SC (richer modality), both retained as comparative additions; neither
moves the headline numbers.

For per-region trust calibration, see [`docs/whats_in_the_box.md`](docs/whats_in_the_box.md).
The TL;DR: trust is regional and explicitly bounded.

## Headline numbers

| Metric | Value | Caveat |
|---|---|---|
| Held-out anchor-candidate ranking accuracy (top-1) | **81%** | Restricted to held-out anchor columns, not full 2094-node space |
| Full-space top-1 (argmax over all 2094 human nodes) | 2.4% | Mean rank of correct anchor = 206/2094 |
| FC translation Pearson r (subject-CV held-out) | **0.32** | (in-sample r = 0.36) |
| Bootstrap argmax stability (40 subject resamples) | **97.8%** | 88% of mouse rows have identical argmax across all 40 |
| z-score vs permuted-anchor null | **+17.8** | Real signal, not solver artefact |
| External validation (Beauchamp 2022, anchored regions) | **11.8× chance** at top-1 | Real cross-species biology where supervised |

Four configs (`fc_only`, `fc_plus_xyz_gw`, `fc_plus_network_mask`,
`fc_plus_SC`) differ by ≤1 of 42 anchors and McNemar-test as
statistically tied. We default to `fc_plus_SC` because SC adds modest
within-network FC translation gains.

## Installation

```bash
git clone <this-repo> homer && cd homer
conda env create -f env.yml && conda activate homer
pip install -e ".[dev]"
pytest -q                   # 117 tests, ~10 s
```

Optional comparative methods:

```bash
pip install fugw torch      # for homer.models.FUGWModel
```

## Quick example

```python
import numpy as np
from homer.data import load_cached
from homer.models import MultimodalFGW

M, _ = load_cached("mouse", cache_dir="outputs/anndata")
H, _ = load_cached("human", cache_dir="outputs/anndata")
costs = np.load("outputs/anndata/full_costs.npz")

model = MultimodalFGW(use_sc=True, sc_weight=0.3, fc_weight=0.7)
model.fit(M, H, Cm_SC=costs["Cm_SC"], Ch_SC=costs["Ch_SC"])

print(model.pi.shape)                              # (1864, 2094)
print(model.evaluate(eval_kind="anchor"))          # {top1, top5, ...}
print(model.evaluate(eval_kind="translation"))     # {pearson_r_overall, ...}

model.save("outputs/coupling/pi_my_run.npy")
```

For a full replication recipe (data prep → fit → evaluate → bootstrap →
artefacts), see [`docs/pipeline.md`](docs/pipeline.md). For interactive
exploration, run any of the four notebooks in `notebooks/`.

## Project layout

```
homer/
├── src/homer/                   # library
│   ├── data/                    # I/O, anchor index, network labels, supplementary anchors
│   ├── costs/                   # within-species + cross-species cost matrices
│   ├── models/                  # 4 sklearn-style FGW model classes (+ FUGW comparative)
│   ├── eval/                    # CV, full-space metrics, FC translation, trust score
│   └── viz/                     # 3D viewer + notebook plots
├── pipeline/                    # numbered, end-to-end replication scripts
│   ├── 00_external/             #   data downloads (Allen, Domhof, Knox)
│   ├── 02_build_anndata.py      #   per-species AnnData
│   ├── 03*_build_costs.py       #   FC + xyz + SC + gene cost matrices
│   ├── 04_solve_production.py   #   fit MultimodalFGW
│   ├── 05*.py                   #   evaluate (anchor CV, FC translation, nulls,
│   │                            #     full-space metrics, Knox vs Allen, Beauchamp,
│   │                            #     trust score)
│   ├── 06_bootstrap.py          #   subject-level bootstrap stability
│   └── 07*.py                   #   comparison table + interactive viewer
├── experiments/                 # research scripts (A/B/C/D, M1, M4, anchor split)
│   └── archive/                 # obsolete stepping-stones
├── tests/                       # pytest, ~10 s on synthetic fixture
├── notebooks/                   # 4 interactive .ipynb walkthroughs
├── docs/                        # see "Documentation" below
├── config/                      # supplementary-anchor YAML configs
└── outputs/                     # all generated artefacts (π, JSONs, figures, viewer)
```

## The four model levels

`homer.models` exposes four sklearn-style classes:

| Class | Use when | Beauchamp top-1 |
|---|---|---|
| `UnsupervisedGW` | Sanity check (no anchors, no xyz) | ~14% |
| `SupervisedFGW` | Anchors + xyz; no SC | 79% |
| **`MultimodalFGW`** | **Production: FC + SC + anchors + xyz** | **81%** |
| `HierarchicalFGW` | Per-network sub-solves (best within-net FC) | 45% LONO, r=0.55 within-net |

All four share `.fit() · .pi · .predict_human_fc() · .evaluate() · .save() · .load()`.
See `notebooks/03_compare_model_levels.ipynb` for side-by-side fits.

Two comparative additions:

| Class | Why | Verdict |
|---|---|---|
| `FUGWModel` | Soft probabilistic π via unbalanced FGW | Better coverage, no anchor-recovery improvement |
| Knox SC variant | Voxel-level SC instead of summary-structure | 1.03× resolution gain, no recovery improvement |

## Documentation

Recommended reading order:

1. **[`docs/whats_in_the_box.md`](docs/whats_in_the_box.md)** — plain-English summary: what the model does, is it generalising, where to trust it. Start here.
2. [`docs/methods_writeup.md`](docs/methods_writeup.md) — paper-style methods note.
3. [`docs/external_validation.md`](docs/external_validation.md) — Beauchamp 2022 validation.
4. [`docs/anchor_expansion_2026-05-06.md`](docs/anchor_expansion_2026-05-06.md) — supplementary anchor experiments (M1 + hippocampal subfields).
5. [`docs/diagnostics_supervised_failures.md`](docs/diagnostics_supervised_failures.md) — why motor + tectum failed despite anchors.
6. [`docs/comparative_methods.md`](docs/comparative_methods.md) — FUGW + Knox SC comparisons.
7. [`docs/results.md`](docs/results.md) — comprehensive comparison table.
8. [`docs/audit_2026-05-06.md`](docs/audit_2026-05-06.md) — end-to-end audit.
9. [`docs/pipeline.md`](docs/pipeline.md) — replication recipe.
10. [`docs/methods.md`](docs/methods.md) — internal methods reference.
11. [`docs/extending.md`](docs/extending.md) — adding new modalities, species, model classes.
12. [`ROADMAP.md`](ROADMAP.md) — full open-issue list.

## Notebooks

Four `.ipynb` files in [`notebooks/`](notebooks/), all using the library API:

- **`01_quickstart.ipynb`** — load production π, click on regions, see top-K partners.
- **`02_explore_results.ipynb`** — comparison table + per-network heatmap + null calibration + bootstrap stability + **trust map** (per-parcel reliability tier).
- **`03_compare_model_levels.ipynb`** — fit all four model levels side-by-side.
- **`04_methodology_walkthrough.ipynb`** — step-by-step FGW solve from raw FC → costs → M → solver → eval.

## Status & limitations

This is research software. Specifically:

- **Validated only against Beauchamp 2022.** Other published correspondences (Mars 2018, Coletta 2020) are roadmap items.
- **Cerebellum is excluded** from our parcellation; 14 of Beauchamp's 36 pairs cannot be evaluated.
- **xyz cost is uniformly misleading for cross-species topology.** The model overcomes this in supervised regions via anchors + FC; in spatially-inverted regions (midbrain, hippocampus without supplementary anchors), it fails. See `docs/diagnostics_supervised_failures.md`.
- **Per-parcel correspondence is a region-level claim**, not a strict 1:1 statement. Mean argmax distance is 25-45 mm even in well-anchored regions.

The model is most usefully framed as:

> An interpolator within the support of the 21 anchor pair_ids in the joint mouse×human FC/SC manifold, with empirical accuracy bounded by the per-region Beauchamp validation.

## Citing

If you use HOMER in published work, please cite:

```
TODO — manuscript in preparation.
```

## License

MIT. See [`LICENSE`](LICENSE).

## Acknowledgements

- Garin et al. for the curated bilateral homologue anchors.
- Beauchamp et al. (2022, eLife) for the published mouse↔human region pairs used for external validation.
- POT (Python Optimal Transport) for the FGW solvers.
- The `fugw` PyPI package (Thual et al. 2022 NeurIPS) for the comparative unbalanced FGW solver.
- Knox et al. (2019) for the voxel-resolved mouse connectome model used in the comparative SC experiment.
