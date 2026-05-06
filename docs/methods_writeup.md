# Cross-species brain region mapping via Fused Gromov–Wasserstein

**Methods note · v0.1**

This document describes the production pipeline that produces a soft cross-species
coupling π between mouse and human brain parcels from resting-state functional
connectivity (FC), structural connectivity (SC), and a small set of
anatomically-curated anchor pairs. It accompanies the open-source
implementation in this repository and is intended to be readable on its own.

---

## 1. Problem and contribution

We have two distinct brain atlases:

- **Garin mouse** atlas: 1864 nodes in CCFv3 voxel space, with
  resting-state FC matrices averaged over 105 mice.
- **Human** atlas: 2094 nodes in MNI152 voxel space, with resting-state
  FC matrices averaged over 113 subjects.

The atlases were built independently in their respective species and do not
share a coordinate frame. The Garin atlas authors identify 42 nodes per
species — 21 *pair-id* bilateral pairs (e.g. left + right primary visual
cortex) — as putative cross-species **homologue anchors**.

We want to learn a (1864 × 2094) **soft coupling π** such that for every
mouse parcel `i`, `π[i, :]` is a probability distribution over human parcels
expressing the model's belief about cross-species correspondence. This is
useful as (a) a region-level cross-species translation table, (b) a vehicle
for translating any mouse-side signal (FC pattern, gene expression, lesion
effect) onto the human atlas, and (c) a hypothesis generator for
follow-on experiments.

**Headline result.** Anchor-supervised semirelaxed Fused Gromov–Wasserstein
on combined FC + SC achieves:
- 81% leave-one-network-out anchor recovery (z = +17.8 vs permuted-anchor null),
- FC translation Pearson r = 0.36 (within-network 0.45),
- 97.6% mean per-cell stability under 40 subject-level bootstrap iterations,
- 4 percentage-point generalisation gap under 5-fold subject CV.

Multiple independent ablations (item A: anchor-relationship M cost;
item B: iterative co-clustering; item C: confidence-weighted FC; item M_gene:
cross-species gene cosine cost) all returned negative or null results,
suggesting we are at or near the information-theoretic ceiling of the
42-anchor supervision signal on this dataset.

---

## 2. Data

### 2.1 Functional connectivity

Mouse and human FC matrices were provided pre-processed by the experimental
collaborator. Each subject contributes a (n_nodes × n_nodes) matrix of Pearson
correlations between resting-state BOLD time series at each pair of parcels.
We work with the across-subjects mean FC throughout; per-cell observation
counts are near-uniform (mouse: exactly 105 everywhere; human: 100–113, with
85% of cells at 113 and all 42 anchor nodes at row coverage 99.8%).

We verified upstream FC quality with three checks (`homer.data.eda`):
- subject-to-subject FC similarity matrix per species (3 outliers flagged
  per species via MAD threshold; not excluded from the production solve);
- left/right hemispheric symmetry per pair-id (median Pearson r = 0.87 in
  human, 0.66 in mouse);
- cross-species anchor-anchor inter-FC Spearman correlation (ρ = 0.38, p < 1e-30
  on 861 pairs) — the signal we are about to amplify with FGW.

### 2.2 Structural connectivity

- **Mouse SC**: Allen Mouse Connectivity Atlas summary-structure-level
  connection matrix (Knox et al. 2019 voxel-resolved variant is parked as
  future work). Aggregated to the 1864 Garin parcels.
- **Human SC**: Domhof et al. (2024) HCP-eNKI publicly-available
  group-averaged streamline matrix at the Schaefer-400 + 17 subcortical
  parcellation, re-aggregated to our 2094-node atlas.

### 2.3 Anchors

The 42 Garin anchors form 21 pair-ids × 2 hemispheres. We hand-grouped the
21 pair-ids into 11 coarse functional networks (auditory, brainstem,
frontal-DMN, frontoparietal, limbic, olfactory, salience, sensorimotor,
subcortical, temporal-DMN, visual) following standard human-brain
network atlases (Yeo, Glasser) and mouse-rat homologue work (Stafford,
Grandjean). The grouping is intentionally coarse — it serves as a
network-level prior, not a finer-grained relabel.

### 2.4 Gene expression (optional modality)

- **Mouse**: Allen ISH grid data via `02b_mouse_genes_direct.py`. We
  achieved 51 of 73 well-known markers, then expanded to 61 of 255 (Allen's
  ~25% 3D-reconstruction rate is the rate-limiting step; documented in
  `ROADMAP.md` items D1–D4).
- **Human**: abagen-derived expression on the Schaefer-400 atlas.
- **Orthologs**: HCOP-derived 1-to-1 ortholog table; ~50 ortholog pairs
  available between the two species' gene sets.

The gene-coexpression modality is exposed as an opt-in flag (`use_gene_gw`,
`M_gene_weight`) but **not** in the production config — the comparison-table
results show it hurts overall CV (60-64% top-1 vs 81% baseline) because it
tanks subcortical recovery (100% → 20%).

---

## 3. The model

### 3.1 Formulation

Given:
- `C_m, C_h` — within-species relational cost matrices (here `1 - r` on FC,
  optionally combined with SC),
- `M` — (n_m, n_h) cross-species feature cost (xyz spatial distance + anchor
  supervision; optional gene/network terms),
- `p` — fixed mouse marginal `1/n_m`,
- α ∈ [0, 1] — FGW mixing weight,
- ε > 0 — entropic regularisation strength,

we solve

$$
\pi^* = \arg\min_{\pi \ge 0, \, \pi \mathbf{1} = p}
  \;\;
  (1-\alpha) \cdot \langle M, \pi\rangle
  \;+\; \alpha \cdot \sum_{i,j,k,l} (C_m[i,k] - C_h[j,l])^2 \, \pi[i,j] \, \pi[k,l]
  \;-\; \varepsilon \cdot H(\pi).
$$

The human-side marginal is **free** (semirelaxed). This matches the data:
the 1864-node mouse atlas is smaller than the 2094-node human atlas, and
many human parcels have no clear mouse counterpart. We don't want to force
the model to spread mass to fake correspondences. The cost is that ~36% of
human nodes end up with `col_mass ≈ 0` ("uncovered") in the production
solve; the benefit is that every mouse row has a sharp, interpretable
distribution over human nodes.

### 3.2 Cost matrices

**Within-species relational (`C_m`, `C_h`)**. Each modality contributes a
symmetric, zero-diagonal, [0, 1]-normalised distance matrix:

- `correlation_distance(FC) = 1 - r`, normalised by max off-diagonal.
- `sc_correlation_distance(SC)`: log1p the heavy-tailed counts (streamline
  density), then Pearson correlation of each node's SC fingerprint, then
  `1 - r`.
- `gene_correlation_distance(expr)`: per-gene + per-node z-score, then
  Pearson, then `1 - r`. NaN-row handling via off-diagonal median.

The production config uses `0.7·FC + 0.3·SC`.

**Cross-species (`M`)**. Three components, summed:

- *xyz spatial prior*: per-species-normalised Euclidean distance between
  (x, y, z) coordinates. Weight 0.5 in production.
- *anchor supervision*: for each visible anchor mouse position `mp` with
  known correct human partner position `hp`, set `M[mp, :] = λ_anchor` (a
  high penalty for any other column) and `M[mp, hp] = 0` (free for the
  correct one). Symmetric handling on the column side. λ_anchor = 1.0,
  large compared to the [0, 1]-normalised feature scale.
- *(optional)* network mismatch penalty, gene cosine cost, anchor-relationship
  cosine cost — all off in production.

### 3.3 Solver

POT's `entropic_semirelaxed_fused_gromov_wasserstein` with a projected
gradient descent inner loop. We use a single uniform-init solve in
production; multistart with 4 random Sinkhorn-projected G0 inits + 1
anchor-warm init was tested (item M1) and converges to within ~1e-6
relative loss across all 6 inits — anchor supervision + xyz make the
objective globally well-identified in practice.

### 3.4 Production hyperparameters

| Parameter            | Value | Rationale                                                              |
|----------------------|-------|------------------------------------------------------------------------|
| α                    | 0.5   | Equal weight to relational + feature terms                             |
| ε                    | 5e-3  | Small → mostly-deterministic π. Larger ε softens π but reduces accuracy. |
| `xyz_weight`         | 0.5   | Spatial prior strong enough to disambiguate, not so strong it swamps FC. |
| `lam_anchor`         | 1.0   | Forbidden-cell penalty, large vs the [0, 1] cost scale.                  |
| `fc_weight`          | 0.7   | Production FC + SC mix.                                                |
| `sc_weight`          | 0.3   | Production FC + SC mix.                                                |
| `cost_normalisation` | "max" | Each cost matrix divided by its max off-diagonal entry.                |
| `max_iter`           | 25    | Solutions converge well within 25 iterations.                          |
| `tol`                | 1e-5  | Loss-change tolerance.                                                 |

These choices are reflected in the `MultimodalFGW` defaults in
`homer.models.multimodal`.

---

## 4. Validation pipeline

We use three independent metrics, each measuring a different aspect of
generalisation.

### 4.1 Held-out anchor cross-validation (LONO)

For each of 11 functional networks: withhold all anchor pair-ids assigned
to that network (so the model receives no anchor supervision in that
region), fit FGW on the remaining 32 visible anchors, then evaluate on the
held-out network's anchors. The model must recover them via the FC + SC +
xyz signal alone.

**Two distinct metrics** that answer different questions:

#### 4.1a Restricted-anchor ranking metric (`homer.data.anchors.held_out_metrics_graded`)

Argmax restricted to the held-out human anchor columns only. Answers:
"among the held-out anchor candidates, did the model rank the correct one
first?" — i.e., a discrimination task on a small candidate set (typically
2-10 candidates depending on which network was held out).

- **restricted-top-1**: argmax over held-out columns hits the correct partner.
- **restricted-top-5**: correct in the top 5 of held-out columns.
- **mean rank** (in held-out columns): 1 = perfect, max = #held-out.
- **mean xyz distance**: per-species-normalised Euclidean distance from
  predicted to correct anchor xyz.

This is what we report as 81% in the headline. **It is NOT per-voxel
mapping accuracy.**

#### 4.1b Full-space recovery metric (`homer.eval.full_space_metrics`)

Argmax taken over ALL 2094 human nodes. Answers: "where in the full
human atlas did the model send the held-out mouse anchor?"

- **full-top-1**: full-space argmax IS the correct anchor.
- **full-top-5**: correct in the global top-5.
- **mean rank in full space**: out of 2094.
- **mean mass on correct anchor**: π[mp, hp_correct] / row sum — the
  actual probability the model assigned to the correct partner regardless
  of whether it was the argmax.
- **frac_argmax_is_anchor**: of the held-out mouse anchors, what fraction
  had a full-space argmax that was *any* anchor (not just the right one).
- **frac_in_neighborhood**: argmax within 5% of brain extent of the
  correct anchor by xyz.

For our production model (`fc_plus_SC`):
| Metric | Restricted (4.1a) | Full-space (4.1b) |
|---|---|---|
| top-1 | 81% | **2.4%** |
| top-5 | 100% | 11.9% |
| mean rank | 1.24 | 206/2094 |
| mean xyz distance | 0.020 | 0.255 |
| argmax is any anchor | — | 4.8% |
| in 5%-of-brain neighborhood | — | 7.1% |

The huge gap (81% vs 2.4%) is because the model's full-space argmax
typically lands on a non-anchor grid node *near* the correct anchor —
e.g. for held-out mouse V1 the argmax goes to grid node `L_971` rather
than the V1 anchor itself, even though they're spatially adjacent.

**Honest summary statement**: the production model reliably *ranks the
correct held-out anchor first among held-out anchor candidates* (81%);
it does not reliably select the correct anchor as the global argmax over
all human nodes (2.4%). If your downstream use depends on per-voxel
correctness rather than candidate ranking, the appropriate number to
quote is the full-space metric.

### 4.2 FC translation quality

Anchor-independent. The coupling π lets us *push* a mouse FC matrix
through to predict the human FC matrix:

$$
\hat{F}_h[j, k] \;=\; \frac{\sum_{i,l} \pi[i, j] \, \pi[l, k] \, F_m[i, l]}
                            {q[j] \cdot q[k]},
\qquad q[j] = \sum_i \pi[i, j].
$$

Pearson correlation of upper-triangle `\hat{F}_h` vs the actual human FC.
Within-network and cross-network breakdowns isolate where the prediction
quality lives.

This is the closest thing to a downstream-task validation we have: the FC
matrix the model is asked to predict was never explicitly used in the solve
(only the 1-D anchor labels were).

### 4.3 Subject-level K-fold CV

For each of K = 5 random 80/20 subject splits per species: build mean FC
on the 80% train subset, fit FGW with full anchor supervision, then
evaluate FC translation on the held-out 20% test subset. Tests
generalisation across cohorts of subjects (orthogonal to anchor
generalisation).

### 4.4 Null distributions

Two principled nulls (`homer.eval.nulls`):
- **Random π**: uniform-sampled doubly-stochastic-ish matrices satisfying
  the mouse marginal. 50 trials per network. Tests whether *any* coherent π
  matters, vs random.
- **Permuted anchor**: shuffle the cross-species anchor correspondence
  before solving FGW. 5 trials per network. Tests whether the *specific*
  anchor pairings drive the result vs "having anchor supervision in
  general".

Z-scores are computed against the per-trial weighted-mean top-1 across the
11 networks.

### 4.5 Bootstrap stability

40 iterations of subject-level bootstrap resampling per species; refit
the production model on each resample; aggregate per-cell mean and std of
π. Stability = 1 − std/std_max.

---

## 5. Results

### 5.1 Headline comparison

Weighted means across 11 LONO folds. FC-translation = production solve
with full anchors. **Production marked bold.** See
`outputs/comparison/comprehensive_table.csv` for the full table.

| Config                              | Top-1 | Top-5 | Pair | Mean rank | xyz_d | FC-r | FC-r within-net | FC-r cross-net |
|-------------------------------------|-------|-------|------|-----------|-------|------|------------------|-----------------|
| Baseline (FC only)                  | 79%   | 100%  | 79%  | 1.26      | 0.021 | 0.36 | 0.45             | 0.20            |
| FC + xyz GW                         | 81%   | 100%  | 81%  | 1.24      | 0.020 | 0.37 | 0.45             | 0.20            |
| FC + network mask                   | 81%   | 100%  | 81%  | 1.24      | 0.020 | 0.38 | 0.49             | 0.17            |
| **FC + SC (production)**            | **81%** | **100%** | **81%** | **1.24** | **0.020** | **0.36** | **0.44** | **0.20** |
| FC + M_gene                         | 60%   | —     | 64%  | —         | —     | —    | —                | —               |
| FC + M_anchor (item A — leak-fixed) | 69%   | 100%  | 69%  | 1.60      | 0.031 | —    | —                | —               |
| Hierarchical (per-network)          | 45%   | 93%   | 67%  | 2.36      | 0.160 | 0.39 | 0.55             | 0.16            |
| Iterative co-clustering (item B)    | 81%   | 100%  | 81%  | 1.24      | 0.020 | —    | —                | (no-op)         |

### 5.2 Null calibration

Each null trial = a per-network top-1 weighted-mean across all 11 networks.

| Null kind         | n trials | Real top-1 | Null mean | Null std | z-score   |
|-------------------|----------|-----------:|----------:|---------:|----------:|
| Random π          | 50       | 81%        | 28%       | 7%       | **+7.5**  |
| Permuted anchor   | 5        | 81%        | 31%       | 3%       | **+17.8** |

The z = +17.8 vs permuted-anchor null is the headline: it tells us the
*specific* mouse↔human anchor pairings drive the result, not just "having
anchor supervision in general".

### 5.3 Subject CV (K=5, 80/20)

| Config       | train r           | test r            | gap             | within-net (test) | cross-net (test) |
|--------------|-------------------|-------------------|-----------------|-------------------|------------------|
| `fc_only`    | 0.360 ± 0.002     | 0.319 ± 0.006     | −0.041 ± 0.007  | 0.420 ± 0.011     | 0.166 ± 0.007    |
| `fc_plus_SC` | 0.357 ± 0.002     | 0.318 ± 0.006     | −0.039 ± 0.008  | 0.417 ± 0.011     | 0.166 ± 0.008    |

The gap is small (~4 pp) and stable across folds; `fc_only` and
`fc_plus_SC` are statistically indistinguishable on this metric.

### 5.4 Bootstrap stability (production)

40 iterations:
- mean per-cell stability: **0.976**
- median: **1.000**
- frac stable above 0.8: **94.9%**
- frac stable above 0.5: **99.2%**

---

## 6. What the project tried that didn't work

Five methodological extensions were tested in dedicated experiments. All
returned negative or null results. Documenting them is part of the
project's contribution: a reader considering the same extensions can read
why they didn't help us before reproducing the dead end.

### 6.1 Anchor-relationship cross-species cost (item A)

For each node, build a vector of FC values to the 42 anchors. Since
anchors are in known cross-species correspondence, those vectors are
directly comparable across species. Cosine distance produces an
(n_m, n_h) cross-species cost matrix.

**Result**: top-1 dropped from 81% to 69% after the leak fix (the first
implementation accidentally used all 42 anchor positions including the
held-out ones at CV time). The information in 32 visible anchors is not
enough to predict the held-out ones better than xyz alone. Visual went
50% → 25%, subcortical 100% → 60%.

### 6.2 Iterative co-clustering / soft anchor expansion (item B)

EM-style: solve FGW once → identify high-confidence non-anchor mouse rows
(row-max concentration > threshold) → add them as soft anchors with low λ
→ re-solve → iterate.

**Result**: π changes by < 1e-5 across iterations. The first-pass π is
already 97.6% concentrated under our production ε = 5e-3, so re-injecting
the model's own confident predictions as constraints is a no-op. The
bottleneck for held-out anchor recovery is information available *to the
held-out row's GW + xyz signal*, not lack of self-confidence about the
rest of the map.

### 6.3 Confidence-weighted FC via `n_obs` (item C)

Bayesian-flavored shrinkage: pull each FC cell `r[i, j]` toward 0 in
proportion to its coverage deficit (`n_obs / n_max`).

**Result**: structural no-op. Mouse `n_obs` is uniform → mathematically
zero change. Human: 99.97% Pearson correlation between unweighted and
shrunk cost matrices. The colleague's preprocessing already removed the
coverage variation that this experiment was designed to exploit.

### 6.4 Cross-species gene cost (M_gene)

Cosine distance over ortholog-aligned gene expression vectors as a term
in M.

**Result**: hurts overall CV. Drops top-1 from 81% to 60-64% across all
M_gene-bearing configs because subcortical anchor recovery collapses
(100% → 20%). Selective M_gene (mask out cells where either species lacks
ortholog data) helps marginally but not enough.

### 6.5 Hierarchical / per-network FGW (item M4)

Solve a separate FGW within each functional network's sub-block of nodes.

**Result**: a *trade-off*, not a strict win. Best within-network FC
translation (r = 0.55 vs 0.45 flat) but worst leave-one-network-out CV
(45% vs 81%), because the held-out network has zero anchor supervision
in its own sub-block. Coverage halved (787 vs 1450 covered human nodes)
because the block-diagonal structure forbids cross-network mass.

A complementary tool, useful when full anchor supervision is available
and within-network fidelity matters more than cross-network coverage.

---

## 7. Interpretation

Five lines of evidence converge on the conclusion that the production
model is at or near the **information-theoretic ceiling** for the
42-anchor supervision signal on this dataset:

1. **Multiple methodology variants converge to 79–81%**. None of the
   extensions in §6 moved the headline number.
2. **The hard regions are the same across all configs**: brainstem,
   subcortical, salience, sensorimotor, visual all bottleneck at 25–60%
   regardless of modality choice. The other 6 networks are at 100%.
3. **z = +17.8 vs permuted-anchor null** says the supervision is genuinely
   informative, not just "any 42 constraints help".
4. **Bootstrap stability is 97.6%** — the solution isn't fragile to
   subject sampling.
5. **Subject-level generalisation gap is small** (~4 pp) — the model
   doesn't overfit to specific subjects.

**Three plausible paths to break past 81%** (in increasing order of
engineering cost):

1. **More anchors**. Garin's 42 is the rate-limiting supervision signal.
   Adding even 10 more high-confidence cross-species pairs (from
   external atlases or expert curation) should move every headline metric.
2. **Higher-resolution per-node modality data**. The Allen
   *summary-structure* SC we use treats all 47 visual-cortex parcels
   identically (one SC vector for all visual-cortex nodes); Knox 2019's
   voxel-resolved mouse SC model would give per-node SC fingerprints and
   should help the within-visual disambiguation that flat FGW currently
   loses. Roadmap item M5.
3. **External validation showing some held-out anchors are genuinely
   ambiguous**. If the published cross-species literature also disagrees
   on the V1 vs V2 boundary in the mouse, then our 50% top-1 for the
   visual network is closer to ceiling than to our error. Roadmap item E5.

---

## 8. Reproducibility

The complete pipeline is deterministic given the input data. From a clean
checkout:

```bash
conda env create -f env.yml && conda activate homer
pip install -e ".[dev]"

# data prep (~3 hours, network-bound)
python pipeline/00_external/...        # one-off external downloads
python pipeline/02_build_anndata.py     # ~1 min
python pipeline/03_build_costs.py        # ~2 min

# solve + evaluate
python pipeline/04_solve_production.py   # ~15 s
python pipeline/05_evaluate.py           # ~15 min
python pipeline/06_bootstrap.py --n-iter 40   # ~10 min

# build artefacts
python pipeline/07_build_artefacts.py
python pipeline/07b_build_viewer.py
```

Programmatic API:

```python
from homer.data import load_cached
from homer.models import MultimodalFGW
import numpy as np

M, _ = load_cached("mouse", cache_dir="outputs/anndata")
H, _ = load_cached("human", cache_dir="outputs/anndata")
costs = np.load("outputs/anndata/full_costs.npz")

model = MultimodalFGW(use_sc=True, sc_weight=0.3, fc_weight=0.7,
                      epsilon=5e-3, xyz_weight=0.5)
model.fit(M, H, Cm_SC=costs["Cm_SC"], Ch_SC=costs["Ch_SC"])
print(model.evaluate(eval_kind="anchor"))     # 100% under full supervision
print(model.evaluate(eval_kind="translation")) # r ≈ 0.36
model.save("outputs/coupling/pi_my_run.npy")
```

Notebooks `notebooks/01_quickstart.ipynb` through `04_methodology_walkthrough.ipynb`
exercise the API end-to-end. The test suite (`pytest tests/`) runs in ~10
seconds and covers the full library.

---

## 9. What this work doesn't address

To set expectations clearly:

- **Single best mapping per node**. We deliver π as a soft probability
  distribution but the production setting (small ε) makes π ~97% one-hot
  in practice. For genuinely soft mappings (uncertainty propagation,
  Bayesian downstream), a larger-ε re-solve is needed. This costs ~2 pp
  on held-out anchor accuracy.
- **Cellular-level correspondences**. We work at the parcel level (~1864
  mouse / 2094 human nodes); single-cell or single-neuron alignment is a
  different problem.
- **Functional task-related correspondence**. Resting-state FC reflects
  intrinsic network organisation; task-related correspondences may differ.
- **Causal claims**. π is a similarity-based correspondence; it does not
  claim that two mapped regions are evolutionarily homologous in the
  developmental-biology sense.

---

## 10. References

- Vayer, Chapel, Flamary, Tavenard, Courty (2019).
  *Optimal Transport for structured data with application on graphs.*
  ICML.
- Sejourne, Vialard, Peyré (2021). *The Unbalanced Gromov-Wasserstein
  Distance.* NeurIPS.
- Flamary et al. (2021). *POT: Python Optimal Transport.* JMLR.
- Garin et al. (2021). *Cross-species translation of fMRI: a dual
  whole-brain mapping framework for resting-state networks.* MICCAI.
- Knox et al. (2019). *High-resolution data-driven model of the mouse
  connectome.* Network Neuroscience.
- Domhof et al. (2024). *Open-access HCP+eNKI structural connectivity dataset.*
  EBRAINS.
- Markello, Hansen, Liu et al. (2021). *abagen: a toolbox for the Allen
  brain atlas genetics data.* eLife.
- Beauchamp et al. (2022). *Whole-brain comparison of rodent and human
  brains using spatial transcriptomics.* eLife. *(planned for E5
  external validation.)*
- Coletta et al. (2020). *Network structure of the mouse brain connectome
  with voxel resolution.* Science Advances.
- Stafford et al. (2014). *Large-scale topology and the default-mode
  network in the mouse connectome.* PNAS.

---

## Appendix A — Repository layout

```
src/homer/                # the library
├── data/                  # io · anchors · networks · eda
├── costs/                 # relational · crossspecies · normalisation
├── models/                # base + 4 model classes + _solver
├── eval/                  # anchor_cv · subject_cv · translation · nulls · bootstrap
└── viz/                   # viewer · notebook · reports

pipeline/                  # numbered, end-to-end replication
├── 00_external/           # data downloads
├── 02_build_anndata.py
├── 03_build_costs.py
├── 04_solve_production.py
├── 05_evaluate.py + 05a/05b/05c sub-steps
├── 06_bootstrap.py
└── 07_build_artefacts.py + 07b_build_viewer.py

experiments/               # research scripts (A, B, C, D, M1, M4 + archive)
notebooks/                 # 4 .ipynb files exercising the public API
docs/                      # pipeline.md · methods.md · results.md · extending.md · this file
tests/                     # 105 pytest cases on synthetic fixture, ~10 s
outputs/                   # generated artefacts (anndata, coupling, eval logs, figures, viewer)
```

## Appendix B — Output JSONs and what they contain

| File                                          | Producer                       | What                                                         |
|-----------------------------------------------|--------------------------------|--------------------------------------------------------------|
| `outputs/anndata/{mouse,human}.h5ad`          | `02_build_anndata.py`          | Per-species AnnData with mean FC                             |
| `outputs/anndata/full_costs.npz`              | `03_build_costs.py`            | All cost matrices (FC, SC, gene, xyz, M_gene, M_anchor)      |
| `outputs/coupling/pi_*.npy`                   | `04_solve_production.py`       | Production coupling + alternates                             |
| `outputs/coupling/bootstrap_aggregate.npz`    | `06_bootstrap.py`              | Per-cell mean + std + stability                              |
| `outputs/logs/multimodal_cv.json`             | `05a_anchor_cv.py`             | LONO CV per (config, network)                                |
| `outputs/logs/fc_translation.json`            | `05b_fc_translation.py`        | FC-translation r per config; subject-CV sub-key              |
| `outputs/logs/null_distributions.json`        | `05c_null_distributions.py`    | Random π + permuted-anchor trial-level results               |
| `outputs/logs/bootstrap_summary_<config>.json` | `06_bootstrap.py`              | Per-config bootstrap stability summary (`fc_plus_SC` is production) |
| `outputs/comparison/*.csv,*.md`               | `07_build_artefacts.py`        | Wide + long comparison tables + markdown summary             |
| `outputs/figures/{13,14}_*.png`               | `07_build_artefacts.py`        | 4-panel comparison + per-network heatmap                     |
| `outputs/viewer/index.html`                   | `07b_build_viewer.py`          | Self-contained interactive 3D viewer                         |
