# Methods. Cross-species brain region mapping via Fused Gromov–Wasserstein

## Problem

We have two distinct brain atlases, the **Garin mouse** atlas (1864 nodes
in CCFv3 voxel space, 105 subjects) and a **human** atlas (2094 nodes in
MNI152 voxel space, 113 subjects), and resting-state functional connectivity
(FC) matrices for each. The 42 **Garin anchors** (21 pair_ids × 2 hemispheres)
are putative cross-species homologues.

We want a (1864 × 2094) **soft coupling π** that, for every mouse parcel, gives
a probability distribution over human parcels indicating cross-species
correspondence.

## Why optimal transport

Mapping atlases of different sizes and modalities is a classic OT setup:
- The two spaces don't share coordinates (mouse CCFv3 ≠ human MNI152).
- We have *intrinsic structure* on each side (FC matrix = pairwise relational
  information) but no node-to-node correspondence outside the 42 anchors.
- We want a probability distribution, not a hard one-to-one map, because
  cross-species mapping is uncertain in many regions.

**Gromov-Wasserstein** matches *intrinsic distance matrices* (FC-derived costs
in our case). **Fused** GW adds a cross-species feature cost matrix M
(spatial distance, gene similarity, etc.). We use the **semirelaxed** variant
because the mouse marginal is a fixed uniform distribution but the human
marginal can float (the model is allowed to leave human nodes uncovered if no
mouse parcel naturally maps there).

## Formulation

Given:
- `C_m, C_h`, within-species relational cost matrices (here `1 - FC`, optionally
  combined with structural-connectivity costs)
- `M`, (n_m × n_h) cross-species feature cost (xyz, network mask, optional gene)
- `p`, fixed mouse marginal `1/n_m`
- α ∈ [0, 1]. FGW mixing weight (1 = pure relational, 0 = pure feature)
- ε > 0, entropic regularisation strength

We solve:

$$
\pi^* = \arg\min_{\pi}
  (1-\alpha) \cdot \langle M, \pi\rangle
  + \alpha \cdot \sum_{i,j,k,l} (C_m[i,k] - C_h[j,l])^2 \, \pi[i,j] \, \pi[k,l]
  - \varepsilon \cdot H(\pi)
$$

subject to `π·1 = p` and `π ≥ 0` (the column marginal is free in the
semirelaxed setting). `H(π)` is the negentropy regulariser.

We use [POT](https://pythonot.github.io/) (Python Optimal Transport) for the
underlying solver, specifically
`ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein`.

## Anchor supervision

The 42 Garin anchors are encoded as **forbidden cells** in M:

For each visible anchor mouse-position `mp` with correct human-anchor
position `hp_correct`:
- `M[mp, :] = lam_anchor` (high penalty for any other column)
- `M[mp, hp_correct] = 0` (free for the correct human partner)
- And symmetrically along columns

`lam_anchor = 1.0` is large enough relative to the other cost components that
visible-anchor mouse rows are forced to point at their correct human partner.
The held-out (CV) anchors get no such constraint and must find their partner
purely through the FC + xyz + SC signal.

### Region anchors (optional, soft by default)

A region anchor (`homer.data.region_anchors`) generalises a point anchor to a
*set* of mouse parcels mapping to a *set* of human parcels. Encoding in M for
each region anchor with mouse-set `Mset` and human-set `Hset`:

- `M[mp, :] = lam_outside` for `mp ∈ Mset` (mild penalty everywhere else)
- `M[mp, hp] = 0` for `mp ∈ Mset, hp ∈ Hset` (free within the region)
- Symmetrically along human columns.

`lam_outside = 0.15` is the default ("soft region anchor"). Compared to the
legacy hard variant (`lam_outside = 1.0`), the soft constraint produces
better-calibrated probability tails (held-out region CV mean rank ↓ 43 %)
while leaving the trained-π argmax unchanged, see
[iteration log §5.6.0a](archive/iteration_log.md)
for the sweep. Pass `region_lam_outside=1.0` to recover the hard wall.

## Modality combinations

Each modality contributes either to the relational cost (within-species) or to
M (cross-species). The four model levels in `homer.models`:

| Class                   | Relational                | M                           | Headline top-1 |
|-------------------------|---------------------------|-----------------------------|----------------|
| `UnsupervisedGW`        | FC                        | (none)                      | ~14% (Garin only) |
| `SupervisedFGW`         | FC                        | xyz + anchors               | 79% |
| **`MultimodalFGW`**     | **0.7·FC + 0.3·SC**       | **0.5·xyz + anchors**       | **81%** |
| `HierarchicalFGW`       | per-network FC + SC       | per-network xyz + anchors   | 45% (LONO) but 0.55 within-net FC |

Optional terms (off by default in the production config, available as ablations):
- `gene_gw_weight`, gene-coexpression-derived within-species GW cost
- `M_gene_weight`, cross-species cosine cost on ortholog-aligned gene vectors
- `M_anchor_weight`, cross-species cost on anchor-relationship FC features
- `network_mask_weight`, cross-network penalty in M

## Hyperparameters used in production

| Parameter        | Value    | Rationale                                             |
|------------------|----------|-------------------------------------------------------|
| α                | 0.5      | Equal weight to relational + feature terms            |
| ε                | 5e-3     | Small → mostly-deterministic π. Larger ε softens π but loses anchor-CV accuracy. |
| `xyz_weight`     | 0.5      | Spatial prior strong enough to disambiguate, not so strong it swamps FC |
| `lam_anchor`     | 1.0      | Point-anchor forbidden-cell penalty; large vs the [0, 1] cost scale |
| `region_lam_outside` | 0.15 | Region-anchor outside-region penalty (soft default, see archive/iteration_log.md §5.6.0a) |
| `fc_weight`      | 0.7      | Production FC + SC mix                                 |
| `sc_weight`      | 0.3      | Production FC + SC mix                                 |
| `cost_normalisation` | "max" | Each cost matrix divided by its max off-diagonal entry |
| `max_iter`       | 25       | Solutions converge well within 25 iterations           |
| `tol`            | 1e-5     | Loss change tolerance                                 |

## Cost matrices

### Relational (within-species)

- `correlation_distance(FC)` = `1 - r`, symmetrised, zero-diagonal. Output in [0, 2].
- `sc_correlation_distance(SC)` = log1p the heavy-tailed counts, then `1 - r`
  on the row-z'd SC fingerprints.
- `gene_correlation_distance(expr)` = per-gene + per-node z-score, then `1 - r`.
  NaN-row handling via the off-diagonal median.

### Cross-species (M)

- xyz: per-species-normalised Euclidean distance between (x, y, z) coords,
  scaled to roughly [0, 1].
- `cross_species_anchor_M`: cosine distance over anchor-relationship FC vectors.
- `cross_species_gene_cost`: cosine distance over ortholog-aligned gene vectors.
- `network_mismatch_mask`: boolean (n_m, n_h) mask, True for cross-network pairs.

All cost matrices are normalised with `normalise_cost(scheme="max")` before
combination so weighting parameters are interpretable as relative shares.

## Why semirelaxed (not balanced)

With balanced FGW, the human marginal is fixed at uniform `1/n_h`, forcing the
solver to assign positive mass to every human node, including the ~20% of
human parcels that don't have a clear mouse counterpart. This produces a
"smeared" π with no clean per-mouse-row interpretation.

Semirelaxed lets the human marginal float. The cost: ~36% of human nodes end up
with `col_mass ≈ 0` in our production solve (we call them "uncovered"). The
benefit: every mouse row has a sharp, interpretable distribution over human
nodes, and we don't force fake correspondences.

## Validation pipeline

Three independent metrics, each described in [`docs/03_results.md`](03_results.md):

1. **Held-out anchor CV** (binary `top-k`, graded `mean_rank`, `mean_xyz_dist`):
   leave-one-network-out, model gets all anchors except those in the held-out
   network. Most stringent, held-out anchors have *no* M signal pulling them
   to the correct partner. 79–81% top-1 in production.

2. **FC translation quality**: anchor-independent. Push mouse FC through π:
   `Fh_pred = πᵀ · Fm · π / (q · qᵀ)`. Pearson correlation of upper-triangle
   vs actual human FC. r = 0.36 in production (vs r = 0 for random π).

3. **Subject-level K-fold CV**: 80/20 random subject splits, K=5. Tests
   whether the model generalises across cohorts of subjects (not across
   anchors). ~4 pp generalisation gap.

Plus null distributions (random π, permuted anchors) for z-score reporting.

## Citations

- Vayer et al. 2019, *Optimal Transport for structured data with application
  on graphs*, the FGW formulation.
- Sejourne et al. 2021, *Unbalanced Optimal Transport, from Theory to
  Numerics*, semirelaxed variant.
- Flamary et al. 2021, *POT: Python Optimal Transport*, the solver library.
- Garin et al. 2021, *MICCAI*, the 21 cross-species anchor pairs.
- Cutuli, Schaefer, Domhof 2024. Domhof human structural connectivity dataset.
- Knox et al. 2019, *Network Neuroscience*. Allen mouse SC voxel model
  (planned for future M5 work; currently we use Allen *summary structure* SC).
