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

## Optimal transport setting

Mapping atlases of different sizes and modalities is an optimal transport problem:
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
- ε > 0, Kullback-Leibler proximal weight of the solver

We solve:

$$
\pi^* = \arg\min_{\pi}
  (1-\alpha) \cdot \langle M, \pi\rangle
  + \alpha \cdot \sum_{i,j,k,l} (C_m[i,k] - C_h[j,l])^2 \, \pi[i,j] \, \pi[k,l]
$$

subject to `π·1 = p` and `π ≥ 0` (the column marginal is free in the
semirelaxed setting). The objective carries no entropy term.

The solver is a Bregman proximal-point iteration. Each step multiplies the
current coupling entrywise by `exp(-grad F(pi) / eps)` and rescales its rows to
`p`, so ε weights a KL penalty tying each iterate to the one before it rather
than penalising the entropy of the coupling. The released coupling is the 25th
iterate at ε = 0.05. It is not the converged solution: carried to `tol` the same
fit takes 591 iterations and returns an almost deterministic coupling that does
not score better on the held-out benchmark (AUROC 0.894 against 0.899, mean
centroid displacement 9.88 mm against 8.83 mm). Diffuseness comes from stopping
early, not from a fixed point.

The iteration approximates a continuous flow in τ = iterations / ε, so ε and the
iteration count trade off against one another. At matched τ = 500 the couplings
obtained at ε = 0.05, 0.2 and 1.0 agree on the objective and on the median
top-ranked probability to three significant figures. The agreement is close but
not exact, because the discretisation is path dependent. τ is used below as a
shorthand for the pair (iterations, ε), not as an exact invariant.

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

### Regional correspondence entries

A region anchor (`otter.data.region_anchors`) generalises a point anchor to a
*set* of mouse parcels mapping to a *set* of human parcels. Encoding in M for
each region anchor with mouse-set `Mset` and human-set `Hset`:

- `M[mp, :] = lam_outside` for `mp ∈ Mset` (mild penalty everywhere else)
- `M[mp, hp] = 0` for `mp ∈ Mset, hp ∈ Hset` (free within the region)
- Symmetrically along human columns.

`lam_outside = 0.15` is the canonical soft penalty. The registry contains 26
entries from 15 comparative-anatomy modules; see
[04_anchor_packs.md](04_anchor_packs.md).

## Canonical recipe

The relational cost is a 0.7:0.3 mixture of functional- and structural-
connectivity distances. The cross-species cost contains the anchor-warped
spatial distance, Garin point-anchor penalties and the 26 regional entries.
Gene expression is used in transfer analyses, not in the canonical fit.

## Hyperparameters

| Parameter        | Value    | Rationale                                             |
|------------------|----------|-------------------------------------------------------|
| α                | 0.5      | Equal weight to relational + feature terms            |
| ε                | 0.05     | KL proximal weight of the solver. With `max_iter` = 25 this gives τ = 500. The five-fold grid search most often selected ε = 0.2; the released ε = 0.05 had nearly identical benchmark accuracy and a more concentrated parcel-level coupling. |
| `xyz_weight`     | 0.25     | Selected most often across the five Beauchamp folds. Applied after a thin-plate-spline warp fitted to the 42 bilateral Garin coordinate pairs. |
| `lam_anchor`     | 1.0      | Point-anchor forbidden-cell penalty; large vs the [0, 1] cost scale |
| `region_lam_outside` | 0.15 | Region-anchor outside-region penalty (soft default) |
| `fc_weight`      | 0.7      | Functional-connectivity share of the relational cost   |
| `sc_weight`      | 0.3      | Structural-connectivity share of the relational cost   |
| `cost_normalisation` | "max" | Each cost matrix divided by its max off-diagonal entry |

| `max_iter`       | 25       | Outer solver iterations. With ε = 0.05 this is τ = 500. The released fit stops here rather than running to `tol`. |
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

## Semirelaxed versus balanced formulations

With balanced FGW, the human marginal is fixed at uniform `1/n_h`, forcing the
solver to assign positive mass to every human node, including human parcels
that have no clear mouse counterpart. This produces a smeared π with no
per-mouse-row interpretation.

Semirelaxed lets the human marginal float, so the coupling can leave human parcels poorly
reconstructed rather than forcing mass onto them. We report reconstruction accuracy
(docs/03_results.md §5) rather than an uncovered-parcel percentage, which would depend on the
threshold chosen. Each column of π is normalised before the push-forward, so the score reflects
whether some mouse tissue is wired like the human parcel rather than how much mass that parcel
received, and §5 of `03_results.md` builds its central measurement on it. Every mouse row keeps
an interpretable distribution over human nodes.

## Evaluation design

The 19 scorable Beauchamp correspondences provide a region-level scoring frame
and inform the hyperparameter grid. They are not supplied as anatomical
constraints, but some territories overlap the Garin-derived spatial scaffold or
regional entries. The principal generalisation checks therefore refit after
removing each overlapping target's supervision and leave out each scorable
Garin class or regional entry in turn.

Additional analyses test coupling organisation, cross-species map transfer,
mouse-based reconstruction of human functional connectivity and forward or
reverse map routing. [03_results.md](03_results.md) links each analysis to its
current producer and provenance-stamped result log.

## Citations

- Vayer et al. 2019, *Optimal Transport for structured data with application
  on graphs*, the FGW formulation.
- Sejourne et al. 2021, *Unbalanced Optimal Transport, from Theory to
  Numerics*, semirelaxed variant.
- Flamary et al. 2021, *POT: Python Optimal Transport*, the solver library.
- Garin et al. 2021, *MICCAI*, the 21 cross-species anchor pairs.
- Cutuli, Schaefer, Domhof 2024. Domhof human structural connectivity dataset.
- Knox et al. 2019, *Network Neuroscience*. Allen mouse SC voxel model. The eleven cortical
  Garin anchor classes take leaf-level connectivity from this model, which resolves cortical
  areas the summary atlas does not separate; subcortical and brainstem parcels keep Allen
  *summary structure* SC.
