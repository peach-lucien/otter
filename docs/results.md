# Results — consolidated summary

This document consolidates the headline numbers from
[`outputs/comparison/comparison_summary.md`](../outputs/comparison/comparison_summary.md)
and the per-experiment notes from `ROADMAP.md`. Re-run
`pipeline/07_build_artefacts.py` to regenerate the table from current results.

> **Important caveats added after external review.**
>
> 1. The "top-1" column below is **restricted-anchor ranking accuracy** (argmax
>    among held-out anchor columns only). The full-space top-1 (argmax over all
>    2094 human nodes) for the production `fc_plus_SC` model is **2.4%**, with
>    a mean rank of **206/2094**. See [`methods_writeup.md`](methods_writeup.md#41-held-out-anchor-cross-validation-lono)
>    for both metrics. The production model reliably *ranks the correct anchor
>    first among held-out anchor candidates*; it does NOT reliably pick the
>    correct anchor as the global argmax.
>
> 2. The 4 best configs (`fc_only`, `fc_plus_xyz_gw`, `fc_plus_network_mask`,
>    `fc_plus_SC`) differ by ≤1 of 42 anchors. Paired McNemar tests give
>    p ≈ 1.00 between adjacent configs — **treat them as statistically tied**,
>    not as a strict ordering.
>
> 3. The reported 97.6% bootstrap stability was for the **FC-only** solve, not
>    `fc_plus_SC`. A 40-iter re-run with the production FC+SC config now gives
>    **97.8% argmax-row stability** (88% of mouse rows have identical argmaxes
>    across all 40 samples; 95% have stability > 0.8). FC+SC is essentially
>    indistinguishable from FC-only on this metric — the difference (97.8%
>    vs 97.6%) is well within sampling noise.
>
> 4. The headline FC translation r = 0.36 is **in-sample** (uses the same
>    human FC matrix to build C_h that it evaluates against). The held-out
>    held-out test r is the subject-CV number: **0.32 ± 0.006**.
>
> 5. **External validation** against Beauchamp 2022's published mouse↔human
>    region pairs gives **11.8× chance enrichment at top-1** for the 15
>    anchor-overlapping pairs (927 mouse parcels) and **0× for the 4
>    hippocampal subfield pairs** (no Garin anchor). See
>    [`external_validation.md`](external_validation.md) for the per-pair
>    table — this is the cleanest demonstration that the model captures real
>    cross-species biology where supervised but cannot generalise to
>    unanchored anatomy.

## Headline table

Weighted means across 11 networks; FC-translation = production solve, full
anchors. Production config marked **bold**.

| Config | Top-1 | Top-5 | Pair | Hemi | Rank | xyz_d | FC-r overall | FC-r within | FC-r cross | Subj-CV test r | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline (FC only) | 79% | 100% | 79% | 100% | 1.26 | 0.021 | 0.36 | 0.45 | 0.20 | 0.32 | — |
| FC + xyz GW | 81% | 100% | 81% | 100% | 1.24 | 0.020 | 0.37 | 0.45 | 0.20 | — | — |
| FC + network mask | 81% | 100% | 81% | 100% | 1.24 | 0.020 | 0.38 | 0.49 | 0.17 | — | — |
| **FC + SC (production)** | **81%** | **100%** | **81%** | **100%** | **1.24** | **0.020** | **0.36** | **0.44** | **0.20** | **0.32** | production |
| FC + gene GW | 76% | — | 81% | 95% | — | — | — | — | — | — | — |
| FC + M_gene | 60% | — | 64% | 93% | — | — | — | — | — | — | — |
| FC + SC + M_gene | 62% | — | 69% | 90% | — | — | — | — | — | — | — |
| all modalities (FC+xyz+SC+gene) | 64% | — | 71% | 90% | — | — | — | — | — | — | — |
| FC + selective M_gene | 60% | — | 64% | 93% | — | — | — | — | — | — | — |
| FC + SC + selective M_gene | 62% | — | 69% | 90% | — | — | — | — | — | — | — |
| FC + M_anchor (item A) | 69% | 100% | 69% | 95% | 1.60 | 0.031 | — | — | — | — | negative |
| FC + SC + M_anchor (item A) | 69% | 100% | 69% | 95% | 1.60 | 0.034 | — | — | — | — | negative |
| Hierarchical (per-network) | 45% | 93% | 67% | 64% | 2.36 | 0.160 | 0.39 | 0.55 | 0.16 | — | M4: cleaner WN, hurts CV |
| Iterative hard (lam=1.00, item B) | 81% | 100% | 81% | 100% | 1.24 | 0.020 | — | — | — | — | no-op (identical to production) |

## Null calibration (production = `fc_plus_SC`)

Each cell of the null is a per-trial weighted-mean top-1 across all 11 networks.

| Null kind | n trials | Real top-1 | Null mean | Null std | z-score |
|---|---|---|---|---|---|
| random_pi | 50 | 81% | 28% | 7% | **+7.5** |
| permuted_anchors | 5 | 81% | 31% | 3% | **+17.8** |

The z=17 vs permuted-anchor null is the headline: it tells us the *specific*
pairings of which mouse anchor maps to which human anchor are doing the
work, not just "having anchor supervision in general".

## Bootstrap stability (production solve)

40 subject-level bootstrap iterations:
- mean per-cell stability: **0.976**
- median stability: **1.000**
- frac stable above 0.8: **94.9%**
- frac stable above 0.5: **99.2%**

## Per-experiment notes

### Methodology improvements that worked

- **`fc + xyz_M`** (vs FC only) — top-1 79% → 81%. xyz spatial prior in M is the
  cheapest +2pp we found.
- **Multistart sanity** — loss spread across 6 diverse inits is < 1e-6 nats.
  Anchor supervision + xyz makes the FGW objective well-identified; single-shot
  solutions are trustworthy.
- **Hierarchical** — gives the best within-network FC translation (r=0.55 vs
  0.45 flat) but at the cost of leave-one-network-out CV (because held-out
  networks have no supervision in their sub-block). A complementary tool, not a
  strict improvement.

### Methodology improvements that failed

| Item | What | Why it failed |
|------|------|---------------|
| **A** | Anchor-relationship M cost | Once leak-fixed, hurts CV by ~10pp. The 32 visible anchors' FC patterns aren't enough to predict held-out anchors better than xyz. |
| **B** | Iterative co-clustering | π is already 97.6% concentrated after the first solve — there's no information to recycle. |
| **C** | Confidence-weighted FC | Mouse `n_obs` is uniform, human is 99.97% correlated with the unweighted version. Structural no-op. |
| **M_gene / selective M_gene** | Cross-species gene cosine cost | Helps visual/sensorimotor a bit but tanks subcortical (100% → 20%). |

### Generalisation properties

- **Subject-level CV** (item D): 4 pp generalisation gap (0.36 train r → 0.32
  test r). Robust across folds (std = 0.006). `fc_only` and `fc_plus_SC`
  indistinguishable on this metric.

## What this all means

The production model is at or near the **information-theoretic ceiling** for
the 42-anchor supervision signal on this dataset. Five lines of evidence
converge on this conclusion:

1. **Multiple methodology variants converge to 79–81%**. None of the
   ablations we tried — gene M, M_anchor, iterative co-clustering,
   confidence-weighted FC — moved the needle.
2. **The hard regions are the same across all configs**. Brainstem,
   subcortical, salience, sensorimotor, visual all bottleneck at 25–60%
   regardless of modality choice. The other 6 networks are at 100%.
3. **z = +17.8 vs permuted-anchor null** says the supervision is
   genuinely informative, not just "any 42 constraints help".
4. **Bootstrap stability is 97.6%** — the solution isn't fragile to
   subject sampling.
5. **Subject-level generalisation gap is small** (~4 pp) — the model
   doesn't overfit to the specific subjects it was trained on.

## To break past 81% would need one of

- (i) **More anchors** — Garin's 42 is the rate-limiting supervision signal
- (ii) **Higher-resolution per-node modality data** — Knox 2019 voxel-level
  mouse SC (parked as M5 in `ROADMAP.md`)
- (iii) **External validation** that some held-out anchors are genuinely
  ambiguous (E5)

## Remaining work

The full open list is in [`ROADMAP.md`](../ROADMAP.md). Highest-value
remaining items:
- **V1** — comparison vs colleague's spectral pipeline
- **V2** — methods writeup
- **E5** — external validation against published cross-species maps
- **M5** — Knox 2019 voxel-level mouse SC
