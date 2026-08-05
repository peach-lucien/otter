# Results

Headline numbers and per-experiment notes. The raw tables live in
`outputs/comparison/comprehensive_table.csv`; re-run
`pipeline/07_build_artefacts.py` to regenerate them after any new evaluation.

> **Caveats — read these first.**
>
> 1. **Top-K, not top-1, is the right metric for π.** This is a *soft probabilistic* mapping — π[i, :] is a probability distribution over 2094 human parcels. Asking "did the *single most-probable* parcel land exactly in the published target sphere?" (top-1) is brittle: a parcel can be in the next-most-probable cell and score 0. Top-5 and top-10 tell you whether the correct human is *in the model's short list*, which is what a downstream user actually consumes. We report all three; top-5 / top-10 are the headline numbers.
> 2. The "top-1" column below is **restricted-anchor ranking accuracy** (argmax among held-out anchor columns only). Full-space top-1 (argmax over all 2094 human nodes) is **2.4%**, mean rank **206/2094**. The model reliably *ranks the correct anchor first among held-out anchor candidates*; it does NOT reliably pick the correct anchor as the global argmax.
> 3. The 4 best configs (`fc_only`, `fc_plus_xyz_gw`, `fc_plus_network_mask`, `fc_plus_SC`) differ by ≤1 of 42 anchors. McNemar p ≈ 1.00 between adjacent configs — **statistically tied**.
> 4. FC translation r = 0.36 is **in-sample**; held-out subject-CV is **0.32 ± 0.006**.
> 5. Bootstrap stability is 97.8% (40-iter, fc_plus_SC).
> 6. **External validation against Beauchamp 2022.** In supervised regions (15 pairs, 927 parcels) the model lands a Beauchamp-target parcel in its top-5 about **22%** of the time and top-10 about **27%** (top-1 is 12%, ≈11.8× chance for those regions). The "all 19 pairs" aggregate that mixes in the 4 novel hippocampal pairs (top-1/5/10 all 0%) gives 11.5% / 20% / 24%. 0× enrichment for novel hippocampal regions cleanly demonstrates the model captures real cross-species biology where supervised but cannot generalise to unanchored anatomy. Details below.

---

## 1. Headline configurations

Weighted means across 11 networks. Production = `fc_plus_SC`.

| Config | Top-1 | Top-5 | Pair | Hemi | Rank | xyz_d | FC-r overall | FC-r within | FC-r cross | Subj-CV test r | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline (FC only) | 79% | 100% | 79% | 100% | 1.26 | 0.021 | 0.36 | 0.45 | 0.20 | 0.32 | — |
| FC + xyz GW | 81% | 100% | 81% | 100% | 1.24 | 0.020 | 0.37 | 0.45 | 0.20 | — | — |
| FC + network mask | 81% | 100% | 81% | 100% | 1.24 | 0.020 | 0.38 | 0.49 | 0.17 | — | — |
| **FC + SC (production)** | **81%** | **100%** | **81%** | **100%** | **1.24** | **0.020** | **0.36** | **0.44** | **0.20** | **0.32** | production |
| FC + gene GW | 76% | — | 81% | 95% | — | — | — | — | — | — | — |
| FC + M_gene | 60% | — | 64% | 93% | — | — | — | — | — | — | — |
| FC + M_anchor (item A) | 69% | 100% | 69% | 95% | 1.60 | 0.031 | — | — | — | — | negative |
| Hierarchical (per-network) | 45% | 93% | 67% | 64% | 2.36 | 0.160 | 0.39 | 0.55 | 0.16 | — | best within-net FC, hurts CV |

## 2. Null calibration

Each cell is a per-trial weighted-mean top-1 across all 11 networks (production = `fc_plus_SC`).

| Null kind | n trials | Real top-1 | Null mean | Null std | z-score |
|---|---|---|---|---|---|
| random_pi | 50 | 81% | 28% | 7% | **+7.5** |
| permuted_anchors | 5 | 81% | 31% | 3% | **+17.8** |

The z=17.8 vs permuted-anchor null is the headline statistic: it confirms the *specific* mouse↔human anchor pairings drive the result, not just "having any 42 anchor constraints".

## 3. Bootstrap stability (production fc_plus_SC, 40 subject-bootstrap iterations)

- Mean argmax-row stability: **97.8%**
- 88% of mouse rows have *identical* argmax across all 40 samples
- 95% have stability > 0.8; 99.4% have stability > 0.5

## 4. External validation against Beauchamp 2022

**Source.** Beauchamp et al. 2022 (*eLife*) curated 36 canonical mouse↔human region pairs in `MouseHumanTranscriptomicSimilarity/create_neuro_pairs.R`. We use the **22 non-cerebellar pairs** (cerebellum is excluded from our parcellation).

**Method** (`pipeline/05f_beauchamp_validation.py`).
- Mouse side: each Beauchamp DSURQE region → set of label IDs in DSURQE hierarchy → overlap with our 1864 parcels via spatial mapping into `DSURQE_CCFv3_labels_200um.mnc` (origin offset (-0.027, -2.334, +1.018) calibrated from 6 unambiguous L/R-Visual/Motor/Auditory anchors).
- Human side: each Beauchamp AHBA region → hand-curated MNI152 centroid + radius (e.g. precentral gyrus = (±35, -20, 55), r=15mm). Membership = our parcels within radius of either L or R centroid.
- For each pair: top-K of π[mouse_region, :] hits human_region; mean rank of best human-region parcel; xyz distance from argmax to human-region centroid.

### 4.1 Per-pair recovery (production π)

`outputs/logs/beauchamp_validation.json`

| Beauchamp pair | n_m | n_h | top-1 | top-5 | top-10 | mean rank | dist_mm | category |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| **Thalamus → thalamus** | **110** | **22** | **33%** | **48%** | **55%** | **381** | **24.4** | **A** |
| Primary auditory area → Heschl's gyrus | 9 | 18 | 22% | 22% | 22% | 1084 | 67.2 | A |
| Primary somatosensory area → postcentral gyrus | 155 | 39 | 20% | 41% | 47% | 573 | 45.3 | A |
| Anterior cingulate area → cingulate gyrus | 23 | 24 | 13% | 22% | 35% | 37 | 36.5 | A |
| Caudoputamen → caudate nucleus | 149 | 24 | 13% | 28% | 32% | 757 | 36.5 | A |
| Hypothalamus → hypothalamus | 52 | 4 | 12% | 19% | 19% | 1000 | 29.1 | A |
| Striatum ventral region → nucleus accumbens | 26 | 2 | 8% | 38% | 58% | 374 | 20.1 | A |
| Visual areas → cuneus | 54 | 36 | 7% | 7% | 7% | 1294 | 66.7 | A |
| Pallidum → globus pallidus | 44 | 6 | 5% | 7% | 9% | 1071 | 26.6 | A |
| Pons → pons | 69 | 6 | 3% | 3% | 3% | 1397 | 51.6 | A |
| Primary motor area → precentral gyrus | 53 | 45 | 0% | 2% | 9% | 1088 | 35.6 | A |
| Cortical subplate-other → amygdala | 54 | 6 | 0% | 7% | 9% | 907 | 46.6 | A |
| Inferior colliculus → inferior colliculus | 29 | 4 | 0% | 0% | 0% | 1431 | 72.3 | A |
| Superior colliculus → superior colliculus | 53 | 2 | 0% | 0% | 0% | 1872 | 59.8 | A |
| Piriform area → piriform cortex | 47 | 13 | 0% | 17% | 28% | 657 | 47.1 | A |
| Subiculum → subiculum | 29 | 8 | 0% | 0% | 0% | 1251 | 57.9 | N |
| Field CA1 → CA1 field | 15 | 6 | 0% | 0% | 0% | 1297 | 66.3 | N |
| Field CA3 → CA3 field | 26 | 4 | 0% | 0% | 0% | 1102 | 49.9 | N |
| Dentate gyrus → dentate gyrus | 22 | 4 | 0% | 0% | 0% | 1300 | 54.9 | N |

A = anchor-overlapping; N = novel (no Garin anchor; hippocampal subfield).

### 4.2 Aggregate (weighted by n_mouse_parcels)

|  | Anchor (n=15, 927 parcels) | Novel (n=4, 92 parcels) | All (n=19, 1019 parcels) | Chance | Enrichment |
|---|---:|---:|---:|---:|---:|
| top-1  | 12% | 0%  | 11% | 0.9% | **11.5×** |
| top-5  | 22% | 0%  | 20% | 4.5% | **4.5×**  |
| top-10 | 27% | 0%  | 24% | 8.7% | **2.8×**  |

The anchor-vs-novel split is starkly clean: 11.8× chance on anchor-overlapping regions, 0× on novel (hippocampal) regions. This is exactly what the supervision-density hypothesis predicts.

### 4.3 Sanity check: permuted-π null

Re-running the same validation on a row-shuffled version of π gives 0.6× chance enrichment (as expected ≈ 1×). The 11.8× signal is real, not an artefact of the chance-baseline calculation.

---

## 5. Anchor expansion experiments

### 5.1 SPLIT-1: narrow M1 anchor (pid=22)

**Hypothesis** (per `docs/diagnostics.md`): the existing Motor anchor (pid=2) is a union of M1 + premotor + FEF + SCEF + Area 6 subdivisions; its centroid (-30.8, -6.4, +52.2) drifts ~14mm anterior of canonical M1 (-35, -20, +55). Adding a narrow M1 anchor should pull non-anchor mouse-motor parcels back to M1.

**Choice.** Mouse L=L_708 (-1.47, +2.61, +1.80), Mouse R=R_808 (+1.53, +2.61, +2.40); Human L=L_935 (-36, -18, +54) — **2.4mm from canonical M1**.

**Result.** Primary motor area → precentral gyrus, top-1: **0% → 4%**, top-5: 2% → 6%, top-10: 9% → **13%**. All 14 other anchor-overlapping pairs unchanged.

### 5.2 EXP-1: hippocampal subfield anchors (pid=23-26)

Added 8 supplementary anchor parcels (4 pair_ids × L+R) for CA1, CA3, Dentate, Subiculum.

| pid | Subfield | Mouse L / R | Human L / R MNI |
|---|---|---|---|
| 23 | CA1 | L_660 / R_546 | (±36, -18, -9) |
| 24 | CA3 | L_326 / R_326 | (±27, -18, -9) |
| 25 | Dentate | L_548 / R_548 | (±27, -27, -9) |
| 26 | Subiculum | L_311 / R_422 | (±18, -36, -9) |

| Beauchamp pair | Before | After | Δ |
|---|---:|---:|---:|
| Subiculum → subiculum, top-1 | 0% | **7%** | +7 pp |
| Field CA1 → CA1 field, top-1 | 0% | 0% | — (didn't help; flagged as S5) |
| Field CA3 → CA3 field, top-1 | 0% | **8%** | +8 pp |
| Dentate gyrus → dentate gyrus, top-1 | 0% | **9%** | +9 pp |

**Aggregate:** novel-pair top-1 went from **0% → 7%** (0× → **24.4× chance enrichment**). Anchor-overlapping pairs and other 14 unchanged.

### 5.3 S4: Region-anchor supervision

The supplementary anchors above (SPLIT-1, EXP-1) are still **point-to-point**: one mouse parcel ↔ one human parcel. Garin's anchors and Beauchamp's region pairs are natively *region* objects (sets of parcels), so we built a region-anchor mechanism (`otter.data.region_anchors`): each mouse parcel in a declared region is allowed to map to *any* human parcel in the matching declared region, forbidden elsewhere.

**Proof of concept**: declare ALL 53 mouse-Motor parcels (DSURQE "Primary motor area") ↔ ALL 44 human-precentral parcels (within 15mm of canonical M1) as one region anchor (pid=30). Re-solve and re-validate.

| Metric | Production π | + region anchor for Motor |
|---|---:|---:|
| Beauchamp Motor pair top-1 | 0% | **100%** (by construction) |
| Beauchamp Motor mean rank | 1088 / 2094 | **1.0 / 2094** |
| All-pair top-1 enrichment | 11.5× chance | **15.8× chance** |
| Anchor-overlapping aggregate top-1 | 12% | **16%** (12.0× → 16.3× chance) |
| Pairs unchanged | — | 13 of 14 other anchored pairs |
| Pairs degraded | — | Somatosensory 20% → 12% (adjacent parcels redistributed; see S5) |
| Novel pairs (hippocampal) | 0% | 0% (unchanged — still no region anchor for them) |

By construction the Motor parcels argmax within the declared region; the meaningful finding is that **other anchored regions are largely preserved** (13 of 14) and the one degradation (somatosensory) is a spatially-adjacent parcel-stealing effect, not a deep failure.

The region-anchor mechanism generalises immediately to:
- Hippocampal subfields (replace point anchors with region anchors)
- Tectum (declare mouse-tectum-region ↔ human-midbrain-region, working around the spatial-inversion problem identified in `diagnostics.md`)
- Any of Beauchamp's region pairs as native multi-parcel anchors

YAML configs in `config/region_anchors_*.yaml`. 7 unit tests in `tests/test_region_anchors.py`.

### 5.4 Infrastructure

`otter.data.supplementary_anchors` (point) and `otter.data.region_anchors` (region) both let you promote existing non-anchor parcels to anchors with new pair_ids, without modifying the underlying FC matrix or atlas. The point form goes via `MultimodalFGW.fit(M_aug, H_aug, ...)` after augmenting the var; the region form goes via `MultimodalFGW.fit(..., region_anchors=[entry, ...])` directly. Tests in `tests/test_supplementary_anchors.py` (6) + `tests/test_region_anchors.py` (7).

### 5.5 S7 — Region anchors via real atlas labels (no fabricated centroids)

`otter.data.atlas_regions.build_garin_region_anchors_from_atlases(M.var, H.var)` builds region anchors using **only published atlases**:
- **Mouse**: DSURQE labels (Beauchamp 2022 repo) — already used by the validation pipeline.
- **Human**: Schaefer-400 (cortical) + JuBrain-184 (where Schaefer is missing). Both are in the Domhof bundle we already use as the human-FC source.

Each Garin anchor's region is defined as: *all our parcels with the same atlas label as the anchor parcel itself* — not a fabricated MNI sphere.

Coverage: **15 of 21 Garin pairs** convert to region anchors. The 6 missing ones (Septum, Periarchicortex, Striatum, Pallidum, Thalamus, Pons) have no Schaefer or JuBrain label at the anchor location and stay as point anchors — we don't fabricate.

**Two atlas-resolution issues** (warned at build time):
- Visual striate (pid=5) and Visual pre/extra (pid=6) share 17 human parcels in Schaefer-400 (same Schaefer ID covers both).
- Insula (pid=9) and Claustrum (pid=16) share 8 (claustrum is buried inside insular cortex at Schaefer's resolution).

These overlaps create ambiguous constraints — the solver can satisfy at most one of each pair. They're a real atlas-resolution limit, not a bug.

### 5.6 S8 — Held-out region CV

Built the analogue of leave-one-network-out CV but for region anchors. For each of the 15 region anchors, hold it out of the supervision set, re-solve, then score: do the held-out region's mouse parcels argmax to its declared human parcels?

This is a methodological probe — held-in supervision gives 100% by construction (the constraint is enforced), so that number can't separate "model captures homology" from "model enforces what we told it". Held-out CV asks the latter question by removing the constraint for one region at a time. Useful for understanding the mechanism; not the right metric for users who'll query π *with* its full supervision active.

| pid | Region | Held-out top-1 | Held-out top-5 | Mean rank /2094 | Held-in top-1 |
|---|---|---:|---:|---:|---:|
| 31 | mPFC | **33%** | 33% | 23 | 100% |
| 32 | Motor | 4% | 6% | 59 | 100% |
| 33 | Somatosensory | 11% | 26% | 29 | 100% |
| 34 | Posterior parietal | 5% | 9% | 218 | 100% |
| 35 | Visual striate | 0% | 0% | 1434 | 0% (overlap with pid=36) |
| 36 | Visual pre/extra | 0% | 0% | 1546 | 100% |
| 37 | Auditory | **22%** | 22% | 286 | 100% |
| 38 | Middle/Inf temporal | 10% | 10% | 962 | 100% |
| 39 | Insula | 2% | 2% | 787 | 100% |
| 41 | Olfactory | 0% | 0% | 1388 | 0% (FC mass on amygdala) |
| 44 | Basal forebrain | 8% | 17% | 22 | 100% |
| 46 | Claustrum | 0% | 0% | 1393 | 100% |
| 47 | Amygdala | 4% | 4% | 83 | 100% |
| 48 | Hypothalamus | 2% | 2% | 1830 | 100% |
| 51 | Tectum | 2% | 2% | 591 | 100% |

**Weighted aggregate (n=15 pairs, 786 mouse parcels):**

|  | Held-out (real generalisation) | Held-in (by construction) |
|---|---:|---:|
| top-1 | **3.4%** | 80% |
| top-5 | 5.5% | 87% |
| top-10 | 6.6% | 88% |

### 5.6.0a SOFT-1 — Soft region anchors (hard penalty → mild penalty)

The original `apply_region_supervision` set `M[mp, h_outside] = 1.0` (hard prohibitive) and `M[mp, h_inside] = 0` (free). This makes the constraint a 0/1 wall: the optimizer literally cannot put any mass outside the declared region. When atlas regions overlap (Visual striate ↔ pre/extra, Insula ↔ Claustrum in our setup) or the Garin anchor is mis-placed relative to Beauchamp's target (Motor; DIAG-1), the hard wall actively *misdirects* mass.

Soft alternative: replace the 0/1 wall with a mild penalty `lam_outside < 1`. The optimizer still prefers in-region cells (cost = 0 vs lam_outside) but *can* violate the constraint if structural cost (Cm, Ch) strongly disagrees.

Held-out region CV sweep (15 region anchors × 7 `lam_outside` values × 5 s each):

| `lam_outside` | top-1 | top-5 | top-10 | Mean rank |
|---|---:|---:|---:|---:|
| 0.05 (very soft) | 3.4% | 5.5% | 6.5% | 509 |
| **0.10** | **3.4%** | **5.5%** | **6.6%** | **477** ← best |
| 0.20 | 3.4% | 5.5% | 6.6% | 483 |
| 0.30 | 3.4% | 5.5% | 6.6% | 707 |
| 0.50 | 3.4% | 5.5% | 6.6% | 836 |
| 0.70 | 3.4% | 5.5% | 6.6% | 836 |
| 1.00 (hard, legacy) | 3.4% | 5.5% | 6.6% | 843 |

**Findings:**

- **Discrete top-K is unchanged** at any `lam_outside`. The argmax / top-5 / top-10 of held-out regions is determined by FC/SC structure; the strength of nearby region constraints doesn't change which cells "win" the discrete ranking.
- **Mean rank improves substantially** with softer constraints — from 843 / 2094 (hard) to 477 / 2094 (best at `lam_outside = 0.1`). That's a 43% reduction. The correct human partner moves from "top 40%" of π's row to "top 22%". The full *probability distribution* π is better-calibrated, even though the argmax is unchanged.
- The sweet spot is `lam_outside ≈ 0.1–0.2`. Below that the constraint loses effect; above 0.3 it starts behaving like the hard version.

**Default (committed)**: `region_lam_outside = 0.15` is now the default in both `apply_region_supervision()` and `MultimodalFGW._solve()`. Hard constraints (1.0) remain available for cases where you want explicit enforcement (e.g., known biological priors that should override structure) — pass `region_lam_outside=1.0`.

**Implication for the trust map**: a softer region anchor doesn't change which parcel "wins" the per-row argmax — so the trust map (which is argmax-based) won't move. But for downstream tasks that use the full π distribution (FC translation, region-set lookups, soft homology probabilities), the soft version provides better-calibrated answers.

**Important caveat — the held-in trained π is near-identical between hard and soft on our dataset.**
The mean-rank improvement above is from the **held-out** CV sweep, where the region is *not* in the supervision set and only nearby region constraints leak in. When we actually re-solve the production model (FC + SC + 15 atlas region anchors) with `region_lam_outside=0.15` vs `1.0`, the two trained π files agree to solver precision: max abs diff `1.3e-7`, mean abs diff `3.3e-12`, argmax overlap 100 % (1864 / 1864 rows), Beauchamp top-K identical. The reason: once a region is *in* the supervision set, FC + SC structure already concentrates mass on the same in-region cells regardless of whether outside is penalised at 0.15 or 1.0; the hard wall is essentially redundant. So the default change is principled (it's the right default for new users adding new region anchors on data where structure is weak) but largely cosmetic for anyone querying the shipped π — both regimes produce the same numbers downstream.

**When the choice actually matters:**

- **Fitting on new data / new region anchors** where FC/SC support for the region is thin — soft (0.15) leaves room for structure to push back, hard (1.0) forbids that.
- **Held-out / leave-one-region-out evaluations** — soft gives 43 % lower mean rank for the held-out region (the nearby-constraint regime above).
- **Anywhere you consume the full π distribution, not just argmax** — entropy, region-set probability sums, soft homology — soft is mildly better-calibrated.

### 5.6.0 Ablation: does the source marginal weighting matter?

By default we use `p = 1/n_m` uniformly — every mouse parcel contributes equal mass. Tested two non-uniform alternatives via held-out region CV:

- **Volume-weighted**: `p_i ∝` # voxels in parcel i (anatomically natural — parcels span 12-2837 voxels in mouse, 236× dynamic range).
- **Stability-weighted**: `p_i ∝` per-row bootstrap stability of the production π (more reliable parcels carry more weight).

| Config | Held-out top-1 | Mean rank |
|---|---:|---:|
| Uniform (current default) | **3.4%** | 843 / 2094 |
| Volume-weighted | 3.3% | 838 |
| Stability-weighted | 3.4% | 843 |

**Per-region: 14 of 15 pairs show 0.0 pp difference between any of the three.** Marginal weighting is a no-op for held-out generalisation in our setup. Reason: anchor supervision dominates the cross-species cost for the ~52 anchored mouse parcels, and the semirelaxed solver lets the human marginal float freely — so the source marginal just adjusts per-row weight in the loss without changing which solution is found. Useful finding: users don't need to compute parcel volumes or per-row stabilities to get the same π.

### 5.6.1 Ablation: does SC contribute?

Re-ran S8's held-out region CV with `use_sc=False` (FC only):

| Metric (held-out, n=15 regions, 786 mouse parcels) | FC only | FC + SC | Δ |
|---|---:|---:|---:|
| Top-1 | **3.6%** | 3.4% | -0.1 pp |
| Mean rank | 853 / 2094 | 843 / 2094 | -10 |

Per-region top-1 is **identical** for FC-only and FC+SC across all 15 regions (Δ = 0 pp on each pair, within rounding). The mean-rank improvement of ~10 is negligible (~0.5% of search space).

**Conclusion**: SC is a no-op for held-out generalisation in our setup. This is consistent with the held-in tie observed in the original multi-modal CV (`fc_only` and `fc_plus_SC` differ by ≤1 of 42 anchors). On this dataset, the simpler `SupervisedFGW(xyz_weight=0.5)` is equivalent to the production `MultimodalFGW(use_sc=True, sc_weight=0.3, fc_weight=0.7)` — and Occam favours the simpler one. Caveat: this is specific to the 1864/2094-node parcellation + Allen summary-structure SC + Domhof FC; richer SC (e.g., Knox voxel-level cortex-only) might change the story for cortical regions. We've shown it doesn't (`docs/results.md §6.2`), but worth re-checking on different datasets.

### 5.7 What S7 + S8 actually tell us

**Two numbers, two questions, both useful:**

**11.8× chance enrichment (held-in / trained model):** "If I query π for a parcel in a supervised region, how often does the model's prediction land in the human region Beauchamp says it should?" This is what a downstream user experiences when using π for translation work — the trained model with all supervision active. ~12% top-1 means roughly 1-in-8 lookups land in the published target.

**3.4% chance enrichment (held-out CV):** "If we remove a region's supervision and re-solve, how often does FC/SC structure alone recover the homology?" This is a methodological probe of what the underlying graph structure encodes vs what supervision contributes. ~7× chance overall, with mPFC (33%), Auditory (22%), Somatosensory (11%) showing strong unsupervised signal.

The held-in figure is what users should look at when deciding whether to trust a specific prediction (Trust map / `docs/whats_in_the_box.md`). The held-out figure is what we cite when claiming "the FGW framework captures real cross-species biology" rather than "supervision wholly determines results".

**Caveat on both:** Beauchamp 2022's 22 pairs are themselves a published hypothesis (derived from gene-expression similarity), not ground truth. Different validation sources (Mars 2018 white-matter, Coletta 2020 FC) might give different numbers. Adding more independent validation is roadmap item S3.

### 5.10 Region-level evaluation — predicting human *regions*, not parcels

Parcel-level top-K (§4-§5.7) asks "is the right human *parcel* in the model's short list?". That's harsh for a soft probabilistic mapping where π spreads mass across a region rather than nailing a single cell. The natural region-level question is:

> Given a mouse region (set of parcels), which *human region*, out of a candidate set of named regions, does the model predict?

**Implementation** (`otter.eval.region_level`, `pipeline/05j_region_level_eval.py`). For mouse region M with parcel indices `M_idx`:

  `pi_M = π[M_idx, :].sum(axis=0); pi_M /= pi_M.sum()` → distribution over 2094 human parcels.

For each candidate human region `H_i`, score = `pi_M[H_i_mask].sum()`. Rank candidates by score; report top-K. Fold enrichment = `score / (|H_i|/n_h)` (mass on `H_i` relative to uniform expectation).

**Candidate set: Beauchamp-22** — the same 22 hand-curated human regions used in §4-§5.7, used as candidates against each other. 21 evaluable (medulla excluded; not in our atlas vocabulary). Chance top-1 varies by region size (≈ 1/21 ≈ 4.8% for equal-size).

**Two flavours of top-K hit reported:**

- **Rank-only top-K**: `rank(H_true) ≤ k`.
- **Qualified top-K**: `rank ≤ k` AND `fold_enrichment ≥ 1.0`. Filters out "vacant" wins where the model put near-zero mass on every candidate and the true region won the noise (e.g. Motor, Inferior Colliculus — both rank 1 but fold ≈ 0).

#### 5.10.1 Production π (FC + SC + 21 Garin point anchors)

| Metric | All evaluable pairs (19) | Anchor-overlapping (15) | Novel hippocampal (4) |
|---|---:|---:|---:|
| Rank top-1 | 46.5% | 51.1% | 0.0% |
| Rank top-3 | 87.8% | 89.4% | 71.7% |
| Rank top-5 | 90.4% | 89.4% | 100.0% |
| **Qualified top-1** | **33.9%** | **37.2%** | **0.0%** |
| **Qualified top-3** | **63.5%** | **69.8%** | **0.0%** |
| **Qualified top-5** | **63.5%** | **69.8%** | **0.0%** |
| Mean rank (of 21) | 2.01 | 1.91 | 2.97 |
| Median rank | 2.0 | — | — |
| Mean fold enrichment | 14.4× | 15.8× | 0.0× |
| Median fold enrichment | 4.3× | — | — |
| Mean candidate-set mass coverage | 20.1% | — | — |

The qualified metric is the honest read: it correctly identifies novel hippocampal pairs as having no real signal (rank-only top-3 of 71.7% is a candidate-overlap artefact — hippocampal subfields are spatially clustered and overlap each other in the candidate set).

#### 5.10.2 Null calibration

Two nulls, both run with n=100 trials. (`pipeline/05j_region_level_eval.py`):

| Null | Null top-1 | Null top-3 | Null top-5 | Null fold | z top-1 | z top-3 | z top-5 | z fold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Column-permuted (shuffle `pi_M`'s column order) | 8.4 ± 9.0% | 24.8 ± 13.4% | 38.0 ± 12.5% | 1.27 ± 0.90× | +4.2 | +4.7 | +4.2 | +14.6 |
| Source-permuted (score `H_true` against another mouse region's `pi_M`) | 11.2 ± 8.0% | 52.1 ± 11.2% | 66.6 ± 11.1% | 1.93 ± 2.13× | +4.4 | +3.2 | +2.1 | +5.9 |

The source-permuted null is the stronger test — it asks whether the mass on `H_true` is *specific* to its matching mouse region M, or a generic bias the model has regardless of source. **The model passes both nulls cleanly at top-1 (+4σ) and at fold enrichment (+6σ source / +15σ column).** Top-5 is only marginally above source-permuted (+2.1σ), reflecting that the candidate set has many similar-size adjacent regions.

#### 5.10.3 What this changes about the story

The parcel-level numbers (top-1 = 12 %, top-5 = 20 %, top-10 = 24 % in supervised regions) describe the worst case: "how often is the *single* most-probable human parcel correct?". Region-level numbers describe the use case for any downstream user who works at region granularity:

- **The model picks the right human region as its top hypothesis in 34 % of cases** (qualified top-1), and the right region is in its top-3 in 64 %.
- **Mean fold enrichment is 14×** — when the model scores `H_true`, it does so at ~14× the mass a uniform π would put there.
- **Mean rank is 2.0 / 21** — the right region is almost always in the model's top two candidates.

The novel-region 0 % qualified shows that this is real region-specific signal, not a candidate-set artefact: where supervision exists, the model identifies the right region; where it doesn't, the qualified metric correctly reports nothing.

#### 5.10.4 Region anchors on this metric

Re-running with the soft-region-anchor π (`pi_fc_plus_SC_with_soft_atlas_regions.npy`):

| Metric | Production (point anchors) | + soft atlas region anchors |
|---|---:|---:|
| Qualified top-1 (anchor-overlapping) | 37.2 % | 31.8 % |
| Qualified top-3 (anchor-overlapping) | 69.8 % | 58.8 % |
| Rank top-3 (anchor-overlapping) | 89.4 % | 95.3 % |
| Mean fold (anchor-overlapping) | 15.8× | **40.8×** |
| Mean candidate-set mass coverage | 20.1 % | 24.5 % |

Region anchors do exactly what they promise: they *concentrate* more mass inside the supervised region (mean fold 15.8× → 40.8×) and push rank top-3 up from 89% → 95%. Qualified top-K drops slightly because tighter concentration produces more rank-1 ties (penalised by the conservative "ties favour truth" rule we use for ranking). For anyone using π at region granularity, the soft-atlas-region version is strictly more concentrated on the right answer.

#### 5.10.5 V2 candidate set — JuBrain ∪ Beauchamp-extras (~151 regions)

The Beauchamp-22 set is small (21 evaluable) and only covers ~10 % of the 2 094 human parcels (mean candidate-set mass coverage on production π is just 20 %). For a sterner test, we re-run with the **JuBrain-184 atlas** as the candidate set. JuBrain regions (≥ 3 parcels) yield ~130 candidates; we union in 7 hand-curated Beauchamp targets (Pallidum, NAc, Caudate, IC, SC, Pons, Hypothalamus, Thalamus, CA-fields) that JuBrain doesn't cover, plus the matched JuBrain region (selected by best Jaccard overlap) for the other Beauchamp targets. Total = **151 candidates** (chance top-1 ≈ 0.7 %).

| Metric (production π, anchor-overlapping pairs, 927 parcels) | Beauchamp-22 (21 candidates) | JuBrain ∪ extras (151 candidates) |
|---|---:|---:|
| Qualified top-1 | 37.2 % | **27.9 %** |
| Qualified top-3 | 69.8 % | 36.0 % |
| Qualified top-5 | 69.8 % | **60.6 %** |
| Mean rank | 1.91 / 21 (top 9 %) | 7.72 / 151 (**top 5 %**) |
| Mean fold enrichment | 15.8× | 15.9× |
| Candidate-set mass coverage | 20 % | **67 %** |

| Metric (production π, anchor-overlapping) | Real | Col-perm null | Src-perm null | z (vs col) | z (vs src) |
|---|---:|---:|---:|---:|---:|
| top-1 | 25.4 % | 1.1 ± 3.4 % | 2.5 ± 5.0 % | **+7.1** | +4.6 |
| top-3 | 32.8 % | 3.8 ± 6.7 % | 3.3 ± 5.2 % | +4.3 | **+5.7** |
| top-5 | 55.2 % | 7.9 ± 8.7 % | 5.5 ± 6.6 % | +5.5 | **+7.6** |

(Values shown over all 19 evaluable pairs including novel.)

**The right region is in the model's top 5 % of candidates (rank 7.72/151) — at +4–8σ above both nulls, including the strict source-permuted null.** This is a much sterner test than Beauchamp-22 (151-way vs 21-way classification), and the model passes cleanly.

**An informative negative — Motor and Auditory fall out of top-K under JuBrain.** With Beauchamp-22, the hand-curated MNI balls for "precentral gyrus" and "Heschl's gyrus" capture our published targets exactly. With JuBrain, the best Jaccard match is `jubrain_92` (Motor) and `jubrain_15` (Auditory), and the model puts most of its mass elsewhere — at JuBrain regions adjacent to but not identical with these. This is consistent with `docs/diagnostics.md`: the model doesn't have the *exact* JuBrain neighbourhood right for these regions, even though it has the gross anatomical neighbourhood (Beauchamp-22 ball) right. The Beauchamp-22 result was generous; the JuBrain result is honest.

**Mean fold enrichment is candidate-set-invariant (15.8× under both).** Fold doesn't depend on the candidate ranking — only on the mass on H_true and its size. This confirms it's the most stable headline metric.

**Soft region anchors look worse here.** Re-running the soft-atlas-region π with JuBrain candidates gives qualified top-1 = 17.5 %, top-5 = 38.4 % — substantially worse than production's 27.9 % / 60.6 %. The soft anchors concentrate mass onto the Beauchamp-22 hand-curated balls (mean fold 40.8× under Beauchamp-22) but those balls don't coincide cleanly with JuBrain regions, so under a JuBrain candidate set the concentration looks misdirected. This is the parcellation-disagreement problem: a model that's "right" under one parcellation can look "wrong" under another. For users querying π, the Beauchamp targets are the right metric if they want named-anatomical hits; JuBrain is the right metric if they want atlas-grade region identification. The model is good at the former and only fair at the latter, and that's an honest read.

### 5.11 TOPO-1 — Per-region xyz ablation (convergent negative)

`docs/diagnostics.md` argues that Motor / Superior Colliculus / Olfactory (and others) fail because the xyz cost is *signed* — mouse and human atlases have inverted dorsoventral organisation in midbrain, so the spatial prior actively pulls non-anchor parcels into the wrong cross-species neighbourhood. The natural fix: weight xyz per parcel, lower for regions where it misleads.

We tested this directly (`experiments/per_region_xyz/01_ablation.py`):

**Step 1 — global xyz=0 sanity check.** Re-fit the production model with `xyz_weight=0` everywhere (FC + SC + anchors only). Beauchamp top-1 changes per pair:

| Region | prod | xyz=0 global | Δ pp |
|---|---:|---:|---:|
| Thalamus | 33 % | 5 % | **−28** |
| Somatosensory | 20 % | 3 % | **−17** |
| Caudate | 13 % | 1 % | **−12** |
| Hypothalamus | 12 % | 4 % | **−8** |
| NAc | 8 % | 0 % | **−8** |
| ACG | 13 % | 9 % | −4 |
| Visual | 7 % | 7 % | 0 |
| Auditory | 22 % | 22 % | 0 |
| Pallidum | 5 % | 5 % | 0 |
| Pons | 3 % | 3 % | 0 |
| **Motor** | **0 %** | **4 %** | **+4** |
| **Tectum (SC)** | **0 %** | **6 %** | **+6** |
| **Piriform** | **0 %** | **13 %** | **+13** |

xyz overall is **net positive** — it helps thalamus, S1, caudate, hypothalamus, NAc, ACG much more than it hurts Motor / Tectum / Piriform. The xyz cost isn't generally a topology-inversion problem; it's a real spatial prior that works for most regions and underperforms for ~3.

**Step 2 — per-region xyz=0 for the three "helped" regions.** Use the new `xyz_weight_per_mouse_parcel` kwarg to zero xyz only for parcels nearest Garin pair_ids {2 (Motor), 11 (Piriform/Olfactory), 21 (Tectum)} — 308 of 1864 parcels. Re-fit:

| Region | prod | xyz=0 global | xyz=0 per-region (targeted) |
|---|---:|---:|---:|
| Motor | 0 % | +4 → 4 % | **0 %** (no change) |
| Tectum (SC) | 0 % | +6 → 6 % | **0 %** (no change) |
| Piriform | 0 % | +13 → 13 % | **0 %** (no change) |
| Thalamus | 33 % | -28 | 31 % (-2) |
| S1 | 20 % | -17 | 20 % (no change) |
| NAc | 8 % | -8 | 12 % (+4) |

**The per-region intervention does not reproduce the global effect on the targeted regions.** Motor / Tectum / Piriform stay at 0 % top-1 even though zeroing xyz *globally* lifts them by 4 / 6 / 13 pp. The most plausible explanation: the FGW solver finds a joint coupling π that depends on the *full* M, not just on each row independently. Zeroing xyz for some rows changes their local cost but leaves the human marginal pulled in the same direction by the other 1556 parcels' xyz contributions, so the targeted parcels end up in the same equilibrium.

**Conclusion.** Per-row xyz weighting *is not the right mechanism* for fixing topology-inverted regions in OTTER. The infrastructure (`xyz_weight_per_mouse_parcel` kwarg, `otter.data.build_xyz_weight_array` helper) is kept in the API as a general tool — and the unit tests confirm it works as intended at the cost-matrix level — but this specific application is a convergent negative. To actually fix Motor / Tectum / Piriform you would need to change the cost in a way that affects the global equilibrium (e.g. learned per-region xyz affine transforms, or removing xyz uniformly and accepting the cost of −28 pp on Thalamus). Both are research-grade interventions, not an evening's work.

The most-likely-to-help next direction is therefore *not* per-region xyz weighting but the one we already established works empirically: **more anchors in the failure regions** — specifically Motor (pair_id 2), Tectum (21), Olfactory (11), via hand-curated anatomical homologue pairs from the cytoarchitecture literature.

### 5.12 BICCN-MOTOR-1 — Motor sub-region anchors from Bakken 2021

Bakken et al. 2021 (*Nature*; the BICCN Motor Cortex Consortium) identified two conserved mouse↔human motor sub-region homologies via cross-species single-cell transcriptomics: **mouse Primary motor area (M1) ↔ human Area 4 (BA4)** and **mouse Secondary motor area (M2) ↔ human Area 6 dorsal premotor (PMd)**. We translated these into OTTER region anchors at pair_ids 30 (M1↔BA4, 53 mouse × 12 human parcels) and 31 (M2↔PMd, 48 mouse × 23 human parcels). Mouse-side sets come from the DSURQE atlas overlay (same source as the Beauchamp validation); human-side sets come from MNI spheres around the canonical cytoarchitectural centroids (Mayka 2006, Glasser HCP-MMP360).

Helper: ``otter.data.atlas_regions.build_biccn_motor_region_anchors(M.var, H.var)``.

Experiment: ``experiments/biccn_motor/01_add_motor_subregion_anchors.py``.

#### 5.12.1 What changes vs production point-anchor π

| Beauchamp pair | prod top-1 | +BICCN top-1 | Δ | top-5 prod → +BICCN |
|---|---:|---:|---:|---:|
| Primary motor area → precentral gyrus | 0 % | **100 %** | **+100** | 2 % → **100 %** |
| Striatum ventral → NAc | 8 % | 12 % | +4 | 38 % → 38 % |
| Primary somatosensory → postcentral | 20 % | 21 % | +1 | 41 % → 39 % |
| All other anchor-overlapping Beauchamp pairs (12) | — | — | unchanged within ±1 pp | — |

Region-level eval (Beauchamp-22 candidate set) on the new π:

| Metric (anchor-overlapping, 15 pairs, 927 parcels) | Production | + BICCN motor |
|---|---:|---:|
| Qualified top-1 | 37.2 % | **53.9 %** |
| Qualified top-3 | 69.8 % | **75.5 %** |
| Mean fold enrichment | 15.8× | 18.9× |

Motor itself goes from "rank 1 with 0 % mass" (vacant — qualified=NO) under production to "rank 2 with 100 % mass and 46× fold enrichment" (qualified=YES) under the BICCN π.

#### 5.12.2 Honest caveat — the 100 % is partly tautological

The mouse-side set for the BICCN M1 anchor is the **same 53-parcel DSURQE "Primary motor area" set** that Beauchamp's "Primary motor area → precentral gyrus" validation uses. The human-side 12-parcel BA4 set is a **strict subset** of Beauchamp's 45-parcel precentral mask. So the soft anchor (lam_outside=0.15) forces ~all mass from those 53 mouse parcels into a subset of the validation target — and Beauchamp top-1 is then 100 % by anatomy, not by independent FC/SC recovery.

**Held-out generalization test**: re-fit with M2 anchor only (M1 anchor held out). Beauchamp Motor top-1 stays at 0 %. So **structure does not recover M1 ↔ BA4 from FC + SC + Garin anchors + M2**. The 100 % is the anchor's contribution, not the model's.

#### 5.12.3 What this is and isn't

**This is** a real practical improvement. Downstream users querying OTTER for "where does mouse M1 map to?" now get a defensible BA4-centred answer rather than a near-uniform misdirected distribution. The mechanism is clean (region anchors via the existing `region_anchors` API), the citation is solid (Bakken 2021 BICCN consortium published cell-type-conserved correspondences), and the off-target cost is small (-2 pp on S1 top-5, otherwise stable).

**This isn't** evidence that OTTER's FC + SC structure encodes motor cross-species biology. The held-out test cleanly shows the opposite: motor is unrecoverable without the explicit anchor. So the path forward for unanchored motor sub-regions (or any region without published correspondences) remains: more curated anchors. Mouse M1 ↔ human BA4 is now a published, anatomically-grounded, parcellation-aligned anchor — usable as a building block for any user who wants trustworthy motor-region queries.

**Implications for further work**: the same mechanism applies to Tectum (Stein 2009 cross-species optic tectum/SC), Olfactory (Mori 2014 piriform layers), Cingulate sub-areas (Vogt 2019). Each one needs a published correspondence + an MNI centroid + ~30 minutes of curation. We deliberately did not anchor every Garin region this way; the goal is to make OTTER queryable in a few key regions, not to exhaustively over-supervise.

### 5.13 TECTUM-1 — Superior + Inferior Colliculus anchors

The tectum is one of OTTER's most-documented failure regions — both Superior Colliculus (SC) and Inferior Colliculus (IC) sit at 0 % Beauchamp top-1 under the production point-anchor π, with mean ranks of 1872 / 2094 (SC) and 1431 / 2094 (IC). ``docs/diagnostics.md`` attributes this to spatial inversion (mouse SC is dorsal, human SC is ventral in MNI space) and lack of finer sub-region supervision than the single Garin Tectum anchor at pair_id 21.

We added a tectum anchor pack with two entries (``otter.data.anchor_packs.tectum``):

  - pid 32: Mouse Superior Colliculus sensory (53 parcels) ↔ Human SC at MNI(±5, -30, -2) r=6 mm (2 parcels)
  - pid 33: Mouse Inferior Colliculus (29 parcels) ↔ Human IC at MNI(±5, -35, -8) r=8 mm (4 parcels)

Mouse-side sets come from the DSURQE atlas overlay; human-side centroids are from Mai/Paxinos and standard SC/IC anatomy (May 2006 *Vision Research*; Schreiner & Winer 2007 *Trends in Neurosciences*). The human balls are deliberately tight because the colliculi are small structures and broader radii would capture unrelated midbrain parcels.

#### 5.13.1 What changes

| Beauchamp pair | prod top-1 | + tectum top-1 | Δ | + SC only (held-out IC) |
|---|---:|---:|---:|---:|
| Superior Colliculus → SC | 0 % | **100 %** | +100 | 100 % |
| Inferior Colliculus → IC | 0 % | **100 %** | +100 | **0 %** ← honest |
| Thalamus → thalamus | 33 % | 29 % | -4 | 30 % |
| All other 12 anchor-overlapping Beauchamp pairs | — | unchanged within ±1 pp | — | — |

Same caveat as §5.12: 100 % is largely tautological. The mouse-side SC and IC sets are *identical* to Beauchamp's validation sets, and the human-side balls overlap the validation balls. The honest test is the **held-out IC column**: when we fit with only the SC anchor and check IC, IC stays at 0 %. So structure + 21 Garin anchors + the SC region anchor do *not* propagate to IC — IC homology is unrecoverable without its own anchor.

#### 5.13.2 Same conclusion, generalised

Combined with §5.12, the picture across two anchor packs is consistent:

- **Region anchors do what they say**: anchored regions become trustworthy for downstream queries (100 % top-1 by construction).
- **Off-target cost is small**: thalamus -4 pp here, S1 -2 pp top-5 for BICCN motor. Everything else stable.
- **Held-out generalization is null**: dropping one of the two anchors in either pack gives 0 % top-1 for the held-out region. Structure does not bridge across un-anchored sub-regions.
- **Implication for future packs**: every region we want OTTER to handle reliably needs its own pack. The cost is ~30 minutes per region (find canonical MNI centroid, validate the parcel set is non-empty, add the helper). The benefit is bounded but real — a trustworthy answer for that specific region.

The pid registry is now:

| pid range | pack |
|---|---|
| 1..21 | Garin point anchors |
| 30, 31 | BICCN motor (M1, M2) |
| 32, 33 | Tectum (SC, IC) |
| 34, 35 | Olfactory (Piriform, AON) |

Future packs reserve ≥ 36. See ``otter.data.anchor_packs/__init__.py`` for the registry and the "adding a new pack" recipe.

### 5.14 OLFACTORY-1 — Piriform + Anterior Olfactory Nucleus

Piriform cortex (primary olfactory cortex) is OTTER's third documented failure region — 0 % Beauchamp top-1 under production point-anchor π, mean rank 657 / 2094. Cross-species olfactory homology is among the most conserved in mammalian neuroanatomy (Mori 2014 *The Olfactory System*; Carlén 2017 *Science*). The olfactory anchor pack (``otter.data.anchor_packs.olfactory``) adds:

  - pid 34: Mouse Piriform area (47 parcels via DSURQE) ↔ Human Piriform cortex at MNI(±25, 5, -20) r=10 mm (13 parcels)
  - pid 35: Mouse Anterior olfactory nucleus (9 parcels) ↔ Human AON at MNI(±15, 25, -15) r=10 mm (6 parcels)

#### 5.14.1 What changes

| Beauchamp pair | prod top-1 | + olfactory top-1 | Δ | + Piriform only |
|---|---:|---:|---:|---:|
| Piriform area → piriform cortex | 0 % | **100 %** | +100 | 100 % |
| All other 13 anchor-overlapping Beauchamp pairs | — | unchanged within ±0 pp | — | unchanged |

The cleanest result among the three packs: **zero off-target cost on any other Beauchamp pair**. The olfactory cortex is anatomically segregated enough from motor / sensory / midbrain / thalamic regions that the new constraint doesn't compete for mass anywhere else. Compare to BICCN motor (S1 top-5 −2 pp) and Tectum (Thalamus −4 pp) which both had small but real off-target effects.

Same tautology caveat as §5.12-§5.13: the Piriform 100 % is by construction because the mouse-side set used here is identical to Beauchamp's validation set and the human-side ball overlaps Beauchamp's. AON is not in Beauchamp's 22 pairs so the AON entry's effect cannot be directly measured against Beauchamp — its inclusion is justified by anatomical homology, not by Beauchamp-recovery.

The held-out test (Piriform anchor only, AON omitted) reports 100 % on Piriform — but this is *not* a generalization test because Piriform was kept; the only thing varying is AON, which Beauchamp doesn't validate. So this pack doesn't have a clean held-out signal. Future olfactory work would need an independent validation source (e.g. olfactory-specific cell-type homology from BICCN extensions).

#### 5.14.2 Three packs, consistent picture

| Pack | Direct gain | Off-target cost | Held-out structure recovery |
|---|---|---|---|
| BICCN motor (§5.12) | Motor 0 → 100 % | S1 top-5 −2 pp | None (M1 not recovered without anchor) |
| Tectum (§5.13) | SC + IC 0 → 100 % | Thalamus −4 pp | None (IC not recovered with SC alone) |
| Olfactory (§5.14) | Piriform 0 → 100 % | **None** | Not measurable (AON not in Beauchamp) |

All three confirm the same lesson: **region anchors deliver trustworthy queries by construction; they do not encode independent cross-species recovery**. The picture is honest and the cost-benefit consistent — each pack costs ~30 minutes of curation and delivers reliable downstream queries for one failure region with near-zero side effects.

### 5.15 HIPPOCAMPAL-1 — Subiculum + CA1 + CA3 + Dentate gyrus

Hippocampal subfields are OTTER's cleanest documented failure region: all four (Subiculum, CA1, CA3, Dentate) sit at 0 % Beauchamp top-1 under production point-anchor π. Earlier work (EXP-1 / SPLIT-1, §5.2) added four hippocampal *point* anchors and moved 3 of 4 from 0 → 7–9 % top-1. The hippocampal pack (``otter.data.anchor_packs.hippocampal``) is the region-anchor analogue, forcing each subfield's full mouse parcel set into the matching human subfield MNI ball:

  - pid 39: Subiculum (29 mouse / 8 human)
  - pid 40: CA1 (15 / 6)
  - pid 41: CA3 (26 / 4)
  - pid 42: Dentate gyrus (22 / 4)

Mouse-side via DSURQE; human-side via small Iglesias 2015 hippocampal-subfield MNI centroids.

#### 5.15.1 What changes

| Beauchamp pair | prod | +hippocampal | Δ | +Subi only (held-out) |
|---|---:|---:|---:|---:|
| Subiculum | 0 % | **100 %** | +100 | 100 % |
| CA1 | 0 % | **100 %** | +100 | **0 %** ← held-out |
| CA3 | 0 % | **100 %** | +100 | **0 %** ← held-out |
| Dentate gyrus | 0 % | **100 %** | +100 | **0 %** ← held-out |
| All 14 non-hippocampal Beauchamp pairs | — | unchanged within ±1 pp | — | unchanged |

Same tautology caveat as the earlier packs. The held-out columns confirm structure does NOT propagate across hippocampal subfields: anchoring Subiculum alone leaves CA1, CA3, and Dentate at 0 %. This contradicts a naïve "anchored hippocampus generalises to subfields" hypothesis — each subfield needs its own anchor.

This is the largest single pack (4 entries vs 2 in the others) and lifts the most validation pairs (4 vs 1–2). Total off-target cost: Thalamus +1 pp ripple, everything else stable. Cost-benefit is excellent.

### 5.16 CINGULATE-1 — Subgenual ACC + Retrosplenial (Vogt 2019)

The cingulate pack (``otter.data.anchor_packs.cingulate``) anchors two best-conserved cingulate sub-domains identified by Vogt (2019):

  - pid 36: Mouse ACA ventral (15 parcels) ↔ Human subgenual ACC at (±5, 10, 35) r=10 mm (6 parcels)
  - pid 37: Mouse Retrosplenial (27 parcels) ↔ Human RSC at (±15, –55, 10) r=10 mm (8 parcels)

We deliberately use *subgenual* ACC rather than pregenual ACC because the pregenual MNI ball (±5, 25, 25) overlaps the human mPFC parcel anchored by Garin pair_id 1 — adding a region anchor there would conflict with the existing supervision.

#### 5.16.1 What changes — first pack where region anchors *hurt* a Beauchamp metric

| Beauchamp pair | prod | +cingulate | Δ | +RSC only |
|---|---:|---:|---:|---:|
| Anterior cingulate area → cingulate gyrus | 13 % | **9 %** | **−4** | 13 % |
| All other 14 anchor-overlapping pairs | — | unchanged | — | unchanged |

**The 4-pp drop is the cleanest signal in the codebase that Beauchamp validation is target-specific, not region-specific.** Beauchamp's "cingulate gyrus" validation ball is centred at (±5, 25, 25) — *pregenual* ACC. Our anchor pulls mass towards subgenual ACC at (±5, 10, 35). Anatomically both are part of cingulate gyrus and both are part of Vogt's "ACC" domain, but Beauchamp's argmax-distance metric punishes mass that lands in subgenual rather than pregenual. So this is not the pack hurting the model — it's the validation measuring a different sub-region than the anchor targets.

The held-out column (RSC anchor only, ACC omitted) confirms this: ACG stays at 13 %, so the drop is fully attributable to the ACC subgenual anchor and not to RSC or to a global side effect.

#### 5.16.2 What this means for the manuscript narrative

The cingulate pack is the first case where the standard "anchor → +100 %" pattern doesn't apply, because the anchor target and the validation target are different sub-regions of the same coarse "cingulate" anatomical area. Two readings:

1. **The pack is broken — don't ship it.** If the only goal is to maximise Beauchamp top-1, drop this pack. Currently the production point-anchor π gives ACG = 13 %, which beats the cingulate pack's 9 %.

2. **The pack is right and the validation is the problem.** If a downstream user actually wants to map mouse ACA-ventral → human subgenual ACC (which is the literature-supported correspondence), the cingulate pack delivers that. The Beauchamp validation pair is testing pregenual ACC mapping; that's a different question. We ship the pack documented as such, and let users choose.

We've taken position (2): the cingulate pack is part of the codebase but is *not* enabled by default in the "compose all packs" recipe. Users opt in explicitly.

### 5.17 AMYGDALA-1 — Cortical subplate / amygdala (closing pack)

Beauchamp pair "Cortical subplate-other → amygdala" was the last 0 % top-1 failure region without dedicated sub-region supervision. Cross-species amygdala homology is uncontroversial (Janak & Tye 2015 *Nature*; Pessoa & Adolphs 2010 *Nature Reviews Neuroscience*). The amygdala pack (``otter.data.anchor_packs.amygdala``) is a single-entry pack — DSURQE doesn't distinguish basolateral / central / lateral amygdaloid nuclei, so we use the broader "Cortical subplate" set that Beauchamp itself uses:

  - pid 38: Mouse Cortical subplate (54 parcels) ↔ Human amygdala at MNI(±25, –5, –20) r=8 mm (6 parcels)

#### 5.17.1 What changes

| Beauchamp pair | prod | +amygdala | Δ |
|---|---:|---:|---:|
| Cortical subplate-other → amygdala | 0 % | **100 %** | +100 |
| All 18 other Beauchamp pairs | — | unchanged within ±0 pp | — |

Cleanest result so far — zero off-target cost, same tautological 100 % by construction. The pack is a single entry so there's no internal held-out test possible. The mouse Cortical subplate set matches Beauchamp's validation set 1:1.

**Composition caveat**: the amygdala human ball at (±25, –5, –20) r=8 mm overlaps 2 parcels (L/R Olfactory cortex) with the olfactory pack's piriform ball at (±25, +5, –20) r=10 mm. When both packs are composed together, those 2 shared parcels receive conflicting soft constraints — the FGW solver handles this, but mass on those 2 specific parcels will be intermediate rather than fully concentrated on either target.

### 5.18 Pid registry (final state — 6 packs)

| pid range | pack | source |
|---|---|---|
| 1..21 | Garin point anchors | Garin 2021 |
| 30, 31 | BICCN motor (M1, M2) | Bakken 2021 |
| 32, 33 | Tectum (SC, IC) | May 2006; Schreiner 2007 |
| 34, 35 | Olfactory (Piriform, AON) | Mori 2014 |
| 36, 37 | Cingulate (subgenual ACC, RSC) — **opt-in** | Vogt 2019 |
| 38 | Amygdala (Cortical subplate) | Janak & Tye 2015 |
| 39, 40, 41, 42 | Hippocampal (Subi, CA1, CA3, DG) | Strange 2014 |
| ≥ 43 | (reserved for future packs) | |

#### 5.18.1 Six packs, consistent picture

| Pack | Lifts validation pairs | Off-target | Held-out generalises? |
|---|---|---|---|
| BICCN motor | Motor 0→100 % | S1 top-5 −2 pp | No |
| Tectum | SC + IC 0→100 % | Thalamus −4 pp | No |
| Olfactory | Piriform 0→100 % | None | Not measurable |
| Cingulate (opt-in) | ACG **13→9 %** | None | Yes — anchor ≠ validation target |
| Amygdala | Amg 0→100 % | None | N/A (single entry) |
| Hippocampal | Subi/CA1/CA3/DG 0→100 % (4 pairs) | Thalamus +1 pp | No (per-subfield) |

**Coverage**: with all default packs composed (everything except opt-in cingulate), every 0 % Beauchamp failure pair is now anchored at sub-region level. The aggregate effect on Beauchamp top-1 for anchor-overlapping pairs goes from 12 % (production point-anchor) to a substantial improvement, weighted by the parcel counts of the newly-anchored regions.

**Generalised conclusion across 14 entries / 6 packs**: anchored regions become trustworthy by construction; structure does NOT propagate across un-anchored sub-regions; each region we want OTTER to handle reliably needs its own pack entry. The cingulate finding (where anchor and validation target different things) adds nuance: the Beauchamp metric is a noisier validation target than its 22-pair design suggests, and disagreements between anchor and validation can produce informative metric drops rather than indictments of the anchor.

### 5.19 COMPOSE-ALL — full default-pack π

The headline result of the anchor-pack series. Fit ``MultimodalFGW`` with all 5 default packs (everything except opt-in cingulate) composed on top of the 21 Garin point anchors. 11 region-anchor entries total. Experiment in ``experiments/compose_all/01_compose_all_default_packs.py``; saved π in ``outputs/coupling/pi_fc_plus_SC_with_all_packs.npy``.

#### 5.19.1 Beauchamp aggregate (anchor-overlapping, 15 pairs / 927 parcels)

| Metric | Production (point anchors) | + all default packs | Δ |
|---|---:|---:|---:|
| top-1 | 11.7 % | **36.5 %** | **×3.1** |
| top-5 | 22.2 % | **45.7 %** | ×2.1 |
| top-10 | 26.9 % | **49.6 %** | ×1.8 |
| Mean rank (of 2 094) | 871 | **85** | **×10 (smaller is better)** |

#### 5.19.2 Region-level eval (Beauchamp-22 candidate set, anchor-overlapping)

| Metric | Production | + all default packs |
|---|---:|---:|
| Qualified top-1 | 37.2 % | **82.2 %** |
| Qualified top-3 | 69.8 % | **100.0 %** |
| Qualified top-5 | 69.8 % | **100.0 %** |
| Mean fold enrichment | 15.8× | **122.7×** |

For the 15 supervised pairs, *every* one is now in the model's top-3 region candidates with fold ≥ 1. Mean fold enrichment up by 7.7×.

#### 5.19.3 Per-pair table

| Pair | prod | + all packs | Δ | Pack-anchored? |
|---|---:|---:|---:|---|
| Motor → precentral | 0 % | **100 %** | +100 | ✓ BICCN |
| Superior Colliculus | 0 % | **100 %** | +100 | ✓ Tectum |
| Inferior Colliculus | 0 % | **100 %** | +100 | ✓ Tectum |
| Piriform → piriform | 0 % | **100 %** | +100 | ✓ Olfactory |
| Amygdala | 0 % | **100 %** | +100 | ✓ Amygdala |
| Subiculum | 0 % | **100 %** | +100 | ✓ Hippocampal |
| CA1 | 0 % | **100 %** | +100 | ✓ Hippocampal |
| CA3 | 0 % | **100 %** | +100 | ✓ Hippocampal |
| Dentate gyrus | 0 % | **100 %** | +100 | ✓ Hippocampal |
| Thalamus | 33 % | 30 % | **−3** | (no pack) |
| Somatosensory → postcentral | 20 % | 19 % | **−1** | (no pack) |
| Caudate | 13 % | 13 % | −1 | (no pack) |
| All other 6 anchor-overlapping pairs | unchanged ±0 | | | |

**Off-target cost summary**: −5 pp summed across 3 non-pack regions. Compare to ~+900 pp summed across 9 pack-anchored regions. Net cost-benefit is overwhelmingly positive even when the headline gains are construction artefacts.

#### 5.19.4 Composition overlaps (audit log)

The fit produced 7 pid-pair overlaps where two packs share at least one human parcel:

| pids | overlap | reason |
|---|---:|---|
| 34 (Piriform) ∩ 38 (Amygdala) | 3 parcels | adjacent ventral anatomy (piriform / amygdala interface) |
| 39, 40, 41, 42 (Hippocampal) pairwise | 2-4 parcels each | adjacent subfields in tight MNI ball |

The FGW solver handles these (soft constraints, not hard walls). Mass on shared parcels ends up intermediate between the two anchor targets. The above Beauchamp / region-level numbers are computed under these compositional constraints, so they represent the actual deployable π.

#### 5.19.5 What this π is and isn't

**Is**: a deployable mouse↔human coupling that returns trustworthy human partners for 9 previously-zero-recovery Beauchamp regions plus the 6 regions already at non-zero top-1. Mean rank 85 / 2094 means the right human partner is in the top 4 % of candidates on average. For downstream users querying π for any of the 11 anchored sub-regions, this is the recommended π.

**Isn't**: an unsupervised demonstration that OTTER's FC + SC structure recovers cross-species homology. Every per-region 100 % is by-construction, and per-region held-out tests (e.g. §5.13 Tectum's IC) confirm that structure does not propagate across un-anchored sub-regions. The validation gain is real because the *anchor sources* are credible (published cross-species correspondences); it's not a methodological claim about the FGW solver's recovery power.

This is the right π for the "best mouse↔human mapping we can deliver with current evidence" use case. Multi-source validation against Mars 2018 / Coletta 2020 / BICCN cell-types is the next step beyond this.

### 5.20 LATERAL-PFC — non-Beauchamp coverage (opt-in OFC + dlPFC)

The first pack added for regions with no direct Beauchamp validation pair — anchored on published cross-species cytoarchitecture and connectivity correspondences for downstream use cases that need lateral PFC predictions (decision-making, working memory, executive control, reward).

  - pid 45: Mouse Orbital area lateral (21 parcels) ↔ Human OFC BA11/47 at (±25, 35, –15) r=10 mm (8 parcels) — Wallis 2012 *Nat Rev Neurosci*. High confidence.
  - pid 46: Mouse Prelimbic area (11 parcels) ↔ Human dlPFC BA9/46 at (±40, 25, 35) r=10 mm (12 parcels) — Carlén 2017 *Science*. **Contested**: Preuss 1995 argues rodents lack proper dlPFC; Carlén / Laubach argue functional homology. Users who consider PL non-homologous to dlPFC should exclude this entry.

Impact: zero off-target effect on any Beauchamp pair (the lateral PFC regions don't overlap any Beauchamp validation target). The pack is purely additive — anchored regions become queryable; nothing else changes.

**Insular pack — not viable**. We investigated adding anterior vs posterior insula but DSURQE doesn't expose sub-divisions of mouse Agranular insular area at the resolution we'd need (no "ventral/posterior part" entries). The single Garin pid 9 (Insula) anchor remains the only insular supervision available.

### 5.21 Multi-source trust map and network coherence

The Beauchamp top-1 metric is one validation source. The "best mouse↔human mapping" framing requires combining multiple signals to estimate, per-parcel and per-network, where the mapping is trustworthy and where it isn't.

#### 5.21.1 Multi-source trust map (v1, ``otter.eval.compute_multisource_trust``)

Augments the internal-signal trust score (``compute_trust_score``: bootstrap stability + argmax concentration + FC similarity to nearest anchor) with two external signals:

- **Anchor presence**: parcel is in the Garin 42 OR in any region-anchor pack's ``mouse_indices``.
- **Beauchamp region validation**: parcel sits in a Beauchamp validation pair whose top-1 > 0.

Combined into 5 evidence tiers. Distribution on the all-packs π over 1864 mouse parcels:

| Tier | n_parcels | % | Interpretation |
|---|---:|---:|---|
| anchored_and_validated | 354 | 19.0 % | Pack-anchor + Beauchamp top-1 > 0 — highest confidence |
| anchored_only | 65 | 3.5 % | Pack-anchor but no Beauchamp validation (lateral PFC, AON, RSC subgenual) |
| validated_only | 665 | 35.7 % | Beauchamp top-1 > 0 but no specific anchor — coverage by FC structure + nearby Garin |
| structural | 233 | 12.5 % | High internal trust (bootstrap + concentration + FC) but no anchor and no Beauchamp |
| low_evidence | 547 | 29.3 % | No supervision and weak internal signal — use with caution |

This is the right trust map to ship: it tells users *why* to trust each parcel rather than collapsing everything into a single composite score. The 19 % anchored_and_validated parcels are the strictest "we have multi-source evidence" set; the 35.7 % validated_only is the next tier; low_evidence parcels (~30 %) should be flagged as out-of-distribution.

#### 5.21.2 Network coherence (v2, ``otter.eval.network_compactness``)

Coletta 2020 (*Sci Adv*) identified ~7 mouse functional networks. A good mouse↔human mapping should preserve these — within-network mouse parcels should map to a *compact* region in human space. We measure compactness as median pairwise distance and mean centroid spread.

Per-network compactness for production π vs all-packs π:

| Network | n_mouse | Prod median pairwise (mm) | +all packs median (mm) | Δ |
|---|---:|---:|---:|---:|
| **olfactory** | 72 | 64.5 | **46.8** | **−17.7** ← biggest gain |
| **limbic** | 364 | 70.4 | **57.6** | **−12.8** |
| temporal_dmn | 81 | 90.4 | 83.5 | −6.9 |
| salience | 117 | 56.9 | 54.0 | −2.9 |
| subcortical | 408 | 46.8 | 45.9 | −0.9 |
| sensorimotor | 230 | 48.5 | 52.5 | +4.0 |
| brainstem | 242 | 41.2 | 45.9 | +4.7 |
| frontal_dmn | 87 | 32.4 | 37.1 | +4.7 |
| frontoparietal | 115 | 31.0 | 37.1 | +6.1 |
| auditory | 65 | 65.4 | 72.0 | +6.6 |
| visual | 83 | 54.0 | 64.3 | +10.3 |

Pattern: adding packs *substantially improves* network coherence for olfactory and limbic (the networks directly covered by our packs — Piriform / AON for olfactory, Hippocampal subfields + Amygdala for limbic), and *mildly fragments* networks where packs anchor only specific sub-regions (visual, auditory, sensorimotor — these have a Garin point anchor but no region pack, so the new anchored regions pull mass into specific spots, leaving the rest of the network distributed). This is the same competition-for-mass dynamic we saw at the per-pair level — net positive for anchored regions, mildly negative for unanchored adjacent ones.

The pack-induced limbic compactness improvement (−12.8 mm) is the cleanest non-Beauchamp signal that the anchor packs encode something real about brain anatomy: the hippocampal + amygdala + olfactory pack collectively map mouse "limbic" parcels into a coherent human limbic region.

#### 5.21.3 What v2 doesn't include — Mars 2018 and BICCN cell types

Two further external validation sources we identified earlier but could not integrate:

- **Mars 2018 / Folloni 2019** (cross-species white-matter homologies, human↔macaque): the supplementary tables aren't directly downloadable via web search; manual extraction from the paper would be required. Worth pursuing as a discrete follow-up if you want a fully external transitive validation.
- **BICCN cell-type composition at parcel level**: Yao 2023 / Siletti 2023 give cross-species cell-type taxonomies but integration with our 1864 × 2094 parcellation requires non-trivial alignment work.

Neither is required to claim "best mapping" — multi-source trust + network coherence already give us evidence layered beyond just Beauchamp top-1. They're listed as concrete future work in the roadmap.

### 5.22 Pid registry (final, 7 packs)

| pid range | pack | status |
|---|---|---|
| 1..21 | Garin point anchors | default |
| 30, 31 | BICCN motor (M1, M2) | default |
| 32, 33 | Tectum (SC, IC) | default |
| 34, 35 | Olfactory (Piriform, AON) | default |
| 36, 37 | Cingulate (subgenual ACC, RSC) | **opt-in** (hurts Beauchamp ACG) |
| 38 | Amygdala | default |
| 39-42 | Hippocampal (Subi, CA1, CA3, DG) | default |
| 45, 46 | Lateral PFC (OFC, dlPFC) | **opt-in for dlPFC** |
| ≥ 47 | (reserved for future packs) | — |

15 region-anchor entries total (default + opt-in), covering 13 distinct named brain regions beyond the 21 Garin point anchors.

---

## 6. Comparative methods (kept as additions, neither moves the headline)

### 6.1 FUGW (`otter.models.FUGWModel`)

Drop-in alternative using the [Thual et al. 2022 NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/8906cac4ca58dcaf17e97a0486ad57ca-Abstract-Conference.html) unbalanced FGW solver via the `fugw` PyPI package. `pip install fugw torch` to use it.

| Model | Restricted top-1 (visual held-out) | Full top-1 | Mean rank /2094 | Row max conc. | Uncovered humans |
|---|---|---|---|---|---|
| `MultimodalFGW` (semirelaxed) | 50% | 0% | 682 | 0.977 | 762 (36%) |
| `FUGWModel` (rho_s=rho_t=1) | 50% | 0% | 511 | **0.077** | **0** |

FUGW gives a genuinely soft π with full human coverage (0 uncovered) but **does not improve anchor identification**. Useful if a downstream task wants probabilistic mass with full coverage; not a replacement for production.

### 6.2 Knox 2019 leaf-level cortical SC

Augments our SC by replacing 22 cortical anchor SC fingerprints with Knox's voxel-resolved cortical leaves (43 leaves). Cost-matrix unique fingerprints: Allen 454 → Knox-augmented 469 (+1.03×). All 11 networks (n=42 LONO folds):

| Aggregate metric | Allen | Knox |
|---|---|---|
| Full top-1 | 2.4% | 2.4% |
| Full top-5 | 11.9% | 11.9% |
| Mean rank /2094 | 205.9 | 205.5 (Δ −0.4 = noise) |
| `frac_argmax_is_anchor` | 4.8% | 4.8% |

**Statistically indistinguishable.** The 1.03× cost-matrix resolution gain doesn't translate into anchor identification.

### 6.3 What this rules out

Combined with the M_anchor / iterative co-clustering / confidence-weighted-FC negatives (items A/B/C), the convergent picture is:

- **Modality data** is not the bottleneck (Knox SC ≈ Allen SC; gene M actively hurts).
- **Solver formulation** is not the bottleneck (FUGW ≈ semirelaxed on identification).
- **Anchor density IS the bottleneck** — confirmed by the hippocampal anchor expansion (0% → 7-9% for 3/4 pairs).

---

## 7. Per-experiment notes (worked / failed)

### Methodology improvements that worked
- **`fc + xyz_M`** (vs FC only) — top-1 79% → 81%. xyz spatial prior in M is the cheapest +2pp we found.
- **Multistart sanity** — loss spread across 6 diverse inits is < 1e-6 nats; FGW is well-identified given anchors + xyz.
- **Hierarchical** — best within-network FC translation (r=0.55 vs 0.45 flat) at the cost of LONO CV.
- **Supplementary anchors** (Phase S in roadmap) — moves Beauchamp-failing pairs from 0% to 4-9% top-1 without disturbing existing anchors.

### Methodology improvements that failed (clean negatives)

| Item | What | Why it failed |
|------|------|---------------|
| **A** | Anchor-relationship M cost | Once leak-fixed, hurts CV by ~10pp. The 32 visible anchors' FC patterns aren't enough to predict held-out anchors better than xyz. |
| **B** | Iterative co-clustering | π is already 97.8% concentrated after the first solve — no information to recycle. |
| **C** | Confidence-weighted FC | Mouse `n_obs` is uniform, human is 99.97% correlated with the unweighted version. Structural no-op. |
| **M_gene / selective M_gene** | Cross-species gene cosine cost | Helps visual/sensorimotor a bit but tanks subcortical (100% → 20%). |
| **FUGW** | Unbalanced FGW formulation | Gives soft π with full coverage but doesn't improve anchor identification. |
| **Knox SC** | Voxel-level cortical SC | 1.03× cost-matrix resolution gain ≈ noise; doesn't affect recovery. |

---

## 8. What this all means

The production model is **interpolating within the support of the 21 Garin anchor pair_ids** in the joint mouse×human FC/SC manifold. Five lines of evidence converge:

1. **Multiple methodology variants converge to 79–81%** restricted top-1.
2. **Hard regions are the same across all configs** (brainstem, subcortical, salience, sensorimotor, visual all bottleneck at 25–60%).
3. **z = +17.8 vs permuted-anchor null** says the supervision is genuinely informative.
4. **Bootstrap stability 97.8%** — solution is stable under subject sampling.
5. **External validation (Beauchamp): 11.8× chance enrichment for anchored regions, 0× for novel.** Direct demonstration that the model captures real biology where supervised, nothing where not.

To break past 81% restricted top-1 requires either (i) **more anchors** (Phase S — already moved hippocampal pairs from 0% → 7-9%), or (ii) a fundamentally different framework (Phase V — comparison vs spectral pipeline).

Open follow-ups: external validation against more cross-species reference sets (Mars 2018 et al.); comparison vs alternative cross-species frameworks (e.g. spectral / connectivity-blueprint methods); methods writeup. See `docs/dev/roadmap_history.md` (local-only) for the development backlog.
