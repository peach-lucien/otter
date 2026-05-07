# Results

Headline numbers and per-experiment notes. The raw tables live in
`outputs/comparison/comprehensive_table.csv`; re-run
`pipeline/07_build_artefacts.py` to regenerate them after any new evaluation.

> **Caveats — read these first.**
>
> 1. The "top-1" column below is **restricted-anchor ranking accuracy** (argmax among held-out anchor columns only). Full-space top-1 (argmax over all 2094 human nodes) is **2.4%**, mean rank **206/2094**. The model reliably *ranks the correct anchor first among held-out anchor candidates*; it does NOT reliably pick the correct anchor as the global argmax.
> 2. The 4 best configs (`fc_only`, `fc_plus_xyz_gw`, `fc_plus_network_mask`, `fc_plus_SC`) differ by ≤1 of 42 anchors. McNemar p ≈ 1.00 between adjacent configs — **statistically tied**.
> 3. FC translation r = 0.36 is **in-sample**; held-out subject-CV is **0.32 ± 0.006**.
> 4. Bootstrap stability is 97.8% (40-iter, fc_plus_SC).
> 5. **External validation against Beauchamp 2022** gives 11.8× chance enrichment for anchored regions, 0× for novel — the cleanest demonstration that the model captures real cross-species biology where supervised but cannot generalise to unanchored anatomy. Details below.

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

The supplementary anchors above (SPLIT-1, EXP-1) are still **point-to-point**: one mouse parcel ↔ one human parcel. Garin's anchors and Beauchamp's region pairs are natively *region* objects (sets of parcels), so we built a region-anchor mechanism (`homer.data.region_anchors`): each mouse parcel in a declared region is allowed to map to *any* human parcel in the matching declared region, forbidden elsewhere.

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

`homer.data.supplementary_anchors` (point) and `homer.data.region_anchors` (region) both let you promote existing non-anchor parcels to anchors with new pair_ids, without modifying the underlying FC matrix or atlas. The point form goes via `MultimodalFGW.fit(M_aug, H_aug, ...)` after augmenting the var; the region form goes via `MultimodalFGW.fit(..., region_anchors=[entry, ...])` directly. Tests in `tests/test_supplementary_anchors.py` (6) + `tests/test_region_anchors.py` (7).

### 5.5 S7 — Region anchors via real atlas labels (no fabricated centroids)

`homer.data.atlas_regions.build_garin_region_anchors_from_atlases(M.var, H.var)` builds region anchors using **only published atlases**:
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

---

## 6. Comparative methods (kept as additions, neither moves the headline)

### 6.1 FUGW (`homer.models.FUGWModel`)

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
