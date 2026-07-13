# Results

Every number on this page is read from a JSON in `outputs/logs/`, written by the script that computed it. The notebooks in `notebooks/` recompute each headline value and **assert** it against that log. If you find a number here that a notebook does not reproduce, the notebook is right and this page is a bug — please open an issue.

---

## The organising claim

**π is a connectional correspondence: connectional organisation transfers through it; microstructure does not.**

π is fitted on functional and structural connectivity. What it was made of travels through it. What it never saw does not. Every result below is an instance of this, including the failures — and the failures are load-bearing, because a coupling that transferred *everything* would just be a smoothing operator.

## Synthesis

HOMER produces a probabilistic mouse↔human parcel coupling π (1,864 × 2,094) by semi-relaxed Fused Gromov–Wasserstein optimal transport, supervised on 21 Garin homology classes plus 26 region-anchor entries from 15 curated packs. The mouse marginal is fixed; the human marginal is free, which is what allows the coupling to say that a human parcel has **no** mouse counterpart.

Six findings, one per figure:

1. **The coupling is calibrated.** Sharp (top partner > 0.5 probability for 92 % of mouse parcels), homology-respecting (0.26 mean self-mass vs 0.048 uniform), topographically faithful (r = 0.61), and confidence-graded by evidence external to the solver.
2. **Connectivity and space carry *which region*; curation carries *which parcel*.** Two metrics that move independently under ablation, and the distinction survives withholding every anchor.
3. **Connectional organisation transfers; microstructure does not.** Networks and the principal gradient clear a spin null; myelin and cytoarchitecture do not.
4. **Each method wins on the modality it encodes.** Against TransBrain, a transcriptomic translator: it leads region identity, HOMER leads everything connectional.
5. **Where π has no support, the deficit is connectional, not molecular.** Association cortex is not molecularly alien to the mouse; it is connectionally reorganised.
6. **That measurement predicts disease.** Bipolar disorder and schizophrenia — and only those, out of 15 conditions — sit where the mouse cannot reach. Their subcortical signature does not.

## Two π files

| File | Anchors | Use when |
|---|---|---|
| `outputs/coupling/pi_fc_plus_SC.npy` | 21 Garin point anchors only | Strict baseline; benchmarking the FGW method itself |
| `outputs/coupling/pi_fc_plus_SC_with_all_packs.npy` | 21 Garin + 26 region-anchor entries (15 packs) | **Recommended for downstream queries** |

`load_pi()` defaults to the recommended one. These are **different matrices** and give different answers; do not mix them in a single analysis.

Produced by `pipeline/04_solve_production.py` + `experiments/anchor_packs/compose_all.py`.

---

## 1 · The coupling is calibrated

*Notebook: `fig1_coupling.ipynb`. Logs: `coupling_summary.json`, `fig1_coupling_matrix.json`, `evidence_tiers_v2.json`.*

| property | value |
|---|---:|
| shape | 1,864 × 2,094 |
| median top-target probability | **1.00** |
| mouse parcels with top probability > 0.5 | **92 %** |
| mean self-mass on the 21-class homology diagonal | **0.26** (5.5× the 0.048 expected under a uniform mapping) |
| topographic fidelity (mouse pairwise distance → routed human pairwise distance) | **r = 0.61**, against a permuted-coupling null of ≈ 0 |

On Beauchamp 2022's external transcriptomic homology benchmark — 19 mouse↔human pairs derived from whole-brain gene expression, a modality HOMER does not use — the recommended coupling reaches **region-level AUROC 0.85**, mass-in-region 0.46, and **45.7 %** parcel-level top-1, significant for **18 of 19** regions against a parcel-set permutation null (FDR q < 0.05). Its errors are graceful: routed mass falls a mean of **17 mm** from the expected homologue and a miss lands on an anatomically adjacent structure rather than scattering.

### Evidence tiers

| Tier | n | % | What it means |
|---|---:|---:|---|
| **anchored_and_validated** | 577 | **31 %** | In a curated anchor AND independently reproduces a published homology |
| **validated_only** | 401 | **22 %** | Reproduces a published homology, no anchor |
| **anchored_only** | 238 | **13 %** | In a curated anchor, no independent validation pair |
| **structural** | 259 | **14 %** | High internal trust, no external evidence |
| **low_evidence** | 389 | **21 %** | No supervision, weak internal signal — use with caution |

The two validated tiers cover **52 %** of the brain.

**What the tier grades is *resolution*, not existence.** Region-level recovery is essentially equal across the two validated tiers (AUROC **0.87** vs **0.88**); parcel-exact recovery is not (top-1 **0.69** vs **0.18**). Curation buys parcel precision, not region-level correspondence. Query at parcel granularity only in `anchored_and_validated`; at region granularity across both validated tiers.

**Trust cannot be read from the solver.** At the production regularisation the coupling is sharply peaked *everywhere*, so intrinsic confidence is uncorrelated with accuracy. The grades are external by necessity.

```python
trust = np.load("outputs/coupling/trust_multisource_all_packs.npz", allow_pickle=True)
reliable_parcels = np.where(trust["evidence_tier"] == "anchored_and_validated")[0]
```

---

## 2 · What carries cross-species homology

*Notebook: `fig2_what_carries_homology.ipynb`. Logs: `ablation_ladder_battery.json`, `anchor_recovery_loo_combined.json`, `beauchamp_metric_battery.json`, `beauchamp_metric_battery_loro.json`, `ablation_auroc.json`.*

Ablate the cost terms and score with the full metric battery. Two quantities move independently:

| cost terms | region-level (AUROC) | parcel-exact (top-1) | centroid displacement |
|---|---:|---:|---:|
| connectivity only (GW on FC + SC) | **0.67** — chance | 0.8 % | 35 mm |
| + spatial position | **0.87** — saturated | 8 % | 28 mm |
| + curated anchors | 0.84 | 10 % | 27 mm |
| + region packs | 0.85 — **unchanged** | **46 %** | **17 mm** |

**Connectivity and spatial position carry *which human region*. Curation carries *which parcel*.**

Connectivity alone is not uninformative — it is **unidentifiable**. Gromov–Wasserstein aligns two connectomes only up to relabelling, so with nothing to fix the global orientation the coupling cannot be *placed*. The sharpest demonstration is in the factorial (ED2): gene-coexpression connectivity **alone** scores AUROC **0.28**, below chance. A connectivity matrix, however biologically meaningful, cannot place itself without a cross-species reference.

### Withholding the curation

Remove each of the **41** combined supervision units (**15** Garin homology classes + **26** region packs) in turn, re-fit the full model, and score the held-out unit from connectivity and space alone:

| | held-out AUROC | held-out top-1 |
|---|---:|---:|
| Garin classes (n = 15) | 0.74 | 1.6 % |
| region packs (n = 26) | 0.72 | 1.4 % |
| **overall (n = 41)** | **0.73** | **1.5 %** |

These are **parcel-count-weighted** means (the unweighted overall AUROC is 0.72). Region-level recovery holds well above chance; parcel-exact recovery collapses. The connectomes and the spatial scaffold reconstruct the region-level correspondence without any anchor at all.

Read the collapse carefully: top-1 is the wrong metric for a held-out unit. Scored by displacement rather than top-1, the held-out coupling is displaced (median 34 mm vs 17 mm for the full model) but **not lost**. Connectivity and space give coarse localisation; anchors sharpen it to the parcel.

### The memorisation control

Several region packs were curated *on* Beauchamp regions, so the benchmark is not fully independent until that overlap is removed. Delete each region's overlapping curation, re-fit, re-score: aggregate AUROC **0.85 → 0.78**. Recovery is largely retained, so agreement with the transcriptomic benchmark reflects cross-species signal the coupling **reconstructs**, not curation it memorised.

---

## 3 · What transfers through π

*Notebook: `fig3_what_transfers.ipynb`. Logs: `coletta_2020_cross_species_rsn.json`, `fair_nulls_coletta_test2c.json`, `margulies_2016_gradient.json`, `margulies_discrete_reframe.json`, `fulcher_2019_gradient.json`, `spin_test_gradients.json`, `published_map_validation.json`.*

Three published cross-species relationships, all on data HOMER never saw, all tested against **spatial-autocorrelation-preserving spin nulls**.

| test | modality | \|r\| or count | spin p | verdict |
|---|---|---:|---:|---|
| Mouse resting-state networks → human (Coletta 2020) | connectivity | 6/10 vs 1.2 expected | **0.002** | **translates** |
| Principal FC gradient → human (Margulies / Huntenburg) | connectivity | \|r\| = 0.54 (0.62 region-level) | **0.004** | **translates** |
| T1w:T2w myelin → human myelin (Fulcher 2019) | microstructure | r = 0.37 | 0.11 | **does not clear the null** |
| Cytoarchitecture → human myelin (Fulcher 2019) | microstructure | r = 0.36 | 0.10 | **does not clear the null** |

### The gradient translates

The mouse principal gradient routed through π predicts the observed human gradient at **|r| = 0.54** across 1,244 parcels (0.62 at region level), exceeding both a permuted-π null (|r| = 0.03, p < 0.001) and a spin null (**p = 0.004**). It survives reduction to discrete structure too: the rank order of the nine human networks along the gradient is recovered at ρ = 0.73 (spin p = 0.043), and a three-tier discretisation is classified at 52 % against 33 % chance and a 34 % spin null (p = 0.001).

> ⚠️ **A correction.** An earlier version of this analysis took the **first** non-trivial eigenvector of the FC graph Laplacian as the principal gradient. In this data that is an **anterior–posterior spatial axis**; the unimodal→transmodal hierarchy is the **second** component. Routing an A–P spatial axis and then testing it against a *spatial-autocorrelation-preserving* null is close to tautological, and it manufactured a confident false negative ("the gradient does not translate", |r| = 0.41, p = 0.15) that this page previously reported. The component is now **selected** against each species' own T1w:T2w map, an external reference; both species independently select component 2, which reproduces the published Margulies gradient at |ρ| = 0.93 (the old component-1 map scored 0.12). `experiments/validation/00_validate_published_maps.py` now asserts this on every named external map.

### Microstructure does not

Two independent mouse measurements from Fulcher et al. — the T1w:T2w myelin proxy and cytoarchitectural type, **neither an input to HOMER** — routed through π and compared with the observed human myelin map over 205 Schaefer regions. Both *resemble* it (r = 0.37, 0.36) and both crush a permuted-π null (empirical p = 0.000). **Neither clears a spatial null** (spin p = 0.11, 0.10). Under the designated translation null (spin the mouse input, route through the real π) the verdict is the same: T1w:T2w fails (p = 0.086), cytoarchitecture sits exactly on the boundary (p = 0.050).

We therefore **do not claim that microstructure translates**. The routed maps are consistent with the human myelin map, but not beyond what the spatial smoothness of both maps already supplies.

> ⚠️ **A correction.** This page previously reported spin p = **0.021 / 0.010** for these two tests and called the structural correspondence "specific". Those p-values were **hardcoded literals in a figure script** and existed in no output file anywhere. The real values are 0.11 and 0.10.

### Cellular reach

Routing Allen ISH markers through π, the **excitatory − inhibitory** contrast translates (r = 0.26, spin p = 0.001). Neuronal − glial, dopaminergic-hotspot, laminar and areal-type contrasts do not. Conservation reaches broad cell classes but not finer composition or lamination — which is consistent with the microstructure result, not in tension with it.

### The pattern

What transfers is **connectional organisation**, the modality π was fitted on. What does not transfer is **microstructure**, which π never saw. π is a connectional correspondence and behaves like one: it carries connectivity across species and does not silently import a microstructural correspondence it has no evidence for.

---

## 4 · HOMER versus TransBrain

*Notebook: `fig4_vs_transbrain.ipynb`. Logs: `transbrain_benchmark_summary.json`, `transbrain_bn_distributions.json`, `transbrain_roundtrip_maps.json`, `transbrain_2025_benchmark.json`.*

TransBrain (Nat Methods 2025) is a **transcriptomic** translator. HOMER is a **connectional** one. §3 predicts the outcome: each should win on the modality it encodes.

| axis | HOMER | TransBrain | winner |
|---|---:|---:|---|
| region-level identity (AUROC) | 0.79 | **0.84** | TransBrain (n.s.) |
| principal-gradient translation (\|r\|) | **0.55** | 0.42 | HOMER |
| round-trip fidelity — gradient / opto / Magel2 | **0.98 / 0.95 / 0.97** | 0.89 / 0.82 / 0.83 | HOMER |
| prediction sharpness (effective target regions) | **≈ 3** | ≈ 60 | HOMER |
| spatial resolution | **parcel (2,094)** | region (~120) | HOMER |
| per-prediction confidence | **evidence tiers** | none | HOMER |
| whole-brain coverage / absence detection | **yes** | none | HOMER |

Two honest caveats.

**The region-identity benchmark is TransBrain's own**, so its lead there has home advantage — and even so, the difference is **not significant** (paired Wilcoxon on per-region AUROC, p = 0.17, n = 24 regions). TransBrain wins 16 regions, HOMER 8. HOMER nonetheless places more mass on the correct region (0.11 vs 0.07).

**We do not claim HOMER localises better.** We once did; it was a **reduction artefact**. TransBrain's output is region-level, so scoring a localisation metric at parcel resolution flatters HOMER by construction — the comparison measures output granularity, not mapping quality. The panel was reframed around *sharpness*, which is a real and large difference (≈ 3 vs ≈ 60 effective target regions).

The round-trip result is HOMER's clearest advantage: translate a mouse phenotype mouse→human→mouse and correlate with the original. The margin is narrowest on the smooth principal gradient (+0.09) and widest on the two focal maps, an optogenetic agranular-insula circuit and the Magel2 autism-model pattern (+0.13 each), where region-level smoothing destroys the spatial detail a focal phenotype lives in.

> ⚠️ **A correction.** The round-trip previously scored HOMER on the 52 mouse regions its parcellation covers but TransBrain on all 68 of `Config.MOUSE_REGIONS` — 16 of them mean-filled for the gradient. The two numbers were not comparable. Both methods are now scored on the identical 52 regions where the phenotype is measured. TransBrain's values moved from 0.87 / 0.91 / 0.81 to 0.89 / 0.82 / 0.83; HOMER's are unchanged.

These are **complementary instruments, not competitors**: a transcriptomic translator is the better instrument for region identity, a connectional one for connectional organisation.

---

## 5 · Where π has no support

*Notebook: `fig5_coverage.ipynb`. Logs: `section5_coverage_nulls.json`, `section5_connectional_vs_molecular.json`, `section5_evolution_battery.json`, `section5_coverage_catalogue.json`, `biccn_contrast_reframe.json`, `balsters_2020_mfc_divergence.json`.*

> ### ⚠️ Coverage is a MASS-NORMALISED MEAN, never a sum
> Coverage of a human parcel is its column-sum of π (`pi.sum(0)`). To aggregate parcels into a region you must take the **mean**, not the sum. Summing makes coverage scale with *how many parcels a region happens to contain* — a parcellation artefact, not biology. It is not a free parameter: the §6 disorder result is ρ = **+0.64** with the mean and ρ = **+0.05** with the sum. An earlier version of this analysis summed, and reported a null.

Semi-relaxed FGW frees the human marginal, so the coupling may leave human parcels uncovered. **53 %** of them receive negligible mouse mass (mass < 1e-6), concentrated over association cortex.

### The territory is organised along the hierarchy

Ordering cortex by the T1w:T2w myelin proxy, coverage in the association tertile is **6.7 log-units** below the sensorimotor tertile (spin p = **0.002**). The *continuous* coverage–myelin correlation does **not** exceed the spin null (r = 0.13, p = 0.076), so the claim is a contrast between the hierarchy's extremes, not a smooth gradient. We report the tertile gap and flag the continuous correlation as spin-fragile.

### The deficit is connectional, not molecular

The same sensorimotor − association contrast, computed two ways over the same 884 cortical parcels:

| measure | gap | spin p | |
|---|---:|---:|---|
| **connectivity coverage** | +0.47 SD | **0.016** | significant |
| **transcriptomic similarity to mouse** | −0.16 SD | 0.45 | n.s. |
| **the dissociation itself** | **+0.64 SD** | **0.038** | significant |

If the absence were molecular, both would collapse. Only one does. **Association cortex is not molecularly alien to the mouse — it is connectionally reorganised.** The mouse has the parts; it does not have the wiring.

Note the third row. It is not enough that one gap is significant and the other is not — that is the "difference between significant and non-significant is not itself significant" error. The *difference of the gaps* is tested directly against its own spin null, and it holds.

### It coincides with human cortical evolution

Correlating coverage against a battery of published cortical maps, each spin-tested: coverage aligns with the principal gradient (ρ = −0.12, p = 0.009), the Sydnor sensorimotor–association axis (−0.14, p = 0.013), the T1w:T2w hierarchy (+0.11, p = 0.037) and macaque→human evolutionary expansion (Hill et al.; −0.18, p = 0.046). **Four of seven maps clear a conservative spin null, and one runs counter** (Xu et al. mouse→human expansion, −0.05, p = 0.56).

Read this conservatively. Every individual correlation is *modest* (|ρ| 0.05–0.18). What carries the claim is the **consistency of direction across seven independent maps**, not any single effect size. So: coverage aligns with the cortical hierarchy, and that alignment is **corroborated by**, rather than driven by, evolutionary expansion.

### A falsification control

If HOMER were simply smearing mouse mass over human cortex, mouse medial-frontal cortex would leak into human dlPFC. It routes essentially **none** there (mass fraction ≈ 0 %, versus 1.1 % expected under a permuted-coupling null), sending it instead to premotor (20 %), medial-prefrontal (10 %) and mid-cingulate (28 %) targets — consistent with the absence of a rodent granular prefrontal homologue. This is a place the model could have embarrassed itself and did not.

---

## 6 · Which parts of a human disorder a mouse can reach

*Notebook: `fig6_disease.ipynb`. Logs: `section6_double_dissociation.json`, `section6_selectivity_battery.json`, `section6_robustness.json`.*

Correlate coverage with ENIGMA case-control **cortical thickness Cohen's d** across the 30 Desikan–Killiany regions HOMER resolves, under a spin null. A *positive* ρ means thinning is more severe where less mouse mass arrives.

| | cortex (n = 30) | subcortex (n = 7) | interaction |
|---|---:|---:|---:|
| **bipolar disorder** | ρ = **+0.64** (spin p < 0.001, FDR q = 0.004) | ρ = **−0.68** | Fisher z p = **0.003** |
| **schizophrenia** | ρ = **+0.52** (spin p = 0.002, FDR q = 0.004) | ρ = **−0.79** | Fisher z p = **0.002** |
| the other **13** conditions | all null | | |

**Selectivity is the result.** If coverage predicted thinning in every disorder it would be a generic "association cortex is vulnerable" statement. It does not. The well-powered nulls are what make it specific — most sharply **22q11 deletion syndrome**, the largest known genetic risk factor for psychosis (|d| = 0.39), which coverage does **not** flag. Coverage indexes *anatomy*, not diagnostic category: 22q11 simply does not share the cortical topography that bipolar disorder and schizophrenia share.

**The sign reverses in subcortex.** In cortex the disorder is worse where the mouse cannot reach; in subcortex, worse where it can. So: *a mouse model cannot address the cortical signature of bipolar disorder or schizophrenia, but it can address their subcortical signature.* That is a statement about which **component** of a disorder is modellable.

**Validation and discovery.** The cortical selectivity **replicates** van den Heuvel et al. (*Brain* 2019) — human-specific cortical connectivity features are implicated in schizophrenia and not in ASD, OCD, MDD, bvFTD or Alzheimer's — reached here from mouse connectivity alone, with no human disorder data in the model. The cortex/subcortex decomposition is the new part.

### Controls

- **Is it just the hierarchy?** Published hierarchy maps do predict this thinning (Sydnor ρ = −0.57, Margulies −0.52 in bipolar disorder). Coverage attains the largest correlation of any predictor tested, is the only map significant in both disorders, and survives partialling both out (ρ = +0.60 / +0.43). **We do not claim it is the better predictor**: at 30 regions the two are not statistically distinguishable (Williams' test, p = 0.33). The control establishes non-reducibility, not superiority.
- **Anchor distance.** Coverage correlates with distance from the 42 curated anchors (ρ = −0.41), and anchor distance predicts bipolar thinning on its own. The association survives adjustment (ρ = +0.53 / +0.49).
- **Analysis choices.** Rescue radius (2–6 mm), minimum-parcel threshold (≥3, ≥10, ≥20) and leave-one-region-out all leave the conclusion intact (bipolar +0.59 to +0.70).
- **Parcel count.** Summed rather than mass-normalised coverage abolishes the effect entirely (+0.05 / +0.02). See the box in §5.

### Limitations, stated plainly

The subcortical arm is **n = 7 regions**. The two arms use **different metrics** (cortical thickness vs subcortical volume; ENIGMA publishes no subcortical thickness), and a sign reversal across two different measurements is weaker evidence than one within a single measurement. And we show **no mechanism** — a correspondence between connectional reorganisation and disorder anatomy, not a cause.

A per-disorder "translatability index" was built and **failed**: Parkinson's disease scored no better than schizophrenia, because ENIGMA's subcortical panel contains no substantia nigra. The index measured where a disorder is *visible to volumetric MRI*, not where it is. It is not reported as a result, and should not be resurrected without phenotype data that samples the structures each disorder actually occupies.

---

## Application: re-subtyping autism mouse models

*Notebook: `discussion_pagani.ipynb`.*

Pagani et al. (2026) split 20 autism mouse models into hyper-connected (n = 9) and hypo-connected (n = 11) subtypes. Routing each subtype through π into human network space, the **hyper**-connected subtype translates *specifically* (predicted-hyper correlates +0.35 with observed hyper and −0.25 with observed hypo). The **hypo**-connected subtype does **not** — its prediction points the wrong way (+0.21 with observed hyper, −0.13 with observed hypo). One of the two works; we report both.

This analysis has been wrong twice: the subtype labels were once **inverted**, and a 1,491-feature decode was **debunked**. The current result is independent of the decode and the labels are verified from the data. Read `discussion_pagani.ipynb` before citing anything from it.

---

## Caveats

1. **The headline Beauchamp top-1 is partly by construction.** The recommended π is supervised on published homologues that overlap the validation set. The honest generalisation numbers are the leave-one-region-out (AUROC 0.73) and the curation-removed re-fit (0.85 → 0.78), both above.
2. **Parcel-level claims need the right tier.** Per-parcel argmax displacement is 17 mm on the benchmark and larger elsewhere. Trust parcel granularity only in `anchored_and_validated`; use region granularity across the validated tiers.
3. **The spatial prior is doing real work.** Zeroing xyz collapses region-level recovery to chance. This is not a defect — it is the identifiability result of §2 — but it means HOMER is not unsupervised homology discovery.
4. **Microstructure and fine molecular detail do not translate.** See §3. Do not read them out of π.
5. **Cerebellum and medulla** are excluded from the parcellation. **dlPFC homology** is contested and opt-in only.

## How to use this map

1. **Load the recommended coupling** — `load_pi()`. Not the base coupling.
2. **Check the tier** before you trust a prediction. `anchored_and_validated` → parcel-level. Either validated tier → region-level. `low_evidence` → treat as a hypothesis.
3. **Ask whether your phenotype is connectional.** If it is (networks, gradients, connectivity-derived maps), π will carry it. If it is microstructural or fine-molecular, it will not — §3 is the evidence, and it is a real limit, not a caveat to wave through.
4. **Check coverage before translating into association cortex.** If the target region receives negligible mouse mass, the absence of a homologue is the finding, not a failure of the query.
5. **Spin-test every spatial correlation** (`homer.eval.nulls.spin_null`) before reading it as significant. A permuted-π null is too lenient for a smooth map — that is how two of the errors on this page happened.
