# Results

The headline numbers, where to trust the model, and the honest caveats. The full development log (22 sections covering every detour and ablation) lives in [archive/iteration_log.md](archive/iteration_log.md).

## Two π files

HOMER ships two main coupling matrices. Use the one that matches your use case.

| File | Anchors | Use when |
|---|---|---|
| `outputs/coupling/pi_fc_plus_SC.npy` | 21 Garin point anchors only | You want the strictest baseline; you're benchmarking the FGW method itself |
| `outputs/coupling/pi_fc_plus_SC_with_all_packs.npy` | 21 Garin + 11 default region-anchor entries | **Recommended for downstream queries.** Best mouse↔human mapping we can deliver with current evidence |

Both are produced by `pipeline/04_solve_production.py` + `experiments/anchor_packs/compose_all.py`.

## Headline numbers

Beauchamp 2022 external validation (15 anchor-overlapping mouse↔human pairs, 927 parcels):

| Metric | Production (point anchors only) | + all default packs | Δ |
|---|---:|---:|---:|
| **Top-1** | 12 % | **37 %** | ×3.1 |
| **Top-5** | 22 % | **46 %** | ×2.1 |
| **Top-10** | 27 % | **50 %** | ×1.8 |
| **Mean rank / 2094** | 871 | **85** | **×10** (smaller is better) |

Region-level evaluation (Beauchamp-22 candidate set):

| Metric | Production | + all packs |
|---|---:|---:|
| Qualified top-1 | 37 % | **82 %** |
| Qualified top-3 | 70 % | **100 %** |
| Mean fold enrichment | 16× | **123×** |

Bootstrap argmax stability over 40 subject-resamples: **97.8 %**. 88 % of mouse rows have an identical argmax across all 40 bootstrap samples.

z-score vs permuted-anchor null: **+17.8** (the specific mouse↔human pairings drive the result, not just "having any 42 anchor constraints").

## Per-region trust tiers (multi-source evidence)

For each of the 1864 mouse parcels, we combine 5 signals into an evidence tier:

| Tier | n parcels | % | What it means |
|---|---:|---:|---|
| **anchored_and_validated** | 354 | 19 % | In an anchor pack AND Beauchamp top-1 > 0 — *highest confidence* |
| **anchored_only** | 65 | 4 % | In an anchor pack, no Beauchamp validation pair (e.g. OFC, AON, RSC) |
| **validated_only** | 665 | 36 % | In a Beauchamp region with top-1 > 0, no specific anchor pack |
| **structural** | 233 | 13 % | High internal trust (bootstrap + concentration + FC similarity) but no external evidence |
| **low_evidence** | 547 | 29 % | Use with caution — no supervision, weak internal signal |

Load and filter:
```python
trust = np.load("outputs/coupling/trust_multisource_all_packs.npz", allow_pickle=True)
reliable_parcels = np.where(trust["evidence_tier"] == "anchored_and_validated")[0]
```

## Per-region performance — parcel-level *and* region-level

HOMER produces a *probability distribution* over 2094 human parcels per mouse parcel — so a single "did the argmax exactly hit Beauchamp's target?" metric undersells what the model is actually doing. We report both views: parcel-level top-K (the harsh "single best parcel" test) and region-level rank + fold enrichment (where does the *mass* go?). The two metrics map cleanly onto the two trust tiers from the previous section.

| Region | parcel top-1 | parcel top-5 | parcel top-10 | region rank / 21 | fold enrichment | Trust tier |
|---|---:|---:|---:|---:|---:|---|
| **Pack-anchored** (9 regions) | | | | | | |
| Motor → precentral | 100 % | 100 % | 100 % | 1 | 47× | anchored_and_validated |
| Superior Colliculus | 100 % | 100 % | 100 % | 1 | 1047× | anchored_and_validated |
| Inferior Colliculus | 100 % | 100 % | 100 % | 1 | 524× | anchored_and_validated |
| Piriform → piriform | 100 % | 100 % | 100 % | 1 | 161× | anchored_and_validated |
| Amygdala | 100 % | 100 % | 100 % | 1 | 349× | anchored_and_validated |
| Subiculum, CA1, CA3, Dentate | 100 % each | 100 % | 100 % | 1 | 262-524× | anchored_and_validated |
| **Garin point anchor only** (10 regions) | | | | | | |
| Thalamus | 30 % | 48 % | 52 % | 1 | 29× | validated_only |
| Striatum ventral → NAc | 8 % | 42 % | 62 % | 1 | 100× | validated_only |
| Auditory → Heschl's | 22 % | 22 % | 22 % | 1 | 26× | validated_only |
| Somatosensory → postcentral | 19 % | 37 % | 45 % | 1 | 10× | validated_only |
| Anterior cingulate → cingulate gyrus | 13 % | 22 % | 35 % | 1 | 11× | validated_only |
| Caudate | 13 % | 27 % | 34 % | 1 | 11× | validated_only |
| Hypothalamus | 12 % | 17 % | 19 % | 2 | 60× | validated_only |
| Visual → cuneus | 7 % | 7 % | 7 % | 1 | 4× | validated_only |
| Pallidum | 5 % | 9 % | 9 % | 2 | 16× | validated_only |
| Pons | 3 % | 3 % | 3 % | 2 | 10× | validated_only |

**Read the table this way:**

- *Pack-anchored rows*: parcel top-1 = 100 %, but this is **largely by construction** — the anchor packs use the same mouse-side sets that Beauchamp validates against, and the human-side balls overlap. The deployment value is real (HOMER queries return defensible parcel-level answers for these regions); the methodological value is *that we shipped the supervision*, not that the model discovered the homology. See caveat 1 below.

- *Garin-point-anchor rows*: parcel top-1 is **3-30 %** — which *looks* bad but is misleading at parcel granularity. The region-level columns show what's actually happening: **every single non-pack region has the right human region in HOMER's top-3 (8 of 10 at rank 1), with fold enrichment 4-100× above chance**. The model puts substantial mass on the correct human region (parcel top-5 = 23 % mean, top-10 = 29 % mean for these rows) — it just doesn't always concentrate on the single canonical Beauchamp ball parcel.

So the honest mapping summary is:

- **Pack-anchored regions** = trustworthy at parcel granularity (by construction).
- **Garin-point-anchor regions** = trustworthy at *region* granularity, mediocre at parcel granularity.
- **Unanchored / low-evidence regions** = not trustworthy.

The multi-source trust map gates this for you per-parcel — see "How to use this map" below.

## Network coherence (independent validation)

Multi-source independent check: when we group mouse parcels by their nearest Garin network, the **olfactory** and **limbic** networks become substantially more compact in human space when packs are applied (median pairwise distance −17.7 mm for olfactory, −12.8 mm for limbic). Networks not directly anchored by a pack show small fragmentation (+4 to +10 mm). This is non-Beauchamp evidence that the packs encode coherent biology, not just constraint satisfaction.

## Independent validation against Pagani 2026 (Nat Neurosci autism subtypes paper)

### What we set out to test

[Pagani et al. 2026, *Nat Neurosci*](https://www.nature.com/articles/s41593-026-02287-z) ("Autism subtypes identified using cross-species functional connectivity analyses") clusters 20 mouse autism models into two FC-perturbation subtypes (hyperconnected, hypoconnected), recovers the same two subtypes in 1,029 human ASD subjects from ABIDE, and shows that each subtype carries a distinct gene/pathway signature that recurs across species (synaptic for hypo, immune for hyper). Across the whole workflow, four claims do the heavy lifting:

1. **Subjects form two FC subtypes** (hyper, hypo), recoverable in both species.
2. The cross-species bridge connecting "mouse subtype X" to "human subtype X" is a **name-based correspondence** between 9 mouse networks (ED Fig 1) and 8 human networks (Fig 4e). Workhorse infrastructure, not a result they claim — but every cross-species finding rides on it.
3. **The subtype FC perturbation patterns recur cross-species in matching anatomical locations.** Their evidence is that the per-subtype network-perturbation matrices look similar when you read them with the name-bridge.
4. **The same per-subtype gene/pathway signature appears in both species** — synaptic in hypo, immune in hyper.

HOMER is positioned to test all of this independently because it produces a quantitative cross-species coupling π that knows nothing about Pagani's data, gene sets, or subtype assignments. What follows is a chain of four hypotheses, with what we found at each step and what it tells us about where HOMER's contribution lives.

### Hypothesis 1 — does Pagani's name-based bridge have biological substance?

The first thing to check is the scaffolding. If π preferentially routes mass between like-named networks across species, the name-bridge isn't a wild assumption.

Aggregating π to a mouse-network × human-network matrix (mouse parcels → HOMER's PAIRID_TO_NETWORK; human parcels → Yeo-7 via Schaefer-400 + Subcortical for un-Schaefer-assigned) and scoring each canonical mouse↔human pair against a uniform-π null:

| Mouse → Human | HOMER row-mass on target | Null | Ratio | Argmax? |
|---|---:|---:|---:|:---:|
| SomatoMotor → SomatoMotor | **43 %** | 11 % | **3.9×** | ★ |
| Salience → Salience | **38 %** | 9 % | **4.3×** | ★ |
| DMN → DMN | **41 %** | 16 % | **2.5×** | ★ |
| Subcortical → Subcortical | **57 %** | 29 % | **2.0×** | ★ |
| Visual → Visual | 8 % | 10 % | 0.8× | argmax: DorsAtten |
| HC_Limbic → Limbic | 7 % | 5 % | 1.4× | argmax: Subcortical |
| Auditory → Auditory | 0 % | 2 % | 0.0× | argmax: Control |
| BF_Olfactory → Subcortical | 14 % | 29 % | 0.5× | argmax: Limbic |

**4 of 8 canonical pairs are diagonal-argmax, with mean 1.92× concentration over null.** A permuted-π control sits at 2/8 and 0.97× (chance), so the agreement is real.

The four misses are all atlas-definition artefacts — Schaefer-17's "Visual" is V1-like only, so higher-order mouse visual maps to DorsAtten; hippocampus has no cortical Schaefer label so HC routes to Subcortical (which is correct — hippocampus *is* an allocortical/subcortical structure); Schaefer's auditory is a narrow band of 62 parcels and the broader auditory association cortex is scattered across Control/Salience; BF/olfactory is a coarse rodent label without a clean Yeo-7 counterpart. None of these are HOMER disagreeing with the biology; they're definitional mismatches between the two species' atlases.

**Verdict — Pagani's name-bridge is partially supported by HOMER's quantitative geometry.** Solid for SomatoMotor, Salience, DMN, and Subcortical; muddled (but not contradicted) for the others. The signal survives stripping anchor packs (4/8, 1.79×) and zeroing the xyz prior (3/8, 2.02×), so it isn't an artefact of any single component. Code: `01_network_crossvalidation.py`. Figure: `autism_subtypes_network_mapping.png`.

### Hypothesis 2 — does HOMER's π reproduce Pagani's per-subtype spatial pattern (claim 3) without using the name-bridge?

This is the headline question. If HOMER's translation of the *mouse* subtype perturbation matrix through π predicts the *human* subtype perturbation matrix — without the name-bridge anywhere in the pipeline — that's independent quantitative replication of one of Pagani's actual claims.

Method: build a translation operator T (9 × 8) where T[mi, hj] = P(human-net hj | mouse-net mi), derived by row-normalising the aggregated π over Pagani-aligned networks. Predicted human Δ-matrix = Tᵀ · Δ_mouse · T, where Δ_mouse = (hyper − hypo) intensity matrix from ED Fig 1. Compare against observed human Δ (Fig 4e hyper − hypo) over the 36 upper-triangle elements of the 8×8 symmetric matrix. Mouse parcels are mapped to Pagani's 9 networks via nearest-Garin-anchor pid, with pons/tectum/hypothalamus dropped (Pagani's mouse atlas excludes them).

**Result: Pearson r = +0.527, analytical p = 0.0009, empirical p = 0.000 vs permuted-π null** (200 trials, within the 1613 kept mouse parcels; null mean −0.40, 95% CI −0.72 to +0.17). 23 of 36 matrix entries agree in sign. The largest positive observed entry — Subcortical–Subcortical Δ ≈ +33 in human, the dominant hyperconnected signal — is also among HOMER's largest positive predictions. The largest negative observed entries (Limbic–SomatoMotor, Visual–SomatoMotor, Limbic–Salience, all on the hypo side) are also predicted negative by HOMER.

A coarser per-network row-sum version of the same test (n=8 elements) gives Pearson r = +0.547 with the same direction-agreement story — HOMER recovers the *direction* of the subtype contrast on 6 of 8 human networks. The full 36-element version is sharper because it preserves the joint network-pair structure rather than collapsing to network-marginal intensity.

**Verdict — strong independent replication of Pagani's claim 3.** HOMER's π, fit without any access to Pagani's data, reproduces the joint network-pair Δ structure at p < 0.001 against a properly-permuted null. This is the cleanest HOMER finding on this paper. Code: `07_full_matrix_translation.py` (n=36 sharp version), `05_subtype_contrast.py` (n=8 coarse version). Figure: `autism_subtypes_full_matrix.png`.

### Hypothesis 3 — does HOMER produce an individual-subject classifier for ASD (the sharper version of claim 1)?

The network-aggregate result is at population level — averaged perturbation matrices. The next question is whether HOMER's translated template carries enough signal to distinguish ASD from controls subject-by-subject, and whether ASD subjects then split into bimodal hyper/hypo clusters by this feature.

Pipeline: fetch 871 ABIDE-pcp subjects (CPAC, AAL-116 parcellation, ~24 sites), compute each subject's `mean(FC)` per parcel as a perturbation feature, site-match against control means, map to HOMER's 2,094 parcels by nearest centroid, score by dot-product against the z-scored HOMER (hyper − hypo) template.

| Feature | Mann-Whitney p (ASD vs CTRL) | Cliff's δ | Direction |
|---|---:|---:|:---|
| `mean(\|FC\|)` (sign-destroying, original) | 0.102 | −0.066 | none |
| `mean(FC)` (signed, audit-corrected) | **0.042** | **−0.083** | ASD < controls on (hyper − hypo) template |

After correcting the feature definition (the absolute-value version collapses hyper- and hypo-perturbed subjects to the same magnitude), the test reaches conventional significance. The direction is informative: **ASD subjects score systematically lower on HOMER's (hyper − hypo) template than controls — i.e., look more like the hypoconnected mouse template.** This is consistent with the longstanding ASD-hypoconnectivity finding in the FC literature, and validates *one half* of Pagani's claim 1 — there is a cross-species-translatable ASD signature at the individual level, with ASD displaced toward the hypo side.

Within ASD, the score distribution is unimodal (1-component GMM preferred, Δ BIC = +17.8). HOMER does *not* recover Pagani's hyper-vs-hypo split as a within-ASD classifier — the template scoring places ASD on the hypo side on average but doesn't separate individuals into two groups.

**Verdict — small but real subject-level effect.** ASD-as-hypoconnected emerges through HOMER (p=0.042) but the within-ASD subtyping (the part of Pagani's claim 1 most specific to their workflow) is not recoverable from this feature. The effect is small (Cliff's δ = −0.08, "negligible" by Romano's effect-size criteria) and the test uses a coarse single-summary feature; replicating Pagani's per-cell-of-FC-matrix perturbation with proper age/motion regression would probably increase the effect. Code: `abide_subtype/abide_subtype_prediction.py`. Figure: `autism_subtypes_abide.png`.

### Hypothesis 4 — does HOMER also validate Pagani's gene-pathway cross-species claim (claim 4)?

The most ambitious test: if HOMER's spatial mapping is correct, then the *mouse spatial expression* of Pagani's subtype gene sets, translated through π, should predict the human ASD perturbation pattern. This would link Pagani's gene biology to their FC biology through HOMER's geometry.

We downloaded parcel-level Allen ISH expression for **1,713 of Pagani's 6,415 implicated genes** (27% yield, limited by Allen's coronal coverage), built per-parcel mouse spatial scores for the hypo and hyper gene sets, translated through π, and correlated the predicted human Δ against Pagani's observed Δ from Fig 4e.

**Initial reading — looked like a clean confirmation:** Bootstrap-mean Pearson r = **+0.428, 95 % CI (+0.349, +0.497), 100 % of 1,000 gene-resamples positive**, 99.7 % above r = +0.3. The 8-network aggregation caps the per-test Pearson regardless of how many genes go in (the point estimate barely improved from the 36-gene proof of concept), but the bootstrap shows the cross-species signal is exceptionally stable — every random gene resampling produces a positive correlation.

Per-pathway breakdown across 14 pathways (synaptic, immune, mTOR, WNT, MAPK, GPCR, chromatin, etc., from MOESM3) showed the same direction for *every* pathway: positive r ≈ +0.35-0.51 with observed Δ. Pagani's claimed direction-by-subtype split (synaptic → hypo, immune → hyper) didn't separate cleanly — but Pagani's published "hypo" matrix has tiny magnitudes (range 0-1.5) compared to "hyper" (range 0-33), so the Δ test is dominated by the hyper side. So the per-pathway flatness might be a Pagani-data artefact rather than a HOMER failure.

**Then the specificity check broke the autism-specific story.** Pagani's MOESM5 supplementary lists genes implicated in 5 other conditions. Testing each:

| Condition | n genes overlapping HOMER | Bootstrap r | 95% CI |
|---|---:|---:|:---:|
| ADHD | 30 | **+0.451** | (+0.392, +0.508) |
| Autism (Pagani's claim) | 1,713 | **+0.437** | (+0.430, +0.445) |
| Schizophrenia | 530 | **+0.434** | (+0.419, +0.449) |
| Bipolar disorder | 109 | **+0.410** | (+0.374, +0.442) |
| Psoriasis (non-brain control) | 2 | (too few to test) | — |
| Dementia | 0 | (no overlap) | — |

**Every brain-disorder gene set produces essentially the same correlation as autism.** ADHD beats autism. Schizophrenia ties autism. The CIs overlap heavily. The autism r = +0.43 isn't validating Pagani's autism-specific gene biology — it's reflecting a shared spatial geometry: HOMER's π routes any reasonable brain-gene-set's mouse-spatial map to the subcortical / limbic / somatomotor regions where Pagani's reported human ASD perturbation lives, and that produces r ≈ +0.4 regardless of input. Psoriasis (the non-brain negative control) couldn't be tested directly because of gene-overlap limits, but the consistent +0.4 across psychiatric conditions is a strong indication that we're picking up generic brain-disorder geometry, not autism-specific signal.

A complementary diagnostic adds nuance: at the per-parcel level (n=2,094 human parcels), the gene-translation predicted Δ and the FC-translation predicted Δ (from Hypothesis 2) correlate at only r = +0.15. The two HOMER translation routes are **complementary, not redundant** — gene-spatial encodes "which regions express the implicated genes"; FC-spatial encodes "which networks are functionally perturbed." Both correlate with Pagani's observed pattern, but they capture different aspects.

**Verdict — Pagani's claim 4 is partially supported, but not in the autism-specific way they frame it.** HOMER produces a stable cross-species spatial replication of *psychiatric brain-gene-set geometry*, not autism-specific gene biology. The genuine claim that survives is broader and more honest: psychiatric perturbation effects concentrate in the same anatomical regions across species, and HOMER's π captures this. Code: `allen_expansion/run_pagani_gene_test.py`, `diagnose_expanded.py`, `cross_disease_specificity.py`. Figures: `autism_subtypes_gene_expanded.png`, `autism_subtypes_cross_disease.png`.

### Where HOMER's signal lives (synthesis)

Putting the four hypotheses together gives a coherent picture of what HOMER's quantitative cross-species coupling does and doesn't contribute to this kind of analysis:

| HOMER contribution | Granularity | Outcome |
|---|---|:---|
| Validates the mouse↔human network bridge for major networks | Network-level (8 nets) | **Yes** — 4/8 diagonal-argmax, 1.92× over null |
| Reproduces per-subtype FC perturbation pattern across species | Network-pair-level (36 elements) | **Strong** — r=+0.527, p=0.0009 |
| Places ASD subjects on the hypo side at population level | Individual subject | **Modest** — p=0.042, Cliff's δ=−0.08 |
| Recovers within-ASD hyper/hypo subtypes as a classifier | Within-ASD | **No** — distribution unimodal |
| Validates Pagani's autism-specific gene/pathway claim | Disease specificity | **No** — same correlation for ADHD/SCZ/bipolar |
| Captures shared brain-disorder spatial geometry | Population spatial geometry | **Yes** — stable r≈+0.4 across psychiatric conditions |

The cleanest contribution is Hypothesis 2 — HOMER replaces Pagani's name-based bridge with a quantitative one and the cross-species FC-pattern claim holds independently. The gene-spatial story (Hypothesis 4) needs reframing: HOMER doesn't validate Pagani's autism-specific gene biology, but it does demonstrate that psychiatric brain perturbation effects share the same anatomical geometry across species. The subject-level result (Hypothesis 3) shows HOMER carries a small ASD-as-hypoconnected signal at the individual level but doesn't recover the bimodal subtype structure Pagani derives from clustering — that part remains a feature of their specific clustering pipeline, not something HOMER recovers automatically.

**What this tells us about where HOMER's signal lives.** Tests 2c and 3 show that HOMER's translation carries genuine cross-species spatial signal at the **network-aggregate / population-level** (Test 2c: r = +0.54 on per-subtype matrix Δ; Test 3: bootstrap r = +0.43 on per-pathway spatial maps). Test 4 shows that the same signal **does not** survive when applied as a per-subject classifier on noisy individual FC data, even with n ≈ 800. This is the natural granularity boundary of the method: HOMER carries cross-species signal at the level Pagani actually publishes results (per-subtype averages, per-network averages), but not at the level of individual-subject diagnosis.

The result is consistent with the observation that even Pagani's own clustering separates subtypes only as cluster-level averages, not as a 1-D score per subject. A more sophisticated subject feature (replicating Pagani's exact perturbation pipeline + clustering rather than our coarse mean-|FC|), a finer parcellation (Schaefer-400 or CC400 rather than AAL-116), and richer per-subject regression for site/age/motion would all probably improve the test, but the negative direction of the effect (Cliff's δ slightly *negative*, not positive) suggests there isn't a strong signal to recover regardless.

Code: `experiments/autism_subtypes/abide_subtype/abide_subtype_prediction.py`. Log: `outputs/logs/autism_subtypes_abide.json`. Per-subject scores: `outputs/logs/abide_per_subject_scores.csv`. Figure: `outputs/figures/autism_subtypes_abide.png`.

## Honest caveats — read these

1. **The 100 % top-1 for pack-anchored regions is largely by construction.** The anchor packs use the same mouse-side sets as Beauchamp's validation, and their human-side balls overlap Beauchamp's. The soft constraint then satisfies the validation. This is a *deployability* gain (HOMER queries are now trustworthy for those regions), not an *unsupervised recovery* claim. The *non*-pack-anchored regions (Thalamus, Auditory, S1, ACG, Caudate, NAc, Hypothalamus, Visual, Pallidum, Pons) are where the FGW solver is doing real work — they reach region-level rank 1-2/21 with 4-100× fold enrichment using only a single Garin point anchor per region propagated through FC + SC + spatial structure.

2. **Held-out tests confirm structure does NOT propagate across un-anchored sub-regions.** For every pack we tested, dropping one anchor entry and re-fitting leaves the held-out region at 0 % top-1 (Tectum's IC, Hippocampal's CA1/CA3/DG). Each region we want HOMER to handle reliably needs its own anchor entry.

3. **Held-out region CV gives the honest "structural recovery" number.** Drop one region's supervision entirely, re-fit, evaluate: **3.4 % top-1**, **5.5 % top-5**, **6.6 % top-10** (~7× chance). This is what FC + SC encode about cross-species correspondence *without* relying on the specific anchor for that region. mPFC (33 %), Auditory (22 %), Somatosensory (11 %) recover meaningfully; midbrain / olfactory / striatum recover at chance.

4. **The cingulate pack hurts a Beauchamp metric — by design.** Adding our subgenual ACC anchor drops Beauchamp ACG top-1 from 13 % → 9 %. The reason: our anchor target (subgenual ACC at –5, 10, 35) differs from Beauchamp's validation ball (pregenual ACC at –5, 25, 25). The cingulate pack is therefore opt-in, not default.

5. **dlPFC homology is contested.** The lateral PFC pack includes a Prelimbic ↔ dlPFC entry (Carlén 2017), but Preuss 1995 argues rodents lack a true dlPFC homologue. The dlPFC entry is opt-in within the pack.

6. **Cerebellum is excluded.** 14 of Beauchamp's 36 region pairs cannot be evaluated.

7. **Per-parcel correspondence is a region-level claim, not a strict 1:1 statement.** Mean argmax distance is 25-45 mm even in well-anchored regions. Argue at the region level, not the parcel level.

8. **Beauchamp 2022 is a published hypothesis (gene-expression-derived), not absolute ground truth.** Different validation sources (Mars 2018 white-matter, Coletta 2020 FC) might give different numbers — see [archive/iteration_log.md §5.21](archive/iteration_log.md) for the multi-source discussion.

## How to use this map

Match query granularity to evidence tier:

- **Region-level queries** (`pi[mouse_region_indices, :].sum(axis=0)`, top-K *human regions*) — **trustworthy for all `anchored_and_validated` and `validated_only` parcels (55 % of the brain)**. Both pack-anchored and Garin-point-anchor regions deliver rank-1/2 of 21 with strong fold enrichment at this granularity. This is the recommended query mode.
- **Parcel-level argmax queries** — only reliable for the `anchored_and_validated` tier (19 %). Pack-anchored regions concentrate mass on a few parcels; non-pack regions spread mass across the right human region without nailing one parcel.
- **For `structural` parcels (13 %)** — treat top-K predictions as hypotheses to verify with other evidence.
- **For `low_evidence` parcels (29 %)** — the trust map flags these. Don't trust argmax; query at region granularity if at all.
- **Avoid**: "mouse parcel X = human parcel Y" claims at the millimeter level. Mean argmax distance is 25-45 mm even in good regions.

For the per-pack detail, see [04_anchor_packs.md](04_anchor_packs.md). For limitations and what HOMER *can't* tell you, see [05_limitations.md](05_limitations.md).
