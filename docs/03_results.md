# Results

The headline numbers and the caveats. Full per-paper detail lives in the showcase notebooks (`notebooks/05-15`) and per-experiment READMEs.

## Synthesis

HOMER produces a soft probabilistic mouse↔human parcel coupling π (1,864 × 2,094) via Fused Gromov-Wasserstein optimal transport, supervised on 21 Garin 2021 point anchors + 26 region-anchor entries from 15 cross-species packs. On Beauchamp 2022's external 22-pair gene-expression benchmark, the recommended π hits 45.7 % parcel-level top-1 (50.6× over null) and 100 % qualified top-3 at region level, with 98.2 % bootstrap stability and +17.8 σ above the permuted-anchor null. It was tested against twelve independent cross-species datasets spanning FC, gene expression, cell types, cortical layers, connectivity gradients, intracortical myelin, and psychiatric disorders, plus two negative-control tests (frontal-cortex homology and the tethering hypothesis) and a head-to-head comparison with the TransBrain method. Translation is strongest at the regional/area/network level: the Margulies/Huntenburg principal gradient at |r| = 0.402 (parcel) and 0.433 (region), and Coletta 2020 resting-state networks at 6/10 diagonal-argmax. Gene-spatial patterns translate at moderate strength. Allen ISH cell-type markers (BICCN) translate for 13 of 23 markers (mean r = +0.089), strongest for interneuron and glutamatergic classes (Drd1 +0.227, Slc17a7 +0.221, Vip +0.201, Calb2 +0.199, Pvalb +0.198). Cortical layer markers (Hodge 2019) translate for 6 of 7 (mean r = +0.119). Disorder-specific signal does not survive: predicted human maps for autism, schizophrenia, bipolar and ADHD correlate at r > 0.97 (ENIGMA), so the model carries a shared psychiatric geometry rather than disorder-specific biology.

## Two π files

| File | Anchors | Use when |
|---|---|---|
| `outputs/coupling/pi_fc_plus_SC.npy` | 21 Garin point anchors only | Strict baseline; benchmarking the FGW method itself |
| `outputs/coupling/pi_fc_plus_SC_with_all_packs.npy` | 21 Garin + 26 region-anchor entries (15 packs) | **Recommended for downstream queries** |

Produced by `pipeline/04_solve_production.py` + `experiments/anchor_packs/compose_all.py`.

## Headline numbers

Beauchamp 2022 external validation (15 anchor-overlapping mouse↔human pairs, 927 parcels):

| Metric | Point anchors only | + all 15 packs | Δ |
|---|---:|---:|---:|
| **Top-1** | 12 % | **39 %** | ×3.3 |
| **Top-5** | 22 % | **47 %** | ×2.1 |
| **Top-10** | 27 % | **50 %** | ×1.8 |
| **Mean rank / 2094** | 871 | **106** | **×8** (lower is better) |

Region-level evaluation (Beauchamp-22 candidate set): qualified top-1 jumps **37 % → 81 %**, qualified top-3 **70 % → 100 %**, mean fold enrichment **16× → 122×**.

Bootstrap argmax stability over 40 subject-resamples: **98.2 %** (89 % of mouse rows have identical argmax across all 40 resamples).

z-score vs permuted-anchor null: **+17.8** — the specific mouse↔human pairings matter, not just having any 42 anchor constraints.

## Per-region trust tiers

Each of 1,864 mouse parcels gets a 5-tier evidence label combining anchor membership, Beauchamp top-1, bootstrap stability, concentration, and FC similarity:

| Tier | n | % | What it means |
|---|---:|---:|---|
| **anchored_and_validated** | 587 | 31 % | In an anchor pack AND Beauchamp top-1 > 0 — *highest confidence* |
| **anchored_only** | 228 | 12 % | In an anchor pack, no Beauchamp validation pair (e.g. OFC, AON, RSC) |
| **validated_only** | 443 | 24 % | Beauchamp top-1 > 0, no specific anchor pack |
| **structural** | 241 | 13 % | High internal trust but no external evidence |
| **low_evidence** | 364 | 20 % | Use with caution — no supervision, weak internal signal |

```python
trust = np.load("outputs/coupling/trust_multisource_all_packs.npz", allow_pickle=True)
reliable_parcels = np.where(trust["evidence_tier"] == "anchored_and_validated")[0]
```

## Per-region performance — parcel-level vs region-level

HOMER returns a probability distribution over 2,094 human parcels per mouse parcel. The "did the argmax hit Beauchamp's target?" metric undersells what's actually happening; the region-level columns show where the *mass* goes.

| Region | parcel top-1 | parcel top-5 | Anchor pack |
|---|---:|---:|---|
| **Pack-anchored, high recovery** | | | |
| Motor / Inf. colliculus / Piriform / Auditory / CA1 / CA3 / Dentate gyrus | 100 % | 100 % | dedicated pack |
| Superior colliculus | 96 % | 100 % | tectum |
| Amygdala | 96 % | 100 % | amygdala |
| Subiculum | 97 % | 100 % | hippocampal |
| **Pack-anchored, sub-region trade-off** (anchor target sits outside Beauchamp's broad ball — see caveat 5) | | | |
| Caudate | 33 % | 37 % | striatum |
| Somatosensory → postcentral | 15 % | 15 % | somatosensory |
| Anterior cingulate | 9 % | 74 % | cingulate |
| Visual → cuneus | 4 % | 20 % | visual |
| **Garin point-anchor only** | | | |
| Thalamus | 30 % | 48 % | — |
| Striatum-ventral → NAc | 8 % | 46 % | — |
| Hypothalamus | 12 % | 19 % | — |
| Pallidum | 5 % | 14 % | — |
| Pons | 3 % | 3 % | — |

**Pack-anchored, high-recovery regions** are trustworthy at parcel granularity (largely *by construction* — see caveat 1). **Sub-region-trade-off packs** (striatum, somatosensory, cingulate, visual) are anatomically defensible but score low on Beauchamp's *parcel* metric because the anchor target sits outside Beauchamp's coarse validation ball — see caveat 5. **Garin point-anchor-only regions** are trustworthy at *region* granularity, mediocre at parcel granularity. The multi-source trust map gates all of this per-parcel; see "How to use this map" below.

Additionally, when mouse parcels are grouped by their nearest Garin network, the **olfactory** and **limbic** networks become substantially more compact in human space after packs are applied (median pairwise distance −17.7 mm for olfactory, −12.8 mm for limbic) — non-Beauchamp evidence that the packs encode coherent biology.

## Independent third-party validation

We tested HOMER against 12 cross-species neuroscience papers using completely independent data (no overlap with Beauchamp's transcriptomic-similarity dataset or HOMER's anchor inputs). Each test asks a different cross-species question at a different granularity — nine look for a positive signal, two (Balsters 2020, Buckner & Krienen 2013) are negative-control / falsification tests, and one (TransBrain 2025) is a head-to-head comparison against a sibling method. The combined picture is a clean resolution-boundary statement.

| Paper | Test | Result | Verdict |
|---|---|---|---|
| **Pagani 2026** (Nat Neurosci) | Subtype perturbation Δ predicted from gene sets via π | Pearson r = +0.822 (p = 0.012), Spearman ρ = +0.524 (n.s.), over 8 networks using the 36 Pagani genes in HOMER's panel | Moderate, small n |
| Pagani 2026 | ABIDE per-subject scoring | p = 0.042, Cliff's δ = −0.083 (ASD < CTRL) | Modest direction |
| Pagani 2026 | Cross-disease gene-set spatial | autism r ≈ ADHD r ≈ SCZ r ≈ +0.43 | **No disorder-specificity** |
| **Margulies 2016 + Huntenburg 2021** | Principal connectivity gradient | **\|r\| = 0.402** parcel-level (region-level 0.433), empirical p = 0.000 (13× null) | **Strong (brain-wide)** |
| **Fulcher 2019** (PNAS) | Mouse T1w:T2w + cytoarchitecture → human myelin | **r = +0.373 / +0.362**, empirical p = 0.000 (205 regions) | **Strong (structural)** |
| **Balsters 2020** (PNAS) | Falsification — does mouse MFC avoid human dlPFC? | **0 %** mouse-MFC mass → dlPFC (0/46 argmax); enrichment ×0.0, p = 0.985; mass goes to premotor / medial PFC / cingulate | **Pass (falsification)** |
| **TransBrain 2025** (Nat Methods) | Head-to-head vs a sibling method + its homology benchmark | predicted-centroid **25.3 mm vs 39.8 mm null** (**p < 0.001**); top-3 **41 %**; head-to-head gradient \|r\| 0.393 (HOMER) vs 0.463 (TransBrain) | **Methods comparison** |
| **Buckner & Krienen 2013** (TICS) | Negative control — is π sparsest over untethered association cortex? | sensorimotor−association coverage gap **6.7 log units** (p = 3.4×10⁻⁷, empirical p = 0.000) | **Pass (negative control)** |
| **Coletta 2020** (Sci Adv) | Labeled mouse-net → Yeo-7 + coherence | **6/10 diagonal-argmax**, **9/11 nets compact** vs null | **Strong** |
| **BICCN** (Yao 2023 + Siletti 2023) | Cell-type marker spatial | 13/23 markers significant, mean r = +0.089; glutamatergic 4/4, interneuron 4/7 (Drd1 +0.227, Slc17a7 +0.221, Vip +0.201, Pvalb +0.198) | Moderate |
| BICCN | Broadly-cortical interneurons | Pvalb, Sst, Vip null | Null — pan-cortical limit |
| Whitesell 2021 (Neuron) | DMN refinement | Yeo-DMN 23.9 %; DMN-aligned cortical territory 54.5 % | Methodological note |
| **ENIGMA Phase 1** | Cross-disorder predictions at parcel level | Off-diagonal r = +0.988 (autism ≈ SCZ ≈ ADHD ≈ bipolar) | **Confirms no disorder-specificity** |
| Hodge 2019 (Nature) | Cortical layer markers | 6/7 markers significant, mean r = +0.119 (L2/3 +0.083/+0.176/+0.189, L4 +0.090, L5 +0.168, L6 +0.108) | Moderate |
| Pagani per-model | 20 mouse models × 1,491 features | Decoded as per-voxel global-connectivity; voxel-ordering pending mask file | Pending external data |

Each validation has its own showcase notebook (see `notebooks/05–15`) with the full method, figure, and discussion. Brief snapshots below.

### Pagani 2026 (autism subtypes, Nat Neurosci)

Four-hypothesis arc against the paper's claims. The subtype test takes Pagani's hypo- and hyper-connected subtype gene sets, keeps the 36 genes that overlap HOMER's curated panel (10 hypo, 26 hyper), translates the mouse hypo/hyper gene scores through π into human-parcel space, aggregates to Pagani's 8 networks, and correlates the predicted hyper−hypo Δ with the observed network Δ. Pearson r = +0.822 (p = 0.012), Spearman ρ = +0.524 (not significant), over 8 networks — a small-n, proof-of-concept result that depends on a 36-gene overlap, not a full pathway-spatial test. Per-subject scoring on 817 ABIDE subjects shows ASD on the hypo side of HOMER's template (p = 0.042). A cross-disease gene-spatial check gives r ≈ +0.4 for all four disorders (autism, SCZ, bipolar, ADHD) against Pagani's observed pattern, so the signal is shared psychiatric geometry rather than autism-specific. Showcase: `notebooks/05_pagani_2026_validation.ipynb`.

### Margulies 2016 + Huntenburg 2021 (principal connectivity gradient)

A brain-wide ordering test orthogonal to anchor pairs. Diffusion-map embedding of FC in each species → first non-trivial eigenvector spans sensorimotor → DMN. Translating the mouse gradient through π as a **transport-weighted average** reproduces the observed human gradient at **|r| = 0.402** at parcel resolution (region-level 0.433, n = 1,244 parcels) — 13× the permuted-π null mean (0.031), empirical p = 0.000. An earlier un-normalised `mouse_grad @ π` routing scored only r = 0.144; normalising by π's per-column mass removes a confound and roughly trebles the correlation. Establishes HOMER preserves the cross-species cortical organisational axis brain-wide, not only at the 22 Beauchamp anchor pairs. Showcase: `notebooks/07_margulies_huntenburg_gradient.ipynb`.

### Fulcher 2019 (multimodal cortical gradient, PNAS)

An anchor-orthogonal, *structural* test of whether π carries the mouse cortical hierarchy. Two independent mouse modalities from Fulcher et al. — the T1w:T2w intracortical-myelin proxy (40 areas) and Goulas cytoarchitectural type (38 areas) — are routed through π and compared against the independent HCP S1200 human myelin map at Schaefer-400 region resolution. Both converge: T1w:T2w → human myelin r = +0.373, cytoarchitecture → human myelin r = +0.362, each empirical p = 0.000 against a 200-trial permuted-π null (205 regions). Neither modality is a HOMER input (π is built from FC + SC), so two unrelated structural measurements converging on the same human target rules out a single-measurement artefact. Side finding: π concentrates the 417 mouse isocortical parcels onto only 174 of 400 Schaefer regions, and that territory spans half the brain-wide range of the principal gradient — a quantitative echo of the disproportionate expansion of human association cortex. Showcase: `notebooks/12_fulcher_2019_multimodal_gradient.ipynb`.

### Balsters 2020 (rodent MFC divergence, PNAS)

The one falsification test in the suite. Balsters et al. showed with whole-brain FC that rodent medial frontal cortex does *not* correspond to primate dorsolateral PFC — it resembles premotor cortex. We route HOMER's 46 mouse rodent-MFC parcels (ACAd/ACAv/PL/ILA) through π and ask where the mass lands. Under the recommended π, **0.0 %** reaches human dlPFC (BA9/46) — 0 of 46 parcels argmax there (enrichment ×0.0, p = 0.985) — while premotor, medial PFC and mid-cingulate carry the mass.

This holds *because the recommended composition deliberately excludes the contested Prelimbic→dlPFC anchor*: the `lateral_pfc` pack ships **OFC-only** (see caveat 5 and `docs/04_anchor_packs.md`). Forcing that anchor in (`build_lateral_pfc_region_anchors(..., include_dlpfc=True)`) instead routes 23 % of mouse-MFC mass to human dlPFC by construction — Balsters 2020 is the independent FC evidence behind the decision to leave it out. Showcase: `notebooks/13_balsters_2020_mfc_divergence.ipynb`.

### TransBrain 2025 (sibling-method benchmark, Nat Methods)

An honest methods-landscape comparison rather than a validation. TransBrain (Huang et al. 2025) is a published region-level mouse↔human phenotype translator — a direct sibling of HOMER built on graph embeddings + dual regression. Two tests. On TransBrain's literature-curated homology benchmark — classic mouse↔human homologous region pairs, never seen by HOMER and independent of the Garin anchors — HOMER lands its predicted human centroid 25.3 mm from the literature homolog vs 39.8 mm for the permuted-π null (p < 0.001) — region-neighbourhood accurate, consistent with HOMER's stated ~25–45 mm resolution. The stricter top-3 rank metric on the fine 127-region Brainnetome atlas is 41 % (p < 0.001) — markedly better than the 5-pack model, since the all-15 composition anchors far more cortical territory. In head-to-head translation of a shared mouse phenotype the two methods agree moderately; on the resting-fMRI gradient TransBrain — purpose-built for region-level translation — scores higher (|r| 0.463 vs 0.393). An advanced follow-up adds a clear HOMER strength: on bidirectional cycle-consistency — round-tripping a phenotype mouse→human→mouse, an even-handed ground-truth-free metric — HOMER recovers the original at r ≈ 0.98 across three phenotypes vs ≈ 0.81–0.91 for TransBrain. The frameworks are complementary: TransBrain for region-level phenotype translation, HOMER for soft anchored parcel-level couplings with per-parcel trust tiers and a more internally coherent coupling. Showcase: `notebooks/14_transbrain_2025_benchmark.ipynb`.

### Buckner & Krienen 2013 (tethering hypothesis, TICS)

A negative-control / falsification test. Buckner & Krienen argue human association cortex expanded so much it became evolutionarily "untethered" — implying no clean mouse homologue exists for it, so a faithful coupling should be confident over sensorimotor cortex and sparse over association cortex. If HOMER's π were uniformly confident everywhere, that would signal over-fitting. For every human cortical parcel we measure HOMER's coverage — the total π mass it receives from the mouse brain — along the sensorimotor → association axis (HCP myelin). Coverage collapses toward association cortex: the sensorimotor tertile receives log₁₀ coverage −12.3, the association tertile −19.0 — a gap of 6.7 log units (Mann-Whitney p = 3.4×10⁻⁷; empirical p = 0.000 vs a permuted-axis null). HOMER is *not* confident everywhere — it is dramatically sparser over the association cortex the field says has no mouse homologue. (π's per-parcel entropy is flat — it is the amount of coverage, not its diffuseness, that carries the signal; HOMER's sensorimotor-weighted anchors also contribute to the gap.) Showcase: `notebooks/15_buckner_krienen_2013_tethering.ipynb`.

### Coletta 2020 (cross-species RSN, Sci Adv)

Three sub-tests at network resolution. **Labeled correspondence** (HOMER PAIRID × Yeo-7): 6/10 canonical pairs diagonal-argmax, with olfactory → Limbic at 7.5× null, Salience → Salience at 4.3×. **Data-driven ICA**: weaker (2/7) because ICA components mix anatomical regions. **Network coherence**: 9/11 networks have HOMER-mapped images more compact than permuted-π null (frontoparietal at 0.58× null, frontal_dmn at 0.63×). Combined verdict: HOMER preserves cross-species RSN correspondence under both labeled and coherence tests. Showcase: `notebooks/08_coletta_2020_cross_species_rsn.ipynb`.

### BICCN cell-type markers (Yao 2023 + Siletti 2023)

23 cell-type-defining markers tested via Allen ISH (mouse) → π → AHBA (human). 13 of 23 markers translate at empirical p < 0.05, with a mean Pearson r of +0.089 across all 23. The strongest are glutamatergic and interneuron markers: Drd1 +0.227, Slc17a7 +0.221, Vip +0.201, Calb2 +0.199, Pvalb +0.198, Slc17a6 +0.196, Camk2a +0.182, Grin1 +0.174; weaker but significant are Gad1 +0.092, Drd2 +0.079, Reln +0.069, Plp1 +0.063, Gfap +0.051. By class, glutamatergic markers translate 4/4 (mean +0.193) and interneuron 4/7 (mean +0.107); oligodendrocyte (1/4) and microglia (0/1) markers are mostly null. Showcase: `notebooks/10_biccn_cell_type_markers.ipynb`.

### ENIGMA cross-disorder (Phase 1 in-sandbox; Phase 2 external)

Predicted human spatial patterns at parcel resolution for autism, bipolar, schizophrenia, and ADHD gene sets. **Cross-disorder Pearson r = +0.988 (off-diagonal mean)**. HOMER's per-disorder predictions are essentially identical — confirming the cross-disease specificity finding from Pagani at sharp parcel resolution. Phase 2 (comparison against ENIGMA cortical-thickness Cohen's d maps) scaffolded; runs once ENIGMA Toolbox CSVs are placed in `data_external/enigma/`. Showcase: `notebooks/11_enigma_cross_disorder.ipynb`.

### Whitesell 2021 DMN refinement (methodological note)

Whitesell's broad mouse-DMN (mPFC + ACC + RSC + PPC + dorsal hippocampus + entorhinal) routes through π to give Yeo-DMN 23.9 % — *lower* than Pagani's PAIRID-DMN at 41 %. But the DMN-aligned cortical territory (Yeo-DMN + DorsAttn + Subcortical) gets 54.5 % of Whitesell-DMN mass, well above Pagani's 41 % on Yeo-DMN alone. Interpretation: HOMER preserves Whitesell's broad DMN at the cortical-territory level; Yeo-7 fragments that territory across labels because Schaefer-17 places PPC in DorsAttn and hippocampus has no cortical label. Not a HOMER failure — two valid definitions of "mouse DMN" partitioning differently. We chose NOT to add a `whitesell_dmn` anchor pack because forcing PPC into Yeo-DMN would override the Yeo/Krienen 2011 consensus.

### Hodge 2019 cortical layer markers

Tested CUX1, CUX2, SATB2 (L2/3), RORB (L4), FEZF2 (L5), TBR1, FOXP2 (L6). Six of seven translate at parcel resolution (mean r = +0.119, mean null r ≈ 0): the three upper-layer markers at +0.083, +0.176 and +0.189, L4 at +0.090, L5 at +0.168, and one L6 marker at +0.108; the other L6 marker is not significant (r = +0.019, p = 0.25). The Schaefer-400 parcellation does not separate layers within an area, so this measures the area-level spatial distribution of these layer-marker genes, not within-area lamination. Showcase: `notebooks/06_hodge_2019_layer_markers.ipynb`.

### Pagani per-mouse-model (decoded, awaiting mask)

Pagani's Figura 1c provides 20 mouse models × 1,491-feature matrix. Decoded from `rsfMRI-global-local-connectivity` repo + supplementary methods: features are per-voxel global connectivity (Cole 2009 weighted-degree centrality) within `chd8_functional_template_mask_wo_cerebellum.nii.gz` at ~700 μm isotropic. Per-model HOMER translation needs the mask file for voxel→parcel mapping; data-request email drafted. Once the mask is in hand, ~2 hours of code to produce per-model predictions over HOMER's 2,094 human parcels. Showcase: `notebooks/09_pagani_per_model_translation.ipynb` (currently exploratory; subtype-average resolution).

## Honest caveats — read these

1. **The 100 % top-1 for pack-anchored regions is largely by construction.** The anchor packs use the same mouse-side sets as Beauchamp's validation, and their human-side balls overlap. This is a *deployability* gain, not an *unsupervised recovery* claim. The non-pack-anchored regions (Thalamus, NAc, Hypothalamus, Pallidum, Pons) are where the FGW solver does real work — reaching region rank 1-2/21 using one Garin point anchor each.

2. **Held-out region CV gives the honest "structural recovery" number: 3.4 % top-1, 5.5 % top-5, 6.6 % top-10 (~7× chance).** This is what FC + SC encode about cross-species correspondence *without* the specific anchor for that region. mPFC (33 %), Auditory (22 %), Somatosensory (11 %) recover meaningfully; midbrain / olfactory / striatum recover at chance. HOMER is genuinely supervised; FC + SC alone are too weak to cross the species gap.

3. **Per-parcel correspondence is a region-level claim, not a strict 1:1 statement.** Mean argmax distance is 25-45 mm even in well-anchored regions. Argue at the region level, not the parcel level.

4. **Cerebellum is excluded** from the parcellation (14 of Beauchamp's 36 region pairs cannot be evaluated). Adding cerebellar coverage is a ~1-week scope expansion.

5. **Three packs carry a Beauchamp-metric trade-off; one anchor is excluded as contested.** Cingulate, somatosensory and visual subdivide a Beauchamp-validated region into sub-targets that sit outside Beauchamp's coarse validation ball, so they lower the *parcel-level* Beauchamp score for those regions even though the anchoring is anatomically defensible — they are kept because the broader multi-benchmark evidence (notably TransBrain's region-level homology benchmark) favours including them. Separately, the `lateral_pfc` pack ships **OFC-only**: its Prelimbic→dlPFC entry is excluded from the recommended composition because rodent dlPFC homology is contested (Preuss 1995) and independently contradicted by the Balsters 2020 falsification test. Pass `include_dlpfc=True` to add it back for ablations.

6. **Beauchamp 2022 is a published hypothesis (gene-expression-derived), not absolute ground truth.** Multi-source validation against Mars 2018 white-matter and Coletta 2020 FC partly addresses this — see `docs/archive/iteration_log.md` §5.21.

## How to use this map

Match query granularity to evidence tier:

- **Region-level queries** (`pi[mouse_region_indices, :].sum(axis=0)`, top-K *human regions*) — **trustworthy for all `anchored_and_validated` and `validated_only` parcels (55 % of the brain)**. Recommended query mode.
- **Parcel-level argmax queries** — only reliable for the `anchored_and_validated` tier (31 %).
- **`structural` parcels (13 %)** — treat top-K predictions as hypotheses to verify with other evidence.
- **`low_evidence` parcels (20 %)** — query at region granularity if at all.
- **Avoid** "mouse parcel X = human parcel Y" at the millimeter level. Mean argmax distance is 25-45 mm even in good regions.
