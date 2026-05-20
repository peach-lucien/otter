# Results

The headline numbers, the resolution boundary, and the honest caveats. Full per-paper detail lives in the showcase notebooks (`notebooks/05-11`) and per-experiment READMEs.

## Synthesis

HOMER produces a soft probabilistic mouse↔human parcel coupling π (1,864 × 2,094) via Fused Gromov-Wasserstein optimal transport, supervised on 21 Garin 2021 point anchors + 11 region-anchor entries from 7 published cross-species packs. On Beauchamp 2022's external 22-pair gene-expression benchmark, the recommended π hits 37 % parcel-level top-1 (3.1× over baseline) and 100 % qualified top-3 at region level, with 97.8 % bootstrap stability and +17.8 σ above the permuted-anchor null. Independent third-party validation against **eight cross-species papers spanning FC, gene expression, cell types, cortical layers, gradients, and psychiatric disorders** establishes a clean resolution boundary: HOMER preserves cross-species signal at the **regional / area / network level** (Pagani 2026 r=+0.527 on subtype perturbation matrices, Margulies/Huntenburg r=+0.144 on the brain-wide gradient, Coletta 2020 6/10 diagonal-argmax on RSN correspondence, BICCN region-concentrated cell types like Th at r=+0.105) but does not translate **broadly-distributed cortical class markers** (Pvalb, Sst, Vip interneurons), **within-area lamination** (Hodge layer markers), or **disorder-specific signal** (predictions for autism vs schizophrenia vs ADHD correlate at r > 0.97 — HOMER captures a generic psychiatric perturbation geometry, not autism-specific biology).

## Two π files

| File | Anchors | Use when |
|---|---|---|
| `outputs/coupling/pi_fc_plus_SC.npy` | 21 Garin point anchors only | Strict baseline; benchmarking the FGW method itself |
| `outputs/coupling/pi_fc_plus_SC_with_all_packs.npy` | 21 Garin + 11 default region-anchor entries | **Recommended for downstream queries** |

Produced by `pipeline/04_solve_production.py` + `experiments/anchor_packs/compose_all.py`.

## Headline numbers

Beauchamp 2022 external validation (15 anchor-overlapping mouse↔human pairs, 927 parcels):

| Metric | Point anchors only | + all default packs | Δ |
|---|---:|---:|---:|
| **Top-1** | 12 % | **37 %** | ×3.1 |
| **Top-5** | 22 % | **46 %** | ×2.1 |
| **Top-10** | 27 % | **50 %** | ×1.8 |
| **Mean rank / 2094** | 871 | **85** | **×10** (lower is better) |

Region-level evaluation (Beauchamp-22 candidate set): qualified top-1 jumps **37 % → 82 %**, qualified top-3 **70 % → 100 %**, mean fold enrichment **16× → 123×**.

Bootstrap argmax stability over 40 subject-resamples: **97.8 %** (88 % of mouse rows have identical argmax across all 40 resamples).

z-score vs permuted-anchor null: **+17.8** — the specific mouse↔human pairings matter, not just having any 42 anchor constraints.

## Per-region trust tiers

Each of 1,864 mouse parcels gets a 5-tier evidence label combining anchor membership, Beauchamp top-1, bootstrap stability, concentration, and FC similarity:

| Tier | n | % | What it means |
|---|---:|---:|---|
| **anchored_and_validated** | 354 | 19 % | In an anchor pack AND Beauchamp top-1 > 0 — *highest confidence* |
| **anchored_only** | 65 | 4 % | In an anchor pack, no Beauchamp validation pair (e.g. OFC, AON, RSC) |
| **validated_only** | 665 | 36 % | Beauchamp top-1 > 0, no specific anchor pack |
| **structural** | 233 | 13 % | High internal trust but no external evidence |
| **low_evidence** | 547 | 29 % | Use with caution — no supervision, weak internal signal |

```python
trust = np.load("outputs/coupling/trust_multisource_all_packs.npz", allow_pickle=True)
reliable_parcels = np.where(trust["evidence_tier"] == "anchored_and_validated")[0]
```

## Per-region performance — parcel-level vs region-level

HOMER returns a probability distribution over 2,094 human parcels per mouse parcel. The "did the argmax hit Beauchamp's target?" metric undersells what's actually happening; the region-level columns show where the *mass* goes.

| Region | parcel top-1 | parcel top-5 | region rank / 21 | fold enrichment | Trust tier |
|---|---:|---:|---:|---:|---|
| **Pack-anchored** (9 regions) | | | | | |
| Motor / Tectum / Piriform / Amygdala / Hipp subfields | 100 % | 100 % | 1 | 47–1047× | anchored_and_validated |
| **Garin point-anchor only** (10 regions) | | | | | |
| Thalamus | 30 % | 48 % | 1 | 29× | validated_only |
| Striatum-ventral → NAc | 8 % | 42 % | 1 | 100× | validated_only |
| Auditory → Heschl's | 22 % | 22 % | 1 | 26× | validated_only |
| Somatosensory → postcentral | 19 % | 37 % | 1 | 10× | validated_only |
| Anterior cingulate | 13 % | 22 % | 1 | 11× | validated_only |
| Caudate | 13 % | 27 % | 1 | 11× | validated_only |
| Hypothalamus | 12 % | 17 % | 2 | 60× | validated_only |
| Visual → cuneus | 7 % | 7 % | 1 | 4× | validated_only |
| Pallidum | 5 % | 9 % | 2 | 16× | validated_only |
| Pons | 3 % | 3 % | 2 | 10× | validated_only |

**Pack-anchored regions** = trustworthy at parcel granularity (largely *by construction* — see caveat 1). **Garin-point-anchor regions** = trustworthy at *region* granularity (rank 1-2/21 with 4-100× fold enrichment), mediocre at parcel granularity. **Unanchored / low-evidence** = not trustworthy. The multi-source trust map gates this per-parcel; see "How to use this map" below.

Additionally, when mouse parcels are grouped by their nearest Garin network, the **olfactory** and **limbic** networks become substantially more compact in human space after packs are applied (median pairwise distance −17.7 mm for olfactory, −12.8 mm for limbic) — non-Beauchamp evidence that the packs encode coherent biology.

## Independent third-party validation

We tested HOMER against 8 cross-species neuroscience papers using completely independent data (no overlap with Beauchamp's transcriptomic-similarity dataset or HOMER's anchor inputs). Each test asks a different cross-species question at a different granularity. The combined picture is a clean resolution-boundary statement.

| Paper | Test | Result | Verdict |
|---|---|---|---|
| **Pagani 2026** (Nat Neurosci) | Subtype FC perturbation matrix via π | **r = +0.527** on 36 network-pair Δ, p = 0.0009, empirical p < 0.005 vs permuted-π null | **Strong** |
| Pagani 2026 | ABIDE per-subject scoring | p = 0.042, Cliff's δ = −0.083 (ASD < CTRL) | Modest direction |
| Pagani 2026 | Cross-disease gene-set spatial | autism r ≈ ADHD r ≈ SCZ r ≈ +0.43 | **No disorder-specificity** |
| **Margulies 2016 + Huntenburg 2021** | Principal connectivity gradient | r = +0.144, ρ = +0.343, **p = 4 × 10⁻¹¹** (10× null) | **Strong (brain-wide)** |
| **Coletta 2020** (Sci Adv) | Labeled mouse-net → Yeo-7 + coherence | **6/10 diagonal-argmax**, **9/11 nets compact** vs null | **Strong** |
| **BICCN** (Yao 2023 + Siletti 2023) | Cell-type marker spatial | Region-concentrated cells (Th, Aqp4, Plp1) at r ≈ +0.05-0.10, p < 0.001 | **Strong (regional)** |
| BICCN | Broadly-cortical interneurons | Pvalb, Sst, Vip null | Null — pan-cortical limit |
| Whitesell 2021 (Neuron) | DMN refinement | Yeo-DMN 23.9 %; DMN-aligned cortical territory 54.5 % | Methodological note |
| **ENIGMA Phase 1** | Cross-disorder predictions at parcel level | Off-diagonal r = +0.987 (autism ≈ SCZ ≈ ADHD ≈ bipolar) | **Confirms no disorder-specificity** |
| Hodge 2019 (Nature) | Cortical layer markers | 6/7 markers null; only RORB (L4) significant | Null — laminar limit |
| Pagani per-model | 20 mouse models × 1,491 features | Decoded as per-voxel global-connectivity; voxel-ordering pending mask file | Pending external data |

Each validation has its own showcase notebook (see `notebooks/05–11`) with the full method, figure, and discussion. Brief snapshots below.

### Pagani 2026 (autism subtypes, Nat Neurosci)

Four-hypothesis arc against the paper's claims. **Strongest result**: translating the mouse subtype Δ-matrix through π reproduces the human subtype Δ-matrix at r = +0.527 over 36 network-pair elements (analytical p = 0.0009; empirical p < 0.005 against permuted-π null) — independent quantitative replication of their claim 3 (FC subtypes recur cross-species at matching anatomical locations) without using their name-based bridge. Per-subject scoring on 817 ABIDE subjects shows ASD on the hypo side of HOMER's template (p = 0.042). Gene-spatial cross-disease check breaks the autism-specific framing — all 4 brain disorders (autism, SCZ, bipolar, ADHD) give r ≈ +0.4 against Pagani's observed pattern, so HOMER captures shared psychiatric geometry, not autism-specific biology. Showcase: `notebooks/05_pagani_2026_validation.ipynb`.

### Margulies 2016 + Huntenburg 2021 (principal connectivity gradient)

A brain-wide ordering test orthogonal to anchor pairs. Diffusion-map embedding of FC in each species → first non-trivial eigenvector spans sensorimotor → DMN. HOMER's translation of the mouse gradient through π correlates with the observed human gradient at Pearson r = +0.144 (Spearman ρ = +0.343, analytical p = 4 × 10⁻¹¹ at n = 2,094) — 10× the permuted-π null mean (0.015) and well outside the null 95 % CI. Establishes HOMER preserves the cross-species cortical organisational axis brain-wide, not only at the 22 Beauchamp anchor pairs. Showcase: `notebooks/07_margulies_huntenburg_gradient.ipynb`.

### Coletta 2020 (cross-species RSN, Sci Adv)

Three sub-tests at network resolution. **Labeled correspondence** (HOMER PAIRID × Yeo-7): 6/10 canonical pairs diagonal-argmax, with olfactory → Limbic at 7.5× null, Salience → Salience at 4.3×. **Data-driven ICA**: weaker (2/7) because ICA components mix anatomical regions. **Network coherence**: 9/11 networks have HOMER-mapped images more compact than permuted-π null (frontoparietal at 0.58× null, frontal_dmn at 0.63×). Combined verdict: HOMER preserves cross-species RSN correspondence under both labeled and coherence tests. Showcase: `notebooks/08_coletta_2020_cross_species_rsn.ipynb`.

### BICCN cell-type markers (Yao 2023 + Siletti 2023)

23 cell-type-defining markers tested via Allen ISH (mouse) → π → AHBA (human) comparison. **Regionally-concentrated cells translate**: Th (dopaminergic, midbrain) r = +0.105 (p < 0.001), Aqp4 (astrocyte) r = +0.080, Plp1 (oligodendrocyte) r = +0.058, Slc6a3 (DAT) r = +0.061, Olig2 r = +0.052. **Broadly-cortical class markers don't**: Pvalb, Sst, Vip, Camk2a all null. Establishes that HOMER preserves region-localised signals but not pan-cortical class distributions. Showcase: `notebooks/10_biccn_cell_type_markers.ipynb`.

### ENIGMA cross-disorder (Phase 1 in-sandbox; Phase 2 external)

Predicted human spatial patterns at parcel resolution for autism, bipolar, schizophrenia, and ADHD gene sets. **Cross-disorder Pearson r = +0.987 (off-diagonal mean)**. HOMER's per-disorder predictions are essentially identical — confirming the cross-disease specificity finding from Pagani at sharp parcel resolution. Phase 2 (comparison against ENIGMA cortical-thickness Cohen's d maps) scaffolded; runs once ENIGMA Toolbox CSVs are placed in `data_external/enigma/`. Showcase: `notebooks/11_enigma_cross_disorder.ipynb`.

### Whitesell 2021 DMN refinement (methodological note)

Whitesell's broad mouse-DMN (mPFC + ACC + RSC + PPC + dorsal hippocampus + entorhinal) routes through π to give Yeo-DMN 23.9 % — *lower* than Pagani's PAIRID-DMN at 41 %. But the DMN-aligned cortical territory (Yeo-DMN + DorsAttn + Subcortical) gets 54.5 % of Whitesell-DMN mass, well above Pagani's 41 % on Yeo-DMN alone. Interpretation: HOMER preserves Whitesell's broad DMN at the cortical-territory level; Yeo-7 fragments that territory across labels because Schaefer-17 places PPC in DorsAttn and hippocampus has no cortical label. Not a HOMER failure — two valid definitions of "mouse DMN" partitioning differently. We chose NOT to add a `whitesell_dmn` anchor pack because forcing PPC into Yeo-DMN would override the Yeo/Krienen 2011 consensus.

### Hodge 2019 cortical layer markers (informative null)

Tested CUX1, CUX2, SATB2 (L2/3), RORB (L4), FEZF2 (L5), TBR1, FOXP2 (L6). Cross-species translation null for 6 of 7; only RORB significant (r = +0.07, empirical p = 0.002) — because L4 is concentrated in primary sensory cortices (V1/S1/A1) which HOMER anchors strongly bind, so it gets translated as an area signal not a laminar one. Cortex-only and layer-group composite refinements don't recover the others. **Establishes HOMER's lower resolution boundary**: HOMER works at the area / network level but does NOT preserve within-area laminar structure because the Garin anchors are area-level, not layer-level. Showcase: `notebooks/06_hodge_2019_layer_markers.ipynb`.

### Pagani per-mouse-model (decoded, awaiting mask)

Pagani's Figura 1c provides 20 mouse models × 1,491-feature matrix. Decoded from `rsfMRI-global-local-connectivity` repo + supplementary methods: features are per-voxel global connectivity (Cole 2009 weighted-degree centrality) within `chd8_functional_template_mask_wo_cerebellum.nii.gz` at ~700 μm isotropic. Per-model HOMER translation needs the mask file for voxel→parcel mapping; data-request email drafted. Once the mask is in hand, ~2 hours of code to produce per-model predictions over HOMER's 2,094 human parcels. Showcase: `notebooks/09_pagani_per_model_translation.ipynb` (currently exploratory; subtype-average resolution).

## Honest caveats — read these

1. **The 100 % top-1 for pack-anchored regions is largely by construction.** The anchor packs use the same mouse-side sets as Beauchamp's validation, and their human-side balls overlap. This is a *deployability* gain, not an *unsupervised recovery* claim. The non-pack-anchored regions (Thalamus, Auditory, S1, ACG, Caudate, NAc, Hypothalamus, Visual, Pallidum, Pons) are where the FGW solver does real work — reaching rank 1-2/21 with 4-100× fold enrichment using one Garin point anchor each.

2. **Held-out region CV gives the honest "structural recovery" number: 3.4 % top-1, 5.5 % top-5, 6.6 % top-10 (~7× chance).** This is what FC + SC encode about cross-species correspondence *without* the specific anchor for that region. mPFC (33 %), Auditory (22 %), Somatosensory (11 %) recover meaningfully; midbrain / olfactory / striatum recover at chance. HOMER is genuinely supervised; FC + SC alone are too weak to cross the species gap.

3. **Per-parcel correspondence is a region-level claim, not a strict 1:1 statement.** Mean argmax distance is 25-45 mm even in well-anchored regions. Argue at the region level, not the parcel level.

4. **Cerebellum is excluded** from the parcellation (14 of Beauchamp's 36 region pairs cannot be evaluated). Adding cerebellar coverage is a ~1-week scope expansion.

5. **The cingulate and lateral PFC packs are opt-in.** Cingulate hurts the Beauchamp ACG metric by anatomical design (subgenual vs pregenual ACC); dlPFC homology is contested (Carlén 2017 vs Preuss 1995). Both packs are biologically defensible but ship as opt-in.

6. **Beauchamp 2022 is a published hypothesis (gene-expression-derived), not absolute ground truth.** Multi-source validation against Mars 2018 white-matter and Coletta 2020 FC partly addresses this — see `docs/archive/iteration_log.md` §5.21.

## How to use this map

Match query granularity to evidence tier:

- **Region-level queries** (`pi[mouse_region_indices, :].sum(axis=0)`, top-K *human regions*) — **trustworthy for all `anchored_and_validated` and `validated_only` parcels (55 % of the brain)**. Recommended query mode.
- **Parcel-level argmax queries** — only reliable for the `anchored_and_validated` tier (19 %).
- **`structural` parcels (13 %)** — treat top-K predictions as hypotheses to verify with other evidence.
- **`low_evidence` parcels (29 %)** — query at region granularity if at all.
- **Avoid** "mouse parcel X = human parcel Y" at the millimeter level. Mean argmax distance is 25-45 mm even in good regions.
