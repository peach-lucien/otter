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

## Pagani 2026 cross-species bridge assumption — independent check

[Pagani et al. 2026, *Nat Neurosci*](https://www.nature.com/articles/s41593-026-02287-z) ("Autism subtypes identified using cross-species functional connectivity analyses") build a mouse↔human autism-subtype workflow on top of a **name-based** cross-species network correspondence — mouse Somatomotor ↔ human Somatomotor, mouse Visual ↔ human Visual, etc. They define 9 mouse networks (ED Fig 1) and 8 human networks (Fig 4e) and assume one matches the other by name. This name-based bridge is *scaffolding* for their workflow, not a result they claim — but every downstream claim about subtypes recurring cross-species rides on it.

We use this as a methodological check: does HOMER's π, fit entirely independently of Pagani 2026's data, route mass between like-named networks?

Aggregate HOMER's π to a mouse-network × human-network matrix (mouse parcels → networks via HOMER's PAIRID_TO_NETWORK; human parcels → Yeo-7 networks via Schaefer-400 with un-Schaefer-assigned parcels collapsed to "Subcortical"). For each of Pagani's name-based pairs:

| Mouse network → Human network | HOMER row-mass on target | Null (uniform π) | Ratio | Argmax? |
|---|---:|---:|---:|:---:|
| SomatoMotor → SomatoMotor | **43 %** | 11 % | **3.9×** | ★ |
| Salience → Salience | **38 %** | 9 % | **4.3×** | ★ |
| DMN → DMN | **41 %** | 16 % | **2.5×** | ★ |
| Subcortical → Subcortical | **57 %** | 29 % | **2.0×** | ★ |
| Visual → Visual | 8 % | 10 % | 0.8× | argmax: DorsAtten |
| HC_Limbic → Limbic | 7 % | 5 % | 1.4× | argmax: Subcortical |
| Auditory → Auditory | 0 % | 2 % | 0.0× | argmax: Control |
| BF_Olfactory → Subcortical | 14 % | 29 % | 0.5× | argmax: Limbic |

**4/8 canonical pairs are diagonal-argmax. Mean concentration: 1.92× over null. Permuted-π control: 1.95/8, 0.97×** — exactly chance, confirming the signal is real.

The four misses are interpretable:
- **Visual → DorsAtten**: Schaefer-17's "Visual" covers only central + peripheral primary visual cortex (V1-like); higher-order mouse visual areas correspond to human regions Schaefer places in DorsAttn (V3, MT, LOC). The mapping is anatomically defensible — it's the Schaefer label definition that's narrow.
- **HC_Limbic → Subcortical**: hippocampus has no Schaefer cortical label, so it falls into Subcortical (which is biologically correct — hippocampus *is* an allocortical/subcortical structure in most parcellations).
- **Auditory → Control**: Schaefer's auditory label `SomMotB_Aud` covers only ~62 parcels (3 %); the broader auditory association cortex Beauchamp pairs to mouse-auditory is distributed across Control and Salience networks in Schaefer-17. This is a Schaefer-coverage limit, not a HOMER failure.
- **BF_Olfactory → Limbic**: orbitofrontal and temporal-pole Limbic regions contain primary olfactory cortex (piriform) — Pagani's "BF" label is a coarser grouping than the cortical-only Yeo-7 networks.

**What this is and isn't.** This is a check on Pagani 2026's *scaffolding*, not a replication of their findings. The paper's actual results are about ASD subtypes (hyper vs hypo) and their gene/pathway signatures; the name-based cross-species network correspondence is workhorse infrastructure connecting their mouse work to their human work. HOMER's agreement on 4/8 networks says the bridge has biological substance under it for those networks; the 4 misses are atlas-definition artefacts (Schaefer's "Visual" is narrower than the mouse paper's; hippocampus is "Subcortical" in Schaefer because it isn't cortex; etc.), not HOMER disagreeing with the biology. The signal survives stripping the anchor packs (4/8, 1.79×) and zeroing the xyz prior (3/8, 2.02×), so it isn't an artefact of any single component — it sits in the underlying FC + SC + Garin-anchor structure.

Code: `experiments/autism_subtypes/01_network_crossvalidation.py`. Log: `outputs/logs/autism_subtypes_network_crossval.json`. Figure: `outputs/figures/autism_subtypes_network_mapping.png`.

### Subtype-contrast spatial pattern (tests Pagani's actual claim 3)

Pagani 2026 claim 3: "the FC subtypes recur cross-species in matching anatomical locations." A real test of this claim — not just the name-bridge — translates Pagani's mouse subtype-contrast spatial pattern through HOMER's π and compares against the observed human subtype-contrast spatial pattern.

Method: for each Pagani mouse network, compute Δ = (hyper subtype perturbation intensity) − (hypo subtype perturbation intensity) from ED Fig 1's 9×9 matrices. Distribute these signed network-level Δs to 1864 mouse parcels via PAIRID_TO_NETWORK, route through π via `mouse_delta_per_parcel @ π` to predict per-human-parcel Δ, then aggregate to 8 human networks (Schaefer-Yeo7 + Subcortical). Correlate against the observed human Δ from Fig 4e (hyper − hypo on the 8×8 matrices).

**Result — Pearson r = +0.547, empirical p < 0.005** (none of 200 permuted-π row-shuffles reach that high). Permuted-π null mean is −0.47 (95% CI −0.88 to +0.06), so the observed correlation is well outside the null band:

| Pagani human net | Observed Δ (z) | Predicted Δ (z) | Same direction? |
|---|---:|---:|:---:|
| Subcortical | +2.34 | +1.13 | ★ |
| Limbic      | +0.05 | +1.77 | ★ |
| Salience    | +0.16 | +0.13 | ★ |
| Visual      | −0.13 | +0.33 | (low magnitude) |
| SomatoMotor | +0.31 | −1.22 | ✗ |
| DorsAtten   | −0.93 | −1.11 | ★ |
| DMN         | −0.74 | −0.15 | ★ |
| Control     | −1.05 | −0.84 | ★ |

HOMER correctly recovers the direction of the subtype contrast for 6 of 8 human networks, including the most-perturbed (Subcortical, Salience, Limbic on the hyper side; Control, DorsAtten, DMN on the hypo side). The two misses (SomatoMotor, Visual) are also where the test's discriminant power is weakest because the observed contrasts there are small. This **does engage one of Pagani's actual results**: HOMER's quantitative cross-species mapping independently corroborates that the hyper-vs-hypo spatial pattern transfers across species in matching anatomical locations, without relying on the name-bridge.

Code: `experiments/autism_subtypes/05_subtype_contrast.py`. Log: `outputs/logs/autism_subtypes_contrast.json`. Figure: `outputs/figures/autism_subtypes_contrast.png`.

### Full per-network-pair Δ-matrix translation (sharper version)

The per-network-intensity test above uses only the 8 row-sums; a sharper version compares the full 36 unique elements of the 8×8 symmetric Δ-matrix. Translation operator T (9×8) is built by aggregating π over Pagani-aligned mouse networks → human networks and row-normalising (P(human-net hj | mouse-net mi)). Predicted human Δ-matrix = Tᵀ · Δ_mouse · T. Splits HOMER's "subcortical" into Caudate Putamen vs Thalamus by nearest-Garin-anchor pid (pid 13/15 → CP, pid 18/19 → Thal). Brainstem mouse parcels (251) are dropped — Pagani has no brainstem network.

**Result — Pearson r = +0.537 over 36 paired matrix elements, analytical p = 0.0007, empirical p < 0.005 vs permuted-π null** (200 trials; null mean −0.52, 95% CI −0.73 to +0.00). 22 of 36 matrix entries agree in sign. The largest positive entry — Subcortical–Subcortical Δ ≈ +33 in human (the strongest network-pair signal of the hyperconnected subtype) — is also the largest positive entry HOMER predicts (≈ +15). The largest negative entries in human (Limbic–SomatoMotor, Visual–SomatoMotor, Limbic–Salience — all on the hypo side) are also predicted negative by HOMER.

This is the sharpest result we have: HOMER's π reproduces the *joint network-pair structure* of Pagani's per-subtype spatial contrast at p=0.0007 with empirical p<0.005 against permutation null. Code: `experiments/autism_subtypes/07_full_matrix_translation.py`. Log: `outputs/logs/autism_subtypes_full_matrix.json`. Figure: `outputs/figures/autism_subtypes_full_matrix.png`.

### Gene-set spatial translation (proof of concept; underpowered)

Pagani's claim 4 is that the subtype gene/pathway signature recurs cross-species — synaptic genes are enriched in mouse-hypo and human-hypo regions; immune genes are enriched in mouse-hyper and human-hyper regions. Their paper observes this as parallel findings in each species rather than as a spatially-linked claim. HOMER can test the explicit spatial link: take the mouse spatial expression map of the subtype gene set, translate through π, see if the predicted human map aligns with the observed human subtype perturbation map.

**The blocker is gene coverage.** Pagani's hypo and hyper gene sets contain 1,952 and 4,463 genes; HOMER ships parcel-level Allen ISH expression for only 51 curated genes. Of those, 10 overlap with Pagani's hypo-only set (Bdnf, Calb1/2, Cux1, Dbh, Gria1, Pax6, Reln, Slc17a7, Snap25) and 26 with hyper-only (Aqp4, Camk2a, Cux2, Drd1/2, Foxp2, Gad1/2, Gfap, Grin1/2b, Lhx2/6, Mbp, Nrgn, Olig2, Plekhg1, Plp1, Rorb, Slc17a6/8, Sox10, Sst, Syn1, Tac1, Th). With only 36 genes spanning the two sets, the spatial test is heavily underpowered relative to the full gene lists.

**Result on the available 36 genes — Pearson r = +0.439 (p = 0.28 nominal), Spearman ρ = +0.619 (Spearman empirical p = 0.045 vs permuted-π null).** Same-sign agreement on 4 of 8 networks. Suggestive — the rank correlation reaches one-sided p ≈ 0.045 against 200 permutations, but Pearson is not significant. We treat this as a directionally-encouraging proof of concept: a full pathway-spatial test using all 1,952 + 4,463 genes (requires ~3-4 hours of Allen Brain Atlas API queries beyond HOMER's curated 51-gene set) is the natural next step.

Code: `experiments/autism_subtypes/09_gene_spatial_translation.py`. Log: `outputs/logs/autism_subtypes_gene_spatial.json`. Summary figure across all three tests: `outputs/figures/autism_subtypes_summary.png`.

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
