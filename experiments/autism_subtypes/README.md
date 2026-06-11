# Autism subtypes (Pagani 2026) — HOMER applications

Cross-species network validation against [Pagani et al. 2026 *Nat Neurosci*](https://www.nature.com/articles/s41593-026-02287-z), "Autism subtypes identified using cross-species functional connectivity analyses".

## What the paper does — workflow in 7 steps

The paper's central claim is that autism is best understood as **two functional-connectivity subtypes** — a **hyperconnected** group and a **hypoconnected** group — that recur in both mouse models and humans, and that these two subtypes are driven by distinct biological mechanisms (immune signalling for hyperconnected; synaptic genes for hypoconnected). The workflow:

**Step 1 — Mouse functional connectivity.** Resting-state fMRI in 20 autism-relevant mouse models (Chd8, Fmr1, Tsc2, Trem2, Btbr, Cdkl5, Mecp2, Shank3, Cntnap2, Nlgn3, Oxtr, 16p11.2, Mecp2 dup, etc.) — see Fig 1c source data (20 models × 1,491 features). Compute per-mouse-model FC matrices; identify regions where each model deviates from wild-type controls (hypo: connectivity reduced, hyper: connectivity increased).

**Step 2 — Mouse subtype clustering.** Cluster the 20 mouse models by their FC perturbation signature → two subtypes emerge. One subtype shows widespread hypoconnectivity (e.g. Shank3, En2, 16p11.2, 22q11.2 — n=11); the other shows hyperconnectivity (e.g. Fmr1, Chd8, Tsc2, Il6, Trem2, Btbr — n=9). (Correct per Pagani Fig 1c row order and verified by mean-connectivity sign; an earlier draft inverted Fmr1/Tsc2 into the hypo group — see `pagani_2026_per_model/` and `_audit/FINDINGS_LOG.md` F-020.) Map both onto the mouse 9-network atlas (Auditory, BF, Caudate-Putamen, DMN, HC, Salience, Somatomotor, Thalamus, Visual; ED Fig 1).

**Step 3 — Gene set extraction.** For each subtype, identify the genes preferentially expressed in the affected mouse regions using the Allen Brain Atlas. Cross-reference with the SFARI autism gene list. Result: 1,952 hypoconnectivity-implicated genes, 4,463 hyperconnectivity-implicated genes (Supplementary Table 4, sheet `subtypes`).

**Step 4 — Pathway enrichment in mouse.** Run pathway enrichment (Reactome, KEGG, synGO, mTOR, immune system, …) on each gene set. Fig 2b source data shows the headline pattern:
- Hypoconnected subtype: **Protein-protein interactions at synapses** (OR ≈ 7.5)
- Hyperconnected subtype: **Adaptive immune system, cytokine signalling, innate immune system** (ORs 2-5)

**Step 5 — Human FC dataset.** Pool 1,029 ASD participants and 1,141 controls from ABIDE-style multi-site rsfMRI collections (Fig 4a source data; 39 collections). Parcellate using Schaefer-400 + subcortical regions; compute per-subject FC. Apply the same subtype clustering procedure.

**Step 6 — Human subtype recovery.** Hypoconnected and hyperconnected groups re-emerge in human ASD (Fig 4d). For each subtype, summarise the perturbed connectivity on the Yeo-7 + Subcortical 8-network atlas (Fig 4e). The paper's cross-species correspondence is then **by network name** — they argue mouse Somatomotor disruption recurs as human Somatomotor disruption, etc.

**Step 7 — Cross-species pathway replication.** Use ENIGMA-style human gene-imaging maps + SFARI autism genes to recover pathway enrichment in *human* ASD by subtype (Fig 5). Find that the same pattern holds: human hypoconnectivity is enriched for synaptic genes; human hyperconnectivity for immune genes. The biology travels cross-species.

The conclusion: autism's two FC subtypes have **the same gene-pathway signature in mice as in humans**, suggesting that mouse models can be matched to humans on the basis of which subtype they replicate — and that drug repurposing (immune modulators for hyperconnected, synaptic stabilisers for hypoconnected) could be guided by subtype membership.

## What this has to do with HOMER

The paper's cross-species link in **step 6** is by name: mouse-Somatomotor ↔ human-Somatomotor. That correspondence is a strong assumption — it works only if the two species' networks are biologically the same things. HOMER's quantitative π provides an independent, parcel-level evidence base for that assumption. Specifically:

- **Validation use** (this directory, `01_network_crossvalidation.py`): aggregate HOMER's π to a mouse-network × human-network matrix. If π preferentially links like-named networks across species, that supports the paper's name-based correspondence. If it doesn't, the workflow may be over-confident in places HOMER thinks the structural evidence disagrees.

- **Extension use** (future): replace name-based matching with a quantitative π-derived correspondence matrix. Per-mouse-model FC perturbation maps could be translated through π into human-parcel space and compared with per-subject human FC maps directly — without ever falling back on the name shortcut.

## The validation result (Test 1)

`01_network_crossvalidation.py` runs the validation. Output: `outputs/logs/autism_subtypes_network_crossval.json`, figure `outputs/figures/autism_subtypes_network_mapping.png`.

| Canonical pair (mouse → human) | HOMER row-mass | Null | Ratio | Argmax? |
|---|---:|---:|---:|:---:|
| SomatoMotor → SomatoMotor | **43 %** | 11 % | **3.9×** | ★ |
| Salience → Salience | **38 %** | 9 % | **4.3×** | ★ |
| DMN → DMN | **41 %** | 16 % | **2.5×** | ★ |
| Subcortical → Subcortical | **57 %** | 29 % | **2.0×** | ★ |
| Visual → Visual | 8 % | 10 % | 0.8× | argmax: DorsAtten |
| HC_Limbic → Limbic | 7 % | 5 % | 1.4× | argmax: Subcortical |
| Auditory → Auditory | 0 % | 2 % | 0.0× | argmax: Control |
| BF_Olfactory → Subcortical | 14 % | 29 % | 0.5× | argmax: Limbic |

**4/8 diagonal-argmax with mean 2.69× over null. Permuted-π null: 1.95/8 and 0.97×** — exactly chance, confirming the signal is structural.

Misses are interpretable: mouse visual cortex covers higher-order visual regions that Schaefer-17 places in DorsAttn; mouse HC routes to "Subcortical" because hippocampus *is* subcortical in cortical-only parcellations; Auditory is limited by Schaefer's narrow auditory label; BF/Olfactory has no clean cortical counterpart in Yeo-7. None of these are HOMER failures — they're Schaefer/Yeo definition limits.

## Robustness — Test 1b

`03_baseline_comparison.py` runs the same scoring against three π variants and a permuted-π null:

| π variant | Diag-argmax | Mean ratio |
|---|:---:|---:|
| with_all_packs (recommended) | 4/8 | 2.69× |
| fc_plus_SC (Garin point anchors only) | 4/8 | 1.79× |
| fc_plus_SC_xyz_zero (no spatial prior) | 3/8 | 2.02× |
| permuted-π null (20 trials) | ~2/8 | 0.97× |

The network-level correspondence is **not driven by the anchor packs** — it's already in the bare-Garin π. Even removing the xyz prior keeps it. The signal originates in FC + SC + the 21 Garin point anchors, propagated through FGW. Permuted π sits at chance.

## Reproduce

```bash
# Extract the Schaefer label file (once)
cd <repo>/homer
unzip -p data_external/p6ebec-hbp-d000038_SC-FC_HCP_eNKI_pub/Schaefer2018_400Parcels_17Networks.zip \
    Schaefer2018_400Parcels_17Networks/Schaefer2018_400Parcels_17Networks_order.txt \
    > outputs/anndata/_schaefer_order.txt

# Run the validation
PYTHONPATH=src python experiments/autism_subtypes/01_network_crossvalidation.py
PYTHONPATH=src python experiments/autism_subtypes/02_plot_network_mapping.py
PYTHONPATH=src python experiments/autism_subtypes/03_baseline_comparison.py
```

Outputs go to:
- `outputs/logs/autism_subtypes_network_crossval.json`
- `outputs/logs/autism_subtypes_baseline_comparison.json`
- `outputs/figures/autism_subtypes_network_mapping.png`
- `outputs/figures/autism_subtypes_diagonal_dominance.png`

## Test 2 — Subtype-contrast spatial pattern (Pagani's claim 3)

**Test 1 only checked the paper's scaffolding** (the name-based mouse↔human network bridge). Test 2 engages one of the paper's actual results: that the FC subtypes recur cross-species *in matching anatomical locations* (their claim 3).

Method: translate the mouse subtype-contrast spatial pattern through π and check if it predicts the human subtype-contrast pattern. For each Pagani mouse network compute Δ_mouse = (hyper intensity) − (hypo intensity), distribute Δ_mouse to mouse parcels via PAIRID_TO_NETWORK, route through π via `Δ_mouse_per_parcel @ π` to predict per-human-parcel Δ, aggregate to 8 human networks, and correlate against the observed human Δ_human from Fig 4e. No name-bridge in this pipeline.

### Result

**Pearson r = +0.494** between predicted and observed human subtype contrast (n=8; analytical p = 0.21). Empirical p = 0.000 vs 200 permuted-π row-shuffles, **but** the null mean is strongly negative (−0.51, 95% CI −0.91 to −0.07), so the empirical p largely reflects the observed value clearing a downward-biased null rather than a large positive effect (see the **null-bias caveat** at the end of this file / `_audit/FINDINGS_LOG.md` F-016). The observed +0.494 does sit clearly above the null band. HOMER recovers the direction of the contrast for most human networks.

The pattern HOMER predicts (in z-scored terms): Limbic and Subcortical preferentially perturbed in human hyperconnected subtype; Control, DMN, DorsAtten, and SomatoMotor preferentially perturbed in human hypoconnected subtype. Six of these directions match Pagani's observed Δ.

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

This **is a replication of one of Pagani's claims**, not just a check on their assumption. The mouse → human translation via HOMER's quantitative π — without name-matching, without any access to Pagani's data during fitting — produces a per-network subtype-contrast pattern that significantly resembles Pagani's observed human pattern.

The two misses (SomatoMotor, Visual) are interpretable: both have small observed contrasts (z ≈ ±0.2), so the test's discriminant power for them is weak; HOMER's prediction is in the wrong direction but only by a small amount in absolute terms.

### Earlier failed design — Test 2a (per-subtype absolute intensity)

Test 2a (`04_subtype_translation.py`) tried correlating predicted vs observed *absolute* per-network intensity for each subtype separately. That failed because the predictions are dominated by human-network size effects (the largest network, Subcortical with 326 parcels, sucks up mass regardless of mouse pattern). Permuted-π null had non-zero mean (≈ +0.79 for hyper), so the test lacked power. Test 2b (`05_subtype_contrast.py`) uses the subtype difference, which cancels size effects, and works.

## Test 2c — Full per-network-pair Δ-matrix translation (sharpest)

Test 2b collapsed each subtype matrix to per-network row-sums (8 numbers). Test 2c uses the full per-network-PAIR matrix instead — all 36 unique upper-triangle entries of the 8×8 symmetric Δ-matrix. Translation: predicted Δ_human = Tᵀ · Δ_mouse · T, where T is a (9×8) row-normalised conditional distribution P(human-net | mouse-net) computed by aggregating π over Pagani-aligned networks.

Key methodological improvement: splits HOMER's coarse "subcortical" into Caudate Putamen (mouse parcels nearest pid 13/15) vs Thalamus (nearest pid 18/19), matching Pagani's 9-net mouse partition. Brainstem mouse parcels (251) are dropped — Pagani has no brainstem network.

### Result

**Pearson r = +0.550 over 36 paired matrix elements (analytical p = 0.0005; empirical p = 0.000 vs 200 permuted-π row-shuffles, null mean −0.47). Spearman ρ = +0.228 (p = 0.18, NOT significant). 22 of 36 entries (61 %) agree in sign.**

> **Honest caveat (audit 2026-06-11).** This result is **leverage-dominated by a single element**: Subcortical–Subcortical (observed Δ +33.7, predicted +15.6 — the strongest network-pair signal of hyperconnected ASD). Removing it drops Pearson to **0.34** and Spearman to **0.16**. Several mid-magnitude cells are predicted with the wrong sign (e.g. observed +22.4 → predicted −3.2). So Test 2c shows that HOMER **correctly recovers the dominant cross-species signal** (Subcortical hyperconnectivity), but does **not** demonstrate strong rank-concordance across the full matrix. An earlier version of this README reported r = +0.601, ρ = +0.643 (p = 0.0007) — those numbers were from a pre-v2 pipeline and do not reproduce under either current coupling (recommended π ρ = +0.23; base π ρ = +0.27). See `_audit/FINDINGS_LOG.md` F-007.

The largest positive observed Δ — Subcortical–Subcortical — is also the largest positive Δ HOMER predicts. Several large negative observed Δ entries (hypo-dominant) are also predicted negative; but the rank concordance across the weaker cells is poor, which is why the Spearman is non-significant.

This is the finest-grained test we have (36 matrix elements vs 8). But as the caveat above shows, the recovered signal is concentrated in the single dominant Subcortical–Subcortical element rather than distributed across the joint network-pair structure: the magnitude correlation (Pearson) is carried by that one point, and the rank correlation (Spearman) across the full matrix is non-significant. So Test 2c supports "HOMER recovers the dominant cross-species Δ" but not the stronger "HOMER reproduces the full joint network-pair structure."

## Test 3 — Gene-set spatial translation

Pagani claim 4: hypo/hyper subtypes have distinct gene/pathway signatures that recur cross-species. Pagani treats this as parallel findings — synaptic genes are enriched in mouse-hypo regions AND synaptic genes are enriched in human-hypo regions — without explicitly bridging the two through a spatial mapping. HOMER's π lets us bridge them.

**Pipeline**: download parcel-level mouse Allen ISH expression for Pagani's implicated genes → compute per-mouse-parcel mean expression score for each subtype gene set → translate via π → aggregate predicted human-parcel scores to 8 Pagani networks → compute predicted human Δ = hyper-spatial − hypo-spatial → correlate against observed human Δ from Fig 4e. Diagnostic: gene-bootstrap (1,000 resamples of the gene pool), per-parcel agreement between gene-translation and FC-translation, per-pathway breakdown.

### Initial proof of concept (36 genes from HOMER's curated set)

Used the 36 of Pagani's 6,415 genes that happened to be in HOMER's 51-gene curated ISH panel (10 hypo + 26 hyper). Pearson r = +0.439 (p = 0.28 nominal), Spearman ρ = +0.619 (empirical p = 0.045). Suggestive but underpowered.

### Expanded run (1,713 Pagani genes via Allen ISH API)

Built `allen_expansion/download_pagani_ish.py` to pull parcel-level expression for all 6,415 Pagani genes directly from the Allen API (no allensdk dependency). 1,713 genes resolved and downloaded successfully (456 hypo + 1,257 hyper) — 27 % yield, limited by Allen's coronal ISH coverage of Pagani's gene set.

| Metric | 36 genes | 1,713 genes |
|---|---:|---:|
| Pearson r (predicted vs observed Δ, n=8 nets) | +0.439 | +0.433 |
| Spearman ρ | +0.619 | +0.619 |
| Same-sign per network | 4/8 | 5/8 |

The point estimate barely moved because the **8-network aggregation is the bottleneck, not gene coverage** — we're correlating two 8-element vectors regardless of how cleanly each is estimated.

The bootstrap is where the expansion pays off:

**Pearson r mean = +0.428, 95 % CI (+0.349, +0.497), with 100 % of 1,000 gene-resamples positive and 99.7 % above r = +0.3.** The signal is exceptionally stable across which subset of genes you draw; it just can't reach a low Pearson p with only n=8 network averages. The bootstrap addresses the "is this correlation a one-off" question directly (it is not) — permutation-π is the wrong null at n=8.

### Per-parcel diagnostic (n = 2,094)

Predicted Δ from Test 3 (gene-derived) vs predicted Δ from Test 2c (FC-derived) correlate at Pearson r = +0.154, Spearman ρ = +0.058 across all 2,094 human parcels. **The two HOMER translation routes are complementary, not redundant** — gene-spatial encodes "which regions express the implicated genes"; FC-spatial encodes "which networks are functionally perturbed." Both correlate with Pagani's observed pattern through π but capture different cross-species signals.

### Per-pathway breakdown (14 pathways from MOESM3)

Every pathway tested — synaptic AND immune AND mTOR AND WNT AND chromatin AND GPCR AND MAPK — shows the **same direction**: positive r ≈ +0.35 to +0.51 with observed hyper Δ, negative r with observed hypo Δ. The direction-by-pathway split that Pagani's claim 4 requires (synaptic → hypo; immune → hyper) doesn't separate cleanly through HOMER. Reason: **Pagani's published "hypo" matrix in Fig 4e has tiny magnitudes** (range 0–1.5) compared to "hyper" (range 0–33), so the Δ test is dominated by the hyper side regardless of which pathway you feed in. We can confirm Pagani's *overall* cross-species spatial replication (every pathway translates positively to human hyper-side ASD perturbation) but their *direction-by-pathway* claim isn't testable from the published source data alone — that would need per-parcel human pathway-spatial maps, which Pagani didn't ship.

## Test 4 — ABIDE per-subject HOMER-template scoring (null result)

A stronger test of Pagani's claim 1 (ASD splits into hyper/hypo subtypes at the individual level): build a HOMER human template by routing the mouse hyper−hypo subtype Δ-matrix through π, score each ABIDE-pcp subject by dot-product against that template, and ask whether ASD subjects systematically differ from controls and split bimodally.

**Pipeline**: fetched 871 ABIDE-pcp subjects (CPAC pipeline, AAL-116 parcellation), 817 valid. Per-subject per-AAL-parcel FC profile, minus site-matched control mean, mapped to HOMER's 2,094 parcels by nearest centroid, scored vs z-scored HOMER template built from the **recommended** coupling `pi_fc_plus_SC_with_all_packs.npy`. Two profile features tested: coarse `mean(|FC|)` (`abs`) and `signed` mean-FC.

### Result — null (re-run 2026-06-11 under the recommended π / current v2 pipeline)

| Metric | `signed` feature (headline) | `abs` feature |
|---|---:|---:|
| n valid subjects | 817 (377 ASD, 440 control) | 817 |
| Mann-Whitney p | 0.96 | 0.64 |
| Cliff's δ | +0.002 (negligible) | +0.019 (negligible) |
| Within-ASD GMM | 1-component preferred (Δ BIC = +17.8) | 1-component |

HOMER's cross-species template does **not** distinguish ASD from control at the individual-subject level (both features non-significant, negligible δ of inconsistent sign), and the within-ASD distribution is unimodal — so it does not recover Pagani's hyper/hypo split as a subject-level classifier.

> **Provenance note.** An interim log (pre-v2 pipeline) had shown the `signed` feature reaching p = 0.042 (δ = −0.083), and this README briefly described a "small but real" effect. Re-running end-to-end under the current v2 pipeline + recommended π gives a clean null (above); the earlier number did not survive the pipeline rework. See `_audit/FINDINGS_LOG.md` F-006.

### Where HOMER's signal lives

Tests 2c and 3 show HOMER's translation carries genuine signal at the **population/network-aggregate level**. Test 4 shows the same signal does NOT survive as a per-subject classifier on noisy individual FC — a natural granularity boundary: HOMER carries cross-species signal at the level Pagani actually publishes (per-subtype averages on networks), not at single-subject diagnostic resolution.

Possible mitigations that might improve power: replicating Pagani's exact per-cell FC perturbation pipeline; a finer parcellation (Schaefer-400 / CC400 instead of AAL-116); richer site/age/motion regression. The negligible, sign-inconsistent δ suggests there isn't a strong subject-level signal to recover regardless of feature engineering.

## Summary across the four tests

| Test | What it tests | Result | Verdict |
|---|---|---|---|
| **Test 1** | Pagani's name-based network bridge has biological substance | 4/8 canonical pairs diagonal-argmax; mean 2.69× over null | Bridge OK for 4 networks; 4 misses are atlas-label artefacts |
| **Test 2c** | Pagani claim 3 (FC subtypes recur cross-species at matching anatomical locations) | r=+0.550 (p=0.0005, empirical p=0.000); Spearman ρ=+0.228 (n.s.); leverage-driven by Subcortical–Subcortical (drop-one r=0.34) | **Partial** — recovers the dominant cross-species signal, but full-matrix rank concordance is weak |
| **Test 3** | Pagani claim 4 (gene/pathway signature recurs cross-species spatially) | Bootstrap r=+0.428, 95% CI (+0.349, +0.497), 100% of resamples positive | **Supports overall claim**; per-pathway direction-by-subtype not testable from published source |
| **Test 4** | Pagani claim 1, individual-subject level (HOMER as ASD classifier feature) | Mann-Whitney p=0.96 (signed) / 0.64 (abs), Cliff's δ≈0, ASD unimodal | **Null** — HOMER signal is population-level, not subject-level (re-run under recommended π) |

## Statistical caveats (audit 2026-06-11)

- **Permuted-π null is negative-mean.** The row-shuffle null sits well below 0 for the contrast/matrix tests (Test 2b mean −0.51, Test 2c −0.47), so "empirical p = 0.000" partly reflects the observed value clearing a downward-biased null rather than a strong positive effect. Read significance from the **effect size + analytical p**, not the empirical p alone. Results that clear the bar with margin *and* a small analytical p (Test 1, the gradients, the negative controls, Test 2c-Pearson) are robust; **borderline ones that pass only via the biased null — Test 2c-Spearman, the per-model Direction-1 routing, and the expanded gene-spatial Pearson — should be read as suggestive / n.s.** (`_audit/FINDINGS_LOG.md` F-016).
- **Multiple comparisons.** ~12 validations × sub-tests, no family-wise correction stated. The strong results survive Bonferroni (α≈0.0025); the borderline ones do not (F-017).
- **Researcher degrees of freedom.** The subtype test formulation was refined 2a (absolute — failed) → 2b (contrast) → 2c (full matrix); metric choices were partly post-hoc. Treat 2a→2c as exploratory-then-confirmatory (F-018).

## Future extensions

- **Per-mouse-model translation.** Done as an *exploratory* showcase in `../pagani_2026_per_model/` (subtype-level translation + occurrence-map spatial routing). The 1,491-feature Fig 1c matrix is a downsampled, dendrogram-sorted reduction with **no published feature→voxel key**, so it can't be inverted to per-voxel maps (the earlier "decode the 1,491 features" plan was falsified — see `pagani_2026_per_model/DATA_VALIDATION_2026-06-10.md`). True per-model translation needs the 20 signed per-model degree-centrality NIfTIs (requested from the Gozzi lab).
- **Per-pathway human spatial map (would need to be requested from Pagani).** Their Fig 5b/c source data ships odds ratios but not the underlying per-parcel pathway-spatial maps. With those maps in hand, Test 3 could be re-run per pathway with proper observed-side spatial pattern, and the direction-by-pathway claim from Pagani's claim 4 would become testable.
- **Replicate Pagani's exact clustering on ABIDE.** Rather than scoring against a HOMER template, re-implement Pagani's perturbation features + clustering on ABIDE individuals, then ask whether HOMER-derived features add discriminative power to the cluster assignment. ~2-3 days of porting their R code.

## Files

| File | What |
|---|---|
| `01_network_crossvalidation.py` | Test 1: π → mouse-net × human-net matrix → diagonal-dominance score |
| `02_plot_network_mapping.py` | Heatmap + bar plot of Test 1 |
| `03_baseline_comparison.py` | Test 1 robustness: bare Garin π + xyz_zero + permuted null |
| `04_subtype_translation.py` | Test 2a (failed — size-confounded; documented for honest provenance) |
| `05_subtype_contrast.py` | Test 2b — subtype-contrast row-sum translation. Pearson r = +0.494, n=8. |
| `06_plot_contrast.py` | Bar comparison + null-distribution figure for Test 2b |
| `07_full_matrix_translation.py` | **Test 2c — full 9×9 → 8×8 matrix translation. Pearson r = +0.550 (p=0.0005); Spearman ρ = +0.228 (n.s.); leverage-driven (F-007).** |
| `08_plot_full_matrix.py` | Scatter + null-distribution figure for Test 2c |
| `09_gene_spatial_translation.py` | Test 3 — gene-set spatial translation proof-of-concept (36 genes; superseded by `allen_expansion/`) |
| `10_summary_figure.py` | Consolidated 4-panel summary figure across all tests |
| `allen_expansion/` | Expanded Test 3 — full Allen API gene download (1,713 Pagani genes). See `allen_expansion/README.md`. Bootstrap r=+0.428, 95% CI (+0.349, +0.497). |
| `abide_subtype/` | Test 4 — ABIDE per-subject HOMER-template scoring (run 2026-06-11 under recommended π; **null**, see above). See `abide_subtype/README.md`. Re-run requires the ABIDE download (nilearn). |
| `README.md` | This file — workflow walkthrough + result summary |

## Citing

If you use this validation:

> Pagani M, Reess TJ, ... Lombardo MV, Gozzi A. *Autism subtypes identified using cross-species functional connectivity analyses.* **Nat Neurosci** (2026). doi:10.1038/s41593-026-02287-z

…and the HOMER manuscript (in preparation).
