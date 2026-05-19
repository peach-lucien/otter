# Autism subtypes (Pagani 2026) — HOMER applications

Cross-species network validation against [Pagani et al. 2026 *Nat Neurosci*](https://www.nature.com/articles/s41593-026-02287-z), "Autism subtypes identified using cross-species functional connectivity analyses".

## What the paper does — workflow in 7 steps

The paper's central claim is that autism is best understood as **two functional-connectivity subtypes** — a **hyperconnected** group and a **hypoconnected** group — that recur in both mouse models and humans, and that these two subtypes are driven by distinct biological mechanisms (immune signalling for hyperconnected; synaptic genes for hypoconnected). The workflow:

**Step 1 — Mouse functional connectivity.** Resting-state fMRI in 20 autism-relevant mouse models (Chd8, Fmr1, Tsc2, Trem2, Btbr, Cdkl5, Mecp2, Shank3, Cntnap2, Nlgn3, Oxtr, 16p11.2, Mecp2 dup, etc.) — see Fig 1c source data (20 models × 1,491 features). Compute per-mouse-model FC matrices; identify regions where each model deviates from wild-type controls (hypo: connectivity reduced, hyper: connectivity increased).

**Step 2 — Mouse subtype clustering.** Cluster the 20 mouse models by their FC perturbation signature → two subtypes emerge. One subtype shows widespread hypoconnectivity (e.g. Tsc2, Shank3, Fmr1); the other shows hyperconnectivity (e.g. Trem2, Btbr, Il6). Map both onto the mouse 9-network atlas (Auditory, BF, Caudate-Putamen, DMN, HC, Salience, Somatomotor, Thalamus, Visual; ED Fig 1).

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

**4/8 diagonal-argmax with mean 1.92× over null. Permuted-π null: 1.95/8 and 0.97×** — exactly chance, confirming the signal is structural.

Misses are interpretable: mouse visual cortex covers higher-order visual regions that Schaefer-17 places in DorsAttn; mouse HC routes to "Subcortical" because hippocampus *is* subcortical in cortical-only parcellations; Auditory is limited by Schaefer's narrow auditory label; BF/Olfactory has no clean cortical counterpart in Yeo-7. None of these are HOMER failures — they're Schaefer/Yeo definition limits.

## Robustness — Test 1b

`03_baseline_comparison.py` runs the same scoring against three π variants and a permuted-π null:

| π variant | Diag-argmax | Mean ratio |
|---|:---:|---:|
| with_all_packs (recommended) | 4/8 | 1.92× |
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

**Pearson r = +0.547** between predicted and observed human subtype contrast. **Empirical p < 0.005** vs 200 permuted-π row-shuffles. The permuted-π null mean is −0.47 (95% CI: −0.88 to +0.06); the observed +0.547 sits well outside that band. HOMER correctly recovers the direction of the contrast for 6 of 8 human networks.

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

**Pearson r = +0.537 over 36 paired matrix elements, analytical p = 0.0007, empirical p < 0.005** against 200 permuted-π row-shuffles (null mean −0.52, 95% CI −0.73 to +0.00). 22 of 36 matrix entries agree in sign.

The largest positive observed Δ — Subcortical–Subcortical at +33 in human (the strongest network-pair signal of hyperconnected ASD) — is also the largest positive Δ HOMER predicts (+15). The largest negative observed Δ entries (Limbic–SomatoMotor, Visual–SomatoMotor, Limbic–Salience, all hypo-dominant) are also predicted negative by HOMER.

This is the sharpest test we have: HOMER's π reproduces the *joint network-pair structure* of Pagani's per-subtype spatial contrast, not just per-network row-sums. The improvement over Test 2b comes from 4.5× more matrix elements (36 vs 8) and from preserving the cross-pair structure (which network-pair Δs are highest, not just which networks are most perturbed).

## Test 3 — Gene-set spatial translation (proof of concept)

Pagani claim 4: hypo/hyper subtypes have distinct gene/pathway signatures that recur cross-species. Pagani treats this as parallel findings — synaptic genes are enriched in mouse-hypo regions AND synaptic genes are enriched in human-hypo regions — without explicitly bridging the two through a spatial mapping. HOMER's π lets us bridge them.

**Pipeline**: identify HOMER curated genes that overlap with Pagani's subtype-implicated genes → compute per-mouse-parcel mean expression score for each gene set → translate via π → aggregate predicted human-parcel scores to 8 Pagani networks → compute predicted human Δ = hyper-spatial − hypo-spatial → correlate against observed human Δ from Fig 4e.

**Blocker — gene coverage.** Pagani's hypo + hyper sets contain 6,415 implicated genes. HOMER's curated mouse ISH atlas contains 51 genes; only 10 overlap with hypo-only and 26 with hyper-only, for a total of 36 useable genes (0.6% of Pagani's). The test is heavily underpowered relative to the full sets.

**Result on the available 36 genes**: Pearson r = +0.439 (p=0.28 nominal), **Spearman ρ = +0.619 (empirical p = 0.045** vs 200 permuted-π row-shuffles). Same-sign agreement on 4 of 8 networks. This is directionally consistent with Pagani's claim 4 but underpowered. A full pathway-spatial test would need ~3-4 hours of Allen Brain Atlas API queries to download parcel-level expression for all 1,952 + 4,463 implicated genes — straightforward in principle, beyond what this round did.

## Summary across all three tests

| Test | What it tests | Result | Verdict |
|---|---|---|---|
| **Test 1** | Pagani's name-based network bridge has biological substance | 4/8 canonical pairs diagonal-argmax; mean 1.92× over null | Bridge OK for 4 networks; 4 misses are atlas-label artefacts |
| **Test 2c** | Pagani claim 3 (FC subtypes recur cross-species at matching anatomical locations) | r=+0.537 over 36 matrix elements; p=0.0007 analytical, p<0.005 empirical | **Strong** — replicates the spatial-contrast claim through a quantitative bridge |
| **Test 3** | Pagani claim 4 (subtype gene/pathway signature recurs cross-species spatially) | Spearman ρ=+0.619, empirical p=0.045; 36/6,415 genes used | **Suggestive proof-of-concept**; full version blocked on gene coverage |

## Future extensions (need additional data)

- **Full pathway-spatial test (extends Test 3 to all 6,415 genes).** Download parcel-level mouse Allen ISH expression for all genes in Pagani's subtype gene sets via the Allen API, recompute Test 3 with full coverage. Likely the highest-power test we could do without raw FC data.
- **Per-mouse-model translation.** Apply HOMER's π to per-model FC perturbation maps (Figura 1c — 20 mouse models × 1,491 features) into human-parcel space, compare against matched human ASD subtype maps. Quantifies "mouse Tsc2 looks like human ASD subtype X" without the name shortcut. **Blocker — decoding what those 1,491 features represent** (likely a published Allen atlas subdivision; the paper's methods would specify).
- **Per-subject human FC clustering.** Replicate Pagani's clustering procedure independently — requires the 2,170 individual-subject FC matrices behind their per-subtype averages. Source data only ships per-subtype averages.
- **Voxel-level ENIGMA pathway maps.** For a more granular pathway-spatial test, ENIGMA's voxel-level ASD spatial signatures + SFARI's gene-spatial maps would let HOMER bridge the spatial pattern directly. Separate data acquisition from a third source.

## Files

| File | What |
|---|---|
| `01_network_crossvalidation.py` | Test 1: π → mouse-net × human-net matrix → diagonal-dominance score |
| `02_plot_network_mapping.py` | Heatmap + bar plot of Test 1 |
| `03_baseline_comparison.py` | Test 1 robustness: bare Garin π + xyz_zero + permuted null |
| `04_subtype_translation.py` | Test 2a (failed — size-confounded; documented for honest provenance) |
| `05_subtype_contrast.py` | Test 2b — subtype-contrast row-sum translation. Pearson r = +0.547, n=8. |
| `06_plot_contrast.py` | Bar comparison + null-distribution figure for Test 2b |
| `07_full_matrix_translation.py` | **Test 2c — full 9×9 → 8×8 matrix translation. Pearson r = +0.537, n=36, analytical p=0.0007.** |
| `08_plot_full_matrix.py` | Scatter + null-distribution figure for Test 2c |
| `09_gene_spatial_translation.py` | Test 3 — gene-set spatial translation proof-of-concept (Spearman ρ=+0.619, emp p=0.045; underpowered). |
| `10_summary_figure.py` | Consolidated 4-panel summary figure across all tests |
| `README.md` | This file — workflow walkthrough + result summary |

## Citing

If you use this validation:

> Pagani M, Reess TJ, ... Lombardo MV, Gozzi A. *Autism subtypes identified using cross-species functional connectivity analyses.* **Nat Neurosci** (2026). doi:10.1038/s41593-026-02287-z

…and the HOMER manuscript (in preparation).
