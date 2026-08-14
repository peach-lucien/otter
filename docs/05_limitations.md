# Limitations

Constraints on what OTTER predictions support.

## 1. Supervision dependence of parcel-exact recovery

Hold out each of the 41 supervision units (15 Garin homology classes + 26 region packs) in turn and re-fit. Region-level recovery holds at a held-out mean AUROC of 0.74 (chance 0.5), while parcel-exact recovery collapses to roughly 10 % top-1. Seven of the 41 units fall below chance, predominantly hippocampal subfields and fine somatotopic subdivisions.

Curation buys parcel precision rather than region-level correspondence. Connectivity and spatial position reconstruct *which human region* a mouse region corresponds to without any anchor at all; the anchors sharpen that to *which parcel*. OTTER is therefore not a landmark look-up. Neither is it an unsupervised translator, and a prediction in an un-anchored region should be read at region granularity rather than parcel granularity.

Top-1 is not an informative metric for a held-out unit. Scored by displacement instead, the held-out coupling is displaced (median 34 mm vs 17 mm for the full model) but not lost. The top-1 collapse alone overstates the failure.

## 2. Circularity of anchor-pack validation

Every default anchor pack gives 100 % top-1 for its target Beauchamp pair, but this is largely tautological, since the anchor's mouse-side set is identical to the validation's mouse-side set, and the human-side ball overlaps the validation's. Per-pack held-out tests (e.g. anchor Subiculum only, check CA1) confirm that structure does not propagate across un-anchored sub-regions.

Extending OTTER to new regions requires adding new anchor packs (see [04_anchor_packs.md](04_anchor_packs.md)). It will not extrapolate.

## 3. Granularity of per-parcel correspondence

Mean centroid displacement from the expected homologue is 8.83 mm parcel-weighted for the production model, against a derived chance displacement of 25 mm. Individual regions range more widely, and reconstruction accuracy agrees between homotopic regions across hemispheres at only ρ = 0.36. Mouse parcels are 12 to 2,837 voxels each.

Statements of the form "mouse parcel 1234 = human parcel 5678" are not supported at the millimetre level. Aggregate over regions (`pi[mouse_region_indices, :].sum(axis=0)`) and report top-K *human regions*.

## 4. Cerebellar exclusion

The parcellation we work with (1864 mouse / 2094 human parcels from the colleague's preprocessing) excludes cerebellum entirely.

14 of Beauchamp 2022's 36 region pairs cannot be evaluated. OTTER returns no answer for any cerebellar query.

## 5. Spatially inverted regions

Mouse tectum is dorsal while human tectum is ventral in MNI space, so the xyz cross-species cost actively misleads non-anchor parcels in these regions. Per-region xyz weighting does not reproduce the global xyz effect. Tectum, Olfactory and Motor instead have dedicated packs that override the misleading xyz prior.

Any new region where mouse and human anatomy are spatially inverted (or where xyz is misleading) will require an explicit anchor pack to recover useful predictions.

## 6. dlPFC homology

The lateral PFC pack can supply a Prelimbic ↔ dlPFC entry (Carlén 2017, Laubach 2018), but Preuss 1995 argues rodents lack a true dlPFC homologue. That entry is excluded from the recommended composition (the `lateral_pfc` pack ships OFC-only) because the homology is contested and OTTER's own Schaeffer et al. 2020 falsification test contradicts it. Pass `include_dlpfc=True` to add it back for ablations.

Claims about rodent dlPFC homology should cite both sides of the debate.

## 7. Validation sources

Most validation in this codebase uses Beauchamp 2022's 22 published mouse-human region pairs (gene-expression-derived). We also have:
- Internal held-out anchor CV (above)
- Bootstrap stability (98.2 % argmax stability across 40 subject resamples)
- Network coherence (Coletta-style)

BICCN cell-type composition at parcel level is integrated. Eight BICCN cell-class maps are scored over all 2,094 parcels against the translation spin null and reported in `docs/03_results.md`, spanning −0.03 for microglial density to +0.35 for the neuronal-glial contrast. They constrain what transfers rather than which regions correspond, so they do not replace a homology benchmark.

One independent source remains unintegrated:
- Mars 2018 white-matter homologies (supplementary table requires manual extraction)

Beauchamp validation passes are necessary but not sufficient. A region anchored against Beauchamp may still disagree with Mars 2018 / BICCN cell types.

## 8. Subdivision packs against broad-ball validation

When a pack subdivides a Beauchamp-validated region into multiple human sub-targets, the effect on the Beauchamp metric depends on whether at least one sub-target overlaps Beauchamp's broad ball centre. Three data points:

| Pack | Sub-targets | Beauchamp validation ball | Outcome |
|---|---|---|---|
| **Cingulate** | subgenual ACC (–5, 10, 35), RSC (–15, –55, 10) | "cingulate gyrus" centred at pregenual ACC (–5, 25, 25) r=15 | ACG **13 % → 9 %** (−4 pp), sub-targets sit outside the ball |
| **Striatum** | dorsolateral CP → putamen (±28, 0, 0); ventromedial CP → caudate (±10, 10, 10) | "caudate nucleus" centred at (–15, 10, 10) r=12 | Cau **13 % → 33 %** (+19 pp), ventromedial sub-target overlaps the ball centre |
| **Somatosensory** | face S1 (±55, –15, 25), hand S1 (±40, –25, 55), leg S1 (±10, –40, 70) | "postcentral gyrus" centred at hand S1 (–40, –25, 55) r=15 | S1 **20 % → 15 %** (−5 pp), face S1 and leg S1 sit ~30 mm outside the ball |

The anatomy is right in all three cases. Vogt 2012, Voorn 2004, and Penfield's homunculus are uncontested. Beauchamp's broad-ball validation measures something coarser than the sub-region anchor targets. When the pack's sub-targets happen to land at Beauchamp's ball centre (striatum), the metric improves. When they spread away from it (cingulate, somatosensory), the metric worsens, even though the predictions are anatomically more correct.

A subdivision pack lifts the Beauchamp metric only if at least one sub-target falls inside Beauchamp's broad ball centre. Where anatomy puts the sub-targets outside that ball, the pack remains anatomically defensible but lowers the Beauchamp parcel metric for that region (cingulate, somatosensory, visual). These packs are kept in the recommended composition, since the multi-benchmark evidence favours inclusion, in particular TransBrain's region-level homology benchmark. The trade-off is documented per pack in [`04_anchor_packs.md`](04_anchor_packs.md).

Beauchamp's 22-pair validation is not fine-grained enough to distinguish among cytoarchitecturally defined sub-regions of the same gross anatomical area, so a model that is right by one subdivision can look wrong under another. Sub-region-aware metrics (Mars 2018 transitive, BICCN cell-type composition) would resolve these false negatives.

## 9. Parcellation granularity

Many anatomical sub-regions of interest do not exist as separate parcels in the preprocessing:
- Mouse "Motor and premotor" is a single 2-parcel object containing M1 + M2 + premotor
- Insular sub-divisions (anterior / posterior, gustatory / visceral) are not separately labelled in DSURQE
- Thalamic nuclei, hypothalamic nuclei, brainstem nuclei (LC, raphe, VTA) are aggregated

OTTER cannot deliver predictions at granularities finer than 1864 mouse parcels / 2094 human parcels. Many neurobiologically meaningful questions need finer parcellation than this.

## 10. Extent of the curation

We have curated 15 anchor packs covering ~22 named brain regions beyond the Garin 21 (all 15 in the recommended composition, see `src/otter/data/anchor_packs/registry.py`). Plenty of regions remain anchored only at the coarse Garin point-anchor granularity. Adding a new pack is ~30 minutes of literature curation per region, see [06_extending.md](06_extending.md), but the universe of possible packs is bounded by published cross-species correspondences *and* by what DSURQE exposes as labels (per limitation 9).

Coverage grows by curation rather than by training. No scale-up path adds regions without literature work.

## Supported queries by tier

Use the multi-source trust map (`outputs/coupling/trust_multisource_canonical.npz`) to gate queries. For parcels in the `anchored_and_validated` tier (31.5 % of the brain), top-K queries are reliable at *parcel* granularity. For `validated_only` (23.8 %), parcel-exact recovery is much weaker (top-1 0.391 against 0.700), so query these at *region* granularity. For `anchored_only` (12.2 %) and `structural` (10.9 %), treat the prediction as a research starting point. For `low_evidence` (21.6 %), assume no signal. The two validated tiers cover 55.3 % of the brain.

For region-level queries on any anchored region, the predictions are correct by construction. That covers the ~22 sub-regions supplied by packs plus the 21 Garin regions, more than 40 well-defined cross-species region predictions.
