# Limitations

What HOMER can't tell you. Read this before claiming HOMER predictions in published work.

## 1. Structural recovery is weak

Held-out region CV (drop one region's supervision, re-fit) recovers the correct human partner at **3.4 % top-1, 5.5 % top-5, 6.6 % top-10** — about 7× chance, but well below the supervised numbers. Three regions recover meaningfully without their anchor (mPFC 33 %, Auditory 22 %, Somatosensory 11 %); the rest recover at or near chance.

**Implication**: HOMER is not an unsupervised cross-species translator. Predictions for un-anchored regions are unreliable. The recommended `pi_fc_plus_SC_with_all_packs.npy` is good *because* it has 11 region-anchor entries on top of the 21 Garin point anchors, not because the FGW solver discovered the homologies.

## 2. Anchor packs work by construction

Every default anchor pack gives 100 % top-1 for its target Beauchamp pair, but this is **largely tautological**: the anchor's mouse-side set is identical to the validation's mouse-side set, and the human-side ball overlaps the validation's. Per-pack held-out tests (e.g. anchor Subiculum only, check CA1) confirm that **structure does not propagate across un-anchored sub-regions**.

**Implication**: extending HOMER to new regions requires adding new anchor packs (see [04_anchor_packs.md](04_anchor_packs.md)). It will not extrapolate.

## 3. Per-parcel correspondence is a region-level claim

Mean argmax distance is 25-45 mm even in well-anchored regions. Mouse parcels are ~12-2837 voxels each.

**Implication**: don't make "mouse parcel 1234 = human parcel 5678" statements at the millimeter level. Aggregate over regions (`pi[mouse_region_indices, :].sum(axis=0)`) and report top-K *human regions*.

## 4. Cerebellum is excluded

The parcellation we work with (1864 mouse / 2094 human parcels from the colleague's preprocessing) excludes cerebellum entirely.

**Implication**: 14 of Beauchamp 2022's 36 region pairs cannot be evaluated. HOMER returns no answer for any cerebellar query.

## 5. Spatially-inverted regions remain hard

`docs/archive/diagnostics.md` documents that mouse-tectum is dorsal while human-tectum is ventral in MNI space — so the xyz cross-species cost actively misleads non-anchor parcels in these regions. We attempted to fix this with per-region xyz weighting ([archive/iteration_log.md §5.11](archive/iteration_log.md)) but the local intervention does not reproduce the global xyz effect. The current workaround is anchor-based: Tectum / Olfactory / Motor have dedicated packs that override the misleading xyz prior.

**Implication**: any new region where mouse and human anatomy are spatially inverted (or where xyz is misleading) will require an explicit anchor pack to recover useful predictions.

## 6. dlPFC homology is contested

The lateral PFC pack includes a Prelimbic ↔ dlPFC entry (Carlén 2017, Laubach 2018), but Preuss 1995 argues rodents lack a true dlPFC homologue. The entry is **opt-in** within the pack.

**Implication**: claims about rodent dlPFC homology should cite both sides of the debate.

## 7. Beauchamp is one validation source

Most validation in this codebase uses Beauchamp 2022's 22 published mouse-human region pairs (gene-expression-derived). We also have:
- Internal held-out anchor CV (above)
- Bootstrap stability (97.8 % argmax stability across 40 subject resamples)
- Network coherence (Coletta-style — see [archive/iteration_log.md §5.21](archive/iteration_log.md))

Other independent sources we identified but did not integrate:
- **Mars 2018 white-matter homologies** (supplementary table requires manual extraction)
- **BICCN cell-type composition at parcel level** (requires alignment with Yao 2023 / Siletti 2023; heavy lift)

**Implication**: Beauchamp validation passes are necessary but not sufficient. A region anchored against Beauchamp may still disagree with Mars 2018 / BICCN cell types.

## 8. Subdivision packs vs broad-ball Beauchamp validation — a general pattern

When a pack subdivides a Beauchamp-validated region into multiple human sub-targets, the effect on the Beauchamp metric depends entirely on **whether at least one sub-target overlaps Beauchamp's broad ball centre**. Three documented data points:

| Pack | Sub-targets | Beauchamp validation ball | Outcome |
|---|---|---|---|
| **Cingulate** (opt-in) | subgenual ACC (–5, 10, 35), RSC (–15, –55, 10) | "cingulate gyrus" centred at pregenual ACC (–5, 25, 25) r=15 | ACG **13 % → 9 %** (−4 pp) — sub-targets sit outside the ball |
| **Striatum** (default) | dorsolateral CP → putamen (±28, 0, 0); ventromedial CP → caudate (±10, 10, 10) | "caudate nucleus" centred at (–15, 10, 10) r=12 | Cau **13 % → 33 %** (+19 pp) — ventromedial sub-target overlaps the ball centre |
| **Somatosensory** (opt-in) | face S1 (±55, –15, 25), hand S1 (±40, –25, 55), leg S1 (±10, –40, 70) | "postcentral gyrus" centred at hand S1 (–40, –25, 55) r=15 | S1 **20 % → 15 %** (−5 pp) — face S1 and leg S1 sit ~30 mm outside the ball |

The anatomy is right in all three cases — Vogt 2012, Voorn 2004, and Penfield's homunculus are uncontested. Beauchamp's broad-ball validation just measures something coarser than the sub-region anchor targets. When the pack's sub-targets happen to land at Beauchamp's ball centre (striatum), the metric improves. When they spread away from it (cingulate, somatosensory), the metric worsens — even though the predictions are anatomically more correct.

**Implication for pack design**: if you're building a subdivision pack and want it to *also* lift the Beauchamp metric, place at least one sub-target inside Beauchamp's broad ball centre. If anatomy puts your sub-targets outside that ball, the pack remains anatomically defensible but ships as **opt-in** (cingulate, somatosensory). Users who care about Beauchamp validation skip those; users who care about body-map / cingulate-sub-area distinctions enable them.

**Methodological note**: this pattern is the cleanest evidence that Beauchamp's 22-pair validation, while useful, isn't fine-grained enough to distinguish among cytoarchitecturally-defined sub-regions of the same gross anatomical area. A model that's "right" by one sub-division can look "wrong" under another. Future validation work that uses sub-region-aware metrics (e.g. Mars 2018 transitive, BICCN cell-type composition) would resolve these false negatives.

## 9. Parcellation granularity caps what's possible

Many anatomical sub-regions of interest don't exist as separate parcels in our preprocessing:
- Mouse "Motor and premotor" is a single 2-parcel object containing M1 + M2 + premotor
- Insular sub-divisions (anterior / posterior, gustatory / visceral) are not separately labelled in DSURQE
- Thalamic nuclei, hypothalamic nuclei, brainstem nuclei (LC, raphe, VTA) are aggregated

**Implication**: HOMER cannot deliver predictions at granularities finer than 1864 mouse parcels / 2094 human parcels. Many neurobiologically meaningful questions need finer parcellation than this.

## 10. Anatomical homology curation is finite

We've curated 14 anchor packs covering ~22 named brain regions beyond the Garin 21. Plenty of regions remain anchored only at the coarse Garin point-anchor granularity. Adding a new pack is ~30 minutes of literature curation per region — see [06_extending.md](06_extending.md) — but the universe of possible packs is bounded by published cross-species correspondences *and* by what DSURQE exposes as labels (per limitation 9).

**Implication**: HOMER's coverage grows by careful curation, not by training. There's no "scale up" path that adds regions without literature work.

## What HOMER *can* tell you reliably

Use the multi-source trust map (`outputs/coupling/trust_multisource_all_packs.npz`) to gate queries. For parcels in the `anchored_and_validated` tier (19 % of the brain), top-K queries are reliable. For `validated_only` (36 %), top-K is moderately reliable. For `structural` (13 %) treat as a research starting point. For `low_evidence` (29 %), assume no signal.

For region-level queries on any anchored region, the predictions are essentially correct by construction. That's worth a lot for the ~22 sub-regions covered by packs + the 21 Garin regions — well over 40 well-defined cross-species region predictions, which is more than any prior method we're aware of.
