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
