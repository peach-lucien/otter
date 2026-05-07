# What's in the box — a plain-English summary

This document answers two questions a new reader is most likely to have:

1. *What does the model actually do, and is it generalising?*
2. *Where can I trust its predictions?*

For deeper detail see [`methods.md`](methods.md) (formulation),
[`results.md`](results.md) (all empirical results including the Beauchamp
2022 validation, anchor expansion experiments, and comparative-method
ablations), [`diagnostics.md`](diagnostics.md) (why motor + tectum failed
despite anchors), and [`dev/audit_2026-05-06.md`](dev/audit_2026-05-06.md)
(end-to-end audit).

## What we built

A **soft coupling matrix** π of shape (1864 mouse parcels, 2094 human
parcels) computed via **Fused Gromov-Wasserstein optimal transport** on
within-species functional connectivity (FC) and structural connectivity
(SC), with **42 Garin-atlas homologue pairs** as cross-species anchors.

The π is the answer to: *"for each mouse parcel, what's the probability
distribution over human parcels that it corresponds to?"* For most parcels,
this distribution is sharp (the FGW solver is essentially deterministic),
so π[m, :].argmax() gives a single-best human partner per mouse parcel.

## Is it generalising or just memorising?

**Both, depending on which parcel you ask about.** Three layers:

1. **The 52 anchor parcels themselves** (2.8% of mouse parcels): constrained
   by supervision. Their argmax is "we told it this maps here". No
   generalisation, just enforcement.

2. **~897 non-anchor parcels in supervised regions** (15 anchor-overlapping
   Beauchamp pairs cover 927 mouse parcels, of which only 30 are anchors —
   the other 897 are *not* supervised): **these show 11.8× chance enrichment
   on Beauchamp 2022 validation. This is real generalisation.** The OT
   objective pulls them toward "the human partner of the parcel they're
   most FC/SC-similar to in mouse", and that pull lands them in the right
   region 12% of the time (vs 1% chance).

3. **The remaining ~870 mouse parcels** with no nearby anchor: **little to
   no signal**. Beauchamp's 4 hippocampal pairs (no Garin anchor) all
   returned 0% top-1 with the original 21 anchors. After we added 4
   hippocampal supplementary anchors (8 anchor parcels), 3 of 4 moved to
   7-9% top-1 — confirming that the 0% wasn't a model bug but a supervision
   gap.

The right framing: π **interpolates within the support of the 21 anchor
pair_ids** in the joint mouse×human FC/SC manifold, much like a regression
model interpolates within its training support but does not extrapolate.
**Adding anchors expands the support**, demonstrably (the hippocampal
result).

## Where to trust the model — the trust map

`pipeline/05g_compute_trust.py` computes two per-parcel trust signals,
saved to `outputs/coupling/trust_score_*.npz`:

### Regional empirical trust (preferred)

Per-region empirical top-1 accuracy on Beauchamp 2022 validation:

| Tier | Threshold | Regions in production π | Regions after M1+hippocampal anchors |
|---|---|---|---|
| 🟢 high | top-1 ≥ 15% | Thalamus (33%), Auditory (22%), Somatosensory (20%) | (same) |
| 🟡 medium | 3-15% | Caudoputamen, Cingulate, Hypothalamus, Striatum-ventral, Visual, Pallidum | + CA3, Dentate, Subiculum, Motor (after split) |
| 🔴 low | <3% | Pons, Motor (broad), Subplate, Piriform, Inf/Sup colliculus, all 4 hippocampal | Pons, Subplate, Piriform, colliculi, CA1 |
| ⚪ unknown | not in any Beauchamp region | 845 parcels (45%) | (same) |

This signal is calibrated: each parcel inherits the empirical accuracy of
its region. **Use this when you want to know "should I trust the model's
prediction for this specific parcel?"** — the answer is "as well as the
model does in its region".

### Model-confidence trust (auxiliary)

Composite of bootstrap argmax stability + argmax mass concentration + FC
similarity to nearest anchor. Less informative because:

- 88% of parcels have perfect bootstrap stability (1.0)
- 88% of parcels have perfect argmax mass concentration (≈1.0, FGW solver
  is essentially deterministic)
- Most differentiation comes from FC similarity, which doesn't perfectly
  predict Beauchamp accuracy.

We keep it for completeness but the regional view is the actionable one.

### Visualisation

`notebooks/02_explore_results.ipynb` Section 10 shows both trust maps as
3D mouse-brain scatters. Hover any parcel to see its tier.

## Practical trust guidance

| You want to... | Recommendation |
|---|---|
| Find the human partner of a mouse anchor parcel | Always reliable (it's enforced). |
| Find the human partner of a non-anchor parcel in Thalamus, Auditory, Somatosensory | High trust — use as a credible homologue prediction. |
| Find the human partner of a non-anchor parcel in mid-tier regions (cingulate, caudoputamen, striatum, hypothalamus, visual) | Moderate trust — use as a starting hypothesis, verify against literature. |
| Find the human partner of a non-anchor parcel in low-tier regions (Pons, Motor without M1 anchor, Tectum, Hippocampus without supplementary anchors) | Don't trust — model is at chance or below for these. |
| Find the human partner of a parcel in cerebellum or other unsupported anatomy | We have no signal. Don't use the model. |
| Translate a mouse FC pattern to a human prediction | Subject-CV held-out r = 0.32. Usable as a soft prior. |
| Make a "mouse parcel X = human parcel Y" claim at the millimeter level | Don't. The mean argmax distance is 25-45mm even in good regions. Argue at the region level instead. |

## What NOT to use the model for

- **Cerebellar correspondences** — cerebellum was excluded from our 1864/2094 parcellation.
- **Cross-species spatially-inverted regions without anchors** — e.g., midbrain colliculi: mouse-tectum is dorsal but human-tectum is ventral. The xyz cost actively misleads non-anchor parcels into the wrong basin.
- **Strict 1-to-1 parcel correspondence** — we have ~10 mouse parcels and
  ~10 human parcels per Beauchamp region; there's no real 1-to-1 mapping at
  the millimeter level. Argue at the region level.
- **Extrapolation to entirely novel anatomy** — outside the supervision
  basin, the model has no signal. Add anchors first.

## How we know all this

Three independent lines of evidence:

1. **Internal CV** (held-out anchor recovery): 81% restricted top-1.
2. **External validation** (Beauchamp 2022): 11.8× chance enrichment for
   anchored regions, 0× for novel — confirms generalisation within support
   and absence outside it.
3. **Targeted intervention** (supplementary anchors): adding M1 + 4
   hippocampal anchors moves the corresponding pairs from 0% to 4-9% top-1
   without disturbing the existing 21 — confirms the supervision-density
   story.

All three converge on the same conclusion: the model captures real
cross-species biology where it has supervision and reliable structural
similarity, and is bounded by supervision elsewhere.
