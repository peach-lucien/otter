# Diagnostics — supervised-but-failing pairs (May 2026)

`pipeline/05f_beauchamp_validation.py` flagged two surprising failures in the
external validation: **Motor cortex** (anchor pid=2) and **Tectum**
(anchor pid=21). Both score 0% top-1 against Beauchamp's published
mouse↔human pairs *despite* having direct supervision via Garin anchors.

This document records why.

## Headline

**Neither is a solver bug.** Both failures are explained by:

1. **Anchor-boundary mismatches** — our Garin anchors are unions of multiple
   AHBA structures, broader than the targets Beauchamp specifies, so the
   anchor centroid drifts and the surrounding non-anchor parcels follow.
2. **Cross-species spatial inversions** — for midbrain structures, mouse
   tectum is dorsal but human tectum is ventral; xyz cost cannot map this
   correctly without explicit anchor pinning.

Held-in supervision works perfectly in both cases — the anchor parcels
themselves recover their human partners with rank 1/2094 and distance 0.
The failure is for the surrounding non-anchor mouse parcels.

## DIAG-1 — Motor cortex (pid=2)

### Anchor placement

Our human "Motor and premotor" anchor centroid: **(-30.8, -6.4, +52.2) mm**.
Canonical precentral gyrus (M1) per AAL3 / Beauchamp: **(-35, -20, +55)**.
Distance: **14.5 mm** — our anchor is ~14mm *anterior* of the canonical M1
target. This is because our anchor's subregion string is

> `Primary Motor Cortex; Frontal Eye Fields; Premotor Eye Field; Area 55b;
> Supplementary and Cingulate Eye Field; Area 6m anterior; Dorsal area 6;
> Area 6mp; Ventral Area 6; Rostral Area 6; Area IFJp; Area 6 anterior;
> Frontal OPercular Area 1`

— a *union* of M1, premotor, FEF, SCEF, and Area 6 subdivisions. The
centroid of that union is anterior to M1 alone. Beauchamp's "Primary motor
area" maps strictly to the **precentral gyrus** (M1 only), so the centroid
mismatch alone accounts for ~14mm of the ~35mm Beauchamp distance.

### What the 51 non-anchor mouse-Motor parcels do

Of 53 mouse parcels in Beauchamp's "Primary motor area" branch, 0 are our
Garin Motor anchors (anchors are at pos 2, 3 only). The 53 argmax to:

- **None** of our 42 anchors (0/53 hit any anchor)
- xyz mean: roughly (-21, +9, +42), i.e. **anterior-frontal cortex**
- Mean distance from argmax to our L Motor anchor centroid: **38 mm**

The argmax pattern is coherent, not random. These parcels' mouse y-coords
are all positive (~+3.8, anterior in mouse). The xyz cost (`xyz_weight=0.5`)
pushes them toward human positions with similarly anterior y, which puts
them in PFC / anterior frontal cortex, not motor cortex.

### Conclusion

Two effects compound:

- The Motor anchor is too broad → its centroid is shifted anterior of
  canonical M1.
- The non-anchor mouse-motor parcels follow the spatial cost into an even
  more anterior basin, since no anchor pulls them back to M1.

**Beauchamp's own paper notes mouse↔human motor cortex transcriptomic
similarity is unusually weak.** So the "real" homology may genuinely be
weaker for motor than for, say, V1. But the proximate cause of the 0%
top-1 here is the anchor-definition mismatch, not biology.

### Implication for next iteration

If we want the motor anchor to behave well on Beauchamp-style tests, we
should **split** the existing pid=2 anchor into:

- `pid=2A` Primary motor (M1 only — precentral gyrus, around (-35, -20, +55))
- `pid=2B` Premotor / FEF / SCEF / Area 6 (around (-30, +5, +50))

This gives the model two narrower targets instead of one fuzzy one.

## DIAG-2 — Tectum (pid=21)

### Anchor placement

Our human Tectum anchor centroid: **(-5.2, -34.4, -8.1) mm** (correct
midline, posterior, ventral). Canonical superior colliculus: **(-5, -30, -2)**.
Distance: **7.5 mm** — well-placed on its own.

### What the 80 non-anchor mouse-tectum parcels do

53 sup-colliculus parcels, 29 inf-colliculus parcels. Of 53 sup-coll, only
2 are anchors. Their argmax distribution:

| | mouse anatomy | argmax y (MNI) | argmax z (MNI) | dist to anchor |
|---|---|---|---|---|
| Sup colliculus (53 parcels) | dorsal midbrain, mouse z≈+1.7 | mean -60 (posterior) | mean +47 (DORSAL — wrong!) | 63 mm |
| Inf colliculus (29 parcels) | dorsal midbrain, mouse z≈+1.7 | mean -76 (very posterior) | mean +47 (very DORSAL) | 73 mm |

The argmax lands in **dorsal posterior cortex** (parieto-occipital region)
— ~50mm dorsal of the actual human colliculi.

### Why

This is a **cross-species spatial geometry inversion**. In the mouse:

- Tectum is **dorsal** (top of brain, high z)
- Cortex is **also dorsal** (top)

In the human, due to massive cortical expansion + brain folding:

- Cortex is **dorsal** (high z, ~+50)
- Tectum is **ventral** (low z, ~-5) — buried under cortex

xyz cost preserves topology: mouse-dorsal → human-dorsal. So mouse-tectum
(dorsal in mouse) maps to human-dorsal-cortex, NOT to human-tectum.

The Tectum anchor itself was placed correctly in human ventral midbrain by
Garin (it overrides the xyz cost via the anchor supervision term). But the
80 surrounding non-anchor mouse-tectum parcels have no direct supervision,
so they default to xyz topology, which is the wrong basin for tectum.

### Conclusion

For brainstem / midbrain structures, the spatial M term is actively misleading.
The fix isn't about anchor count or anchor definition — it's that **xyz cost
should be down-weighted (or zeroed) for non-cortical mouse parcels** because
mouse-dorsal-midbrain has no human-dorsal homologue.

### Implication for next iteration

Two options:

1. **Add many more brainstem anchors** (sup colliculus, inf colliculus, red
   nucleus, periaqueductal grey, etc.) so the supervision term dominates the
   xyz term in midbrain.
2. **Replace xyz cost with a region-aware spatial cost** that doesn't
   penalise mouse-tectum being topologically far from human-tectum given
   they're both at "tectum-coordinates" within their own brain.

Option 1 is straightforward; option 2 is research.

## Implication for EXP-1 (hippocampal anchors)

Mouse hippocampus is **dorsal** in mouse (top, posterior). Human hippocampus
is **ventral** medial-temporal (low z, lateral). This is the **same
spatial-inversion problem** as tectum.

If we just add CA1/CA3/dentate/subiculum anchors:

- The anchor parcels themselves will recover their human partners (held-in
  works always).
- But the surrounding non-anchor mouse-hippocampal parcels may still argmax
  to wherever their xyz topology pulls them — which is dorsal-posterior-cortex
  in human, not ventral-medial-temporal hippocampus.

**Mitigation options for EXP-1:**

- Add **multiple** hippocampal anchors (4 subfields × 2 hemispheres = 8
  anchors) so the supervision basin is large enough to cover the
  surrounding hippocampal parcels.
- Tag hippocampal mouse parcels and zero their xyz cost (or use only FC/SC).
- Pre-test: before re-solving, check whether the Beauchamp non-anchor
  hippocampal parcels' xyz neighbours in MNI are in / near human
  hippocampus or somewhere else.

## Saved outputs

This document. No code/JSON saved — the diagnostics ran inline in the chat
session against `outputs/coupling/pi_fc_plus_SC.npy`. Reproducing them
requires running the inline blocks in the conversation; if you want a
permanent script, that's a follow-up.

## Pre-flight for EXP-1 (hippocampal anchors)

Two checks ran before committing to EXP-1.

### Check 1 — xyz cost alone

For each mouse hippocampal parcel, find the human parcel that minimises
M_xyz (per-species-normalised xyz Euclidean distance). Does it land in human
hippocampus (within 15mm of canonical (±25, -25, -10))?

| Mouse Beauchamp region | n parcels | n landing in hippocampus by xyz alone |
|---|---:|---:|
| Field CA1 | 15 | **0** |
| Field CA3 | 26 | **0** |
| Dentate gyrus | 22 | **0** |
| Subiculum | 29 | **0** |
| **Total** | **92** | **0/92 (0%)** |

xyz cost alone places every hippocampal mouse parcel in dorsal-posterior
cortex (mean MNI ~(-2, -50, +25)), not in human hippocampus. **Sanity
check**: mouse V1 parcels also land 0/54 in human cuneus by xyz alone, so
this isn't hippocampus-specific — the per-species xyz normalisation is
uniformly misleading for cross-species anatomy. The model relies on
anchor + FC/SC supervision to override it.

### Check 2 — what current π does (with no hippocampal supervision)

Use a 30mm tolerance (broader than 15mm Beauchamp test) and ask: of the
non-anchor mouse parcels in each region, how many argmax within 30mm of
the canonical centroid?

**Anchored regions (current production π):**

| Beauchamp region | n | within 30mm | mean dist |
|---|---:|---:|---:|
| Visual areas (cuneus target) | 54 | 4 (7%) | 62 mm |
| Anterior cingulate | 23 | 8 (35%) | 34 mm |
| Primary motor area | 53 | 44 (83%) | 24 mm |
| Primary somatosensory | 155 | 90 (58%) | 25 mm |
| Caudoputamen | 149 | 103 (69%) | 23 mm |
| Thalamus | 110 | 84 (76%) | 19 mm |

**Hippocampal regions (no anchors yet):**

| Beauchamp region | n | within 30mm of hippocampus | mean dist |
|---|---:|---:|---:|
| Field CA1 | 15 | 0 (0%) | 56 mm |
| Field CA3 | 26 | **8 (31%)** | 34 mm |
| Dentate gyrus | 22 | 1 (5%) | 51 mm |
| Subiculum | 29 | 1 (3%) | 52 mm |

### Interpretation

- **Anchored regions span 7–83% within-30mm**, so even with anchors the
  fan-out is highly variable. Visual is the worst (only 7%), Motor +
  Caudoputamen + Thalamus are >70%. The region's anchor breadth and FC/SC
  structure both matter.
- **CA3 already gets 31% within 30mm without any hippocampal anchor** —
  surprising and important. The model's FC/SC structure is partially
  pulling CA3 toward the right human region, just not strongly enough to
  beat 15mm strict-tolerance.
- The other 3 hippocampal regions (CA1, dentate, subiculum) are <5% within
  30mm — far from the right area.

### Predictions for EXP-1

If we add 4 hippocampal anchors (CA1, CA3, dentate, subiculum × L/R = 8
new anchor parcels), based on the pattern in already-anchored subcortical
regions:

| Metric | Current | Expected post-EXP-1 |
|---|---:|---:|
| Beauchamp **strict** top-1 (15mm) | 0/92 (0%) | 5–15% (matching other anchored subcortical) |
| Within 30mm of hippocampus | 10/92 (11%) | 40–70% (matching Caudoputamen / Thalamus) |
| Held-in self-recovery for 8 new anchors | n/a | 100% |

The strict top-1 won't jump to 90%, but a 5-15% improvement would be a
meaningful 5-15× chance enrichment for previously 0% pairs — clearly
publishable and consistent with the supervision-density story.

The CA3 result (31% already within 30mm without supervision) suggests the
model's structural cost has *some* hippocampal signal already; adding the
anchor should sharpen it.

## Bottom line for next steps

Pre-flight justifies proceeding with EXP-1, with realistic expectations:

1. **EXP-1 (hippocampal anchors)**. Expected: 0% → ~10% top-1, dramatic
   improvement on within-30mm metric. Multi-day effort.

2. **Split the Motor anchor** (pid=2 → pid=2A M1, pid=2B premotor). Simple
   anchor-table edit, doesn't require external data. Likely cleans up the
   motor cortex Beauchamp result.

3. **Brainstem / midbrain xyz weight reduction**. The tectum failure is a
   topology inversion; consider a region-aware xyz cost that downweights
   spatial similarity for non-cortical mouse parcels. Research item.
