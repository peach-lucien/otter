# Buckner & Krienen 2013, tethering hypothesis (negative control)

A negative-control / falsification test. We asked whether OTTER's π is
*appropriately unconfident* over the part of human cortex the field says has
no clear mouse homologue.

## Why this experiment

[Buckner & Krienen 2013, Trends Cogn Sci](https://doi.org/10.1016/j.tics.2013.09.017),
"The evolution of distributed association networks in the human brain"
argue that human association cortex expanded so much it became evolutionarily
**"untethered"** from the sensory hierarchies and molecular gradients that
organise primary cortex. The implication for a mouse↔human mapping: there is
no well-defined mouse homologue of the expanded human association cortex, so a
faithful coupling should be confident over sensorimotor cortex and **sparse /
unconfident over association cortex**.

The falsification logic follows from that. If OTTER's π were *uniformly* confident
everywhere, including over association cortex that has no clear mouse
homologue, that would signal over-fitting. OTTER should know what it can't
map.

## Result

> ## ⚠️ VERDICT REVERSED — 2026-07-18
>
> **This test no longer passes. On the canonical coupling it is INCONCLUSIVE** (the value the
> log itself now records in its `verdict` field).
>
> Every "pass" number previously reported here came from a retired coupling. The re-run on
> `pi_canonical.npy` (sha256 `bb4cae00cbca9f16c6f9cfca3b0124292b41d81643e2ef5d5511686b20f9df77`):
>
> | statistic | retired coupling | **canonical coupling** |
> |---|---:|---:|
> | sensorimotor-tertile coverage (log₁₀) | −8.5 | **−4.54** |
> | association-tertile coverage (log₁₀) | −18.8 | **−5.23** |
> | sensorimotor−association gap, summed | 10.3 log units | *(metric superseded)* |
> | sensorimotor−association gap, mass-normalised mean | 6.74 log units | **0.68 log units** |
> | gap, spatial-autocorrelation (spin) null | p = 0.002 | **p = 0.286 — not significant** |
> | gap, Mann–Whitney (no spatial null) | p = 2.5×10⁻¹⁶ | p = 1.3×10⁻⁶ |
> | coverage vs myelin, continuous ρ | +0.185 | **+0.095** (spin p = 0.169) |
> | entropy vs myelin, continuous ρ | +0.088 ("flat") | **−0.435** |
>
> The 6.7-log-unit gap is **withdrawn**. It is 0.68 log units on the canonical coupling, an
> order of magnitude smaller, and it does not clear a spin null. The tethering prediction is
> neither confirmed nor refuted by this test as currently constructed: the Mann–Whitney
> contrast is still highly significant, but Mann–Whitney ignores spatial autocorrelation, and
> once that is accounted for the gap is within the null. The permuted-*axis* null in the log
> (`null/empirical_p = 0.0`) has the same weakness and should not be quoted as support.
>
> The entropy result also moved and changed sign: entropy is *not* flat along the axis on the
> canonical coupling (ρ = −0.435), so note (1) below is falsified as written.
>
> Everything below this box is the retired-coupling write-up, left in place rather than
> silently re-narrated. Do not quote its numbers.

Three panels (`outputs/figures/buckner_krienen_2013_tethering.png`).

For every human cortical parcel we measure OTTER's **coverage**, the π mass it
receives from the mouse brain (the per-column mass of the coupling), and place it on
the sensorimotor → association axis (the HCP T1w/T2w myelin map; high myelin =
sensorimotor).

> ⚠️ **Coverage must be aggregated as a mass-normalised mean rather than a sum.** Summing
> makes coverage scale with *how many parcels a region happens to contain*, which is a
> parcellation artefact rather than biology. It is not a free parameter: the downstream
> disorder result is ρ = **+0.64** with the mean and ρ = **+0.05** with the sum. The
> numbers below are from the original summed analysis and are **superseded**. ~~the
> canonical figures are … a **6.7 log-unit** sensorimotor−association tertile gap
> (spin p = 0.002), with the *continuous* correlation spin-fragile (r = 0.13,
> p = 0.076)~~ — that text was itself written against the retired coupling and mislabelled
> "canonical". The actual canonical figures (`outputs/logs/section5_coverage_nulls.json`)
> are a **0.68 log-unit** gap at **spin p = 0.286**, with the continuous correlation
> r = 0.14 at spin p = 0.169. Both are null.

OTTER's coverage **collapses toward association cortex**. The sensorimotor
tertile receives a mean log₁₀ coverage of −8.5; the association tertile −18.8,
a gap of **10.3 log units** [SUPERSEDED: summed, and on the retired coupling. The
mass-normalised gap on the canonical coupling is **0.68 log units, spin p = 0.286**, i.e.
null — see the reversal box above]. The contrast
is overwhelming (Mann-Whitney p = 2.5×10⁻¹⁶) and far beyond a permuted-axis
null (95th-percentile gap 1.8 log units; empirical p = 0.000). The decile curve
shows a monotone-ish collapse from sensorimotor to association cortex; Spearman
ρ = +0.185 along the full axis.

| Measure | Result |
|---|---|
| sensorimotor-tertile coverage | log₁₀ −8.5 |
| association-tertile coverage | log₁₀ −18.8 |
| coverage gap (summed, retired π, SUPERSEDED) | 10.3 log units, p = 2.5×10⁻¹⁶ |
| coverage gap (mass-normalised mean, **retired** π, WITHDRAWN) | ~~6.7 log units, spin p = 0.002~~ |
| **coverage gap (mass-normalised mean, canonical π)** | **0.68 log units, spin p = 0.286 (null)** |

~~**OTTER is not confident everywhere**, it is dramatically sparser over human
association cortex, just as the tethering hypothesis predicts. The coupling
"knows" it has no good mouse homologue for the expanded human association
networks.~~

**WITHDRAWN.** On the canonical coupling the sensorimotor−association coverage gap is
0.68 log units and does not clear a spin null, so this test does not support the
tethering prediction. It does not refute it either — the test simply no longer
discriminates. The evidence §5 now rests on is *reconstruction*-coverage, which does track
expansion and hierarchy maps under a spin null (see `docs/03_results.md` §5).

**Notes.** (1) π's per-parcel *entropy*, the diffuseness of a parcel's
mouse origin, is flat along the axis (ρ = +0.088). The *amount* of
coverage carries the tethering signal, and its diffuseness does not. (2) OTTER's
Garin anchors are weighted toward sensorimotor cortex, which contributes to the
sensorimotor–association coverage gap alongside the underlying evolutionary
untethering. The two are not fully separable, but both push the same way: a
faithful coupling and a sensible anchor set agree that association cortex is
the hard, weakly-tethered territory. The same boundary appears in the Fulcher
2019 experiment, where mouse isocortex routes onto a gradient-compressed human
territory that never reaches the transmodal extreme.

## Method

1. Per human cortical parcel, OTTER coverage = log₁₀ of the π column mass
   (Σ_i π[i,j]).
2. Sensorimotor→association axis = HCP T1w/T2w myelin per Schaefer region
   (reused from the Fulcher experiment).
3. Decile curve, association-vs-sensorimotor tertile contrast (Mann-Whitney),
   Spearman correlation, permuted-axis null (1,000 shuffles).

Mostly reuses data already in the repo: π, the human parcellation, and the
Fulcher human myelin map.

## Files

| File | What |
|---|---|
| `01_tethering_test.py` | Per-parcel coverage, decile/tertile/Spearman, permuted-axis null |
| `02_plot.py` | 3-panel figure |
| `README.md` | This file |

## Reproduce

```bash
PYTHONPATH=src python experiments/buckner_krienen_2013_tethering/01_tethering_test.py
PYTHONPATH=src python experiments/buckner_krienen_2013_tethering/02_plot.py
```

Outputs: `outputs/logs/buckner_krienen_2013_tethering.json`,
`outputs/figures/buckner_krienen_2013_tethering.png`.
