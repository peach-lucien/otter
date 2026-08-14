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
no well-defined mouse homologue of the expanded human association cortex, so an
accurate coupling should be confident over sensorimotor cortex and sparse or
unconfident over association cortex.

The falsification logic follows from that. If OTTER's π were uniformly confident
everywhere, including over association cortex that has no clear mouse
homologue, that would signal over-fitting.

## Result

Three panels (`outputs/figures/buckner_krienen_2013_tethering.png`).

For every human cortical parcel the analysis measures OTTER's coverage, the π mass it
receives from the mouse brain (the per-column mass of the coupling), and place it on
the sensorimotor → association axis (the HCP T1w/T2w myelin map; high myelin =
sensorimotor).

Coverage is aggregated as a mass-normalised mean rather than a sum. Summing makes
coverage scale with *how many parcels a region happens to contain*, which is a
parcellation artefact rather than biology.

**The test is inconclusive.** On the canonical coupling `pi_canonical.npy`
(sha256 `bb4cae00cbca9f16c6f9cfca3b0124292b41d81643e2ef5d5511686b20f9df77`) the
sensorimotor tertile receives a mean log₁₀ coverage of −4.54 and the association
tertile −5.23, a gap of **0.68 log units**. The gap does not clear a
spatial-autocorrelation (spin) null, **p = 0.286**. The Mann–Whitney contrast is
still significant (p = 1.3×10⁻⁶), but Mann–Whitney ignores spatial autocorrelation.
The permuted-*axis* null in the log (`null/empirical_p = 0.0`) has the same weakness
and should not be quoted as support.

| Measure | Result |
|---|---|
| sensorimotor-tertile coverage | log₁₀ −4.54 |
| association-tertile coverage | log₁₀ −5.23 |
| coverage gap (mass-normalised mean) | **0.68 log units** |
| gap, spatial-autocorrelation (spin) null | **p = 0.286, not significant** |
| gap, Mann–Whitney (no spatial null) | p = 1.3×10⁻⁶ |
| coverage vs myelin, continuous | ρ = +0.095, r = 0.14, spin p = 0.169 |
| entropy vs myelin, continuous | ρ = −0.435 |

This test does not support the tethering prediction, and it does not refute it. The
test does not discriminate. The coverage claim rests on *reconstruction*-coverage,
which does track expansion and hierarchy maps under a spin null (see
`docs/03_results.md` §5). Canonical figures are in
`outputs/logs/section5_coverage_nulls.json`.

**Notes.** (1) π's per-parcel *entropy*, the diffuseness of a parcel's
mouse origin, correlates with the axis at ρ = −0.435. (2) OTTER's
Garin anchors are weighted toward sensorimotor cortex, which contributes to the
sensorimotor–association coverage gap alongside the underlying evolutionary
untethering. The two are not fully separable, but both push the same way. An
accurate coupling and the anchor set agree that association cortex is the
weakly-tethered territory. The same boundary appears in the Fulcher
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
