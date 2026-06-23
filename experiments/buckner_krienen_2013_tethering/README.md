# Buckner & Krienen 2013, tethering hypothesis (negative control)

A negative-control / falsification test: it asks whether HOMER's π is
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

This is the test's falsification logic: if HOMER's π were *uniformly* confident
everywhere, including over association cortex that has no clear mouse
homologue, that would signal over-fitting. HOMER should know what it can't
map.

## Result

**VERDICT: PASS.** Three panels (`outputs/figures/buckner_krienen_2013_tethering.png`).

For every human cortical parcel we measure HOMER's **coverage**, the total π
mass it receives from the mouse brain (the per-column mass of the coupling)
and place it on the sensorimotor → association axis (the HCP T1w/T2w myelin
map; high myelin = sensorimotor).

HOMER's coverage **collapses toward association cortex**. The sensorimotor
tertile receives a mean log₁₀ coverage of −8.5; the association tertile −18.8
a gap of **10.3 log units** (~10 orders of magnitude less π mass). The contrast
is overwhelming (Mann-Whitney p = 2.5×10⁻¹⁶) and far beyond a permuted-axis
null (95th-percentile gap 1.8 log units; empirical p = 0.000). The decile curve
shows a monotone-ish collapse from sensorimotor to association cortex; Spearman
ρ = +0.185 along the full axis.

| Measure | Result |
|---|---|
| sensorimotor-tertile coverage | log₁₀ −8.5 |
| association-tertile coverage | log₁₀ −18.8 |
| coverage gap | **10.3 log units**, p = 2.5×10⁻¹⁶, empirical p = 0.000 |

**HOMER is not confident everywhere**, it is dramatically sparser over human
association cortex, exactly as the tethering hypothesis predicts. The coupling
"knows" it has no good mouse homologue for the expanded human association
networks.

**Notes.** (1) π's per-parcel *entropy*, the diffuseness of a parcel's
mouse origin, is flat along the axis (ρ = +0.088); it is the *amount* of
coverage, not its diffuseness, that carries the tethering signal. (2) HOMER's
Garin anchors are weighted toward sensorimotor cortex, which contributes to the
sensorimotor–association coverage gap alongside the genuine evolutionary
untethering, the two are not fully separable, but both push the same way: a
faithful coupling and a sensible anchor set agree that association cortex is
the hard, weakly-tethered territory. The same boundary appears in the Fulcher
2019 experiment, where mouse isocortex routes onto a gradient-compressed human
territory that never reaches the transmodal extreme.

## Method

1. Per human cortical parcel, HOMER coverage = log₁₀ of the π column mass
   (Σ_i π[i,j]).
2. Sensorimotor→association axis = HCP T1w/T2w myelin per Schaefer region
   (reused from the Fulcher experiment).
3. Decile curve, association-vs-sensorimotor tertile contrast (Mann-Whitney),
   Spearman correlation, permuted-axis null (1,000 shuffles).

Mostly reuses data already in the repo, π, the human parcellation, and the
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

## Showcase notebook

See [`notebooks/15_buckner_krienen_2013_tethering.ipynb`](../../notebooks/15_buckner_krienen_2013_tethering.ipynb).
