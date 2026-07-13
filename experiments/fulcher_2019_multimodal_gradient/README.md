# Fulcher 2019 multimodal-gradient validation

Tests whether HOMER's π carries the **mouse multimodal cortical hierarchy**
the sensorimotor → prefrontal axis that Fulcher et al. showed is shared across
cytoarchitecture, gene expression, cell density and connectivity, across to
the human cortex.

## Why this experiment

[Fulcher, Murray, Zerbi & Wang 2019 (PNAS)](https://doi.org/10.1073/pnas.1814144116)
showed that several independent modalities of mouse cortex all vary along a
single hierarchical axis from primary sensory to prefrontal areas. Their
modality-spanning measurement is the **T1w:T2w ratio**, an intracortical-myelin
proxy reported for 40 mouse isocortical areas; they also use Goulas et al.'s
**cytoarchitectural type** (eulamination 1–4) for 38 areas.

This is a Beauchamp-independent, anchor-orthogonal test on two counts: both
mouse maps are *structural* (HOMER's π is built from FC + SC), and the human
reference, the [HCP S1200 T1w/T2w myelin map](https://github.com/netneurolab/neuromaps),
is independent published data, not an anchor pair. If HOMER's π is
anatomically faithful, translating the mouse myelin hierarchy through π should
reproduce the human myelin map: heavily myelinated sensory cortex, lightly
myelinated association cortex.

## Result

Three panels (`outputs/figures/fulcher_2019_multimodal_gradient.png`):

**1. Mouse T1w:T2w → human myelin.** Translating the mouse T1w:T2w map through
π reproduces the human HCP myelin map at **Pearson r = +0.373, Spearman ρ =
+0.321** (analytical p = 2.5×10⁻⁵, n = 174 Schaefer regions), with **empirical
p = 0.000** against a 200-trial permuted-π null (null mean r ≈ 0).

> ⚠️ **But a permuted-π null is the wrong null for a smooth map.** Under a
> spatial-autocorrelation-preserving **spin null**, this correlation **does NOT
> survive: spin p = 0.11**. The same is true of the cytoarchitecture panel below
> (spin p = 0.10). Earlier versions of this README and of `docs/03_results.md`
> reported spin p = 0.021 / 0.010 and called the correspondence "specific" — those
> p-values were **hardcoded literals in a figure script** and existed in no output
> file. **We do not claim that microstructure translates.** The routed maps are
> consistent with the human myelin map, but not beyond what the spatial smoothness
> of both maps already supplies. See `outputs/logs/spin_test_gradients.json`.

**2. Routed territory is gradient-compressed.** π concentrates the whole mouse
brain onto a compact human territory: the 417 mouse isocortical parcels map onto
just **174 of 400 Schaefer regions**. On the human principal connectivity
gradient those 174 regions have **half the brain-wide spread** (SD 0.0108 vs
0.0217, compression ×0.50). Mouse isocortex lands on a narrow middle slice of
the human unimodal–transmodal axis, so the principal gradient is *not* a
hierarchy ruler within this territory (predicted-vs-gradient r = −0.204), and
the result echoes the disproportionate evolutionary expansion of human
association cortex.

**3. Cytoarchitecture → human myelin (independent modality).** A second,
independent mouse modality. Goulas cytoarchitectural type, routed through π
also predicts the human myelin map at **r = +0.362, ρ = +0.325** (p = 3.4×10⁻⁴,
empirical p = 0.000). Two unrelated mouse structural modalities converge on the
same human target — but neither clears a spin null, so this convergence rules out a
single-measurement artefact **without** establishing that π carries the hierarchy.
Two smooth maps agreeing at the level spatial smoothness already predicts is not a
correspondence.

**What this experiment actually shows.** π was fitted on connectivity. Microstructure
is the modality it never saw, and it does not transfer. That is not a failure of the
method — it is the negative half of the paper's organising claim (`docs/03_results.md`
§3): *connectional organisation transfers through π; microstructure does not.*

| Panel | Test | Pearson r | Empirical p |
|---|---|---:|:---:|
| 1 | mouse T1w:T2w → human myelin | **+0.373** | 0.000 |
| 2 | routed territory gradient SD vs all-cortex | ×0.50 | |
| 3 | mouse cytoarchitecture → human myelin | **+0.362** | 0.000 |

## Method

1. Assign every HOMER mouse parcel its Allen-acronym value from Fulcher's
   tables (T1w:T2w → 417 parcels / 36 areas; cytoarchitecture → 414 / 35).
   PTLp has no acronym in HOMER's parcellation (split into VISa/VISrl in the
   newer Allen CCF); VISal/PERI/AUDpo carry no parcels.
2. Translate each mouse map through π as a transport-weighted average:
   `predicted_h[j] = Σ_i m[i]·π[i,j] / Σ_i π[i,j]` over the assigned parcels.
3. Aggregate the predicted human map to Schaefer-400 regions; compare against
   the HCP myelin map (Pearson + Spearman).
4. Permuted-π null (200 trials, row shuffle).
5. Panel 2: compare the principal-gradient SD over all cortex vs the routed
   territory.

The human myelin map is the HCP S1200 T1w/T2w annotation (neuromaps), parcellated
onto Domhof's Schaefer-400 17Networks parcellation. HOMER's exact human
parcellation order. See `data_external/fulcher_2019_gradients/SOURCES.md`.

## Files

| File | What |
|---|---|
| `01_gradient_validation.py` | Route both mouse modalities through π, compare to human myelin, characterise the routed territory, permuted-π null |
| `02_plot.py` | 3-panel figure |
| `README.md` | This file |

## Reproduce

```bash
# depends on the Margulies experiment having been run (reuses its human gradient)
PYTHONPATH=src python experiments/margulies_2016_principal_gradient/01_gradient_validation.py
PYTHONPATH=src python experiments/fulcher_2019_multimodal_gradient/01_gradient_validation.py
PYTHONPATH=src python experiments/fulcher_2019_multimodal_gradient/02_plot.py
```

Outputs:
- `outputs/logs/fulcher_2019_gradient.json` (per-region predicted/observed maps + stats)
- `outputs/figures/fulcher_2019_multimodal_gradient.png` (3-panel figure)

