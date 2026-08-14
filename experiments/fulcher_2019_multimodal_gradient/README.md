# Fulcher 2019 multimodal-gradient validation

We ask whether OTTER's π carries the **mouse multimodal cortical hierarchy**, the
sensorimotor to prefrontal axis that Fulcher et al. showed is shared across
cytoarchitecture, gene expression, cell density and connectivity, across to the
human cortex.

## Why this experiment

[Fulcher, Murray, Zerbi & Wang 2019 (PNAS)](https://doi.org/10.1073/pnas.1814144116)
showed that several independent modalities of mouse cortex vary along a single
hierarchical axis from primary sensory to prefrontal areas. Their modality-spanning
measurement is the **T1w:T2w ratio**, an intracortical-myelin proxy reported for 40
mouse isocortical areas. They also use Goulas et al.'s **cytoarchitectural type**, an
ordinal eulamination scale, reported for 38 areas.

The test is Beauchamp-independent and anchor-orthogonal on two counts. Both mouse maps are
structural, while π is built from FC and SC, and the human reference, the
[HCP S1200 T1w/T2w myelin map](https://github.com/netneurolab/neuromaps), is independent
published data rather than an anchor pair.

## Result

π is `outputs/coupling/pi_canonical.npy`, sha256 `bb4cae00cbca9f16c6f9cfca3b0124292b41d81643e2ef5d5511686b20f9df77`.
Three panels (`outputs/figures/fulcher_2019_multimodal_gradient.png`).

**1. Mouse T1w:T2w → human myelin.** The mouse T1w:T2w proxy, covering 570 mouse parcels
across 39 Allen areas, is routed through π and the predicted map compared to the observed
human HCP myelin map. Over 388 Schaefer regions the two agree at **Pearson r = +0.470,
Spearman ρ = +0.494**. None of 200 permuted-π trials reached that r, and the permuted null
has mean r = −0.011.

**2. Routed territory on the principal gradient.** π reaches 388 of 400 Schaefer regions.
On the human principal connectivity gradient those regions have the same spread as the whole
cortex, SD 0.00157 against 0.00157, ratio 1.00, so the routed territory spans the human
unimodal to transmodal axis rather than a slice of it. The predicted T1w:T2w map tracks that
gradient at **r = +0.560** over 388 regions, with the gradient oriented so its high end is
heavily myelinated cortex.

**3. Cytoarchitecture → human myelin.** The same routing applied to a second, independent
mouse modality. Goulas cytoarchitectural type, covering 37 Allen areas, predicts the human
myelin map at **r = +0.473, ρ = +0.560** over 388 regions. None of 200 permuted-π trials
reached that r, and the permuted null has mean r = −0.011.

### Spin nulls

`outputs/logs/out_c2_nulls.json` scores both transfers at parcel level against two spatial
nulls. The translation spin rotates the mouse input and routes the rotated map through the
real coupling. The human-side spin rotates the human target map. Both preserve spatial
autocorrelation, so neither null sits at zero.

| Transfer | \|r\| | parcels | translation spin p | human-side spin p |
|---|---:|---:|---:|---:|
| mouse T1w:T2w → human myelin | 0.505 | 1,789 | 0.005 | < 0.001 |
| mouse cytoarchitecture → human myelin | 0.534 | 1,787 | < 0.001 | < 0.001 |

Null mean \|r\| across these four tests runs from 0.138 to 0.243. The nulls use 1,000
rotations, so p values logged as 0.000 are reported here as p < 0.001.

### Resolution of the mouse inputs

The mouse cytoarchitecture map takes five distinct values and the T1w:T2w proxy takes 39,
each spread over about 1,790 parcels. The parcel count is therefore not a count of
independent observations, and no analytic p-value is quoted for these correlations. The
spin nulls carry the inference.

**What this experiment shows.** Two mouse structural measurements that π never saw, an
intracortical-myelin proxy and cytoarchitectural type, both predict the human HCP myelin
map when routed through π. Each clears a translation spin and a human-side spin at parcel
level, and each clears a permuted-π null at region level. Microstructure transfers through π.

## Method

1. Assign every OTTER mouse parcel its Allen-acronym value from Fulcher's tables. 39 of the
   40 T1w:T2w areas and 37 of the 38 cytoarchitecture areas have parcels in OTTER's mouse
   parcellation.
2. Route each mouse map through π as a transport-weighted average,
   `predicted_h[j] = Σ_i m[i]·π[i,j] / Σ_i π[i,j]` over the assigned parcels.
3. Aggregate the predicted human map to Schaefer-400 regions and compare against the HCP
   myelin map with Pearson and Spearman correlation.
4. Permuted-π null, 200 trials, row shuffle.
5. Panel 2 compares the principal-gradient SD over all cortex against the routed territory.

The human myelin map is the HCP S1200 T1w/T2w annotation (neuromaps), parcellated onto
Domhof's Schaefer-400 17Networks parcellation, in OTTER's human parcellation order. See
`data_external/fulcher_2019_gradients/SOURCES.md`.

## Files

| File | What |
|---|---|
| `01_gradient_validation.py` | Routes both mouse modalities through π, compares to human myelin, characterises the routed territory, runs the permuted-π null. Writes `outputs/logs/fulcher_2019_gradient.json` |
| `02_plot.py` | Reads that log, writes `outputs/figures/fulcher_2019_multimodal_gradient.png` |
| `README.md` | This file |

The parcel-level spin nulls come from `outputs/logs/out_c2_nulls.json`. No producer script
for that log is present in the repo.

## Reproduce

```bash
# depends on the Margulies experiment having been run (reuses its human gradient)
PYTHONPATH=src python experiments/margulies_2016_principal_gradient/01_gradient_validation.py
PYTHONPATH=src python experiments/fulcher_2019_multimodal_gradient/01_gradient_validation.py
PYTHONPATH=src python experiments/fulcher_2019_multimodal_gradient/02_plot.py
```
