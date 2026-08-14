# Whitesell 2021 DMN refinement

Where OTTER's coupling routes a mouse-DMN parcel set taken from [Whitesell et al. 2021, Neuron](https://www.cell.com/neuron/fulltext/S0896-6273(21)00006-X), instead of OTTER's PAIRID-derived `frontal_dmn` and `temporal_dmn` networks.

## Parcel set

Nine DSURQE regions, 176 mouse parcels, 9.4 % of mouse parcels.

| DSURQE region | parcels |
|---|---:|
| Entorhinal area, medial part, dorsal zone | 47 |
| Subiculum | 35 |
| Retrosplenial area | 30 |
| Anterior cingulate area | 23 |
| Field CA1 | 18 |
| Prelimbic area | 11 |
| Posterior parietal association areas | 6 |
| Entorhinal area, medial part, ventral zone | 4 |
| Infralimbic area | 2 |

## Method

Coupling `outputs/coupling/pi_canonical.npy`, sha256 `bb4cae00cbca9f16c6f9cfca3b0124292b41d81643e2ef5d5511686b20f9df77`. The pi rows for the 176 mouse parcels are summed, row-normalised, and aggregated over the human Yeo-7 networks plus a Subcortical category. The null for a network is its share of the 2,094 human parcels. The ratio is row-mass divided by that null.

## Result

| Human network | row-mass | n parcels | null | ratio |
|---|---:|---:|---:|---:|
| DMN | 38.0 % | 439 | 21.0 % | 1.81x |
| Subcortical | 19.3 % | 326 | 15.6 % | 1.24x |
| Salience | 13.2 % | 248 | 11.8 % | 1.12x |
| Limbic | 11.8 % | 60 | 2.9 % | 4.13x |
| Visual | 10.4 % | 255 | 12.2 % | 0.86x |
| Control | 4.3 % | 271 | 12.9 % | 0.33x |
| DorsAtten | 1.9 % | 242 | 11.6 % | 0.17x |
| SomatoMotor | 0.9 % | 253 | 12.1 % | 0.07x |

DMN takes the most mass, at 1.81x its parcel-count null. Limbic has the highest ratio, 4.13x, on 60 parcels. Control, DorsAtten, SomatoMotor and Visual fall below their nulls.

## Comparison

Same row-normalised pi-mass statistic and the same 439-parcel Yeo-DMN mask, on the same coupling. The baselines are read from `outputs/logs/autism_subtypes_network_crossval.json` and `outputs/logs/coletta_2020_cross_species_rsn.json`.

| mouse parcel set | row-mass into Yeo-DMN |
|---|---:|
| Whitesell 2021 | 38.0 % |
| OTTER `DMN` network, Pagani Test 1 | 26.0 % |
| OTTER `frontal_dmn`, Coletta sub-test A | 11.3 % |
| OTTER `temporal_dmn`, Coletta sub-test A | 49.3 % |

## Verdict

The Whitesell parcel set sends 38.0 % of its pi mass into Yeo-DMN, and Yeo-DMN is the network it sends the most mass to. That is above the OTTER `DMN` network at 26.0 % and above `frontal_dmn` at 11.3 %, and below `temporal_dmn` at 49.3 %. The three OTTER definitions do not agree with each other, so the run does not establish that swapping in the Whitesell definition sharpens the DMN to DMN correspondence. The refinement is inconclusive on that question.

## Anchor pack

No `whitesell_dmn` anchor pack was built. The run measures routing only and carries nothing that decides the question either way.

## Files

| File | What |
|---|---|
| `01_whitesell_dmn_refinement.py` | Aggregates pi for the Whitesell-DMN parcels into human Yeo-7 row-mass |
| `README.md` | This file |

## Reproduce

```bash
PYTHONPATH=src python experiments/whitesell_2021_dmn/01_whitesell_dmn_refinement.py
```

Output: `outputs/logs/whitesell_2021_dmn_refinement.json`.
