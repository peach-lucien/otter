# FC/SC/xyz weight selection

This document describes the validation-based parameter optimization introduced
in `pipeline/05i_weight_selection.py`. The goal is to avoid treating the
production relational weights as hand-tuned constants.

## Question

The previous production configuration was:

```text
fc_weight = 0.7
sc_weight = 0.3
xyz_weight = 0.5
```

In `MultimodalFGW`, `fc_weight` and `sc_weight` are normalized across active
relational modalities, so they define the FC/SC mixture ratio. The question is:

```text
Which FC/SC ratio and xyz prior weight gives the best validation tradeoff?
```

## Search Space

The current sweep evaluates:

```text
FC ratios:  1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.0
SC ratios:  0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0
xyz_weight: 0.0, 0.25, 0.5, 0.75, 1.0
```

This is a 40-candidate grid. For each candidate, the script runs
leave-one-network-out anchor CV and a full-model brain-wide FC translation
evaluation.

## Selection Score

The selected candidate maximizes:

```text
score =
  0.45 * anchor_top1
+ 0.35 * FC_translation_r
+ 0.20 * within_network_FC_r
```

This score intentionally balances two concerns:

- anchor recovery, because the model should recover held-out homologous anchors;
- brain-wide FC preservation, because anchor recovery alone can be explained by
  spatial anchor interpolation.

The score weights are explicit CLI parameters:

```bash
PYTHONPATH=src python pipeline/05i_weight_selection.py \
  --score-anchor 0.45 \
  --score-fc 0.35 \
  --score-within 0.20
```

## How To Run

From the repository root:

```bash
PYTHONPATH=src python pipeline/05i_weight_selection.py
```

For a quick smoke test on a subset:

```bash
PYTHONPATH=src python pipeline/05i_weight_selection.py \
  --fc-ratios 0.8,0.7,0.6 \
  --xyz-weights 0.25,0.5 \
  --networks visual,brainstem
```

## Outputs

The optimization writes:

```text
outputs/logs/weight_selection.json
outputs/logs/weight_selection.csv
outputs/logs/weight_selection_selected.json
```

`weight_selection.json` stores the full per-candidate details, including
per-network anchor CV and FC translation metrics.

`weight_selection.csv` is the flat ranked table.

`weight_selection_selected.json` is the small production-facing artifact read by
`pipeline/04_solve_production.py --config fc_plus_SC_selected`.

## Selected Configuration

The full sweep selected:

```text
fc_weight = 0.8
sc_weight = 0.2
xyz_weight = 0.25
```

Metrics for the selected candidate:

| metric | value |
|---|---:|
| selection score | 0.5810 |
| anchor top1 | 81.0% |
| anchor top5 | 100.0% |
| anchor mean rank | 1.238 |
| anchor mean xyz distance | 0.0200 |
| FC translation r | 0.368 |
| within-network FC r | 0.439 |
| cross-network FC r | 0.207 |
| human nodes kept | 1461 |

## Top Candidates

| config | score | anchor top1 | FC r | within-network r |
|---|---:|---:|---:|---:|
| FC 0.8 / SC 0.2 / xyz 0.25 | 0.5810 | 81.0% | 0.368 | 0.439 |
| FC 0.6 / SC 0.4 / xyz 0.5 | 0.5769 | 81.0% | 0.360 | 0.432 |
| FC 1.0 / SC 0.0 / xyz 0.25 | 0.5734 | 78.6% | 0.374 | 0.444 |
| FC 0.9 / SC 0.1 / xyz 0.25 | 0.5722 | 78.6% | 0.371 | 0.444 |
| FC 0.9 / SC 0.1 / xyz 0.5 | 0.5699 | 78.6% | 0.366 | 0.441 |

## Production Use

After running the sweep, fit production with:

```bash
PYTHONPATH=src python pipeline/04_solve_production.py --config fc_plus_SC_selected
```

This reads:

```text
outputs/logs/weight_selection_selected.json
```

and passes the selected weights to `MultimodalFGW`.

## Interpretation

The selected configuration is slightly more FC-heavy and uses a weaker xyz prior
than the previous hardcoded production default:

```text
old production: FC 0.7 / SC 0.3 / xyz 0.5
selected:       FC 0.8 / SC 0.2 / xyz 0.25
```

This means SC is still useful under the validation score, but the sweep prefers
20% SC rather than 30%. It also suggests that `xyz_weight=0.5` may over-emphasize
the spatial prior relative to the combined anchor and FC-translation objectives.

## Caveat

This is a single-level validation sweep over the existing LONO benchmark plus
in-sample FC translation. For a stricter methodological claim, use nested
validation: tune weights inside each outer held-out network, then evaluate the
selected weights on the untouched outer fold.
