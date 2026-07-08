# Anchor-interpolation baseline

This experiment asks how much HOMER gains from FC, SC, and xyz beyond the
information already present in the anchors.

The baseline family here does **not** solve FGW and does **not** use FC or SC.
It builds a full probabilistic coupling from the visible Garin anchors only:

```text
mouse parcel -> nearby visible mouse anchors -> paired human anchor kernels
```

The benchmark uses the same leave-one-network-out Garin-anchor folds as
`pipeline/05a_anchor_cv.py`, so a held-out region's own anchor is not visible to
the baseline.

## Models

| Model | Uses mouse xyz? | Uses human xyz? | Uses FC/SC? | Description |
|---|---:|---:|---:|---|
| `uniform` | no | no | no | every mouse parcel maps uniformly over human parcels |
| `visible_anchor_prior` | no | no | no | every mouse parcel maps uniformly over visible human anchors |
| `nearest_anchor_delta` | yes | no | no | each mouse parcel inherits its nearest visible anchor's paired human anchor |
| `mouse_kernel_delta` | yes | no | no | soft mouse-side interpolation over visible anchors, delta on paired human anchors |
| `mouse_kernel_human_kernel` | yes | yes | no | soft mouse-side interpolation plus human-side spatial smoothing around paired anchors |

The comparison table also pulls already-committed HOMER LONO results from
`outputs/logs/multimodal_cv.json`, `outputs/logs/garin_supervised_cv_no_xyz.json`,
and related logs when present.

## Bandwidth tuning and confidence intervals

Kernel baselines can be tuned with `--tune-bandwidths`. The grid values are
multipliers on the automatic bandwidth, which is the median nearest-neighbour
distance among visible anchors in each fold. The selected candidate maximises
held-out top-1 accuracy, then minimises mean rank and xyz distance.

The script also stores per-anchor rows and percentile bootstrap confidence
intervals over the 42 held-out anchor observations. These intervals describe
sampling variability of the reported anchor observations. They do not correct
for the optimism introduced when the bandwidth grid is selected on the same
LONO benchmark.

## Plots

Generate figures from the latest JSON results with:

```bash
PYTHONPATH=src /opt/anaconda3/envs/retune/bin/python experiments/anchor_interpolation_baseline/plot_results.py
```

The script writes PNG and SVG versions under
`experiments/anchor_interpolation_baseline/results/plots/`:

```text
anchor_accuracy_with_ci
rank_and_xyz_distance
full_space_and_fc_translation
fc_translation_breakdown
per_network_top1_heatmap
bandwidth_tuning_surface
```

`fc_translation_breakdown` is the clearest single plot for the final
HOMER-vs-baseline story: it compares the tuned anchor-only spatial baseline
against the validation-selected HOMER configuration on overall, within-network,
and cross-network FC translation.

## Run

From the repository root:

```bash
python scripts/fetch_data.py
PYTHONPATH=src /opt/anaconda3/envs/retune/bin/python experiments/anchor_interpolation_baseline/run_baselines.py
```

To tune the anchor-only bandwidths and add bootstrap confidence intervals:

```bash
PYTHONPATH=src /opt/anaconda3/envs/retune/bin/python experiments/anchor_interpolation_baseline/run_baselines.py --tune-bandwidths
```

Useful options:

```text
--tau-grid 0.25,0.5,1,2,4,8
--bootstrap 2000
--bootstrap-seed 123
--skip-fc-translation
```

If you use a different environment, replace the Python path. The script writes:

```text
experiments/anchor_interpolation_baseline/results/anchor_interpolation_baseline.json
experiments/anchor_interpolation_baseline/results/summary.md
experiments/anchor_interpolation_baseline/results/plots/
experiments/anchor_interpolation_baseline/results/homer_anchor_baseline_results.xlsx
```

## Interpretation

The decisive comparison is:

```text
anchor interpolation
vs HOMER selected FC/SC/xyz
vs HOMER FC-only
vs HOMER FC+SC
```

If the anchor interpolator is close to HOMER on Beauchamp or anchor-CV metrics,
that benchmark is mostly measuring anchor proximity. If HOMER beats it on FC
translation or independent downstream validations, FC/SC are adding structure
beyond anchor interpolation.

## Results Summary

The tuned anchor-only spatial baseline is `mouse_kernel_human_kernel`. It uses
Gaussian/RBF interpolation from visible mouse anchors to paired human anchor
neighbourhoods, but it does not use FC or SC.

The validation-selected HOMER configuration comes from
`outputs/logs/weight_selection_selected.json`:

```text
HOMER selected = FC 0.8 / SC 0.2 / xyz 0.25
```

Main comparison:

| model | anchor top1 | anchor top5 | FC overall r | within-network r | cross-network r |
|---|---:|---:|---:|---:|---:|
| Tuned anchor spatial baseline | 76.2% | 100.0% | 0.242 | 0.066 | 0.205 |
| HOMER selected | 81.0% | 100.0% | 0.368 | 0.439 | 0.207 |
| HOMER FC-only | 78.6% | 100.0% | 0.364 | 0.447 | 0.199 |
| HOMER FC+SC old production | 81.0% | 100.0% | 0.361 | 0.444 | 0.198 |

This shows two different stories:

1. Anchor recovery is partly explainable by spatial interpolation around anchors.
   The tuned anchor-only baseline reaches 76.2% top1, close to HOMER.
2. Brain-wide FC preservation is not explained by anchor interpolation. The
   anchor-only model has within-network FC r = 0.066, while HOMER selected has
   within-network FC r = 0.439.

So the strongest evidence for HOMER is not merely that it recovers held-out
anchors. The stronger evidence is that it preserves functional structure across
the mapped human connectome better than a smooth anchor-only spatial prior.

## Artifacts

Code:

```text
experiments/anchor_interpolation_baseline/run_baselines.py
experiments/anchor_interpolation_baseline/plot_results.py
```

Results:

```text
experiments/anchor_interpolation_baseline/results/anchor_interpolation_baseline.json
experiments/anchor_interpolation_baseline/results/summary.md
experiments/anchor_interpolation_baseline/results/homer_anchor_baseline_results.xlsx
experiments/anchor_interpolation_baseline/results/plots/
```

The Excel workbook contains compact result tables:

```text
Dashboard
Main Metrics
Brainwide FC
FC Translation Table
Anchor Accuracy Table
Full Space
Tuning Grid
Baseline Definitions
```

## Full-Space Caveat

The full-space panel still uses the existing `HOMER FC+SC` full-space reference
from `outputs/logs/full_space_eval.json`. A separate full-space evaluation has
not yet been run for `HOMER selected`.
