# Anchor-interpolation baseline results

Status: **complete**

## Weighted Leave-One-Network-Out Metrics

| model | tau scales | top1 (95% CI) | top5 | mean rank (95% CI) | full top5 | FC translation r |
|---|---:|---:|---:|---:|---:|---:|
| `uniform` | n/a | 26.2% [11.9%, 40.5%] | 85.7% | 3.12 [2.5, 3.81] | 2.4% | n/a |
| `visible_anchor_prior` | n/a | 26.2% [11.9%, 40.5%] | 85.7% | 3.12 [2.5, 3.81] | 0.0% | n/a |
| `nearest_anchor_delta` | n/a | 26.2% [11.9%, 40.5%] | 85.7% | 3.12 [2.5, 3.81] | 2.4% | 0.537 |
| `mouse_kernel_delta` | m=0.25, h=1 | 26.2% [11.9%, 40.5%] | 85.7% | 3.12 [2.5, 3.81] | 0.0% | 0.541 |
| `mouse_kernel_human_kernel` | m=2, h=4 | 76.2% [64.3%, 88.1%] | 100.0% | 1.33 [1.14, 1.52] | 21.4% | 0.242 |

## Bandwidth Tuning

- `mouse_kernel_delta` selected tau_mouse_scale=0.25, tau_human_scale=1; top1=26.2%, mean_rank=3.12
- `mouse_kernel_human_kernel` selected tau_mouse_scale=2, tau_human_scale=4; top1=76.2%, mean_rank=1.33

## Existing HOMER Log Summaries

- `homer_lono_baseline_fc_only` from `outputs/logs/multimodal_cv.json`
  top1=78.6%, top5=100.0%, pair_id=78.6%, mean_rank=1.26, mean_xyz_dist=0.0212
- `homer_lono_fc_plus_SC` from `outputs/logs/multimodal_cv.json`
  top1=81.0%, top5=100.0%, pair_id=81.0%, mean_rank=1.24, mean_xyz_dist=0.02
- `homer_lono_no_xyz` from `outputs/logs/garin_supervised_cv_no_xyz.json`
- `homer_lono_fc_only_summary` from `outputs/logs/garin_supervised_cv.json`
- `homer_full_space_eval` from `outputs/logs/full_space_eval.json`
- `homer_fc_translation` from `outputs/logs/fc_translation.json`
- `homer_null_distributions` from `outputs/logs/null_distributions.json`

