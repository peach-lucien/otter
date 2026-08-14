# Ablations

Three ablation experiments comparing the production configuration (soft anchors, uniform mouse marginal, area-level xyz weighting) against variants. The production configuration outperforms the variants in all three.

## Files

| Script | Tested | Outcome |
|---|---|---|
| `soft_region_anchors.py` | Soft anchors (λ_outside ≈ 0.15) vs hard anchors (λ_outside → 0) | Soft is the default, keeps anchor supervision conservative about cases where structural cost disagrees with the prescribed pair |
| `marginal_weighting.py` | Uniform mouse marginal (1/n) vs volume-weighted (parcel-volume proportional) | Uniform is the production setting; volume weighting biased the human-side mass distribution toward large parcels without improving Beauchamp top-K |
| `per_region_xyz.py` | Per-region xyz weighting (downweight xyz cost in spatially-inverted regions like tectum) | Local intervention did not reproduce the global xyz effect; superseded by region-anchor packs for inverted regions |

## Reproduce

```bash
PYTHONPATH=src python experiments/ablations/soft_region_anchors.py
PYTHONPATH=src python experiments/ablations/marginal_weighting.py
PYTHONPATH=src python experiments/ablations/per_region_xyz.py
```

Results are written to `outputs/logs/`.
