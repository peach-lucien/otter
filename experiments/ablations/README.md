# Coupling sensitivity analyses

Optional scripts for evaluating alternative coupling settings. They are not required to load or use the released coupling.

| Script | Comparison |
|---|---|
| `soft_region_anchors.py` | Region-anchor penalty strength |
| `marginal_weighting.py` | Uniform, parcel-volume and stability-weighted mouse marginals |
| `per_region_xyz.py` | Global and parcel-specific spatial-cost weighting |

Run from the repository root:

```bash
PYTHONPATH=src python experiments/ablations/soft_region_anchors.py
PYTHONPATH=src python experiments/ablations/marginal_weighting.py
PYTHONPATH=src python experiments/ablations/per_region_xyz.py
```

Outputs are written to `outputs/logs/` and, where applicable, `outputs/coupling/`.
