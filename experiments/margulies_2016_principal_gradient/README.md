# Margulies 2016 + Huntenburg 2021 principal-gradient validation

Tests whether HOMER's π preserves the cross-species principal connectivity gradient — a single brain-wide ordering, orthogonal to specific-pair anchor benchmarks.

## Why this experiment

[Margulies et al. 2016 (PNAS)](https://www.pnas.org/doi/10.1073/pnas.1608282113) introduced the principal connectivity gradient — derived by diffusion-map embedding on the human resting-state FC matrix, it spans from primary sensorimotor cortex (unimodal end) to default-mode network (transmodal end). It's the dominant organisational axis of human cortex.

[Huntenburg et al. 2021 (Nat Comm)](https://www.nature.com/articles/s41467-021-26703-z) extended the same procedure to mouse rsfMRI and showed an analogous gradient exists in mouse, broadly conserved across species.

If HOMER's π is anatomically faithful, then routing the mouse principal gradient through π should approximately reproduce the human principal gradient. This is a **single global number**: correlation of predicted vs observed human gradient over 2,094 parcels.

## Result

**Pearson r = +0.144, Spearman ρ = +0.343, analytical p = 4×10⁻¹¹**
**Permuted-π null |r| mean = +0.015, 95% CI (+0.001, +0.043). Empirical p = 0.000.**

The observed correlation is ≈ 10× the null mean and well outside the null 95% CI. The Spearman ρ being substantially larger than Pearson r indicates the *ordering* of parcels along the gradient is more preserved than absolute values — biologically the more meaningful statistic since the gradient is defined up to monotonic transformation.

**HOMER preserves the broad cross-species cortical organisation gradient.** Combined with the other validations:

| Test | Granularity | Pearson r | Status |
|---|---|---:|:---:|
| Pagani Test 2c | Network-pair Δ (36 elements) | +0.527 | Strong |
| **Margulies/Huntenburg gradient** | **Brain-wide ordering (2,094 parcels)** | **+0.144** | **Modest but p=4e-11** |
| Hodge layer markers | Within-area lamination | ~0 | Null |

This is a *brain-wide* organisational test, not driven by anchor pairs. The Pagani Test 2c (r=+0.527) is the strongest result but operates at the network-pair level. The Margulies gradient is more diffuse but covers the whole brain — establishing that HOMER's cross-species fidelity isn't only at the 22 Beauchamp anchor pairs or the network-aggregated level.

## Method

Standard Margulies-style diffusion-map embedding per species:
1. Fisher-z transform FC correlations
2. Threshold (keep top 10% per row)
3. Cosine similarity of row-profiles → affinity W
4. Symmetric-normalised graph Laplacian L = I − D^{−½} W D^{−½}
5. Second-smallest eigenvalue's eigenvector = principal gradient

Then `mouse_grad @ π` → predicted human gradient. Pearson r vs observed. Permuted-π null.

## Files

| File | What |
|---|---|
| `01_gradient_validation.py` | Compute per-species principal gradient, route mouse through π, correlate, null |
| `02_plot.py` | 3-panel figure (mouse gradient distribution, scatter, null CI) |
| `README.md` | This file |

## Reproduce

```bash
PYTHONPATH=src python experiments/margulies_2016_principal_gradient/01_gradient_validation.py
PYTHONPATH=src python experiments/margulies_2016_principal_gradient/02_plot.py
```

Outputs:
- `outputs/logs/margulies_2016_gradient.json` (full per-parcel gradients + stats)
- `outputs/figures/margulies_2016_gradient.png` (3-panel figure)

## Showcase notebook

See [`notebooks/07_margulies_huntenburg_gradient.ipynb`](../../notebooks/07_margulies_huntenburg_gradient.ipynb) for an interactive walkthrough.
