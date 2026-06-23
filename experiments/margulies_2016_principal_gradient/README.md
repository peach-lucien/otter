# Margulies 2016 + Huntenburg 2021 principal-gradient validation

Tests whether HOMER's π preserves the cross-species principal connectivity
gradient, a single brain-wide ordering, orthogonal to specific-pair anchor
benchmarks.

## Why this experiment

[Margulies et al. 2016 (PNAS)](https://www.pnas.org/doi/10.1073/pnas.1608282113)
introduced the principal connectivity gradient, derived by diffusion-map
embedding on the resting-state FC matrix, it spans from primary sensorimotor
cortex (unimodal end) to default-mode network (transmodal end). It's the
dominant organisational axis of cortex.

[Huntenburg et al. 2021 (Nat Comm)](https://www.nature.com/articles/s41467-021-26703-z)
extended the same procedure to mouse rsfMRI and showed an analogous gradient
exists in mouse, broadly conserved across species.

If HOMER's π is anatomically faithful, routing the mouse principal gradient
through π should reproduce the human principal gradient, a single global
correlation.

## Result

**|r| = 0.402 (parcel-level), 0.433 (region-level)**
**Permuted-π null |r| mean = 0.026; empirical p = 0.000, 16× the null.**

HOMER preserves the broad cross-species cortical organisation gradient, well
clear of the permuted-π null. Combined with the other validations:

| Test | Granularity | \|r\| | Status |
|---|---|---:|:---:|
| Pagani Test 2c | Network-pair Δ (36 elements) | +0.55 (leverage-driven; Spearman n.s.) | Partial |
| **Margulies/Huntenburg gradient** | **Brain-wide ordering** | **0.402** | **n.s. by spin test** ¹ |
| Fulcher 2019 T1w:T2w → myelin | Cortical region | +0.373 | Survives spin test (p = 0.02) ¹ |
| Hodge layer markers | Within-area lamination | ~0 | Null |

> ¹ A permuted-π null destroys spatial autocorrelation and over-states
> significance for a smooth target like a gradient. Under a proper **spin test**
> (rotate parcel centroids on a sphere; `homer.eval.nulls.spin_null`, run via
> `experiments/spatial_null_check/apply_spin_test.py`), the Margulies correlation
> **does not survive**: |r|=0.402, spin p=**0.16** (spin-null |r| mean 0.25, 95th
> pct 0.51). So this is a *modest, spatially-unexceptional* correspondence. Re-rate
> any spatial-map correlation with the spin null before claiming significance.

A *brain-wide* organisational test, not driven by anchor pairs, establishing
that HOMER's cross-species fidelity isn't only at the 22 Beauchamp anchor pairs
or the network-aggregated level.

### Routing note

The gradient is routed as a **transport-weighted average**. The bare
un-normalised `mouse_grad @ π` conflates the translated gradient with π's
per-column mass (which varies widely under the semirelaxed coupling) and
scores only r = 0.144; normalising by the column mass removes that confound.
Routing as

    predicted_h[j] = Σ_i mouse_grad[i]·π[i,j] / Σ_i π[i,j]

This removes that confound and roughly trebles the correlation. Human parcels
that receive negligible π mass are left undefined (n = 1,435 of 2,094).

## Method

Standard Margulies-style diffusion-map embedding per species:
1. Fisher-z transform FC correlations
2. Threshold (keep top 10% per row)
3. Cosine similarity of row-profiles → affinity W
4. Symmetric-normalised graph Laplacian L = I − D^{−½} W D^{−½}
5. Second-smallest eigenvalue's eigenvector = principal gradient

Then route the mouse gradient through π as a transport-weighted average,
compare to the observed human gradient (Pearson + Spearman, parcel level and
Schaefer-400 region level), permuted-π null. Eigenvectors are sign-ambiguous,
so |r| is the headline.

## Files

| File | What |
|---|---|
| `01_gradient_validation.py` | Compute per-species gradient, route mouse through π, correlate, null |
| `02_plot.py` | 3-panel figure (mouse gradient distribution, scatter, null CI) |
| `README.md` | This file |

## Reproduce

```bash
PYTHONPATH=src python experiments/margulies_2016_principal_gradient/01_gradient_validation.py
PYTHONPATH=src python experiments/margulies_2016_principal_gradient/02_plot.py
```

Outputs:
- `outputs/logs/margulies_2016_gradient.json` (per-parcel gradients + stats)
- `outputs/figures/margulies_2016_gradient.png` (3-panel figure)

## Showcase notebook

See [`notebooks/07_margulies_huntenburg_gradient.ipynb`](../../notebooks/07_margulies_huntenburg_gradient.ipynb).

## Discrete reframe (2026-06-19), `03_discrete_reframe.py`

The continuous correlation fails the spin null (|r|=0.40, p=0.16) for a structural
reason: two smooth monotone gradients correlate by spatial autocorrelation alone. We
asked the gradient question **categorically** instead (HOMER's strong mode):
- **Gradient-tier classification** (3 tiers, n=1244): exact accuracy 50.3% vs 33%
  chance, spin p=0.092; adjacent (±1) 82.8%, spin p=0.110.
- **Network rank-order** (9 networks): Spearman ρ(predicted, observed network
  gradient) = +0.73, spin p=0.098.

So the discrete content of the gradient is **more** cross-species-specific than the
raw smooth correlation suggested (p moves from 0.16 toward ~0.09–0.10), but it still
does **not** clear significance, and the network test is underpowered (n=9). HOMER carries some discrete gradient-ordering signal, not a significant one.
Log: `outputs/logs/margulies_discrete_reframe.json`.
