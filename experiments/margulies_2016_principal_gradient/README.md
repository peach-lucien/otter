# Margulies 2016 + Huntenburg 2021 principal-gradient validation

We asked whether HOMER's π preserves the cross-species principal connectivity
gradient, a single brain-wide ordering that is orthogonal to the specific-pair
anchor benchmarks.

## Why this experiment

[Margulies et al. 2016 (PNAS)](https://www.pnas.org/doi/10.1073/pnas.1608282113)
introduced the principal connectivity gradient, derived by diffusion-map
embedding on the resting-state FC matrix. It spans from primary sensorimotor
cortex (unimodal end) to default-mode network (transmodal end), and it is the
dominant organisational axis of cortex.

[Huntenburg et al. 2021 (Nat Comm)](https://www.nature.com/articles/s41467-021-26703-z)
extended the same procedure to mouse rsfMRI and showed an analogous gradient
exists in mouse, broadly conserved across species.

If HOMER's π is anatomically faithful, routing the mouse principal gradient
through π should reproduce the human principal gradient. That is a single global
correlation, with no anchor pair involved.

## Result

We routed the mouse principal gradient through π and correlated the predicted
human gradient with the observed one.

**|r| = 0.54 (parcel-level, n = 1,244), |r| = 0.62 (region-level)**
**Permuted-π null |r| mean = 0.03 (p < 0.001). Spin null: p = 0.004, so the correlation survives.**

The principal functional-connectivity gradient, the unimodal→transmodal axis,
translates across species through π. It also survives reduction to discrete
structure: the rank order of the nine human networks along the gradient is
recovered at ρ = 0.73 (spin p = 0.043), and a three-tier discretisation is
classified at 52 % against 33 % chance and a 34 % spin null (p = 0.001).

| Test | Granularity | \|r\| | Spin null |
|---|---|---:|:---:|
| **Margulies/Huntenburg gradient** | **Brain-wide ordering** | **0.54** | **p = 0.004, survives** |
| Coletta 2020 resting-state networks | Network correspondence | 6/10 vs 1.2 expected | p = 0.002, survives |
| Fulcher 2019 T1w:T2w → myelin | Cortical region | +0.37 | p = 0.11, **does not survive** |
| Fulcher 2019 cytoarchitecture → myelin | Cortical region | +0.36 | p = 0.10, **does not survive** |

Those four rows are the paper's organising claim: **connectional organisation
transfers through π; microstructure does not.** π was fitted on connectivity.
See `docs/03_results.md` §3.

This is a *brain-wide* organisational test that no anchor pair drives, so it
establishes that HOMER's cross-species fidelity extends beyond the Beauchamp
anchor pairs and the network-aggregated level.

---

## ⚠️ The bug this experiment once had

This README previously reported **|r| = 0.402, spin p = 0.16, "does not survive"**,
and the method section below prescribed the **second-smallest eigenvalue's
eigenvector** as the principal gradient.

That instruction is **wrong for this data**. The first non-trivial component here is
an **anterior–posterior spatial axis**; the unimodal→transmodal hierarchy is the
**second**. We verified this in both species:

| component | vs published Margulies G1 | vs that species' own T1w:T2w |
|---|---:|---:|
| comp 1 (what the old code took) | \|ρ\| = 0.12 | human −0.13 / mouse −0.28 |
| **comp 2 (correct)** | **\|ρ\| = 0.93** | **human +0.59 / mouse +0.57** |

The consequence was severe. Routing an A–P **spatial** axis and then testing it
against a **spatial-autocorrelation-preserving** spin null is close to tautological.
It produced a confident *false negative* that the project believed for months, and
the result went unexamined because it agreed with what the project expected of the
gradient at the time.

The component index is no longer hard-coded. `principal_gradient()` in
`01_gradient_validation.py` *selects* the component by its correlation with an
external hierarchy reference (that species' own T1w:T2w map), and
`experiments/validation/00_validate_published_maps.py` asserts that every named
"published" map still matches the source it is named after.

### Routing note

The gradient is routed as a **transport-weighted average**. The bare
un-normalised `mouse_grad @ π` conflates the translated gradient with π's
per-column mass (which varies widely under the semirelaxed coupling) and
scores only r = 0.144. Normalising by the column mass removes that confound.
Routing as

    predicted_h[j] = Σ_i mouse_grad[i]·π[i,j] / Σ_i π[i,j]

roughly trebles the correlation. Human parcels
that receive negligible π mass are left undefined (n = 1,435 of 2,094).

## Method

Standard Margulies-style diffusion-map embedding per species:
1. Fisher-z transform FC correlations
2. Threshold (keep top 10% per row)
3. Cosine similarity of row-profiles → affinity W
4. Symmetric-normalised graph Laplacian L = I − D^{−½} W D^{−½}
5. Take the first `n_comp` non-trivial diffusion components (D^-1/2 · u_k, and not the
   raw symmetric-Laplacian eigenvector; the two differ by a degree weighting)
6. **SELECT** the unimodal→transmodal component by its |ρ| with that species' own
   T1w:T2w myelin map, a reference external to the FC data. Both species select
   component 2. It is not safe to assume the first component; see the bug note above.

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


## Discrete reframe (2026-06-19), `03_discrete_reframe.py`

The continuous correlation fails the spin null (|r|=0.40, p=0.16) for a structural
reason: two smooth monotone gradients correlate by spatial autocorrelation alone. We
therefore asked the gradient question **categorically** instead (HOMER's strong mode):
- **Gradient-tier classification** (3 tiers, n=1244): exact accuracy 50.3% vs 33%
  chance, spin p=0.092; adjacent (±1) 82.8%, spin p=0.110.
- **Network rank-order** (9 networks): Spearman ρ(predicted, observed network
  gradient) = +0.73, spin p=0.098.

So the discrete content of the gradient is **more** cross-species-specific than the
raw smooth correlation suggested (p moves from 0.16 toward ~0.09–0.10), but it still
does **not** clear significance, and the network test is underpowered (n=9). HOMER
carries some discrete gradient-ordering signal, and that signal is not significant.
Log: `outputs/logs/margulies_discrete_reframe.json`.
