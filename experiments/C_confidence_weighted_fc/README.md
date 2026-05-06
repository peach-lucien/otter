# C — Confidence-weighted FC via `fc_n_obs`

**Result: structural no-op.** The colleague's preprocessing produced
near-uniform FC coverage, so weighting by per-cell observation count
barely moves the resulting cost matrix. CV experiment was unnecessary.

## What was planned

Bayesian-flavored shrinkage: pull each FC cell `r[i,j]` toward 0 in proportion
to its coverage deficit (`n_obs / n_max`), then re-derive the FC cost matrix C
and run CV. The intuition: cells with fewer subject observations are noisier
estimates and shouldn't drive the GW relational term as strongly.

## Findings

Diagnosis showed the experiment was a literal no-op for mouse and a near-no-op
for human:

| Species | n_obs range | mean | row_cov min | anchor row_cov | corr(C_orig, C_shrunk) |
|---------|-------------|------|-------------|----------------|------------------------|
| mouse   | 105 (uniform) | 105.0 | 1.000     | 1.000          | 1.0 (literal no-op)    |
| human   | 100 – 113   | 112.6 | 0.928       | 0.998          | 0.999966               |

Mouse is uniform → mathematically zero change. Human: only 0.9% of nodes have
<95% row coverage; all 42 anchors have 99.8%. Cost-matrix correlation is
r=0.99997 with mean abs diff of 0.00018 on a [0, 2] scale.

## Decision

Park as future work IF upstream preprocessing changes (e.g. coverage-imbalanced
FC matrices from a new cohort). For the current data, no signal to extract.

## How to re-verify

```bash
PYTHONPATH=src python experiments/C_confidence_weighted_fc/confidence_weighted_fc_check.py
```

Results saved to `outputs/logs/confidence_weighted_fc_check.json`.
