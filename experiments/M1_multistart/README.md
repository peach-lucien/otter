# M1 — Multistart entropic FGW

**Result: meaningful no-op.** All 6 diverse inits (uniform + 4 random
Sinkhorn-projected + 1 anchor-warm) converge to within ~5e-7 relative loss
of each other. Identical metrics across restarts. The 6× compute cost
buys no measurable gain.

## What was tried

`homer.models._solver.entropic_semirelaxed_fgw_multistart` runs:

1. one default uniform-init solve
2. `n_random_inits` Sinkhorn-projected random G0 inits
3. (optionally) one anchor-warmstart run

Picks the lowest-loss solution. Tested on visual + brainstem (the two
hardest CV folds).

## Findings

- **Loss spread across 6 inits: < 1e-6** (relative).
- **Identical** top-1, top-5, mean_rank, mean_xyz_dist across all restarts.

This is a *substantive* finding about the methodology, not a failure: anchor
supervision + xyz spatial feature in M makes the FGW objective globally
well-identified in practice. Single-shot solutions are trustworthy. (Compare
unsupervised GW from earlier in the project where restarts found genuinely
different optima.)

## Decision

Keep single-shot FGW for production. The multistart helper remains available
for future diagnostics or for harder regimes (e.g., if we drop anchor
supervision).

For the methods writeup: *"solution stability verified via multistart
(loss spread < 1e-6 across 6 diverse initialisations)."*

## How to re-verify

```bash
PYTHONPATH=src python pipeline/05a_anchor_cv.py \
    --configs baseline_fc_only --networks visual,brainstem \
    --n-restarts 5 --cache-suffix _ms
```

Results stored in `outputs/logs/multimodal_cv.json` under the
`baseline_fc_only_ms` key.

The clean library API also supports this:
```python
from homer.models import SupervisedFGW
m = SupervisedFGW(use_multistart=True, n_restarts=4)
m.fit(M, H)
print(m.fit_info_.extra)   # {'best_init': ..., 'loss_spread': ..., ...}
```
