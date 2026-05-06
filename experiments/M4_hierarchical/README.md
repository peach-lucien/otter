# M4 — Hierarchical / per-network FGW

**Result: clean trade-off, not a strict improvement.** Hierarchical solves
each functional network as an isolated sub-problem (60–410 nodes each),
producing a block-sparse coupling. Wins on within-network FC translation,
loses on leave-one-network-out CV (because held-out networks have zero
visible anchors in their sub-block).

## What was tried

Per-network within-species semirelaxed FGW. Each of 11 networks solved
independently, then assembled into a (1864, 2094) coupling.

## Results

- **Leave-one-network-out CV: HURT** (top-1 79% → 45%) — the held-out
  network has zero visible anchors in its sub-block, so no supervision.
- **Production FC translation: HELPED** (overall r 0.36 → 0.40, within-net
  0.45 → 0.55) — each network's sub-FGW gets focused optimization.
- **Cross-network FC: HURT** (r 0.20 → 0.16) — block-diagonal by construction.
- **Coverage: HALVED** (1450 → 787 human nodes kept) — same reason.

## Decision

Hierarchical is a **complementary** tool, not a strict improvement. Use when:
- full anchor supervision is available, AND
- you care more about within-network FC fidelity than cross-network coverage.

The flat solver remains the production choice for general use.

## How to re-run

Functional API:
```bash
PYTHONPATH=src python experiments/M4_hierarchical/hierarchical_cv.py --mode cv
PYTHONPATH=src python experiments/M4_hierarchical/hierarchical_cv.py --mode production
```

Results saved to `outputs/logs/hierarchical_cv.json` and
`outputs/coupling/pi_hierarchical.npy`. The `hierarchical_fc_only` entry in
`outputs/logs/fc_translation.json` carries the FC-translation comparison.

Clean class API:
```python
from homer.models import HierarchicalFGW
m = HierarchicalFGW(epsilon=5e-3, xyz_weight=0.5)
m.fit(M, H)            # full supervision, flat use
m.fit(M, H, holdout_pair_ids=[5, 6])   # leave-one-network-out (visual)
```
