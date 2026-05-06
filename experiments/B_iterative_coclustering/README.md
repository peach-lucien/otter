# B — Iterative anchor expansion / soft co-clustering

**Result: clean no-op.** Adding the model's own high-confidence pairings back
as soft (or hard) anchors and re-solving leaves π byte-for-byte identical
across iterations.

## What was tried

EM-style refinement:
1. Solve FGW with the visible anchors (lam=1.0 hard supervision)
2. Pick the top-K non-anchor mouse rows by row-max concentration
3. Re-solve with those rows added as soft (lam=0.30) or hard (lam=1.0) anchors
4. Iterate 2-3 times

## Findings

Per-network top-1 with K=200, conf≥0.95, lam=1.0 over 2 iterations is
**identical (to 16 decimals)** to the single-shot `fc_plus_SC` baseline
across ALL 11 networks. Confirmed both at production ε=5e-3 and at the
softer ε=5e-2 sanity check.

## Why it's a no-op

With production ε=5e-3 + anchor supervision, the first-pass π already has
mean row-max concentration **0.977** — every "high confidence" row is one-hot
at exactly the human node it would settle on again. Adding lam_soft to the
OTHER columns of those rows changes M but not π (the solver was already
going to its assigned column anyway). Verified analytically:
|π_iter1 − π_iter0|_max = 1e-5 even with 200 rows modified in M.

## Honest interpretation

Held-out anchor recovery is bottlenecked by information available *to the
held-out row's GW + xyz signal*, not by lack of self-confidence about the
rest of the map. Iterative co-clustering would only help in a regime where
the initial solution is genuinely ambiguous — which isn't ours after the
M_xyz term + anchor supervision.

## How to re-run

```bash
PYTHONPATH=src python experiments/B_iterative_coclustering/iterative_cv.py \
    --config fc_plus_SC --n-iter 2 --top-k 200 --conf-thresh 0.95 --lam-soft 1.0
```

Results saved to `outputs/logs/iterative_cv.json`.
