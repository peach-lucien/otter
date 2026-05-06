# D — Subject-level cross-validation

**Result: small but real generalisation gap (~4 pp).** Training on 80% of
subjects and testing on the held-out 20% loses about 4 percentage points of
FC translation Pearson r. `fc_only` and `fc_plus_SC` are statistically
indistinguishable on this metric.

## What was tested

Independent of held-out anchor CV (which tests SPATIAL generalisation). This
tests SUBJECT generalisation: does π trained on 80% of subjects predict the
held-out 20%'s mean FC?

For each of K=5 random folds:
1. Random 80/20 subject split per species (seeded)
2. Stream train mean FC; derive test mean by subtraction from total
3. Re-build C cost matrices on the train FC
4. Solve FGW with full anchor supervision
5. FC-translation Pearson r on (train, test) FC

## Results (mean ± std across 5 folds)

| config       | train r           | test r            | gap             | test within-net | test cross-net |
|--------------|-------------------|-------------------|-----------------|-----------------|----------------|
| `fc_only`    | 0.360 ± 0.002     | 0.319 ± 0.006     | −0.041 ± 0.007  | 0.420 ± 0.011   | 0.166 ± 0.007  |
| `fc_plus_SC` | 0.357 ± 0.002     | 0.318 ± 0.006     | −0.039 ± 0.008  | 0.417 ± 0.011   | 0.166 ± 0.008  |

## Interpretation

- Within-network drops more (0.45 → 0.42) than cross-network (0.20 → 0.17,
  already near floor).
- `fc_only` and `fc_plus_SC` indistinguishable, confirming E1's all-subjects
  finding. SC's win in held-out anchor CV does NOT translate to better
  subject-level generalisation.
- Test-set width across folds is ~0.006 — robust to which specific subjects
  are seen. Good news for downstream use: a similar-sized new cohort would
  give r in the 0.31–0.36 range.

π is mostly limited by *anchor structure*, not by *subject-level noise*.
Subject-CV adds ~4 pp of generalisation overhead on top of the anchor-driven
limits.

## How to re-run

```bash
# Run K folds incrementally — each fold ~26 s
PYTHONPATH=src python experiments/D_subject_cv/subject_cv.py --config fc_plus_SC --k-folds 5
PYTHONPATH=src python experiments/D_subject_cv/subject_cv.py --config fc_only    --k-folds 5
```

Results saved to `outputs/logs/subject_cv.json` and aggregated into
`outputs/logs/fc_translation.json` under the `subject_cv` key.

The clean library API also exposes this:
```python
from homer.eval import subject_kfold_cv
from homer.models import MultimodalFGW
res = subject_kfold_cv(lambda: MultimodalFGW(use_sc=True),
                        cache_dir="outputs/anndata", k_folds=5)
```
