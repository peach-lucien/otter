# Experiments

Each subfolder is one research experiment from the project's iteration history.
Items in this directory are **not** part of the production pipeline — they're
documented and kept for reproducibility. Most returned negative or no-op results
(see [`docs/results.md`](../docs/results.md) for the consolidated comparison table).

## Layout

| Folder                          | What                                                    | Result      |
|---------------------------------|---------------------------------------------------------|-------------|
| `A_anchor_M_cost/`              | Anchor-relationship features as cross-species M term    | negative    |
| `B_iterative_coclustering/`     | Bootstrap high-confidence pairs back as soft anchors    | no-op       |
| `C_confidence_weighted_fc/`     | Bayesian shrinkage of FC by per-cell `n_obs`            | structural no-op |
| `D_subject_cv/`                 | K-fold subject-level CV on FC translation               | small gap (~4pp) |
| `M1_multistart/`                | Multistart entropic FGW (5 random + uniform inits)      | no-op (loss spread <1e-6) |
| `M4_hierarchical/`              | Per-network sub-FGW solves                              | trade-off (CV worse, within-net FC better) |
| `archive/`                      | Stepping-stone scripts superseded by current pipeline   | obsolete    |

## How to re-run

Each script that produced a substantive result is preserved here. Re-running:

```bash
PYTHONPATH=src python experiments/B_iterative_coclustering/iterative_cv.py --config fc_plus_SC --networks visual,brainstem
PYTHONPATH=src python experiments/D_subject_cv/subject_cv.py --config fc_plus_SC --k-folds 5
```

For items A and M1 (no dedicated script), see the per-folder README — they
were configuration flags of `pipeline/05a_anchor_cv.py`.
