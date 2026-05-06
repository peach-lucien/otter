# A — Anchor-relationship features as cross-species M term

**Result: clean negative.** Adding `M_anchor` (cosine distance over the
n-anchor FC fingerprint of every node) to the cross-species cost made the
production CV *worse*: top-1 dropped from 81% (`fc_plus_SC`) to 69%.

## What was tried

For each node, build a feature vector containing its FC value to each of the
visible anchors. Since anchors are in known 1-to-1 cross-species correspondence,
those vectors are directly comparable. Cosine distance between them gives an
(n_m, n_h) cross-species cost matrix to add into M.

The intuition: V1 and V2 might have similar global FC profiles (hard for the
GW relational term), but their FC profiles to specific anchors (LGN/thalamus,
higher visual areas) should differ.

## How it was tested

This experiment lives entirely in two `pipeline/05a_anchor_cv.py` configs:

```python
"fc_plus_M_anchor":         dict(M={"xyz": 0.5, "anchor_feat": 0.5}),
"fc_plus_SC_plus_M_anchor": dict(relational={"FC": 0.7, "SC": 0.3},
                                  M={"xyz": 0.5, "anchor_feat": 0.5}),
```

Run with:
```bash
PYTHONPATH=src python pipeline/05a_anchor_cv.py --configs fc_plus_M_anchor,fc_plus_SC_plus_M_anchor
```

## Important caveat

The first version of this experiment leaked: `M_anchor` was pre-computed in
`build_multimodal_costs.py` using ALL 42 anchors, including the held-out ones.
Top-1 jumped to a suspicious 100% across all networks. The fix (in
`MultimodalFGW`): recompute `M_anchor` per-fold using only `visible_pair_ids`.

After the leak fix, both `fc_plus_M_anchor` and `fc_plus_SC_plus_M_anchor`
land at top-1 = 69%, vs baseline 79% / production 81%. Subcortical drops
100% → 60%, visual drops 50% → 25%. Helps salience (25% → 50%) but the
regressions outweigh the gains.

## Interpretation

The 32 visible anchors' FC patterns aren't enough to predict held-out anchors
better than xyz alone. Worth keeping the helper available
(`homer.costs.cross_species_anchor_M`) as a diagnostic — if a config relies
heavily on M_anchor and improves, it's a sign the held-out anchors are too
easy / leaked.

See [`docs/results.md`](../../docs/results.md) for the full comparison table.
