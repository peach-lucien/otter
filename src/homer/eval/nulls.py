"""Null distributions for held-out anchor CV.

Two principled nulls:
  - random_pi_null      — sample uniform random π satisfying mouse marginal
  - permuted_anchor_null — shuffle anchor pair_ids before solving FGW

These give the reference distributions for z-scores on the real top-1.
Headline numbers from the comparison table:
    real top-1 81% vs random_pi 28%±7% (z=+7.5)
    real top-1 81% vs permuted_anchor 31%±3% (z=+17.8)
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from homer.data.anchors import (
    AnchorIndex, get_anchor_index, held_out_metrics_graded,
)


def random_pi_null(
    mouse_ad,
    human_ad,
    *,
    held_out_pair_ids: Sequence[int],
    n_trials: int = 50,
    seed: int = 0,
) -> dict:
    """Sample n_trials random π satisfying the mouse uniform marginal,
    evaluate held-out anchor metrics. Returns mean ± std + per-trial list.
    """
    idx_m = get_anchor_index(mouse_ad.var)
    idx_h = get_anchor_index(human_ad.var)
    n_m = mouse_ad.uns["n_nodes"]; n_h = human_ad.uns["n_nodes"]
    p = np.full(n_m, 1.0 / n_m, dtype=np.float64)

    trials = []
    for t in range(n_trials):
        rng = np.random.default_rng(seed + t)
        A = rng.uniform(0.5, 1.5, size=(n_m, n_h))
        A = A * (p / A.sum(axis=1).clip(min=1e-12))[:, None]
        pi_anchor = A[np.ix_(idx_m.pos, idx_h.pos)]
        m = held_out_metrics_graded(
            pi_anchor, idx_m, idx_h, held_out_pair_ids, var_h=human_ad.var,
        )
        trials.append({k: m[k] for k in ("top1", "top5", "pair_id", "hemisphere",
                                          "mean_rank", "mean_xyz_dist") if k in m})
    return _summarise(trials)


def permuted_anchor_null(
    mouse_ad,
    human_ad,
    solve_fn,
    *,
    held_out_pair_ids: Sequence[int],
    n_trials: int = 5,
    seed: int = 0,
) -> dict:
    """Shuffle the anchor cross-species correspondence (pair_ids permuted),
    re-solve FGW, evaluate. Tests whether the *specific* mouse↔human anchor
    pairings drive the result vs "having anchor supervision in general".

    `solve_fn(visible_pair_ids, idx_m_perm, idx_h_perm) -> pi` is supplied by
    the caller (lets you wire in any model/config you want).
    """
    idx_m = get_anchor_index(mouse_ad.var)
    idx_h = get_anchor_index(human_ad.var)

    visible_pair_ids = sorted(int(p) for p in idx_m.pair_ids
                               if int(p) not in set(held_out_pair_ids))

    trials = []
    for t in range(n_trials):
        rng = np.random.default_rng(seed + t)
        perm = rng.permutation(len(idx_m.pos))
        # Permute idx_m's positions to scramble the anchor correspondence
        idx_m_perm = AnchorIndex(
            pos=idx_m.pos[perm],
            pair_ids=idx_m.pair_ids[perm],
            hemispheres=idx_m.hemispheres[perm],
            keys=[idx_m.keys[i] for i in perm],
        )
        pi = solve_fn(visible_pair_ids, idx_m_perm, idx_h)
        pi_anchor = pi[np.ix_(idx_m.pos, idx_h.pos)]
        m = held_out_metrics_graded(
            pi_anchor, idx_m, idx_h, held_out_pair_ids, var_h=human_ad.var,
        )
        trials.append({k: m[k] for k in ("top1", "top5", "pair_id", "hemisphere",
                                          "mean_rank", "mean_xyz_dist") if k in m})
    return _summarise(trials)


def _summarise(trials: list[dict]) -> dict:
    out: dict = {"n_trials": len(trials), "trials": trials}
    if not trials:
        return out
    keys = trials[0].keys()
    for k in keys:
        vals = np.array([t[k] for t in trials], dtype=float)
        out[f"{k}_mean"] = float(np.nanmean(vals))
        out[f"{k}_std"]  = float(np.nanstd(vals))
    return out
