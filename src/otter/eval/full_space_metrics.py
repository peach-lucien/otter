"""Full-space recovery metrics for held-out anchors.

The companion :func:`otter.data.anchors.held_out_metrics_graded` restricts the
``argmax`` search to the held-out human anchor columns only, so its top-1
answers the question "among the held-out anchor candidates, did the model rank
the correct one first?". This module answers a strictly harder question:
"among ALL n_h human nodes (anchors + ~2000 grid nodes), where did the model
send the held-out mouse anchor?".

Both versions are valid and answer different questions. The restricted version
is what we report as "anchor-candidate ranking accuracy" (~81% on production);
the full-space version typically lands at 0–5% top-1 because the model
naturally lands on a grid node *near* the correct anchor rather than the
anchor itself.

Public:
    full_space_metrics(pi, idx_m, idx_h, held_out_pair_ids, *, var_h=None,
                        top_k=5, neighborhood_xyz_dist=0.05) -> dict
    full_space_metrics_per_anchor(...) -> pd.DataFrame
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from otter.data.anchors import AnchorIndex, held_out_indices


def full_space_metrics(
    pi: np.ndarray,
    idx_m: AnchorIndex,
    idx_h: AnchorIndex,
    held_out_pair_ids: Sequence[int],
    *,
    var_h=None,
    top_k: int = 5,
    neighborhood_xyz_dist: float = 0.05,
) -> dict:
    """Recovery metrics with argmax taken over **all human nodes**, not just
    held-out anchor candidates.

    Parameters
    ----------
    pi : (n_m, n_h) ndarray
        The full coupling. Rows are mouse positional indices, cols human.
    idx_m, idx_h : AnchorIndex
    held_out_pair_ids : list of pair_id ints
        Anchors withheld from supervision (the LONO fold's held set).
    var_h : pandas DataFrame, optional
        Human var (with x/y/z + garin_anchor cols). Required for
        mean_xyz_dist_full + neighborhood_hit metrics.
    top_k : int
        K for the top-K hit rate.
    neighborhood_xyz_dist : float
        Per-species-normalised xyz distance threshold defining "the argmax
        landed close to the correct anchor". Default 0.05 ≈ 5% of brain extent.

    Returns
    -------
    dict with:
        n                                  number of held-out mouse anchors
        full_top1                          frac whose full-space argmax IS the correct anchor
        full_topK                          frac whose correct anchor is in the full-space top-K
        mean_rank_full                     mean rank of correct anchor in full π row
        median_rank_full                   "
        mean_xyz_dist_full                 mean per-species-normalised xyz distance from
                                            argmax to correct anchor (lower = better)
        median_xyz_dist_full
        mean_mass_on_correct_anchor        mean π[mp, hp_correct] across held-out anchors
                                            (the model's actual probability assigned to the
                                            correct partner, regardless of whether it's the argmax)
        frac_argmax_is_anchor              what fraction of full-space argmaxes land on
                                            ANY anchor (not necessarily the correct one)
        frac_in_neighborhood               argmax within neighborhood_xyz_dist of correct anchor
        n_h_total                          for context
    """
    visible_pair_ids = [p for p in idx_m.pair_ids
                        if int(p) not in set(int(x) for x in held_out_pair_ids)]
    m_held_local, h_held_local = held_out_indices(idx_m, idx_h, visible_pair_ids)
    if len(m_held_local) == 0:
        return {"n": 0}

    m_held_pos     = idx_m.pos[m_held_local]              # mouse positions in pi
    h_correct_pos  = idx_h.pos[h_held_local]              # correct human positions
    n_h            = pi.shape[1]
    all_anchor_h   = set(int(p) for p in idx_h.pos.tolist())

    # 1. Full-space ranks
    sub_full = pi[m_held_pos, :]                          # (n_held, n_h)
    order = np.argsort(-sub_full, axis=1)                 # rank 0 = argmax
    full_argmax = order[:, 0]                             # (n_held,)

    # Rank of correct anchor in each row (1-indexed)
    ranks = np.zeros(len(m_held_pos), dtype=np.int64)
    for i, hp in enumerate(h_correct_pos):
        ranks[i] = int(np.where(order[i] == hp)[0][0]) + 1

    # 2. Mass on correct anchor (the actual π weight, not just rank)
    masses = sub_full[np.arange(len(m_held_pos)), h_correct_pos]
    # Normalise by row sum so it's comparable across rows (semirelaxed: row sums = 1/n_m)
    row_sums = sub_full.sum(axis=1).clip(min=1e-12)
    mass_normalised = masses / row_sums                   # in [0, 1]

    # 3. Did the argmax hit ANY anchor (not necessarily the correct one)?
    argmax_is_anchor = np.array([int(p_) in all_anchor_h for p_ in full_argmax.tolist()])

    out = {
        "n":                            int(len(m_held_pos)),
        "n_h_total":                    int(n_h),
        "full_top1":                    float((full_argmax == h_correct_pos).mean()),
        f"full_top{top_k}":             float((ranks <= top_k).mean()),
        "mean_rank_full":               float(ranks.mean()),
        "median_rank_full":             float(np.median(ranks)),
        "max_rank_possible_full":       int(n_h),
        "mean_mass_on_correct_anchor":  float(mass_normalised.mean()),
        "median_mass_on_correct_anchor":float(np.median(mass_normalised)),
        "frac_argmax_is_anchor":        float(argmax_is_anchor.mean()),
    }

    # 4. xyz-distance metrics (require var_h)
    if var_h is not None:
        xyz = var_h[["x", "y", "z"]].values.astype(np.float64)
        lo = xyz.min(0, keepdims=True); hi = xyz.max(0, keepdims=True)
        xyz_n = (xyz - lo) / np.maximum(hi - lo, 1e-9)        # per-species [0, 1]^3
        dist_to_correct = np.linalg.norm(
            xyz_n[full_argmax] - xyz_n[h_correct_pos], axis=1
        )
        out["mean_xyz_dist_full"]   = float(dist_to_correct.mean())
        out["median_xyz_dist_full"] = float(np.median(dist_to_correct))
        out["frac_in_neighborhood"] = float((dist_to_correct < neighborhood_xyz_dist).mean())

    return out


def full_space_metrics_per_anchor(
    pi: np.ndarray,
    idx_m: AnchorIndex,
    idx_h: AnchorIndex,
    held_out_pair_ids: Sequence[int],
    *,
    var_m=None,
    var_h=None,
):
    """Per-held-anchor breakdown, useful for spotting which anchors fail
    catastrophically vs which are merely off by a neighbouring node.

    Returns a pandas DataFrame, one row per held-out mouse anchor.
    """
    import pandas as pd
    visible_pair_ids = [p for p in idx_m.pair_ids
                        if int(p) not in set(int(x) for x in held_out_pair_ids)]
    m_held_local, h_held_local = held_out_indices(idx_m, idx_h, visible_pair_ids)
    if len(m_held_local) == 0:
        return pd.DataFrame()

    m_held_pos    = idx_m.pos[m_held_local]
    h_correct_pos = idx_h.pos[h_held_local]
    n_h           = pi.shape[1]
    all_anchor_h  = set(int(p) for p in idx_h.pos.tolist())

    rows = []
    for i, mp in enumerate(m_held_pos):
        row = pi[mp, :]
        order = np.argsort(-row)
        argmax_pos = int(order[0])
        rank = int(np.where(order == h_correct_pos[i])[0][0]) + 1
        mass = float(row[h_correct_pos[i]] / max(row.sum(), 1e-12))
        info = {
            "pair_id":             int(idx_m.pair_ids[m_held_local[i]]),
            "hemisphere":          str(idx_m.hemispheres[m_held_local[i]]),
            "mouse_pos":           int(mp),
            "correct_h_pos":       int(h_correct_pos[i]),
            "argmax_h_pos":        argmax_pos,
            "argmax_is_correct":   bool(argmax_pos == h_correct_pos[i]),
            "argmax_is_any_anchor":bool(argmax_pos in all_anchor_h),
            "rank_full":           rank,
            "mass_on_correct":     mass,
            "argmax_pi_value":     float(row[argmax_pos] / max(row.sum(), 1e-12)),
        }
        if var_m is not None:
            info["mouse_region"] = str(var_m.iloc[mp]["region"])
        if var_h is not None:
            info["correct_h_region"] = str(var_h.iloc[h_correct_pos[i]]["region"])
            info["argmax_h_region"]  = str(var_h.iloc[argmax_pos]["region"])
            xyz = var_h[["x", "y", "z"]].values.astype(np.float64)
            lo = xyz.min(0, keepdims=True); hi = xyz.max(0, keepdims=True)
            xyz_n = (xyz - lo) / np.maximum(hi - lo, 1e-9)
            info["xyz_dist_argmax_to_correct"] = float(
                np.linalg.norm(xyz_n[argmax_pos] - xyz_n[h_correct_pos[i]])
            )
        rows.append(info)
    return pd.DataFrame(rows)
