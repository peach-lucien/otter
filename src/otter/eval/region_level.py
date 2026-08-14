"""Region-level evaluation of the cross-species coupling π.

Parcel-level top-K asks whether the correct human parcel is in the top-K of
π[m, :]. That question suits a user who wants a specific parcel prediction,
but it is strict for a soft probabilistic mapping in which the model spreads
mass across a region of human parcels rather than concentrating it on one
cell.

Region-level top-K asks instead:

    Given a mouse region M (set of parcels), which *human region* does the
    model predict, out of a candidate set of named human regions?

Aggregation
-----------
For mouse region ``M`` with parcel indices ``M_idx`` (size ``k``):

  pi_M     = sum_{m in M_idx} π[m, :]    # (n_h,)  total mass mapped from M
  pi_M    /= pi_M.sum()                  # normalize → distribution over human

(Sum and mean agree up to a constant; the sum is normalized at the end so the
result is a probability distribution.)

Scoring
-------
For each candidate human region ``H_i`` with mask over the 2094 human parcels:

  score(H_i) = sum(pi_M[H_i])                       # mass on H_i
  fold(H_i)  = score(H_i) / (|H_i| / n_h)           # vs uniform expectation
  rank(H_i)  = 1 + #{ H_j : score(H_j) > score(H_i) }

Top-K hit if ``rank(H_true) <= K``.

Nulls
-----
- ``column_permute`` shuffles the column order of ``pi_M`` (destroys spatial
  structure; preserves total mass). Tests whether the observed mass on the
  target region is above what a same-magnitude random distribution would give.

- ``source_permute`` scores ``H_true`` against ``pi_M`` aggregated from a
  *different* mouse region. Tests whether the mass on ``H_true`` is specific
  to ``M``, vs being a generic model bias toward H_true regardless of input.

Notes
-----
- Candidate regions may overlap (e.g. hippocampal subfields). The fold and
  top-K metrics are well-defined under overlap, though not independent across
  candidate columns.
- The candidate set defines chance level. With Beauchamp-22 it is roughly
  1/22 = 4.5%; with JuBrain-184 it is lower. Both are reported.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Core primitives


def aggregate_pi_over_mouse_region(
    pi: np.ndarray, mouse_mask: np.ndarray
) -> np.ndarray:
    """Aggregate π[mouse_mask, :] to a single distribution over human parcels.

    Parameters
    ----------
    pi : (n_m, n_h) ndarray
        Coupling matrix.
    mouse_mask : (n_m,) bool ndarray, or int ndarray of indices
        Mouse region membership.

    Returns
    -------
    pi_M : (n_h,) ndarray
        Probability distribution over human parcels (sums to 1).

    Notes
    -----
    Uses sum-then-normalize rather than mean-then-normalize; they're
    equivalent up to a scalar so the normalized result is identical.
    """
    pi_sub = pi[mouse_mask]                  # (k, n_h)
    if pi_sub.shape[0] == 0:
        raise ValueError("mouse_mask selects 0 parcels")
    pi_M = pi_sub.sum(axis=0)
    total = pi_M.sum()
    if total <= 0:
        raise ValueError("pi_M total mass is non-positive, degenerate region")
    return pi_M / total


def score_candidate_human_regions(
    pi_M: np.ndarray, candidate_masks: dict[str, np.ndarray]
) -> dict[str, float]:
    """For each candidate human region, return its total probability mass under pi_M."""
    return {name: float(pi_M[mask].sum()) for name, mask in candidate_masks.items()}


def fold_enrichment_candidate_regions(
    pi_M: np.ndarray, candidate_masks: dict[str, np.ndarray]
) -> dict[str, float]:
    """Fold enrichment of mass on each candidate region vs uniform expectation."""
    n_h = pi_M.shape[0]
    out = {}
    for name, mask in candidate_masks.items():
        size = int(mask.sum())
        if size == 0:
            out[name] = float("nan")
            continue
        observed = float(pi_M[mask].sum())
        expected = size / n_h
        out[name] = observed / expected if expected > 0 else float("nan")
    return out


def rank_candidate_regions(
    scores: dict[str, float], true_region: str
) -> int:
    """1-indexed rank of true_region by descending score; ties go to true_region."""
    if true_region not in scores:
        raise KeyError(f"true_region {true_region!r} not in candidate set")
    s_true = scores[true_region]
    # "<" not "<=", so ties favour the true region: two tied regions place the
    # truth at the better rank.
    n_strictly_better = sum(1 for n, s in scores.items() if n != true_region and s > s_true)
    return n_strictly_better + 1


# ---------------------------------------------------------------------------
# Per-pair evaluation


@dataclass
class RegionLevelPairResult:
    """Per-pair region-level metrics."""

    pair: str                 # "mouse_name -> human_name"
    n_mouse_parcels: int
    n_human_parcels: int
    rank: int                 # 1..|candidate set|
    score_true: float         # observed mass on true H
    fold_enrichment: float    # score_true / (|H_true|/n_h)
    top_k_hits: dict[int, bool]
    qualified_top_k_hits: dict[int, bool] = None    # rank <= k AND fold >= 1
    total_mass_in_candidates: float = 0.0
    is_anchor_overlapping: Optional[bool] = None


def region_topk(
    pi: np.ndarray,
    mouse_mask: np.ndarray,
    candidate_masks: dict[str, np.ndarray],
    true_region: str,
    *,
    k_list: Sequence[int] = (1, 3, 5),
    pair_label: Optional[str] = None,
    is_anchor_overlapping: Optional[bool] = None,
    fold_threshold: float = 1.0,
) -> RegionLevelPairResult:
    """Region-level top-K for one mouse region.

    Two flavours of top-K hit are reported:

    - ``top_k_hits[k]``: rank-only hit (``rank(H_true) <= k``).
    - ``qualified_top_k_hits[k]``: requires *both* ``rank <= k`` *and*
      ``fold_enrichment >= fold_threshold``. This filters out "vacant"
      wins in which the model put near-zero mass on every candidate and the
      true region leads only on noise.

    Also reports ``total_mass_in_candidates``: how much of the model's
    aggregated π over the mouse region lands inside *any* candidate
    region. If this is small (say < 0.2), the candidate set is too sparse
    for region-level evaluation, because the model is putting mass on
    parcels outside the candidate vocabulary.
    """
    pi_M = aggregate_pi_over_mouse_region(pi, mouse_mask)
    scores = score_candidate_human_regions(pi_M, candidate_masks)
    fold = fold_enrichment_candidate_regions(pi_M, candidate_masks)
    rank = rank_candidate_regions(scores, true_region)
    fold_true = fold[true_region]
    qualified = fold_true >= fold_threshold
    union_mask = np.zeros_like(next(iter(candidate_masks.values())))
    for m in candidate_masks.values():
        union_mask = union_mask | m
    total_mass_in_candidates = float(pi_M[union_mask].sum())

    return RegionLevelPairResult(
        pair=pair_label or f"-> {true_region}",
        n_mouse_parcels=int(np.asarray(mouse_mask).sum() if mouse_mask.dtype == bool
                            else len(mouse_mask)),
        n_human_parcels=int(candidate_masks[true_region].sum()),
        rank=rank,
        score_true=scores[true_region],
        fold_enrichment=fold_true,
        top_k_hits={int(k): rank <= int(k) for k in k_list},
        qualified_top_k_hits={int(k): (rank <= int(k)) and qualified for k in k_list},
        total_mass_in_candidates=total_mass_in_candidates,
        is_anchor_overlapping=is_anchor_overlapping,
    )


# ---------------------------------------------------------------------------
# Pipeline-level evaluation


def evaluate_region_level(
    pi: np.ndarray,
    pairs: Sequence[tuple[str, str]],
    mouse_masks: dict[str, np.ndarray],
    candidate_masks: dict[str, np.ndarray],
    *,
    k_list: Sequence[int] = (1, 3, 5),
    anchor_overlap: Optional[dict[str, bool]] = None,
) -> dict:
    """Run region-level evaluation across all pairs.

    Parameters
    ----------
    pi : (n_m, n_h) ndarray
    pairs : list of (mouse_name, human_name), the canonical homologue pairs
    mouse_masks : {mouse_name: (n_m,) bool ndarray}
    candidate_masks : {candidate_human_name: (n_h,) bool ndarray}
        Must include every ``human_name`` in ``pairs`` plus any other
        candidates to rank against. Larger candidate sets make the task
        harder (chance top-1 ≈ 1/|candidate set|).
    k_list : K values to report (top-1, top-3, top-5, ...).
    anchor_overlap : optional {mouse_name: bool}, whether this pair overlaps
        with the anchor supervision (for the anchor-vs-novel breakdown).

    Returns
    -------
    {
      "per_pair": [RegionLevelPairResult dicts],
      "aggregate": {
        "top_k": {1: ..., 3: ..., 5: ...},
        "mean_rank": float,
        "median_rank": float,
        "mean_fold_enrichment": float,
        "median_fold_enrichment": float,
      },
      "anchor_vs_novel": optional breakdown,
      "n_candidates": int,
      "n_pairs_evaluated": int,
    }
    """
    per_pair: list[RegionLevelPairResult] = []
    skipped: list[str] = []
    for m_name, h_name in pairs:
        if m_name not in mouse_masks:
            skipped.append(f"{m_name}: mouse mask missing")
            continue
        if h_name not in candidate_masks:
            skipped.append(f"{h_name}: not in candidate set")
            continue
        m_mask = mouse_masks[m_name]
        if m_mask.sum() == 0:
            skipped.append(f"{m_name}: empty mouse mask")
            continue
        if candidate_masks[h_name].sum() == 0:
            skipped.append(f"{h_name}: empty human mask")
            continue
        is_anc = (anchor_overlap or {}).get(m_name)
        res = region_topk(
            pi, m_mask, candidate_masks, h_name,
            k_list=k_list,
            pair_label=f"{m_name} -> {h_name}",
            is_anchor_overlapping=is_anc,
        )
        per_pair.append(res)

    if not per_pair:
        return {
            "per_pair": [], "aggregate": {}, "n_candidates": len(candidate_masks),
            "n_pairs_evaluated": 0, "skipped": skipped,
        }

    # Weight aggregate metrics by n_mouse_parcels (a bigger region carries more weight).
    weights = np.array([p.n_mouse_parcels for p in per_pair], dtype=float)
    wsum = weights.sum()
    ranks = np.array([p.rank for p in per_pair], dtype=float)
    folds = np.array([p.fold_enrichment for p in per_pair], dtype=float)

    coverage = np.array([p.total_mass_in_candidates for p in per_pair], dtype=float)
    agg = {
        "top_k": {
            int(k): float(
                sum(weights[i] * float(per_pair[i].top_k_hits[int(k)])
                    for i in range(len(per_pair))) / wsum
            )
            for k in k_list
        },
        "qualified_top_k": {
            int(k): float(
                sum(weights[i] * float(per_pair[i].qualified_top_k_hits[int(k)])
                    for i in range(len(per_pair))) / wsum
            )
            for k in k_list
        },
        "mean_rank": float((ranks * weights).sum() / wsum),
        "median_rank": float(np.median(ranks)),
        "mean_fold_enrichment": float((folds * weights).sum() / wsum),
        "median_fold_enrichment": float(np.median(folds)),
        "mean_total_mass_in_candidates": float((coverage * weights).sum() / wsum),
    }

    out = {
        "per_pair": [_pair_to_dict(p) for p in per_pair],
        "aggregate": agg,
        "n_candidates": len(candidate_masks),
        "n_pairs_evaluated": len(per_pair),
        "skipped": skipped,
    }

    # Anchor-vs-novel breakdown
    if anchor_overlap is not None and any(p.is_anchor_overlapping is not None for p in per_pair):
        for label, subset in [
            ("anchor_overlapping",
             [p for p in per_pair if p.is_anchor_overlapping is True]),
            ("novel",
             [p for p in per_pair if p.is_anchor_overlapping is False]),
        ]:
            if not subset:
                continue
            w = np.array([p.n_mouse_parcels for p in subset], dtype=float)
            ws = w.sum()
            r = np.array([p.rank for p in subset], dtype=float)
            f = np.array([p.fold_enrichment for p in subset], dtype=float)
            out[label] = {
                "n_pairs": len(subset),
                "n_parcels": int(ws),
                "top_k": {
                    int(k): float(
                        sum(w[i] * float(subset[i].top_k_hits[int(k)])
                            for i in range(len(subset))) / ws
                    )
                    for k in k_list
                },
                "qualified_top_k": {
                    int(k): float(
                        sum(w[i] * float(subset[i].qualified_top_k_hits[int(k)])
                            for i in range(len(subset))) / ws
                    )
                    for k in k_list
                },
                "mean_rank": float((r * w).sum() / ws),
                "mean_fold_enrichment": float((f * w).sum() / ws),
            }
    return out


def _pair_to_dict(p: RegionLevelPairResult) -> dict:
    return {
        "pair": p.pair,
        "n_mouse_parcels": p.n_mouse_parcels,
        "n_human_parcels": p.n_human_parcels,
        "rank": p.rank,
        "score_true": p.score_true,
        "fold_enrichment": p.fold_enrichment,
        "top_k_hits": {str(k): bool(v) for k, v in p.top_k_hits.items()},
        "qualified_top_k_hits": {str(k): bool(v) for k, v in (p.qualified_top_k_hits or {}).items()},
        "total_mass_in_candidates": float(p.total_mass_in_candidates),
        "is_anchor_overlapping": p.is_anchor_overlapping,
    }


# ---------------------------------------------------------------------------
# Nulls


def column_permuted_null(
    pi: np.ndarray,
    pairs: Sequence[tuple[str, str]],
    mouse_masks: dict[str, np.ndarray],
    candidate_masks: dict[str, np.ndarray],
    *,
    k_list: Sequence[int] = (1, 3, 5),
    n_trials: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Null distribution from column-permuted pi_M.

    For each pair, shuffle the column order of ``pi_M`` n_trials times,
    re-score the candidate regions, re-rank. Report null mean / std of
    top-K hit rate and fold enrichment.

    This null preserves total mass but destroys *where* mass falls, so it
    gives the chance of the model's mass-magnitude landing in a region of
    that size at random.
    """
    rng = rng or np.random.default_rng(0)
    n_h = pi.shape[1]

    null_topk = {int(k): [] for k in k_list}
    null_fold = []
    for trial in range(n_trials):
        perm = rng.permutation(n_h)
        per_pair_hits = {int(k): [] for k in k_list}
        per_pair_folds = []
        per_pair_weights = []
        for m_name, h_name in pairs:
            if m_name not in mouse_masks or h_name not in candidate_masks:
                continue
            m_mask = mouse_masks[m_name]
            if m_mask.sum() == 0:
                continue
            pi_M = aggregate_pi_over_mouse_region(pi, m_mask)
            pi_M_shuf = pi_M[perm]
            scores = score_candidate_human_regions(pi_M_shuf, candidate_masks)
            rank = rank_candidate_regions(scores, h_name)
            size = int(candidate_masks[h_name].sum())
            obs = scores[h_name]
            exp = size / n_h
            fold = obs / exp if exp > 0 else float("nan")
            per_pair_folds.append(fold)
            per_pair_weights.append(int(m_mask.sum()))
            for k in k_list:
                per_pair_hits[int(k)].append(rank <= int(k))
        if not per_pair_folds:
            continue
        w = np.array(per_pair_weights, dtype=float)
        wsum = w.sum()
        null_fold.append(float((np.array(per_pair_folds) * w).sum() / wsum))
        for k in k_list:
            v = np.array(per_pair_hits[int(k)], dtype=float)
            null_topk[int(k)].append(float((v * w).sum() / wsum))

    return {
        "n_trials": n_trials,
        "null_topk_mean": {k: float(np.mean(v)) for k, v in null_topk.items()},
        "null_topk_std": {k: float(np.std(v)) for k, v in null_topk.items()},
        "null_fold_mean": float(np.mean(null_fold)) if null_fold else float("nan"),
        "null_fold_std":  float(np.std(null_fold))  if null_fold else float("nan"),
    }


def source_permuted_null(
    pi: np.ndarray,
    pairs: Sequence[tuple[str, str]],
    mouse_masks: dict[str, np.ndarray],
    candidate_masks: dict[str, np.ndarray],
    *,
    k_list: Sequence[int] = (1, 3, 5),
    n_trials: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Null distribution from mouse-region permuted pairings.

    For each trial, shuffle the (mouse_name -> human_name) mapping. Tests
    whether the mass-on-H_true is specific to the matching M_true, vs being
    a generic model bias toward H_true that any mouse region would also
    produce. It is the stronger of the two nulls, requiring the model to be
    selective rather than merely non-uniform.
    """
    rng = rng or np.random.default_rng(0)

    # Pre-aggregate pi_M for each mouse region so we don't repeat work
    pi_M_by_mouse = {}
    for m_name, _ in pairs:
        if m_name in mouse_masks and mouse_masks[m_name].sum() > 0:
            pi_M_by_mouse[m_name] = aggregate_pi_over_mouse_region(pi, mouse_masks[m_name])

    valid_pairs = [
        (m, h) for m, h in pairs
        if m in pi_M_by_mouse and h in candidate_masks
    ]
    if len(valid_pairs) < 2:
        return {"n_trials": 0, "null_topk_mean": {}, "null_topk_std": {},
                "null_fold_mean": float("nan"), "null_fold_std": float("nan")}

    null_topk = {int(k): [] for k in k_list}
    null_fold = []
    m_names = [m for m, _ in valid_pairs]
    h_names = [h for _, h in valid_pairs]

    for trial in range(n_trials):
        perm = rng.permutation(len(valid_pairs))
        per_pair_hits = {int(k): [] for k in k_list}
        per_pair_folds = []
        per_pair_weights = []
        for i, (m_true, h_true) in enumerate(valid_pairs):
            # Use pi_M from a *permuted* mouse region but score against the
            # original h_true. If perm[i] == i it's the real pairing, accept
            # it (this is bounded by the # fixed points which is small for
            # large permutations).
            m_other = m_names[perm[i]]
            pi_M_other = pi_M_by_mouse[m_other]
            scores = score_candidate_human_regions(pi_M_other, candidate_masks)
            rank = rank_candidate_regions(scores, h_true)
            size = int(candidate_masks[h_true].sum())
            obs = scores[h_true]
            exp = size / pi.shape[1]
            fold = obs / exp if exp > 0 else float("nan")
            per_pair_folds.append(fold)
            per_pair_weights.append(int(mouse_masks[m_true].sum()))
            for k in k_list:
                per_pair_hits[int(k)].append(rank <= int(k))
        w = np.array(per_pair_weights, dtype=float)
        wsum = w.sum()
        null_fold.append(float((np.array(per_pair_folds) * w).sum() / wsum))
        for k in k_list:
            v = np.array(per_pair_hits[int(k)], dtype=float)
            null_topk[int(k)].append(float((v * w).sum() / wsum))

    return {
        "n_trials": n_trials,
        "null_topk_mean": {k: float(np.mean(v)) for k, v in null_topk.items()},
        "null_topk_std": {k: float(np.std(v)) for k, v in null_topk.items()},
        "null_fold_mean": float(np.mean(null_fold)) if null_fold else float("nan"),
        "null_fold_std":  float(np.std(null_fold))  if null_fold else float("nan"),
    }
