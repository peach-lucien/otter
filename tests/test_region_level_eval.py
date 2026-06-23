"""Tests for homer.eval.region_level."""
from __future__ import annotations

import numpy as np
import pytest

from homer.eval.region_level import (
    aggregate_pi_over_mouse_region,
    score_candidate_human_regions,
    fold_enrichment_candidate_regions,
    rank_candidate_regions,
    region_topk,
    evaluate_region_level,
    column_permuted_null,
    source_permuted_null,
)


# ---------------------------------------------------------------------------
# Aggregation


def test_aggregate_normalizes_to_distribution():
    rng = np.random.default_rng(0)
    pi = rng.uniform(0, 1, (5, 10))
    m_mask = np.array([True, True, False, False, True])
    pi_M = aggregate_pi_over_mouse_region(pi, m_mask)
    assert pi_M.shape == (10,)
    assert pi_M.sum() == pytest.approx(1.0)
    assert (pi_M >= 0).all()


def test_aggregate_uses_only_masked_rows():
    pi = np.zeros((4, 6))
    pi[0] = [1, 0, 0, 0, 0, 0]
    pi[1] = [0, 1, 0, 0, 0, 0]
    pi[2] = [9, 9, 9, 9, 9, 9]   # excluded
    pi[3] = [0, 0, 1, 0, 0, 0]
    m_mask = np.array([True, True, False, True])
    pi_M = aggregate_pi_over_mouse_region(pi, m_mask)
    # Each of rows 0, 1, 3 contributed one unit; after normalization each is 1/3
    expected = np.array([1, 1, 1, 0, 0, 0]) / 3
    assert np.allclose(pi_M, expected)


def test_aggregate_empty_mask_raises():
    pi = np.full((3, 4), 0.25)
    with pytest.raises(ValueError, match="0 parcels"):
        aggregate_pi_over_mouse_region(pi, np.array([False, False, False]))


def test_aggregate_zero_mass_raises():
    pi = np.zeros((3, 4))
    with pytest.raises(ValueError, match="non-positive"):
        aggregate_pi_over_mouse_region(pi, np.array([True, False, False]))


# ---------------------------------------------------------------------------
# Scoring


def test_score_sums_pi_within_each_candidate_mask():
    pi_M = np.array([0.1, 0.2, 0.3, 0.4])
    masks = {
        "A": np.array([True, False, False, False]),
        "B": np.array([False, True, True, False]),
        "C": np.array([False, False, False, True]),
    }
    scores = score_candidate_human_regions(pi_M, masks)
    assert scores == {"A": pytest.approx(0.1),
                      "B": pytest.approx(0.5),
                      "C": pytest.approx(0.4)}


def test_fold_enrichment_basic():
    # 10 human parcels, uniform pi_M; mass on any 2-parcel region is 0.2,
    # expected is 2/10 = 0.2, fold = 1.0.
    pi_M = np.full(10, 0.1)
    masks = {
        "two": np.array([True]*2 + [False]*8),
        "five": np.array([True]*5 + [False]*5),
    }
    fold = fold_enrichment_candidate_regions(pi_M, masks)
    assert fold["two"] == pytest.approx(1.0)
    assert fold["five"] == pytest.approx(1.0)


def test_fold_enrichment_concentrated():
    # All mass on parcel 0 → "A" (covering parcel 0) gets fold = n_h / |A|.
    pi_M = np.zeros(10); pi_M[0] = 1.0
    masks = {
        "A": np.array([True] + [False]*9),
        "B": np.array([False]*5 + [True]*5),
    }
    fold = fold_enrichment_candidate_regions(pi_M, masks)
    assert fold["A"] == pytest.approx(10.0)
    assert fold["B"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Ranking


def test_rank_highest_score_is_rank_1():
    scores = {"A": 0.5, "B": 0.3, "C": 0.2}
    assert rank_candidate_regions(scores, "A") == 1
    assert rank_candidate_regions(scores, "B") == 2
    assert rank_candidate_regions(scores, "C") == 3


def test_rank_ties_favor_truth():
    # Two regions tied; truth is one of them. Should be rank 1, not rank 2.
    scores = {"A": 0.5, "B": 0.5, "C": 0.2}
    assert rank_candidate_regions(scores, "A") == 1
    assert rank_candidate_regions(scores, "B") == 1


def test_rank_unknown_region_raises():
    scores = {"A": 0.5}
    with pytest.raises(KeyError):
        rank_candidate_regions(scores, "missing")


# ---------------------------------------------------------------------------
# Per-pair top-K


def test_region_topk_perfect_prediction():
    # Mouse region of 1 parcel mapped entirely to human parcels 5..8.
    pi = np.full((3, 10), 1e-6)
    pi[0, 5:9] = 0.25                          # all mass on H_target
    masks = {
        "H_target": np.array([False]*5 + [True]*4 + [False]),
        "H_decoy":  np.array([True]*4 + [False]*6),
    }
    res = region_topk(pi, np.array([True, False, False]), masks, "H_target",
                      k_list=(1, 3))
    assert res.rank == 1
    assert res.top_k_hits[1] is True
    assert res.top_k_hits[3] is True
    assert res.fold_enrichment > 2.0    # large


def test_region_topk_wrong_region_argmax():
    # Mass concentrated on H_decoy; H_target should be rank 2.
    pi = np.full((2, 10), 1e-6)
    pi[0, 0:4] = 0.25
    masks = {
        "H_target": np.array([False]*5 + [True]*4 + [False]),
        "H_decoy":  np.array([True]*4 + [False]*6),
    }
    res = region_topk(pi, np.array([True, False]), masks, "H_target",
                      k_list=(1, 2))
    assert res.rank == 2
    assert res.top_k_hits[1] is False
    assert res.top_k_hits[2] is True


# ---------------------------------------------------------------------------
# Pipeline-level


def _toy_problem(n_m=20, n_h=30, n_pairs=4, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    # 4 mouse regions of 5 parcels each, 4 human regions of 5 parcels each.
    pi = rng.uniform(0, 1e-4, (n_m, n_h))
    mouse_masks, candidate_masks, pairs = {}, {}, []
    for i in range(n_pairs):
        m = np.zeros(n_m, bool); m[i*5:(i+1)*5] = True
        h = np.zeros(n_h, bool); h[i*5:(i+1)*5] = True
        mouse_masks[f"M{i}"] = m
        candidate_masks[f"H{i}"] = h
        pairs.append((f"M{i}", f"H{i}"))
        # Strong signal: each mouse region maps to its matching human region
        pi[i*5:(i+1)*5, i*5:(i+1)*5] += 1.0
    return pi, pairs, mouse_masks, candidate_masks


def test_evaluate_region_level_strong_signal():
    pi, pairs, mouse_masks, candidate_masks = _toy_problem(n_pairs=4)
    out = evaluate_region_level(pi, pairs, mouse_masks, candidate_masks,
                                 k_list=(1, 2))
    assert out["n_pairs_evaluated"] == 4
    assert out["aggregate"]["top_k"][1] == 1.0
    assert out["aggregate"]["mean_rank"] == 1.0
    assert out["aggregate"]["mean_fold_enrichment"] > 4.0


def test_evaluate_region_level_skips_missing():
    pi, pairs, mouse_masks, candidate_masks = _toy_problem(n_pairs=3)
    # Add a pair whose human region is not in candidate_masks
    pairs = list(pairs) + [("M0", "H_missing")]
    out = evaluate_region_level(pi, pairs, mouse_masks, candidate_masks)
    assert out["n_pairs_evaluated"] == 3
    assert any("not in candidate set" in s for s in out["skipped"])


def test_evaluate_region_level_anchor_breakdown():
    pi, pairs, mouse_masks, candidate_masks = _toy_problem(n_pairs=4)
    anchor_overlap = {"M0": True, "M1": True, "M2": False, "M3": False}
    out = evaluate_region_level(pi, pairs, mouse_masks, candidate_masks,
                                 anchor_overlap=anchor_overlap)
    assert "anchor_overlapping" in out
    assert "novel" in out
    assert out["anchor_overlapping"]["n_pairs"] == 2
    assert out["novel"]["n_pairs"] == 2


# ---------------------------------------------------------------------------
# Nulls


def test_column_permuted_null_is_at_chance():
    """A uniform pi should give chance-level null top-K."""
    n_m, n_h, n_pairs = 20, 30, 4
    pi = np.full((n_m, n_h), 1.0 / n_h)   # uniform
    mouse_masks, candidate_masks, pairs = {}, {}, []
    for i in range(n_pairs):
        m = np.zeros(n_m, bool); m[i*5:(i+1)*5] = True
        h = np.zeros(n_h, bool); h[i*5:(i+1)*5] = True
        mouse_masks[f"M{i}"] = m
        candidate_masks[f"H{i}"] = h
        pairs.append((f"M{i}", f"H{i}"))
    null = column_permuted_null(pi, pairs, mouse_masks, candidate_masks,
                                 k_list=(1, 2), n_trials=20)
    # Uniform pi has fold = 1 exactly regardless of permutation
    assert null["null_fold_mean"] == pytest.approx(1.0, abs=1e-6)
    # Top-K hits driven by ties, all candidate regions get identical scores,
    # so the rank-1 hit happens for *every* call (ties favour truth).
    assert null["null_topk_mean"][1] == pytest.approx(1.0, abs=1e-6)


def test_source_permuted_null_breaks_signal():
    """With a strong signal in pi, source-permuted null should be much worse."""
    pi, pairs, mouse_masks, candidate_masks = _toy_problem(n_pairs=5, rng_seed=7)
    # Real evaluation: top-1 = 100%
    real = evaluate_region_level(pi, pairs, mouse_masks, candidate_masks,
                                  k_list=(1,))
    null = source_permuted_null(pi, pairs, mouse_masks, candidate_masks,
                                 k_list=(1,), n_trials=30)
    # Real top-1 should be substantially above null top-1
    assert real["aggregate"]["top_k"][1] > 0.8
    assert null["null_topk_mean"][1] < 0.5
