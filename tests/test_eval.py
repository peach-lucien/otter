"""Tests for otter.eval, translation, anchor CV, nulls."""
import numpy as np
import pytest

from otter.eval.translation import (
    fc_translation_quality,
    predict_human_fc,
    uniform_pi_baseline,
)


def test_predict_human_fc_identity_is_self():
    """If π = identity (n × n) / n, predicted human FC = mouse FC exactly."""
    n = 10
    rng = np.random.default_rng(0)
    fc = rng.uniform(-0.5, 0.5, size=(n, n))
    fc = 0.5 * (fc + fc.T); np.fill_diagonal(fc, 1.0)
    pi = np.eye(n) / n
    pred = predict_human_fc(pi, fc)
    np.testing.assert_allclose(pred, fc, atol=1e-5)


def test_fc_translation_perfect_correlation_when_pi_is_identity():
    # n>=15 so the upper-tri (105 pairs) clears the >100 threshold inside
    # fc_translation_quality (which avoids reporting r on tiny samples).
    n = 15
    rng = np.random.default_rng(0)
    fc = rng.uniform(-0.5, 0.5, size=(n, n))
    fc = 0.5 * (fc + fc.T); np.fill_diagonal(fc, 1.0)
    pi = np.eye(n) / n
    res = fc_translation_quality(pi, fc, fc)
    assert abs(res["pearson_r_overall"] - 1.0) < 1e-6


def test_fc_translation_random_pi_has_lower_r():
    n = 20
    rng = np.random.default_rng(0)
    fc = rng.uniform(-0.5, 0.5, size=(n, n))
    fc = 0.5 * (fc + fc.T); np.fill_diagonal(fc, 1.0)
    pi_random = rng.uniform(0, 1, size=(n, n))
    pi_random /= pi_random.sum()
    res = fc_translation_quality(pi_random, fc, fc)
    # With a random π, r should be far from 1
    assert res["pearson_r_overall"] < 0.5


def test_uniform_pi_baseline_gives_constant_prediction():
    n = 15
    rng = np.random.default_rng(0)
    fc = rng.uniform(-0.5, 0.5, size=(n, n))
    fc = 0.5 * (fc + fc.T); np.fill_diagonal(fc, 1.0)
    p = np.full(n, 1.0 / n)
    res = uniform_pi_baseline(fc, fc, p=p, q=p)
    # Predicted FC is constant → r is undefined or near 0; we expect small std
    assert "pred_std" in res
    assert res["pred_std"] < 1e-5


def test_anchor_loo_cv_smoke(mouse_ad, human_ad):
    """The CV harness runs end-to-end on one network."""
    from otter.eval.anchor_cv import anchor_loo_cv
    from otter.models import SupervisedFGW
    # Use only 1 network for speed; pick 'frontal_dmn' (pair_id 1)
    res = anchor_loo_cv(
        model_factory=lambda: SupervisedFGW(epsilon=1e-2),
        mouse_ad=mouse_ad, human_ad=human_ad,
        networks=["frontal_dmn"],
    )
    assert "per_network" in res
    assert "frontal_dmn" in res["per_network"]
    assert "weighted" in res
    assert "top1" in res["per_network"]["frontal_dmn"]


def test_random_pi_null_smoke(mouse_ad, human_ad):
    from otter.eval.nulls import random_pi_null
    res = random_pi_null(mouse_ad, human_ad,
                          held_out_pair_ids=[1], n_trials=3, seed=0)
    assert "top1_mean" in res
    assert "top1_std" in res
    assert res["n_trials"] == 3


# ---------------------------------------------------------------------------
# Multi-source trust (v1), augments compute_trust_score


def test_compute_multisource_trust_returns_evidence_tiers(mouse_ad, human_ad):
    """With no external evidence, tiers fall back to structural / low_evidence."""
    from otter.eval.trust_score import compute_multisource_trust
    n_m = mouse_ad.uns["n_nodes"]
    n_h = human_ad.uns["n_nodes"]
    pi = np.full((n_m, n_h), 1.0 / n_h)

    out = compute_multisource_trust(mouse_ad, human_ad, pi)
    assert out["evidence_tier"].shape == (n_m,)
    expected = {"anchored_and_validated", "anchored_only", "validated_only",
                "structural", "low_evidence"}
    assert set(out["evidence_tier"].tolist()).issubset(expected)
    # 10 Garin anchors in the fixture (5 pair_ids × 2 hemispheres)
    assert out["garin_anchored"].sum() == 10
    assert out["pack_anchored"].sum() == 0
    assert np.isnan(out["beauchamp_top1"]).all()


def test_compute_multisource_trust_with_pack_anchored(mouse_ad, human_ad):
    """Passing region_anchor entries flags those parcels as pack_anchored."""
    from otter.eval.trust_score import compute_multisource_trust
    from otter.data.region_anchors import RegionAnchorEntry
    n_m = mouse_ad.uns["n_nodes"]
    n_h = human_ad.uns["n_nodes"]
    pi = np.full((n_m, n_h), 1.0 / n_h)

    entries = [RegionAnchorEntry(pair_id=30, label="test",
                                  mouse_indices=[12, 13, 14],
                                  human_indices=[0, 1])]
    out = compute_multisource_trust(mouse_ad, human_ad, pi,
                                     region_anchor_entries=entries)
    assert out["pack_anchored"][12]
    assert out["pack_anchored"][13]
    assert out["pack_anchored"][14]
    assert not out["pack_anchored"][0]
    assert out["evidence_tier"][12] == "anchored_only"


# ---------------------------------------------------------------------------
# Network coherence (v2)


def test_network_compactness_smoke(mouse_ad, human_ad):
    """network_compactness returns per-network compactness metrics."""
    from otter.eval.network_coherence import network_compactness
    n_m = mouse_ad.uns["n_nodes"]
    n_h = human_ad.uns["n_nodes"]
    rng = np.random.default_rng(7)
    pi = rng.uniform(0, 1, (n_m, n_h))

    out = network_compactness(pi, mouse_ad.var, human_ad.var)
    assert len(out) >= 1
    for net, m in out.items():
        assert m["n_mouse"] > 0
        assert m["median_pairwise_dist_mm"] >= 0
        assert m["mean_centroid_spread_mm"] >= 0


def test_compare_network_compactness_delta_zero(mouse_ad, human_ad):
    """Comparing pi vs pi yields zero delta everywhere."""
    from otter.eval.network_coherence import compare_network_compactness
    n_m = mouse_ad.uns["n_nodes"]
    n_h = human_ad.uns["n_nodes"]
    rng = np.random.default_rng(11)
    pi = rng.uniform(0, 1, (n_m, n_h))

    cmp = compare_network_compactness(pi, pi, mouse_ad.var, human_ad.var,
                                       label_a="x", label_b="y")
    for net, m in cmp.items():
        assert abs(m["delta_med"]) < 1e-9
        assert abs(m["delta_spread"]) < 1e-9
