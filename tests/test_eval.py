"""Tests for homer.eval — translation, anchor CV, nulls."""
import numpy as np
import pytest

from homer.eval.translation import (
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
    from homer.eval.anchor_cv import anchor_loo_cv
    from homer.models import SupervisedFGW
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
    from homer.eval.nulls import random_pi_null
    res = random_pi_null(mouse_ad, human_ad,
                          held_out_pair_ids=[1], n_trials=3, seed=0)
    assert "top1_mean" in res
    assert "top1_std" in res
    assert res["n_trials"] == 3
