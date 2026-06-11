"""Tests for the spatial-autocorrelation-preserving spin null (eval.nulls.spin_null)."""
import numpy as np
from scipy.spatial import cKDTree

from homer.eval.nulls import spin_null


def _sphere_coords(n=300, seed=0):
    rng = np.random.default_rng(seed)
    xyz = rng.standard_normal((n, 3))
    return xyz


def _smooth_map(coords, seed):
    """A spatially-autocorrelated map: each parcel = mean of its 30 neighbours."""
    rng = np.random.default_rng(seed)
    nbr = cKDTree(coords).query(coords, k=30)[1]
    return rng.standard_normal(len(coords))[nbr].mean(1)


def test_spin_null_smooth_vs_itself_is_significant():
    coords = _sphere_coords()
    m = _smooth_map(coords, seed=1)
    res = spin_null(m, m, coords, n_trials=300, seed=0)
    assert abs(res["r_observed"]) > 0.99
    assert res["p_spin"] < 0.05          # perfect alignment beats the spin null


def test_spin_null_two_independent_smooth_maps_not_significant():
    coords = _sphere_coords()
    m1 = _smooth_map(coords, seed=1)
    m2 = _smooth_map(coords, seed=2)
    res = spin_null(m1, m2, coords, n_trials=300, seed=0)
    # two independent smooth maps should NOT correlate beyond spatial chance
    assert res["p_spin"] > 0.05


def test_translation_spin_null_runs_and_is_sane():
    from homer.eval.nulls import translation_spin_null
    rng = np.random.default_rng(0)
    n_m, n_h = 120, 150
    coords = rng.standard_normal((n_m, 3))
    mouse_map = _smooth_map(coords, seed=3)
    pi = rng.random((n_m, n_h))
    pi /= pi.sum()
    obs_human = rng.standard_normal(n_h)
    res = translation_spin_null(mouse_map, obs_human, pi, coords, n_trials=100, seed=0)
    assert np.isfinite(res["r_observed"])
    assert 0.0 < res["p_translation_spin"] <= 1.0
    assert res["n_trials"] == 100


def test_spin_null_handles_nans_and_shapes():
    coords = _sphere_coords(n=120)
    a = _smooth_map(coords, 1); b = _smooth_map(coords, 1)
    a[:10] = np.nan
    res = spin_null(a, b, coords, n_trials=100, seed=0)
    assert np.isfinite(res["r_observed"])
    assert 0.0 < res["p_spin"] <= 1.0
    assert res["n_trials"] == 100
