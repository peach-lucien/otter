"""Tests for homer.costs (relational + cross-species + normalisation)."""
import numpy as np

from homer.costs import (
    correlation_distance,
    cross_species_anchor_M,
    cross_species_gene_cost,
    fisher_z_distance,
    gene_correlation_distance,
    geodesic_fc_distance,
    normalise_cost,
    sc_correlation_distance,
)


def _is_valid_cost_matrix(d, n):
    """A cost matrix should be (n, n), symmetric, zero-diagonal, finite, non-negative."""
    assert d.shape == (n, n)
    assert np.allclose(d, d.T, atol=1e-8)
    assert np.allclose(np.diag(d), 0.0, atol=1e-8)
    assert np.isfinite(d).all()
    assert (d >= 0).all()


def test_correlation_distance(mouse_ad):
    fc = mouse_ad.uns["fc_mean"].astype(np.float64)
    d = correlation_distance(fc)
    _is_valid_cost_matrix(d, 20)
    # 1 - r, where r ∈ [-1,1] → d ∈ [0, 2]
    assert (d <= 2.0 + 1e-6).all()


def test_correlation_distance_rejects_asymmetric():
    n = 5
    rng = np.random.default_rng(0)
    fc = rng.uniform(-1, 1, size=(n, n))   # not symmetric
    import pytest
    with pytest.raises(ValueError, match="not symmetric"):
        correlation_distance(fc)


def test_fisher_z_distance(mouse_ad):
    fc = mouse_ad.uns["fc_mean"].astype(np.float64)
    d = fisher_z_distance(fc)
    _is_valid_cost_matrix(d, 20)


def test_geodesic_fc_distance(mouse_ad):
    fc = mouse_ad.uns["fc_mean"].astype(np.float64)
    d = geodesic_fc_distance(fc, threshold=0.05)
    _is_valid_cost_matrix(d, 20)


def test_sc_correlation_distance():
    rng = np.random.default_rng(0)
    sc = rng.poisson(5, size=(15, 15)).astype(np.float64)
    sc = sc + sc.T
    np.fill_diagonal(sc, 0)
    d = sc_correlation_distance(sc)
    _is_valid_cost_matrix(d, 15)


def test_gene_correlation_distance():
    rng = np.random.default_rng(0)
    expr = rng.uniform(0, 10, size=(15, 50))
    d = gene_correlation_distance(expr)
    _is_valid_cost_matrix(d, 15)


def test_gene_correlation_with_nan_rows():
    """NaN rows get filled with median distance — no NaNs in output."""
    rng = np.random.default_rng(0)
    expr = rng.uniform(0, 10, size=(15, 50))
    expr[2, :] = np.nan        # one all-NaN row
    d = gene_correlation_distance(expr)
    _is_valid_cost_matrix(d, 15)


def test_cross_species_anchor_M_shapes(mouse_ad, human_ad):
    fc_m = mouse_ad.uns["fc_mean"]
    fc_h = human_ad.uns["fc_mean"]
    pos_m = np.array([0, 2, 4, 6, 8])         # use 5 anchors (a subset)
    pos_h = np.array([0, 2, 4, 6, 8])
    M = cross_species_anchor_M(fc_m, fc_h, pos_m, pos_h)
    assert M.shape == (20, 25)
    assert np.isfinite(M).all()
    assert (M >= 0).all() and (M <= 2).all()


def test_cross_species_gene_cost_shape_mismatch_raises():
    rng = np.random.default_rng(0)
    em = rng.uniform(0, 1, size=(10, 5))
    eh = rng.uniform(0, 1, size=(15, 8))           # wrong number of genes
    import pytest
    with pytest.raises(ValueError, match="shape mismatch"):
        cross_species_gene_cost(em, eh)


def test_cross_species_gene_cost_same_species():
    """Mouse vs mouse with identical orthologs → cosine sim ≈ 1 → distance ≈ 0
    on the diagonal."""
    rng = np.random.default_rng(0)
    e = rng.uniform(0, 1, size=(8, 12))
    d = cross_species_gene_cost(e, e)
    assert d.shape == (8, 8)
    # Self-distance should be ≈ 0 on the diagonal
    np.testing.assert_allclose(np.diag(d), 0.0, atol=1e-6)


def test_normalise_cost_max_scheme():
    rng = np.random.default_rng(0)
    d = rng.uniform(0, 5, size=(10, 10))
    d = 0.5 * (d + d.T); np.fill_diagonal(d, 0)
    n = normalise_cost(d, scheme="max")
    off = n[~np.eye(10, dtype=bool)]
    assert np.isclose(off.max(), 1.0)


def test_normalise_cost_none_passes_through():
    rng = np.random.default_rng(0)
    d = rng.uniform(0, 5, size=(10, 10))
    n = normalise_cost(d, scheme="none")
    np.testing.assert_array_equal(d, n)


def test_normalise_cost_invalid_scheme_raises():
    import pytest
    d = np.eye(5)
    with pytest.raises(ValueError, match="unknown scheme"):
        normalise_cost(d, scheme="bogus")
