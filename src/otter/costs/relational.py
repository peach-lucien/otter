"""Intra-species relational distance / cost matrices for FGW.

Each function takes a (n_nodes, n_nodes) FC-like similarity matrix (or a
(n_nodes, n_features) feature matrix for gene expression) and returns a
symmetric, zero-diagonal, finite (n_nodes, n_nodes) distance matrix.
"""
from __future__ import annotations

import numpy as np


def _symmetrise(d: np.ndarray) -> np.ndarray:
    return 0.5 * (d + d.T)


def correlation_distance(fc: np.ndarray) -> np.ndarray:
    """1 - r, symmetrised, zero-diagonal. The basic FC-derived cost.

    Output values lie in [0, 2], since r ∈ [-1, 1]. Negative correlations give
    distance > 1, which GW admits.
    """
    if not np.allclose(fc, fc.T, equal_nan=True, atol=1e-4):
        raise ValueError("input FC is not symmetric")
    d = 1.0 - fc
    d = _symmetrise(d)
    np.fill_diagonal(d, 0.0)
    if not np.isfinite(d).all():
        m = np.nanmean(d[~np.eye(d.shape[0], dtype=bool)])
        d = np.where(np.isfinite(d), d, m)
    return d.astype(np.float64)


def fisher_z_distance(fc: np.ndarray, *, clip: float = 0.999999) -> np.ndarray:
    """Distance based on Fisher-z'd correlations: |arctanh(r_i) - arctanh(r_j)|."""
    z = np.arctanh(np.clip(fc, -clip, clip)).astype(np.float64)
    sq = (z * z).sum(axis=1, keepdims=True)
    d2 = sq + sq.T - 2.0 * (z @ z.T)
    d2 = np.clip(d2, 0.0, None)
    d = np.sqrt(_symmetrise(d2))
    np.fill_diagonal(d, 0.0)
    return d


def geodesic_fc_distance(fc: np.ndarray, *, threshold: float = 0.2) -> np.ndarray:
    """Threshold |FC| to a sparse weighted graph, edge weight = 1/|r|, then
    return all-pairs shortest path distance."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import shortest_path

    a = np.abs(fc)
    keep = a >= threshold
    np.fill_diagonal(keep, False)
    w = np.where(keep, 1.0 / np.maximum(a, 1e-3), 0.0)
    d = shortest_path(sp.csr_matrix(w), directed=False)
    if not np.isfinite(d).all():
        finite_max = np.nanmax(d[np.isfinite(d)])
        d = np.where(np.isfinite(d), d, finite_max * 1.5)
    d = _symmetrise(d)
    np.fill_diagonal(d, 0.0)
    return d


def sc_correlation_distance(sc: np.ndarray, *, log_transform: bool = True,
                              eps: float = 1e-6) -> np.ndarray:
    """Within-species relational distance from a structural connectome matrix.

    SC values are heavy-tailed counts. log_transform=True applies log1p first,
    then computes Pearson correlation of each node's SC fingerprint, returns 1-r.
    """
    if not np.allclose(sc, sc.T, atol=1e-3):
        raise ValueError("SC matrix is not symmetric")
    x = np.log1p(sc) if log_transform else sc.copy()
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True).clip(min=eps)
    z = (x - mu) / sd
    r = (z @ z.T) / z.shape[1]
    d = 1.0 - r
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)
    return np.clip(d, 0.0, 2.0).astype(np.float64)


def gene_correlation_distance(expr: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """Within-species relational distance from a (n_nodes, n_genes) expression
    matrix. Pearson correlation between gene-expression fingerprints, then 1-r.
    """
    n = expr.shape[0]
    valid = np.isfinite(expr).all(axis=1)
    x = expr.copy().astype(np.float64)
    col_mu = np.nanmean(x[valid], axis=0, keepdims=True)
    col_sd = np.nanstd(x[valid], axis=0, keepdims=True).clip(min=eps)
    x = (x - col_mu) / col_sd
    row_mu = np.nanmean(x, axis=1, keepdims=True)
    row_sd = np.nanstd(x, axis=1, keepdims=True).clip(min=eps)
    z = (x - row_mu) / row_sd
    z[~np.isfinite(z)] = 0.0
    r = (z @ z.T) / z.shape[1]
    d = 1.0 - r
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)
    if (~valid).any():
        med = float(np.nanmedian(d[valid][:, valid]))
        d[~valid, :] = med
        d[:, ~valid] = med
        np.fill_diagonal(d, 0.0)
    return np.clip(d, 0.0, 2.0).astype(np.float64)


def anchor_relationship_features(fc_mean: np.ndarray, anchor_pos: np.ndarray) -> np.ndarray:
    """For each node, return its FC values to each anchor.

    Returns (n_nodes, n_anchors) array. Row i = FC[i, anchor_k] for k in
    sorted anchor order.
    """
    return fc_mean[:, anchor_pos].astype(np.float64)
