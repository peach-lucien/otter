"""Cross-species cost matrices M used by the FGW W-term.

Public:
    cross_species_anchor_M, cosine distance over anchor-relationship features
    cross_species_gene_cost, cosine distance over ortholog-aligned gene vectors

The xyz cross-species cost is computed inline in pipeline scripts (it's a
straightforward Euclidean distance between per-species-normalised xyz coordinates
plus a normalisation factor) so it doesn't need its own helper.
"""
from __future__ import annotations

import numpy as np

from otter.costs.relational import anchor_relationship_features


def cross_species_anchor_M(
    fc_m: np.ndarray, fc_h: np.ndarray,
    anchor_pos_m: np.ndarray, anchor_pos_h: np.ndarray,
    *, eps: float = 1e-6,
) -> np.ndarray:
    """Cross-species cost matrix from anchor-relationship feature vectors.

    Each node gets a vector of FC values to each anchor. Since the 42 anchors
    are in known 1-to-1 cross-species correspondence (sorted by pair_id+hemi
    in both species), these vectors are directly comparable between species.
    Returns (n_m, n_h) cosine-distance matrix.
    """
    af_m = anchor_relationship_features(fc_m, anchor_pos_m)
    af_h = anchor_relationship_features(fc_h, anchor_pos_h)
    af_m = (af_m - af_m.mean(0, keepdims=True)) / af_m.std(0, keepdims=True).clip(min=eps)
    af_h = (af_h - af_h.mean(0, keepdims=True)) / af_h.std(0, keepdims=True).clip(min=eps)
    af_m = af_m / np.linalg.norm(af_m, axis=1, keepdims=True).clip(min=eps)
    af_h = af_h / np.linalg.norm(af_h, axis=1, keepdims=True).clip(min=eps)
    cos = af_m @ af_h.T
    return (1.0 - cos).clip(0.0, 2.0).astype(np.float64)


def cross_species_gene_cost(expr_m: np.ndarray, expr_h: np.ndarray,
                             *, eps: float = 1e-6) -> np.ndarray:
    """Cross-species cosine distance between mouse and human ortholog vectors.
    Inputs must already be ortholog-aligned (same ordered set of genes).
    Returns (n_m, n_h) cost matrix, normalised to roughly [0, 2].
    """
    if expr_m.shape[1] != expr_h.shape[1]:
        raise ValueError(f"shape mismatch: mouse {expr_m.shape}, human {expr_h.shape}")

    def _std(x):
        x = x.copy().astype(np.float64)
        valid = np.isfinite(x).all(axis=1)
        mu = np.nanmean(x[valid], axis=0, keepdims=True)
        sd = np.nanstd(x[valid], axis=0, keepdims=True).clip(min=eps)
        z = (x - mu) / sd
        z[~np.isfinite(z)] = 0.0
        return z, valid

    zm, vm = _std(expr_m)
    zh, vh = _std(expr_h)
    zm = zm / np.linalg.norm(zm, axis=1, keepdims=True).clip(min=eps)
    zh = zh / np.linalg.norm(zh, axis=1, keepdims=True).clip(min=eps)
    cos = zm @ zh.T
    d = 1.0 - cos
    if (~vm).any(): d[~vm, :] = 1.0
    if (~vh).any(): d[:, ~vh] = 1.0
    return np.clip(d, 0.0, 2.0).astype(np.float64)
