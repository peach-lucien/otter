"""FC translation quality, anchor-independent evaluation of π.

A coupling π pushes mass from mouse nodes to human nodes. If π is biologically
meaningful, then *aggregating* mouse FC values via π should reproduce human FC.

Mathematically:
    Fh_pred[j, k] = Σ_{i,l} π[i,j] · π[l,k] · Fm[i,l]  /  (q[j] · q[k])

where q[j] = Σ_i π[i,j] is the human-side marginal. The metric is the Pearson
correlation between Fh_pred[upper-tri] and the actual Fh[upper-tri].

Why this matters: every other metric we have (anchor recovery, hemisphere
preservation, spot-check) uses the 42 anchors in some form. This one uses the
*entire* π and a held-out signal (the human FC matrix), the closest thing to
"downstream task" validation.

Public API:
    predict_human_fc(pi, fc_mouse)         -> (n_h, n_h) prediction
    fc_translation_quality(pi, Fm, Fh)     -> dict of Pearson r + breakdown
    random_pi_baseline(...)                -> mean of permuted-π baseline
    uniform_pi_baseline(...)               -> trivial p ⊗ q baseline
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def predict_human_fc(pi: np.ndarray, fc_mouse: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Push mouse FC through the coupling to predict human FC.

    `Fh_pred[j,k] = (πᵀ @ Fm @ π)[j,k] / (q[j] · q[k])`
    """
    if pi.shape[0] != fc_mouse.shape[0]:
        raise ValueError(f"pi {pi.shape} and fc_mouse {fc_mouse.shape} mismatch")
    pi = pi.astype(np.float64, copy=False)
    Fm = fc_mouse.astype(np.float64, copy=False)
    num = pi.T @ Fm @ pi
    q = pi.sum(axis=0).clip(min=eps)
    denom = np.outer(q, q)
    return (num / denom).astype(np.float32)


def fc_translation_quality(
    pi: np.ndarray,
    fc_mouse: np.ndarray,
    fc_human: np.ndarray,
    *,
    network_labels_h: Optional[np.ndarray] = None,
    min_marginal: float = 1e-6,
) -> dict:
    """Evaluate Pearson r between predicted and actual human FC.

    Returns a dict with overall + within/cross-network breakdown.
    """
    fc_pred = predict_human_fc(pi, fc_mouse)
    n_h = fc_human.shape[0]

    q = pi.sum(axis=0)
    keep = q > min_marginal
    n_kept = int(keep.sum())

    iu_full = np.triu_indices(n_h, k=1)
    keep_pair = keep[iu_full[0]] & keep[iu_full[1]]
    iu = (iu_full[0][keep_pair], iu_full[1][keep_pair])
    actual = fc_human[iu]
    pred = fc_pred[iu]
    valid = np.isfinite(actual) & np.isfinite(pred)
    if valid.sum() < 100:
        return {"pearson_r_overall": float("nan"), "n_pairs_used": int(valid.sum()),
                "n_human_nodes_kept": n_kept}

    overall_r = float(np.corrcoef(actual[valid], pred[valid])[0, 1])

    out = {
        "pearson_r_overall":   overall_r,
        "n_pairs_used":        int(valid.sum()),
        "n_human_nodes_kept":  n_kept,
        "pred_mean":           float(np.nanmean(pred[valid])),
        "pred_std":            float(np.nanstd(pred[valid])),
        "actual_mean":         float(np.nanmean(actual[valid])),
        "actual_std":          float(np.nanstd(actual[valid])),
    }

    if network_labels_h is not None:
        same_net = network_labels_h[iu[0]] == network_labels_h[iu[1]]
        same_net &= valid
        diff_net = (network_labels_h[iu[0]] != network_labels_h[iu[1]]) & valid
        if same_net.sum() > 50:
            out["pearson_r_within_net"] = float(
                np.corrcoef(actual[same_net], pred[same_net])[0, 1]
            )
            out["n_within_net"] = int(same_net.sum())
        if diff_net.sum() > 50:
            out["pearson_r_cross_net"] = float(
                np.corrcoef(actual[diff_net], pred[diff_net])[0, 1]
            )
            out["n_cross_net"] = int(diff_net.sum())

    return out


def random_pi_baseline(
    fc_mouse: np.ndarray, fc_human: np.ndarray,
    *, p: np.ndarray, q: np.ndarray, n_trials: int = 20, seed: int = 0,
    network_labels_h: Optional[np.ndarray] = None,
) -> dict:
    """Mean Pearson r when π is sampled from random feasible doubly-stochastic
    matrices with the given marginals.
    """
    n_m, n_h = fc_mouse.shape[0], fc_human.shape[0]
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n_trials):
        A = rng.uniform(0.5, 1.5, size=(n_m, n_h))
        for _ in range(50):
            A = A * (p / A.sum(axis=1).clip(min=1e-12))[:, None]
            A = A * (q / A.sum(axis=0).clip(min=1e-12))[None, :]
        rs.append(fc_translation_quality(A, fc_mouse, fc_human,
                                         network_labels_h=network_labels_h)["pearson_r_overall"])
    rs = np.asarray(rs)
    return {
        "pearson_r_mean": float(rs.mean()),
        "pearson_r_std":  float(rs.std()),
        "n_trials": int(n_trials),
    }


def uniform_pi_baseline(
    fc_mouse: np.ndarray, fc_human: np.ndarray,
    *, p: np.ndarray, q: np.ndarray,
    network_labels_h: Optional[np.ndarray] = None,
) -> dict:
    """π = p ⊗ q (independent marginals, every mouse node spreads uniformly).
    Predicted human FC is constant; Pearson r should be 0.
    """
    pi_unif = np.outer(p, q)
    return fc_translation_quality(pi_unif, fc_mouse, fc_human,
                                   network_labels_h=network_labels_h)
