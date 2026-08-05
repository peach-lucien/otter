"""Null distributions for held-out anchor CV.

Two principled nulls:
  - random_pi_null, sample uniform random π satisfying mouse marginal
  - permuted_anchor_null, shuffle anchor pair_ids before solving FGW

These give the reference distributions for z-scores on the real top-1.
Headline numbers from the comparison table:
    real top-1 81% vs random_pi 28%±7% (z=+7.5)
    real top-1 81% vs permuted_anchor 31%±3% (z=+17.8)
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from otter.data.anchors import (
    AnchorIndex, get_anchor_index, held_out_metrics_graded,
)


def random_pi_null(
    mouse_ad,
    human_ad,
    *,
    held_out_pair_ids: Sequence[int],
    n_trials: int = 50,
    seed: int = 0,
) -> dict:
    """Sample n_trials random π satisfying the mouse uniform marginal,
    evaluate held-out anchor metrics. Returns mean ± std + per-trial list.
    """
    idx_m = get_anchor_index(mouse_ad.var)
    idx_h = get_anchor_index(human_ad.var)
    n_m = mouse_ad.uns["n_nodes"]; n_h = human_ad.uns["n_nodes"]
    p = np.full(n_m, 1.0 / n_m, dtype=np.float64)

    trials = []
    for t in range(n_trials):
        rng = np.random.default_rng(seed + t)
        A = rng.uniform(0.5, 1.5, size=(n_m, n_h))
        A = A * (p / A.sum(axis=1).clip(min=1e-12))[:, None]
        pi_anchor = A[np.ix_(idx_m.pos, idx_h.pos)]
        m = held_out_metrics_graded(
            pi_anchor, idx_m, idx_h, held_out_pair_ids, var_h=human_ad.var,
        )
        trials.append({k: m[k] for k in ("top1", "top5", "pair_id", "hemisphere",
                                          "mean_rank", "mean_xyz_dist") if k in m})
    return _summarise(trials)


def permuted_anchor_null(
    mouse_ad,
    human_ad,
    solve_fn,
    *,
    held_out_pair_ids: Sequence[int],
    n_trials: int = 5,
    seed: int = 0,
) -> dict:
    """Shuffle the anchor cross-species correspondence (pair_ids permuted),
    re-solve FGW, evaluate. Tests whether the *specific* mouse↔human anchor
    pairings drive the result vs "having anchor supervision in general".

    `solve_fn(visible_pair_ids, idx_m_perm, idx_h_perm) -> pi` is supplied by
    the caller (lets you wire in any model/config you want).
    """
    idx_m = get_anchor_index(mouse_ad.var)
    idx_h = get_anchor_index(human_ad.var)

    visible_pair_ids = sorted(int(p) for p in idx_m.pair_ids
                               if int(p) not in set(held_out_pair_ids))

    trials = []
    for t in range(n_trials):
        rng = np.random.default_rng(seed + t)
        perm = rng.permutation(len(idx_m.pos))
        # Permute idx_m's positions to scramble the anchor correspondence
        idx_m_perm = AnchorIndex(
            pos=idx_m.pos[perm],
            pair_ids=idx_m.pair_ids[perm],
            hemispheres=idx_m.hemispheres[perm],
            keys=[idx_m.keys[i] for i in perm],
        )
        pi = solve_fn(visible_pair_ids, idx_m_perm, idx_h)
        pi_anchor = pi[np.ix_(idx_m.pos, idx_h.pos)]
        m = held_out_metrics_graded(
            pi_anchor, idx_m, idx_h, held_out_pair_ids, var_h=human_ad.var,
        )
        trials.append({k: m[k] for k in ("top1", "top5", "pair_id", "hemisphere",
                                          "mean_rank", "mean_xyz_dist") if k in m})
    return _summarise(trials)


def _summarise(trials: list[dict]) -> dict:
    out: dict = {"n_trials": len(trials), "trials": trials}
    if not trials:
        return out
    keys = trials[0].keys()
    for k in keys:
        vals = np.array([t[k] for t in trials], dtype=float)
        out[f"{k}_mean"] = float(np.nanmean(vals))
        out[f"{k}_std"]  = float(np.nanstd(vals))
    return out


# ---------------------------------------------------------------------------
# Spatial-autocorrelation-preserving (spin) null for parcel-map correlations
# ---------------------------------------------------------------------------
def _haar_rotation(rng: np.random.Generator) -> np.ndarray:
    """A uniformly-random proper 3x3 rotation (Haar measure) via QR."""
    Q, R = np.linalg.qr(rng.standard_normal((3, 3)))
    Q = Q @ np.diag(np.sign(np.diag(R)))          # fix QR sign ambiguity
    if np.linalg.det(Q) < 0:                        # enforce proper rotation
        Q[:, 0] = -Q[:, 0]
    return Q


def spin_null(
    map_a,
    map_b,
    coords,
    *,
    n_trials: int = 1000,
    seed: int = 0,
) -> dict:
    """Spin test (Alexander-Bloch / Vázquez-Rodríguez) for the correlation
    between two parcel-resolved brain maps.

    The permuted-π / row-shuffle null destroys spatial autocorrelation, which
    inflates significance when comparing two *smooth* maps (gradients, myelin).
    The spin null instead preserves spatial autocorrelation: it projects parcel
    centroids onto a sphere, applies a random rotation, reassigns each original
    parcel to the nearest rotated parcel, permutes ``map_b`` by that assignment,
    and recomputes the correlation. The resulting null distribution has the same
    spatial smoothness as the real map, so a high observed |r| only beats it if
    the cross-map alignment exceeds what smoothness alone produces.

    Parameters
    ----------
    map_a, map_b : array (n_parcels,)
        The two maps to correlate (e.g. observed vs predicted human gradient).
        NaNs are allowed; only entries finite in both are used.
    coords : array (n_parcels, 3)
        Parcel centroid coordinates (e.g. MNI mm). Centred + projected to a unit
        sphere internally. Whole-brain rotation (subcortex included).
    n_trials : int
        Number of random rotations.

    Returns
    -------
    dict with observed Pearson r, spin p-value (two-sided on |r|), and the null
    summary. Compare ``p_spin`` against the (over-optimistic) permuted-π p.
    """
    from scipy.spatial import cKDTree
    from scipy.stats import pearsonr

    a = np.asarray(map_a, dtype=float)
    b = np.asarray(map_b, dtype=float)
    xyz = np.asarray(coords, dtype=float)
    if not (a.shape == b.shape == (xyz.shape[0],)):
        raise ValueError("map_a, map_b, coords must share n_parcels")

    # Project centroids to a unit sphere (centre, then normalise).
    c = xyz - np.nanmean(xyz, axis=0)
    nrm = np.linalg.norm(c, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    sph = c / nrm

    def _corr(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            return np.nan
        return float(pearsonr(x[m], y[m])[0])

    r_obs = _corr(a, b)
    rng = np.random.default_rng(seed)
    null = np.empty(n_trials, dtype=float)
    for t in range(n_trials):
        rot = sph @ _haar_rotation(rng).T          # rotate the sphere
        # nearest rotated parcel for each original parcel position
        _, perm = cKDTree(rot).query(sph)
        null[t] = _corr(a, b[perm])

    absnull = np.abs(null)
    p_spin = (np.sum(absnull >= abs(r_obs)) + 1) / (n_trials + 1)
    return {
        "r_observed": r_obs,
        "p_spin": float(p_spin),
        "n_trials": int(n_trials),
        "null_mean": float(np.nanmean(null)),
        "null_abs_mean": float(np.nanmean(absnull)),
        "null_abs_p95": float(np.nanpercentile(absnull, 95)),
        "null_ci95": [float(np.nanpercentile(null, 2.5)),
                       float(np.nanpercentile(null, 97.5))],
    }


def _route_normalized(mouse_vec: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Transport-weighted average: predicted[j] = Σ_i mouse[i]·π[i,j] / Σ_i π[i,j]."""
    den = pi.sum(axis=0)
    out = np.full(pi.shape[1], np.nan)
    ok = den > 1e-12
    out[ok] = (mouse_vec @ pi)[ok] / den[ok]
    return out


def translation_spin_null(
    mouse_map,
    observed_human_map,
    pi,
    mouse_coords,
    *,
    n_trials: int = 1000,
    seed: int = 0,
) -> dict:
    """The *fair* null for a cross-species TRANSLATION claim.

    Three nulls are possible when asking "does routing ``mouse_map`` through π
    predict ``observed_human_map``?", and they test different hypotheses:

      C. **Fully shuffle the mouse input** (replace it with spatial noise) →
         tests only "does a smooth map beat noise?". Too lenient: any smooth
         brain map clears it, because routed noise predicts nothing. This is the
         original permuted-input/permuted-π behaviour and it INFLATES significance.
      A. **Spin the observed human map** (`spin_null`) → tests "do two smooth maps
         align beyond spatial autocorrelation?". Spatially fair but ignores π's
         structure in the null.
      B. **Spin the mouse input and route it through the REAL π** (this function)
         → tests "is it *this specific* mouse spatial pattern, not a rotated one,
         that, through OTTER's actual coupling, predicts the human map?". This
         keeps both the spatial autocorrelation AND π, breaking only the specific
         mouse→human correspondence. It is the appropriate null for a translation
         claim.

    Empirically A and B agree closely (both control spatial autocorrelation),
    while C is far more lenient. Report B (and/or A); do NOT rely on C.

    Returns observed |r|, the spin-B p-value, and the null summary.
    """
    from scipy.spatial import cKDTree
    from scipy.stats import pearsonr

    m = np.asarray(mouse_map, dtype=float)
    obs = np.asarray(observed_human_map, dtype=float)
    pi = np.asarray(pi, dtype=float)
    mc = np.asarray(mouse_coords, dtype=float)

    c = mc - np.nanmean(mc, axis=0)
    nrm = np.linalg.norm(c, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    sph = c / nrm

    def _corr(x, y):
        ok = np.isfinite(x) & np.isfinite(y)
        return float(pearsonr(x[ok], y[ok])[0]) if ok.sum() >= 3 else np.nan

    r_obs = _corr(_route_normalized(m, pi), obs)
    rng = np.random.default_rng(seed)
    null = np.empty(n_trials, dtype=float)
    for t in range(n_trials):
        rot = sph @ _haar_rotation(rng).T
        _, perm = cKDTree(rot).query(sph)
        null[t] = _corr(_route_normalized(m[perm], pi), obs)

    absnull = np.abs(null)
    p = (np.sum(absnull >= abs(r_obs)) + 1) / (n_trials + 1)
    return {
        "r_observed": r_obs,
        "p_translation_spin": float(p),
        "n_trials": int(n_trials),
        "null_abs_mean": float(np.nanmean(absnull)),
        "null_abs_p95": float(np.nanpercentile(absnull, 95)),
    }
