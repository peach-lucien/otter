"""Null distributions for held-out anchor CV.

Two nulls:
  - random_pi_null, sample uniform random π satisfying mouse marginal
  - permuted_anchor_null, shuffle anchor pair_ids before solving FGW

These give the reference distributions for z-scores on the real top-1.
Values from the comparison table:
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
    the caller, so any model or configuration can be substituted.
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


_MIRROR = np.diag([-1.0, 1.0, 1.0])

_WHOLE_ALIASES = ("whole", "whole_volume", "none", "off")
_PRESERVE_ALIASES = ("preserve", "preserving", "per_hemisphere", "hemisphere")


def _project_to_sphere(xyz: np.ndarray) -> np.ndarray:
    """Centre a coordinate block on its own centroid and normalise to a unit sphere."""
    c = xyz - np.nanmean(xyz, axis=0)
    nrm = np.linalg.norm(c, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return c / nrm


def _hemisphere_groups(xyz_centred: np.ndarray, hemi_labels) -> tuple:
    """Index arrays for the left, right and midline parcels.

    ``hemi_labels`` is used when supplied, matching on the first letter of each
    label so that "L"/"R", "left"/"right" and "lh"/"rh" all work. Otherwise the
    sign of the centred x coordinate decides. Parcels sitting exactly on the
    midline, or carrying a label that is neither left nor right, go into the
    midline group.
    """
    n = int(xyz_centred.shape[0])
    if hemi_labels is not None:
        lab = np.asarray(hemi_labels).ravel()
        if lab.shape[0] != n:
            raise ValueError("hemi_labels must have one entry per parcel")
        first = np.array([str(v).strip().lower()[:1] if v is not None else ""
                          for v in lab])
        left = first == "l"
        right = first == "r"
    else:
        x = xyz_centred[:, 0]
        left = x < 0
        right = x > 0
    mid = ~(left | right)
    return np.flatnonzero(left), np.flatnonzero(right), np.flatnonzero(mid)


def _spin_permuter(coords, *, hemisphere: str = "whole", hemi_labels=None):
    """Build the rotation to permutation map used by the spin nulls.

    Returns ``(permute, info)``. Calling ``permute(rng)`` draws one Haar
    rotation from ``rng`` and returns an index array ``perm`` so that
    ``values[perm]`` is one spun surrogate of ``values``.

    Modes
    -----
    ``hemisphere="whole"``
        One rotation applied to the whole volume. All parcel centroids are
        centred on the global centroid, projected to a single sphere, rotated
        together, and each parcel takes the value of its nearest rotated
        neighbour. This preserves the spatial autocorrelation of the map and
        the global shape of the parcellation. It does not preserve hemisphere,
        so a parcel can inherit a value from the opposite side.

    ``hemisphere="preserve"``
        Each hemisphere is centred on its own centroid and projected to its own
        sphere. The left hemisphere is rotated by Q and the right by MQM with
        M = diag(-1, 1, 1), so the two hemispheres receive mirror image
        rotations and the surrogate stays symmetric under reflection. Nearest
        neighbour lookup runs within a hemisphere, so no assignment crosses the
        midline. This preserves spatial autocorrelation, hemisphere membership
        and the left/right symmetry of the rotation. It is the right choice
        when either map is bilaterally symmetric, because a whole volume spin
        then leaves the two hemispheres of the surrogate aligned with each
        other and inflates the false positive rate.

    Both modes reassign with replacement, as in Alexander-Bloch et al. The
    surrogate is a nearest neighbour relabelling, not a bijection.
    """
    from scipy.spatial import cKDTree

    xyz = np.asarray(coords, dtype=float)
    mode = str(hemisphere).strip().lower()

    if mode in _WHOLE_ALIASES:
        sph = _project_to_sphere(xyz)

        def permute(rng: np.random.Generator) -> np.ndarray:
            rot = sph @ _haar_rotation(rng).T
            return cKDTree(rot).query(sph)[1]

        return permute, {"hemisphere": "whole", "n_midline": 0,
                         "n_left": 0, "n_right": 0}

    if mode not in _PRESERVE_ALIASES:
        raise ValueError(
            f"hemisphere must be one of {_WHOLE_ALIASES + _PRESERVE_ALIASES}, "
            f"got {hemisphere!r}")

    centred = xyz - np.nanmean(xyz, axis=0)
    left, right, mid = _hemisphere_groups(centred, hemi_labels)
    sph = np.zeros_like(centred)
    for grp in (left, right):
        if grp.size:
            sph[grp] = _project_to_sphere(centred[grp])
    n = int(xyz.shape[0])

    def permute(rng: np.random.Generator) -> np.ndarray:
        Q = _haar_rotation(rng)
        perm = np.arange(n)
        for grp, rot_mat in ((left, Q), (right, _MIRROR @ Q @ _MIRROR)):
            if grp.size == 0:
                continue
            s = sph[grp]
            rotated = s @ rot_mat.T
            perm[grp] = grp[cKDTree(rotated).query(s)[1]]
        return perm            # midline parcels keep their own value

    return permute, {"hemisphere": "preserve", "n_midline": int(mid.size),
                     "n_left": int(left.size), "n_right": int(right.size)}


def spin_null(
    map_a,
    map_b,
    coords,
    *,
    n_trials: int = 1000,
    seed: int = 0,
    hemisphere: str = "whole",
    hemi_labels=None,
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
        NaNs are allowed; only entries finite in both are used. ``map_b`` is the
        map that gets spun.
    coords : array (n_parcels, 3)
        Parcel centroid coordinates (e.g. MNI mm). Centred + projected to a unit
        sphere internally.
    n_trials : int
        Number of random rotations.
    hemisphere : {"whole", "preserve"}
        ``"whole"`` rotates every centroid together about the global centroid.
        It preserves spatial autocorrelation and the shape of the parcellation,
        and it allows a parcel to inherit a value from the other hemisphere.
        ``"preserve"`` rotates each hemisphere about its own centroid, the left
        by Q and the right by MQM with M = diag(-1, 1, 1), so the two rotations
        are mirror images and every assignment stays inside its hemisphere. It
        preserves spatial autocorrelation, hemisphere membership and left/right
        symmetry, and it is the calibrated choice for bilaterally symmetric

        maps. The default is ``"whole"``.
    hemi_labels : sequence, optional
        One hemisphere label per parcel, matched on the first letter, e.g. the
        ``hemisphere`` column of a var table. Only used when
        ``hemisphere="preserve"``. Falls back to the sign of the centred x
        coordinate when omitted. Parcels on the midline keep their own value
        and are counted in ``n_midline``.

    Returns
    -------
    dict with observed Pearson r, spin p-value (two-sided on |r|), and the null
    summary. Compare ``p_spin`` against the (over-optimistic) permuted-π p.
    """
    from scipy.stats import pearsonr

    a = np.asarray(map_a, dtype=float)
    b = np.asarray(map_b, dtype=float)
    xyz = np.asarray(coords, dtype=float)
    if not (a.shape == b.shape == (xyz.shape[0],)):
        raise ValueError("map_a, map_b, coords must share n_parcels")

    permute, info = _spin_permuter(xyz, hemisphere=hemisphere,
                                   hemi_labels=hemi_labels)

    def _corr(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3:
            return np.nan
        return float(pearsonr(x[m], y[m])[0])

    r_obs = _corr(a, b)
    rng = np.random.default_rng(seed)
    null = np.empty(n_trials, dtype=float)
    for t in range(n_trials):
        null[t] = _corr(a, b[permute(rng)])

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
        **info,
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
    hemisphere: str = "whole",
    hemi_labels=None,
) -> dict:
    """Null for a cross-species translation claim.

    Three nulls are available when asking "does routing ``mouse_map`` through π
    predict ``observed_human_map``?", and they test different hypotheses:

      C. Fully shuffle the mouse input (replace it with spatial noise) → tests
         whether a smooth map beats noise. Any smooth brain map clears it,
         because routed noise predicts nothing, so it overstates significance.
      A. Spin the observed human map (`spin_null`) → tests whether two smooth
         maps align beyond spatial autocorrelation. It controls spatial
         autocorrelation but ignores π's structure in the null.
      B. Spin the mouse input and route it through the real π (this function)
         → tests whether this specific mouse spatial pattern, rather than a
         rotated one, predicts the human map through the coupling. It retains
         both the spatial autocorrelation and π, breaking only the
         mouse→human correspondence.

    A and B agree closely, both controlling spatial autocorrelation, while C is
    more lenient. B and A are the nulls reported.

    ``hemisphere`` and ``hemi_labels`` control the rotation of the mouse input
    exactly as in :func:`spin_null`. ``"whole"`` rotates the whole mouse volume
    about its centroid and lets a mouse parcel inherit the value of a parcel in
    the other hemisphere. ``"preserve"`` rotates each mouse hemisphere about its
    own centroid with mirror image rotations Q and MQM, so hemisphere membership

    and left/right symmetry survive and nothing crosses the midline. The
    default is ``"whole"``.

    Returns observed |r|, the spin-B p-value, and the null summary.
    """
    from scipy.stats import pearsonr

    m = np.asarray(mouse_map, dtype=float)
    obs = np.asarray(observed_human_map, dtype=float)
    pi = np.asarray(pi, dtype=float)
    mc = np.asarray(mouse_coords, dtype=float)

    permute, info = _spin_permuter(mc, hemisphere=hemisphere,
                                   hemi_labels=hemi_labels)

    def _corr(x, y):
        ok = np.isfinite(x) & np.isfinite(y)
        return float(pearsonr(x[ok], y[ok])[0]) if ok.sum() >= 3 else np.nan

    r_obs = _corr(_route_normalized(m, pi), obs)
    rng = np.random.default_rng(seed)
    null = np.empty(n_trials, dtype=float)
    for t in range(n_trials):
        null[t] = _corr(_route_normalized(m[permute(rng)], pi), obs)

    absnull = np.abs(null)
    p = (np.sum(absnull >= abs(r_obs)) + 1) / (n_trials + 1)
    return {
        "r_observed": r_obs,
        "p_translation_spin": float(p),
        "n_trials": int(n_trials),
        "null_abs_mean": float(np.nanmean(absnull)),
        "null_abs_p95": float(np.nanpercentile(absnull, 95)),
        **info,
    }
