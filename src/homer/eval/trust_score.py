"""Per-parcel trust score for the production π.

Combines three independent signals to estimate, for each mouse parcel, how
much we should trust its predicted human partner:

1. **Bootstrap argmax stability** (`per_row_stability` from
   `outputs/coupling/bootstrap_aggregate_*.npz`): how consistent is the
   argmax across 40 subject-bootstrap samples?  High → π is reproducible.

2. **Argmax mass concentration** (mass on argmax / mass on whole row):
   how peaked is the row's distribution? A sharp peak means the model is
   confident; a diffuse row means it's torn between many candidates.

3. **FC similarity to nearest mouse anchor** (Pearson r of the parcel's
   mouse-FC profile against the nearest anchor's mouse-FC profile):
   high → the parcel is in the same FC-coherent neighborhood as the anchor,
   so the supervision signal is well-supported by FC structure.

The composite score is in [0, 1] (higher = more trustworthy).  Three tiers
(high / medium / low) are assigned by quantile cuts.

Note: "distance to nearest mouse anchor in mm" is deliberately not used as a
component, it is uninformative because every mouse parcel is within ~4mm of
*some* anchor (the mouse brain is small). Argmax mass concentration is a much
better signal of model confidence.

Usage::

    from homer.eval.trust_score import compute_trust_score
    out = compute_trust_score(M, H, pi, bootstrap_path="outputs/coupling/bootstrap_aggregate_fc_plus_SC.npz")
    out["trust"]    # (n_m,) float in [0, 1]
    out["tier"]     # (n_m,) array of {"high","medium","low"}
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from homer.data.anchors import get_anchor_index


def _nearest_anchor_distance_mm(var, anchor_pos: np.ndarray) -> np.ndarray:
    """Per-parcel min Euclidean distance (mm) to any anchor in the same atlas."""
    xyz = var[["x", "y", "z"]].to_numpy()                    # (n, 3)
    anchor_xyz = xyz[anchor_pos]                              # (n_anchors, 3)
    # (n, n_anchors) pairwise distances
    d = np.linalg.norm(xyz[:, None, :] - anchor_xyz[None, :, :], axis=-1)
    return d.min(axis=1)


def _nearest_anchor_fc_similarity(fc_mean: np.ndarray, anchor_pos: np.ndarray) -> np.ndarray:
    """For each mouse parcel, return the max Pearson r between its FC row
    and any anchor's FC row.

    fc_mean : (n, n) mean functional connectivity matrix.
    anchor_pos : (n_anchors,) positional indices of anchor parcels.
    """
    fc = fc_mean.astype(np.float64)
    # standardize each row (n, n)
    eps = 1e-9
    mu = fc.mean(axis=1, keepdims=True)
    sd = fc.std(axis=1, keepdims=True).clip(min=eps)
    z = (fc - mu) / sd
    # corr(row_i, row_j) = z[i] @ z[j] / n_features
    n_feat = z.shape[1]
    anchor_z = z[anchor_pos]                                  # (n_anchors, n)
    # (n, n_anchors) pairwise correlations
    r = z @ anchor_z.T / n_feat
    return r.max(axis=1)


def _normalise_to_unit(x: np.ndarray, *, lower_pct: float = 5, upper_pct: float = 95) -> np.ndarray:
    """Clip to [p5, p95] then linear-scale to [0, 1]. Robust to outliers."""
    lo = np.percentile(x, lower_pct)
    hi = np.percentile(x, upper_pct)
    if hi <= lo:
        return np.full_like(x, 0.5)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _argmax_concentration(pi: np.ndarray) -> np.ndarray:
    """Per-row mass concentration: pi[i, argmax_i] / sum(pi[i])."""
    row_sum = pi.sum(axis=1).clip(min=1e-12)
    row_max = pi.max(axis=1)
    return row_max / row_sum


def compute_trust_score(
    M_anndata,
    H_anndata,
    pi: np.ndarray,
    *,
    bootstrap_path: Optional[str | Path] = None,
    weights: tuple[float, float, float] = (0.4, 0.4, 0.2),
    tier_quantiles: tuple[float, float] = (0.33, 0.67),
) -> dict:
    """Compute a per-parcel trust score for a mouse → human coupling.

    Parameters
    ----------
    M_anndata, H_anndata : AnnData
        With var columns (x, y, z, garin_anchor) and ``uns['fc_mean']``.
    pi : (n_m, n_h) ndarray
        The coupling matrix to evaluate.
    bootstrap_path : path or None
        Path to a bootstrap_aggregate_*.npz file containing
        ``per_row_stability``. If None, bootstrap component is set to a
        constant 0.5 (uninformative).
    weights : (w_boot, w_concentration, w_fc_sim)
        Convex weights for the three components.
    tier_quantiles : (q_low, q_high)
        Score quantiles defining the {low, medium, high} tier boundaries.

    Returns
    -------
    dict with keys:
        trust         : (n_m,) float in [0, 1]
        tier          : (n_m,) of strings 'low' / 'medium' / 'high'
        bootstrap     : (n_m,) per-row bootstrap stability (0..1)
        concentration : (n_m,) row mass on argmax / row sum
        concentration_norm : (n_m,) ditto, normalised to [0, 1]
        fc_sim        : (n_m,) Pearson r to nearest anchor FC profile
        fc_sim_norm   : (n_m,) ditto, normalized to [0, 1]
        n_anchors     : int, number of anchors used
    """
    assert sum(weights) > 0, "weights must sum to >0"
    weights = np.array(weights, dtype=np.float64)
    weights = weights / weights.sum()

    idx_m = get_anchor_index(M_anndata.var)
    n_m = pi.shape[0]

    # Component 1: bootstrap row-stability
    if bootstrap_path is not None and Path(bootstrap_path).exists():
        b = np.load(bootstrap_path)
        if "per_row_stability" not in b.files:
            raise KeyError(f"per_row_stability not in {bootstrap_path}")
        boot = b["per_row_stability"].astype(np.float64)
    else:
        boot = np.full(n_m, 0.5, dtype=np.float64)
    if boot.shape != (n_m,):
        raise ValueError(f"per_row_stability shape {boot.shape} != ({n_m},)")

    # Component 2: argmax mass concentration (per row)
    concentration = _argmax_concentration(pi)
    concentration_norm = _normalise_to_unit(concentration)

    # Component 3: FC similarity to nearest anchor. Larger = more trust.
    fc_mean = np.asarray(M_anndata.uns["fc_mean"])
    fc_sim = _nearest_anchor_fc_similarity(fc_mean, idx_m.pos)
    fc_sim_norm = _normalise_to_unit(fc_sim)

    # Composite (convex combination of normalised components)
    trust = weights[0] * boot + weights[1] * concentration_norm + weights[2] * fc_sim_norm

    # Tier assignment by quantile
    q_low, q_high = np.percentile(trust, [tier_quantiles[0]*100, tier_quantiles[1]*100])
    tier = np.array(["medium"] * n_m, dtype=object)
    tier[trust < q_low]  = "low"
    tier[trust >= q_high] = "high"

    return {
        "trust":              trust,
        "tier":               tier,
        "bootstrap":          boot,
        "concentration":      concentration,
        "concentration_norm": concentration_norm,
        "fc_sim":             fc_sim,
        "fc_sim_norm":        fc_sim_norm,
        "n_anchors":          int(len(idx_m)),
        "weights":            tuple(weights.tolist()),
        "tier_quantiles":     tier_quantiles,
    }


def compute_multisource_trust(
    M_anndata,
    H_anndata,
    pi: np.ndarray,
    *,
    bootstrap_path: Optional[str | Path] = None,
    region_anchor_entries: Optional[list] = None,
    beauchamp_per_pair: Optional[dict] = None,
    mouse_dsurqe_labels: Optional[np.ndarray] = None,
    beauchamp_region_to_mouse_dsurqe: Optional[dict] = None,
    weights: tuple[float, float, float] = (0.4, 0.4, 0.2),
) -> dict:
    """Multi-source per-parcel trust map (v1).

    Layers external supervision signals (anchor presence, Beauchamp
    region-validation) on top of the existing internal ``compute_trust_score``
    composite. Produces a multi-tier classification:

      ``"anchored_and_validated"``, parcel is in an anchor pack AND its
        Beauchamp region has top-1 > 0
      ``"anchored_only"``, parcel is in an anchor pack but its
        Beauchamp region (if any) is still 0 % or it's outside any
        Beauchamp region
      ``"validated_only"``, parcel is in a Beauchamp region with
        top-1 > 0 but is not in any anchor pack
      ``"structural"``, parcel has high internal trust
        (bootstrap + concentration + FC) but no anchor and no Beauchamp
        validation: pure structural confidence
      ``"low_evidence"``, none of the above

    Parameters
    ----------
    region_anchor_entries : list of RegionAnchorEntry (optional)
        If provided, parcels in any entry's mouse_indices are flagged as
        ``pack_anchored``.
    beauchamp_per_pair : dict (optional)
        Loaded ``outputs/logs/beauchamp_validation_*.json``. Keys are
        "Mouse region -> Human region" strings; values include "top1".
    mouse_dsurqe_labels : (n_m,) ndarray of int (optional)
        Per-parcel DSURQE label. Needed (with ``beauchamp_region_to_mouse_dsurqe``)
        to attach each parcel to its Beauchamp validation pair.
    beauchamp_region_to_mouse_dsurqe : dict (optional)
        {Beauchamp mouse name: set of DSURQE label IDs}. Use
        ``pipeline.05f_beauchamp_validation.parse_dsurqe_tree`` to build.

    Returns
    -------
    dict with all keys from ``compute_trust_score`` plus:
        garin_anchored   : (n_m,) bool, parcel is one of the 42 Garin anchors
        pack_anchored    : (n_m,) bool, in any region_anchor entry
        beauchamp_top1   : (n_m,) float, its Beauchamp pair's top-1 (NaN if N/A)
        evidence_tier    : (n_m,) of strings (5 tiers, see above)
    """
    # Start with the internal composite trust score
    base = compute_trust_score(
        M_anndata, H_anndata, pi,
        bootstrap_path=bootstrap_path, weights=weights,
    )
    n_m = pi.shape[0]

    # ---- Garin anchored flag
    idx_m = get_anchor_index(M_anndata.var)
    garin_anchored = np.zeros(n_m, dtype=bool)
    garin_anchored[idx_m.pos] = True

    # ---- Pack-anchored flag
    pack_anchored = np.zeros(n_m, dtype=bool)
    if region_anchor_entries:
        for e in region_anchor_entries:
            pack_anchored[list(e.mouse_indices)] = True

    # ---- Beauchamp per-parcel top-1 (via mouse-region membership)
    beau_top1 = np.full(n_m, np.nan, dtype=np.float64)
    if (beauchamp_per_pair is not None
            and mouse_dsurqe_labels is not None
            and beauchamp_region_to_mouse_dsurqe is not None):
        # Map each Beauchamp mouse-region name to its top-1
        for pair_str, payload in beauchamp_per_pair.items():
            if pair_str.startswith("_"): continue
            if "skip_reason" in payload: continue
            top1 = payload.get("top1")
            if top1 is None: continue
            mouse_name = pair_str.split(" -> ")[0]
            labels = beauchamp_region_to_mouse_dsurqe.get(mouse_name)
            if not labels: continue
            mask = np.isin(mouse_dsurqe_labels, list(labels))
            beau_top1[mask] = float(top1)

    # ---- Evidence tier
    evidence_tier = np.full(n_m, "low_evidence", dtype=object)
    is_validated = (beau_top1 > 0)
    is_anchored  = garin_anchored | pack_anchored
    is_structural = base["trust"] >= np.percentile(base["trust"], 67)

    evidence_tier[is_anchored & is_validated]  = "anchored_and_validated"
    evidence_tier[is_anchored & ~is_validated] = "anchored_only"
    evidence_tier[~is_anchored & is_validated] = "validated_only"
    evidence_tier[~is_anchored & ~is_validated & is_structural] = "structural"
    # rest stay "low_evidence"

    return {
        **base,
        "garin_anchored": garin_anchored,
        "pack_anchored":  pack_anchored,
        "beauchamp_top1": beau_top1,
        "evidence_tier":  evidence_tier,
    }


def calibrate_trust_against_validation(
    trust: np.ndarray,
    tier: np.ndarray,
    pi: np.ndarray,
    expected_h_indices: dict[int, set[int]],
) -> dict:
    """Calibration: do high-trust parcels actually achieve higher top-1 accuracy?

    Parameters
    ----------
    trust : (n_m,), composite trust score
    tier  : (n_m,), 'low' / 'medium' / 'high'
    pi    : (n_m, n_h), coupling matrix
    expected_h_indices : {mouse_parcel_idx: set of expected human parcel indices}
        Validation ground truth. Only mouse parcels in this dict are scored.

    Returns
    -------
    dict with per-tier {n, top1_accuracy}.
    """
    by_tier = {"low": [], "medium": [], "high": []}
    for m_idx, h_set in expected_h_indices.items():
        argmax_h = int(pi[m_idx].argmax())
        is_correct = argmax_h in h_set
        by_tier[tier[m_idx]].append(int(is_correct))
    out = {}
    for t, hits in by_tier.items():
        if not hits:
            out[t] = {"n": 0, "top1_accuracy": float("nan")}
        else:
            out[t] = {"n": len(hits), "top1_accuracy": float(np.mean(hits))}
    return out


def regional_empirical_accuracy(
    parcel_to_region: np.ndarray,
    pi: np.ndarray,
    expected_h_indices: dict[int, set[int]],
) -> dict[str, dict]:
    """Per-region empirical Beauchamp top-1 accuracy for each region.

    This is the "trust signal", for each region, how often does
    the model's argmax fall in the published-correct human region. A
    parcel's trust is then the accuracy of its region.

    Parameters
    ----------
    parcel_to_region : (n_m,) array of strings (region label for each parcel,
        or None / "" for parcels not in any validation region).
    pi : (n_m, n_h)
    expected_h_indices : {mouse_idx: set of correct human idx} (from Beauchamp
        validation pre-processing).

    Returns
    -------
    {region_name: {n, top1_accuracy, parcel_indices}}.
    """
    out = {}
    by_region: dict[str, list[tuple[int, int]]] = {}
    for m_idx, h_set in expected_h_indices.items():
        region = parcel_to_region[m_idx]
        if region in (None, "", b""): continue
        argmax_h = int(pi[m_idx].argmax())
        is_correct = int(argmax_h in h_set)
        by_region.setdefault(str(region), []).append((m_idx, is_correct))

    for region, items in by_region.items():
        idxs = [i for i, _ in items]
        hits = [c for _, c in items]
        out[region] = {
            "n": len(items),
            "top1_accuracy": float(np.mean(hits)) if hits else float("nan"),
            "parcel_indices": idxs,
        }
    return out


def assign_regional_trust(
    n_m: int,
    parcel_to_region: np.ndarray,
    regional_accuracy: dict[str, dict],
    *,
    high_threshold: float = 0.15,
    low_threshold: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each mouse parcel a regional-trust score (the empirical top-1
    of its Beauchamp-validated region) and a tier.

    Parcels not in any validated region get NaN score and 'unknown' tier.

    Parameters
    ----------
    high_threshold : float, top-1 ≥ this → 'high' tier (default 15%).
    low_threshold  : float, top-1 < this → 'low' tier (default 3%, about
        3× chance for typical region size). Between → 'medium'.

    Returns
    -------
    score : (n_m,), empirical accuracy of the parcel's region, NaN if unknown.
    tier  : (n_m,), 'high' / 'medium' / 'low' / 'unknown'.
    """
    score = np.full(n_m, np.nan, dtype=np.float64)
    tier = np.full(n_m, "unknown", dtype=object)
    for region, info in regional_accuracy.items():
        acc = info["top1_accuracy"]
        for idx in info["parcel_indices"]:
            score[idx] = acc
            if acc >= high_threshold:
                tier[idx] = "high"
            elif acc < low_threshold:
                tier[idx] = "low"
            else:
                tier[idx] = "medium"
    return score, tier
