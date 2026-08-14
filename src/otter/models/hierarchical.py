"""HierarchicalFGW, per-network sub-FGW solves, assembled into a block-sparse π.

Motivation
----------
The flat FGW solver handles the entire 1864 × 2094 problem at once. Within-network
disambiguation (V1 vs V2, motor vs somato) is a *local* problem that can drown
in the noise of cross-network nodes. Hierarchical OT solves each functional
network in isolation, giving the within-network optimization full attention.

Trade-off: cross-network constraints are lost. In leave-one-network-out CV,
when an entire network is held out, that network's sub-solve has zero anchor
supervision. For *standard* CV where some anchors of every network are visible,
hierarchical can do better on within-network FC translation.

Headline results from the comparison table:
    Anchor CV (LONO):   45% top-1 (HURTS, the held network has no supervision)
    FC translation:    r=0.39 overall, r=0.55 within-network (HELPS)
    Coverage:          787 human nodes kept (vs 1450 for flat. HALVED)

Use this when: full anchor supervision is available AND you care more about
within-network FC accuracy than cross-network coverage.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import ot

from otter.data.anchors import AnchorIndex, get_anchor_index
from otter.data.networks import NETWORKS, assign_networks
from otter.models.base import FGWModel, FitInfo
from otter.models.supervised import _build_xyz_M


def _build_M_subblock(
    M_xyz_sub: np.ndarray,
    idx_m_local: np.ndarray, idx_h_local: np.ndarray,
    pair_ids_m_local: np.ndarray, pair_ids_h_local: np.ndarray,
    hemi_m_local: np.ndarray, hemi_h_local: np.ndarray,
    visible_pair_ids: set[int],
    *, lam_anchor: float = 1.0, xyz_w: float = 0.5,
) -> np.ndarray:
    """Build M for a within-network sub-FGW."""
    M = (xyz_w * M_xyz_sub).astype(np.float64)

    for lm in idx_m_local:
        if int(pair_ids_m_local[lm]) not in visible_pair_ids:
            continue
        target_pid = pair_ids_m_local[lm]
        target_hemi = hemi_m_local[lm]
        match = np.where((pair_ids_h_local[idx_h_local] == target_pid) &
                         (hemi_h_local[idx_h_local] == target_hemi))[0]
        if len(match) == 0:
            continue
        lh = idx_h_local[match[0]]
        M[lm, :] = lam_anchor
        M[lm, lh] = 0.0

    for lh in idx_h_local:
        if int(pair_ids_h_local[lh]) not in visible_pair_ids:
            continue
        target_pid = pair_ids_h_local[lh]
        target_hemi = hemi_h_local[lh]
        match = np.where((pair_ids_m_local[idx_m_local] == target_pid) &
                         (hemi_m_local[idx_m_local] == target_hemi))[0]
        if len(match) == 0:
            continue
        lm = idx_m_local[match[0]]
        M[M[:, lh] < lam_anchor, lh] = lam_anchor
        M[lm, lh] = 0.0
    return M


def hierarchical_semirelaxed_fgw(
    Cm_full: np.ndarray,
    Ch_full: np.ndarray,
    M_xyz_full: np.ndarray,
    idx_m: AnchorIndex,
    idx_h: AnchorIndex,
    var_m, var_h,
    visible_pair_ids,
    *,
    alpha: float = 0.5,
    epsilon: float = 5e-3,
    lam_anchor: float = 1.0,
    xyz_w: float = 0.5,
    max_iter: int = 25,
    tol: float = 1e-5,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """Functional API for hierarchical FGW. Used by HierarchicalFGW class
    and also exported for backward-compatibility with existing scripts.
    """
    visible_set = set(int(p) for p in visible_pair_ids)
    net_m = assign_networks(var_m, idx_m)
    net_h = assign_networks(var_h, idx_h)

    n_m, n_h = Cm_full.shape[0], Ch_full.shape[0]
    pi_full = np.zeros((n_m, n_h), dtype=np.float32)

    pair_ids_m_full = np.zeros(n_m, dtype=np.int32)
    pair_ids_h_full = np.zeros(n_h, dtype=np.int32)
    hemi_m_full = np.array(["?"] * n_m, dtype="<U1")
    hemi_h_full = np.array(["?"] * n_h, dtype="<U1")
    for k, mp in enumerate(idx_m.pos):
        pair_ids_m_full[mp] = idx_m.pair_ids[k]
        hemi_m_full[mp]     = idx_m.hemispheres[k]
    for k, hp in enumerate(idx_h.pos):
        pair_ids_h_full[hp] = idx_h.pair_ids[k]
        hemi_h_full[hp]     = idx_h.hemispheres[k]

    info = {"per_network": {}}

    for net_id, net_name in enumerate(NETWORKS):
        m_nodes = np.where(net_m == net_id)[0]
        h_nodes = np.where(net_h == net_id)[0]
        if len(m_nodes) == 0 or len(h_nodes) == 0:
            if verbose: print(f"  {net_name}: skip (m={len(m_nodes)} h={len(h_nodes)})")
            continue

        Cm_sub  = Cm_full[np.ix_(m_nodes, m_nodes)].astype(np.float64)
        Ch_sub  = Ch_full[np.ix_(h_nodes, h_nodes)].astype(np.float64)
        M_xyz_sub = M_xyz_full[np.ix_(m_nodes, h_nodes)]

        m_anchor_global = np.intersect1d(m_nodes, idx_m.pos)
        h_anchor_global = np.intersect1d(h_nodes, idx_h.pos)
        m_global_to_local = {g: i for i, g in enumerate(m_nodes)}
        h_global_to_local = {g: i for i, g in enumerate(h_nodes)}
        m_anchor_local = np.array([m_global_to_local[g] for g in m_anchor_global], dtype=np.int64)
        h_anchor_local = np.array([h_global_to_local[g] for g in h_anchor_global], dtype=np.int64)

        pair_ids_m_local = pair_ids_m_full[m_nodes]
        pair_ids_h_local = pair_ids_h_full[h_nodes]
        hemi_m_local     = hemi_m_full[m_nodes]
        hemi_h_local     = hemi_h_full[h_nodes]

        M_sub = _build_M_subblock(
            M_xyz_sub, m_anchor_local, h_anchor_local,
            pair_ids_m_local, pair_ids_h_local, hemi_m_local, hemi_h_local,
            visible_set, lam_anchor=lam_anchor, xyz_w=xyz_w,
        )

        p_sub = np.full(len(m_nodes), 1.0 / len(m_nodes), dtype=np.float64)

        try:
            pi_sub, log = ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein(
                M=M_sub, C1=Cm_sub, C2=Ch_sub, p=p_sub,
                alpha=alpha, epsilon=epsilon,
                max_iter=max_iter, tol=tol, log=True,
            )
        except Exception as e:
            if verbose: print(f"  {net_name}: solve failed: {e}")
            continue

        scale = len(m_nodes) / n_m
        pi_full[np.ix_(m_nodes, h_nodes)] = (pi_sub * scale).astype(np.float32)

        n_visible_anchors = sum(1 for k in m_anchor_local
                                 if int(pair_ids_m_local[k]) in visible_set)
        info["per_network"][net_name] = {
            "n_m":              int(len(m_nodes)),
            "n_h":              int(len(h_nodes)),
            "n_anchors_total":  int(len(m_anchor_local)),
            "n_anchors_visible": int(n_visible_anchors),
            "fgw_dist":         float(log.get("srfgw_dist", -1)),
        }
        if verbose:
            print(f"  {net_name:15s} m={len(m_nodes):>4d} h={len(h_nodes):>4d} "
                  f"anchors_visible={n_visible_anchors}/{len(m_anchor_local)} "
                  f"loss={log.get('srfgw_dist', -1):.5f}")

    return pi_full, info


class HierarchicalFGW(FGWModel):
    """Per-network hierarchical FGW.

    Parameters
    ----------
    Same as SupervisedFGW (alpha, epsilon, xyz_weight, lam_anchor, max_iter, tol).

    Pass `holdout_pair_ids=[5, 6]` to fit() to do leave-one-network-out CV
    (where 5+6 are visual). Held-out anchors get no supervision in their
    network sub-solve.
    """

    _name = "HierarchicalFGW"

    def __init__(
        self,
        *,
        alpha: float = 0.5,
        epsilon: float = 5e-3,
        xyz_weight: float = 0.5,
        lam_anchor: float = 1.0,
        max_iter: int = 25,
        tol: float = 1e-5,
        cost_normalisation: str = "max",
    ):
        super().__init__(
            alpha=alpha, epsilon=epsilon, xyz_weight=xyz_weight,
            lam_anchor=lam_anchor, max_iter=max_iter, tol=tol,
            cost_normalisation=cost_normalisation,
        )

    def _solve(self, *, mouse_ad, human_ad,
               holdout_pair_ids: Optional[Sequence[int]] = None,
               Cm: Optional[np.ndarray] = None,
               Ch: Optional[np.ndarray] = None,
               M_xyz: Optional[np.ndarray] = None, **kw):
        from otter.costs.relational import correlation_distance
        from otter.costs.normalisation import normalise_cost

        idx_m = get_anchor_index(mouse_ad.var)
        idx_h = get_anchor_index(human_ad.var)
        all_pairs = set(int(p) for p in idx_m.pair_ids)
        held = set(int(p) for p in (holdout_pair_ids or []))
        visible = sorted(all_pairs - held)

        if Cm is None:
            Cm = normalise_cost(
                correlation_distance(mouse_ad.uns["fc_mean"].astype(np.float64)),
                scheme=self.config["cost_normalisation"],
            )
        if Ch is None:
            Ch = normalise_cost(
                correlation_distance(human_ad.uns["fc_mean"].astype(np.float64)),
                scheme=self.config["cost_normalisation"],
            )
        if M_xyz is None:
            M_xyz = _build_xyz_M(mouse_ad.var, human_ad.var)

        pi, info = hierarchical_semirelaxed_fgw(
            Cm, Ch, M_xyz, idx_m, idx_h, mouse_ad.var, human_ad.var, visible,
            alpha=self.config["alpha"], epsilon=self.config["epsilon"],
            lam_anchor=self.config["lam_anchor"], xyz_w=self.config["xyz_weight"],
            max_iter=self.config["max_iter"], tol=self.config["tol"],
        )
        # Aggregate per-network losses
        per_net = info.get("per_network", {})
        losses = [v.get("fgw_dist", float("nan")) for v in per_net.values()]
        finite = [l for l in losses if np.isfinite(l)]
        agg_loss = float(np.mean(finite)) if finite else float("nan")
        return pi, FitInfo(
            loss=agg_loss, n_iter=self.config["max_iter"], converged=True,
            extra={"per_network": per_net, "n_networks_solved": len(per_net),
                   "n_visible_anchors": len(visible)},
        )
