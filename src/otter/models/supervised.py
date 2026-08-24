"""Anchor-supervised semirelaxed FGW with an optional spatial cost."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import ot

from otter.costs.relational import correlation_distance
from otter.costs.normalisation import normalise_cost
from otter.data.anchors import get_anchor_index
from otter.models.base import FGWModel, FitInfo
from otter.models._solver import entropic_semirelaxed_fgw_multistart

def _convergence(log, max_iter, tol):
    """Iterations run and whether the step fell to tol, from the POT solver log.

    The solver records the step norm every tenth iteration, so ``log["err"]`` holds one
    entry per ten iterations.
    """
    errs = list(log.get("err", []))
    n_iter = (len(errs) - 1) * 10 if errs else int(max_iter)
    converged = bool(errs) and float(errs[-1]) <= float(tol) and n_iter < int(max_iter)
    return n_iter, converged



def _build_xyz_M(var_m, var_h) -> np.ndarray:
    """Per-species-normalised xyz Euclidean distance, shape (n_m, n_h),
    scaled so values lie roughly in [0, 1].
    """
    def _norm(var):
        xyz = var[["x", "y", "z"]].values.astype(np.float64)
        lo = xyz.min(0, keepdims=True); hi = xyz.max(0, keepdims=True)
        return (xyz - lo) / np.maximum(hi - lo, 1e-9)
    cm = _norm(var_m); ch = _norm(var_h)
    sq_m = (cm * cm).sum(1, keepdims=True)
    sq_h = (ch * ch).sum(1, keepdims=True)
    d2 = sq_m + sq_h.T - 2.0 * cm @ ch.T
    d = np.sqrt(np.clip(d2, 0.0, None))
    return d / max(d.max(), 1e-9)


def _apply_anchor_supervision(
    M: np.ndarray, idx_m, idx_h, visible_pair_ids: Sequence[int],
    *, lam: float = 1.0,
) -> np.ndarray:
    """In-place style: writes forbidden cells (lam) and free cells (0) for
    visible anchor rows/columns. Returns the modified M.
    """
    visible = set(int(p) for p in visible_pair_ids)
    for k, mp in enumerate(idx_m.pos):
        if int(idx_m.pair_ids[k]) in visible:
            M[mp, :] = lam
            M[mp, idx_h.pos[k]] = 0.0
    for k, hp in enumerate(idx_h.pos):
        if int(idx_h.pair_ids[k]) in visible:
            mp_correct = idx_m.pos[k]
            col_mask = M[:, hp] < lam
            M[col_mask, hp] = lam
            M[mp_correct, hp] = 0.0
    return M


class SupervisedFGW(FGWModel):
    """Anchor-supervised semirelaxed FGW with xyz spatial prior.

    Parameters
    ----------
    alpha : float, default 0.5
        FGW mixing weight. 1 = pure GW (relational only), 0 = pure W (M only).
    epsilon : float, default 5e-3
    xyz_weight : float, default 0.5
        Weight of the xyz term in M. Set to 0 to drop the spatial prior.
    lam_anchor : float, default 1.0
        Penalty for forbidden anchor cells.
    use_multistart : bool, default False
        If True, run multistart with n_restarts random inits + anchor warm.
    n_restarts : int, default 4
    cost_normalisation : str, default 'max'
    max_iter : int, default 25
    tol : float, default 1e-5
    """

    _name = "SupervisedFGW"

    def __init__(
        self,
        *,
        alpha: float = 0.5,
        epsilon: float = 5e-3,
        xyz_weight: float = 0.5,
        lam_anchor: float = 1.0,
        use_multistart: bool = False,
        n_restarts: int = 4,
        cost_normalisation: str = "max",
        max_iter: int = 25,
        tol: float = 1e-5,
    ):
        super().__init__(
            alpha=alpha, epsilon=epsilon, xyz_weight=xyz_weight,
            lam_anchor=lam_anchor, use_multistart=use_multistart,
            n_restarts=n_restarts, cost_normalisation=cost_normalisation,
            max_iter=max_iter, tol=tol,
        )

    def _solve(self, *, mouse_ad, human_ad,
               holdout_pair_ids: Optional[Sequence[int]] = None,
               Cm: Optional[np.ndarray] = None,
               Ch: Optional[np.ndarray] = None,
               M_xyz: Optional[np.ndarray] = None, **kw):
        idx_m = get_anchor_index(mouse_ad.var)
        idx_h = get_anchor_index(human_ad.var)

        # Determine which anchors are visible
        all_pairs = set(int(p) for p in idx_m.pair_ids)
        held = set(int(p) for p in (holdout_pair_ids or []))
        visible = sorted(all_pairs - held)

        # Cost matrices
        if Cm is None:
            fc_m = mouse_ad.uns["fc_mean"].astype(np.float64)
            Cm = normalise_cost(correlation_distance(fc_m),
                                 scheme=self.config["cost_normalisation"])
        if Ch is None:
            fc_h = human_ad.uns["fc_mean"].astype(np.float64)
            Ch = normalise_cost(correlation_distance(fc_h),
                                 scheme=self.config["cost_normalisation"])

        # xyz term in M
        if M_xyz is None and self.config["xyz_weight"] != 0:
            M_xyz = _build_xyz_M(mouse_ad.var, human_ad.var)
        M = (self.config["xyz_weight"] * M_xyz).astype(np.float64) \
            if M_xyz is not None else np.zeros((Cm.shape[0], Ch.shape[0]),
                                                dtype=np.float64)

        # Anchor supervision
        M = _apply_anchor_supervision(
            M, idx_m, idx_h, visible, lam=self.config["lam_anchor"],
        )

        # Solve
        n_m = Cm.shape[0]
        p = np.full(n_m, 1.0 / n_m, dtype=np.float64)

        if self.config["use_multistart"]:
            vis_mask = np.array([int(p_) in set(visible) for p_ in idx_m.pair_ids])
            anchor_warm = ((idx_m.pos[vis_mask], idx_h.pos[vis_mask])
                            if vis_mask.any() else None)
            pi, ms_info = entropic_semirelaxed_fgw_multistart(
                M=M, C1=Cm, C2=Ch, p=p,
                alpha=self.config["alpha"], epsilon=self.config["epsilon"],
                max_iter=self.config["max_iter"], tol=self.config["tol"],
                n_random_inits=self.config["n_restarts"],
                anchor_warm=anchor_warm,
            )
            return pi, FitInfo(
                loss=ms_info["best_loss"], n_iter=self.config["max_iter"],
                converged=True, n_restarts=ms_info["n_restarts"],
                extra={"best_init": ms_info["best_init"],
                       "loss_spread": ms_info["loss_spread"],
                       "n_visible_anchors": int(vis_mask.sum())},
            )

        pi, log = ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein(
            M=M, C1=Cm, C2=Ch, p=p,
            alpha=self.config["alpha"], epsilon=self.config["epsilon"],
            max_iter=self.config["max_iter"], tol=self.config["tol"], log=True,
        )
        loss = float(log.get("srfgw_dist", log.get("fgw_dist", float("nan"))))
        n_iter, converged = _convergence(log, self.config["max_iter"], self.config["tol"])
        return pi, FitInfo(loss=loss, n_iter=n_iter, converged=converged,
                           extra={"final_err": float(log["err"][-1]) if log.get("err") else None})
