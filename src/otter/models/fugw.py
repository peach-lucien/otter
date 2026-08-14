"""FUGWModel. Fused Unbalanced Gromov–Wasserstein wrapper for cross-species coupling.

Wraps the [Thual et al. 2022 NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/8906cac4ca58dcaf17e97a0486ad57ca-Abstract-Conference.html)
implementation from the ``fugw`` PyPI package as a drop-in alternative to
`MultimodalFGW`. **Comparative method, not a replacement**, fits into the
existing `FGWModel` API so the same anchor CV / FC translation / null
distribution / bootstrap evaluation works without modification.

The semirelaxed solver fixes the mouse marginal at uniform and lets the human
marginal float freely, so it has no incentive to spread mass evenly across
human nodes; held-out anchors then land on non-anchor grid nodes near the
correct anchor rather than on the anchor itself.

FUGW formulates the problem as **unbalanced** in both directions, with two
KL-divergence penalties (`rho_s`, `rho_t`) controlling how strictly each
marginal must match a target. ``rho → ∞`` recovers balanced FGW; ``rho → 0``
recovers fully unconstrained mass. For brain alignment the recommended setting
balances both terms.

Public:
    FUGWModel(*, alpha=0.5, epsilon=5e-3, rho_s=1.0, rho_t=1.0,
              fc_weight=0.7, sc_weight=0.3, use_sc=True, xyz_weight=0.5,
              lam_anchor=1.0, nits_bcd=10, nits_uot=1000)
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from otter.costs.relational import correlation_distance
from otter.costs.normalisation import normalise_cost
from otter.data.anchors import get_anchor_index
from otter.models.base import FGWModel, FitInfo
from otter.models.supervised import _build_xyz_M, _apply_anchor_supervision


def _to_torch(x):
    """Lazy torch conversion. Importing torch is heavy so do it on demand."""
    import torch
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.astype(np.float32))
    return x


class FUGWModel(FGWModel):
    """Fused Unbalanced Gromov–Wasserstein wrapper.

    Drop-in alternative to `MultimodalFGW` for comparative experiments. Uses
    the `fugw` PyPI package's ``FUGWSolver``.

    Parameters
    ----------
    alpha : float, default 0.5
        FGW mixing weight (1 = pure GW, 0 = pure W). Same meaning as in the
        existing semirelaxed model.
    epsilon : float, default 5e-3
        Entropic regularisation strength.
    rho_s : float, default 1.0
        Source-marginal relaxation strength (mouse). High → balanced (forces
        each mouse row to sum to its prescribed mass); low → unconstrained.
        ``rho_s == float('inf')`` reproduces balanced FGW behaviour on the
        source side.
    rho_t : float, default 1.0
        Target-marginal relaxation (human). Setting ``rho_t = float('inf')``
        forces uniform human coverage, the opposite of the semirelaxed
        configuration, which lets the human marginal float entirely free.
    fc_weight, sc_weight, use_sc : as in MultimodalFGW.
    xyz_weight, lam_anchor : as in SupervisedFGW.
    nits_bcd : int, default 10
        Outer block-coordinate-descent iterations.
    nits_uot : int, default 1000
        Inner unbalanced-OT iterations per BCD step.
    cost_normalisation : str, default 'max'
    """

    _name = "FUGWModel"

    def __init__(
        self,
        *,
        alpha: float = 0.5,
        epsilon: float = 5e-3,
        rho_s: float = 1.0,
        rho_t: float = 1.0,
        fc_weight: float = 0.7,
        sc_weight: float = 0.3,
        use_sc: bool = True,
        xyz_weight: float = 0.5,
        lam_anchor: float = 1.0,
        nits_bcd: int = 10,
        nits_uot: int = 1000,
        cost_normalisation: str = "max",
    ):
        super().__init__(
            alpha=alpha, epsilon=epsilon,
            rho_s=rho_s, rho_t=rho_t,
            fc_weight=fc_weight, sc_weight=sc_weight, use_sc=use_sc,
            xyz_weight=xyz_weight, lam_anchor=lam_anchor,
            nits_bcd=nits_bcd, nits_uot=nits_uot,
            cost_normalisation=cost_normalisation,
        )

    def _solve(self, *, mouse_ad, human_ad,
               holdout_pair_ids: Optional[Sequence[int]] = None,
               Cm_FC: Optional[np.ndarray] = None,
               Ch_FC: Optional[np.ndarray] = None,
               Cm_SC: Optional[np.ndarray] = None,
               Ch_SC: Optional[np.ndarray] = None,
               M_xyz: Optional[np.ndarray] = None, **kw):
        # Lazy import, torch + fugw are heavy
        from fugw.solvers.dense import FUGWSolver
        import torch

        cfg = self.config
        idx_m = get_anchor_index(mouse_ad.var)
        idx_h = get_anchor_index(human_ad.var)
        all_pairs = set(int(p) for p in idx_m.pair_ids)
        held = set(int(p) for p in (holdout_pair_ids or []))
        visible = sorted(all_pairs - held)

        # ---- Within-species relational cost (FC, optionally + SC) ----
        if Cm_FC is None:
            Cm_FC = normalise_cost(
                correlation_distance(mouse_ad.uns["fc_mean"].astype(np.float64)),
                scheme=cfg["cost_normalisation"],
            )
        if Ch_FC is None:
            Ch_FC = normalise_cost(
                correlation_distance(human_ad.uns["fc_mean"].astype(np.float64)),
                scheme=cfg["cost_normalisation"],
            )
        weights = {"FC": cfg["fc_weight"]}
        if cfg["use_sc"]:
            if Cm_SC is None or Ch_SC is None:
                raise ValueError("use_sc=True but Cm_SC/Ch_SC not supplied to fit()")
            weights["SC"] = cfg["sc_weight"]
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("all relational weights are 0")
        weights = {k: v / total for k, v in weights.items()}

        Ds = weights["FC"] * Cm_FC.astype(np.float64)
        Dt = weights["FC"] * Ch_FC.astype(np.float64)
        if "SC" in weights:
            Ds = Ds + weights["SC"] * Cm_SC.astype(np.float64)
            Dt = Dt + weights["SC"] * Ch_SC.astype(np.float64)

        # ---- Cross-species feature cost (M = xyz + anchor supervision) ----
        n_m, n_h = Ds.shape[0], Dt.shape[0]
        F = np.zeros((n_m, n_h), dtype=np.float64)
        if cfg["xyz_weight"] != 0:
            if M_xyz is None:
                M_xyz = _build_xyz_M(mouse_ad.var, human_ad.var)
            F = F + cfg["xyz_weight"] * M_xyz
        F = _apply_anchor_supervision(F, idx_m, idx_h, visible, lam=cfg["lam_anchor"])

        # ---- Marginals ----
        ws = np.full(n_m, 1.0 / n_m, dtype=np.float64)
        wt = np.full(n_h, 1.0 / n_h, dtype=np.float64)

        # ---- Solve ----
        solver = FUGWSolver(
            nits_bcd=cfg["nits_bcd"],
            nits_uot=cfg["nits_uot"],
            tol_bcd=1e-7,
            tol_uot=1e-7,
            eval_bcd=1, eval_uot=10,
        )
        result = solver.solve(
            alpha=cfg["alpha"],
            rho_s=float(cfg["rho_s"]),
            rho_t=float(cfg["rho_t"]),
            eps=cfg["epsilon"],
            reg_mode="joint",
            divergence="kl",
            F=_to_torch(F),
            Ds=_to_torch(Ds),
            Dt=_to_torch(Dt),
            ws=_to_torch(ws),
            wt=_to_torch(wt),
            solver="sinkhorn",
            verbose=False,
        )
        # FUGWSolver returns a dict with keys: 'pi' (Tensor), 'gamma' (Tensor),
        # 'duals_pi', 'duals_gamma', 'loss' (dict of component breakdowns),
        # 'loss_val', 'loss_steps', 'loss_times'. The 'loss' subdict has keys
        # 'wasserstein', 'gromov_wasserstein', 'marginal_constraint_dim1',
        # 'marginal_constraint_dim2', 'regularization', 'total'.
        if not isinstance(result, dict) or "pi" not in result:
            raise RuntimeError(f"unexpected FUGWSolver return: {type(result)}")
        pi = result["pi"]
        if hasattr(pi, "detach"):
            pi = pi.detach().cpu().numpy()
        pi = np.asarray(pi, dtype=np.float64)

        loss_dict = result.get("loss", {})
        total = loss_dict.get("total", float("nan")) if isinstance(loss_dict, dict) else float("nan")
        if hasattr(total, "item"):
            total = float(total.item())
        elif hasattr(total, "__iter__"):
            # Could be a list of per-iter losses; take the last
            try:
                total = float(list(total)[-1])
            except Exception:
                total = float("nan")
        else:
            total = float(total) if total is not None else float("nan")

        # Normalise so that mouse rows sum to 1/n_m (the output convention of
        # the other models, so downstream eval code works unchanged)
        row_sums = pi.sum(axis=1, keepdims=True).clip(min=1e-12)
        pi = pi / row_sums / n_m

        return pi, FitInfo(
            loss=total,
            n_iter=cfg["nits_bcd"],
            converged=True,
            extra={"weights": weights,
                    "rho_s": cfg["rho_s"], "rho_t": cfg["rho_t"],
                    "n_visible_anchors": len(visible),
                    "loss_breakdown": {
                        k: float(v.item()) if hasattr(v, 'item') else
                            float(v) if not hasattr(v, '__iter__') else
                            float(list(v)[-1]) if v else float("nan")
                        for k, v in loss_dict.items()
                    } if isinstance(loss_dict, dict) else {}},
        )
