"""UnsupervisedGW, plain entropic GW on FC alone.

The simplest possible model: no anchor supervision, no spatial prior, no SC.
Just Gromov-Wasserstein matching of the two species' FC distance matrices.

This is the historical baseline (≈ 14% top-1 with anchors as the eval set,
matching the pre-supervised "Garin only" experiment). Provided here so users
can see what the relational signal alone gives, before any supervision.

Optional multistart (use_multistart=True) reduces local-minima sensitivity.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from homer.costs.relational import correlation_distance
from homer.costs.normalisation import normalise_cost
from homer.models._solver import (
    entropic_gw, entropic_gw_multistart,
)
from homer.models.base import FGWModel, FitInfo


class UnsupervisedGW(FGWModel):
    """Plain entropic GW on FC.

    Parameters
    ----------
    epsilon : float, default 5e-3
        Entropic regularisation strength. Lower = harder solutions.
    max_iter : int, default 1000
    tol : float, default 1e-9
    use_multistart : bool, default False
        If True, run 5 restarts and pick the lowest-loss result.
    n_restarts : int, default 5
        Only used if use_multistart=True.
    cost_normalisation : str, default 'max'
        How to normalise the FC cost matrices before solving. Choices: max,
        mean, median, none. See :func:`homer.costs.normalise_cost`.
    """

    _name = "UnsupervisedGW"

    def __init__(
        self,
        *,
        epsilon: float = 5e-3,
        max_iter: int = 1000,
        tol: float = 1e-9,
        use_multistart: bool = False,
        n_restarts: int = 5,
        cost_normalisation: str = "max",
    ):
        super().__init__(
            epsilon=epsilon, max_iter=max_iter, tol=tol,
            use_multistart=use_multistart, n_restarts=n_restarts,
            cost_normalisation=cost_normalisation,
        )

    def _solve(self, *, mouse_ad, human_ad,
               Cm: Optional[np.ndarray] = None,
               Ch: Optional[np.ndarray] = None, **kw):
        # Build FC cost matrices if not supplied
        if Cm is None:
            fc_m = mouse_ad.uns["fc_mean"].astype(np.float64)
            Cm = correlation_distance(fc_m)
            Cm = normalise_cost(Cm, scheme=self.config["cost_normalisation"])
        if Ch is None:
            fc_h = human_ad.uns["fc_mean"].astype(np.float64)
            Ch = correlation_distance(fc_h)
            Ch = normalise_cost(Ch, scheme=self.config["cost_normalisation"])

        kwargs = dict(
            epsilon=self.config["epsilon"],
            max_iter=self.config["max_iter"],
            tol=self.config["tol"],
        )

        if self.config["use_multistart"]:
            best, all_results = entropic_gw_multistart(
                Cm, Ch, n_restarts=self.config["n_restarts"], **kwargs,
            )
            losses = [r.loss for r in all_results]
            info = FitInfo(
                loss=best.loss, n_iter=best.n_iter, converged=best.converged,
                n_restarts=len(all_results),
                extra={"loss_spread": float(max(losses) - min(losses)),
                       "best_init_seed": int(best.init_seed)},
            )
            return best.pi, info
        else:
            res = entropic_gw(Cm, Ch, **kwargs)
            return res.pi, FitInfo(
                loss=res.loss, n_iter=res.n_iter, converged=res.converged,
            )
