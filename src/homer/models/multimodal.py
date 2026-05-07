"""MultimodalFGW — the production model.

Adds optional modalities on top of SupervisedFGW (anchors + xyz):
  - SC (structural connectivity) — mixed into the relational cost C
  - gene-expression GW          — mixed into C as a separate within-species cost
  - M_gene                       — cross-species cosine cost on ortholog vectors
  - M_anchor                     — cross-species cost on anchor-relationship FC features

The headline production config (from the comparison table) is:
    MultimodalFGW(use_sc=True, sc_weight=0.3, fc_weight=0.7,
                  xyz_weight=0.5, lam_anchor=1.0)

This achieves 81% top-1 on leave-one-network-out CV and r=0.36 FC translation
quality. Other modality switches (gene, M_gene, M_anchor) are opt-in for
ablations / experiments — see the comprehensive_table.csv for their per-config
results.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import ot

from homer.costs.crossspecies import (
    cross_species_anchor_M, cross_species_gene_cost,
)
from homer.costs.normalisation import normalise_cost
from homer.costs.relational import (
    correlation_distance, gene_correlation_distance, sc_correlation_distance,
)
from homer.data.anchors import get_anchor_index
from homer.data.networks import assign_networks, network_mismatch_mask
from homer.models._solver import entropic_semirelaxed_fgw_multistart
from homer.models.base import FitInfo
from homer.data.region_anchors import apply_region_supervision
from homer.models.supervised import SupervisedFGW, _apply_anchor_supervision, _build_xyz_M


class MultimodalFGW(SupervisedFGW):
    """Production multimodal semirelaxed FGW.

    Parameters
    ----------
    fc_weight, sc_weight, gene_gw_weight : float
        Mixing weights for the relational (within-species) GW term. Must sum
        to a non-negative value; will be auto-normalised so they sum to 1
        across active modalities (those with weight > 0).
    use_sc : bool, default True
        If True, sc_weight contributes to the relational cost. Requires
        Cm_SC and Ch_SC to be passed to fit() (or precomputed via
        :func:`homer.costs.sc_correlation_distance`).
    use_gene_gw : bool, default False
        If True, mix gene-expression-derived within-species GW cost.
        Requires Cm_gene, Ch_gene to be passed to fit().
    xyz_weight : float, default 0.5
        Weight of xyz term in M.
    network_mask_weight : float, default 0.0
        Per-cell penalty for cross-network pairings in M.
    M_gene_weight : float, default 0.0
        Cross-species gene cost in M (requires M_gene to be passed).
    M_anchor_weight : float, default 0.0
        Anchor-relationship cross-species cost in M. Computed per-fold using
        ONLY the visible anchors (CV-fair).
    selective_M_gene : bool, default False
        If True, mask M_gene to zero where either species lacks ortholog data.
    Other params (alpha, epsilon, lam_anchor, ...) inherited from SupervisedFGW.
    """

    _name = "MultimodalFGW"

    def __init__(
        self,
        *,
        # Relational mixing
        fc_weight: float = 0.7,
        sc_weight: float = 0.3,
        gene_gw_weight: float = 0.0,
        use_sc: bool = True,
        use_gene_gw: bool = False,
        # M-term weights
        xyz_weight: float = 0.5,
        network_mask_weight: float = 0.0,
        M_gene_weight: float = 0.0,
        M_anchor_weight: float = 0.0,
        selective_M_gene: bool = False,
        # Inherit rest from SupervisedFGW defaults
        alpha: float = 0.5,
        epsilon: float = 5e-3,
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
        self.config.update(dict(
            fc_weight=fc_weight, sc_weight=sc_weight, gene_gw_weight=gene_gw_weight,
            use_sc=use_sc, use_gene_gw=use_gene_gw,
            network_mask_weight=network_mask_weight,
            M_gene_weight=M_gene_weight, M_anchor_weight=M_anchor_weight,
            selective_M_gene=selective_M_gene,
        ))

    def _solve(self, *, mouse_ad, human_ad,
               holdout_pair_ids: Optional[Sequence[int]] = None,
               # Pre-built cost matrices — anything not given is computed
               Cm_FC: Optional[np.ndarray] = None,
               Ch_FC: Optional[np.ndarray] = None,
               Cm_SC: Optional[np.ndarray] = None,
               Ch_SC: Optional[np.ndarray] = None,
               Cm_gene: Optional[np.ndarray] = None,
               Ch_gene: Optional[np.ndarray] = None,
               M_xyz: Optional[np.ndarray] = None,
               M_gene: Optional[np.ndarray] = None,
               M_gene_valid: Optional[np.ndarray] = None,
               net_mask: Optional[np.ndarray] = None,
               # Region-anchor supervision (S4): list of RegionAnchorEntry
               # applied to M *after* point-anchor supervision. Each entry
               # forces the supervised mouse parcels to map only to parcels
               # in the supervised human set (lam elsewhere).
               region_anchors: Optional[Sequence] = None,
               # Source marginal — defaults to uniform 1/n_m. Override with a
               # length-n_m probability vector for volume/stability weighting.
               p: Optional[np.ndarray] = None,
               **kw):
        idx_m = get_anchor_index(mouse_ad.var)
        idx_h = get_anchor_index(human_ad.var)
        all_pairs = set(int(p) for p in idx_m.pair_ids)
        held = set(int(p) for p in (holdout_pair_ids or []))
        visible = sorted(all_pairs - held)

        # ---- Build relational cost: weighted combo of FC, SC, gene-GW ----
        if Cm_FC is None:
            Cm_FC = normalise_cost(
                correlation_distance(mouse_ad.uns["fc_mean"].astype(np.float64)),
                scheme=self.config["cost_normalisation"],
            )
        if Ch_FC is None:
            Ch_FC = normalise_cost(
                correlation_distance(human_ad.uns["fc_mean"].astype(np.float64)),
                scheme=self.config["cost_normalisation"],
            )
        weights = {"FC": self.config["fc_weight"]}
        if self.config["use_sc"]:
            if Cm_SC is None or Ch_SC is None:
                raise ValueError("use_sc=True but Cm_SC/Ch_SC not supplied to fit()")
            weights["SC"] = self.config["sc_weight"]
        if self.config["use_gene_gw"]:
            if Cm_gene is None or Ch_gene is None:
                # Try to compute from the AnnData if expressions are present
                raise ValueError(
                    "use_gene_gw=True but Cm_gene/Ch_gene not supplied to fit()"
                )
            weights["gene"] = self.config["gene_gw_weight"]
        # auto-normalise weights to sum to 1 across active modalities
        total = sum(weights.values())
        if total <= 0:
            raise ValueError(f"all relational weights are 0: {weights}")
        weights = {k: v / total for k, v in weights.items()}

        Cm = weights["FC"] * Cm_FC.astype(np.float64)
        Ch = weights["FC"] * Ch_FC.astype(np.float64)
        if "SC" in weights:
            Cm = Cm + weights["SC"] * Cm_SC.astype(np.float64)
            Ch = Ch + weights["SC"] * Ch_SC.astype(np.float64)
        if "gene" in weights:
            Cm = Cm + weights["gene"] * Cm_gene.astype(np.float64)
            Ch = Ch + weights["gene"] * Ch_gene.astype(np.float64)

        # ---- Build M ----
        n_m, n_h = Cm.shape[0], Ch.shape[0]
        M = np.zeros((n_m, n_h), dtype=np.float64)

        if self.config["xyz_weight"] != 0:
            if M_xyz is None:
                M_xyz = _build_xyz_M(mouse_ad.var, human_ad.var)
            M += self.config["xyz_weight"] * M_xyz

        if self.config["network_mask_weight"] != 0:
            if net_mask is None:
                net_m = assign_networks(mouse_ad.var, idx_m)
                net_h = assign_networks(human_ad.var, idx_h)
                net_mask = network_mismatch_mask(net_m, net_h)
            M += self.config["network_mask_weight"] * net_mask.astype(np.float64)

        if self.config["M_gene_weight"] != 0:
            if M_gene is None:
                raise ValueError("M_gene_weight>0 but M_gene not supplied to fit()")
            gene_term = M_gene.astype(np.float64)
            if self.config["selective_M_gene"]:
                if M_gene_valid is None:
                    raise ValueError("selective_M_gene=True but M_gene_valid not supplied")
                valid = M_gene_valid.astype(bool)
                mean_valid = float(gene_term[valid].mean()) if valid.any() else 0.5
                gene_term = np.where(valid, gene_term - mean_valid, 0.0)
            M += self.config["M_gene_weight"] * gene_term

        if self.config["M_anchor_weight"] != 0:
            # CV-fair: only use visible anchors
            visible_set = set(visible)
            visible_local = [k for k, pid in enumerate(idx_m.pair_ids)
                              if int(pid) in visible_set]
            if visible_local:
                pos_m = np.asarray(idx_m.pos)[visible_local]
                pos_h = np.asarray(idx_h.pos)[visible_local]
                fc_m = mouse_ad.uns["fc_mean"]
                fc_h = human_ad.uns["fc_mean"]
                M_anchor = cross_species_anchor_M(fc_m, fc_h, pos_m, pos_h)
                M_anchor = M_anchor / max(float(M_anchor.max()), 1e-9)
                M += self.config["M_anchor_weight"] * M_anchor

        # Anchor supervision (point anchors)
        M = _apply_anchor_supervision(M, idx_m, idx_h, visible,
                                       lam=self.config["lam_anchor"])
        # Region-anchor supervision (S4) — applied after point anchors so
        # region constraints can extend or refine the existing point ones.
        if region_anchors:
            M = apply_region_supervision(
                M, region_anchors, lam=self.config["lam_anchor"])

        # ---- Solve ----
        if p is None:
            p = np.full(n_m, 1.0 / n_m, dtype=np.float64)
        else:
            p = np.asarray(p, dtype=np.float64)
            if p.shape != (n_m,):
                raise ValueError(f"p shape {p.shape} != ({n_m},)")
            if p.sum() < 1e-9 or (p < 0).any():
                raise ValueError("p must be non-negative and sum to >0")
            p = p / p.sum()    # normalise just in case

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
                       "weights": weights,
                       "n_visible_anchors": int(vis_mask.sum())},
            )

        pi, log = ot.gromov.entropic_semirelaxed_fused_gromov_wasserstein(
            M=M, C1=Cm, C2=Ch, p=p,
            alpha=self.config["alpha"], epsilon=self.config["epsilon"],
            max_iter=self.config["max_iter"], tol=self.config["tol"], log=True,
        )
        loss = float(log.get("srfgw_dist", log.get("fgw_dist", float("nan"))))
        return pi, FitInfo(
            loss=loss, n_iter=self.config["max_iter"], converged=True,
            extra={"weights": weights, "n_visible_anchors": len(visible)},
        )
