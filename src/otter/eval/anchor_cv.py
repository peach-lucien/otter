"""Leave-one-network-out anchor CV.

A model is fit with one functional network's anchor pair_ids withheld
(no supervision), then evaluated on whether it can recover those held-out
anchors via the relational + spatial signal alone.

Public API:
    anchor_loo_cv(model, mouse_ad, human_ad, *, networks=None) -> dict per network
    held_out_metrics_graded, re-export from otter.data.anchors
"""
from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np

from otter.data.anchors import (
    get_anchor_index, held_out_metrics_graded,
)
from otter.data.networks import NETWORKS, PAIRID_TO_NETWORK


def anchor_loo_cv(
    model_factory,
    mouse_ad,
    human_ad,
    *,
    networks: Optional[Sequence[str]] = None,
    fit_kwargs: Optional[dict] = None,
    verbose: bool = False,
) -> dict:
    """Leave-one-network-out CV: for each functional network, withhold its
    anchor pair_ids and re-fit a fresh model.

    Parameters
    ----------
    model_factory : callable returning a fresh FGWModel
        e.g. `lambda: MultimodalFGW(use_sc=True)`. A factory rather than an
        instance, because each fold solves a fresh model.
    mouse_ad, human_ad : AnnData
    networks : list of network names to use as folds (default: all 11)
    fit_kwargs : passed to each model.fit()
    verbose : print per-fold progress

    Returns
    -------
    {
        'per_network': {net_name: {top1, top5, pair_id, mean_rank, mean_xyz_dist, n, elapsed}},
        'weighted': {top1, top5, pair_id, mean_rank, mean_xyz_dist},
    }
    """
    fit_kwargs = fit_kwargs or {}
    networks = networks or NETWORKS
    idx_h = get_anchor_index(human_ad.var)

    # Build network → pair_ids
    net_to_pairs: dict[str, list[int]] = {n: [] for n in NETWORKS}
    for pid, name in PAIRID_TO_NETWORK.items():
        net_to_pairs[name].append(pid)

    per_net: dict[str, dict] = {}
    for net_name in networks:
        held = sorted(net_to_pairs[net_name])
        t = time.time()
        model = model_factory()
        model.fit(mouse_ad, human_ad,
                   holdout_pair_ids=held, **fit_kwargs)
        m = model.evaluate(held_out_pair_ids=held, eval_kind="anchor")
        elapsed = time.time() - t
        per_net[net_name] = {
            "n_anchors_held":  m["n"],
            "n_pair_ids_held": len(held),
            **{k: m[k] for k in ("top1", "top5", "pair_id", "hemisphere",
                                 "mean_rank", "median_rank", "mean_xyz_dist",
                                 "median_xyz_dist") if k in m},
            "elapsed":         round(elapsed, 1),
        }
        if verbose:
            print(f"  {net_name:15s} (n={m['n']}): "
                  f"top1={m['top1']:.0%} top5={m.get('top5', float('nan')):.0%} "
                  f"pair={m['pair_id']:.0%}  ({elapsed:.1f}s)")

    # Weighted aggregate
    weights = np.array([per_net[n]["n_anchors_held"] for n in networks], dtype=float)
    wt = weights.sum()
    weighted = {}
    for k in ("top1", "top5", "pair_id", "hemisphere", "mean_rank", "mean_xyz_dist"):
        vals = np.array([per_net[n].get(k, np.nan) for n in networks])
        if np.isnan(vals).all():
            weighted[k] = float("nan")
        else:
            weighted[k] = float(np.nansum(vals * weights) / wt)

    return {"per_network": per_net, "weighted": weighted}
