"""Within-network compactness — Coletta-style multi-source check.

For each Garin functional network, measures how tightly the predicted human
partners cluster in MNI space. Network-preserving mapping means mouse
parcels in network X end up close to each other in human space.

This is an *internal* multi-source check — it doesn't rely on Beauchamp
validation. Two complementary metrics per network:

  - median_pairwise_distance_mm: median pairwise Euclidean distance among
    predicted human partners (sampled to 50 if larger). Smaller = tighter.
  - mean_centroid_spread_mm: mean distance from each predicted human partner
    to the network's predicted-human centroid. Smaller = tighter.

A "network-coherent" mapping has both metrics small. Comparing two π values
on these metrics reveals whether structural / supervision changes preserve
or degrade within-network coherence.

Citation context
----------------
Coletta et al. 2020 (*Sci Adv*) identified ~7 mouse functional networks
that mirror the canonical human resting-state networks. A good mouse↔human
mapping should map within-network mouse parcels to within-network human
parcels. We use the Garin 21 pair_ids → network assignment from
``homer.data.networks.PAIRID_TO_NETWORK`` as the within-mouse network
labelling (a curated proxy for Coletta-style network membership).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from homer.data.anchors import (
    AnchorIndex, get_anchor_index, assign_parcels_to_nearest_anchor_region,
)
from homer.data.networks import PAIRID_TO_NETWORK


def network_compactness(
    pi: np.ndarray,
    M_var: pd.DataFrame,
    H_var: pd.DataFrame,
    *,
    anchor_index: Optional[AnchorIndex] = None,
    sample_size: int = 50,
    rng_seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Per-network compactness of predicted human partners.

    Parameters
    ----------
    pi : (n_m, n_h) ndarray
        Coupling matrix.
    M_var, H_var : DataFrame
        Must have x/y/z columns and the standard HOMER anchor metadata.
    anchor_index : AnchorIndex (optional)
        Defaults to ``get_anchor_index(M_var)``.
    sample_size : int, default 50
        Pairwise-distance computation is O(k^2); subsample networks larger
        than this to keep runtime predictable.

    Returns
    -------
    {network_name: {
        n_mouse:                        # parcels in this network
        median_pairwise_dist_mm:        # tight = coherent
        mean_centroid_spread_mm:        # tight = coherent
    }}
    """
    if anchor_index is None:
        anchor_index = get_anchor_index(M_var)
    rng = np.random.default_rng(rng_seed)

    mouse_pid = assign_parcels_to_nearest_anchor_region(M_var, anchor_index)
    mouse_net = np.array([PAIRID_TO_NETWORK[int(p)] for p in mouse_pid])
    argmax_h = pi.argmax(axis=1)
    h_xyz = H_var[["x", "y", "z"]].to_numpy()

    out = {}
    for net in sorted(set(mouse_net.tolist())):
        mask = mouse_net == net
        h_pts = h_xyz[argmax_h[mask]]
        n_m = int(mask.sum())
        if n_m == 0:
            continue
        if len(h_pts) > sample_size:
            samp = rng.choice(len(h_pts), sample_size, replace=False)
            h_samp = h_pts[samp]
        else:
            h_samp = h_pts
        d = np.linalg.norm(h_samp[:, None, :] - h_samp[None, :, :], axis=-1)
        iu = np.triu_indices(len(h_samp), k=1)
        med_d = float(np.median(d[iu])) if len(iu[0]) else 0.0
        c = h_pts.mean(axis=0)
        spread = float(np.linalg.norm(h_pts - c, axis=1).mean())
        out[net] = {
            "n_mouse":                     n_m,
            "median_pairwise_dist_mm":     med_d,
            "mean_centroid_spread_mm":     spread,
        }
    return out


def compare_network_compactness(
    pi_a: np.ndarray, pi_b: np.ndarray,
    M_var: pd.DataFrame, H_var: pd.DataFrame,
    *,
    label_a: str = "A", label_b: str = "B",
    **kwargs,
) -> dict[str, dict[str, float]]:
    """Side-by-side network compactness for two π values.

    Returns ``{network: {a, b, delta_med, delta_spread}}``.
    Negative delta = pi_b is *more compact* than pi_a on that network.
    """
    cca = network_compactness(pi_a, M_var, H_var, **kwargs)
    ccb = network_compactness(pi_b, M_var, H_var, **kwargs)
    out = {}
    for net in cca:
        if net not in ccb: continue
        a, b = cca[net], ccb[net]
        out[net] = {
            f"{label_a}_med":    a["median_pairwise_dist_mm"],
            f"{label_b}_med":    b["median_pairwise_dist_mm"],
            "delta_med":         b["median_pairwise_dist_mm"] - a["median_pairwise_dist_mm"],
            f"{label_a}_spread": a["mean_centroid_spread_mm"],
            f"{label_b}_spread": b["mean_centroid_spread_mm"],
            "delta_spread":      b["mean_centroid_spread_mm"] - a["mean_centroid_spread_mm"],
            "n_mouse":           a["n_mouse"],
        }
    return out
