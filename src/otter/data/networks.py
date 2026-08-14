"""Functional network labels and assignment over the 21 Garin anchor pair_ids.

Coarse network groupings, defined from the human-brain functional network
atlases (Yeo 7 / 17, Glasser 360) and mouse-rat homologue work (Stafford 2014
mouse DMN, Grandjean atlases).

Public API:
    NETWORKS, sorted list of unique network names
    PAIRID_TO_NETWORK, dict pair_id (1..21) → network name
    assign_networks(var, idx_anchor) -> int array of network ids per node
    network_mismatch_mask(net_m, net_h) -> bool (n_m, n_h) cross-network mask
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from otter.data.anchors import AnchorIndex


# Curated mapping from anchor pair_id (1..21) → coarse functional network.
# The labels are a prior, not a relabelling of anchor identity.
PAIRID_TO_NETWORK: dict[int, str] = {
    1:  "frontal_dmn",      # Medial PFC
    2:  "sensorimotor",     # Motor & premotor
    3:  "sensorimotor",     # Somatosensory
    4:  "frontoparietal",   # Posterior parietal
    5:  "visual",           # Visual striate (V1)
    6:  "visual",           # Visual extra-striate (V2)
    7:  "auditory",         # Auditory cortex
    8:  "temporal_dmn",     # MIPT
    9:  "salience",         # Insula
    10: "limbic",           # Septum
    11: "olfactory",        # Olfactory cortex
    12: "limbic",           # Periarchicortex
    13: "subcortical",      # Striatum
    14: "subcortical",      # Basal forebrain
    15: "subcortical",      # Pallidum
    16: "salience",         # Claustrum
    17: "limbic",           # Amygdala
    18: "subcortical",      # Hypothalamus
    19: "subcortical",      # Thalamus
    20: "brainstem",        # Pons
    21: "brainstem",        # Tectum
}
NETWORKS: list[str] = sorted(set(PAIRID_TO_NETWORK.values()))


def assign_networks(var: Any, idx_anchor: AnchorIndex) -> np.ndarray:
    """Return an integer network-id array of length len(var). Anchors get the
    network of their pair_id; grid nodes inherit the network of their nearest
    anchor by per-species-normalised xyz distance.
    """
    pair_to_net = {p: NETWORKS.index(PAIRID_TO_NETWORK[p]) for p in PAIRID_TO_NETWORK}
    n = len(var)
    out = np.full(n, -1, dtype=np.int32)

    anchor_pos = idx_anchor.pos
    for k, pos in enumerate(anchor_pos):
        out[pos] = pair_to_net[int(idx_anchor.pair_ids[k])]

    coords = var[["x", "y", "z"]].values.astype(np.float64)
    lo = coords.min(0, keepdims=True); hi = coords.max(0, keepdims=True)
    cn = (coords - lo) / np.maximum(hi - lo, 1e-9)
    anchor_xyz = cn[anchor_pos]
    grid_mask = ~var["garin_anchor"].values
    grid_pos = np.where(grid_mask)[0]
    sq_a = (anchor_xyz**2).sum(1, keepdims=True)
    sq_b = (cn[grid_pos]**2).sum(1, keepdims=True)
    d2 = sq_b + sq_a.T - 2.0 * cn[grid_pos] @ anchor_xyz.T
    nearest_anchor_local = d2.argmin(axis=1)
    for grid_i, k in zip(grid_pos, nearest_anchor_local):
        out[grid_i] = pair_to_net[int(idx_anchor.pair_ids[k])]
    assert (out >= 0).all(), "some nodes unlabelled"
    return out


def network_mismatch_mask(net_m: np.ndarray, net_h: np.ndarray) -> np.ndarray:
    """Boolean (n_m, n_h) mask: True where the two nodes are in different networks.
    Multiply by a penalty λ_net and add to M to bias against cross-network mappings.
    """
    return (net_m[:, None] != net_h[None, :])
