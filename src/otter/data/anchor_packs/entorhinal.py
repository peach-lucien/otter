"""Entorhinal regional correspondence entry.

Pair ID 49 links mouse entorhinal parcels to a human entorhinal MNI target. Primary source: Franjic et al., Neuron (2022), doi:10.1016/j.neuron.2021.10.036."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from otter.data.region_anchors import RegionAnchorEntry
from otter.data.anchor_packs._dsurqe import (
    mouse_parcels_in_dsurqe_region,
    human_parcels_in_mni_sphere,
)


def build_entorhinal_region_anchors(
    M_var: pd.DataFrame,
    H_var: pd.DataFrame,
    *,
    atlas_root: Path | str = ".",
) -> list[RegionAnchorEntry]:
    """Build the single entorhinal region anchor.

    Returns ``[Entorhinal entry]`` at pid 49.
    """
    mouse_idx = mouse_parcels_in_dsurqe_region(M_var, "Entorhinal area", atlas_root)
    human_idx = human_parcels_in_mni_sphere(H_var, (-20, -10, -30), 10.0)

    if not (mouse_idx and human_idx):
        raise ValueError(
            f"empty set for entorhinal anchor "
            f"(|mouse|={len(mouse_idx)}, |human|={len(human_idx)})"
        )

    return [
        RegionAnchorEntry(
            pair_id=49,
            # This label is a stored key that appears in logs. Crossref dates
            # the paper 2022; see the reference above.
            label="Entorhinal cortex (Franjic 2021)",
            mouse_indices=mouse_idx, human_indices=human_idx,
        ),
    ]
