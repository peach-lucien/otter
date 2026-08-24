"""Perirhinal regional correspondence entry.

Pair ID 55 links mouse and human perirhinal-cortex targets. Primary source: Burwell et al., Hippocampus (1995), doi:10.1002/hipo.450050503."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from otter.data.region_anchors import RegionAnchorEntry
from otter.data.anchor_packs._dsurqe import (
    mouse_parcels_in_dsurqe_region,
    human_parcels_in_mni_sphere,
)


def build_perirhinal_region_anchors(
    M_var: pd.DataFrame,
    H_var: pd.DataFrame,
    *,
    atlas_root: Path | str = ".",
) -> list[RegionAnchorEntry]:
    """Build the perirhinal cortex anchor.

    Returns ``[Perirhinal entry]`` at pid 55.
    """
    mouse_idx = mouse_parcels_in_dsurqe_region(M_var, "Perirhinal area", atlas_root)
    human_idx = human_parcels_in_mni_sphere(H_var, (-35, -10, -30), 10.0)

    if not (mouse_idx and human_idx):
        raise ValueError(
            f"empty set for perirhinal anchor "
            f"(|mouse|={len(mouse_idx)}, |human|={len(human_idx)})"
        )

    return [
        RegionAnchorEntry(
            pair_id=55,
            label="Perirhinal cortex (Burwell 1995)",
            mouse_indices=mouse_idx, human_indices=human_idx,
        ),
    ]
