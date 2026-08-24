"""Posterior-parietal regional correspondence entry.

Pair ID 61 links mouse posterior parietal association parcels to a human BA7 target. Primary source: Whitlock, Current Biology (2017), doi:10.1016/j.cub.2017.06.007."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from otter.data.region_anchors import RegionAnchorEntry
from otter.data.anchor_packs._dsurqe import (
    mouse_parcels_in_dsurqe_region,
    human_parcels_in_mni_sphere,
)


def build_ppc_region_anchors(
    M_var: pd.DataFrame,
    H_var: pd.DataFrame,
    *,
    atlas_root: Path | str = ".",
) -> list[RegionAnchorEntry]:
    """Build the posterior parietal cortex anchor.

    Returns ``[PPC entry]`` at pid 61.
    """
    mouse_idx = mouse_parcels_in_dsurqe_region(
        M_var, "Posterior parietal association areas", atlas_root,
    )
    human_idx = human_parcels_in_mni_sphere(H_var, (-35, -55, 50), 10.0)

    if not (mouse_idx and human_idx):
        raise ValueError(
            f"empty set for PPC anchor "
            f"(|mouse|={len(mouse_idx)}, |human|={len(human_idx)})"
        )

    return [
        RegionAnchorEntry(
            pair_id=61,
            label="Posterior parietal cortex / BA7 (Whitlock 2017)",
            mouse_indices=mouse_idx, human_indices=human_idx,
        ),
    ]
