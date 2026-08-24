"""Periaqueductal-gray regional correspondence entry.

Pair ID 54 links mouse and human PAG targets at gross-structure resolution. Primary source: Ezra et al., Human Brain Mapping (2015), doi:10.1002/hbm.22855."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from otter.data.region_anchors import RegionAnchorEntry
from otter.data.anchor_packs._dsurqe import (
    mouse_parcels_in_dsurqe_region,
    human_parcels_in_mni_sphere,
)


def build_pag_region_anchors(
    M_var: pd.DataFrame,
    H_var: pd.DataFrame,
    *,
    atlas_root: Path | str = ".",
) -> list[RegionAnchorEntry]:
    """Build the periaqueductal gray anchor.

    Returns ``[PAG entry]`` at pid 54.
    """
    pag_mouse = mouse_parcels_in_dsurqe_region(M_var, "Periaqueductal gray", atlas_root)
    pag_human = human_parcels_in_mni_sphere(H_var, (-5, -30, -10), 6.0)

    if not (pag_mouse and pag_human):
        raise ValueError(
            f"empty set for PAG anchor "
            f"(|pag_m|={len(pag_mouse)}, |pag_h|={len(pag_human)})"
        )

    return [
        RegionAnchorEntry(
            pair_id=54,
            label="Periaqueductal gray (Ezra 2015; Kingsbury 2011)",
            mouse_indices=pag_mouse, human_indices=pag_human,
        ),
    ]
