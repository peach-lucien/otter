"""Extrastriate visual regional correspondence entry.

Pair ID 52 links mouse lateral visual area LM to a human V2 target. Primary source: Wang and Burkhalter, Journal of Comparative Neurology (2007), doi:10.1002/cne.21286."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from otter.data.region_anchors import RegionAnchorEntry
from otter.data.anchor_packs._dsurqe import (
    mouse_parcels_in_dsurqe_region,
    human_parcels_in_mni_sphere,
)


def build_visual_region_anchors(
    M_var: pd.DataFrame,
    H_var: pd.DataFrame,
    *,
    atlas_root: Path | str = ".",
) -> list[RegionAnchorEntry]:
    """Build the mouse-LM ↔ human-V2 visual extrastriate anchor.

    Returns ``[LM↔V2 entry]`` at pid 52.
    """
    lm_mouse = mouse_parcels_in_dsurqe_region(M_var, "Lateral visual area", atlas_root)
    v2_human = human_parcels_in_mni_sphere(H_var, (-20, -85, 10), 10.0)

    if not (lm_mouse and v2_human):
        raise ValueError(
            f"empty set for visual anchor "
            f"(|lm_m|={len(lm_mouse)}, |v2_h|={len(v2_human)})"
        )

    return [
        RegionAnchorEntry(
            pair_id=52,
            label="Lateral visual area / V2 (Wang & Burkhalter 2007)",
            mouse_indices=lm_mouse, human_indices=v2_human,
        ),
    ]
