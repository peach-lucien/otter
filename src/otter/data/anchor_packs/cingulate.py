"""Cingulate regional correspondence entries.

Pair IDs 36 and 37 link mouse ventral anterior cingulate and retrosplenial parcels to human subgenual ACC and retrosplenial targets. Primary source: Vogt et al., Brain Structure and Function (2012), doi:10.1007/s00429-012-0493-3."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from otter.data.region_anchors import RegionAnchorEntry
from otter.data.anchor_packs._dsurqe import (
    mouse_parcels_in_dsurqe_region,
    human_parcels_in_mni_sphere,
)


def build_cingulate_region_anchors(
    M_var: pd.DataFrame,
    H_var: pd.DataFrame,
    *,
    atlas_root: Path | str = ".",
) -> list[RegionAnchorEntry]:
    """Build the two cingulate region anchors (subgenual ACC + RSC).

    Returns ``[ACC entry, RSC entry]`` (pid 36 and 37 respectively).
    """
    acc_mouse = mouse_parcels_in_dsurqe_region(
        M_var, "Anterior cingulate area, ventral part", atlas_root,
    )
    rsc_mouse = mouse_parcels_in_dsurqe_region(M_var, "Retrosplenial area", atlas_root)
    acc_human = human_parcels_in_mni_sphere(H_var, (-5,  10, 35), 10.0)
    rsc_human = human_parcels_in_mni_sphere(H_var, (-15, -55, 10), 10.0)

    if not (acc_mouse and acc_human and rsc_mouse and rsc_human):
        raise ValueError(
            f"empty set for cingulate anchor, check atlas alignment "
            f"(|acc_m|={len(acc_mouse)}, |acc_h|={len(acc_human)}, "
            f"|rsc_m|={len(rsc_mouse)}, |rsc_h|={len(rsc_human)})"
        )

    return [
        RegionAnchorEntry(
            pair_id=36, label="Subgenual ACC / BA24-25 (Vogt 2012)",
            mouse_indices=acc_mouse, human_indices=acc_human,
        ),
        RegionAnchorEntry(
            pair_id=37, label="Retrosplenial cortex / BA29-30 (Vogt 2012)",
            mouse_indices=rsc_mouse, human_indices=rsc_human,
        ),
    ]
