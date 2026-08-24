"""Tectal regional correspondence entries.

Pair IDs 32 and 33 link mouse superior and inferior colliculus parcels to the corresponding human targets. Primary source: Isa et al., Current Biology (2021), doi:10.1016/j.cub.2021.04.001."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from otter.data.region_anchors import RegionAnchorEntry
from otter.data.anchor_packs._dsurqe import (
    mouse_parcels_in_dsurqe_region,
    human_parcels_in_mni_sphere,
)


def build_tectum_region_anchors(
    M_var: pd.DataFrame,
    H_var: pd.DataFrame,
    *,
    atlas_root: Path | str = ".",
) -> list[RegionAnchorEntry]:
    """Build the two tectum region anchors (SC + IC).

    Returns ``[SC entry, IC entry]`` (pid 32 and 33 respectively). Raises
    ``FileNotFoundError`` if the Beauchamp DSURQE atlas isn't present at
    ``{atlas_root}/data_external/MouseHumanTranscriptomicSimilarity/``.
    """
    sc_mouse = mouse_parcels_in_dsurqe_region(
        M_var, "Superior colliculus, sensory related", atlas_root,
    )
    ic_mouse = mouse_parcels_in_dsurqe_region(
        M_var, "Inferior colliculus", atlas_root,
    )
    sc_human = human_parcels_in_mni_sphere(H_var, (-5, -30, -2), 6.0)
    ic_human = human_parcels_in_mni_sphere(H_var, (-5, -35, -8), 8.0)

    if not (sc_mouse and sc_human and ic_mouse and ic_human):
        raise ValueError(
            f"empty set for tectum anchor, check atlas alignment "
            f"(|sc_m|={len(sc_mouse)}, |sc_h|={len(sc_human)}, "
            f"|ic_m|={len(ic_mouse)}, |ic_h|={len(ic_human)})"
        )

    return [
        RegionAnchorEntry(
            pair_id=32, label="Superior Colliculus (Isa 2021)",
            mouse_indices=sc_mouse, human_indices=sc_human,
        ),
        RegionAnchorEntry(
            pair_id=33, label="Inferior Colliculus (Winer & Schreiner 2005)",
            mouse_indices=ic_mouse, human_indices=ic_human,
        ),
    ]
