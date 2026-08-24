"""Lateral-prefrontal regional correspondence entries.

Pair ID 45 links mouse lateral orbital cortex to human orbitofrontal cortex and is canonical. Pair ID 46 links prelimbic cortex to dlPFC and is optional because that correspondence is contested. Sources: Wallis, Nature Neuroscience (2011), doi:10.1038/nn.2956; Preuss, Journal of Cognitive Neuroscience (1995), doi:10.1162/jocn.1995.7.1.1; Carlen, Science (2017), doi:10.1126/science.aan8868."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from otter.data.region_anchors import RegionAnchorEntry
from otter.data.anchor_packs._dsurqe import (
    mouse_parcels_in_dsurqe_region,
    human_parcels_in_mni_sphere,
)


def build_lateral_pfc_region_anchors(
    M_var: pd.DataFrame,
    H_var: pd.DataFrame,
    *,
    atlas_root: Path | str = ".",
    include_dlpfc: bool = False,
) -> list[RegionAnchorEntry]:
    """Build the lateral PFC region anchors.

    Returns the OFC anchor (pid 45) only. The Prelimbic↔dlPFC anchor
    (pid 46) is excluded by default: rodent dlPFC homology is disputed
    (Preuss 1995) and is contradicted by the Schaeffer 2020 falsification test,
    so the canonical composition does not assert it. Pass
    ``include_dlpfc=True`` to add it, for example in ablations.
    """
    ofc_mouse = mouse_parcels_in_dsurqe_region(
        M_var, "Orbital area, lateral part", atlas_root,
    )
    ofc_human = human_parcels_in_mni_sphere(H_var, (-25, 35, -15), 10.0)
    if not (ofc_mouse and ofc_human):
        raise ValueError(
            f"empty set for OFC anchor, check atlas alignment "
            f"(|ofc_m|={len(ofc_mouse)}, |ofc_h|={len(ofc_human)})"
        )
    entries = [
        RegionAnchorEntry(
            pair_id=45, label="OFC / BA11-47 (Wallis 2011)",
            mouse_indices=ofc_mouse, human_indices=ofc_human,
        ),
    ]

    if include_dlpfc:
        pl_mouse = mouse_parcels_in_dsurqe_region(M_var, "Prelimbic area", atlas_root)
        dl_human = human_parcels_in_mni_sphere(H_var, (-40, 25, 35), 10.0)
        if not (pl_mouse and dl_human):
            raise ValueError(
                f"empty set for dlPFC anchor, check atlas alignment "
                f"(|pl_m|={len(pl_mouse)}, |dl_h|={len(dl_human)})"
            )
        entries.append(RegionAnchorEntry(
            pair_id=46, label="dlPFC / BA9-46 (Carlén 2017; contested homology)",
            mouse_indices=pl_mouse, human_indices=dl_human,
        ))

    return entries
