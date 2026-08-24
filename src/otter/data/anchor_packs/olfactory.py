"""Olfactory regional correspondence entries.

Pair IDs 34 and 35 link mouse piriform and anterior-olfactory-nucleus parcels to the corresponding human targets. Sources: Mori, The Olfactory System (2014); Carlen, Science (2017), doi:10.1126/science.aan8868."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from otter.data.region_anchors import RegionAnchorEntry
from otter.data.anchor_packs._dsurqe import (
    mouse_parcels_in_dsurqe_region,
    human_parcels_in_mni_sphere,
)


def build_olfactory_region_anchors(
    M_var: pd.DataFrame,
    H_var: pd.DataFrame,
    *,
    atlas_root: Path | str = ".",
) -> list[RegionAnchorEntry]:
    """Build the two olfactory region anchors (Piriform + AON).

    Returns ``[Piriform entry, AON entry]`` (pid 34 and 35 respectively).
    Raises ``FileNotFoundError`` if the Beauchamp DSURQE atlas isn't
    present at ``{atlas_root}/data_external/MouseHumanTranscriptomicSimilarity/``.
    """
    pir_mouse = mouse_parcels_in_dsurqe_region(M_var, "Piriform area", atlas_root)
    aon_mouse = mouse_parcels_in_dsurqe_region(
        M_var, "Anterior olfactory nucleus", atlas_root,
    )
    pir_human = human_parcels_in_mni_sphere(H_var, (-25,  5, -20), 10.0)
    aon_human = human_parcels_in_mni_sphere(H_var, (-15, 25, -15), 10.0)

    if not (pir_mouse and pir_human and aon_mouse and aon_human):
        raise ValueError(
            f"empty set for olfactory anchor, check atlas alignment "
            f"(|pir_m|={len(pir_mouse)}, |pir_h|={len(pir_human)}, "
            f"|aon_m|={len(aon_mouse)}, |aon_h|={len(aon_human)})"
        )

    return [
        RegionAnchorEntry(
            pair_id=34, label="Piriform cortex (Mori 2014; Carlén 2017)",
            mouse_indices=pir_mouse, human_indices=pir_human,
        ),
        RegionAnchorEntry(
            pair_id=35, label="Anterior olfactory nucleus (Mori 2014)",
            mouse_indices=aon_mouse, human_indices=aon_human,
        ),
    ]
