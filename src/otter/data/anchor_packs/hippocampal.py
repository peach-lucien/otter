"""Hippocampal regional correspondence entries.

Pair IDs 39 to 42 link mouse subiculum, CA1, CA3 and dentate-gyrus parcels to the corresponding human subfield targets. Primary sources: Strange et al., Nature Reviews Neuroscience (2014), doi:10.1038/nrn3785; Iglesias et al., NeuroImage (2015), doi:10.1016/j.neuroimage.2015.04.042."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from otter.data.region_anchors import RegionAnchorEntry
from otter.data.anchor_packs._dsurqe import (
    mouse_parcels_in_dsurqe_region,
    human_parcels_in_mni_sphere,
)


# Subfield specs: (mouse DSURQE region name, human MNI centroid_xL, y, z, radius_mm)
_SUBFIELDS = [
    ("Subiculum",      "Subiculum (Strange 2014)",     -22, -32,  -8, 8.0),
    ("Field CA1",      "CA1 (Strange 2014)",           -30, -25, -10, 8.0),
    ("Field CA3",      "CA3 (Strange 2014)",           -25, -22, -10, 8.0),
    ("Dentate gyrus",  "Dentate gyrus (Strange 2014)", -25, -28, -10, 8.0),
]


def build_hippocampal_region_anchors(
    M_var: pd.DataFrame,
    H_var: pd.DataFrame,
    *,
    atlas_root: Path | str = ".",
) -> list[RegionAnchorEntry]:
    """Build the four hippocampal subfield region anchors.

    Returns ``[Subiculum, CA1, CA3, Dentate gyrus]`` at pids 39-42.
    """
    out: list[RegionAnchorEntry] = []
    for i, (dsurqe_name, label, xL, y, z, r) in enumerate(_SUBFIELDS):
        mouse_idx = mouse_parcels_in_dsurqe_region(M_var, dsurqe_name, atlas_root)
        human_idx = human_parcels_in_mni_sphere(H_var, (xL, y, z), r)
        if not mouse_idx or not human_idx:
            raise ValueError(
                f"empty set for hippocampal anchor {label!r}: "
                f"|mouse|={len(mouse_idx)}, |human|={len(human_idx)}"
            )
        out.append(RegionAnchorEntry(
            pair_id=39 + i, label=label,
            mouse_indices=mouse_idx, human_indices=human_idx,
        ))
    return out
