"""Amygdala regional correspondence entry.

Pair ID 38 links mouse cortical-subplate parcels to a human amygdala MNI sphere. Primary sources: Janak and Tye, Nature (2015), doi:10.1038/nature14188; Pessoa and Adolphs, Nature Reviews Neuroscience (2010), doi:10.1038/nrn2920."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from otter.data.region_anchors import RegionAnchorEntry
from otter.data.anchor_packs._dsurqe import (
    mouse_parcels_in_dsurqe_region,
    human_parcels_in_mni_sphere,
)


def build_amygdala_region_anchors(
    M_var: pd.DataFrame,
    H_var: pd.DataFrame,
    *,
    atlas_root: Path | str = ".",
) -> list[RegionAnchorEntry]:
    """Build the amygdala region anchor.

    Returns ``[Amygdala entry]`` (pid 38).
    """
    mouse_idx = mouse_parcels_in_dsurqe_region(M_var, "Cortical subplate", atlas_root)
    human_idx = human_parcels_in_mni_sphere(H_var, (-25, -5, -20), 8.0)

    if not (mouse_idx and human_idx):
        raise ValueError(
            f"empty set for amygdala anchor, check atlas alignment "
            f"(|mouse|={len(mouse_idx)}, |human|={len(human_idx)})"
        )

    return [
        RegionAnchorEntry(
            pair_id=38,
            label="Amygdala / Cortical subplate (Janak & Tye 2015)",
            mouse_indices=mouse_idx, human_indices=human_idx,
        ),
    ]
