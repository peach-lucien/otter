"""Motor-cortex regional correspondence entries.

Pair IDs 30 and 31 link mouse M1 and M2 parcels to human BA4 and dorsal premotor targets. Primary source: Bakken et al., Nature (2021), doi:10.1038/s41586-021-03465-8."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from otter.data.region_anchors import RegionAnchorEntry
from otter.data.anchor_packs._dsurqe import (
    mouse_parcels_in_dsurqe_region,
    human_parcels_in_mni_sphere,
)


def build_biccn_motor_region_anchors(
    M_var: pd.DataFrame,
    H_var: pd.DataFrame,
    *,
    atlas_root: Path | str = ".",
) -> list[RegionAnchorEntry]:
    """Build the two BICCN-aligned motor region anchors.

    Returns ``[M1 entry, M2 entry]`` (pid 30 and 31 respectively). Raises
    ``FileNotFoundError`` if the Beauchamp DSURQE atlas isn't present at
    ``{atlas_root}/data_external/MouseHumanTranscriptomicSimilarity/``.
    """
    m1_idx = mouse_parcels_in_dsurqe_region(M_var, "Primary motor area", atlas_root)
    m2_idx = mouse_parcels_in_dsurqe_region(M_var, "Secondary motor area", atlas_root)
    ba4_idx = human_parcels_in_mni_sphere(H_var, (-37, -22, 55), 10.0)
    pmd_idx = human_parcels_in_mni_sphere(H_var, (-28,  -5, 62), 12.0)

    if not (m1_idx and ba4_idx and m2_idx and pmd_idx):
        raise ValueError(
            f"empty set for BICCN motor anchor, check atlas alignment "
            f"(|m1|={len(m1_idx)}, |ba4|={len(ba4_idx)}, "
            f"|m2|={len(m2_idx)}, |pmd|={len(pmd_idx)})"
        )

    return [
        RegionAnchorEntry(
            pair_id=30, label="M1 / BA4 (BICCN, Bakken 2021)",
            mouse_indices=m1_idx, human_indices=ba4_idx,
        ),
        RegionAnchorEntry(
            pair_id=31, label="M2 / PMd Area 6 (BICCN, Bakken 2021)",
            mouse_indices=m2_idx, human_indices=pmd_idx,
        ),
    ]
