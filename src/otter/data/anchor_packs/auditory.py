"""Auditory regional correspondence entries.

Pair IDs 56 and 57 link mouse primary and secondary auditory parcels to human auditory core and belt targets. Primary sources: Hackett et al., Journal of Comparative Neurology (2001), doi:10.1002/cne.1407; Kaas and Hackett, PNAS (2000), doi:10.1073/pnas.97.22.11793."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from otter.data.region_anchors import RegionAnchorEntry
from otter.data.anchor_packs._dsurqe import (
    mouse_parcels_in_dsurqe_region,
    human_parcels_in_mni_sphere,
)


def build_auditory_region_anchors(
    M_var: pd.DataFrame,
    H_var: pd.DataFrame,
    *,
    atlas_root: Path | str = ".",
) -> list[RegionAnchorEntry]:
    """Build the two auditory subdivision anchors (core + belt).

    Returns ``[A1↔core, A2↔belt]`` at pids 56 and 57.
    """
    a1_mouse = mouse_parcels_in_dsurqe_region(M_var, "Primary auditory area", atlas_root)
    a2d_mouse = mouse_parcels_in_dsurqe_region(M_var, "Dorsal auditory area", atlas_root)
    a2v_mouse = mouse_parcels_in_dsurqe_region(M_var, "Ventral auditory area", atlas_root)
    a2_mouse = sorted(set(a2d_mouse) | set(a2v_mouse))
    a1_human = human_parcels_in_mni_sphere(H_var, (-48, -22, 6), 6.0)
    belt_human = human_parcels_in_mni_sphere(H_var, (-55, -15, 0), 8.0)

    if not (a1_mouse and a1_human and a2_mouse and belt_human):
        raise ValueError(
            f"empty set for auditory anchor "
            f"(|a1_m|={len(a1_mouse)}, |a1_h|={len(a1_human)}, "
            f"|a2_m|={len(a2_mouse)}, |belt_h|={len(belt_human)})"
        )

    return [
        RegionAnchorEntry(
            pair_id=56,
            label="Primary auditory / A1 core (Hackett 2001)",
            mouse_indices=a1_mouse, human_indices=a1_human,
        ),
        RegionAnchorEntry(
            pair_id=57,
            label="Auditory belt (A2 dorsal+ventral; Kaas & Hackett 2000)",
            mouse_indices=a2_mouse, human_indices=belt_human,
        ),
    ]
