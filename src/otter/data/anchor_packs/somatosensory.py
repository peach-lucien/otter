"""Somatosensory regional correspondence entries.

Pair IDs 58 to 60 link mouse face, upper-limb and lower-limb S1 parcels to human face, hand and leg S1 targets. Primary sources: Penfield and Boldrey (1937); Seelke et al., PLOS ONE (2012), doi:10.1371/journal.pone.0032322."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from otter.data.region_anchors import RegionAnchorEntry
from otter.data.anchor_packs._dsurqe import (
    mouse_parcels_in_dsurqe_region,
    human_parcels_in_mni_sphere,
)


def build_somatosensory_region_anchors(
    M_var: pd.DataFrame,
    H_var: pd.DataFrame,
    *,
    atlas_root: Path | str = ".",
) -> list[RegionAnchorEntry]:
    """Build the three S1 body-map anchors (face, hand, leg).

    Returns ``[Face S1, Hand S1, Leg S1]`` at pids 58, 59, 60.
    """
    barrel = mouse_parcels_in_dsurqe_region(M_var, "Primary somatosensory area, barrel field", atlas_root)
    nose = mouse_parcels_in_dsurqe_region(M_var, "Primary somatosensory area, nose", atlas_root)
    face_mouse = sorted(set(barrel) | set(nose))

    hand_mouse = mouse_parcels_in_dsurqe_region(M_var, "Primary somatosensory area, upper limb", atlas_root)
    leg_mouse = mouse_parcels_in_dsurqe_region(M_var, "Primary somatosensory area, lower limb", atlas_root)

    face_human = human_parcels_in_mni_sphere(H_var, (-55, -15, 25), 8.0)
    hand_human = human_parcels_in_mni_sphere(H_var, (-40, -25, 55), 8.0)
    leg_human = human_parcels_in_mni_sphere(H_var, (-10, -40, 70), 10.0)

    if not (face_mouse and face_human and hand_mouse and hand_human and leg_mouse and leg_human):
        raise ValueError(
            f"empty set in somatosensory pack "
            f"(face_m={len(face_mouse)}, face_h={len(face_human)}, "
            f"hand_m={len(hand_mouse)}, hand_h={len(hand_human)}, "
            f"leg_m={len(leg_mouse)}, leg_h={len(leg_human)})"
        )

    return [
        RegionAnchorEntry(
            pair_id=58,
            label="S1 face (barrel + nose) ↔ BA3b ventral (Penfield; Seelke 2012)",
            mouse_indices=face_mouse, human_indices=face_human,
        ),
        RegionAnchorEntry(
            pair_id=59,
            label="S1 hand (upper limb) ↔ BA3b mid (Penfield)",
            mouse_indices=hand_mouse, human_indices=hand_human,
        ),
        RegionAnchorEntry(
            pair_id=60,
            label="S1 leg (lower limb) ↔ BA3b medial / paracentral (Penfield)",
            mouse_indices=leg_mouse, human_indices=leg_human,
        ),
    ]
