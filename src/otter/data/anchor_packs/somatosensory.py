"""Somatosensory body-map anchor pack (Penfield 1937; Roe et al. 2007).

Primary somatosensory cortex (S1, BA3b) has a conserved somatotopic
organization across mammals: a body map with foot/leg medial, trunk adjacent,
hand/forelimb mid-lateral, and face/lips most lateral. Penfield 1937
established this in humans; electrophysiology and fMRI report a similar
organization in rodents, non-human primates and other mammals (Freire 2024;
Seelke 2012; Gordon 2023).

  pid 58: Mouse Barrel field + Nose S1 ↔ Human Face S1 (BA3b ventral)
  pid 59: Mouse Upper limb S1 ↔ Human Hand S1 (BA3b mid)
  pid 60: Mouse Lower limb S1 ↔ Human Leg S1 (BA3b medial / paracentral)

The mouse Barrel field is the vibrissa (whisker) representation, a
specialised face-related sensory map. Cross-species, the barrel field
maps onto the primate face/lip region of S1 (the mouse whisker
representation occupies the cortical space that primate face S1 occupies).
Barrel field and nose are combined for the face anchor.

The mouse trunk has 1 DSURQE parcel, too small to anchor; the trunk
representation falls under the "Primary somatosensory area" anchor (Garin
pid 3) and is not split out here.

Mouse-side: DSURQE atlas overlay.
  Barrel field: 61 parcels (whisker representation)
  Nose: 27 parcels (rostral face S1)
  Upper limb: 24 parcels (forelimb)
  Lower limb: 14 parcels (hindlimb)

Human-side: MNI spheres at Penfield-aligned body-map centroids.
  Face S1 (BA3b ventral):  (±55, –15, 25) r=8 mm
  Hand S1 (BA3b mid):      (±40, –25, 55) r=8 mm
  Leg S1 (paracentral):    (±10, –40, 70) r=10 mm

Empirical effect, opt-in
-------------------------
Beauchamp validates "Primary somatosensory area → postcentral gyrus" using all
155 mouse S1 parcels onto a single broad postcentral ball centred at hand S1
(±40, –25, 55) r=15. This pack splits 126 of those 155 parcels into 3
body-map-specific human balls (face at ±55, hand at ±40, leg at ±10). The face
S1 ball at (±55, –15, 25) lies outside Beauchamp's r=15 validation ball, at a
distance of about 35 mm, so mass concentrated there counts as a Beauchamp miss.
Beauchamp S1 drops from 20 % to 15 %.

The anchor target differs from the validation target, as in the cingulate pack.
The pack ships as opt-in because it lowers the Beauchamp metric, and applies
where body-map-specific S1 queries matter more than Beauchamp's broad-ball
validation.

References:
  - Roe, A. W., Chen, L. M., & Friedman, R. M. (2007). Somatosensory
    cortex from a comparative perspective. *Trends in Neurosciences* /
    *Behav Brain Res* (cross-species somatotopic organization).
  - Seelke, A. M. H. et al. (2012). The Emergence of Somatotopic Maps
    of the Body in S1 in Rats. *PLOS ONE* 7, e32322. (77 citations).
    DOI: 10.1371/journal.pone.0032322.
  - Gordon, E. M. et al. (2023). A somato-cognitive action network
    alternates with effector regions in motor cortex. *Nature* 617,
    351-359. DOI: 10.1038/s41586-023-05964-2. (345 citations).
    Confirms cross-species somatotopic alignment.
  - Freire, M. et al. (2024). Organization of Somatosensory Cortex in
    the South American Rodent Paca. *Brain, Behavior and Evolution*.
    Cross-species somatotopy.
"""
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
