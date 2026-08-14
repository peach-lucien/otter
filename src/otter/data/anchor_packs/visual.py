"""Mouse extrastriate visual area anchor pack (Wang & Burkhalter 2007).

Wang & Burkhalter 2007 (*J Comp Neurol*, 481 cit) mapped mouse visual cortex
and identified mouse Lateral Visual area (LM) as the homologue of primate V2,
from retinotopic mapping, V1 input patterns and laminar organisation. The
remaining extrastriate areas (AL, AM, P, RL, A) are higher-order visual regions
whose primate homologues are less certain.

  pid 52: Mouse Lateral Visual area (LM) ↔ Human V2

The more debated mappings (AL ↔ V3, AM ↔ V4) are not included; pid 53 is
reserved for them.

Mouse-side: DSURQE atlas overlay.
  Lateral visual area: 9 parcels

Human-side: MNI sphere at canonical V2 / BA18 centroid.
  V2: (±20, –85, 10) r=10 mm → 8 parcels (medial+lateral occipital)

Garin overlap: pid 5 (Visual striate) and pid 6 (Visual extra-striate) are
existing point anchors. Pid 6's mouse-side point parcel lies in the "Visual
extra-striate" region, and the Garin V2 anchor constrains one parcel, while the
LM region anchor constrains 9 LM parcels. The two are compatible, since they
constrain different rows of M.

Reference:
  Wang, Q. & Burkhalter, A. (2007). Area map of mouse visual cortex.
  *Journal of Comparative Neurology* 502, 339-357.
  DOI: 10.1002/cne.21286. (481 citations).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from otter.data.region_anchors import RegionAnchorEntry
from otter.data.anchor_packs._dsurqe import (
    mouse_parcels_in_dsurqe_region,
    human_parcels_in_mni_sphere,
)


def build_visual_region_anchors(
    M_var: pd.DataFrame,
    H_var: pd.DataFrame,
    *,
    atlas_root: Path | str = ".",
) -> list[RegionAnchorEntry]:
    """Build the mouse-LM ↔ human-V2 visual extrastriate anchor.

    Returns ``[LM↔V2 entry]`` at pid 52.
    """
    lm_mouse = mouse_parcels_in_dsurqe_region(M_var, "Lateral visual area", atlas_root)
    v2_human = human_parcels_in_mni_sphere(H_var, (-20, -85, 10), 10.0)

    if not (lm_mouse and v2_human):
        raise ValueError(
            f"empty set for visual anchor "
            f"(|lm_m|={len(lm_mouse)}, |v2_h|={len(v2_human)})"
        )

    return [
        RegionAnchorEntry(
            pair_id=52,
            label="Lateral visual area / V2 (Wang & Burkhalter 2007)",
            mouse_indices=lm_mouse, human_indices=v2_human,
        ),
    ]
