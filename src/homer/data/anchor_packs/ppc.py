"""Posterior parietal cortex (PPC) anchor pack (Whitlock 2017).

Mouse posterior parietal cortex has documented cross-species homology
to primate PPC (BA7), serving similar functions in spatial attention,
navigation, and sensorimotor integration. The Allen mouse atlas
"Posterior parietal association areas" corresponds anatomically to
the PPC region described in cross-species reviews.

  pid 61: Mouse Posterior parietal association ↔ Human PPC BA7

Garin pid 4 (Posterior parietal) gives a single point anchor; this
region pack constrains all 10 mouse PPC parcels onto the human BA7
ball.

Mouse-side: DSURQE atlas overlay.
  Posterior parietal association areas: 10 parcels

Human-side: MNI sphere at BA7 centroid (superior parietal lobule).
  PPC BA7: (±35, –55, 50) r=10 mm → 14 parcels

References (verified Consensus search 2026):
  - Whitlock, J. R. (2017). Posterior parietal cortex. *Current
    Biology* 27, R691-R695. DOI: 10.1016/j.cub.2017.06.007.
    Cross-species PPC review.
  - Lyamzin, D. & Benucci, A. (2019). The mouse posterior parietal
    cortex: Anatomy and functions. *Neuroscience Research* 140,
    14-22. DOI: 10.1016/j.neures.2018.10.008. Mouse-specific PPC
    review.

Anatomical-credibility supervision — PPC isn't in Beauchamp's 22
pairs, so direct lift can't be measured. The Garin pid 4 anchor gets
expanded supervision via the 10-parcel region anchor.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from homer.data.region_anchors import RegionAnchorEntry
from homer.data.anchor_packs._dsurqe import (
    mouse_parcels_in_dsurqe_region,
    human_parcels_in_mni_sphere,
)


def build_ppc_region_anchors(
    M_var: pd.DataFrame,
    H_var: pd.DataFrame,
    *,
    atlas_root: Path | str = ".",
) -> list[RegionAnchorEntry]:
    """Build the posterior parietal cortex anchor.

    Returns ``[PPC entry]`` at pid 61.
    """
    mouse_idx = mouse_parcels_in_dsurqe_region(
        M_var, "Posterior parietal association areas", atlas_root,
    )
    human_idx = human_parcels_in_mni_sphere(H_var, (-35, -55, 50), 10.0)

    if not (mouse_idx and human_idx):
        raise ValueError(
            f"empty set for PPC anchor "
            f"(|mouse|={len(mouse_idx)}, |human|={len(human_idx)})"
        )

    return [
        RegionAnchorEntry(
            pair_id=61,
            label="Posterior parietal cortex / BA7 (Whitlock 2017)",
            mouse_indices=mouse_idx, human_indices=human_idx,
        ),
    ]
