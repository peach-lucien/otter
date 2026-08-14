"""Amygdala anchor pack.

Cross-species amygdala homology is established:

  - Janak, P. H. & Tye, K. M. (2015). From circuits to behaviour in the
    amygdala. *Nature* 517, 284-292. DOI: 10.1038/nature14188.
    (1701 citations).
  - Pessoa, L. & Adolphs, R. (2010). Emotion processing and the amygdala:
    from a 'low road' to 'many roads' of evaluating biological significance.
    *Nature Reviews Neuroscience* 11, 773-783. DOI: 10.1038/nrn2920.
    (1663 citations).

The pack covers the remaining 0 % Beauchamp top-1 pair without dedicated
sub-region supervision, "Cortical subplate-other → amygdala".

  pid 38: Mouse Cortical subplate ↔ Human amygdala

No sub-nuclear pairs
--------------------
The DSURQE atlas tree does not distinguish basolateral, central and lateral
amygdaloid nuclei, carrying only the broader "Cortical subplate" category (54
mouse parcels) and a few entries too small to constrain (Medial amygdalar
nucleus, 6 parcels; Cortical amygdalar area, 2 parcels). The mouse-side set is
therefore the same as Beauchamp's validation set (Cortical subplate, 54
parcels), which makes the Beauchamp top-1 recovery tautological as with the
other packs.

Mouse-side: DSURQE atlas overlay.
  Cortical subplate: 54 parcels

Human-side: MNI sphere at canonical amygdala centroid (Mai/Paxinos).
  (±25, -5, -20) r=8 mm → 6 parcels

Overlap with the olfactory pack
-------------------------------
The amygdala MNI ball at (±25, -5, -20) r=8 captures 2 parcels named
"L/R_Olfactory cortex" that also fall in the olfactory pack's piriform ball at
(±25, +5, -20) r=10; the piriform and amygdala centroids are 10 mm apart in MNI
space. When both packs are composed, the 2 shared parcels receive conflicting
soft constraints. The constraints are soft rather than hard walls, and mass on
those 2 parcels is intermediate between the two anchor targets rather than
concentrated on either.
"""
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
