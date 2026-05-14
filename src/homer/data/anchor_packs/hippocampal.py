"""Hippocampal subfield anchor pack.

Cross-species correspondence of hippocampal subfields with conserved
laminar organisation between rodent and primate (verified Consensus
search 2026):

  - Strange, B. A., Witter, M. P., Lein, E. S., & Moser, E. I. (2014).
    Functional organization of the hippocampal longitudinal axis. *Nature
    Reviews Neuroscience* 15, 655-669. DOI: 10.1038/nrn3785.
    (1503 citations).
  - Iglesias, J. E. et al. (2015). A computational atlas of the hippocampal
    formation using ex vivo, ultra-high resolution MRI. *NeuroImage* 115,
    117-137. DOI: 10.1016/j.neuroimage.2015.04.042. (1139 citations).
    Provides the MNI subfield centroids used for the human balls.

  pid 39:  Subiculum
  pid 40:  CA1
  pid 41:  CA3
  pid 42:  Dentate gyrus

We skip CA2 (not present in our DSURQE tree — its parcels likely fall
into CA3 in this atlas resolution).

Why this is HOMER's most-tested failure region
----------------------------------------------
All five hippocampal subfields show 0 % Beauchamp top-1 under production
point-anchor π — the cleanest documented evidence that HOMER's FC/SC
structure does NOT recover hippocampal homology unsupervised. Earlier
work (EXP-1 / SPLIT-1, docs/results §5.2) added four hippocampal *point*
anchors and moved 3 of 4 from 0 → 7-9 % top-1. This pack is the
region-anchor analogue: each subfield's full DSURQE parcel set forced to
map into the matching human subfield MNI ball.

Mouse-side: DSURQE atlas overlay.
  Subiculum: 29 parcels
  Field CA1: 15 parcels
  Field CA3: 26 parcels
  Dentate gyrus: 22 parcels

Human-side: MNI spheres at canonical hippocampal subfield centroids
(Iglesias et al. 2015 *NeuroImage* hippocampal subfield atlas):

  Subiculum:     (±22, -32, -8) r=8 mm  →  8 parcels
  CA1:           (±30, -25, -10) r=8 mm →  6 parcels
  CA3:           (±25, -22, -10) r=8 mm →  4 parcels
  Dentate gyrus: (±25, -28, -10) r=8 mm →  4 parcels

Honest caveat: as with all other anchor packs, the mouse-side sets here
are identical to Beauchamp's validation sets, so Beauchamp top-1 → 100 %
for each anchored subfield is largely tautological. The pack's value is
practical (HOMER queries for hippocampal subfield parcels become
trustworthy). Independent confirmation would require non-Beauchamp
validation (e.g. Iglesias 2015 hippocampal subfield atlas tracing).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from homer.data.region_anchors import RegionAnchorEntry
from homer.data.anchor_packs._dsurqe import (
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
