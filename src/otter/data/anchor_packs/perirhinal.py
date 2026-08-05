"""Perirhinal cortex anchor pack (Burwell 1995; Kealy & Commins 2011).

Perirhinal cortex (Brodmann areas 35 and 36 in primates; rostral
rhinal sulcus cortex in rodents) is a medial temporal lobe memory
region with established cross-species homology. Burwell et al. 1995
(*Hippocampus*, 544 cit) is the canonical rat-monkey comparative
neuroanatomy reference; Kealy & Commins 2011 (*Progress in
Neurobiology*, 123 cit) reviews rat perirhinal anatomy/physiology
including primate homology.

  pid 55: Mouse Perirhinal area ↔ Human perirhinal cortex

Completes OTTER's medial temporal lobe coverage: combined with the
hippocampal (Subi/CA1/CA3/DG) and entorhinal packs, all major MTL
memory structures are now anchored.

Mouse-side: DSURQE atlas overlay.
  Perirhinal area: 6 parcels (small but anatomically distinct)

Human-side: MNI sphere at canonical perirhinal centroid (rostral medial
temporal lobe, just lateral to entorhinal cortex):
  Perirhinal cortex: (±35, -10, -30) r=10 mm → 6 parcels

References (verified Consensus search 2026):
  - Burwell, R. D., Witter, M. P., & Amaral, D. G. (1995). Perirhinal
    and postrhinal cortices of the rat: A review of the neuroanatomical
    literature and comparison with findings from the monkey brain.
    *Hippocampus* 5, 390-408. DOI: 10.1002/hipo.450050503.
    (544 citations). Canonical rat-monkey perirhinal homology paper.
  - Kealy, J. & Commins, S. (2011). The rat perirhinal cortex: A review
    of anatomy, physiology, plasticity, and function. *Progress in
    Neurobiology* 93, 522-548. DOI: 10.1016/j.pneurobio.2011.03.002.
    (123 citations).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from otter.data.region_anchors import RegionAnchorEntry
from otter.data.anchor_packs._dsurqe import (
    mouse_parcels_in_dsurqe_region,
    human_parcels_in_mni_sphere,
)


def build_perirhinal_region_anchors(
    M_var: pd.DataFrame,
    H_var: pd.DataFrame,
    *,
    atlas_root: Path | str = ".",
) -> list[RegionAnchorEntry]:
    """Build the perirhinal cortex anchor.

    Returns ``[Perirhinal entry]`` at pid 55.
    """
    mouse_idx = mouse_parcels_in_dsurqe_region(M_var, "Perirhinal area", atlas_root)
    human_idx = human_parcels_in_mni_sphere(H_var, (-35, -10, -30), 10.0)

    if not (mouse_idx and human_idx):
        raise ValueError(
            f"empty set for perirhinal anchor "
            f"(|mouse|={len(mouse_idx)}, |human|={len(human_idx)})"
        )

    return [
        RegionAnchorEntry(
            pair_id=55,
            label="Perirhinal cortex (Burwell 1995)",
            mouse_indices=mouse_idx, human_indices=human_idx,
        ),
    ]
