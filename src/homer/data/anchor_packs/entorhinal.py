"""Entorhinal cortex anchor pack (Franjic et al. 2021).

The entorhinal cortex (EC) is the major gateway between hippocampus and
neocortex, with conserved cytoarchitecture and connectivity across mouse,
macaque, pig, and human. Franjic et al. (2021, *Neuron*) profiled
single-nucleus transcriptomes across the hippocampal-entorhinal system
in three species and identified subregion-specific cell types and
transitional changes from the three-layered archicortex to the
six-layered neocortex.

  pid 49: Mouse Entorhinal area ↔ Human entorhinal cortex

This is a single-entry pack because DSURQE exposes "Entorhinal area" (84
parcels) and "Entorhinal area, lateral part" (33 parcels) but NOT a
distinct "medial part" label, so we can't cleanly split into the lateral
EC (object/contextual memory) vs medial EC (spatial / grid-cell) subdivisions
that Franjic 2021 and Ohara et al. 2021 distinguish. The whole-entorhinal
anchor still captures the broad EC↔EC homology, and is non-conflicting
with all existing packs (entorhinal is anatomically distinct from any
other anchored region).

Mouse-side: DSURQE atlas overlay.
  Entorhinal area: 84 parcels

Human-side: MNI sphere at the anterior medial temporal lobe entorhinal
centroid (Mai/Paxinos):
  Entorhinal cortex: (±20, –10, –30) r=10 mm → 6 parcels

References (verified Consensus search 2026):
  - Franjic, D. et al. (2021). Transcriptomic taxonomy and neurogenic
    trajectories of adult human, macaque, and pig hippocampal and
    entorhinal cells. *Neuron* 110, 452-469.e14.
    DOI: 10.1016/j.neuron.2021.10.036. (222 citations).
  - Ohara, S. et al. (2021). Laminar Organization of the Entorhinal Cortex
    in Macaque Monkeys Based on Cell-Type-Specific Markers and Connectivity.
    *Frontiers in Neural Circuits* 15:649571.
    DOI: 10.3389/fncir.2021.649571. (15 citations). Companion paper for
    laminar correspondences.

Future extension
----------------
If DSURQE adds the "Entorhinal area, medial part" label in a future atlas
release, this pack can be extended to two entries (lateral EC ↔ anterolateral
human EC; medial EC ↔ posteromedial human EC) along the lines proposed by
Ohara 2021. The pid range 49-50 is reserved for that.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from homer.data.region_anchors import RegionAnchorEntry
from homer.data.anchor_packs._dsurqe import (
    mouse_parcels_in_dsurqe_region,
    human_parcels_in_mni_sphere,
)


def build_entorhinal_region_anchors(
    M_var: pd.DataFrame,
    H_var: pd.DataFrame,
    *,
    atlas_root: Path | str = ".",
) -> list[RegionAnchorEntry]:
    """Build the single entorhinal region anchor.

    Returns ``[Entorhinal entry]`` at pid 49.
    """
    mouse_idx = mouse_parcels_in_dsurqe_region(M_var, "Entorhinal area", atlas_root)
    human_idx = human_parcels_in_mni_sphere(H_var, (-20, -10, -30), 10.0)

    if not (mouse_idx and human_idx):
        raise ValueError(
            f"empty set for entorhinal anchor "
            f"(|mouse|={len(mouse_idx)}, |human|={len(human_idx)})"
        )

    return [
        RegionAnchorEntry(
            pair_id=49,
            label="Entorhinal cortex (Franjic 2021)",
            mouse_indices=mouse_idx, human_indices=human_idx,
        ),
    ]
