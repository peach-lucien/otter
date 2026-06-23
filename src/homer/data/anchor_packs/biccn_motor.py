"""BICCN motor cortex anchor pack (Bakken et al. 2021).

The BICCN Motor Cortex Consortium (Bakken et al. 2021, *Nature*) identified
two strongly conserved mouse↔human motor sub-region homologies via
cross-species single-cell transcriptomics.

Reference (verified Consensus search 2026):
  Bakken, T. E. et al. (2021). Comparative cellular analysis of motor cortex
  in human, marmoset and mouse. *Nature* 598, 111-119.
  DOI: 10.1038/s41586-021-03465-8.

  pid 30:  Mouse Primary motor area (M1) ↔ Human Area 4 / BA4 (primary motor)
  pid 31:  Mouse Secondary motor area (M2) ↔ Human Area 6 dorsal premotor (PMd)

Mouse-side sets come from the DSURQE atlas overlay (53 parcels for M1,
48 for M2). Human-side sets come from MNI spheres around canonical
cytoarchitectural centroids (Mayka 2006; Glasser HCP-MMP360): BA4 at
(±37, -22, 55) r=10 mm → 12 parcels; PMd at (±28, -5, 62) r=12 mm → 23
parcels.

Caveat (docs/archive/iteration_log.md §5.12): the mouse M1 set is identical to the
set used by Beauchamp 2022's "Primary motor area → precentral gyrus"
validation, and the BA4 human set is a subset of Beauchamp's precentral
ball. Top-1 = 100 % after fitting is largely *tautological*; the held-out
control (M2 anchor only, M1 omitted) gives Motor top-1 = 0 %, structure
does NOT independently recover M1 ↔ BA4. The pack is useful as a
practical mechanism (HOMER queries for motor parcels return defensible
BA4-centred answers), not as evidence of unsupervised cross-species
recovery.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from homer.data.region_anchors import RegionAnchorEntry
from homer.data.anchor_packs._dsurqe import (
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
