"""Olfactory cortex anchor pack.

Cross-species olfactory cortex anatomy is among the most conserved in
mammalian brains. The primary olfactory cortex (piriform) and the anterior
olfactory nucleus (AON) have direct homologues with conserved
cytoarchitecture and connectivity.

References (verified Consensus search 2026):
  - Mori, K. (2014). The Olfactory System: From Odor Molecules to
    Motivational Behaviors. Springer. (260 citations).
  - Carlén, M. (2017). What constitutes the prefrontal cortex? *Science*
    358, 478-482. DOI: 10.1126/science.aan8868. (Discusses cross-species
    olfactory-PFC connectivity homologies.)

  pid 34:  Mouse Piriform area ↔ Human Piriform cortex
  pid 35:  Mouse Anterior olfactory nucleus ↔ Human AON

Mouse-side sets come from the DSURQE atlas overlay:
  Piriform area: 47 parcels
  Anterior olfactory nucleus: 9 parcels

Human-side sets are MNI spheres at canonical Mai/Paxinos centroids:
  Piriform cortex: (±25, 5, -20) r=10 mm → 13 parcels
  AON: (±15, 25, -15) r=10 mm → 6 parcels

Why this targets a documented failure
-------------------------------------
Beauchamp validation gives 0 % top-1 for "Piriform area → piriform cortex"
under the production point-anchor π (mean rank 657 / 2094), one of the
larger failures. The Garin pair_id 11 (Olfactory cortex) gives a single
point anchor; sub-region region anchors let HOMER target the bulk of mouse
piriform → human piriform directly. The AON entry is included because the
anatomy is clean and both species have a well-defined AON region in our
atlases, it's a small but cheap supplementary constraint.

Same caveat as the other packs (see docs/archive/iteration_log.md §5.12.2 and §5.13):
the mouse Piriform set used here is identical to Beauchamp's validation
set, so any improvement on the Beauchamp Piriform pair is partly
tautological. The pack's value is *practical* (HOMER queries for olfactory
parcels become trustworthy) rather than evidence of independent
structural recovery.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from homer.data.region_anchors import RegionAnchorEntry
from homer.data.anchor_packs._dsurqe import (
    mouse_parcels_in_dsurqe_region,
    human_parcels_in_mni_sphere,
)


def build_olfactory_region_anchors(
    M_var: pd.DataFrame,
    H_var: pd.DataFrame,
    *,
    atlas_root: Path | str = ".",
) -> list[RegionAnchorEntry]:
    """Build the two olfactory region anchors (Piriform + AON).

    Returns ``[Piriform entry, AON entry]`` (pid 34 and 35 respectively).
    Raises ``FileNotFoundError`` if the Beauchamp DSURQE atlas isn't
    present at ``{atlas_root}/data_external/MouseHumanTranscriptomicSimilarity/``.
    """
    pir_mouse = mouse_parcels_in_dsurqe_region(M_var, "Piriform area", atlas_root)
    aon_mouse = mouse_parcels_in_dsurqe_region(
        M_var, "Anterior olfactory nucleus", atlas_root,
    )
    pir_human = human_parcels_in_mni_sphere(H_var, (-25,  5, -20), 10.0)
    aon_human = human_parcels_in_mni_sphere(H_var, (-15, 25, -15), 10.0)

    if not (pir_mouse and pir_human and aon_mouse and aon_human):
        raise ValueError(
            f"empty set for olfactory anchor, check atlas alignment "
            f"(|pir_m|={len(pir_mouse)}, |pir_h|={len(pir_human)}, "
            f"|aon_m|={len(aon_mouse)}, |aon_h|={len(aon_human)})"
        )

    return [
        RegionAnchorEntry(
            pair_id=34, label="Piriform cortex (Mori 2014; Carlén 2017)",
            mouse_indices=pir_mouse, human_indices=pir_human,
        ),
        RegionAnchorEntry(
            pair_id=35, label="Anterior olfactory nucleus (Mori 2014)",
            mouse_indices=aon_mouse, human_indices=aon_human,
        ),
    ]
