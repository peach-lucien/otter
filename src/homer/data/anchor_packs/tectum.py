"""Tectum (midbrain colliculi) anchor pack.

Both Superior Colliculus and Inferior Colliculus are anatomically homologous
across mouse and primate brains, with conserved layered organisation:

  - SC: Isa et al. 2021, *Current Biology*, "The tectum/superior colliculus
    as the vertebrate solution for spatial sensory integration and action"
    (cross-species SC review across vertebrates).
  - IC: Winer & Schreiner 2005, "The inferior colliculus" (Springer book;
    the canonical IC reference covering cross-species anatomy).

  pid 32:  Mouse Superior Colliculus (sensory) ↔ Human Superior Colliculus
  pid 33:  Mouse Inferior Colliculus            ↔ Human Inferior Colliculus

Mouse-side sets come from the DSURQE atlas overlay (53 parcels for the
sensory SC; 29 for IC). Human-side sets are *tight* MNI spheres at canonical
Mai/Paxinos centroids (these are small brainstem structures, broader balls
would capture unrelated midbrain parcels):

  SC: (±5, -30, -2) r=6 mm  →  2 human parcels
  IC: (±5, -35, -8) r=8 mm  →  4 human parcels

References (verified Consensus search 2026):
  - Isa, T. et al. (2021). The tectum/superior colliculus as the vertebrate
    solution for spatial sensory integration and action. *Current Biology*
    31, R741-R762. DOI: 10.1016/j.cub.2021.04.020.
  - Winer, J. A. & Schreiner, C. E. (Eds.) (2005). *The Inferior Colliculus*.
    Springer. ISBN: 978-0-387-22038-1.

Why this targets a documented failure
-------------------------------------
``docs/archive/diagnostics.md`` calls out tectum as spatially-inverted between
species: mouse SC is dorsal whereas human SC is ventral in MNI space, so
the xyz cross-species cost actively misleads non-anchor tectum parcels.
The Garin pair_id 21 (Tectum) gives a single point anchor for both
colliculi combined; sub-region anchors let HOMER target SC↔SC and IC↔IC
distinctly. Beauchamp validation gives 0 % top-1 for both colliculi under
the production point-anchor π, and we expect this pack to lift those.

Same caveat as biccn_motor: the mouse-side set for each anchor is
identical to the set Beauchamp 2022's validation uses, so any improvement
on the Beauchamp Tectum pairs is partly *tautological*. The pack's value
is practical (HOMER queries for tectum parcels become trustworthy), not
methodological.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from homer.data.region_anchors import RegionAnchorEntry
from homer.data.anchor_packs._dsurqe import (
    mouse_parcels_in_dsurqe_region,
    human_parcels_in_mni_sphere,
)


def build_tectum_region_anchors(
    M_var: pd.DataFrame,
    H_var: pd.DataFrame,
    *,
    atlas_root: Path | str = ".",
) -> list[RegionAnchorEntry]:
    """Build the two tectum region anchors (SC + IC).

    Returns ``[SC entry, IC entry]`` (pid 32 and 33 respectively). Raises
    ``FileNotFoundError`` if the Beauchamp DSURQE atlas isn't present at
    ``{atlas_root}/data_external/MouseHumanTranscriptomicSimilarity/``.
    """
    sc_mouse = mouse_parcels_in_dsurqe_region(
        M_var, "Superior colliculus, sensory related", atlas_root,
    )
    ic_mouse = mouse_parcels_in_dsurqe_region(
        M_var, "Inferior colliculus", atlas_root,
    )
    sc_human = human_parcels_in_mni_sphere(H_var, (-5, -30, -2), 6.0)
    ic_human = human_parcels_in_mni_sphere(H_var, (-5, -35, -8), 8.0)

    if not (sc_mouse and sc_human and ic_mouse and ic_human):
        raise ValueError(
            f"empty set for tectum anchor, check atlas alignment "
            f"(|sc_m|={len(sc_mouse)}, |sc_h|={len(sc_human)}, "
            f"|ic_m|={len(ic_mouse)}, |ic_h|={len(ic_human)})"
        )

    return [
        RegionAnchorEntry(
            pair_id=32, label="Superior Colliculus (Isa 2021)",
            mouse_indices=sc_mouse, human_indices=sc_human,
        ),
        RegionAnchorEntry(
            pair_id=33, label="Inferior Colliculus (Winer & Schreiner 2005)",
            mouse_indices=ic_mouse, human_indices=ic_human,
        ),
    ]
