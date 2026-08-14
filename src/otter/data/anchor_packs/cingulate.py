"""Cingulate cortex anchor pack (Vogt et al. 2012).

Vogt et al. (2012, *Brain Structure and Function*) systematised the
cytoarchitectural sub-division of cingulate cortex across mouse, rat, and
human, identifying subgenual ACC (s32 in primates / v32 in rodents) and
retrosplenial cortex as homologous domains across species. Companion paper
Vogt et al. 2013 (*J Comp Neurology*) extends this to macaque area 32
subdivisions. van Heukelum et al. 2020 (*Trends in Neurosciences*)
provides a recent cross-species framework.

References (verified Consensus search 2026):
  - Vogt, B. A., Hof, P. R., Zilles, K., Vogt, L. J., Herold, C., & Palomero-
    Gallagher, N. (2012). Cytoarchitecture of mouse and rat cingulate cortex
    with human homologies. *Brain Structure and Function* 219, 185-192.
    DOI: 10.1007/s00429-012-0493-3.
  - van Heukelum, S. et al. (2020). Where is Cingulate Cortex? A Cross-
    Species View. *Trends in Neurosciences* 43, 285-299.
    DOI: 10.1016/j.tins.2020.03.007.

  pid 36:  Mouse ACA ventral ↔ Human subgenual ACC (BA24/25)
  pid 37:  Mouse Retrosplenial area ↔ Human RSC (BA29/30)

Why we deliberately avoid pregenual ACC
---------------------------------------
The most-studied cingulate sub-division is pregenual ACC (BA32). Its
canonical MNI centroid (±5, 25, 25) sits inside our human "Medial
prefrontal cortex" parcel, which is the *same* parcel anchored by
Garin pair_id 1 (Medial PFC). Adding a region anchor for pregenual ACC
would conflict with the existing Garin point anchor, and the soft
constraints would compete rather than compose. Subgenual ACC at
(±5, 10, 35) is anatomically distinct from mPFC in our parcellation
and pairs cleanly with mouse ACA ventral.

Why no PCC entry
----------------
Mouse "Posterior parietal association areas" is not a defensible
homologue of primate posterior cingulate cortex (Vogt & Paxinos 2014).
The cleanest cingulate pairs are subgenual ACC + RSC. We skip PCC; if a
specific user needs it, a dedicated pack should curate a primate
specifically-trained correspondence.

Mouse-side: DSURQE atlas overlay.
  ACA ventral: 15 parcels
  Retrosplenial area: 27 parcels
Human-side: MNI spheres.
  Subgenual ACC: (±5, 10, 35) r=10 mm  → 6 parcels
  RSC:           (±15, -55, 10) r=10 mm → 8 parcels

Caveat: for Beauchamp validation, the ACC subgenual anchor's mouse-side set
(ACA ventral) and human-side ball do NOT exactly match the Beauchamp
"Anterior cingulate area → cingulate gyrus" validation pair (which uses full mouse ACA and a different MNI
centroid). So Beauchamp recovery for ACG is NOT tautological for this
pack, actual measurement (rather than mechanical satisfaction).
Retrosplenial isn't in Beauchamp's 22 pairs so its effect is also not
directly measurable.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from otter.data.region_anchors import RegionAnchorEntry
from otter.data.anchor_packs._dsurqe import (
    mouse_parcels_in_dsurqe_region,
    human_parcels_in_mni_sphere,
)


def build_cingulate_region_anchors(
    M_var: pd.DataFrame,
    H_var: pd.DataFrame,
    *,
    atlas_root: Path | str = ".",
) -> list[RegionAnchorEntry]:
    """Build the two cingulate region anchors (subgenual ACC + RSC).

    Returns ``[ACC entry, RSC entry]`` (pid 36 and 37 respectively).
    """
    acc_mouse = mouse_parcels_in_dsurqe_region(
        M_var, "Anterior cingulate area, ventral part", atlas_root,
    )
    rsc_mouse = mouse_parcels_in_dsurqe_region(M_var, "Retrosplenial area", atlas_root)
    acc_human = human_parcels_in_mni_sphere(H_var, (-5,  10, 35), 10.0)
    rsc_human = human_parcels_in_mni_sphere(H_var, (-15, -55, 10), 10.0)

    if not (acc_mouse and acc_human and rsc_mouse and rsc_human):
        raise ValueError(
            f"empty set for cingulate anchor, check atlas alignment "
            f"(|acc_m|={len(acc_mouse)}, |acc_h|={len(acc_human)}, "
            f"|rsc_m|={len(rsc_mouse)}, |rsc_h|={len(rsc_human)})"
        )

    return [
        RegionAnchorEntry(
            pair_id=36, label="Subgenual ACC / BA24-25 (Vogt 2012)",
            mouse_indices=acc_mouse, human_indices=acc_human,
        ),
        RegionAnchorEntry(
            pair_id=37, label="Retrosplenial cortex / BA29-30 (Vogt 2012)",
            mouse_indices=rsc_mouse, human_indices=rsc_human,
        ),
    ]
