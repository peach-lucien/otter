"""Lateral prefrontal cortex anchor pack.

The Garin atlas covers prefrontal cortex with one point anchor at mPFC
(pair_id 1). This pack adds two lateral PFC sub-region anchors based on
published cross-species cytoarchitecture and connectivity correspondences:

  pid 45: Mouse Orbital area, lateral part ↔ Human OFC (BA11/47)
  pid 46: Mouse Prelimbic area ↔ Human dlPFC (BA9/46)  [**see homology caveat**]

This is the first pack added for regions with NO direct Beauchamp
validation pair — purely anatomical-credibility-driven. Coverage of
lateral PFC matters for downstream users studying decision-making,
working memory, executive control, and reward processing.

OFC homology (pid 45) — high confidence
---------------------------------------
Orbitofrontal cortex homology between rodents and primates is among the
best-established in PFC. Both species have a cytoarchitecturally
defined OFC with similar afferents from amygdala / thalamus and similar
roles in value-based decision-making. DSURQE "Orbital area, lateral
part" maps cleanly onto human BA11/47.

  Reference: Wallis, J. D. (2011). Cross-species studies of orbitofrontal
  cortex and value-based decision-making. *Nature Neuroscience* 15,
  13-19. DOI: 10.1038/nn.2956. Note this paper is from 2011 / Nature
  Neuroscience, NOT 2012 / Nat Rev Neurosci as may appear elsewhere.

dlPFC homology (pid 46) — *contested*
-------------------------------------
Whether rodents have a direct homologue of primate dorsolateral PFC
(BA9/46) is a long-running debate:

  - **Preuss (1995, *J Cogn Neurosci* 7, 1-24, DOI 10.1162/jocn.1995.7.1.1)**:
    rodents lack a granular cortex equivalent to primate dlPFC; mouse
    Prelimbic is more like an extended cingulate / mPFC region.
  - **Carlén (2017, *Science* 358, 478-482, DOI 10.1126/science.aan8868)** and
    Laubach et al. (2018, *eNeuro*): PL is the *functional* homologue based
    on cross-species working-memory / cognitive-control tasks, even if the
    cytoarchitecture differs.

We include this anchor with the caveat. Users who consider rodent PL not
homologous to primate dlPFC should exclude this entry (it's a separate
pair_id, easy to filter out).

Mouse-side: DSURQE atlas overlay.
  Orbital area, lateral part: 21 parcels
  Prelimbic area:              11 parcels

Human-side: MNI spheres at Petrides cytoarchitectural centroids.
  OFC BA11/47:    (±25, 35, -15) r=10 mm →  8 parcels
  dlPFC BA9/46:   (±40, 25,  35) r=10 mm → 12 parcels

Composition caveat — overlap with Garin pid 1 (mPFC)
----------------------------------------------------
Mouse Prelimbic is anatomically close to the mouse mPFC parcel that
hosts Garin pid 1 (Medial PFC). The Prelimbic anchor's mouse-side set
may include the Garin pid 1 anchor parcel, in which case the soft
region anchor constrains a parcel that already has a Garin point
anchor. The FGW solver handles this; expect mass on that parcel to be
intermediate between the Garin pid 1 target and our dlPFC ball.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from homer.data.region_anchors import RegionAnchorEntry
from homer.data.anchor_packs._dsurqe import (
    mouse_parcels_in_dsurqe_region,
    human_parcels_in_mni_sphere,
)


def build_lateral_pfc_region_anchors(
    M_var: pd.DataFrame,
    H_var: pd.DataFrame,
    *,
    atlas_root: Path | str = ".",
    include_dlpfc: bool = False,
) -> list[RegionAnchorEntry]:
    """Build the lateral PFC region anchors.

    Returns the OFC anchor (pid 45) only. The Prelimbic↔dlPFC anchor
    (pid 46) is **excluded by default**: rodent dlPFC homology is disputed
    (Preuss 1995) and is independently contradicted by the Balsters 2020
    falsification test, so the recommended composition does not assert it.
    Pass ``include_dlpfc=True`` to add it back (e.g. for ablations).
    """
    ofc_mouse = mouse_parcels_in_dsurqe_region(
        M_var, "Orbital area, lateral part", atlas_root,
    )
    ofc_human = human_parcels_in_mni_sphere(H_var, (-25, 35, -15), 10.0)
    if not (ofc_mouse and ofc_human):
        raise ValueError(
            f"empty set for OFC anchor — check atlas alignment "
            f"(|ofc_m|={len(ofc_mouse)}, |ofc_h|={len(ofc_human)})"
        )
    entries = [
        RegionAnchorEntry(
            pair_id=45, label="OFC / BA11-47 (Wallis 2011)",
            mouse_indices=ofc_mouse, human_indices=ofc_human,
        ),
    ]

    if include_dlpfc:
        pl_mouse = mouse_parcels_in_dsurqe_region(M_var, "Prelimbic area", atlas_root)
        dl_human = human_parcels_in_mni_sphere(H_var, (-40, 25, 35), 10.0)
        if not (pl_mouse and dl_human):
            raise ValueError(
                f"empty set for dlPFC anchor — check atlas alignment "
                f"(|pl_m|={len(pl_mouse)}, |dl_h|={len(dl_human)})"
            )
        entries.append(RegionAnchorEntry(
            pair_id=46, label="dlPFC / BA9-46 (Carlén 2017; contested homology)",
            mouse_indices=pl_mouse, human_indices=dl_human,
        ))

    return entries
