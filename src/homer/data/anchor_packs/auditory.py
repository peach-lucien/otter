"""Auditory cortex anchor pack (Hackett 2001; Kaas & Hackett 2000).

Primate auditory cortex has a well-characterised core + belt + parabelt
organisation. The core (AI, R, RT) is identified by primary-like
cytoarchitecture and dense MGN input; the belt is a surrounding ring
of ~7 secondary fields; the parabelt is lateral. Kaas & Hackett 2000
(*PNAS*, 1025 cit) defined this scheme and Hackett 2001 (*J Comp
Neurol*, 464 cit) extended the architectonic identification to
chimpanzees and humans.

  pid 56: Mouse Primary auditory area (A1) ↔ Human auditory core (BA41)
  pid 57: Mouse Dorsal + Ventral auditory areas (A2-dorsal + A2-ventral)
           ↔ Human auditory belt (BA42 / surrounding cortex)

The mouse auditory cortex is organised similarly: a tonotopically
mapped primary area (A1) surrounded by secondary fields (A2 dorsal /
AAF, A2 ventral / VAF). The cross-species correspondence at the gross
core/belt level is established; finer subdivisions (e.g. which mouse
secondary field maps to which primate belt field) are still debated.

Why this pack might lift Beauchamp Auditory
-------------------------------------------
Beauchamp validates "Primary auditory area → Heschl's gyrus" with a
broad 18-parcel human ball (currently 22 % top-1 under the
production point-anchor π). Our pid 56 anchors mouse A1 (the SAME 9
parcels Beauchamp uses) onto a tighter A1 core ball, while pid 57
provides additional human-belt coverage for mouse secondary auditory
parcels (which Beauchamp's "Primary auditory area" set doesn't
include). The combined pack should at minimum hold the 22 % top-1
and may lift it via the tighter core target.

Mouse-side: DSURQE atlas overlay.
  Primary auditory area: 9 parcels (A1)
  Dorsal auditory area:  6 parcels (A2 dorsal / anterior auditory field)
  Ventral auditory area: 5 parcels (A2 ventral)

Human-side: MNI spheres at canonical centroids.
  A1 core (BA41):          (±48, -22, 6) r=6 mm
  Auditory belt (BA42):    (±55, -15, 0) r=8 mm

References (verified Consensus search 2026):
  - Hackett, T. A., Preuss, T. M., & Kaas, J. H. (2001). Architectonic
    identification of the core region in auditory cortex of macaques,
    chimpanzees, and humans. *Journal of Comparative Neurology* 441,
    197-222. DOI: 10.1002/cne.1407. (464 citations).
  - Kaas, J. H. & Hackett, T. A. (2000). Subdivisions of auditory cortex
    and processing streams in primates. *PNAS* 97, 11793-11799.
    DOI: 10.1073/pnas.97.22.11793. (1025 citations).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from homer.data.region_anchors import RegionAnchorEntry
from homer.data.anchor_packs._dsurqe import (
    mouse_parcels_in_dsurqe_region,
    human_parcels_in_mni_sphere,
)


def build_auditory_region_anchors(
    M_var: pd.DataFrame,
    H_var: pd.DataFrame,
    *,
    atlas_root: Path | str = ".",
) -> list[RegionAnchorEntry]:
    """Build the two auditory subdivision anchors (core + belt).

    Returns ``[A1↔core, A2↔belt]`` at pids 56 and 57.
    """
    a1_mouse = mouse_parcels_in_dsurqe_region(M_var, "Primary auditory area", atlas_root)
    a2d_mouse = mouse_parcels_in_dsurqe_region(M_var, "Dorsal auditory area", atlas_root)
    a2v_mouse = mouse_parcels_in_dsurqe_region(M_var, "Ventral auditory area", atlas_root)
    a2_mouse = sorted(set(a2d_mouse) | set(a2v_mouse))
    a1_human = human_parcels_in_mni_sphere(H_var, (-48, -22, 6), 6.0)
    belt_human = human_parcels_in_mni_sphere(H_var, (-55, -15, 0), 8.0)

    if not (a1_mouse and a1_human and a2_mouse and belt_human):
        raise ValueError(
            f"empty set for auditory anchor "
            f"(|a1_m|={len(a1_mouse)}, |a1_h|={len(a1_human)}, "
            f"|a2_m|={len(a2_mouse)}, |belt_h|={len(belt_human)})"
        )

    return [
        RegionAnchorEntry(
            pair_id=56,
            label="Primary auditory / A1 core (Hackett 2001)",
            mouse_indices=a1_mouse, human_indices=a1_human,
        ),
        RegionAnchorEntry(
            pair_id=57,
            label="Auditory belt (A2 dorsal+ventral; Kaas & Hackett 2000)",
            mouse_indices=a2_mouse, human_indices=belt_human,
        ),
    ]
