"""Periaqueductal Gray anchor pack (Bandler & Shipley 1994; Ezra 2015).

The midbrain periaqueductal gray (PAG) is a key homeostatic structure
involved in pain modulation, autonomic function, and defensive behaviour.
Its columnar organisation (dorsolateral, lateral, ventrolateral) is
conserved from rodent through human:

  - Bandler & Shipley 1994 *TINS* — established the columnar functional
    model in rodent.
  - Ezra et al. 2015, *Human Brain Mapping* — diffusion-MRI-based
    segmentation of human PAG into 4 columns concordant with the
    rodent model (76 cit).
  - Kingsbury et al. 2011, *PLOS ONE* — extends the columnar model to
    birds, confirming pan-amniote conservation (93 cit).

  pid 54: Mouse Periaqueductal gray ↔ Human PAG

Caveat from Ezra 2015 — columnar *structure* is conserved but cortical
connectivity differs between humans and other mammals. So the gross
PAG↔PAG anchor is defensible; sub-column splits (dorsolateral vs lateral
vs ventrolateral) are *not* attempted here. They could be added if a
human PAG-column atlas becomes available.

Mouse-side: DSURQE atlas overlay.
  Periaqueductal gray: 16 parcels

Human-side: MNI sphere at canonical PAG centroid (Mai/Paxinos).
  PAG: (±5, –30, –10) r=6 mm → ~4-6 parcels (tight ball; PAG is small)

References (verified Consensus search 2026):
  - Ezra, M., Faull, O. K., Jbabdi, S., & Pattinson, K. T. (2015).
    Connectivity-based segmentation of the periaqueductal gray matter
    in human with brainstem optimized diffusion MRI. *Human Brain Mapping*
    36, 3459-3471. DOI: 10.1002/hbm.22855. (76 citations).
  - Kingsbury, M. A., Kelly, A. M., Schrock, S. E., & Goodson, J. L.
    (2011). Mammal-like organization of the avian midbrain central gray
    and a reappraisal of the intercollicular nucleus. *PLOS ONE* 6,
    e20720. DOI: 10.1371/journal.pone.0020720. (93 citations).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from homer.data.region_anchors import RegionAnchorEntry
from homer.data.anchor_packs._dsurqe import (
    mouse_parcels_in_dsurqe_region,
    human_parcels_in_mni_sphere,
)


def build_pag_region_anchors(
    M_var: pd.DataFrame,
    H_var: pd.DataFrame,
    *,
    atlas_root: Path | str = ".",
) -> list[RegionAnchorEntry]:
    """Build the periaqueductal gray anchor.

    Returns ``[PAG entry]`` at pid 54.
    """
    pag_mouse = mouse_parcels_in_dsurqe_region(M_var, "Periaqueductal gray", atlas_root)
    pag_human = human_parcels_in_mni_sphere(H_var, (-5, -30, -10), 6.0)

    if not (pag_mouse and pag_human):
        raise ValueError(
            f"empty set for PAG anchor "
            f"(|pag_m|={len(pag_mouse)}, |pag_h|={len(pag_human)})"
        )

    return [
        RegionAnchorEntry(
            pair_id=54,
            label="Periaqueductal gray (Ezra 2015; Kingsbury 2011)",
            mouse_indices=pag_mouse, human_indices=pag_human,
        ),
    ]
