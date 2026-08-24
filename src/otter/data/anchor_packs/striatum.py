"""Striatal regional correspondence entries.

Pair IDs 47 and 48 link dorsolateral and ventromedial subsets of mouse caudoputamen to human putamen and anterior caudate targets. Primary source: Voorn et al., Trends in Neurosciences (2004), doi:10.1016/j.tins.2004.06.006."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from otter.data.region_anchors import RegionAnchorEntry
from otter.data.anchor_packs._dsurqe import (
    mouse_parcels_in_dsurqe_region,
    human_parcels_in_mni_sphere,
)


def build_striatum_region_anchors(
    M_var: pd.DataFrame,
    H_var: pd.DataFrame,
    *,
    atlas_root: Path | str = ".",
) -> list[RegionAnchorEntry]:
    """Build the two Voorn-aligned striatum subdivision anchors.

    Returns ``[Dorsolateral (putamen), Ventromedial (caudate)]`` at pids 47 and 48.
    """
    cp_all = mouse_parcels_in_dsurqe_region(M_var, "Caudoputamen", atlas_root)
    if not cp_all:
        raise ValueError("mouse Caudoputamen empty, check DSURQE atlas")

    # Spatial subdivision: dorsolateral vs ventromedial by (|x|, z) thresholds
    cp_var = M_var.iloc[cp_all]
    abs_x = cp_var["x"].abs().to_numpy()
    z = cp_var["z"].to_numpy()
    med_abs_x = float(np.median(abs_x))
    med_z = float(np.median(z))

    dl_mask = (abs_x > med_abs_x) & (z > med_z)
    vm_mask = (abs_x <= med_abs_x) & (z <= med_z)

    cp_arr = np.asarray(cp_all)
    dl_mouse = [int(i) for i in cp_arr[dl_mask]]
    vm_mouse = [int(i) for i in cp_arr[vm_mask]]

    # Human putamen (lateral, sensorimotor): MNI ±28, 0, 0
    putamen = human_parcels_in_mni_sphere(H_var, (-28, 0, 0), 10.0)
    # Human anterior caudate (medial, associative): MNI ±10, 10, 10
    caudate = human_parcels_in_mni_sphere(H_var, (-10, 10, 10), 10.0)

    if not (dl_mouse and putamen and vm_mouse and caudate):
        raise ValueError(
            f"empty striatum subset (|dl_m|={len(dl_mouse)}, "
            f"|putamen|={len(putamen)}, |vm_m|={len(vm_mouse)}, "
            f"|caudate|={len(caudate)})"
        )

    return [
        RegionAnchorEntry(
            pair_id=47,
            label="Caudoputamen dorsolateral / Putamen (Voorn 2004)",
            mouse_indices=dl_mouse, human_indices=putamen,
        ),
        RegionAnchorEntry(
            pair_id=48,
            label="Caudoputamen ventromedial / Caudate anterior (Voorn 2004)",
            mouse_indices=vm_mouse, human_indices=caudate,
        ),
    ]
