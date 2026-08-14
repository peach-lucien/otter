"""Look up atlas labels for each parcel and build region-anchor specs.

Two atlases for the human side, both in
``data_external/_domhof_extracted/``, extracted from the Domhof bundle, the
same source as the human FC, SC and parcellation:

  - **Schaefer-400** (cortical): 400 cortical parcels in 17-network order,
    in MNI152 2mm space.
  - **JuBrain (Julich-Brain) 184**: 184 cyto-architectonically-defined regions
    including some subcortical and cerebellar coverage, in MNI152 2mm space.

Both are needed: Schaefer covers cortex broadly (~84% of the parcels) but not
subcortex; JuBrain's cytoarchitectonic regions cover some subcortex but miss
some cortical landmarks (e.g. auditory). Together they cover ~88% of the
parcels with at least one atlas label.

The mouse side uses the DSURQE atlas extracted from the Beauchamp 2022 repo
(see ``pipeline/05f_beauchamp_validation.py``).

Region anchors are defined only where an atlas supplies labels. Regions
without atlas coverage (Septum, Striatum, Pallidum, Thalamus, Pons, Tectum,
and others) stay as point anchors.

Public:
    assign_atlas_labels(var, atlas) -> (n,) int array of atlas IDs
    build_garin_region_anchors_from_schaefer(M, H) -> list[RegionAnchorEntry]
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd

from otter.data.anchors import get_anchor_index
from otter.data.region_anchors import RegionAnchorEntry


# Locations of the extracted atlas volumes (relative to repo root)
ATLAS_PATHS = {
    "schaefer_400": "data_external/_domhof_extracted/Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.nii.gz",
    "jubrain_184":  "data_external/_domhof_extracted/JuBrain_Atlas_MNI_2mm_Version_4.nii.gz",
}


def _lookup_atlas_id(arr: np.ndarray, affine: np.ndarray, xyz: np.ndarray,
                     *, radius: int = 2) -> np.ndarray:
    """For each xyz point, return the most-frequent non-zero atlas label
    in a ±radius cube around its corresponding voxel."""
    inv = np.linalg.inv(affine)
    voxels = (inv @ np.c_[xyz, np.ones(len(xyz))].T).T[:, :3]
    i, j, k = (voxels[:, ax].round().astype(int) for ax in range(3))
    sh = arr.shape
    out = np.zeros(len(xyz), dtype=int)
    for p in range(len(xyz)):
        i0, i1 = max(0, i[p]-radius), min(sh[0], i[p]+radius+1)
        j0, j1 = max(0, j[p]-radius), min(sh[1], j[p]+radius+1)
        k0, k1 = max(0, k[p]-radius), min(sh[2], k[p]+radius+1)
        block = arr[i0:i1, j0:j1, k0:k1].ravel()
        nz = block[block > 0]
        if len(nz) == 0:
            continue
        vals, counts = np.unique(nz, return_counts=True)
        out[p] = int(vals[counts.argmax()])
    return out


def assign_atlas_labels(var: pd.DataFrame, atlas: str = "schaefer_400",
                         atlas_path: Optional[Path | str] = None,
                         radius: int = 2) -> np.ndarray:
    """Return (n,) array of atlas label IDs (0 = no overlap) for each parcel
    in ``var``. ``var`` must have x/y/z columns in MNI152 mm.
    """
    if atlas_path is None:
        atlas_path = ATLAS_PATHS[atlas]
    img = nib.load(str(atlas_path))
    arr = np.asarray(img.get_fdata()).astype(int)
    xyz = var[["x", "y", "z"]].to_numpy()
    return _lookup_atlas_id(arr, img.affine, xyz, radius=radius)


def assign_atlas_labels_with_hemisphere(
    var: pd.DataFrame, schaefer_ids: np.ndarray
) -> np.ndarray:
    """Return Schaefer IDs but enforce per-hemisphere consistency.

    Some near-midline parcels can pick up a Schaefer ID from the wrong
    hemisphere via the radius-2 cube lookup. This function zeros out any
    parcel whose Schaefer ID corresponds to the wrong hemisphere
    (Schaefer-400 IDs 1-200 are LH, 201-400 are RH per the official
    17-network ordering).
    """
    out = schaefer_ids.copy()
    hemi = var["hemisphere"].values
    is_lh = (out >= 1) & (out <= 200)
    is_rh = (out >= 201) & (out <= 400)
    wrong_hemi = ((hemi == "L") & is_rh) | ((hemi == "R") & is_lh)
    out[wrong_hemi] = 0
    return out


def build_garin_region_anchors_from_atlases(
    M_var: pd.DataFrame,
    H_var: pd.DataFrame,
    *,
    atlas_root: Path | str = ".",
    pid_offset: int = 30,
    skip_missing: bool = True,
) -> list[RegionAnchorEntry]:
    """For each Garin pair_id, build a region anchor using:

      - Mouse: parcels with the same DSURQE label as the anchor parcel
      - Human: parcels with the same Schaefer-400 or JuBrain label as the
        anchor parcel (whichever is non-zero; Schaefer > JuBrain for
        cortical, JuBrain > Schaefer for sub-cortical regions)

    If the anchor parcel has *no* atlas label on the human side, that
    pair is skipped (returned list is shorter than 21). The caller can
    keep the missing pairs as point anchors.

    Region pair_ids are assigned starting at ``pid_offset`` (default 30,
    well above the 21 Garin point-anchor pair_ids).
    """
    atlas_root = Path(atlas_root)
    schaefer_path = atlas_root / ATLAS_PATHS["schaefer_400"]
    jubrain_path  = atlas_root / ATLAS_PATHS["jubrain_184"]

    # Load mouse DSURQE labels (for mouse parcels)
    dsurqe_path = atlas_root / "data_external/MouseHumanTranscriptomicSimilarity/AMBA/data/imaging/DSURQE_CCFv3_labels_200um.mnc"
    if not dsurqe_path.exists():
        raise FileNotFoundError(
            f"DSURQE atlas not found at {dsurqe_path}. Need the Beauchamp repo "
            f"at data_external/MouseHumanTranscriptomicSimilarity/."
        )
    img_d = nib.load(str(dsurqe_path))
    arr_d = np.asarray(img_d.get_fdata()).astype(int)
    DSURQE_OFFSET = np.array([-0.027, -2.334, +1.018])  # see pipeline/05f_*.py
    mouse_xyz_d = M_var[["x", "y", "z"]].to_numpy() + DSURQE_OFFSET
    mouse_dsurqe = _lookup_atlas_id(arr_d, img_d.affine, mouse_xyz_d, radius=2)

    # Load human atlases
    schaefer_ids = assign_atlas_labels(H_var, "schaefer_400", schaefer_path)
    schaefer_ids = assign_atlas_labels_with_hemisphere(H_var, schaefer_ids)
    jubrain_ids  = assign_atlas_labels(H_var, "jubrain_184",  jubrain_path)

    # Garin anchors
    idx_m = get_anchor_index(M_var); idx_h = get_anchor_index(H_var)
    if idx_m.keys != idx_h.keys:
        raise ValueError("anchor key orderings differ between species")

    out: list[RegionAnchorEntry] = []
    pair_ids_seen: set[int] = set()
    for k, mp in enumerate(idx_m.pos):
        pid = int(idx_m.pair_ids[k])
        hemi = idx_m.hemispheres[k]
        # One region per (pid, hemi); L and R are combined in the entry
        pass

    # Pair-level: one entry per pair_id, combining L and R
    pair_to_pos_m = {}; pair_to_pos_h = {}
    for k in range(len(idx_m)):
        pid = int(idx_m.pair_ids[k])
        pair_to_pos_m.setdefault(pid, []).append((idx_m.hemispheres[k], int(idx_m.pos[k])))
        pair_to_pos_h.setdefault(pid, []).append((idx_h.hemispheres[k], int(idx_h.pos[k])))

    n_built = n_skipped = 0
    skipped_reasons = []
    for pid in sorted(pair_to_pos_m.keys()):
        # For both species, the set of parcels sharing atlas labels with the
        # anchor parcels.
        # Mouse: DSURQE label
        m_anchor_lbls = {mouse_dsurqe[mp] for _, mp in pair_to_pos_m[pid]
                          if mouse_dsurqe[mp] > 0}
        # Human: Schaefer first (preferred), JuBrain only if Schaefer missing
        h_anchor_lbls_sc = {schaefer_ids[hp] for _, hp in pair_to_pos_h[pid]
                             if schaefer_ids[hp] > 0}
        h_anchor_lbls_ju = {jubrain_ids[hp]  for _, hp in pair_to_pos_h[pid]
                             if jubrain_ids[hp]  > 0}

        # If neither species has labels, skip
        if not m_anchor_lbls or (not h_anchor_lbls_sc and not h_anchor_lbls_ju):
            n_skipped += 1
            anchor_name = M_var.iloc[pair_to_pos_m[pid][0][1]]["region"]
            reason = (f"mouse_dsurqe={'OK' if m_anchor_lbls else 'missing'}, "
                       f"human_atlas={'missing' if not (h_anchor_lbls_sc or h_anchor_lbls_ju) else 'OK'}")
            skipped_reasons.append(f"  pid={pid} ({anchor_name}): {reason}")
            if skip_missing:
                continue

        # Mouse-region: parcels with these DSURQE labels
        mouse_set = [int(p) for p in np.where(np.isin(mouse_dsurqe, list(m_anchor_lbls)))[0]]
        # Human-region: parcels with matching Schaefer (if available) else JuBrain
        if h_anchor_lbls_sc:
            human_set = [int(p) for p in np.where(np.isin(schaefer_ids, list(h_anchor_lbls_sc)))[0]]
            human_atlas_used = "schaefer_400"
        else:
            human_set = [int(p) for p in np.where(np.isin(jubrain_ids, list(h_anchor_lbls_ju)))[0]]
            human_atlas_used = "jubrain_184"

        anchor_name = M_var.iloc[pair_to_pos_m[pid][0][1]]["region"].lstrip("LR_")
        out.append(RegionAnchorEntry(
            pair_id=pid_offset + pid,
            label=f"Garin pair_id={pid} ({anchor_name}), mouse=DSURQE, human={human_atlas_used}",
            mouse_indices=mouse_set,
            human_indices=human_set,
        ))
        n_built += 1

    if skipped_reasons:
        print(f"Skipped {n_skipped}/{len(pair_to_pos_m)} Garin pairs (no atlas coverage):")
        for r in skipped_reasons:
            print(r)
    # Detect human-set overlaps: two region anchors sharing parcels create
    # ambiguous constraints that the solver cannot satisfy exactly. The warning
    # lets the caller keep, merge, or drop them.
    n_overlap = 0
    for i in range(len(out)):
        for j in range(i+1, len(out)):
            sh_h = set(out[i].human_indices) & set(out[j].human_indices)
            if sh_h:
                n_overlap += 1
                if n_overlap <= 3:    # cap output noise
                    print(f"  ⚠ overlap: pid={out[i].pair_id} & {out[j].pair_id} share "
                          f"{len(sh_h)} human parcels (atlas resolution limit)")
    if n_overlap:
        print(f"  total {n_overlap} pid-pair overlaps, region constraints are ambiguous "
              f"in those cases (held-out CV will reveal effects)")
    print(f"Built {n_built} region anchors from atlas labels (pid {pid_offset+1}..{pid_offset+max(pair_to_pos_m)}).")
    return out


# Curated per-region anchor packs (BICCN motor, Tectum, etc.) live in
# ``otter.data.anchor_packs.*``, each pack is a small self-contained module.
# This file keeps only the systematic atlas-derived pack
# (``build_garin_region_anchors_from_atlases``).
