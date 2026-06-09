"""Pre-compute, for each of HOMER's 1864 parcels, the set of CCFv3 voxel
indices its rsmask voxels map to after applying Paul's warpfield2SS.

We save these at two CCFv3 resolutions so downstream consumers can use
whichever is appropriate:

  - 200 µm voxel indices: matches Allen ISH energy volumes shape (~67×41×58)
                          and most AllenSDK gridded data. Use this for the
                          gene-matrix rebuild (02_mouse_genes successor).
  - 25 µm voxel indices:  matches the fixed CCFv3 annotation. Use this for
                          fine-grained per-voxel queries / visualisation.

For each parcel we save the deduplicated voxel-index set (after nearest-
neighbour rounding) plus a count. The list-of-lists format is awkward for
parquet, so we save:

  - ``parcel_warped_voxels_200um.npz``  ragged arrays via concat + offsets
  - ``parcel_warped_voxels_25um.npz``   same at 25 µm
  - ``parcel_warped_voxels.json``       metadata + per-parcel n_voxels

Usage:
    PYTHONPATH=src python pipeline/00_external/warp_rebuild/04_warped_voxel_sets.py
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT.parent / "data_crossspecies"
WF   = DATA / "warpfields"
OUT  = ROOT / "data_external" / "_warp_rebuild"
OUT.mkdir(parents=True, exist_ok=True)


def _u16str(f, ref):
    a = np.asarray(f[ref][:]).flatten()
    return bytes(memoryview(a.astype(np.uint16))).decode("utf-16-le", errors="replace").rstrip("\x00")


def main():
    # Load HOMER voxel_indices
    print("loading HOMER voxel_indices...")
    with h5py.File(str(DATA / "corrs_mouse.mat"), "r") as f:
        g = f["m"]
        ht = [_u16str(f, r) for r in np.asarray(g["ht"][:]).flatten()]
        t_refs = np.asarray(g["t"][:])
        n_nodes = t_refs.shape[1]
        ic_indices = ht.index("indices")
        voxel_indices = []
        for n in range(n_nodes):
            vi = np.asarray(f[t_refs[ic_indices, n]][:]).flatten().astype(np.int64)
            voxel_indices.append(vi)
    print(f"  {n_nodes} parcels, total voxels = {sum(len(v) for v in voxel_indices)}")

    rsmask = nib.load(str(DATA / "_mouse_mask" / "rsmask.nii"))
    rsmask_aff = rsmask.affine
    rsmask_shape = rsmask.shape

    wf_img = nib.load(str(WF / "warpfield2SS.nii.gz"))
    wf2ss = np.asarray(wf_img.dataobj)
    wf_aff = wf_img.affine
    wf_inv = np.linalg.inv(wf_aff)
    print(f"warpfield2SS shape {wf2ss.shape}  affine origin {wf_aff[:3, 3]}")

    # Allen 25 µm grid (matches the fixed annotation) is Paul's NS grid:
    #   shape (528, 320, 456) at 25 µm, axcodes PIR (RAS+ affine derived below).
    # Allen 200 µm volumes (from Allen ISH energy) have shape ~(67, 41, 58)
    # but with the same PIR layout and origin at the corner.
    res_25um = 0.025
    res_200um = 0.200
    # NS world coords -> 25 µm voxel index. Reuse the warpfield's affine since
    # that's the NS world frame.
    # The fixed annotation has the SAME affine as warpfield2NS; let's compute
    # voxel index by treating axcodes PIR: world (x, y, z) -> voxel
    #   k = x / res, i = -y / res, j = -z / res
    def ns_world_to_voxel(ns_world, res):
        x, y, z = ns_world[..., 0], ns_world[..., 1], ns_world[..., 2]
        k = (x / res).round().astype(np.int64)        # LR axis (axis 2)
        i = (-y / res).round().astype(np.int64)       # AP axis (axis 0)
        j = (-z / res).round().astype(np.int64)       # DV axis (axis 1)
        return np.stack([i, j, k], axis=-1)

    n_25 = (int(np.ceil(11.4 / res_25um)) + 4,
            int(np.ceil(13.2 / res_25um)) + 4,
            int(np.ceil(8.0 / res_25um)) + 4)   # not used directly
    print(f"using 25 µm voxel resolution and 200 µm voxel resolution for downstream sampling")

    # Per-parcel: voxel_indices (1D MATLAB, Fortran) -> SS world mm -> warpfield -> NS world mm
    # Then convert NS world mm -> CCFv3 voxel index at the requested resolution.
    all_25_i, all_25_j, all_25_k = [], [], []
    all_200_i, all_200_j, all_200_k = [], [], []
    offsets_25  = [0]
    offsets_200 = [0]
    n_vox_25, n_vox_200 = [], []
    n_outside = 0

    for n, vi in enumerate(voxel_indices):
        idx = vi - 1                                    # MATLAB 1-based -> 0
        idx = idx[(idx >= 0) & (idx < int(np.prod(rsmask_shape)))]
        ijk = np.array(np.unravel_index(idx, rsmask_shape, order="F")).T
        homog = np.column_stack([ijk, np.ones(len(ijk))])
        ss_world = (rsmask_aff @ homog.T).T[:, :3]      # SS world mm (~RAS+)

        # SS world -> SS voxel on Paul's 70 µm grid
        ss_homog = np.column_stack([ss_world, np.ones(len(ss_world))])
        ss_vox_f = (wf_inv @ ss_homog.T).T[:, :3]
        ss_vox = np.round(ss_vox_f).astype(int)

        # Look up warpfield -> NS world mm
        ns_world = np.full((len(ss_vox), 3), np.nan)
        for v_i in range(len(ss_vox)):
            x, y, z = ss_vox[v_i]
            if 0 <= x < wf2ss.shape[0] and 0 <= y < wf2ss.shape[1] and 0 <= z < wf2ss.shape[2]:
                ns_world[v_i] = wf2ss[x, y, z]

        # Drop voxels where the warpfield is undefined or returns 0 (outside-brain padding)
        mag = np.linalg.norm(ns_world, axis=1)
        ok = np.isfinite(mag) & (mag > 1e-6)
        if not ok.any():
            n_outside += 1

        ns_ok = ns_world[ok]

        # Convert to 25 µm and 200 µm CCFv3 voxel indices
        v25 = ns_world_to_voxel(ns_ok, res_25um)
        v200 = ns_world_to_voxel(ns_ok, res_200um)

        # Deduplicate (different parcel voxels may map to the same CCFv3 voxel)
        v25_u = np.unique(v25, axis=0) if len(v25) else v25.reshape(0, 3)
        v200_u = np.unique(v200, axis=0) if len(v200) else v200.reshape(0, 3)

        all_25_i.append(v25_u[:, 0])
        all_25_j.append(v25_u[:, 1])
        all_25_k.append(v25_u[:, 2])
        all_200_i.append(v200_u[:, 0])
        all_200_j.append(v200_u[:, 1])
        all_200_k.append(v200_u[:, 2])
        offsets_25.append(offsets_25[-1] + len(v25_u))
        offsets_200.append(offsets_200[-1] + len(v200_u))
        n_vox_25.append(int(len(v25_u)))
        n_vox_200.append(int(len(v200_u)))

        if (n + 1) % 500 == 0:
            print(f"  ...{n+1}/{n_nodes} parcels")

    arr_25_i = np.concatenate(all_25_i)
    arr_25_j = np.concatenate(all_25_j)
    arr_25_k = np.concatenate(all_25_k)
    arr_200_i = np.concatenate(all_200_i)
    arr_200_j = np.concatenate(all_200_j)
    arr_200_k = np.concatenate(all_200_k)

    np.savez(OUT / "parcel_warped_voxels_25um.npz",
             i=arr_25_i.astype(np.int32),
             j=arr_25_j.astype(np.int32),
             k=arr_25_k.astype(np.int32),
             offsets=np.asarray(offsets_25, dtype=np.int64))
    np.savez(OUT / "parcel_warped_voxels_200um.npz",
             i=arr_200_i.astype(np.int32),
             j=arr_200_j.astype(np.int32),
             k=arr_200_k.astype(np.int32),
             offsets=np.asarray(offsets_200, dtype=np.int64))
    meta = {
        "n_parcels": n_nodes,
        "resolution_um_files": {
            "25um": "parcel_warped_voxels_25um.npz",
            "200um": "parcel_warped_voxels_200um.npz",
        },
        "format": (
            "ragged arrays: concatenated i/j/k arrays of length offsets[-1], "
            "where parcel n's voxel indices live in offsets[n]:offsets[n+1]."
        ),
        "n_vox_25um_per_parcel_summary": {
            "min": int(np.min(n_vox_25)), "max": int(np.max(n_vox_25)),
            "mean": float(np.mean(n_vox_25)), "median": float(np.median(n_vox_25)),
        },
        "n_vox_200um_per_parcel_summary": {
            "min": int(np.min(n_vox_200)), "max": int(np.max(n_vox_200)),
            "mean": float(np.mean(n_vox_200)), "median": float(np.median(n_vox_200)),
        },
        "n_parcels_with_no_warped_voxels": int(n_outside),
        "source_warpfield": str(WF / "warpfield2SS.nii.gz"),
        "source_rsmask":    str(DATA / "_mouse_mask" / "rsmask.nii"),
    }
    (OUT / "parcel_warped_voxels.json").write_text(json.dumps(meta, indent=2))
    print(f"\nsaved 25 µm voxel sets:   total {len(arr_25_i)} indices across {n_nodes} parcels")
    print(f"saved 200 µm voxel sets:  total {len(arr_200_i)} indices across {n_nodes} parcels")
    print(f"per-parcel n_vox @ 25 µm: median {np.median(n_vox_25):.0f}, max {np.max(n_vox_25)}")
    print(f"per-parcel n_vox @ 200 µm: median {np.median(n_vox_200):.0f}, max {np.max(n_vox_200)}")


if __name__ == "__main__":
    main()
