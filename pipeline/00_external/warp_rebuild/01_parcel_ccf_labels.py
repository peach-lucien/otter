"""Build a per-parcel CCFv3 acronym table using Paul's nonlinear warpfield.

For each of HOMER's 1864 mouse parcels, this script:

  1. Reads the parcel's voxel_indices (MATLAB 1-based linear indices into rsmask
     grid at 200 µm, but actually living on the same world-coordinate frame as
     Paul's 70 µm SS DSURQE grid).
  2. Converts each voxel to SS world-mm via the rsmask affine.
  3. Looks up Paul's warpfield2SS at the corresponding SS voxel (70 µm grid)
     to read the NS (CCFv3 25 µm) world-mm coordinate.
  4. Looks up the fixed Allen CCFv3 annotation (uint32) at that NS voxel.
  5. Resolves the Allen label id to acronym + full ancestry path.

Outputs (saved to ``data_external/_warp_rebuild/``):
  - ``parcel_ccfv3_labels.parquet``  per-parcel table with warped centre, voxel,
                                       centre-label acronym, majority-vote
                                       acronym across the parcel's voxel set,
                                       agreement fraction, n_voxels in parcel.
  - ``parcel_ccfv3_labels.csv``       same, for easy diffing.
  - ``parcel_ccfv3_labels.json``      provenance: filenames, affines, n_parcels.

This is the input for the diagnostic comparing the new warp against HOMER's
heuristic 48-permutation transform, and a column we can ship in ``M.var`` so
downstream queries can use CCFv3 acronyms directly rather than going through
DSURQE.

Usage:
    PYTHONPATH=src python pipeline/00_external/warp_rebuild/01_parcel_ccf_labels.py
"""
from __future__ import annotations

import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT.parent / "data_crossspecies"
WF   = DATA / "warpfields"
OUT  = ROOT / "data_external" / "_warp_rebuild"
OUT.mkdir(parents=True, exist_ok=True)

FIXED_ANN = DATA / "_orig_ccfv3_2017" / "annotation_25_fixed.nii.gz"
ALLEN_ONTOLOGY_URL = "http://api.brain-map.org/api/v2/structure_graph_download/1.json"


def _deref_u16_string(f: h5py.File, ref) -> str:
    """Decode a MATLAB UTF-16-LE char array referenced by `ref`."""
    a = np.asarray(f[ref][:]).flatten()
    return bytes(memoryview(a.astype(np.uint16))).decode("utf-16-le", errors="replace").rstrip("\x00")


def load_homer_parcels() -> pd.DataFrame:
    """Return DataFrame with columns: numid, region, subregion, centre_x/y/z, voxel_indices."""
    p = DATA / "corrs_mouse.mat"
    with h5py.File(str(p), "r") as f:
        g = f["m"]
        ht = [_deref_u16_string(f, r) for r in np.asarray(g["ht"][:]).flatten()]
        t_refs = np.asarray(g["t"][:])
        n_cols, n_nodes = t_refs.shape
        col = {name: ht.index(name) for name in ("numid", "region", "subregion", "center", "indices")}
        rows = []
        for n in range(n_nodes):
            numid  = int(np.asarray(f[t_refs[col["numid"], n]][:]).flatten()[0])
            region = _deref_u16_string(f, t_refs[col["region"], n])
            subreg = _deref_u16_string(f, t_refs[col["subregion"], n])
            ctr    = np.asarray(f[t_refs[col["center"], n]][:]).flatten().astype(float)
            vox    = np.asarray(f[t_refs[col["indices"], n]][:]).flatten().astype(np.int64)
            rows.append({
                "node":      n,
                "numid":     numid,
                "region":    region,
                "subregion": subreg,
                "centre_x":  ctr[0], "centre_y": ctr[1], "centre_z": ctr[2],
                "n_voxels":  len(vox),
                "voxel_indices": vox,
            })
    return pd.DataFrame(rows)


def load_allen_ontology() -> dict[int, dict]:
    """Allen mouse brain structure graph (id -> {acronym, name, ancestry})."""
    with urllib.request.urlopen(ALLEN_ONTOLOGY_URL, timeout=60) as r:
        tree = json.loads(r.read().decode("utf-8"))

    out = {}
    def walk(node, ancestors=()):
        nid = node.get("id")
        out[nid] = {
            "id": nid,
            "acronym": node.get("acronym"),
            "name":    node.get("name"),
            "ancestor_acronyms": ancestors,
            "ancestor_ids":      tuple(a["id"] for a in [_ for _ in []]),  # filled below
        }
        for c in (node.get("children") or []):
            walk(c, ancestors + ((node.get("acronym"), nid),))
    walk(tree["msg"][0])
    return out


def main():
    print("loading HOMER parcels (1864 nodes)...")
    df = load_homer_parcels()
    print(f"  loaded {len(df)} parcels")

    print(f"\nloading rsmask.nii (200 µm SS grid HOMER uses)...")
    rsmask_img = nib.load(str(DATA / "_mouse_mask" / "rsmask.nii"))
    rsmask_aff = rsmask_img.affine
    rsmask_shp = rsmask_img.shape
    print(f"  shape {rsmask_shp}  origin {rsmask_aff[:3,3]}  axcodes {nib.aff2axcodes(rsmask_aff)}")

    print(f"\nloading warpfield2SS (Paul's nonlinear DSURQE→CCFv3)...")
    wf2ss_img = nib.load(str(WF / "warpfield2SS.nii.gz"))
    wf2ss = np.asarray(wf2ss_img.dataobj)
    wf2ss_aff = wf2ss_img.affine
    wf2ss_inv = np.linalg.inv(wf2ss_aff)
    print(f"  shape {wf2ss.shape}  SS-grid affine origin {wf2ss_aff[:3,3]}  axcodes {nib.aff2axcodes(wf2ss_aff)}")

    print(f"\nloading fixed CCFv3 annotation (uint32, all 672 labels)...")
    ann_img = nib.load(str(FIXED_ANN))
    ann = np.asarray(ann_img.dataobj)
    ann_aff = ann_img.affine
    ann_inv = np.linalg.inv(ann_aff)
    print(f"  shape {ann.shape}  axcodes {nib.aff2axcodes(ann_aff)}")

    print(f"\nfetching Allen ontology (graph 1)...")
    ontology = load_allen_ontology()
    def lid_to_acr(lid):
        if lid == 0: return "(none)"
        info = ontology.get(int(lid))
        return info["acronym"] if info else f"id={lid}"
    print(f"  loaded {len(ontology)} structures")

    def warp_voxel_to_ccf_voxel(vox_1based_F: np.ndarray) -> np.ndarray:
        """Voxel set (1D MATLAB indices, Fortran order) -> per-voxel CCFv3 voxel ijk."""
        idx = vox_1based_F.astype(np.int64) - 1
        valid = (idx >= 0) & (idx < int(np.prod(rsmask_shp)))
        idx = idx[valid]
        ijk = np.array(np.unravel_index(idx, rsmask_shp, order="F")).T   # (n, 3)
        homog = np.column_stack([ijk, np.ones(len(ijk))])
        ss_world = (rsmask_aff @ homog.T).T[:, :3]                       # (n, 3) SS world mm
        # Convert SS world mm -> SS voxel index on Paul's 70 µm grid
        ss_homog = np.column_stack([ss_world, np.ones(len(ss_world))])
        ss_vox_f = (wf2ss_inv @ ss_homog.T).T[:, :3]
        ss_vox   = np.round(ss_vox_f).astype(int)
        # Read warpfield -> NS world mm
        ok = ((ss_vox[:, 0] >= 0) & (ss_vox[:, 0] < wf2ss.shape[0]) &
              (ss_vox[:, 1] >= 0) & (ss_vox[:, 1] < wf2ss.shape[1]) &
              (ss_vox[:, 2] >= 0) & (ss_vox[:, 2] < wf2ss.shape[2]))
        ns_world = np.full((len(ss_vox), 3), np.nan)
        for n_i, (i, j, k) in enumerate(ss_vox):
            if ok[n_i]:
                ns_world[n_i] = wf2ss[i, j, k]
        # Convert NS world mm -> CCFv3 voxel index
        ns_homog = np.column_stack([ns_world, np.ones(len(ns_world))])
        ns_vox_f = (ann_inv @ ns_homog.T).T[:, :3]
        ns_vox   = np.round(np.where(np.isnan(ns_vox_f), -1, ns_vox_f)).astype(int)
        return ss_world, ns_world, ns_vox

    print(f"\nrunning per-parcel warp + label lookup...")
    rows = []
    for _, p in df.iterrows():
        ss_world, ns_world, ns_vox = warp_voxel_to_ccf_voxel(p["voxel_indices"])

        # Centre via the stored centre xyz (more accurate than mean of voxel centroids,
        # since the stored centre may be intensity-weighted).
        ctr_ss = np.array([p["centre_x"], p["centre_y"], p["centre_z"]])
        ctr_ss_vox = np.round((wf2ss_inv @ np.array([*ctr_ss, 1.0]))[:3]).astype(int)
        if (0 <= ctr_ss_vox[0] < wf2ss.shape[0] and
            0 <= ctr_ss_vox[1] < wf2ss.shape[1] and
            0 <= ctr_ss_vox[2] < wf2ss.shape[2]):
            ctr_ns = wf2ss[ctr_ss_vox[0], ctr_ss_vox[1], ctr_ss_vox[2]]
            ctr_ns_vox = np.round((ann_inv @ np.array([*ctr_ns, 1.0]))[:3]).astype(int)
            if (0 <= ctr_ns_vox[0] < ann.shape[0] and
                0 <= ctr_ns_vox[1] < ann.shape[1] and
                0 <= ctr_ns_vox[2] < ann.shape[2]):
                centre_lid = int(ann[ctr_ns_vox[0], ctr_ns_vox[1], ctr_ns_vox[2]])
            else:
                centre_lid = 0
                ctr_ns = np.full(3, np.nan)
        else:
            centre_lid = 0
            ctr_ns = np.full(3, np.nan)
            ctr_ns_vox = np.full(3, -1, dtype=int)

        # Per-voxel labels for majority vote across the parcel
        per_vox_lids = []
        for i, j, k in ns_vox:
            if i < 0 or j < 0 or k < 0:
                continue
            if i >= ann.shape[0] or j >= ann.shape[1] or k >= ann.shape[2]:
                continue
            per_vox_lids.append(int(ann[i, j, k]))

        if per_vox_lids:
            cnt = Counter([l for l in per_vox_lids if l != 0])
            if cnt:
                maj_lid, maj_n = cnt.most_common(1)[0]
                agree = maj_n / len(per_vox_lids)
            else:
                maj_lid, agree = 0, 0.0
        else:
            maj_lid, agree = 0, 0.0

        rows.append({
            "node":           p["node"],
            "numid":          p["numid"],
            "region":         p["region"],
            "subregion":      p["subregion"][:120],
            "n_voxels":       p["n_voxels"],
            "centre_ss_x":    ctr_ss[0], "centre_ss_y": ctr_ss[1], "centre_ss_z": ctr_ss[2],
            "centre_ns_x":    float(ctr_ns[0]) if np.isfinite(ctr_ns[0]) else np.nan,
            "centre_ns_y":    float(ctr_ns[1]) if np.isfinite(ctr_ns[1]) else np.nan,
            "centre_ns_z":    float(ctr_ns[2]) if np.isfinite(ctr_ns[2]) else np.nan,
            "centre_ccfv3_i": int(ctr_ns_vox[0]),
            "centre_ccfv3_j": int(ctr_ns_vox[1]),
            "centre_ccfv3_k": int(ctr_ns_vox[2]),
            "centre_allen_id":      centre_lid,
            "centre_allen_acr":     lid_to_acr(centre_lid),
            "majority_allen_id":    maj_lid,
            "majority_allen_acr":   lid_to_acr(maj_lid),
            "majority_agreement":   float(agree),
            "n_voxels_labelled":   sum(1 for l in per_vox_lids if l != 0),
            "n_voxels_unlabelled": sum(1 for l in per_vox_lids if l == 0),
        })

        if (p["node"] + 1) % 500 == 0:
            print(f"  ...{p['node']+1}/{len(df)} parcels processed")

    out_df = pd.DataFrame(rows)
    out_path_pq = OUT / "parcel_ccfv3_labels.parquet"
    out_path_cs = OUT / "parcel_ccfv3_labels.csv"
    out_path_pq_alt = OUT / "parcel_ccfv3_labels.feather"

    # Try parquet, fall back to feather / csv-only if pyarrow not installed
    try:
        out_df.to_parquet(out_path_pq, index=False)
        print(f"\nsaved {out_path_pq}")
    except Exception as e:
        print(f"\nparquet save failed ({e}); trying feather")
        try:
            out_df.to_feather(out_path_pq_alt)
            print(f"saved {out_path_pq_alt}")
        except Exception as e2:
            print(f"  feather also failed ({e2}); CSV only")

    out_df.to_csv(out_path_cs, index=False)
    print(f"saved {out_path_cs}  ({len(out_df)} rows)")

    # Provenance JSON
    prov = {
        "n_parcels":  len(out_df),
        "rsmask_nii": str(DATA / "_mouse_mask" / "rsmask.nii"),
        "rsmask_shape": list(rsmask_shp),
        "rsmask_affine": rsmask_aff.tolist(),
        "warpfield":  str(WF / "warpfield2SS.nii.gz"),
        "warpfield_shape": list(wf2ss.shape),
        "warpfield_affine": wf2ss_aff.tolist(),
        "fixed_annotation": str(FIXED_ANN),
        "fixed_annotation_shape": list(ann.shape),
        "fixed_annotation_affine": ann_aff.tolist(),
        "ontology_url": ALLEN_ONTOLOGY_URL,
    }
    (OUT / "parcel_ccfv3_labels.json").write_text(json.dumps(prov, indent=2, default=str))

    # Quick headline numbers
    has_centre = (out_df["centre_allen_id"] > 0).sum()
    has_majority = (out_df["majority_allen_id"] > 0).sum()
    high_agreement = ((out_df["majority_agreement"] >= 0.5) & (out_df["majority_allen_id"] > 0)).sum()
    print(f"\nheadline:")
    print(f"  parcels with positive Allen label at centre:   {has_centre}/{len(out_df)} ({100*has_centre/len(out_df):.1f}%)")
    print(f"  parcels with positive majority-vote label:     {has_majority}/{len(out_df)} ({100*has_majority/len(out_df):.1f}%)")
    print(f"  parcels with ≥50% majority agreement:          {high_agreement}/{len(out_df)} ({100*high_agreement/len(out_df):.1f}%)")
    print(f"  unique majority acronyms assigned:             {out_df['majority_allen_acr'].nunique()}")
    print(f"  centre/majority acronym agreement:             "
          f"{(out_df['centre_allen_acr'] == out_df['majority_allen_acr']).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
