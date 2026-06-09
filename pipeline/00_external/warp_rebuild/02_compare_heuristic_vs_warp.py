"""Compare HOMER's heuristic 48-permutation transform against Paul's nonlinear
DSURQE→CCFv3 warpfield, per parcel.

We compare at two levels:

(A) **Summary-structure assignment.** HOMER's `mouse_sc_meta.json` already
    stores ``node_struct_idx`` — which Allen summary structure each of the
    1864 parcels was assigned to under the OLD brute-force transform. For
    the NEW warp, we resolve each parcel's centre acronym (from
    ``parcel_ccfv3_labels.csv``) up the Allen ontology to its summary
    ancestor. Agreement at this level is the most policy-relevant signal
    because the production SC matrix is built per summary structure.

(B) **World-mm displacement.** For each parcel centre, we compute the OLD
    transform's claimed CCFv3 world-mm position and Paul's NEW position,
    converted into a common frame (Paul's NS world coords) so we can take
    a Euclidean distance.

Output:
  - ``data_external/_warp_rebuild/heuristic_vs_warp.csv``  per-parcel diffs.
  - stdout summary: agreement rate, distribution of displacements.

Usage:
    PYTHONPATH=src python pipeline/00_external/warp_rebuild/02_compare_heuristic_vs_warp.py
"""
from __future__ import annotations

import json
import sys
import urllib.request
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


def _deref_u16_string(f, ref):
    a = np.asarray(f[ref][:]).flatten()
    return bytes(memoryview(a.astype(np.uint16))).decode("utf-16-le", errors="replace").rstrip("\x00")


def load_homer_centres():
    """Load (centres mm, region, subregion) for the 1864 parcels."""
    with h5py.File(str(DATA / "corrs_mouse.mat"), "r") as f:
        g = f["m"]
        ht = [_deref_u16_string(f, r) for r in np.asarray(g["ht"][:]).flatten()]
        t_refs = np.asarray(g["t"][:])
        n = t_refs.shape[1]
        col = {k: ht.index(k) for k in ("center", "region", "subregion")}
        centres = np.zeros((n, 3))
        region, subreg = [], []
        for i in range(n):
            centres[i] = np.asarray(f[t_refs[col["center"], i]][:]).flatten().astype(float)
            region.append(_deref_u16_string(f, t_refs[col["region"], i]))
            subreg.append(_deref_u16_string(f, t_refs[col["subregion"], i]))
    return centres, region, subreg


def load_allen_ontology():
    """Returns id -> {acronym, name, parent_id, ancestor_ids (set incl. self)}."""
    with urllib.request.urlopen(
        "http://api.brain-map.org/api/v2/structure_graph_download/1.json", timeout=60
    ) as r:
        tree = json.loads(r.read().decode("utf-8"))
    info = {}
    def walk(node, parent_id=None, ancestor_ids=()):
        nid = node.get("id")
        chain = ancestor_ids + ((nid,) if nid is not None else ())
        info[nid] = {
            "id":         nid,
            "acronym":    node.get("acronym"),
            "name":       node.get("name"),
            "parent_id":  parent_id,
            "ancestor_ids": set(chain),
        }
        for c in (node.get("children") or []):
            walk(c, nid, chain)
    walk(tree["msg"][0])
    return info


def main():
    print("loading HOMER parcels...")
    centres, regions, subregs = load_homer_centres()
    n_nodes = len(centres)
    print(f"  {n_nodes} parcels")

    # --- OLD transform: from data_external/_diagnostics/mouse_to_ccf_transform.json
    old_path = ROOT / "data_external" / "_diagnostics" / "mouse_to_ccf_transform.json"
    if not old_path.exists():
        print(f"OLD transform file not found at {old_path}")
        sys.exit(1)
    old_t = json.loads(old_path.read_text())
    perm  = old_t["perm"]
    signs = old_t["signs"]
    shift = np.asarray(old_t["shift_mm"])
    print(f"\nOLD heuristic transform: perm={perm} signs={signs} shift_mm={shift.tolist()}")

    def apply_old_transform(centres):
        """centres (N,3) in rsmask world mm -> OLD-style CCFv3 'mm' (= voxel-indexed)."""
        out = np.column_stack([
            signs[0] * centres[:, perm[0]],
            signs[1] * centres[:, perm[1]],
            signs[2] * centres[:, perm[2]],
        ])
        return out + shift

    old_ccf = apply_old_transform(centres)
    # In the OLD pipeline, ccf_ijk = (ccf_world / res_mm).astype(int) at 100 µm
    # for the structure annotation (01_mouse_sc.py uses 100 µm).
    res_old = 0.1
    old_ijk = (old_ccf / res_old).astype(int)
    print(f"OLD ccf_ijk bbox: i in [{old_ijk[:,0].min()},{old_ijk[:,0].max()}]  "
          f"j in [{old_ijk[:,1].min()},{old_ijk[:,1].max()}]  k in [{old_ijk[:,2].min()},{old_ijk[:,2].max()}]")

    # --- NEW warp: Paul's warpfield2SS
    print("\nNEW warp: Paul's warpfield2SS")
    wf_img = nib.load(str(WF / "warpfield2SS.nii.gz"))
    wf2ss = np.asarray(wf_img.dataobj)
    wf_aff = wf_img.affine
    wf_inv = np.linalg.inv(wf_aff)

    ss_homog = np.column_stack([centres, np.ones(n_nodes)])
    ss_vox_f = (wf_inv @ ss_homog.T).T[:, :3]
    ss_vox = np.round(ss_vox_f).astype(int)
    ns_world_new = np.full((n_nodes, 3), np.nan)
    for i in range(n_nodes):
        x, y, z = ss_vox[i]
        if 0 <= x < wf2ss.shape[0] and 0 <= y < wf2ss.shape[1] and 0 <= z < wf2ss.shape[2]:
            ns_world_new[i] = wf2ss[x, y, z]
    print(f"NEW NS world bbox: x in [{np.nanmin(ns_world_new[:,0]):.2f},{np.nanmax(ns_world_new[:,0]):.2f}]  "
          f"y in [{np.nanmin(ns_world_new[:,1]):.2f},{np.nanmax(ns_world_new[:,1]):.2f}]  "
          f"z in [{np.nanmin(ns_world_new[:,2]):.2f},{np.nanmax(ns_world_new[:,2]):.2f}]")

    # --- Convert OLD (i,j,k)*100µm into Paul's NS world frame for displacement comparison.
    # OLD output (a,b,c) at 100 µm corresponds to PIR voxel indices (i_AP, j_DV, k_LR) at 100 µm,
    # which in Paul's NS 25 µm voxel space = (4a/0.1, 4b/0.1, 4c/0.1) = (40a, 40b, 40c) voxels,
    # but actually OLD (a,b,c) is already mm so just need to map (mm_AP, mm_DV, mm_LR) to Paul's
    # world (x_LR, y_AP_anterior, z_DV_superior).
    # Paul's NS affine: voxel (i,j,k) -> (0.025*k, -0.025*i, -0.025*j)
    # So Paul-world from voxel = (LR_mm, -AP_mm, -DV_mm)
    # OLD-mm (AP_mm, DV_mm, LR_mm) -> Paul-world (LR_mm, -AP_mm, -DV_mm) = (c, -a, -b)
    old_ns_world = np.column_stack([old_ccf[:, 2], -old_ccf[:, 0], -old_ccf[:, 1]])

    # --- Displacement
    disp = old_ns_world - ns_world_new
    disp_mag = np.linalg.norm(disp, axis=1)
    finite = np.isfinite(disp_mag)
    print(f"\nPer-parcel displacement |OLD - NEW| in Paul's NS world (mm):")
    print(f"  finite values: {finite.sum()}/{n_nodes}")
    print(f"  mean: {np.nanmean(disp_mag):.3f} mm")
    print(f"  median: {np.nanmedian(disp_mag):.3f} mm")
    print(f"  min: {np.nanmin(disp_mag):.3f} mm")
    print(f"  max: {np.nanmax(disp_mag):.3f} mm")
    print(f"  pct >  1 mm: {100*(disp_mag[finite] > 1.0).mean():.1f}%")
    print(f"  pct >  2 mm: {100*(disp_mag[finite] > 2.0).mean():.1f}%")
    print(f"  pct >  5 mm: {100*(disp_mag[finite] > 5.0).mean():.1f}%")

    # --- Summary-structure agreement
    print("\nloading Allen ontology + NEW per-parcel labels...")
    ontology = load_allen_ontology()
    new_df = pd.read_csv(OUT / "parcel_ccfv3_labels.csv")
    new_centre_lid = new_df["centre_allen_id"].astype(int).to_numpy()

    # OLD: load node_struct_idx from mouse_sc_meta.json
    sc_meta = json.loads((ROOT / "data_external" / "mouse_sc_meta.json").read_text())
    old_struct_idx = np.asarray(sc_meta["node_struct_idx"], dtype=np.int64)
    old_struct_acrs = sc_meta["structure_acronyms"]

    # Resolve old struct index to summary id+acronym
    # We need the Allen id for each summary acronym. AllenSDK summary structures
    # are the standard 290-region set; map acronym back to id via the ontology.
    acr_to_id = {info["acronym"]: nid for nid, info in ontology.items() if info["acronym"]}
    old_summary_id_per_parcel = np.zeros(n_nodes, dtype=np.int64)
    old_summary_acr_per_parcel = []
    for i, sidx in enumerate(old_struct_idx):
        if sidx < 0 or sidx >= len(old_struct_acrs):
            old_summary_id_per_parcel[i] = 0
            old_summary_acr_per_parcel.append("(unmapped)")
        else:
            acr = old_struct_acrs[sidx]
            old_summary_acr_per_parcel.append(acr)
            old_summary_id_per_parcel[i] = acr_to_id.get(acr, 0)

    # For NEW: find which summary-structure each centre's fine label ancestrally
    # belongs to. Summary structures are a defined set; for each NEW fine label,
    # walk up the parent chain and return the first ancestor that's in
    # old_struct_acrs (the production summary set).
    summary_acr_set = set(old_struct_acrs)
    summary_id_set  = {acr_to_id.get(a, 0) for a in old_struct_acrs if a in acr_to_id}

    new_summary_acr_per_parcel = []
    for i in range(n_nodes):
        lid = int(new_centre_lid[i])
        if lid == 0:
            new_summary_acr_per_parcel.append("(unlabelled)")
            continue
        info = ontology.get(lid)
        if not info:
            new_summary_acr_per_parcel.append("(unknown)")
            continue
        # Walk ancestors, find first that's in the summary set
        found = None
        # The 'ancestor_ids' set is unordered; for traversal we need parent chain.
        # Rebuild parent chain from parent_id.
        cur = lid
        chain = [cur]
        while cur:
            cur = ontology.get(cur, {}).get("parent_id")
            if cur:
                chain.append(cur)
        for aid in chain:
            acr = ontology.get(aid, {}).get("acronym")
            if acr in summary_acr_set:
                found = acr
                break
        new_summary_acr_per_parcel.append(found or "(no_summary_ancestor)")

    # Agreement
    agree = np.array([old_summary_acr_per_parcel[i] == new_summary_acr_per_parcel[i]
                      for i in range(n_nodes)])
    valid = np.array([old_summary_acr_per_parcel[i] not in ("(unmapped)",)
                       and new_summary_acr_per_parcel[i] not in ("(unlabelled)", "(unknown)", "(no_summary_ancestor)")
                       for i in range(n_nodes)])
    print(f"\nSummary-structure agreement (OLD ↔ NEW):")
    print(f"  parcels with both assignments valid:  {valid.sum()}/{n_nodes}")
    print(f"  agreement among valid:                {100*agree[valid].mean():.1f}%")
    print(f"  unique summary acronyms (OLD):        {len(set(old_summary_acr_per_parcel))}")
    print(f"  unique summary acronyms (NEW):        {len(set(new_summary_acr_per_parcel))}")

    # --- Save per-parcel table
    out_df = pd.DataFrame({
        "node": np.arange(n_nodes),
        "region": regions,
        "subregion": [s[:60] for s in subregs],
        "centre_ss_x": centres[:, 0],
        "centre_ss_y": centres[:, 1],
        "centre_ss_z": centres[:, 2],
        "old_ccf_a_mm": old_ccf[:, 0],
        "old_ccf_b_mm": old_ccf[:, 1],
        "old_ccf_c_mm": old_ccf[:, 2],
        "old_ns_world_x": old_ns_world[:, 0],
        "old_ns_world_y": old_ns_world[:, 1],
        "old_ns_world_z": old_ns_world[:, 2],
        "new_ns_world_x": ns_world_new[:, 0],
        "new_ns_world_y": ns_world_new[:, 1],
        "new_ns_world_z": ns_world_new[:, 2],
        "displacement_mm": disp_mag,
        "old_summary_acr": old_summary_acr_per_parcel,
        "new_summary_acr": new_summary_acr_per_parcel,
        "summary_agree":   agree,
    })
    out_path = OUT / "heuristic_vs_warp.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nsaved {out_path}  ({len(out_df)} rows)")

    # Top regions where old and new disagree
    disagreement = out_df[~out_df["summary_agree"] & valid].copy()
    disagreement["transition"] = disagreement["old_summary_acr"] + " -> " + disagreement["new_summary_acr"]
    top = disagreement["transition"].value_counts().head(15)
    print(f"\nTop summary-structure transitions where OLD and NEW disagree (top 15):")
    for t, c in top.items():
        print(f"  {c:>4d}  {t}")


if __name__ == "__main__":
    main()
