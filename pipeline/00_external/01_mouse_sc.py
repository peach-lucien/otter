"""Build mouse_sc.npy from each parcel's CCFv3 centre voxel.

Reads ``ns_center_ix`` from the mouse ``.mat`` file and looks up the CCFv3
25 µm annotation at that voxel directly. The voxel indices are pre-warped
into CCFv3 space (nonlinear DSURQE→CCFv3 registration), so no coordinate
transform is applied here.

The summary-structure SC matrix itself comes from the Allen Mouse
Connectivity Atlas (Oh et al. 2014) summary-level unionised projection
volumes via AllenSDK; this script maps each parcel to a summary structure
using its CCFv3 centre voxel (``ns_center_ix``).

Outputs:
  - ``data_external/mouse_sc.npy``         shape (1864, 1864) float32
  - ``data_external/mouse_sc_meta.json``   provenance + node_struct_idx

Usage:
    PYTHONPATH=src python pipeline/00_external/01_mouse_sc.py
"""
from __future__ import annotations

import importlib.util
import importlib.machinery
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT.parent / "data_crossspecies"
V2_DIR = DATA / "updated_connectom_0906_26"
OUT = ROOT / "data_external"
OUT.mkdir(parents=True, exist_ok=True)


def _load_io():
    pkg_otter = importlib.util.module_from_spec(importlib.machinery.ModuleSpec("otter", None))
    pkg_data  = importlib.util.module_from_spec(importlib.machinery.ModuleSpec("otter.data", None))
    sys.modules.setdefault("otter", pkg_otter)
    sys.modules.setdefault("otter.data", pkg_data)
    spec = importlib.util.spec_from_file_location("otter.data.io", ROOT / "src/otter/data/io.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["otter.data.io"] = mod
    spec.loader.exec_module(mod)
    return mod


def main(resolution_um: int = 100):
    IO = _load_io()

    try:
        from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
    except ImportError:
        print("ERROR: pip install allensdk")
        sys.exit(1)

    cache_dir = Path.home() / ".allensdk_cache"
    print(f"Allen SDK cache → {cache_dir}")
    mcc = MouseConnectivityCache(
        resolution=resolution_um,
        manifest_file=str(cache_dir / "manifest.json"),
    )

    # -- 1. AllenSDK CCFv3 annotation (still used to derive summary IDs) ----
    # Note: a shipped 25 µm CCFv3 annotation (ANO_ABA_NS.nii.gz) is also
    # available alongside the mouse package, but we use the AllenSDK version
    # for the structure-tree mapping (descendant_ids), which is independent
    # of voxel-level lookups.
    print("downloading CCFv3 annotation via AllenSDK...")
    ann, _ = mcc.get_annotation_volume()
    print(f"  AllenSDK annotation shape={ann.shape} at {resolution_um} µm")

    print("downloading structure tree...")
    structure_tree = mcc.get_structure_tree()
    summary = structure_tree.get_structures_by_set_id([167587189])
    summary_ids = [int(s["id"]) for s in summary]
    summary_names = [s["acronym"] for s in summary]
    print(f"  {len(summary_ids)} summary structures")
    sid_to_idx = {sid: i for i, sid in enumerate(summary_ids)}

    # -- 2. Build structure-level SC from unionised projection volumes ------
    # (Independent of parcel placement.)
    print("building structure-level SC (slow)...")
    experiments = mcc.get_experiments(injection_structure_ids=summary_ids, cre=False)
    n_struct = len(summary_ids)
    SC = np.zeros((n_struct, n_struct), dtype=np.float64)
    counts = np.zeros((n_struct, n_struct), dtype=np.int32)
    from tqdm import tqdm
    for exp in tqdm(experiments, desc="experiments"):
        try:
            unionizes = mcc.get_structure_unionizes(
                experiment_ids=[exp["id"]], structure_ids=summary_ids,
                hemisphere_ids=[3], is_injection=False,
            )
        except Exception as e:
            tqdm.write(f"  skip {exp['id']}: {e}")
            continue
        src_id = exp["structure_id"]
        if src_id not in sid_to_idx:
            continue
        i = sid_to_idx[src_id]
        for _, row in unionizes.iterrows():
            j = sid_to_idx.get(int(row["structure_id"]))
            if j is None:
                continue
            SC[i, j] += float(row["normalized_projection_volume"])
            counts[i, j] += 1
    SC = np.where(counts > 0, SC / np.maximum(counts, 1), 0.0)
    print(f"  SC density (non-zero entries): {(SC > 0).mean():.2%}")

    # -- 3. Project each of 1864 nodes onto a CCFv3 region -----------------
    print("projecting 1864 mouse nodes onto CCFv3 regions via ns_center_ix...")
    meta = IO.load_metadata("mouse")
    if meta["_schema"] != "v2":
        print(f"  WARNING: the mouse parcel table is missing the pre-warped "
              f"voxel-index columns. Point DATA_DIR at the mouse package "
              f"that ships them.")
    df = IO.parse_t_table(meta["t"], meta["ht"])
    n_nodes = len(df)

    # ns_center_ix is 0-based linear index into the 25 µm NS grid (528, 320, 456).
    # AllenSDK gives us the 100 µm version (132, 80, 114), same PIR layout
    # but 4× coarser per axis. Downsample by integer division.
    ns_ix_25um = df["ns_center_ix"].to_numpy().astype(np.int64)
    ijk_25 = np.column_stack(np.unravel_index(ns_ix_25um, IO._NS_SHAPE, order="F"))
    # Allen 100 µm grid is 1/4 the resolution
    scale = 25 // resolution_um if resolution_um <= 25 else resolution_um // 25
    if resolution_um == 25:
        ijk = ijk_25
    elif resolution_um == 100:
        ijk = ijk_25 // 4
    elif resolution_um == 200:
        ijk = ijk_25 // 8
    else:
        raise ValueError(f"unsupported resolution_um={resolution_um}; use 25, 100, or 200")

    in_bounds = ((ijk[:, 0] >= 0) & (ijk[:, 0] < ann.shape[0]) &
                 (ijk[:, 1] >= 0) & (ijk[:, 1] < ann.shape[1]) &
                 (ijk[:, 2] >= 0) & (ijk[:, 2] < ann.shape[2]))
    node_region = np.zeros(n_nodes, dtype=np.int64)
    ok = np.where(in_bounds)[0]
    node_region[ok] = ann[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
    n_in_brain = int((node_region > 0).sum())
    print(f"  {n_in_brain}/{n_nodes} nodes assigned a CCFv3 region "
          f"({n_in_brain/n_nodes:.1%})")

    # -- 4. Map fine CCFv3 region → summary structure (via ancestry) -------
    print("mapping CCFv3 IDs → summary structures via ancestry...")
    descendants = {
        sid: set(structure_tree.descendant_ids([sid])[0]) for sid in summary_ids
    }
    node_struct_idx = np.full(n_nodes, -1, dtype=np.int64)
    n_unmapped = 0
    for i, reg in enumerate(node_region):
        if reg == 0:
            n_unmapped += 1
            continue
        for sid in summary_ids:
            if reg in descendants[sid]:
                node_struct_idx[i] = sid_to_idx[sid]
                break
        else:
            n_unmapped += 1
    print(f"  {n_unmapped}/{n_nodes} nodes had no summary-structure ancestor")

    # -- 5. Build (1864, 1864) SC matrix -----------------------------------
    print("assembling per-node SC matrix...")
    SC_node = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    valid_idx = np.where(node_struct_idx >= 0)[0]
    for i in valid_idx:
        for j in valid_idx:
            SC_node[i, j] = SC[node_struct_idx[i], node_struct_idx[j]]
    SC_node = 0.5 * (SC_node + SC_node.T)

    np.save(OUT / "mouse_sc.npy", SC_node)
    meta_out = {
        "source": "Allen Mouse Connectivity Atlas (Oh et al. 2014, Nature) via AllenSDK",
        "schema_loaded": meta["_schema"],
        "resolution_um":     resolution_um,
        "n_structures":      len(summary_ids),
        "structure_acronyms": summary_names,
        "n_nodes":           int(n_nodes),
        "n_in_brain":        int(n_in_brain),
        "n_unmapped":        int(n_unmapped),
        "node_struct_idx":   node_struct_idx.tolist(),
        "symmetrised":       True,
        "ns_source": "ns_center_ix (nonlinear DSURQE→CCFv3 warp)",
    }
    (OUT / "mouse_sc_meta.json").write_text(json.dumps(meta_out, indent=2, default=str))
    print(f"\nsaved → {OUT / 'mouse_sc.npy'}")
    print(f"        {OUT / 'mouse_sc_meta.json'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution-um", type=int, default=100,
                    help="AllenSDK annotation resolution (25, 100, or 200 µm).")
    main(ap.parse_args().resolution_um)
