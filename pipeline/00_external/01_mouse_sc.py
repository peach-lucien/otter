"""Project the Allen Mouse Connectivity Atlas SC onto the colleague's 1864-node
parcellation, using the mouse → CCFv3 transform from 00c.

.. deprecated:: v2
    LEGACY (v1 only). This script depends on the heuristic 48-permutation
    transform from ``00c_align_mouse_to_ccf.py`` to assign each parcel a
    CCFv3 summary structure. The v2 successor ``01b_mouse_sc_v2.py`` reads
    the pre-warped voxel centre ``ns_center_ix`` directly from
    ``corrs_mouse_v2.mat`` (Paul's nonlinear DSURQE -> CCFv3 warp) and is
    the production path. Use this script only when working from the v1
    mouse package.

Pipeline:
  1. Use AllenSDK to download CCFv3 annotation + structure-level SC matrix
     (Oh et al. 2014 normalised projection volumes, summary structures only).
  2. Load the colleague→CCFv3 transform from 00c.
  3. For each of the 1864 mouse nodes:
       a. Get its (x,y,z) centre in the colleague's bregma-centred mm.
       b. Apply the transform → CCFv3 world mm.
       c. Convert to CCFv3 voxel index.
       d. Look up the CCFv3 region ID at that voxel.
       e. Map up the structure tree to find which 'summary structure' it belongs to.
  4. Build the (1864 × 1864) SC matrix by indexing the structure-level matrix.

Output: data_external/mouse_sc.npy + mouse_sc_meta.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))               # for _mouse_transform

from homer.data import load_metadata, parse_t_table         # noqa: E402
from _mouse_transform import load_transform, apply_transform # noqa: E402

OUT  = ROOT / "data_external"; OUT.mkdir(parents=True, exist_ok=True)
DIAG = OUT / "_diagnostics"


def main(resolution_um: int = 100):
    try:
        from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
    except ImportError:
        print("ERROR: pip install allensdk")
        sys.exit(1)

    transform = load_transform(DIAG)
    print(f"using mouse→CCFv3 transform (coverage at fit: {transform['coverage']:.1%})")

    cache_dir = Path.home() / ".allensdk_cache"
    print(f"Allen SDK cache → {cache_dir}")
    mcc = MouseConnectivityCache(resolution=resolution_um,
                                 manifest_file=str(cache_dir / "manifest.json"))

    # -- 1. CCFv3 annotation + structure tree --------------------------------
    print("downloading CCFv3 annotation...")
    ann, _ = mcc.get_annotation_volume()
    print(f"  annotation shape={ann.shape} (CCFv3 at {resolution_um} µm)")

    print("downloading structure tree...")
    structure_tree = mcc.get_structure_tree()
    summary = structure_tree.get_structures_by_set_id([167587189])
    summary_ids = [int(s["id"]) for s in summary]
    summary_names = [s["acronym"] for s in summary]
    print(f"  {len(summary_ids)} summary structures")
    sid_to_idx = {sid: i for i, sid in enumerate(summary_ids)}

    # -- 2. Build structure-level SC matrix from per-experiment unionizes ----
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
            tqdm.write(f"  skip {exp['id']}: {e}"); continue
        src_id = exp["structure_id"]
        if src_id not in sid_to_idx: continue
        i = sid_to_idx[src_id]
        for _, row in unionizes.iterrows():
            j = sid_to_idx.get(int(row["structure_id"]))
            if j is None: continue
            SC[i, j] += float(row["normalized_projection_volume"])
            counts[i, j] += 1
    SC = np.where(counts > 0, SC / np.maximum(counts, 1), 0.0)
    print(f"  SC density (non-zero entries): {(SC > 0).mean():.2%}")

    # -- 3. Project each of 1864 nodes onto a CCFv3 region -------------------
    print("projecting 1864 mouse nodes onto CCFv3 regions...")
    meta = load_metadata("mouse")
    df = parse_t_table(meta["t"], meta["ht"])
    n_nodes = len(df)
    centres = df[["x", "y", "z"]].values

    # Per-node centre → CCFv3 voxel index
    res_mm = resolution_um / 1000.0
    ccf_world = apply_transform(centres, transform)
    ccf_ijk = (ccf_world / res_mm).astype(np.int64)

    in_bounds = ((ccf_ijk[:, 0] >= 0) & (ccf_ijk[:, 0] < ann.shape[0]) &
                 (ccf_ijk[:, 1] >= 0) & (ccf_ijk[:, 1] < ann.shape[1]) &
                 (ccf_ijk[:, 2] >= 0) & (ccf_ijk[:, 2] < ann.shape[2]))
    node_region = np.zeros(n_nodes, dtype=np.int64)
    ok = np.where(in_bounds)[0]
    node_region[ok] = ann[ccf_ijk[ok, 0], ccf_ijk[ok, 1], ccf_ijk[ok, 2]]
    n_in_brain = int((node_region > 0).sum())
    print(f"  {n_in_brain}/{n_nodes} nodes assigned a CCFv3 region "
          f"({n_in_brain/n_nodes:.1%})")

    # Optional: also use voxel-level lookups for nodes whose centre fell outside
    # CCFv3 brain — sample over all the node's voxels and take the mode.
    # (This is a robustness fallback, not on the hot path; commented out for now.)

    # -- 4. Map fine CCFv3 region → summary structure (via ancestry) ---------
    print("mapping CCFv3 IDs → summary structures via ancestry...")
    descendants = {sid: set(structure_tree.descendant_ids([sid])[0]) for sid in summary_ids}
    node_struct_idx = np.full(n_nodes, -1, dtype=np.int64)
    n_unmapped = 0
    for i, reg in enumerate(node_region):
        if reg == 0: n_unmapped += 1; continue
        for sid in summary_ids:
            if reg in descendants[sid]:
                node_struct_idx[i] = sid_to_idx[sid]; break
        else:
            n_unmapped += 1
    print(f"  {n_unmapped}/{n_nodes} nodes had no summary-structure ancestor")

    # -- 5. Build (1864, 1864) SC matrix -------------------------------------
    print("assembling per-node SC matrix...")
    SC_node = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    valid_idx = np.where(node_struct_idx >= 0)[0]
    for i in valid_idx:
        for j in valid_idx:
            SC_node[i, j] = SC[node_struct_idx[i], node_struct_idx[j]]

    # Symmetrise: SC tracer is directional, but for our FGW use we want a
    # symmetric relational cost. Use the average of the two directions.
    SC_node = 0.5 * (SC_node + SC_node.T)

    np.save(OUT / "mouse_sc.npy", SC_node)
    meta_out = {
        "source": "Allen Mouse Connectivity Atlas (Oh et al. 2014, Nature) via AllenSDK",
        "resolution_um":     resolution_um,
        "transform_used":    transform,
        "n_structures":      len(summary_ids),
        "structure_acronyms": summary_names,
        "n_nodes":           int(n_nodes),
        "n_in_brain":        int(n_in_brain),
        "n_unmapped":        int(n_unmapped),
        "node_struct_idx":   node_struct_idx.tolist(),
        "symmetrised":       True,
    }
    (OUT / "mouse_sc_meta.json").write_text(json.dumps(meta_out, indent=2, default=str))
    print(f"\nsaved → {OUT / 'mouse_sc.npy'}")
    print(f"        {OUT / 'mouse_sc_meta.json'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution-um", type=int, default=100)
    main(ap.parse_args().resolution_um)
