"""Recompute ``node_struct_idx`` (Allen summary-structure index per parcel)
under Paul's nonlinear warp.

The existing ``mouse_sc_meta.json`` stores ``node_struct_idx`` and
``structure_acronyms`` derived under the OLD 48-permutation heuristic. This
script regenerates ``node_struct_idx`` from the NEW warpfield2SS pipeline:

  parcel centre (SS world mm) → warpfield2SS (SS voxel) → NS world mm
        → CCFv3 annotation lookup → structure-tree ancestry → summary id

Once Allen unionised projection volumes are re-downloaded (separate step),
the (n_struct × n_struct) SC matrix can be re-indexed with this new
``node_struct_idx`` to produce a corrected ``mouse_sc.npy`` — no need to
re-run the full unionise download because the structure-level SC matrix
itself doesn't depend on parcel placement.

Output:
  - ``data_external/_warp_rebuild/node_struct_idx_warped.json``
       new index per parcel + summary acronym mapping + diff vs OLD

Usage:
    PYTHONPATH=src python pipeline/00_external/warp_rebuild/03_new_node_struct_idx.py
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT.parent / "data_crossspecies"
WF   = DATA / "warpfields"
OUT  = ROOT / "data_external" / "_warp_rebuild"


def main():
    # Load Allen ontology
    print("loading Allen ontology...")
    with urllib.request.urlopen(
        "http://api.brain-map.org/api/v2/structure_graph_download/1.json", timeout=60
    ) as r:
        tree = json.loads(r.read().decode("utf-8"))

    ontology = {}
    def walk(node, parent=None):
        nid = node.get("id")
        ontology[nid] = {
            "id": nid,
            "acronym": node.get("acronym"),
            "name":    node.get("name"),
            "parent_id": parent,
        }
        for c in (node.get("children") or []):
            walk(c, nid)
    walk(tree["msg"][0])
    print(f"  {len(ontology)} structures")

    # Load the per-parcel labels we computed in script 01
    labels_path = OUT / "parcel_ccfv3_labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Missing {labels_path}. Run "
            f"pipeline/00_external/warp_rebuild/01_parcel_ccf_labels.py first."
        )
    df = pd.read_csv(labels_path)
    print(f"loaded {len(df)} parcel labels")

    # Existing summary structures the production SC matrix was indexed against
    sc_meta = json.loads((ROOT / "data_external" / "mouse_sc_meta.json").read_text())
    summary_acrs = sc_meta["structure_acronyms"]
    acr_to_idx = {a: i for i, a in enumerate(summary_acrs)}
    summary_acr_set = set(summary_acrs)
    print(f"production summary set: {len(summary_acrs)} structures")

    # For each parcel's centre fine label, find first ancestor in the summary set
    new_struct_idx = np.full(len(df), -1, dtype=np.int64)
    new_summary_acr = []
    for i, lid in enumerate(df["centre_allen_id"].astype(int).tolist()):
        if lid == 0:
            new_summary_acr.append("(unlabelled)")
            continue
        cur = lid
        chain = []
        while cur:
            chain.append(cur)
            cur = ontology.get(cur, {}).get("parent_id")
        found = None
        for aid in chain:
            acr = ontology.get(aid, {}).get("acronym")
            if acr in summary_acr_set:
                found = acr
                break
        if found is None:
            new_summary_acr.append("(no_summary_ancestor)")
        else:
            new_struct_idx[i] = acr_to_idx[found]
            new_summary_acr.append(found)

    # Compare to old node_struct_idx
    old_idx = np.asarray(sc_meta["node_struct_idx"], dtype=np.int64)
    valid_old = old_idx >= 0
    valid_new = new_struct_idx >= 0
    both_valid = valid_old & valid_new
    agree = (new_struct_idx == old_idx) & both_valid

    print(f"\nparcels mapped to a summary structure:")
    print(f"  OLD: {valid_old.sum()}/{len(df)}")
    print(f"  NEW: {valid_new.sum()}/{len(df)}")
    print(f"  both valid: {both_valid.sum()}")
    print(f"  agree at summary level among both-valid: {agree[both_valid].sum()}/{both_valid.sum()} "
          f"({100*agree[both_valid].mean():.1f}%)")

    # Save
    out = {
        "n_parcels": int(len(df)),
        "summary_acronyms": summary_acrs,  # same order as production SC matrix indices
        "node_struct_idx_old": old_idx.tolist(),
        "node_struct_idx_new": new_struct_idx.tolist(),
        "node_struct_acr_new": new_summary_acr,
        "n_valid_old": int(valid_old.sum()),
        "n_valid_new": int(valid_new.sum()),
        "n_agree":     int(agree[both_valid].sum()),
        "n_both_valid": int(both_valid.sum()),
        "source": "warpfield2SS centre -> CCFv3 fine label -> walk to first summary ancestor",
        "notes": (
            "Drop-in replacement for mouse_sc_meta.json['node_struct_idx']. The "
            "production (n_struct × n_struct) summary-structure SC matrix is "
            "independent of parcel placement, so re-indexing it with these new "
            "values produces a corrected per-parcel SC matrix without re-running "
            "the Allen unionise download."
        ),
    }
    out_path = OUT / "node_struct_idx_warped.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {out_path}")

    # Top transitions
    df["old_acr"] = [summary_acrs[i] if i >= 0 else "(unmapped)" for i in old_idx]
    df["new_acr"] = new_summary_acr
    diff_mask = (df["old_acr"] != df["new_acr"]) & valid_old & valid_new
    df["transition"] = df["old_acr"] + " -> " + df["new_acr"]
    top = df.loc[diff_mask, "transition"].value_counts().head(20)
    print(f"\nTop 20 OLD -> NEW summary-structure transitions:")
    for t, n in top.items():
        print(f"  {n:>4d}  {t}")


if __name__ == "__main__":
    main()
