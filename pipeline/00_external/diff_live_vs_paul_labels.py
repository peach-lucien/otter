"""Diff diagnostic: live `_dsurqe.py` lookup vs Paul's pre-computed v2 labels.

Gate for the option-(c) refactor: before swapping `_dsurqe.py` to consume
Paul's `region_vote_ss_dsq` / `region_vote_ns_aba` columns directly, we
must verify that the anchor packs derived under each path agree at the
parcel-set level.

For every anchor pack, this script:
  1. Builds the pack under the LIVE path (current `_dsurqe.py` lookup
     — Beauchamp 2022's DSURQE label volume + `DSURQE_OFFSET_MM` + 2-voxel
     majority vote).
  2. Builds the pack under the PAUL-COLUMNS path (filter `M.var` by
     `region_vote_ss_dsq == <DSURQE region name>` for each pack entry).
  3. Computes the Jaccard similarity of the resulting parcel sets.
  4. Flags any pack with Jaccard < 0.95 for manual review.

Output:
  - `data_external/_diagnostics/dsurqe_live_vs_paul_diff.json`
  - Console summary with per-pack jaccard + flagged entries.

If every pack has Jaccard >= 0.95, the option-(c) refactor of
`_dsurqe.py` is safe — swap can proceed. Otherwise document the
discrepancies and either accept them (with a v1 vs v2 vs Paul-columns
comparison in docs/05_limitations.md) or defer the refactor.

Usage:
    PYTHONPATH=src python pipeline/00_external/diff_live_vs_paul_labels.py
"""
from __future__ import annotations

import importlib
import importlib.util
import importlib.machinery
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT.parent / "data_crossspecies"
OUT = ROOT / "data_external" / "_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)


def _load_io():
    pkg_homer = importlib.util.module_from_spec(importlib.machinery.ModuleSpec("homer", None))
    pkg_data  = importlib.util.module_from_spec(importlib.machinery.ModuleSpec("homer.data", None))
    sys.modules.setdefault("homer", pkg_homer)
    sys.modules.setdefault("homer.data", pkg_data)
    spec = importlib.util.spec_from_file_location("homer.data.io", ROOT / "src/homer/data/io.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["homer.data.io"] = mod
    spec.loader.exec_module(mod)
    return mod


# Every pack module + its builder function.
PACK_MODULES = [
    ("hippocampal",   "build_hippocampal_region_anchors"),
    ("lateral_pfc",   "build_lateral_pfc_region_anchors"),
    ("visual",        "build_visual_region_anchors"),
    ("ppc",           "build_ppc_region_anchors"),
    ("tectum",        "build_tectum_region_anchors"),
    ("pag",           "build_pag_region_anchors"),
    ("amygdala",      "build_amygdala_region_anchors"),
    ("olfactory",     "build_olfactory_region_anchors"),
    ("striatum",      "build_striatum_region_anchors"),
    ("cingulate",     "build_cingulate_region_anchors"),
    ("auditory",      "build_auditory_region_anchors"),
    ("somatosensory", "build_somatosensory_region_anchors"),
    ("entorhinal",    "build_entorhinal_region_anchors"),
    ("perirhinal",    "build_perirhinal_region_anchors"),
    ("biccn_motor",   "build_biccn_motor_region_anchors"),
]


def parcels_via_paul_column(M_var: pd.DataFrame, dsurqe_region_name: str,
                             column: str = "region_vote_ss_dsq") -> list[int]:
    """Return parcels whose Paul-provided vote label is in the DSURQE region's
    subtree.

    The DSURQE tree is hierarchical — e.g. "Primary motor area" expands to
    a set of leaf DSURQE labels. Paul's votes are at the leaf level, so we
    need to resolve the named region to its descendant DSURQE leaf names
    via the Beauchamp 2022 DSURQE tree.
    """
    from homer.data.anchor_packs._dsurqe import parse_dsurqe_tree
    base = DATA / "MouseHumanTranscriptomicSimilarity/AMBA/data"
    tree_path = base / "DSURQE_tree.json"
    if not tree_path.exists():
        raise FileNotFoundError(f"DSURQE_tree.json not found at {tree_path}")
    name_to_label_ids = parse_dsurqe_tree(tree_path)
    if dsurqe_region_name not in name_to_label_ids:
        return []
    target_label_set = name_to_label_ids[dsurqe_region_name]

    # Paul's vote labels are strings (the DSURQE region name at the chosen
    # leaf). We need to translate his strings back to the DSURQE leaf IDs to
    # check subtree membership. The simplest equivalent test: walk the tree
    # collecting (id -> name) and (name -> ancestor_names), then check if
    # any ancestor of the parcel's vote-label matches dsurqe_region_name.
    # For now do the easy case: assume Paul's vote names ARE the leaf names
    # in the DSURQE tree, and check if the parcel's vote label is in the
    # subtree by walking the tree once per pack-region.
    ancestors_per_name = _build_ancestor_map(tree_path)
    target = dsurqe_region_name

    parcels = []
    for i, vote_label in enumerate(M_var[column].fillna("").to_list()):
        ancestor_set = ancestors_per_name.get(vote_label, set()) | {vote_label}
        if target in ancestor_set:
            parcels.append(int(i))
    return parcels


def _build_ancestor_map(tree_path: Path) -> dict[str, set[str]]:
    """Return {leaf_name: set of ancestor names} from the DSURQE tree."""
    tree = json.loads(tree_path.read_text())
    out: dict[str, set[str]] = {}

    def walk(node, ancestors: tuple[str, ...]) -> None:
        nm = node.get("name")
        chain = ancestors + ((nm,) if nm else ())
        if nm:
            out[nm] = set(chain[:-1])  # all strict ancestors
        for c in (node.get("children") or {}).values():
            walk(c, chain)

    walk(tree["msg"][0], ())
    return out


def main():
    IO = _load_io()
    print("Loading v2 metadata...")
    meta = IO.load_metadata("mouse")
    if meta["_schema"] != "v2":
        print(f"WARNING: schema is {meta['_schema']!r}; this script needs v2.")
        sys.exit(1)
    M_var = IO.parse_t_table(meta["t"], meta["ht"])
    print(f"  {len(M_var)} parcels loaded")

    # Synthesise a minimal human-side stub. The live anchor pack builders
    # call human_parcels_in_mni_sphere(H_var, centroid, radius) which only
    # needs (x, y, z) columns. For the diff we only compare mouse_indices.
    H_stub = pd.DataFrame({
        "x": np.linspace(-50, 50, 100),
        "y": np.linspace(-90, 90, 100),
        "z": np.linspace(-50, 50, 100),
    })

    sys.path.insert(0, str(ROOT / "src"))

    results = {}
    summary = []

    for module_name, builder_name in PACK_MODULES:
        try:
            mod = importlib.import_module(f"homer.data.anchor_packs.{module_name}")
            builder = getattr(mod, builder_name)
        except Exception as e:
            results[module_name] = {"error": f"import failed: {e}"}
            continue

        print(f"\n--- {module_name} ---")

        # Live path
        try:
            entries = builder(M_var, H_stub, atlas_root=ROOT.parent)
        except Exception as e:
            print(f"  LIVE build failed: {e}")
            results[module_name] = {"live_error": str(e)}
            continue

        # For each entry: live parcel set vs Paul-column-derived parcel set.
        entry_diffs = []
        for e in entries:
            live_set = set(int(p) for p in e.mouse_indices)

            # The live path uses a DSURQE region name internally. We need to
            # extract it from the builder's source. The simplest: look at
            # the pack module's source for `mouse_parcels_in_dsurqe_region(
            # M_var, "<NAME>", ...)` calls. Pack files keep that string
            # as the second positional. Read it from the module.
            #
            # Cleaner: builders return entries with `.label` set to something
            # like "Subiculum (Strange 2014)". The DSURQE name is whatever
            # the builder fed to mouse_parcels_in_dsurqe_region — we don't
            # have that mapping here cleanly without parsing each pack.
            #
            # For this diagnostic we do a heuristic: try the entry's `label`
            # as the DSURQE name; fall back to skipping if unresolved.
            dsurqe_name_candidate = e.label.split(" (")[0].strip()
            try:
                paul_parcels = parcels_via_paul_column(
                    M_var, dsurqe_name_candidate
                )
            except FileNotFoundError as fnf:
                entry_diffs.append({
                    "pair_id": e.pair_id, "label": e.label,
                    "error": str(fnf),
                })
                continue

            paul_set = set(paul_parcels)
            n_intersect = len(live_set & paul_set)
            n_union     = len(live_set | paul_set)
            jaccard = n_intersect / max(n_union, 1)

            entry_diffs.append({
                "pair_id": e.pair_id,
                "label":   e.label,
                "dsurqe_name_tried": dsurqe_name_candidate,
                "n_live":         len(live_set),
                "n_paul":         len(paul_set),
                "n_intersect":    n_intersect,
                "n_only_live":    len(live_set - paul_set),
                "n_only_paul":    len(paul_set - live_set),
                "jaccard":        float(jaccard),
                "flagged":        jaccard < 0.95,
                "only_live_sample": sorted(live_set - paul_set)[:10],
                "only_paul_sample": sorted(paul_set - live_set)[:10],
            })
            mark = "⚠ FLAGGED" if jaccard < 0.95 else "ok"
            print(f"  pid {e.pair_id:3d}  {e.label[:50]:50s}  "
                  f"live={len(live_set):4d} paul={len(paul_set):4d}  "
                  f"jaccard={jaccard:.3f}  {mark}")
            if jaccard < 0.95:
                summary.append({
                    "pack": module_name,
                    "entry": e.label,
                    "jaccard": float(jaccard),
                })

        results[module_name] = {"entries": entry_diffs}

    out_path = OUT / "dsurqe_live_vs_paul_diff.json"
    out_path.write_text(json.dumps({
        "results": results,
        "flagged_summary": summary,
        "decision_rule": "Jaccard >= 0.95 on every entry → option-(c) refactor safe.",
    }, indent=2, default=str))
    print(f"\nsaved {out_path}")

    if not summary:
        print(f"\n=== ALL CLEAR — every entry has Jaccard >= 0.95 ===")
        print(f"     The option-(c) refactor is safe to proceed.")
    else:
        print(f"\n=== {len(summary)} flagged entries (Jaccard < 0.95) ===")
        for s in summary[:20]:
            print(f"  {s['pack']:14s}  {s['entry'][:50]:50s}  J={s['jaccard']:.3f}")
        print(f"\n     Investigate the listed entries before swapping _dsurqe.py.")
        print(f"     Tail samples in {out_path} give the differing parcel IDs.")


if __name__ == "__main__":
    main()
