"""Run every anchor pack under v1 and v2 and diff the selected parcel sets.

B3 in the loader review: small-radius packs (LC, raphe, PAG-adjacent, etc.)
use ``mouse_parcels_in_mouse_sphere`` with radii as small as 0.5 mm.
Under v2 the parcel x/y/z values shift by up to 0.117 mm vs v1, which can
change anchor-pack membership at the boundary.

This script runs every anchor pack builder under both v1 and v2, computes
the set-symmetric-difference of selected parcel IDs, and dumps the result
to ``data_external/_diagnostics/anchor_pack_v1_v2_diff.json``.

Any pack whose diff exceeds 5 % of its set size is flagged for manual
review. Run this BEFORE relying on v2 anchor packs for π refit so we
know which packs (if any) materially change.

Usage:
    PYTHONPATH=src python pipeline/00_external/diff_anchor_packs_v1_vs_v2.py
"""
from __future__ import annotations

import importlib.util
import importlib.machinery
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT.parent / "data_crossspecies"
OUT  = ROOT / "data_external" / "_diagnostics"
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


def _u16(f, ref):
    a = np.asarray(f[ref][:]).flatten()
    return bytes(memoryview(a.astype(np.uint16))).decode("utf-16-le").rstrip("\x00")


def load_v1_var() -> pd.DataFrame:
    """Force-load corrs_mouse.mat (v1) bypassing the v2-first resolver."""
    p = DATA / "corrs_mouse.mat"
    if not p.exists():
        raise FileNotFoundError(f"v1 file not found at {p}")
    with h5py.File(str(p), "r") as f:
        g = f["m"]
        ht = [_u16(f, r) for r in np.asarray(g["ht"][:]).flatten()]
        t_refs = np.asarray(g["t"][:])
        n = t_refs.shape[1]
        cols = {c: ht.index(c) for c in ("numid", "region", "center")}
        numids = np.array([
            int(np.asarray(f[t_refs[cols["numid"], j]][:]).flatten()[0])
            for j in range(n)
        ])
        centres = np.stack([
            np.asarray(f[t_refs[cols["center"], j]][:]).flatten().astype(float)
            for j in range(n)
        ])
        regions = [_u16(f, t_refs[cols["region"], j]) for j in range(n)]
    df = pd.DataFrame({
        "numid": numids, "x": centres[:, 0], "y": centres[:, 1], "z": centres[:, 2],
        "region": regions,
    })
    return df


def main():
    IO = _load_io()

    print("loading v2 metadata...")
    meta = IO.load_metadata("mouse")
    if meta["_schema"] != "v2":
        print(f"NOTE: detected schema {meta['_schema']!r} from the resolver.")
    df_v2 = IO.parse_t_table(meta["t"], meta["ht"])
    print(f"  v2: {len(df_v2)} parcels")

    print("loading v1 t-table (via h5py, bypassing v2-first resolver)...")
    df_v1 = load_v1_var()
    print(f"  v1: {len(df_v1)} parcels")

    if len(df_v1) != len(df_v2):
        raise RuntimeError(
            f"row count mismatch: v1={len(df_v1)} v2={len(df_v2)}"
        )

    # Now run every anchor pack builder under both. Anchor packs that use
    # `mouse_parcels_in_mouse_sphere` (small-radius xyz lookups) are the
    # ones at risk; packs that use `mouse_parcels_in_dsurqe_region` are
    # also at risk if the DSURQE label volume is queried at v2's xyz with
    # a sub-voxel shift.
    #
    # We import the pack builders directly without invoking the full
    # homer package init (avoids ot dependency). Each call returns
    # RegionAnchorEntry instances with mouse_indices / human_indices.

    # Stub the data module's load_cached etc. by exposing what packs need.
    # Pack signatures take (M_var, H_var, atlas_root=...). M_var is the
    # mouse parcel DataFrame (just needs x,y,z and any related columns);
    # H_var the human equivalent (we won't run human-side selection in
    # this diff — only mouse_indices are at risk under v2).

    # Lazy import the pack builders. Some need the DSURQE label volume
    # in data_external/MouseHumanTranscriptomicSimilarity/. If absent,
    # the pack will skip itself.

    sys.path.insert(0, str(ROOT / "src"))

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

    # Synthesise a minimal human-side var DataFrame stub. The pack functions
    # call ``human_parcels_in_mni_sphere(H_var, centroid, radius)`` which
    # only needs columns x, y, z on H_var. For the diff we don't care
    # about human-side membership; we just need the call to succeed.
    H_stub = pd.DataFrame({
        "x": np.linspace(-50, 50, 100),
        "y": np.linspace(-90, 90, 100),
        "z": np.linspace(-50, 50, 100),
    })

    diffs = {}
    for module_name, builder_name in PACK_MODULES:
        try:
            mod = importlib.import_module(f"homer.data.anchor_packs.{module_name}")
            builder = getattr(mod, builder_name)
        except Exception as e:
            diffs[module_name] = {"error": f"import failed: {e}"}
            continue

        print(f"\n--- {module_name} ---")
        for label, M_var in [("v1", df_v1), ("v2", df_v2)]:
            try:
                entries_label = label  # keep both runs scoped
                if label == "v1":
                    entries_v1 = builder(M_var, H_stub, atlas_root=ROOT.parent)
                else:
                    entries_v2 = builder(M_var, H_stub, atlas_root=ROOT.parent)
            except Exception as e:
                diffs.setdefault(module_name, {})[f"{label}_error"] = str(e)

        if module_name in diffs and "v1_error" in diffs[module_name]:
            print(f"  v1 build failed: {diffs[module_name]['v1_error'][:120]}")
            continue
        if module_name in diffs and "v2_error" in diffs[module_name]:
            print(f"  v2 build failed: {diffs[module_name]['v2_error'][:120]}")
            continue

        try:
            entry_diff = []
            for e1, e2 in zip(entries_v1, entries_v2):
                m1 = set(e1.mouse_indices)
                m2 = set(e2.mouse_indices)
                only_v1 = m1 - m2
                only_v2 = m2 - m1
                jaccard = len(m1 & m2) / max(len(m1 | m2), 1)
                entry_diff.append({
                    "pair_id": e1.pair_id,
                    "label":   e1.label,
                    "n_v1":    len(m1),
                    "n_v2":    len(m2),
                    "n_intersect":  len(m1 & m2),
                    "n_only_v1":    len(only_v1),
                    "n_only_v2":    len(only_v2),
                    "jaccard":      float(jaccard),
                    "flagged":      jaccard < 0.95,
                    "only_v1_sample": sorted(only_v1)[:10],
                    "only_v2_sample": sorted(only_v2)[:10],
                })
                print(f"  pid {e1.pair_id}  {e1.label[:50]:50s}  "
                      f"v1={len(m1):3d} v2={len(m2):3d}  "
                      f"jaccard={jaccard:.3f}  "
                      f"{'⚠ FLAGGED' if jaccard < 0.95 else 'ok'}")
            diffs[module_name] = {"entries": entry_diff}
        except Exception as e:
            diffs[module_name] = {
                "diff_error": str(e),
                "n_entries_v1": len(entries_v1) if 'entries_v1' in dir() else None,
                "n_entries_v2": len(entries_v2) if 'entries_v2' in dir() else None,
            }

    out_path = OUT / "anchor_pack_v1_v2_diff.json"
    out_path.write_text(json.dumps(diffs, indent=2, default=str))
    print(f"\nsaved {out_path}")

    flagged = []
    for name, d in diffs.items():
        for e in d.get("entries", []):
            if e.get("flagged"):
                flagged.append((name, e))
    print(f"\n=== summary: {len(flagged)} entries flagged (Jaccard < 0.95) ===")
    for name, e in flagged[:20]:
        print(f"  {name}.{e['label'][:40]}  jaccard={e['jaccard']:.3f}  "
              f"only_v1={e['n_only_v1']} only_v2={e['n_only_v2']}")


if __name__ == "__main__":
    main()
