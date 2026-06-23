"""Build mouse_genes.npy from each parcel's CCFv3 voxel set.

Reads ``AS_ix`` (Allen CCFv3 25 µm voxel indices, pre-warped via a nonlinear
DSURQE→CCFv3 registration) from the mouse ``.mat`` file and samples Allen ISH
energy volumes at 200 µm at the corresponding downsampled voxels.

For each Allen ISH gene:

  1. Locate the energy.zip in the local cache. We reuse the Pagani ISH
     cache (`experiments/autism_subtypes/allen_expansion/pagani_ish_cache/`)
     and fall back to direct download via the Allen API if a gene's
     section_data_set_id is missing.
  2. Open the zip; read `energy.mhd` for shape/dtype + `energy.raw`.
  3. For each of the 1864 parcels, take the parcel's CCFv3 voxel set
     (25 µm grid), downsample each voxel to the 200 µm grid (integer divide
     each axis index by 8), deduplicate, and average the energy values.
  4. Stack per-parcel means into a (1864, n_genes) matrix.

Outputs:

  - ``data_external/mouse_genes.npy``        shape (1864, n_genes)
  - ``data_external/mouse_gene_list.csv``    gene metadata
  - ``data_external/mouse_genes_meta.json``  provenance + cache hit/miss

Usage:
    PYTHONPATH=src python pipeline/00_external/02_mouse_genes.py

The downstream consumers (`03_build_costs.py`, `experiments/*`) read
``mouse_genes.npy`` and ``mouse_gene_list.csv`` only.
"""
from __future__ import annotations

import importlib.util
import importlib.machinery
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_EXT = ROOT / "data_external"
PAGANI_CACHE = ROOT / "experiments/autism_subtypes/allen_expansion/pagani_ish_cache"
ALLENSDK_CACHE = Path.home() / ".allensdk_cache" / "ish_energy"


# Bypass the homer package __init__ (which requires `ot`), load io.py directly.
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


_MHD_DTYPES = {
    "MET_FLOAT":  np.float32,
    "MET_DOUBLE": np.float64,
    "MET_USHORT": np.uint16,
    "MET_SHORT":  np.int16,
    "MET_UCHAR":  np.uint8,
    "MET_CHAR":   np.int8,
    "MET_INT":    np.int32,
    "MET_UINT":   np.uint32,
}


def read_ish_grid(zip_path: Path, variable: str = "energy") -> np.ndarray:
    """Return the 3D energy volume from an Allen ISH grid-data zip."""
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        mhd = next((n for n in names if n.endswith(f"{variable}.mhd")), None)
        raw = next((n for n in names if n.endswith(f"{variable}.raw")), None)
        if mhd is None or raw is None:
            raise FileNotFoundError(f"{zip_path.name}: missing {variable}.mhd/raw (have {names})")
        text = z.read(mhd).decode()
        dim_m  = re.search(r"DimSize\s*=\s*(\d+)\s+(\d+)\s+(\d+)", text)
        type_m = re.search(r"ElementType\s*=\s*(\S+)", text)
        if not dim_m or not type_m:
            raise ValueError(f"unparseable mhd:\n{text[:300]}")
        shape = tuple(int(d) for d in dim_m.groups())
        dtype = _MHD_DTYPES[type_m.group(1)]
        msb = re.search(r"BinaryDataByteOrderMSB\s*=\s*(\S+)", text)
        big = msb is not None and msb.group(1).strip().lower() in {"true", "1"}
        np_dtype = np.dtype(dtype).newbyteorder(">" if big else "<")
        arr = np.frombuffer(z.read(raw), dtype=np_dtype)
        if arr.size != int(np.prod(shape)):
            raise ValueError(f"size mismatch in {zip_path.name}: header {shape} vs {arr.size}")
        # MetaImage (.mhd/.raw) stores voxels column-major (first DimSize axis
        # varies fastest), so the buffer must be reshaped with order="F".
        # NumPy's default C-order would mis-order the volume.
        return arr.reshape(shape, order="F").astype(np.float32)


def find_or_download_gene(sds_id: int, work_dir: Path) -> Path | None:
    """Find ISH zip in caches or download to ``work_dir``."""
    for cache in (PAGANI_CACHE, ALLENSDK_CACHE):
        p = cache / f"sds_{sds_id}_energy.zip"
        if p.exists():
            try:
                with zipfile.ZipFile(p) as z:
                    names = z.namelist()
                    if any(n.endswith(".mhd") for n in names) and any(n.endswith(".raw") for n in names):
                        return p
            except zipfile.BadZipFile:
                pass
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / f"sds_{sds_id}_energy.zip"
    url = f"http://api.brain-map.org/grid_data/download/{sds_id}?include=energy"
    try:
        urllib.request.urlretrieve(url, str(out))
        with zipfile.ZipFile(out) as z:
            if not any(n.endswith(".mhd") for n in z.namelist()):
                out.unlink(missing_ok=True)
                return None
        return out
    except Exception as e:
        print(f"  download fail {sds_id}: {e}", file=sys.stderr)
        return None


def ns_voxels_to_200um(ns_indices_25um: np.ndarray,
                       io_module) -> np.ndarray:
    """Convert NS-grid 25 µm linear indices → deduplicated 200 µm (i,j,k) ijk.

    The loader produces 0-based linear indices in Fortran order into the
    (528, 320, 456) 25 µm grid. Allen ISH energy volumes are at 200 µm with
    shape (67, 41, 58). The mapping is integer-divide by 8 in each axis.
    """
    NS_SHAPE = io_module._NS_SHAPE
    ijk_25 = np.column_stack(np.unravel_index(ns_indices_25um, NS_SHAPE, order="F"))
    ijk_200 = ijk_25 // 8
    # Deduplicate (many 25 µm voxels collapse to the same 200 µm voxel)
    return np.unique(ijk_200, axis=0)


def main():
    IO = _load_io()
    print(f"Loading mouse metadata via {IO.__name__}...")
    meta = IO.load_metadata("mouse")
    schema = meta["_schema"]
    if schema != "v2":
        print(f"WARNING: the mouse parcel table is missing the pre-warped "
              f"voxel-index columns. Point DATA_DIR at the mouse package "
              f"that ships them.")
    df = IO.parse_t_table(meta["t"], meta["ht"])
    n_parcels = len(df)
    print(f"  {n_parcels} parcels (schema={schema})")

    # Read the gene list from an immutable master so re-runs are idempotent.
    # The master is written once and never overwritten; the per-run output
    # (mouse_gene_list.csv) is the NaN-pruned list that aligns 1:1 with the
    # columns of mouse_genes.npy.
    GENE_LIST = DATA_EXT / "mouse_gene_list.csv"
    MASTER = DATA_EXT / "mouse_gene_list_master.csv"
    if MASTER.exists():
        gene_df = pd.read_csv(MASTER)
        print(f"loaded {len(gene_df)} genes from {MASTER.name} (immutable master)")
    else:
        gene_df = pd.read_csv(GENE_LIST)
        gene_df.to_csv(MASTER, index=False)  # seed master from current list, once
        print(f"loaded {len(gene_df)} genes from {GENE_LIST.name}; "
              f"seeded immutable master {MASTER.name}")

    # Pre-compute per-parcel 200 µm voxel sets (NS grid → Allen ISH 67×41×58).
    print("computing per-parcel 200 µm voxel sets from AS_ix...")
    per_parcel_200um = [
        ns_voxels_to_200um(arr, IO) for arr in df["ns_voxel_indices"].to_list()
    ]
    print(f"  median voxels/parcel @ 200 µm: "
          f"{np.median([len(v) for v in per_parcel_200um]):.0f}")

    work_dir = DATA_EXT / "_ish_cache"
    work_dir.mkdir(parents=True, exist_ok=True)

    expr = np.full((n_parcels, len(gene_df)), np.nan, dtype=np.float32)
    cache_hit = 0
    cache_miss = 0
    skipped = []

    for col, row in enumerate(gene_df.itertuples()):
        sds_id = int(row.section_data_set_id)
        sym = row.gene_symbol
        in_pagani = (PAGANI_CACHE / f"sds_{sds_id}_energy.zip").exists()
        if in_pagani: cache_hit += 1
        else:         cache_miss += 1
        zp = find_or_download_gene(sds_id, work_dir)
        if zp is None:
            skipped.append({"sds_id": sds_id, "symbol": sym, "reason": "no energy.zip"})
            continue
        try:
            volume = read_ish_grid(zp, "energy")
        except Exception as e:
            skipped.append({"sds_id": sds_id, "symbol": sym, "reason": f"read fail: {e}"})
            continue

        for p in range(n_parcels):
            ijk = per_parcel_200um[p]
            if len(ijk) == 0: continue
            ok = ((ijk[:, 0] >= 0) & (ijk[:, 0] < volume.shape[0]) &
                  (ijk[:, 1] >= 0) & (ijk[:, 1] < volume.shape[1]) &
                  (ijk[:, 2] >= 0) & (ijk[:, 2] < volume.shape[2]))
            if not ok.any(): continue
            vals = volume[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
            vals = vals[np.isfinite(vals) & (vals >= 0)]
            if len(vals) > 0:
                expr[p, col] = float(vals.mean())

        if (col + 1) % 10 == 0 or col == 0:
            n_nan = np.isnan(expr[:, col]).sum()
            print(f"  [{col+1:2d}/{len(gene_df)}] {sym} ({sds_id}) -> sampled (n_nan {n_nan})")

    # Drop any all-NaN gene columns
    nan_cols = np.isnan(expr).all(axis=0)
    kept_idx = np.where(~nan_cols)[0]
    expr_kept = expr[:, kept_idx]
    gene_df_kept = gene_df.iloc[kept_idx].reset_index(drop=True)

    out_path = DATA_EXT / "mouse_genes.npy"
    np.save(out_path, expr_kept)
    # Pruned list aligned 1:1 with mouse_genes.npy columns. The master
    # (mouse_gene_list_master.csv) is left untouched, so re-runs are idempotent.
    gene_df_kept.to_csv(GENE_LIST, index=False)

    meta_out = {
        "source": "Allen ISH energy at 200 µm sampled via AS_ix (nonlinear DSURQE→CCFv3 warp).",
        "schema_loaded": schema,
        "n_parcels": int(n_parcels),
        "n_genes_attempted": int(len(gene_df)),
        "n_genes_kept":      int(len(kept_idx)),
        "n_genes_skipped":   int(len(skipped)),
        "skipped":           skipped,
        "pagani_cache_hits": int(cache_hit),
        "downloads_needed":  int(cache_miss),
        "ccf_resolution_um": 200,
        "ns_voxel_source": "AS_ix (25 µm) → integer-divide by 8 → 200 µm",
    }
    (DATA_EXT / "mouse_genes_meta.json").write_text(
        json.dumps(meta_out, indent=2, default=str)
    )

    print(f"\ndone:")
    print(f"  kept    {len(kept_idx)}/{len(gene_df)} genes")
    print(f"  cached  {cache_hit}, downloaded {cache_miss}")
    print(f"  saved {out_path}  shape {expr_kept.shape}")


if __name__ == "__main__":
    main()
