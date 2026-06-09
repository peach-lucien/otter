"""Rebuild ``mouse_genes.npy`` using Paul's nonlinear warpfield instead of
HOMER's heuristic 48-permutation transform.

Mechanism:

  1. Load pre-computed per-parcel CCFv3 voxel sets at 200 µm
     (produced by ``04_warped_voxel_sets.py``).
  2. For each Allen ISH gene listed in ``data_external/mouse_gene_list.csv``,
     locate its energy.zip in a cache (preferring the existing
     ``experiments/autism_subtypes/allen_expansion/pagani_ish_cache/``); if
     missing, download directly from the Allen API.
  3. Read the energy volume (shape ~67×41×58 at 200 µm, float32, PIR layout).
  4. Sample at each parcel's warped voxel indices; mean nonneg-finite values
     give the parcel's expression for that gene.
  5. Stack columns into a (1864, n_genes) matrix; save alongside the
     existing ``mouse_genes.npy`` for before/after comparison.

Outputs to ``data_external/``:
  - ``mouse_genes_warped.npy``            shape (1864, n_genes_kept)
  - ``mouse_gene_list_warped.csv``        per-gene metadata
  - ``mouse_genes_warped_meta.json``      provenance + cache hit/miss summary

Usage:
    PYTHONPATH=src python pipeline/00_external/warp_rebuild/05_rebuild_mouse_genes.py
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DATA_EXT = ROOT / "data_external"
WARPED   = DATA_EXT / "_warp_rebuild"
PAGANI_CACHE = ROOT / "experiments/autism_subtypes/allen_expansion/pagani_ish_cache"
ALLENSDK_CACHE = Path.home() / ".allensdk_cache" / "ish_energy"


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
    """Return the 3D energy volume from an Allen ISH grid-data zip.

    Same parsing as in ``02_mouse_genes.py``. Header gives DimSize (Nx, Ny, Nz);
    data stored row-major, float32 in our Allen exports."""
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        mhd = next((n for n in names if n.endswith(f"{variable}.mhd")), None)
        raw = next((n for n in names if n.endswith(f"{variable}.raw")), None)
        if mhd is None or raw is None:
            raise FileNotFoundError(f"{zip_path.name}: missing energy.mhd/raw (have {names})")
        text = z.read(mhd).decode()
        dim_m = re.search(r"DimSize\s*=\s*(\d+)\s+(\d+)\s+(\d+)", text)
        type_m = re.search(r"ElementType\s*=\s*(\S+)", text)
        if not dim_m or not type_m:
            raise ValueError(f"unparseable mhd:\n{text[:300]}")
        shape = tuple(int(d) for d in dim_m.groups())
        elem = type_m.group(1)
        dtype = _MHD_DTYPES.get(elem)
        if dtype is None:
            raise ValueError(f"unsupported MHD ElementType: {elem}")
        msb = re.search(r"BinaryDataByteOrderMSB\s*=\s*(\S+)", text)
        big_endian = msb is not None and msb.group(1).strip().lower() in {"true", "1"}
        np_dtype = np.dtype(dtype).newbyteorder(">" if big_endian else "<")

        buf = z.read(raw)
        arr = np.frombuffer(buf, dtype=np_dtype)
        if arr.size != int(np.prod(shape)):
            raise ValueError(
                f"size mismatch in {zip_path.name}: header {shape} ({np.prod(shape)}) vs {arr.size}"
            )
        return arr.reshape(shape).astype(np.float32)


def find_or_download_gene(sds_id: int, work_dir: Path) -> Path | None:
    """Find ISH zip in the existing caches, or download to ``work_dir``."""
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

    # Download directly from the Allen API
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / f"sds_{sds_id}_energy.zip"
    url = f"http://api.brain-map.org/grid_data/download/{sds_id}?include=energy"
    try:
        urllib.request.urlretrieve(url, str(out))
        with zipfile.ZipFile(out) as z:
            names = z.namelist()
            if not (any(n.endswith(".mhd") for n in names) and any(n.endswith(".raw") for n in names)):
                print(f"  skip {sds_id}: downloaded zip has only {names}", file=sys.stderr)
                out.unlink(missing_ok=True)
                return None
        return out
    except Exception as e:
        print(f"  download fail {sds_id}: {e}", file=sys.stderr)
        return None


def main():
    # Load per-parcel warped voxel sets at 200 µm
    print("loading per-parcel warped voxel sets (200 µm)...")
    vox = np.load(WARPED / "parcel_warped_voxels_200um.npz")
    offsets = vox["offsets"]
    ii, jj, kk = vox["i"], vox["j"], vox["k"]
    n_parcels = len(offsets) - 1
    print(f"  {n_parcels} parcels, total voxel indices {len(ii)}")

    # Gene list
    gene_df = pd.read_csv(DATA_EXT / "mouse_gene_list.csv")
    print(f"loading {len(gene_df)} genes from mouse_gene_list.csv")

    # Working dir for any genes we have to download fresh
    work_dir = WARPED / "_ish_cache"
    work_dir.mkdir(parents=True, exist_ok=True)

    expr = np.full((n_parcels, len(gene_df)), np.nan, dtype=np.float32)
    cache_hit = 0
    cache_miss = 0
    skipped = []

    for col, row in enumerate(gene_df.itertuples()):
        sds_id = int(row.section_data_set_id)
        sym = row.gene_symbol
        # Find cached or download
        in_pagani = (PAGANI_CACHE / f"sds_{sds_id}_energy.zip").exists()
        if in_pagani:
            cache_hit += 1
        else:
            cache_miss += 1
        zp = find_or_download_gene(sds_id, work_dir)
        if zp is None:
            skipped.append({"sds_id": sds_id, "symbol": sym, "reason": "could not obtain energy.zip"})
            print(f"  [{col+1:2d}/{len(gene_df)}] {sym} ({sds_id}): SKIP")
            continue
        try:
            volume = read_ish_grid(zp, "energy")  # shape (67, 41, 58) typically
        except Exception as e:
            skipped.append({"sds_id": sds_id, "symbol": sym, "reason": f"read fail: {e}"})
            print(f"  [{col+1:2d}/{len(gene_df)}] {sym} ({sds_id}): READ FAIL ({e})")
            continue

        # Sample per parcel
        for p in range(n_parcels):
            s, e = offsets[p], offsets[p+1]
            if s == e:
                continue
            pi, pj, pk = ii[s:e], jj[s:e], kk[s:e]
            ok = ((pi >= 0) & (pi < volume.shape[0]) &
                  (pj >= 0) & (pj < volume.shape[1]) &
                  (pk >= 0) & (pk < volume.shape[2]))
            if not ok.any():
                continue
            vals = volume[pi[ok], pj[ok], pk[ok]]
            vals = vals[np.isfinite(vals) & (vals >= 0)]
            if len(vals) > 0:
                expr[p, col] = float(vals.mean())

        if (col+1) % 10 == 0 or col == 0:
            print(f"  [{col+1:2d}/{len(gene_df)}] {sym} ({sds_id}) -> sampled  (n_nan rows: {np.isnan(expr[:, col]).sum()})")

    # Drop genes we couldn't sample (all-NaN)
    nan_cols = np.isnan(expr).all(axis=0)
    kept_idx = np.where(~nan_cols)[0]
    expr_kept = expr[:, kept_idx]
    gene_df_kept = gene_df.iloc[kept_idx].reset_index(drop=True)

    out_path = DATA_EXT / "mouse_genes_warped.npy"
    np.save(out_path, expr_kept)
    gene_df_kept.to_csv(DATA_EXT / "mouse_gene_list_warped.csv", index=False)

    meta = {
        "source":  "Allen ISH energy volumes sampled via Paul's nonlinear DSURQE→CCFv3 warp",
        "n_parcels":   n_parcels,
        "n_genes_attempted": int(len(gene_df)),
        "n_genes_kept":      int(len(kept_idx)),
        "n_genes_skipped":   int(len(skipped)),
        "skipped":           skipped,
        "pagani_cache_hits": int(cache_hit),
        "downloads_needed":  int(cache_miss),
        "warpfield":         "data_crossspecies/warpfields/warpfield2SS.nii.gz",
        "voxel_set_source":  "data_external/_warp_rebuild/parcel_warped_voxels_200um.npz",
        "ccf_resolution_um": 200,
    }
    (DATA_EXT / "mouse_genes_warped_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    print(f"\ndone:")
    print(f"  kept    {len(kept_idx)}/{len(gene_df)} genes")
    print(f"  cached  {cache_hit}, downloaded {cache_miss}")
    print(f"  saved {out_path}  shape {expr_kept.shape}")

    # Quick before/after correlation against the existing mouse_genes.npy
    existing = DATA_EXT / "mouse_genes.npy"
    if existing.exists():
        old = np.load(existing)
        if old.shape[1] >= expr_kept.shape[1]:
            # Match by gene_symbol (in case ordering differs)
            old_list = pd.read_csv(DATA_EXT / "mouse_gene_list.csv")
            sym_to_old_col = {s: i for i, s in enumerate(old_list["gene_symbol"].tolist())}
            common_cols_old = [sym_to_old_col[s] for s in gene_df_kept["gene_symbol"].tolist() if s in sym_to_old_col]
            common_cols_new = [i for i, s in enumerate(gene_df_kept["gene_symbol"].tolist()) if s in sym_to_old_col]
            if len(common_cols_old) > 5:
                old_aligned = old[:, common_cols_old]
                new_aligned = expr_kept[:, common_cols_new]
                # Per-parcel cosine similarity between old and new gene vectors
                from numpy.linalg import norm
                cos = np.array([
                    float(np.dot(old_aligned[p], new_aligned[p]) /
                          max(norm(old_aligned[p]) * norm(new_aligned[p]), 1e-12))
                    for p in range(n_parcels)
                ])
                print(f"\nper-parcel cosine similarity (OLD vs NEW gene vectors), n_genes_compared = {len(common_cols_old)}:")
                print(f"  mean:   {np.nanmean(cos):.3f}")
                print(f"  median: {np.nanmedian(cos):.3f}")
                print(f"  min:    {np.nanmin(cos):.3f}")
                print(f"  pct <0.5: {100*(cos < 0.5).mean():.1f}%")
                print(f"  pct <0.0: {100*(cos < 0.0).mean():.1f}%")
                np.save(WARPED / "gene_vector_cosine_old_vs_new.npy", cos)
                print(f"  saved per-parcel cosines to {WARPED / 'gene_vector_cosine_old_vs_new.npy'}")


if __name__ == "__main__":
    main()
