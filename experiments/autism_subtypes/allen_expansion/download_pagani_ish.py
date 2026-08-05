"""Standalone Allen Mouse ISH downloader for Pagani 2026's 6,415 implicated genes.

**Run this OUTSIDE the conversation sandbox**, it needs ~5-20 GB of disk for
the cache, takes 1-3 days even with parallel workers, and downloads ~1-2 GB
of usable per-parcel expression data.

What it does:
  1. Reads `pagani_gene_list.csv` (6,415 mouse-symbol-cased gene symbols).
  2. For each gene, queries Allen Brain Atlas RMA API for a coronal
     SectionDataSet (no allensdk dependency; uses requests directly).
  3. Downloads the 3D ISH energy grid (.mhd + .raw) as a zip.
  4. Reads the volume, samples per-parcel CCFv3 voxels using OTTER's existing
     mouse → CCFv3 transform.
  5. Saves a (1864, n_genes_kept) expression matrix + a gene list.

Outputs (in this directory):
  - pagani_ish_cache/sds_<id>_energy.zip  (cached zip per gene)
  - pagani_mouse_expr.npy                  (1864, n_genes_kept) float32
  - pagani_gene_list_resolved.csv          per-gene status + sds_id
  - pagani_download_log.json               session-level stats

Recovery: the script is idempotent. Re-running picks up where it left off
(skips genes whose zip already exists and parses successfully).

Usage:
    python experiments/autism_subtypes/allen_expansion/download_pagani_ish.py \\
        --gene-list experiments/autism_subtypes/allen_expansion/pagani_gene_list.csv \\
        --workers 4 --max-genes 6415

For a fast test pull on 50 genes:
    python ... --max-genes 50

The downstream analysis (`09_gene_spatial_translation.py`) just loads
`pagani_mouse_expr.npy` and `pagani_gene_list_resolved.csv`, no other
plumbing needed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import requests
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]   # otter/
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipeline" / "00_external"))

from otter.data import DATA_DIR, load_metadata, parse_t_table       # noqa: E402
from _mouse_transform import load_transform, colleague_voxel_to_ccf_world  # noqa: E402

ALLEN_BASE = "http://api.brain-map.org"
DEFAULT_TIMEOUT = 30
RES_MM = 0.2   # CCFv3 200µm

# ---------------------------------------------------------------------------
# Allen API helpers (no allensdk dependency)
# ---------------------------------------------------------------------------

def query_section_data_set(symbol: str, session: requests.Session,
                             retries: int = 3) -> int | None:
    """Return lowest-id coronal mouse SectionDataSet for a gene symbol, or None."""
    criteria = (
        "[failed$eqfalse],products[id$eq1],"
        "plane_of_section[name$eq'coronal'],"
        f"genes[acronym$eq'{symbol}']"
    )
    url = (
        f"{ALLEN_BASE}/api/v2/data/SectionDataSet/query.xml?"
        f"criteria={urllib.parse.quote(criteria, safe='[]$,')}"
        f"&num_rows=20"
    )
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=DEFAULT_TIMEOUT)
            r.raise_for_status()
            root = ET.fromstring(r.text)
            ids = sorted(int(o.find("id").text) for o in root.iter("section-data-set"))
            return ids[0] if ids else None
        except (requests.RequestException, ET.ParseError) as e:
            if attempt == retries - 1:
                return None
            time.sleep(0.5 * (2 ** attempt))
    return None


def download_ish_grid(sds_id: int, cache_path: Path, session: requests.Session,
                       retries: int = 3) -> bool:
    """Download Allen ISH energy grid as zip and verify .mhd + .raw inside.
    Returns True on success."""
    if cache_path.exists() and cache_path.stat().st_size > 1000:
        # Validate cached
        try:
            with zipfile.ZipFile(cache_path) as zf:
                names = zf.namelist()
            has = any(n.endswith(".mhd") for n in names) and \
                  any(n.endswith(".raw") for n in names)
            if has: return True
            cache_path.unlink()
        except zipfile.BadZipFile:
            cache_path.unlink()
    url = f"{ALLEN_BASE}/grid_data/download/{sds_id}?include=energy"
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=60, stream=True)
            r.raise_for_status()
            ct = r.headers.get("Content-Type", "")
            if "zip" not in ct.lower() and "octet-stream" not in ct.lower():
                # Got HTML/XML error response
                return False
            with cache_path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
            with zipfile.ZipFile(cache_path) as zf:
                names = zf.namelist()
            if any(n.endswith(".mhd") for n in names) and \
               any(n.endswith(".raw") for n in names):
                return True
            cache_path.unlink()
            return False
        except (requests.RequestException, zipfile.BadZipFile):
            if cache_path.exists():
                cache_path.unlink()
            if attempt == retries - 1:
                return False
            time.sleep(1.0 * (2 ** attempt))
    return False


_MHD_DTYPES = {
    "MET_FLOAT":  np.float32, "MET_DOUBLE": np.float64,
    "MET_USHORT": np.uint16,  "MET_SHORT":  np.int16,
    "MET_UCHAR":  np.uint8,   "MET_CHAR":   np.int8,
    "MET_INT":    np.int32,   "MET_UINT":   np.uint32,
}


def read_ish_grid(zip_path: Path, variable: str = "energy") -> np.ndarray:
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        mhd = next(n for n in names if n.endswith(f"{variable}.mhd"))
        raw = next(n for n in names if n.endswith(f"{variable}.raw"))
        text = z.read(mhd).decode()
        dim_m = re.search(r"DimSize\s*=\s*(\d+)\s+(\d+)\s+(\d+)", text)
        type_m = re.search(r"ElementType\s*=\s*(\S+)", text)
        shape = tuple(int(d) for d in dim_m.groups())
        dtype = _MHD_DTYPES[type_m.group(1)]
        msb = re.search(r"ElementByteOrderMSB\s*=\s*(\S+)", text)
        big = msb is not None and msb.group(1).strip().lower() in {"true", "1"}
        np_dtype = np.dtype(dtype).newbyteorder(">" if big else "<")
        buf = z.read(raw)
        # MetaImage (.mhd/.raw) is column-major: reshape with order="F"
        # (NumPy's default C-order would mis-order the volume).
        return np.frombuffer(buf, dtype=np_dtype).reshape(shape, order="F").astype(np.float32)


# ---------------------------------------------------------------------------
# Per-parcel sampling
# ---------------------------------------------------------------------------

def compute_parcel_voxel_indices() -> list[np.ndarray]:
    """Re-compute per-parcel CCFv3 voxel indices (one per parcel; each is a list
    of voxel ijk triples that fall in that parcel)."""
    DIAG = ROOT / "data_external" / "_diagnostics"
    MASK = DATA_DIR / "_mouse_mask" / "rsmask.nii"

    transform = load_transform(DIAG)
    diagnostics = json.loads((DIAG / "mask_info.json").read_text())
    one_based = diagnostics["mouse_voxel_index_check"]["likely_one_based"]
    order = diagnostics["mouse_voxel_index_check"]["recommended_order"]

    if not MASK.exists():
        raise SystemExit(
            f"\n[allen_expansion] Required source mask not found:\n  {MASK}\n\n"
            "This Allen-ISH download step needs the mouse resting-state mask under\n"
            "  data_crossspecies/_mouse_mask/rsmask.nii\n"
            "which is raw source data NOT included in the public data release. This\n"
            "is a maintainer / source-data-only script (and a multi-day Allen API\n"
            "download). See experiments/autism_subtypes/allen_expansion/README.md.\n"
        )
    rsmask = nib.load(str(MASK))
    rsmask_affine = rsmask.affine; rsmask_shape = rsmask.shape
    meta = load_metadata("mouse"); df = parse_t_table(meta["t"], meta["ht"])

    print(f"  Computing CCFv3 voxel indices for {len(df)} mouse parcels...")
    node_ccf_voxels = []
    for vox in df["voxel_indices"]:
        ccf_world = colleague_voxel_to_ccf_world(
            rsmask_affine, np.asarray(vox), rsmask_shape,
            one_based=one_based, order=order, transform=transform,
        )
        ccf_ijk = (ccf_world / RES_MM).astype(np.int64)
        node_ccf_voxels.append(ccf_ijk)
    return node_ccf_voxels


def sample_volume(volume: np.ndarray, node_ccf_voxels: list[np.ndarray]) -> np.ndarray:
    """Return per-parcel mean of `volume` (NaN-aware)."""
    out = np.full(len(node_ccf_voxels), np.nan, dtype=np.float32)
    for i, ccf_ijk in enumerate(node_ccf_voxels):
        in_b = ((ccf_ijk[:, 0] >= 0) & (ccf_ijk[:, 0] < volume.shape[0]) &
                (ccf_ijk[:, 1] >= 0) & (ccf_ijk[:, 1] < volume.shape[1]) &
                (ccf_ijk[:, 2] >= 0) & (ccf_ijk[:, 2] < volume.shape[2]))
        if not in_b.any(): continue
        ok = ccf_ijk[in_b]
        v = volume[ok[:, 0], ok[:, 1], ok[:, 2]]
        v = v[np.isfinite(v) & (v >= 0)]
        if len(v) > 0:
            out[i] = float(v.mean())
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gene-list", default=str(Path(__file__).parent / "pagani_gene_list.csv"))
    p.add_argument("--out-dir", default=str(Path(__file__).parent))
    p.add_argument("--cache-dir", default=None,
                   help="Where to cache ISH zips. Default: <out-dir>/pagani_ish_cache")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel download workers (Allen API has soft rate-limit)")
    p.add_argument("--max-genes", type=int, default=6415,
                   help="Hard cap on genes attempted (use small value for test runs)")
    p.add_argument("--start", type=int, default=0,
                   help="Skip first N genes (for chunked runs across machines)")
    p.add_argument("--sleep-between-queries", type=float, default=0.1)
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "pagani_ish_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    genes = pd.read_csv(args.gene_list)
    genes = genes.iloc[args.start: args.start + args.max_genes].reset_index(drop=True)
    print(f"Targeting {len(genes)} genes from {args.gene_list}")

    # Step 1: resolve symbols → SectionDataSet ids (parallel; very fast)
    print("\nStep 1: resolving gene symbols → SectionDataSet ids ...")
    session = requests.Session()
    resolved_records = []
    fail_ids = []

    def _resolve_one(idx_sym):
        idx, sym = idx_sym
        sds = query_section_data_set(sym, session=session)
        return idx, sym, sds

    # Sequential to be polite to RMA endpoint
    for idx, sym in enumerate(genes["mouse_symbol"]):
        sds = query_section_data_set(sym, session=session)
        if sds is None:
            fail_ids.append((idx, sym, "no_dataset"))
        else:
            resolved_records.append({
                "idx": idx, "mouse_symbol": sym, "section_data_set_id": sds,
                "human_symbol": genes.iloc[idx]["human_symbol"],
                "subtype": genes.iloc[idx]["subtype"],
            })
        if idx % 100 == 0:
            print(f"  {idx}/{len(genes)}: {sym} → {sds}")
        time.sleep(args.sleep_between_queries)
    print(f"\n  Resolved {len(resolved_records)} of {len(genes)} symbols "
          f"({len(fail_ids)} unmatched)")

    # Step 2: per-parcel CCFv3 voxel indices (one-shot, slow first call)
    print("\nStep 2: computing per-parcel CCFv3 voxel indices ...")
    node_ccf_voxels = compute_parcel_voxel_indices()
    n_nodes = len(node_ccf_voxels)

    # Step 3: parallel download + parse + sample
    print(f"\nStep 3: downloading + sampling {len(resolved_records)} grids "
          f"(workers={args.workers})...")
    expr = np.full((n_nodes, len(resolved_records)), np.nan, dtype=np.float32)
    n_ok = n_fail = 0

    def _process(rec):
        sds_id = rec["section_data_set_id"]
        path = cache_dir / f"sds_{sds_id}_energy.zip"
        ok = download_ish_grid(sds_id, path, session=session)
        if not ok:
            return rec["idx"], rec["mouse_symbol"], None
        try:
            vol = read_ish_grid(path)
        except Exception:
            return rec["idx"], rec["mouse_symbol"], None
        sampled = sample_volume(vol, node_ccf_voxels)
        return rec["idx"], rec["mouse_symbol"], sampled

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_process, r): k for k, r in enumerate(resolved_records)}
        for fut in as_completed(futures):
            k = futures[fut]
            idx, sym, sampled = fut.result()
            completed += 1
            if sampled is None:
                n_fail += 1
                if completed % 50 == 0:
                    print(f"  {completed}/{len(resolved_records)} | {sym}: FAILED "
                          f"(running: {n_ok} ok, {n_fail} fail)")
            else:
                expr[:, k] = sampled
                n_ok += 1
                if completed % 50 == 0:
                    print(f"  {completed}/{len(resolved_records)} | {sym}: OK "
                          f"(running: {n_ok} ok, {n_fail} fail)")

    print(f"\n  Final: {n_ok} ok, {n_fail} failed out of {len(resolved_records)}")

    # Save
    valid_idx = [k for k in range(len(resolved_records))
                 if np.isfinite(expr[:, k]).any()]
    expr_kept = expr[:, valid_idx]
    kept_meta = pd.DataFrame(
        [resolved_records[k] for k in valid_idx],
        columns=["idx", "mouse_symbol", "section_data_set_id", "human_symbol", "subtype"],
    )
    np.save(out_dir / "pagani_mouse_expr.npy", expr_kept)
    kept_meta.to_csv(out_dir / "pagani_gene_list_resolved.csv", index=False)
    fail_meta = pd.DataFrame(fail_ids, columns=["idx", "mouse_symbol", "reason"])
    fail_meta.to_csv(out_dir / "pagani_gene_failures.csv", index=False)

    log = {
        "n_genes_input": int(len(genes)),
        "n_resolved":    int(len(resolved_records)),
        "n_downloaded_ok": int(n_ok),
        "n_failed":      int(n_fail),
        "n_in_output":   int(len(valid_idx)),
        "parcels":       int(n_nodes),
        "cache_dir":     str(cache_dir),
        "out_files": [
            "pagani_mouse_expr.npy",
            "pagani_gene_list_resolved.csv",
            "pagani_gene_failures.csv",
        ],
    }
    (out_dir / "pagani_download_log.json").write_text(json.dumps(log, indent=2))
    print(f"\nSaved: {out_dir/'pagani_mouse_expr.npy'} shape={expr_kept.shape}")
    print(f"       {out_dir/'pagani_gene_list_resolved.csv'} ({len(kept_meta)} genes)")
    print(f"       {out_dir/'pagani_gene_failures.csv'} ({len(fail_meta)} failures)")
    print(f"       {out_dir/'pagani_download_log.json'}")


if __name__ == "__main__":
    main()
