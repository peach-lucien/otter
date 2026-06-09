"""Project Allen Mouse Brain ISH gene expression onto the 1864-node parcellation.

.. deprecated:: v2
    LEGACY (v1 only). This script uses the heuristic 48-permutation
    transform from ``00c_align_mouse_to_ccf.py`` to map parcel centres
    into CCFv3 before ISH sampling. The v2 successor
    ``02c_mouse_genes_v2.py`` reads the pre-warped voxel set ``AS_ix``
    directly from ``corrs_mouse_v2.mat`` (Paul's nonlinear DSURQE -> CCFv3
    warp) and is the production path. Use this script only when working
    from the v1 mouse package.

For each gene we get a 3D 'energy' volume in CCFv3 200 µm space. We then sample
the volume at each node's voxel positions (after transforming colleague-mouse
coords → CCFv3 coords).

Output:
  data_external/mouse_genes.npy        (1864, n_genes) float32
  data_external/mouse_gene_list.csv    gene metadata
"""
from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Allen ISH grid-data reader (replaces the deprecated
# GridDataApi.read_brain_atlas_data, which was removed in newer allensdk).
# ---------------------------------------------------------------------------
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


def _read_ish_grid(zip_path: Path, variable: str = "energy") -> np.ndarray:
    """Read a 3D volume out of an Allen ISH grid-data zip.

    The zip contains a `<variable>.mhd` (text header) and `<variable>.raw`
    (binary). MetaImage convention: data stored row-major; shape from header
    is (size_x, size_y, size_z) but the linear ordering is the standard 3D
    flatten — np.fromfile().reshape(shape) works directly.
    """
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        mhd = next((n for n in names if n.endswith(f"{variable}.mhd")), None)
        raw = next((n for n in names if n.endswith(f"{variable}.raw")), None)
        if mhd is None or raw is None:
            raise FileNotFoundError(
                f"missing {variable}.mhd / {variable}.raw in {zip_path.name} (have: {names})"
            )
        text = z.read(mhd).decode()
        dim_m = re.search(r"DimSize\s*=\s*(\d+)\s+(\d+)\s+(\d+)", text)
        type_m = re.search(r"ElementType\s*=\s*(\S+)", text)
        if not dim_m or not type_m:
            raise ValueError(f"could not parse {variable}.mhd:\n{text[:300]}")
        shape = tuple(int(d) for d in dim_m.groups())
        elem = type_m.group(1)
        dtype = _MHD_DTYPES.get(elem)
        if dtype is None:
            raise ValueError(f"unsupported MHD element type: {elem}")
        msb = re.search(r"ElementByteOrderMSB\s*=\s*(\S+)", text)
        big_endian = msb is not None and msb.group(1).strip().lower() in {"true", "1"}
        np_dtype = np.dtype(dtype).newbyteorder(">" if big_endian else "<")

        buf = z.read(raw)
        arr = np.frombuffer(buf, dtype=np_dtype)
        if arr.size != int(np.prod(shape)):
            raise ValueError(
                f"size mismatch in {zip_path.name}: header says {shape} "
                f"({np.prod(shape)} elements), got {arr.size}"
            )
        # MetaImage axis order is (x, y, z) in C-contiguous memory.
        return arr.reshape(shape).astype(np.float32)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from homer.data import DATA_DIR, load_metadata, parse_t_table     # noqa: E402
from _mouse_transform import (                                      # noqa: E402
    load_transform, colleague_voxel_to_ccf_world,
)

OUT  = ROOT / "data_external"; OUT.mkdir(parents=True, exist_ok=True)
DIAG = OUT / "_diagnostics"
MASK = DATA_DIR / "_mouse_mask" / "rsmask.nii"


def _check_smoke_test(k, valid_gene_idx, n_skipped_no_grid, n_skipped_other,
                       smoke_n, smoke_min, progress_every):
    """Bail early if everything is failing; print progress periodically."""
    n_attempted = k + 1
    n_kept = len(valid_gene_idx)
    n_failed = n_skipped_no_grid + n_skipped_other

    # Smoke test: bail if first SMOKE_TEST_N attempts have too few successes
    if n_attempted == smoke_n and n_kept < smoke_min:
        print(f"\n\nSMOKE TEST FAILED: only {n_kept}/{smoke_n} attempts produced "
              f"a usable gene volume ({n_skipped_no_grid} no-grid, "
              f"{n_skipped_other} other failures).\n"
              f"  Likely cause: Allen API issue or all initial datasets lack 3D grid.\n"
              f"  Suggested check: try one SDS_ID manually:\n"
              f"    >>> from allensdk.api.queries.grid_data_api import GridDataApi\n"
              f"    >>> g = GridDataApi()\n"
              f"    >>> g.download_gene_expression_grid_data(79587720, 'energy', '/tmp/test.zip')\n"
              f"    >>> import zipfile; print(zipfile.ZipFile('/tmp/test.zip').namelist())\n"
              f"  Expecting ['energy.mhd', 'energy.raw']; got ['data_set.xml'] = API regression.\n"
              f"  Aborting before wasting more time.")
        sys.exit(2)

    # Periodic progress checkpoint
    if n_attempted % progress_every == 0:
        rate = n_kept / n_attempted * 100
        print(f"\n  [checkpoint @ {n_attempted}] kept={n_kept} ({rate:.1f}%), "
              f"no_grid={n_skipped_no_grid}, other={n_skipped_other}", flush=True)


def main(args):
    try:
        from allensdk.api.queries.rma_api import RmaApi
        from allensdk.api.queries.grid_data_api import GridDataApi
    except ImportError:
        print("ERROR: pip install allensdk")
        sys.exit(1)

    transform = load_transform(DIAG)
    print(f"using mouse→CCFv3 transform (coverage at fit: {transform['coverage']:.1%})")
    diagnostics = json.loads((DIAG / "mask_info.json").read_text())
    one_based = diagnostics["mouse_voxel_index_check"]["likely_one_based"]
    order = diagnostics["mouse_voxel_index_check"]["recommended_order"]

    cache_dir = Path.home() / ".allensdk_cache"
    rma = RmaApi(); grid = GridDataApi()
    # Note: download_expression_grid_data is deprecated and has a string-vs-list
    # bug with the `include` parameter. Use download_gene_expression_grid_data.

    # 1. Gene list -----------------------------------------------------------
    print("listing Allen ISH coronal genes...")
    gene_list_path = OUT / "_diagnostics" / "allen_ish_gene_list.csv"
    gene_list_path.parent.mkdir(parents=True, exist_ok=True)
    if gene_list_path.exists() and not args.refresh:
        genes_df = pd.read_csv(gene_list_path)
    else:
        # The 'storage_directory$ne' filter excludes datasets that don't have
        # a 3D-reconstructed volume on disk — i.e. the ones whose grid_data
        # download returns only `data_set.xml` and no `.mhd`/`.raw`.
        records = rma.model_query(
            "SectionDataSet",
            criteria=("[failed$eqfalse],[storage_directory$ne''],"
                      "products[id$eq1],plane_of_section[name$eq'coronal']"),
            include="genes", num_rows="all", start_row=0,
        )
        flat = []
        for r in records:
            for g in (r.get("genes") or []):
                flat.append({
                    "section_data_set_id": int(r["id"]),
                    "gene_id":   int(g["id"]),
                    "gene_symbol": g.get("acronym", ""),
                    "entrez_id":   g.get("entrez_id"),
                })
        genes_df = pd.DataFrame(flat).drop_duplicates(subset="gene_id")
        genes_df.to_csv(gene_list_path, index=False)
    print(f"  {len(genes_df)} unique genes with coronal ISH")
    if args.max_genes is not None:
        genes_df = genes_df.head(args.max_genes)
        print(f"  restricted to first {args.max_genes} (test mode)")

    # 2. Setup mouse node CCFv3 coordinates ---------------------------------
    rsmask = nib.load(MASK)
    rsmask_affine = rsmask.affine
    rsmask_shape = rsmask.shape
    meta = load_metadata("mouse"); df = parse_t_table(meta["t"], meta["ht"])
    n_nodes = len(df)

    # Pre-compute, per node, its voxels' CCFv3 200 µm voxel indices
    # (gene volumes are at 200 µm)
    print("pre-computing per-node CCFv3 200 µm voxel indices ...")
    res_mm = 0.2
    node_ccf_voxels: list[np.ndarray] = []
    for vox in df["voxel_indices"]:
        ccf_world = colleague_voxel_to_ccf_world(
            rsmask_affine, np.asarray(vox), rsmask_shape,
            one_based=one_based, order=order, transform=transform,
        )
        ccf_ijk = (ccf_world / res_mm).astype(np.int64)
        node_ccf_voxels.append(ccf_ijk)

    # 3. Download per-gene energy volume + sample per node -------------------
    print("downloading + sampling gene volumes ...")
    grid_dir = cache_dir / "ish_energy"
    grid_dir.mkdir(parents=True, exist_ok=True)

    # ---- one-time cache validation ----------------------------------------
    # Pre-fix runs (when the API call had a string-vs-list bug) saved zips
    # containing only `data_set.xml`. Scan & delete any stale broken zips so
    # we re-download them with the now-working API call.
    print("scanning cache for broken zips...")
    n_purged = 0
    for zpath in grid_dir.glob("*.zip"):
        try:
            with zipfile.ZipFile(zpath) as z:
                names = z.namelist()
                ok = any(n.endswith(".mhd") for n in names) and \
                     any(n.endswith(".raw") for n in names)
            if not ok:
                zpath.unlink()
                n_purged += 1
        except (zipfile.BadZipFile, OSError):
            try: zpath.unlink(); n_purged += 1
            except OSError: pass
    print(f"  purged {n_purged} broken zips from {grid_dir}")

    expr = np.full((n_nodes, len(genes_df)), np.nan, dtype=np.float32)
    valid_gene_idx = []
    n_skipped_no_grid = 0
    n_skipped_other = 0

    # Smoke-test thresholds: bail early if the API is broken
    SMOKE_TEST_N    = 10        # check the first N attempts
    SMOKE_TEST_MIN  = 2         # at least this many must succeed
    PROGRESS_EVERY  = 100       # print intermediate stats every N attempts

    from tqdm import tqdm
    for k, (_, row) in enumerate(tqdm(genes_df.iterrows(), total=len(genes_df))):
        sds_id = int(row["section_data_set_id"])
        path = grid_dir / f"sds_{sds_id}_energy.zip"
        if not path.exists():
            try:
                grid.download_gene_expression_grid_data(
                    section_data_set_id=sds_id,
                    volume_type="energy",
                    path=str(path),
                )
            except Exception as e:
                tqdm.write(f"  skip {sds_id}: download fail: {e}")
                n_skipped_other += 1
                _check_smoke_test(k, valid_gene_idx, n_skipped_no_grid, n_skipped_other,
                                  SMOKE_TEST_N, SMOKE_TEST_MIN, PROGRESS_EVERY)
                continue
        try:
            volume = _read_ish_grid(path, variable="energy")
        except FileNotFoundError:
            n_skipped_no_grid += 1
            try: path.unlink()
            except OSError: pass
            _check_smoke_test(k, valid_gene_idx, n_skipped_no_grid, n_skipped_other,
                              SMOKE_TEST_N, SMOKE_TEST_MIN, PROGRESS_EVERY)
            continue
        except Exception as e:
            tqdm.write(f"  skip {sds_id}: read fail: {e}")
            n_skipped_other += 1
            _check_smoke_test(k, valid_gene_idx, n_skipped_no_grid, n_skipped_other,
                              SMOKE_TEST_N, SMOKE_TEST_MIN, PROGRESS_EVERY)
            continue
        # volume is in CCFv3 200 µm space — shape typically (67, 41, 58) or similar
        for i, ccf_ijk in enumerate(node_ccf_voxels):
            in_bounds = ((ccf_ijk[:, 0] >= 0) & (ccf_ijk[:, 0] < volume.shape[0]) &
                         (ccf_ijk[:, 1] >= 0) & (ccf_ijk[:, 1] < volume.shape[1]) &
                         (ccf_ijk[:, 2] >= 0) & (ccf_ijk[:, 2] < volume.shape[2]))
            if not in_bounds.any():
                continue
            ok = ccf_ijk[in_bounds]
            vals = volume[ok[:, 0], ok[:, 1], ok[:, 2]]
            vals = vals[np.isfinite(vals) & (vals >= 0)]
            if len(vals) > 0:
                expr[i, k] = float(vals.mean())
        valid_gene_idx.append(k)
        _check_smoke_test(k, valid_gene_idx, n_skipped_no_grid, n_skipped_other,
                          SMOKE_TEST_N, SMOKE_TEST_MIN, PROGRESS_EVERY)

    # Drop columns for genes we skipped — the per-gene order was preserved
    # by iteration but we want a tight matrix at the end.
    expr_kept = expr[:, valid_gene_idx]
    np.save(OUT / "mouse_genes.npy", expr_kept)
    metadata = genes_df.iloc[valid_gene_idx].reset_index(drop=True)
    metadata.to_csv(OUT / "mouse_gene_list.csv", index=False)
    info = {
        "source":  "Allen Mouse Brain ISH atlas (Lein et al. 2007)",
        "n_nodes": int(n_nodes),
        "n_genes": int(len(valid_gene_idx)),
        "n_attempted":          int(len(genes_df)),
        "n_skipped_no_grid":    int(n_skipped_no_grid),
        "n_skipped_other":      int(n_skipped_other),
        "transform_used": transform,
        "ccf_resolution_um": int(res_mm * 1000),
    }
    print(f"\n  done: {len(valid_gene_idx)} genes kept, "
          f"{n_skipped_no_grid} with no 3D grid, {n_skipped_other} other failures")
    (OUT / "mouse_genes_meta.json").write_text(json.dumps(info, indent=2, default=str))
    print(f"\nsaved → {OUT / 'mouse_genes.npy'}  shape {expr.shape}")
    print(f"        {OUT / 'mouse_gene_list.csv'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-genes", type=int, default=None)
    ap.add_argument("--refresh",   action="store_true")
    main(ap.parse_args())
