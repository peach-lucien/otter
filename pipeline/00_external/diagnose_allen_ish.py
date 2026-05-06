"""30-second diagnostic for Allen ISH grid download.

Tests three things in order:
  1. Direct API call with a KNOWN-GOOD SDS_ID (Bdnf, sds=79587720). This is one
     of the 51 that 02b_mouse_genes_direct.py successfully downloaded. If THIS
     fails, the Allen API itself has rotted.
  2. Direct API call with a SDS_ID from the bulk RmaApi query. If THIS fails,
     the bulk query is returning datasets that don't have grid data.
  3. Compare the bulk-query results to the curated set: are they the same kind
     of dataset?

Run:
    PYTHONPATH=src python scripts/external/diagnose_allen_ish.py

Expected good output (everything works):
    [TEST 1] Bdnf SDS 79587720 → energy.mhd + energy.raw  ✓
    [TEST 2] First bulk-query SDS → energy.mhd + energy.raw  ✓
    [TEST 3] All checks pass — your 02_mouse_genes.py should work after the
             cache cleanup runs at startup.

Expected bad output (API has rotted):
    [TEST 1] Bdnf SDS 79587720 → ['data_set.xml']  ✗
    The Allen API has changed. Need to use a different download method.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pandas as pd


def test_download(sds_id: int, label: str = "") -> bool:
    """Download one zip and inspect it."""
    from allensdk.api.queries.grid_data_api import GridDataApi
    gda = GridDataApi()
    out = Path(f"/tmp/allen_diag_sds_{sds_id}.zip")
    if out.exists(): out.unlink()
    print(f"\n[{label}] downloading SDS {sds_id} ...", flush=True)
    try:
        gda.download_gene_expression_grid_data(
            section_data_set_id=sds_id, volume_type="energy", path=str(out),
        )
    except Exception as e:
        print(f"  ✗ download raised: {e}")
        return False
    if not out.exists():
        print(f"  ✗ no file created at {out}")
        return False
    size = out.stat().st_size
    print(f"  downloaded {size} bytes")
    try:
        with zipfile.ZipFile(out) as z:
            names = z.namelist()
    except zipfile.BadZipFile:
        print(f"  ✗ not a valid zip"); return False
    print(f"  zip contents: {names}")
    has_grid = any(n.endswith(".mhd") for n in names) and \
               any(n.endswith(".raw") for n in names)
    if has_grid:
        print(f"  ✓ contains energy.mhd + energy.raw → THIS GENE WORKS")
        return True
    print(f"  ✗ no .mhd/.raw — Allen returned only metadata for this dataset")
    return False


def main():
    print("=" * 60)
    print(" Allen ISH grid download diagnostic ")
    print("=" * 60)

    # TEST 1: known-good SDS (Bdnf, from 02b's successful run)
    test1 = test_download(79587720, "TEST 1 — known-good (Bdnf)")

    # TEST 2: SDS from the bulk query (look at 02_mouse_genes.py's gene list)
    print("\n[TEST 2 setup] reading bulk query result ...")
    bulk_csv = Path(__file__).resolve().parents[2] / "data_external" / "_diagnostics" / "allen_ish_gene_list.csv"
    if not bulk_csv.exists():
        print(f"  WARN: {bulk_csv} not found — run 02_mouse_genes.py once to generate it")
        print("  Skipping TEST 2 and TEST 3.")
        sys.exit(0 if test1 else 1)

    bulk = pd.read_csv(bulk_csv)
    print(f"  loaded {len(bulk)} bulk-query SDS_IDs")
    # Try a few in order
    bulk_ids = bulk["section_data_set_id"].head(5).tolist()
    print(f"  testing first 5 bulk SDS_IDs: {bulk_ids}")
    test2_passes = []
    for sds in bulk_ids:
        test2_passes.append(test_download(int(sds), f"TEST 2 bulk SDS {sds}"))
    n_pass = sum(test2_passes)

    # TEST 3: compare: does the curated good SDS appear in the bulk query?
    print("\n[TEST 3] is the known-good Bdnf SDS in the bulk query?")
    has_bdnf_sds = (bulk["section_data_set_id"] == 79587720).any()
    print(f"  → {has_bdnf_sds}")
    if not has_bdnf_sds:
        print("  Note: bulk query may be returning a different gene's SDS for Bdnf.")
        bdnf_in_bulk = bulk[bulk["gene_symbol"].str.lower() == "bdnf"]
        print(f"  Bdnf rows in bulk query: {len(bdnf_in_bulk)}")
        if len(bdnf_in_bulk):
            print(bdnf_in_bulk.head())

    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    print(f"  TEST 1 (known-good Bdnf SDS):      {'PASS' if test1 else 'FAIL'}")
    print(f"  TEST 2 (bulk query SDS_IDs):       {n_pass}/{len(bulk_ids)} pass")
    if test1 and n_pass >= 2:
        print("\n  → API works in general AND for bulk query SDS_IDs.")
        print("    Your 02_mouse_genes.py should work after cache cleanup.")
        print("    If it isn't, paste the actual stdout from a fresh run.")
    elif test1 and n_pass == 0:
        print("\n  → API works for curated SDS but NOT for bulk query SDS_IDs.")
        print("    The bulk RmaApi filter is returning bad datasets.")
        print("    Fix: use MouseAtlasApi.get_section_data_sets instead, or")
        print("    expand the curated 02b list to ~300 well-known genes.")
    elif not test1:
        print("\n  → Allen API itself is broken — even known-good SDS fails.")
        print("    May need to wait for upstream fix or use a different data source.")


if __name__ == "__main__":
    main()
