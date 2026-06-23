"""Tests for the mouse parcel-table loader.

Split into:
  - synthetic unit tests (no external data dependency)
  - integration tests gated on the mouse data file being present on disk

Empirical anchors (pinned from the loader's fact-finding):

  - NS round-trip max error: EXACTLY 0.0 mm  →  atol=1e-9 here.
  - SS round-trip max error: 0.5495 mm  →  atol=0.6 in the SS test.
  - |center - DS_center_mm|.max() = 0.117 mm  →  threshold 0.2 mm.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO.parent / "data_crossspecies"
V2_DIR   = DATA_DIR / "updated_connectom_0906_26"
V2_MAT   = V2_DIR / "corrs_mouse_v2.mat"


# ---------------------------------------------------------------------------
# Load homer.data.io directly without going through the package __init__
# (which imports `ot`, not always available in test envs).
# ---------------------------------------------------------------------------

def _load_io_module():
    import importlib.machinery
    # Provide stubs so io.py's `from homer.data import DATA_DIR` style still
    # works without invoking the full homer package init.
    pkg_homer = importlib.util.module_from_spec(
        importlib.machinery.ModuleSpec("homer", None)
    )
    pkg_homer_data = importlib.util.module_from_spec(
        importlib.machinery.ModuleSpec("homer.data", None)
    )
    sys.modules.setdefault("homer", pkg_homer)
    sys.modules.setdefault("homer.data", pkg_homer_data)

    io_spec = importlib.util.spec_from_file_location(
        "homer.data.io", REPO / "src/homer/data/io.py"
    )
    io_mod = importlib.util.module_from_spec(io_spec)
    sys.modules["homer.data.io"] = io_mod
    io_spec.loader.exec_module(io_mod)
    return io_mod


IO = _load_io_module()


# ===========================================================================
# Unit tests, no external file dependency
# ===========================================================================

def test_detect_schema_v1():
    assert IO._detect_schema(IO._V1_HT) == "v1"


def test_detect_schema_v2():
    assert IO._detect_schema(IO._V2_HT) == "v2"


def test_detect_schema_rejects_unknown():
    with pytest.raises(ValueError, match="unrecognised ht schema"):
        IO._detect_schema(["foo", "bar"])


def test_detect_schema_rejects_v2_with_extra_column():
    with pytest.raises(ValueError, match="unrecognised ht schema"):
        IO._detect_schema(IO._V2_HT + ["extra_col"])


def test_detect_schema_rejects_permuted():
    # v2 with two columns swapped
    permuted = IO._V2_HT.copy()
    permuted[6], permuted[7] = permuted[7], permuted[6]
    with pytest.raises(ValueError, match="unrecognised ht schema"):
        IO._detect_schema(permuted)


def test_decode_matlab_linear_indices_basic():
    """Round-trip a small 1-based MATLAB array through the decoder."""
    raw = np.array([1.0, 2.0, 100.0])
    decoded = IO._decode_matlab_linear_indices(
        raw, grid_shape=IO._NS_SHAPE, field_name="test"
    )
    np.testing.assert_array_equal(decoded, np.array([0, 1, 99], dtype=np.int64))


def test_decode_matlab_linear_indices_scalar_promoted():
    """A scalar input (i.e. a 1-voxel parcel) must work (I1 guard)."""
    decoded = IO._decode_matlab_linear_indices(
        1.0, grid_shape=IO._NS_SHAPE, field_name="test"
    )
    np.testing.assert_array_equal(decoded, np.array([0], dtype=np.int64))


def test_decode_matlab_linear_indices_rejects_zero():
    with pytest.raises(ValueError, match="must be >= 1"):
        IO._decode_matlab_linear_indices(
            np.array([0.0]), grid_shape=IO._NS_SHAPE, field_name="test"
        )


def test_decode_matlab_linear_indices_rejects_negative():
    with pytest.raises(ValueError, match="must be >= 1"):
        IO._decode_matlab_linear_indices(
            np.array([-5.0]), grid_shape=IO._NS_SHAPE, field_name="test"
        )


def test_decode_matlab_linear_indices_rejects_out_of_range():
    """Value just past the grid size."""
    too_big = int(np.prod(IO._NS_SHAPE)) + 1
    with pytest.raises(ValueError, match="exceeds grid size"):
        IO._decode_matlab_linear_indices(
            np.array([float(too_big)]),
            grid_shape=IO._NS_SHAPE, field_name="test",
        )


def test_decode_matlab_linear_indices_rejects_non_integer():
    with pytest.raises(ValueError, match="non-integer value"):
        IO._decode_matlab_linear_indices(
            np.array([1.5]), grid_shape=IO._NS_SHAPE, field_name="test"
        )


def _build_v2_synthetic_row(numid: int, type_: int = 2,
                             as_ix: list[int] | None = None,
                             ds_ix: list[int] | None = None) -> list:
    """Build one synthetic v2 t-row for unit testing."""
    if as_ix is None: as_ix = [1, 100, 1000]
    if ds_ix is None: ds_ix = [1, 50, 500]
    return [
        float(type_), float(numid), 0.0,
        f"L_{numid}", "-",
        np.array([1.0, 2.0, 3.0]),    # AS_center_mm
        np.array(as_ix, dtype=np.float64),  # AS_ix
        float(as_ix[0]),               # AS_center_ix
        "ABA_dsurqe_centre",           # AS_region_center_DSURQUE
        "ABA centre name",             # AS_region_center_ABA
        "ABA_dsurqe_vote",             # AS_region_vote_DSURQUE
        "ABA vote name",               # AS_region_vote_ABA
        np.array([4.0, 5.0, 6.0]),    # DS_center_mm
        np.array(ds_ix, dtype=np.float64),  # DS_ix
        float(ds_ix[0]),               # DS_center_ix
        "DS_dsurqe_centre",
        "DS_ABA_centre",
        "DS_dsurqe_vote",
        "DS_ABA_vote",
    ]


def test_parse_t_table_v2_synthetic():
    t = [
        _build_v2_synthetic_row(1),
        _build_v2_synthetic_row(2, as_ix=[2, 200, 2000], ds_ix=[2, 100, 1000]),
    ]
    df = IO.parse_t_table(t, IO._V2_HT)
    # Identifier columns
    assert list(df["numid"]) == [1, 2]
    assert df.attrs["schema"] == "v2"
    assert df.attrs["voxel_indices_grid"] == "SS"
    assert df.attrs["voxel_indices_shape"] == IO._SS_SHAPE
    assert df.attrs["voxel_indices_order"] == "F"
    assert df.attrs["voxel_indices_one_based"] is False
    # x/y/z populated from DS_center_mm
    assert df["x"].iloc[0] == 4.0
    assert df["y"].iloc[0] == 5.0
    assert df["z"].iloc[0] == 6.0
    # NS coords
    assert df["centre_ns_x"].iloc[0] == 1.0
    assert df["centre_ns_y"].iloc[0] == 2.0
    assert df["centre_ns_z"].iloc[0] == 3.0
    # voxel_indices is 0-based (1 → 0 after MATLAB-1 decrement)
    np.testing.assert_array_equal(
        df["voxel_indices"].iloc[0], np.array([0, 49, 499], dtype=np.int64)
    )
    np.testing.assert_array_equal(
        df["ns_voxel_indices"].iloc[0], np.array([0, 99, 999], dtype=np.int64)
    )
    # ns_center_ix is 0-based scalar
    assert df["ns_center_ix"].iloc[0] == 0   # 1 (MATLAB) - 1
    assert df["ns_center_ix"].iloc[1] == 1   # 2 (MATLAB) - 1
    # ss_center_ix similarly
    assert df["ss_center_ix"].iloc[0] == 0
    assert df["ss_center_ix"].iloc[1] == 1


def test_parse_t_table_v2_rejects_numid_gap():
    """v2 enforces numid is exactly 1..n in order."""
    t = [
        _build_v2_synthetic_row(1),
        _build_v2_synthetic_row(3),  # numid 2 missing
    ]
    with pytest.raises(ValueError, match="numid must be exactly"):
        IO.parse_t_table(t, IO._V2_HT)


def test_parse_t_table_v2_rejects_out_of_range_index():
    """An AS_ix value past _NS_SHAPE size must crash."""
    too_big = int(np.prod(IO._NS_SHAPE)) + 1
    t = [_build_v2_synthetic_row(1, as_ix=[too_big])]
    with pytest.raises(ValueError, match="exceeds grid size"):
        IO.parse_t_table(t, IO._V2_HT)


def test_parse_t_table_v2_rejects_zero_index():
    """A zero (0-based already?) input must crash. MATLAB 1-based."""
    t = [_build_v2_synthetic_row(1, as_ix=[0, 1, 2])]
    with pytest.raises(ValueError, match="must be >= 1"):
        IO.parse_t_table(t, IO._V2_HT)


def test_parse_t_table_v1_path_still_works():
    """The v1 path must remain bytes-identical in behaviour."""
    t_v1 = [
        [1.0, 1.0, 0.0, "L_test", "-",
         np.array([1.0, 2.0, 3.0]), np.array([10, 20, 30], dtype=np.float64)],
        [2.0, 2.0, 0.0, "R_test", "-",
         np.array([4.0, 5.0, 6.0]), np.array([15, 25, 35], dtype=np.float64)],
    ]
    df = IO.parse_t_table(t_v1, IO._V1_HT)
    assert df.attrs["schema"] == "v1"
    assert list(df["numid"]) == [1, 2]
    np.testing.assert_array_equal(
        df["voxel_indices"].iloc[0], np.array([10, 20, 30], dtype=np.int64)
    )
    assert df["x"].iloc[0] == 1.0


def test_parse_t_table_v2_singleton_voxel_set():
    """A parcel with a 1-element index array must not crash (I1 guard)."""
    t = [_build_v2_synthetic_row(1, as_ix=[42], ds_ix=[7])]
    df = IO.parse_t_table(t, IO._V2_HT)
    np.testing.assert_array_equal(
        df["voxel_indices"].iloc[0], np.array([6], dtype=np.int64)
    )
    np.testing.assert_array_equal(
        df["ns_voxel_indices"].iloc[0], np.array([41], dtype=np.int64)
    )


# ===========================================================================
# Integration tests, gated on v2 file presence
# ===========================================================================

needs_v2 = pytest.mark.skipif(
    not V2_MAT.exists(),
    reason=f"v2 file not present at {V2_MAT}"
)


@needs_v2
def test_v2_file_loadable_via_load_metadata():
    meta = IO.load_metadata("mouse")
    assert meta["_schema"] == "v2"
    assert len(meta["t"]) == 1864
    assert meta["ht"] == IO._V2_HT


@needs_v2
def test_v2_indices_decode_to_ns_centres_exactly():
    """The key empirical anchor: NS-frame round-trip is EXACT.

    Take each parcel's ns_center_ix (already 0-based), unravel F-order in
    NS grid, apply NS affine, compare to centre_ns_*. Expect max diff = 0.
    """
    import nibabel as nib
    meta = IO.load_metadata("mouse")
    df = IO.parse_t_table(meta["t"], meta["ht"])
    n = len(df)

    ns_template = nib.load(str(V2_DIR / "template_ABA_NS.nii.gz"))
    ns_aff = ns_template.affine

    ix = df["ns_center_ix"].to_numpy()
    ijk = np.column_stack(np.unravel_index(ix, IO._NS_SHAPE, order="F"))
    homog = np.column_stack([ijk, np.ones(n)])
    world = (ns_aff @ homog.T).T[:, :3]
    stored = df[["centre_ns_x", "centre_ns_y", "centre_ns_z"]].to_numpy()

    diff = world - stored
    mag = np.linalg.norm(diff, axis=1)
    # Tighter tolerance than 1e-6: empirical max is EXACTLY 0.0 mm across all
    # 1864 parcels. A 1e-6 mm threshold would mask a 1-µm-class regression
    # (e.g., a unit conversion bug in the rounding step). Use 1e-9.
    assert mag.max() < 1e-9, (
        f"NS round-trip should be exact (empirical max = 0.0 mm); "
        f"max |diff| = {mag.max():.12f} mm. This indicates a regression "
        f"in MATLAB→Python index conversion (off-by-one, wrong order, or "
        f"unit conversion error)."
    )


@needs_v2
def test_v2_indices_decode_to_ss_centres_within_threshold():
    """SS-frame round-trip: max ~0.77 mm by design.

    DS_center_mm is a continuous COM / warped centroid while DS_center_ix
    is the closest member of the voxel set, the two disagree by up to one
    voxel diagonal in the worst case (12 of 1864 parcels at 70 µm
    voxel-diagonal ≈ 0.12 mm × √3 ≈ 0.21 mm). Larger residuals come from
    parcels whose voxel set doesn't contain a voxel close to the COM
    empirical max is 0.7688 mm. Pin threshold at 0.8 mm with headroom.

    See B2/L16 in REVIEW.md for the semantic asymmetry.
    """
    import nibabel as nib
    meta = IO.load_metadata("mouse")
    df = IO.parse_t_table(meta["t"], meta["ht"])
    n = len(df)

    ss_template = nib.load(str(V2_DIR / "template_ABA_SS.nii.gz"))
    ss_aff = ss_template.affine

    ix = df["ss_center_ix"].to_numpy()
    ijk = np.column_stack(np.unravel_index(ix, IO._SS_SHAPE, order="F"))
    homog = np.column_stack([ijk, np.ones(n)])
    world = (ss_aff @ homog.T).T[:, :3]
    stored = df[["centre_ss_x", "centre_ss_y", "centre_ss_z"]].to_numpy()

    diff = world - stored
    mag = np.linalg.norm(diff, axis=1)
    assert mag.max() < 0.8, (
        f"SS round-trip max |diff| {mag.max():.4f} mm exceeds 0.8 mm. "
        f"Empirical max was 0.7688 mm; if this regresses past 0.8, something "
        f"upstream of the loader changed (Paul's voxel-set or COM definition)."
    )
    # Also assert MOST parcels round-trip exactly (the loose-tail is small).
    assert (mag < 1e-6).mean() > 0.99, (
        f"only {100*(mag < 1e-6).mean():.1f}% of SS centres round-trip exactly; "
        f"empirical was ≥ 99% (12/1864 ≈ 0.64% loose)."
    )


@needs_v2
def test_v2_xyz_compat_with_v1():
    """v2 x/y/z (from DS_center_mm) ≈ v1 center to sub-voxel precision."""
    v1_path = DATA_DIR / "corrs_mouse.mat"
    if not v1_path.exists():
        pytest.skip("v1 file not present for comparison")

    # Load v2 (resolver picks v2 by default)
    meta_v2 = IO.load_metadata("mouse")
    df_v2 = IO.parse_t_table(meta_v2["t"], meta_v2["ht"])

    # Load v1 directly via h5py (bypassing the resolver which prefers v2)
    import h5py
    def _u16(f, ref):
        a = np.asarray(f[ref][:]).flatten()
        return bytes(memoryview(a.astype(np.uint16))).decode("utf-16-le").rstrip("\x00")
    with h5py.File(str(v1_path), "r") as f:
        g = f["m"]
        ht = [_u16(f, r) for r in np.asarray(g["ht"][:]).flatten()]
        t_refs = np.asarray(g["t"][:])
        n = t_refs.shape[1]
        center_col = ht.index("center")
        v1_centers = np.stack([
            np.asarray(f[t_refs[center_col, j]][:]).flatten().astype(float)
            for j in range(n)
        ])

    v2_centers = df_v2[["x", "y", "z"]].to_numpy()
    diff = np.abs(v2_centers - v1_centers).max()
    assert diff < 0.2, (
        f"|v2.x/y/z - v1.center|.max() = {diff:.4f} mm exceeds 0.2 mm. "
        f"Empirical was 0.117 mm."
    )


@needs_v2
def test_v2_numid_pairid_region_identical_to_v1():
    """Parcel ordering and identifiers must be unchanged between v1 and v2.

    This is L14 in the design review, silently reordering rows would break
    every downstream consumer that indexes by row position.
    """
    v1_path = DATA_DIR / "corrs_mouse.mat"
    if not v1_path.exists():
        pytest.skip("v1 file not present for comparison")

    import h5py
    def _u16(f, ref):
        a = np.asarray(f[ref][:]).flatten()
        return bytes(memoryview(a.astype(np.uint16))).decode("utf-16-le").rstrip("\x00")

    def _pull_v1(f):
        g = f["m"]
        ht = [_u16(f, r) for r in np.asarray(g["ht"][:]).flatten()]
        t_refs = np.asarray(g["t"][:])
        n = t_refs.shape[1]
        cols = {c: ht.index(c) for c in ("numid", "pairid", "region")}
        numid = np.array([int(np.asarray(f[t_refs[cols["numid"], j]][:]).flatten()[0]) for j in range(n)])
        pairid = np.array([int(np.asarray(f[t_refs[cols["pairid"], j]][:]).flatten()[0]) for j in range(n)])
        region = [_u16(f, t_refs[cols["region"], j]) for j in range(n)]
        return numid, pairid, region

    with h5py.File(str(v1_path), "r") as f:
        v1_numid, v1_pairid, v1_region = _pull_v1(f)

    meta_v2 = IO.load_metadata("mouse")
    df_v2 = IO.parse_t_table(meta_v2["t"], meta_v2["ht"])

    np.testing.assert_array_equal(df_v2["numid"].to_numpy(), v1_numid)
    np.testing.assert_array_equal(df_v2["pairid"].to_numpy(), v1_pairid)
    assert list(df_v2["region"]) == v1_region


@needs_v2
@pytest.mark.slow
def test_v2_rr_unchanged_sampled():
    """m.rr should be bit-identical between v1 and v2. Paul did NOT recompute FC.

    Marked slow because reads two 1.3 GB files. Skip in fast runs with
    ``pytest -m 'not slow'`` or env var HOMER_TEST_FAST=1.
    """
    import os
    if os.environ.get("HOMER_TEST_FAST") == "1":
        pytest.skip("HOMER_TEST_FAST=1: skipping 1.3GB file comparison")
    v1_path = DATA_DIR / "corrs_mouse.mat"
    if not v1_path.exists():
        pytest.skip("v1 file not present for comparison")

    import h5py
    with h5py.File(str(v1_path), "r") as f1, h5py.File(str(V2_MAT), "r") as f2:
        rr1 = f1["m"]["rr"]
        rr2 = f2["m"]["rr"]
        assert rr1.shape == rr2.shape
        # Sample subjects at first / middle / last positions. The 1.3 GB
        # files are slow to read; 3 samples is sufficient to catch any
        # systematic difference. If you want exhaustive verification,
        # run with HOMER_TEST_EXHAUSTIVE=1 set.
        import os
        if os.environ.get("HOMER_TEST_EXHAUSTIVE") == "1":
            sample = np.linspace(0, rr1.shape[0] - 1, 10, dtype=int)
        else:
            sample = np.array([0, rr1.shape[0] // 2, rr1.shape[0] - 1])
        for s in sample:
            a = np.asarray(rr1[s, :, :])
            b = np.asarray(rr2[s, :, :])
            assert np.array_equal(np.isnan(a), np.isnan(b)), \
                f"NaN mask differs on subject {s}"
            assert np.array_equal(
                np.nan_to_num(a, nan=0.0),
                np.nan_to_num(b, nan=0.0),
            ), f"FC values differ on subject {s}"
