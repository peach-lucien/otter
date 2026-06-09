"""Data layer: load mat73-formatted v7.3 .mat files into AnnData.

Public API:
    DATA_DIR             — default location of the data subfolder.
    load_struct(species) — return the raw dict for a species.
    parse_t_table(...)   — turn the cell-array t into a tidy DataFrame.
    build_anndata(...)   — assemble an AnnData per species, optionally cache to disk.
    load_cached(...)     — reload a previously cached AnnData + FC tensor.
    stream_mean_fc(...)  — chunked compute of the mean FC matrix without
                            materialising the full per-subject tensor.

The file naming convention on disk is:
    <species>.h5ad        — AnnData (no FC tensor inside)
    <species>.fc.npy      — float32 FC tensor (n_nodes, n_nodes, n_subjects)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import mat73
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[4] / "data_crossspecies"

# Mapping from spoken species name to top-level key inside the .mat file.
_MAT_TOPKEY = {"mouse": "m", "human": "h"}

# ----- v1 / v2 mouse-table schemas ------------------------------------------
#
# Paul ships v2 with full Allen full-NAME labels in `*_ABA` columns and DSURQE
# atlas-specific labels (e.g., "CA1Or") in `*_DSURQUE` columns. Note Paul's
# spelling is "DSURQUE" (D-S-U-R-Q-U-E — with an extra U) — case-sensitive.
#
# Schema detection branches on exact-equality of the ht column list. v2-ish
# files with permuted/extra columns must be rejected; this keeps the loader
# from silently mis-decoding a partially-migrated future file.

_V1_HT = ["type", "numid", "pairid", "region", "subregion", "center", "indices"]

_V2_HT = [
    "type", "numid", "pairid", "region", "subregion",
    "AS_center_mm", "AS_ix", "AS_center_ix",
    "AS_region_center_DSURQUE", "AS_region_center_ABA",
    "AS_region_vote_DSURQUE",   "AS_region_vote_ABA",
    "DS_center_mm", "DS_ix", "DS_center_ix",
    "DS_region_center_DSURQUE", "DS_region_center_ABA",
    "DS_region_vote_DSURQUE",   "DS_region_vote_ABA",
]

# v2 grid shapes — used both for index-bound validation and for documenting
# what `np.unravel_index(..., order='F')` expects from downstream consumers.
#
# NS = "Native Space" = Allen CCFv3-2017 25 µm.
#   axcodes ('P','I','R'), affine origin (0, 0, 0)
# SS = "Standard Space" = DSURQE 70 µm.
#   axcodes ('R','A','S'), affine origin (-6.27, -8.19, -4.20)
_NS_SHAPE = (528, 320, 456)
_SS_SHAPE = (181, 274, 139)

# Where the v2 file lives by default. `_mat_path` prefers v2 if present in
# DATA_DIR/<this subfolder>/, falls back to v1 in DATA_DIR.
_V2_MOUSE_SUBDIR = "updated_connectom_0906_26"
_V2_MOUSE_FILENAME = "corrs_mouse_v2.mat"


def _detect_schema(ht: list[str]) -> str:
    """Return 'v1' or 'v2' based on exact equality of the ht column list.

    Raises ValueError on any other arrangement — including a v2 list with
    a 20th column appended or columns reordered. The strictness is intentional:
    silent mis-decoding of a half-migrated future file would be much worse
    than a loud failure here.
    """
    ht_list = list(ht)
    if ht_list == _V1_HT:
        return "v1"
    if ht_list == _V2_HT:
        return "v2"
    raise ValueError(
        f"unrecognised ht schema: {ht_list!r}. "
        f"Expected either v1 {_V1_HT!r} or v2 {_V2_HT!r}."
    )


# ---------------------------------------------------------------------------
# raw loader
# ---------------------------------------------------------------------------
def _mat_path(species: str, data_dir: Path | None) -> Path:
    """Return the .mat file path for a species.

    For ``species == "mouse"`` the resolver looks first for the v2 file in
    ``data_dir / _V2_MOUSE_SUBDIR / _V2_MOUSE_FILENAME``, then ``data_dir /
    _V2_MOUSE_FILENAME``, then falls back to v1 ``data_dir / corrs_mouse.mat``.
    For ``"human"`` it returns ``data_dir / corrs_human.mat``.

    Signature unchanged — still returns ``Path`` only. The schema detected at
    that path is available via the sibling helper ``_mat_path_and_schema``.
    """
    p, _schema = _mat_path_and_schema(species, data_dir)
    return p


def _mat_path_and_schema(species: str, data_dir: Path | None) -> tuple[Path, str]:
    """Resolve the .mat file path and report the schema version.

    Returns ``(Path, schema_str)`` where ``schema_str`` is ``"v1"`` or ``"v2"``.
    For ``"human"`` this is always ``"v1"`` (no v2 human file exists yet).

    Raises ``FileNotFoundError`` if no candidate file exists, listing all
    paths tried so the caller can see which were searched.
    """
    if species not in _MAT_TOPKEY:
        raise ValueError(
            f"species must be one of {list(_MAT_TOPKEY)}; got {species!r}"
        )
    data_dir = Path(data_dir) if data_dir else DATA_DIR

    if species == "mouse":
        # Preference order: v2 subfolder, v2 at data_dir root, then v1.
        candidates = [
            (data_dir / _V2_MOUSE_SUBDIR / _V2_MOUSE_FILENAME, "v2"),
            (data_dir / _V2_MOUSE_FILENAME,                    "v2"),
            (data_dir / "corrs_mouse.mat",                     "v1"),
        ]
    else:
        candidates = [(data_dir / f"corrs_{species}.mat", "v1")]

    for path, schema in candidates:
        if path.exists():
            return path, schema

    tried = ", ".join(str(p) for p, _ in candidates)
    raise FileNotFoundError(
        f"no .mat file for species {species!r}. Tried: {tried}"
    )


def load_struct(species: str, *, data_dir: Path | None = None) -> dict[str, Any]:
    """Load the raw struct for one species via mat73 (loads everything into memory)."""
    p = _mat_path(species, data_dir)
    d = mat73.loadmat(str(p))
    top = _MAT_TOPKEY[species]
    if top not in d:
        raise KeyError(f"expected top-level key {top!r} in {p.name}; got {list(d)}")
    return d[top]


def _deref_uint16_string(f: h5py.File, ref) -> str:
    """Decode a MATLAB-stored char array (uint16, UTF-16-LE) referenced by ref."""
    a = np.asarray(f[ref][:]).flatten()
    if a.dtype != np.uint16:
        raise TypeError(f"expected uint16 string ref, got dtype {a.dtype}")
    return bytes(memoryview(a.astype(np.uint16))).decode("utf-16-le", errors="replace").rstrip("\x00")


def load_metadata(species: str, *, data_dir: Path | None = None) -> dict[str, Any]:
    """Load t, ht, dirs, species via h5py with manual ref dereferencing."""
    p = _mat_path(species, data_dir)
    top = _MAT_TOPKEY[species]
    out: dict[str, Any] = {}
    with h5py.File(str(p), "r") as f:
        g = f[top]
        out["ht"] = [_deref_uint16_string(f, r) for r in np.asarray(g["ht"][:]).flatten()]
        sp = np.asarray(g["species"][:]).flatten().astype(np.uint16)
        out["species"] = bytes(memoryview(sp)).decode("utf-16-le", errors="replace").rstrip("\x00")

        t_refs = np.asarray(g["t"][:])
        n_cols, n_nodes = t_refs.shape
        out["t"] = []
        for j in range(n_nodes):
            row: list[Any] = []
            for i in range(n_cols):
                obj = np.asarray(f[t_refs[i, j]][:])
                if obj.dtype == np.uint16:
                    val: Any = bytes(memoryview(obj.flatten().astype(np.uint16))).decode(
                        "utf-16-le", errors="replace"
                    ).rstrip("\x00")
                else:
                    val = obj.flatten().astype(np.float64)
                    if val.shape == (1,):
                        val = float(val[0])
                row.append(val)
            out["t"].append(row)

        # Schema is a function of the ht column list — record alongside
        # so downstream consumers can dispatch without re-deriving.
        out["_schema"] = _detect_schema(out["ht"])

        dirs_refs = np.asarray(g["dirs"][:]).flatten()
        out["dirs"] = []
        for r in dirs_refs:
            try:
                s = _deref_uint16_string(f, r)
            except Exception:
                s = ""
            out["dirs"].append([s] if s else [])

    return out


def stream_mean_fc(
    species: str,
    *,
    data_dir: Path | None = None,
    block: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute mean FC and per-cell observation count without loading the full
    per-subject tensor into memory."""
    p = _mat_path(species, data_dir)
    top = _MAT_TOPKEY[species]
    with h5py.File(str(p), "r") as f:
        rr = f[f"{top}/rr"]
        n_subj, n_nodes, n_nodes2 = rr.shape
        if n_nodes != n_nodes2:
            raise ValueError(f"non-square FC: {rr.shape}")
        if block is None:
            block = rr.chunks[1] if rr.chunks else 256
        sum_fc = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        cnt    = np.zeros((n_nodes, n_nodes), dtype=np.int32)
        for b in range((n_nodes + block - 1) // block):
            j0, j1 = b * block, min((b + 1) * block, n_nodes)
            cd = rr[:, :, j0:j1]
            valid = ~np.isnan(cd)
            sum_fc[:, j0:j1] += np.where(valid, cd, 0.0).sum(axis=0)
            cnt[:, j0:j1]    += valid.sum(axis=0)
    mu = (sum_fc / np.maximum(cnt, 1)).astype(np.float32)
    mu[cnt == 0] = np.nan
    return mu, cnt, int(n_subj)


def stream_mean_fc_subset(
    species: str,
    *,
    exclude_subjects: list[int] | np.ndarray | None = None,
    include_subjects: list[int] | np.ndarray | None = None,
    data_dir: Path | None = None,
    block: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Like stream_mean_fc but lets you exclude (or include only) specific
    subjects by their 0-based index."""
    p = _mat_path(species, data_dir)
    top = _MAT_TOPKEY[species]
    with h5py.File(str(p), "r") as f:
        rr = f[f"{top}/rr"]
        n_subj_total, n_nodes, _ = rr.shape

        if include_subjects is not None and exclude_subjects is not None:
            raise ValueError("pass at most one of include_subjects / exclude_subjects")
        if include_subjects is not None:
            keep = np.zeros(n_subj_total, dtype=bool)
            keep[np.asarray(include_subjects, dtype=int)] = True
        elif exclude_subjects is not None:
            keep = np.ones(n_subj_total, dtype=bool)
            keep[np.asarray(exclude_subjects, dtype=int)] = False
        else:
            keep = np.ones(n_subj_total, dtype=bool)
        n_subj = int(keep.sum())
        if n_subj == 0:
            raise ValueError("no subjects after include/exclude")
        if block is None:
            block = rr.chunks[1] if rr.chunks else 256

        sum_fc = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        cnt    = np.zeros((n_nodes, n_nodes), dtype=np.int32)
        for b in range((n_nodes + block - 1) // block):
            j0, j1 = b * block, min((b + 1) * block, n_nodes)
            cd = rr[:, :, j0:j1]
            cd = cd[keep]
            valid = ~np.isnan(cd)
            sum_fc[:, j0:j1] += np.where(valid, cd, 0.0).sum(axis=0)
            cnt[:, j0:j1]    += valid.sum(axis=0)
    mu = (sum_fc / np.maximum(cnt, 1)).astype(np.float32)
    mu[cnt == 0] = np.nan
    return mu, cnt, n_subj


def stream_subject_nan_stats(
    species: str, *, data_dir: Path | None = None,
) -> dict[str, Any]:
    """Per-subject NaN diagnostics without loading the full tensor."""
    p = _mat_path(species, data_dir)
    top = _MAT_TOPKEY[species]
    with h5py.File(str(p), "r") as f:
        rr = f[f"{top}/rr"]
        n_subj, n_nodes, _ = rr.shape
        nan_per_subj  = np.zeros(n_subj, dtype=np.int64)
        block = rr.chunks[1] if rr.chunks else 256
        for b in range((n_nodes + block - 1) // block):
            j0, j1 = b * block, min((b + 1) * block, n_nodes)
            cd = rr[:, :, j0:j1]
            nan_per_subj += np.isnan(cd).reshape(n_subj, -1).sum(axis=1)
    return {
        "n_subjects": int(n_subj),
        "n_nodes":    int(n_nodes),
        "n_total":    int(nan_per_subj.sum()),
        "n_per_subj": nan_per_subj,
        "subjects_with_any_nan": int((nan_per_subj > 0).sum()),
    }


def parse_t_table(t: list[list[Any]], ht: list[str]) -> pd.DataFrame:
    """Turn the (n_nodes-long) list-of-lists `t` into a tidy DataFrame.

    Two schemas supported (detected from `ht`):

    v1 — 7 columns: ``type, numid, pairid, region, subregion, center, indices``.
        Produced columns:
            type, numid, pairid (ints)
            region, subregion (str)
            x, y, z (float, in species' native space)
            hemisphere ('L' or 'R')
            garin_anchor (bool, True iff type==1)
            anchor_pair_id (Int64; same value for the L/R partners of an anchor pair)
            voxel_indices (object, 1D int64 arrays into the rsmask 200 µm grid)

    v2 — 19 columns (mouse-only). Same identifier columns, but coordinates
        and indices are split into Native Space (NS = Allen CCFv3 25 µm,
        axcodes PIR, affine origin (0,0,0)) and Standard Space (SS = DSURQE
        70 µm, axcodes RAS, affine origin (-6.27, -8.19, -4.20)) variants.

        Produced columns (in addition to v1's identifier columns):
            x, y, z                      — populated from DS_center_mm (SS frame)
                                            for v1 backward compatibility.
                                            Differs from v1 `center` by ≤ 0.16 mm.
            voxel_indices                — populated from DS_ix (SS grid, 0-based,
                                            F-order). DIFFERENT GRID from v1
                                            (181×274×139 at 70 µm vs 62×94×47
                                            at 200 µm). Consumers that previously
                                            unravelled into the rsmask grid must
                                            migrate to SS or NS grid.
            centre_ns_x/y/z              — AS_center_mm, NS world mm.
            centre_ss_x/y/z              — DS_center_mm, SS world mm.
            ns_center_ix                 — scalar 0-based linear index into NS
                                            grid, F-order.
            ss_center_ix                 — scalar 0-based linear index into SS
                                            grid, F-order.
            ns_voxel_indices             — 1D 0-based int64 arrays into NS grid.
            ss_voxel_indices             — 1D 0-based int64 arrays into SS grid
                                            (same content as voxel_indices).
            region_center_ns_aba         — Allen full NAME (not acronym) at NS
                                            centre voxel.
            region_center_ns_dsq         — DSURQE atlas full label at NS centre.
            region_center_ss_aba         — Allen full NAME at SS centre.
            region_center_ss_dsq         — DSURQE atlas full label at SS centre.
            region_vote_ns_aba           — majority-vote Allen full NAME over
                                            NS voxel set.
            region_vote_ns_dsq           — majority-vote DSURQE label over
                                            NS voxel set.
            region_vote_ss_aba           — majority-vote Allen full NAME over
                                            SS voxel set.
            region_vote_ss_dsq           — majority-vote DSURQE label over
                                            SS voxel set.

        IMPORTANT CONVENTIONS for v2 indices:
            - Stored on disk as MATLAB 1-based, Fortran (column-major) linear
              indices. This loader CONVERTS TO 0-BASED at parse time. The
              resulting values are int64 and ready for
              ``np.unravel_index(idx, shape, order='F')`` directly — do NOT
              subtract 1 again downstream.
            - Order convention (F) lives in consumers; not encoded in the value.
            - Grid shape is ``_NS_SHAPE`` for ``ns_*`` / ``AS_*``, ``_SS_SHAPE``
              for ``ss_*`` / ``DS_*``.

        See also ``df.attrs`` set on the returned DataFrame for runtime-checkable
        metadata about index conventions. ``df.attrs`` does NOT survive
        anndata round-trips; the equivalent values are written to ``A.uns``
        by ``build_anndata``.
    """
    schema = _detect_schema(ht)
    if schema == "v1":
        return _parse_t_table_v1(t)
    elif schema == "v2":
        return _parse_t_table_v2(t)
    else:  # pragma: no cover — _detect_schema raises before reaching here
        raise ValueError(f"unsupported schema: {schema!r}")


def _parse_t_table_v1(t: list[list[Any]]) -> pd.DataFrame:
    """v1 path — unchanged from the original implementation. Kept verbatim."""
    types     = np.fromiter((int(row[0]) for row in t), dtype=np.int8,  count=len(t))
    numids    = np.fromiter((int(row[1]) for row in t), dtype=np.int32, count=len(t))
    pairids   = np.fromiter((int(row[2]) for row in t), dtype=np.int32, count=len(t))
    regions   = [str(row[3]) for row in t]
    subreg    = [str(row[4]) for row in t]
    centers   = np.stack([np.asarray(row[5], dtype=np.float64).ravel() for row in t])
    voxels    = [np.asarray(row[6], dtype=np.int64).ravel() for row in t]

    if centers.shape[1] != 3:
        raise ValueError(f"expected 3 coordinates per node, got {centers.shape}")

    df = pd.DataFrame({
        "type":      types,
        "numid":     numids,
        "pairid":    pairids,
        "region":    regions,
        "subregion": subreg,
        "x":         centers[:, 0],
        "y":         centers[:, 1],
        "z":         centers[:, 2],
    })
    df["hemisphere"] = np.where(
        df["region"].str.startswith("L_"), "L",
        np.where(df["region"].str.startswith("R_"), "R", "?"),
    )
    df["garin_anchor"] = df["type"] == 1
    df["anchor_pair_id"] = df["pairid"].where(df["garin_anchor"]).astype("Int64")
    df["voxel_indices"] = pd.Series(voxels, index=df.index, dtype=object)

    df.index = df["numid"].astype(int).astype(str)
    df.index.name = "node_id"

    # Tag schema on the DataFrame for runtime introspection. Note: df.attrs
    # does NOT survive serialisation — used only for in-process loader→consumer
    # hand-off. AnnData equivalents are populated by build_anndata into A.uns.
    df.attrs.update({
        "schema": "v1",
        "voxel_indices_grid": "rsmask",
        "voxel_indices_shape": (62, 94, 47),
        "voxel_indices_order": "F",
        "voxel_indices_one_based": False,  # v1 loader produces 0-based... see note
        "xyz_frame": "rsmask_world",
    })
    return df


def _decode_matlab_linear_indices(raw, *, grid_shape: tuple[int, int, int],
                                   field_name: str) -> np.ndarray:
    """Decode a MATLAB 1-based linear index array → 0-based int64.

    Validates that every value lies in ``[1, prod(grid_shape)]`` BEFORE
    decrement. Raises ValueError on any out-of-bounds value, naming the
    grid and the offending position.

    Robust to:
      - load_metadata producing a Python scalar for a 1-element array
        (I1 guard); promoted to a 1D array.
      - float64 input from h5py / mat73 (cast to int64 after rounding to
        nearest; values must be exactly integer-valued).
    """
    arr = np.atleast_1d(np.asarray(raw, dtype=np.float64)).ravel()
    # IEEE round-to-nearest before cast — guards against 1234.9999 artefacts.
    arr_int = np.rint(arr).astype(np.int64)
    if not np.array_equal(arr_int.astype(np.float64), arr):
        # Find first non-integer value for the error message
        bad_mask = arr_int.astype(np.float64) != arr
        bad_idx = int(np.argmax(bad_mask))
        raise ValueError(
            f"{field_name!r} contains a non-integer value at position {bad_idx}: "
            f"{arr[bad_idx]!r} (expected integer linear index)."
        )

    grid_size = int(np.prod(grid_shape))
    if arr_int.size > 0:
        if arr_int.min() < 1:
            bad = int(np.argmin(arr_int))
            raise ValueError(
                f"{field_name!r} value at position {bad} is {int(arr_int[bad])} "
                f"— MATLAB 1-based indices must be >= 1."
            )
        if arr_int.max() > grid_size:
            bad = int(np.argmax(arr_int))
            raise ValueError(
                f"{field_name!r} value at position {bad} is {int(arr_int[bad])} "
                f"— exceeds grid size {grid_size} (shape {grid_shape})."
            )
    return arr_int - 1  # 0-based


def _parse_t_table_v2(t: list[list[Any]]) -> pd.DataFrame:
    """v2 path — 19 columns, see ``parse_t_table`` docstring for the schema."""
    n = len(t)
    if n == 0:
        raise ValueError("v2 t-table is empty")

    # Validate row width — every row must have exactly 19 cells.
    for j, row in enumerate(t):
        if len(row) != 19:
            raise ValueError(
                f"v2 t-table row {j} has {len(row)} cells; expected 19"
            )

    # Identifier columns (unchanged semantics from v1)
    types   = np.fromiter((int(row[0]) for row in t), dtype=np.int8,  count=n)
    numids  = np.fromiter((int(row[1]) for row in t), dtype=np.int32, count=n)
    pairids = np.fromiter((int(row[2]) for row in t), dtype=np.int32, count=n)
    regions = [str(row[3]) for row in t]
    subregs = [str(row[4]) for row in t]

    # NS / Allen CCFv3 25 µm coordinates and indices
    as_centre_mm = np.stack([
        np.asarray(row[5], dtype=np.float64).ravel() for row in t
    ])
    if as_centre_mm.shape != (n, 3):
        raise ValueError(
            f"AS_center_mm must be (n, 3); got {as_centre_mm.shape}"
        )

    ns_voxel_indices = [
        _decode_matlab_linear_indices(row[6], grid_shape=_NS_SHAPE,
                                       field_name=f"AS_ix[node {j}]")
        for j, row in enumerate(t)
    ]
    ns_center_ix = np.array([
        int(_decode_matlab_linear_indices(row[7], grid_shape=_NS_SHAPE,
                                           field_name=f"AS_center_ix[node {j}]")[0])
        for j, row in enumerate(t)
    ], dtype=np.int64)

    rc_ns_dsq = [str(row[8])  for row in t]
    rc_ns_aba = [str(row[9])  for row in t]
    rv_ns_dsq = [str(row[10]) for row in t]
    rv_ns_aba = [str(row[11]) for row in t]

    # SS / DSURQE 70 µm coordinates and indices
    ds_centre_mm = np.stack([
        np.asarray(row[12], dtype=np.float64).ravel() for row in t
    ])
    if ds_centre_mm.shape != (n, 3):
        raise ValueError(
            f"DS_center_mm must be (n, 3); got {ds_centre_mm.shape}"
        )

    ss_voxel_indices = [
        _decode_matlab_linear_indices(row[13], grid_shape=_SS_SHAPE,
                                       field_name=f"DS_ix[node {j}]")
        for j, row in enumerate(t)
    ]
    ss_center_ix = np.array([
        int(_decode_matlab_linear_indices(row[14], grid_shape=_SS_SHAPE,
                                           field_name=f"DS_center_ix[node {j}]")[0])
        for j, row in enumerate(t)
    ], dtype=np.int64)

    rc_ss_dsq = [str(row[15]) for row in t]
    rc_ss_aba = [str(row[16]) for row in t]
    rv_ss_dsq = [str(row[17]) for row in t]
    rv_ss_aba = [str(row[18]) for row in t]

    # Backward-compat: x/y/z = DS_center_mm (SS frame, matches v1 frame to
    # sub-voxel precision). voxel_indices = ss_voxel_indices (distinct list
    # copy to avoid aliasing — see I8 in the design review).
    df = pd.DataFrame({
        "type":      types,
        "numid":     numids,
        "pairid":    pairids,
        "region":    regions,
        "subregion": subregs,
        "x":         ds_centre_mm[:, 0],
        "y":         ds_centre_mm[:, 1],
        "z":         ds_centre_mm[:, 2],
    })
    df["hemisphere"] = np.where(
        df["region"].str.startswith("L_"), "L",
        np.where(df["region"].str.startswith("R_"), "R", "?"),
    )
    df["garin_anchor"] = df["type"] == 1
    df["anchor_pair_id"] = df["pairid"].where(df["garin_anchor"]).astype("Int64")

    # voxel_indices is a list of SS-grid 0-based int64 arrays. Make a
    # distinct copy so a later mutation of ss_voxel_indices doesn't shadow.
    df["voxel_indices"] = pd.Series(
        [v.copy() for v in ss_voxel_indices],
        index=df.index, dtype=object,
    )

    # New v2-specific columns
    df["centre_ns_x"] = as_centre_mm[:, 0]
    df["centre_ns_y"] = as_centre_mm[:, 1]
    df["centre_ns_z"] = as_centre_mm[:, 2]
    df["centre_ss_x"] = ds_centre_mm[:, 0]
    df["centre_ss_y"] = ds_centre_mm[:, 1]
    df["centre_ss_z"] = ds_centre_mm[:, 2]
    df["ns_center_ix"] = ns_center_ix
    df["ss_center_ix"] = ss_center_ix
    df["ns_voxel_indices"] = pd.Series(ns_voxel_indices, index=df.index, dtype=object)
    df["ss_voxel_indices"] = pd.Series(ss_voxel_indices, index=df.index, dtype=object)
    df["region_center_ns_aba"] = rc_ns_aba
    df["region_center_ns_dsq"] = rc_ns_dsq
    df["region_center_ss_aba"] = rc_ss_aba
    df["region_center_ss_dsq"] = rc_ss_dsq
    df["region_vote_ns_aba"]   = rv_ns_aba
    df["region_vote_ns_dsq"]   = rv_ns_dsq
    df["region_vote_ss_aba"]   = rv_ss_aba
    df["region_vote_ss_dsq"]   = rv_ss_dsq

    # Validation gates
    if not np.array_equal(numids, np.arange(1, n + 1, dtype=np.int32)):
        raise ValueError(
            "v2 t-table numid must be exactly 1..n (in order); "
            f"first mismatch at row {int(np.argmax(numids != np.arange(1, n+1)))}."
        )

    df.index = df["numid"].astype(int).astype(str)
    df.index.name = "node_id"

    df.attrs.update({
        "schema": "v2",
        "voxel_indices_grid": "SS",
        "voxel_indices_shape": _SS_SHAPE,
        "voxel_indices_order": "F",
        "voxel_indices_one_based": False,  # already decremented at load
        "xyz_frame": "SS",
        "ss_center_voxel_is_com": False,   # see B2/L16: max 0.55 mm offset
        "ns_center_voxel_is_com": True,    # NS round-trip is exact
        "ns_grid_shape": _NS_SHAPE,
        "ss_grid_shape": _SS_SHAPE,
        "ns_axcodes": "PIR",
        "ss_axcodes": "RAS",
        "ns_affine_origin": (0.0, 0.0, 0.0),
        "ss_affine_origin": (-6.27, -8.19, -4.20),
    })
    return df


def build_anndata(
    species: str,
    *,
    data_dir: Path | None = None,
    cache_dir: Path | None = None,
    overwrite: bool = False,
    cache_per_subject: bool = False,
    cache_voxels: bool = False,
) -> ad.AnnData:
    """Build the per-species AnnData. Streams the FC tensor in chunks."""
    meta = load_metadata(species, data_dir=data_dir)
    fc_mean, fc_n_obs, n_subj = stream_mean_fc(species, data_dir=data_dir)
    n_nodes = fc_mean.shape[0]

    var = parse_t_table(meta["t"], meta["ht"])
    if len(var) != n_nodes:
        raise ValueError(f"t-table rows ({len(var)}) != FC nodes ({n_nodes})")

    raw_dirs = meta.get("dirs", [[] for _ in range(n_subj)])
    paths = []
    for d in raw_dirs:
        if isinstance(d, list):
            paths.append(d[0] if d else "")
        else:
            paths.append(str(d))
    obs = pd.DataFrame({
        "subject_id":  [f"{species}_{i:04d}" for i in range(n_subj)],
        "source_path": paths,
    }).set_index("subject_id")

    A = ad.AnnData(
        X=np.zeros((n_subj, n_nodes), dtype=np.float32),
        obs=obs,
        var=var.drop(columns=["voxel_indices"]),
        uns={"species": species},
    )
    A.uns["fc_mean"] = fc_mean
    A.uns["fc_z"]    = np.arctanh(np.clip(fc_mean, -0.999999, 0.999999)).astype(np.float32)
    A.uns["fc_n_obs"] = fc_n_obs
    A.uns["n_subjects"] = n_subj
    A.uns["n_nodes"]    = n_nodes
    A.var["n_subjects_min"] = fc_n_obs.min(axis=1).astype(np.int32)
    A.var["n_subjects_max"] = fc_n_obs.max(axis=1).astype(np.int32)
    A.uns["voxel_indices"] = [v.astype(np.int64) for v in var["voxel_indices"]]

    # Record the loader schema and frame metadata in A.uns. Mirrors the
    # df.attrs that parse_t_table sets, but survives h5ad round-trip.
    # Tuples are stored as numpy arrays so the read-back type is deterministic
    # (anndata serialises Python tuples as h5py datasets that round-trip as
    # ndarrays). Use np.array_equal / .tolist() to compare downstream.
    schema = var.attrs.get("schema", "v1")
    A.uns["mouse_schema" if species == "mouse" else f"{species}_schema"] = schema
    A.uns["xyz_frame"]  = var.attrs.get("xyz_frame", "rsmask_world")
    A.uns["voxel_indices_grid"]      = var.attrs.get("voxel_indices_grid", "rsmask")
    A.uns["voxel_indices_shape"]     = np.array(var.attrs.get("voxel_indices_shape", (62, 94, 47)), dtype=np.int64)
    A.uns["voxel_indices_order"]     = var.attrs.get("voxel_indices_order", "F")
    A.uns["voxel_indices_one_based"] = bool(var.attrs.get("voxel_indices_one_based", False))

    if schema == "v2":
        # Per-frame v2 metadata. Stored as ndarrays so h5ad round-trips return
        # ndarrays consistently (not Python tuples).
        A.uns["ns_grid_shape"]    = np.array(_NS_SHAPE, dtype=np.int64)
        A.uns["ss_grid_shape"]    = np.array(_SS_SHAPE, dtype=np.int64)
        A.uns["ns_axcodes"]       = "PIR"
        A.uns["ss_axcodes"]       = "RAS"
        A.uns["ns_affine_origin"] = np.array((0.0, 0.0, 0.0), dtype=np.float64)
        A.uns["ss_affine_origin"] = np.array((-6.27, -8.19, -4.20), dtype=np.float64)
        A.uns["ns_center_voxel_is_com"] = True   # NS round-trip is exact
        A.uns["ss_center_voxel_is_com"] = False  # max 0.55 mm offset; see L16
        # Distinct list copies for aliasing safety (B6 / I8 in REVIEW.md).
        A.uns["ns_voxel_indices"] = [v.astype(np.int64).copy() for v in var["ns_voxel_indices"]]
        A.uns["ss_voxel_indices"] = [v.astype(np.int64).copy() for v in var["ss_voxel_indices"]]
        # Strip the new ragged columns from var (anndata can't serialise ragged
        # object arrays) — the data lives in A.uns instead. The scalar
        # ns_center_ix / ss_center_ix DO stay in var (scalar int64).
        for col in ("ns_voxel_indices", "ss_voxel_indices"):
            if col in A.var.columns:
                A.var = A.var.drop(columns=[col])

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        h5_path = cache_dir / f"{species}.h5ad"
        if h5_path.exists() and not overwrite:
            raise FileExistsError(f"{h5_path} exists; pass overwrite=True to replace")
        # Pop all ragged voxel lists before writing — anndata can't serialise
        # object-dtype arrays of unequal length. Restore after.
        popped: dict[str, Any] = {}
        for key in ("voxel_indices", "ns_voxel_indices", "ss_voxel_indices"):
            if key in A.uns:
                popped[key] = A.uns.pop(key)
        A.write_h5ad(h5_path)
        if cache_voxels:
            # Aliases for backward compatibility: node_{i} is the legacy
            # filename pattern, ns_node_{i} / ss_node_{i} are v2 explicit.
            payload: dict[str, np.ndarray] = {}
            primary = popped.get("voxel_indices") or popped.get("ss_voxel_indices") or []
            for i, v in enumerate(primary):
                payload[f"node_{i}"] = v
            if "ns_voxel_indices" in popped:
                for i, v in enumerate(popped["ns_voxel_indices"]):
                    payload[f"ns_node_{i}"] = v
            if "ss_voxel_indices" in popped:
                for i, v in enumerate(popped["ss_voxel_indices"]):
                    payload[f"ss_node_{i}"] = v
            np.savez_compressed(cache_dir / f"{species}.voxels.npz", **payload)
        # Restore in-memory state.
        for k, v in popped.items():
            A.uns[k] = v

    return A


class CacheSchemaMismatch(RuntimeError):
    """Raised when a cached AnnData's schema doesn't match what's on disk."""


def load_cached(
    species: str,
    *,
    cache_dir: Path,
    data_dir: Path | None = None,
    strict_schema: bool = True,
) -> tuple[ad.AnnData, np.ndarray | None]:
    """Load a previously cached AnnData and (if present) the per-subject FC tensor.

    Parameters
    ----------
    species : str
    cache_dir : Path
        Where the cached h5ad/fc.npy live.
    data_dir : Path | None
        Where the source .mat files live; defaults to ``DATA_DIR``. Used to
        determine the *expected* schema. If passing ``None`` and the layout
        on disk has shifted between cache-build time and now, callers can
        pass ``strict_schema=False`` to suppress the cache-staleness check.
    strict_schema : bool, default True
        If True, raises ``CacheSchemaMismatch`` when the cached AnnData's
        recorded schema doesn't match the schema of the source .mat file
        currently on disk (e.g. cache built from v1 while v2 is now present,
        or vice versa). The error message includes the exact ``rm`` command
        to clear the stale cache.

    Returns
    -------
    (A, rr) : (AnnData, np.ndarray | None)
        ``rr`` is mmap-loaded if ``{species}.fc.npy`` exists; otherwise None.
    """
    cache_dir = Path(cache_dir)
    A = ad.read_h5ad(cache_dir / f"{species}.h5ad")

    if strict_schema:
        try:
            _, expected_schema = _mat_path_and_schema(species, data_dir)
        except FileNotFoundError:
            expected_schema = None
        if expected_schema is not None:
            schema_key = "mouse_schema" if species == "mouse" else f"{species}_schema"
            cached_schema = A.uns.get(schema_key)
            # Treat a missing schema key as legacy v1 cache — accept silently
            # if the on-disk file is also v1. The schema key was added when
            # we introduced v2 support; pre-existing v1 caches don't have it
            # but their content is still correct under v1 semantics.
            if cached_schema is None and expected_schema == "v1":
                pass  # legacy v1 cache compatible with v1 .mat file
            elif cached_schema is None or str(cached_schema) != expected_schema:
                h5ad_path = cache_dir / f"{species}.h5ad"
                fc_path   = cache_dir / f"{species}.fc.npy"
                voxels    = cache_dir / f"{species}.voxels.npz"
                cached_repr = "legacy (no schema tag)" if cached_schema is None else repr(cached_schema)
                raise CacheSchemaMismatch(
                    f"Cached {species!r} AnnData has schema "
                    f"{cached_repr} but the current source .mat file is "
                    f"schema {expected_schema!r}. The cache is stale.\n\n"
                    f"Clear it with:\n"
                    f"  rm -f {h5ad_path} {fc_path} {voxels}\n\n"
                    f"then rebuild via build_anndata(). "
                    f"Pass strict_schema=False to suppress this check."
                )

    fc_path = cache_dir / f"{species}.fc.npy"
    rr = np.load(fc_path, mmap_mode="r") if fc_path.exists() else None
    return A, rr
