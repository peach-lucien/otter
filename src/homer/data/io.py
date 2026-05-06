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


# ---------------------------------------------------------------------------
# raw loader
# ---------------------------------------------------------------------------
def _mat_path(species: str, data_dir: Path | None) -> Path:
    if species not in _MAT_TOPKEY:
        raise ValueError(f"species must be one of {list(_MAT_TOPKEY)}; got {species!r}")
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    p = data_dir / f"corrs_{species}.mat"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


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

    Columns produced:
        type, numid, pairid (ints)
        region, subregion (str)
        x, y, z (float, in species' native space)
        hemisphere ('L' or 'R')
        garin_anchor (bool, True iff type==1)
        anchor_pair_id (Int64; same value for the L/R partners of an anchor pair)
    """
    if list(ht) != ["type", "numid", "pairid", "region", "subregion", "center", "indices"]:
        raise ValueError(f"unexpected header order: {ht}")

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

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        h5_path = cache_dir / f"{species}.h5ad"
        if h5_path.exists() and not overwrite:
            raise FileExistsError(f"{h5_path} exists; pass overwrite=True to replace")
        voxel_list = A.uns.pop("voxel_indices")
        A.write_h5ad(h5_path)
        if cache_voxels:
            np.savez_compressed(
                cache_dir / f"{species}.voxels.npz",
                **{f"node_{i}": v for i, v in enumerate(voxel_list)},
            )
        A.uns["voxel_indices"] = voxel_list

    return A


def load_cached(
    species: str, *, cache_dir: Path
) -> tuple[ad.AnnData, np.ndarray | None]:
    """Load a previously cached AnnData and (if present) the per-subject FC tensor."""
    cache_dir = Path(cache_dir)
    A  = ad.read_h5ad(cache_dir / f"{species}.h5ad")
    fc_path = cache_dir / f"{species}.fc.npy"
    rr = np.load(fc_path, mmap_mode="r") if fc_path.exists() else None
    return A, rr
