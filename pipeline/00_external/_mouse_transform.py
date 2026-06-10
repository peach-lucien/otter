"""Heuristic mouse → CCFv3 coordinate transform (experiment-only).

This module wraps the 48-permutation + centroid-translation alignment
produced by ``00c_align_mouse_to_ccf.py``. The main pipeline does NOT use
it — ``01_mouse_sc.py`` / ``02_mouse_genes.py`` read pre-warped CCFv3 voxel
indices (``ns_center_ix`` / ``AS_ix``) directly from the mouse ``.mat``
file, so no coordinate transform is applied there.

It is retained only because the
``experiments/autism_subtypes/allen_expansion/`` chain
(``download_pagani_ish.py``) still depends on it.

The transform is computed once by 00c_align_mouse_to_ccf.py and saved to
data_external/_diagnostics/mouse_to_ccf_transform.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_transform(diagnostics_dir: Path) -> dict:
    """Load the mouse → CCFv3 transform JSON. Raises FileNotFoundError if 00c
    hasn't been run yet."""
    p = diagnostics_dir / "mouse_to_ccf_transform.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run scripts/external/00c_align_mouse_to_ccf.py first."
        )
    return json.loads(p.read_text())


def apply_transform(centres: np.ndarray, transform: dict) -> np.ndarray:
    """Convert (N, 3) colleague-mouse mm coords → (N, 3) CCFv3 mm coords.

    centres: per-node centres in the colleague's bregma-centred frame.
    transform: dict from load_transform(); keys 'perm', 'signs', 'shift_mm'.
    """
    perm = transform["perm"]; signs = transform["signs"]; shift = np.asarray(transform["shift_mm"])
    out = np.column_stack([
        signs[0] * centres[:, perm[0]],
        signs[1] * centres[:, perm[1]],
        signs[2] * centres[:, perm[2]],
    ])
    return out + shift


def colleague_voxel_to_ccf_world(rsmask_affine: np.ndarray,
                                  voxel_indices_1d: np.ndarray,
                                  rsmask_shape: tuple,
                                  one_based: bool, order: str,
                                  transform: dict) -> np.ndarray:
    """Convert a flat array of MATLAB voxel indices into CCFv3 world (mm) coords.

    1. Decode 1D index → 3D ijk in the colleague's mask using the given order.
    2. Apply rsmask.affine to get colleague-frame world (mm).
    3. Apply the discovered transform → CCFv3 world (mm).
    """
    idx = np.asarray(voxel_indices_1d, dtype=np.int64)
    if one_based: idx = idx - 1
    grid_size = int(np.prod(rsmask_shape))
    # Out-of-bounds indices are a hard error. This function expects indices
    # into the rsmask 200 µm grid; the pre-warped CCFv3/DSURQE voxel indices
    # carried in the mouse .mat file are on different grids and must NOT be
    # passed here — use ns_voxel_indices with the NS affine, or
    # ss_voxel_indices with the SS affine, instead. (Silently filtering
    # would propagate as a NaN-filled gene matrix downstream.)
    if idx.size > 0 and ((idx < 0).any() or (idx >= grid_size).any()):
        n_oob = int(((idx < 0) | (idx >= grid_size)).sum())
        first_bad = int(np.argmax((idx < 0) | (idx >= grid_size)))
        raise ValueError(
            f"colleague_voxel_to_ccf_world received {n_oob} out-of-bounds "
            f"indices for rsmask grid {rsmask_shape} (size {grid_size}). "
            f"First bad index at position {first_bad}: value {int(idx[first_bad])}. "
            f"Pre-warped CCFv3/DSURQE voxel indices must not be passed here — use "
            f"ns_voxel_indices with the NS affine, or ss_voxel_indices "
            f"with the SS affine, instead."
        )
    ijk = np.array(np.unravel_index(idx, rsmask_shape, order=order)).T   # (N, 3)
    homog = np.column_stack([ijk, np.ones(len(ijk))])
    world_colleague = (rsmask_affine @ homog.T).T[:, :3]
    return apply_transform(world_colleague, transform)


def ccf_world_to_voxel(world_mm: np.ndarray, ccf_resolution_um: int) -> np.ndarray:
    """CCFv3 world mm → voxel index (int). Origin is at CCFv3 voxel (0,0,0)."""
    res_mm = ccf_resolution_um / 1000.0
    return (world_mm / res_mm).astype(np.int64)
