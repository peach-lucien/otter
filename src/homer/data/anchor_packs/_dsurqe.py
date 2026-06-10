"""Shared DSURQE atlas helpers for anchor pack modules.

Anchor packs that need to look up mouse parcels by anatomical region name
share the same DSURQE-overlay logic (Beauchamp 2022's atlas in CCFv3 200μm).
This module centralises that so each pack stays small.

Private API (underscore prefix); not exported from the package.
"""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

# Mouse-coord → DSURQE-volume world-coord offset, calibrated from 6 Garin
# anchors (see pipeline/05f_beauchamp_validation.py for the derivation).
# Used by ``assign_dsurqe_labels`` to place each parcel into the DSURQE
# atlas volume for the live region lookup.
DSURQE_OFFSET_MM = np.array([-0.027, -2.334, +1.018])


def parse_dsurqe_tree(path: Path) -> dict[str, set[int]]:
    """Return ``{region_name: set(DSURQE label IDs descended from that node)}``."""
    tree = json.loads(Path(path).read_text())

    def _normlab(L):
        return [] if not L else ([L] if isinstance(L, int) else [int(x) for x in L])

    def _walk(node):
        out = [{"name": node.get("name"), "labels": _normlab(node.get("label"))}]
        for c in (node.get("children") or {}).values():
            out.extend(_walk(c))
        return out

    return {n["name"]: set(n["labels"]) for n in _walk(tree["msg"][0]) if n["labels"]}


def assign_dsurqe_labels(
    M_var: pd.DataFrame, dsurqe_volume_path: Path, *, radius: int = 2,
) -> np.ndarray:
    """Return ``(n_parcels,)`` array of DSURQE label IDs (0 if no overlap)."""
    from collections import Counter
    img = nib.load(str(dsurqe_volume_path))
    labels = np.asarray(img.get_fdata()).astype(np.int32)
    sh = labels.shape
    xyz = M_var[["x", "y", "z"]].to_numpy() + DSURQE_OFFSET_MM
    inv = np.linalg.inv(img.affine)
    voxels = (inv @ np.c_[xyz, np.ones(len(xyz))].T).T[:, :3]
    i, j, k = (voxels[:, ax].round().astype(int) for ax in range(3))
    out = np.zeros(len(xyz), dtype=np.int32)
    for p in range(len(xyz)):
        i0, i1 = max(0, i[p] - radius), min(sh[0], i[p] + radius + 1)
        j0, j1 = max(0, j[p] - radius), min(sh[1], j[p] + radius + 1)
        k0, k1 = max(0, k[p] - radius), min(sh[2], k[p] + radius + 1)
        block = labels[i0:i1, j0:j1, k0:k1].ravel()
        nz = block[block > 0]
        if len(nz):
            out[p] = Counter(nz.tolist()).most_common(1)[0][0]
    return out


def _build_dsurqe_ancestor_map(tree_path: Path) -> dict[str, set[str]]:
    """Return ``{node_name: set(ancestor_names_including_self)}`` from the DSURQE tree.

    Helper for resolving the parcel table's precomputed DSURQE vote labels
    against the tree; currently unused by the production lookup (see
    ``mouse_parcels_in_dsurqe_region`` for why we use the live atlas volume
    instead).
    """
    tree = json.loads(Path(tree_path).read_text())
    out: dict[str, set[str]] = {}

    def _walk(node, ancestors: tuple[str, ...]) -> None:
        nm = node.get("name")
        chain = ancestors + ((nm,) if nm else ())
        if nm:
            out[nm] = set(chain)
        for c in (node.get("children") or {}).values():
            _walk(c, chain)

    _walk(tree["msg"][0], ())
    return out


def _has_precomputed_dsurqe_votes(M_var: pd.DataFrame) -> bool:
    """True iff M_var carries the precomputed DSURQE vote labels."""
    return "region_vote_ss_dsq" in M_var.columns


def mouse_parcels_in_dsurqe_region(
    M_var: pd.DataFrame, region_name: str, atlas_root: Path | str = ".",
) -> list[int]:
    """Return positional indices of mouse parcels in the named DSURQE region.

    Resolves ``region_name`` against the DSURQE hierarchy (so e.g. "Primary
    motor area" picks up all its leaf labels) and returns every parcel
    whose DSURQE label lies in that subtree. Uses Beauchamp 2022's live
    atlas volume + the hand-calibrated ``DSURQE_OFFSET_MM``.

    .. note::
       We resolve regions via the live DSURQE atlas volume rather than the
       parcel table's precomputed DSURQE vote labels (``region_vote_ss_dsq``).
       The precomputed vote vocabulary uses different region names than the
       anchor packs query for — e.g. packs ask for "Caudoputamen" but the
       votes say "striatum" (a parent in the tree), and "Periaqueductal gray"
       (American) vs "periaqueductal grey" (British) — so a direct
       subtree-membership check against those labels returns empty sets for
       most queries. Consuming them directly would first require a
       name-mapping table from pack names to the vote vocabulary (see
       ``_paul_vote_bridge.py``). For the quantitative live-vs-votes
       comparison that backs this choice, see
       ``experiments/dsurqe_lookup_crosscheck/``.

    Raises FileNotFoundError if the Beauchamp 2022 DSURQE atlas isn't
    present at ``data_external/MouseHumanTranscriptomicSimilarity/AMBA/data/``.
    """
    atlas_root = Path(atlas_root)
    base = atlas_root / "data_external/MouseHumanTranscriptomicSimilarity/AMBA/data"
    tree_path = base / "DSURQE_tree.json"
    vol_path  = base / "imaging/DSURQE_CCFv3_labels_200um.mnc"
    if not tree_path.exists() or not vol_path.exists():
        raise FileNotFoundError(
            f"Beauchamp 2022 DSURQE atlas not found under {base!s}. "
            f"Clone github.com/...MouseHumanTranscriptomicSimilarity/ into "
            f"data_external/ to enable mouse-side anchor lookups."
        )
    name_to_lbl = parse_dsurqe_tree(tree_path)
    if region_name not in name_to_lbl:
        raise KeyError(
            f"{region_name!r} not in DSURQE hierarchy. Known regions include: "
            f"{sorted(list(name_to_lbl.keys()))[:5]}..."
        )
    parcel_lbl = assign_dsurqe_labels(M_var, vol_path)
    return [int(i) for i in np.where(np.isin(parcel_lbl, list(name_to_lbl[region_name])))[0]]


def human_parcels_in_mni_sphere(
    H_var: pd.DataFrame, centroid_xyz_left: tuple[float, float, float],
    radius_mm: float, *, mirror_to_right: bool = True,
) -> list[int]:
    """Return positional indices of human parcels within ``radius_mm`` of the
    centroid. If ``mirror_to_right`` (default), the right-hemisphere mirror
    is included automatically.
    """
    h_xyz = H_var[["x", "y", "z"]].to_numpy()
    xL, y, z = centroid_xyz_left
    cL = np.array([xL, y, z]); cR = np.array([-xL, y, z])
    if mirror_to_right:
        d = np.minimum(np.linalg.norm(h_xyz - cL, axis=1),
                        np.linalg.norm(h_xyz - cR, axis=1))
    else:
        d = np.linalg.norm(h_xyz - cL, axis=1)
    return [int(i) for i in np.where(d <= radius_mm)[0]]


def mouse_parcels_in_mouse_sphere(
    M_var: pd.DataFrame, centroid_xyz_left: tuple[float, float, float],
    radius_mm: float, *, mirror_to_right: bool = True,
) -> list[int]:
    """Return positional indices of *mouse* parcels within ``radius_mm`` of
    the centroid, expressed in M_var's coordinate system (NOT bregma-centered
    CCFv3 — apply the DSURQE_OFFSET_MM correction first if your reference
    centroid is in bregma coords).

    Symmetric to ``human_parcels_in_mni_sphere``. Use this for structures
    that DSURQE doesn't expose as labels at the 200μm parcel resolution
    (habenula, locus coeruleus, substantia nigra, claustrum, raphe nuclei).

    Bregma → M_var coordinate translation
    --------------------------------------
    The DSURQE atlas uses bregma-relative coordinates. M_var's xyz are
    offset by the ``DSURQE_OFFSET_MM = [-0.027, -2.334, +1.018]``
    vector (see beauchamp_validation pipeline). To convert a bregma
    coord to our system, *subtract* this offset:

        M_var_xyz = bregma_xyz - DSURQE_OFFSET_MM

    Example: if Allen CCFv3 reports LC at bregma (±0.85, -5.4, -3.85),
    then M_var-space coords are (±0.88, -3.07, -4.87). Pass those to
    this function. We provide ``M_var_coords_from_bregma`` as a helper
    below.
    """
    m_xyz = M_var[["x", "y", "z"]].to_numpy()
    xL, y, z = centroid_xyz_left
    cL = np.array([xL, y, z]); cR = np.array([-xL, y, z])
    if mirror_to_right:
        d = np.minimum(np.linalg.norm(m_xyz - cL, axis=1),
                        np.linalg.norm(m_xyz - cR, axis=1))
    else:
        d = np.linalg.norm(m_xyz - cL, axis=1)
    return [int(i) for i in np.where(d <= radius_mm)[0]]


def M_var_coords_from_bregma(
    bregma_xyz: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Convert a bregma-relative CCFv3 coordinate (mm) to M_var space.

    Our mouse parcel xyz coordinates are offset from bregma by
    ``DSURQE_OFFSET_MM = (-0.027, -2.334, +1.018)`` (calibrated from 6
    Garin anchors with unambiguous DSURQE leaf IDs; see
    pipeline/05f_beauchamp_validation.py).

    To convert from a published bregma centroid to M_var coords:
        M_var_xyz = bregma_xyz - DSURQE_OFFSET_MM
    """
    x, y, z = bregma_xyz
    ox, oy, oz = DSURQE_OFFSET_MM
    return (x - ox, y - oy, z - oz)
