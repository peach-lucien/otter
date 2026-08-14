"""Region-anchor supervision (multi-parcel anchors).

A region anchor declares: *each mouse parcel in mouse_indices is allowed to
map to any human parcel in human_indices, and forbidden everywhere else*.
This is a strict generalisation of point-anchor supervision (in which each
mouse parcel is forced to map to a single human parcel).

Region anchors are used because:
  - Garin's anchors are named anatomical regions, not single points, and
    Beauchamp 2022 supplies region-level pairs natively.
  - Aggregating each region into a single parcel loses the within-region
    structure: every sub-parcel of motor cortex is forced onto a single
    human anchor at the centroid of motor + premotor + FEF, ~14mm
    anterior of canonical M1.
  - With a region anchor, each mouse motor sub-parcel maps to any human
    precentral parcel, and the FC/SC structure selects within the
    supervised set.

Cost-matrix encoding (in cross-species M), conflict-aware across all entries:
    A cell (mp, hp) is *region-supervised* if mp is named by any entry's mouse
    set or hp by any entry's human set. One global compatibility mask is built
    first, then:
      M[mp, hp] = beta_in       if (mp, hp) co-occur in some entry  (allowed)
      M[mp, hp] = lam_outside   if supervised but incompatible
      M[mp, hp] = unchanged     if unsupervised, or point-anchor-protected
    A mouse parcel in several entries is allowed to map to the *union* of
    their human sets, the application is order-independent.

Usage::

    from otter.data.region_anchors import (
        RegionAnchorEntry, parse_region_anchors_config, apply_region_supervision
    )

    entries = parse_region_anchors_config(
        "config/region_anchors_motor.yaml", M.var, H.var,
    )
    M_cost = apply_region_supervision(M_cost, entries, lam=1.0)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


@dataclass
class RegionAnchorEntry:
    """A region anchor: each mouse parcel in `mouse_indices` is allowed
    to map to any human parcel in `human_indices`, forbidden elsewhere.
    """
    pair_id: int
    label: str
    mouse_indices: list[int] = field(default_factory=list)
    human_indices: list[int] = field(default_factory=list)


def _resolve_indices(
    var: pd.DataFrame,
    *,
    node_ids: Optional[Sequence[str]] = None,
    centroid_mm: Optional[Sequence[float]] = None,
    radius_mm: Optional[float] = None,
) -> list[int]:
    """Resolve a region spec to a list of positional indices into var.

    Two forms:
      - node_ids: explicit list of node_id strings (must exist in var.index).
      - centroid_mm + radius_mm: all parcels within `radius_mm` of `centroid_mm`.
    """
    if node_ids is not None:
        out = []
        for nid in node_ids:
            key = nid if nid in var.index else str(nid)
            if key not in var.index:
                raise KeyError(
                    f"node_id {nid!r} not in var.index "
                    f"(first few: {list(var.index[:3])})"
                )
            out.append(int(var.index.get_loc(key)))
        return out

    if centroid_mm is None or radius_mm is None:
        raise ValueError(
            "region spec needs either node_ids or (centroid_mm + radius_mm)"
        )
    xyz = var[["x", "y", "z"]].to_numpy()
    d = np.linalg.norm(xyz - np.asarray(centroid_mm)[None, :], axis=1)
    return np.where(d <= radius_mm)[0].tolist()


def parse_region_anchors_config(
    config: list[dict] | str | Path,
    var_m: pd.DataFrame,
    var_h: pd.DataFrame,
) -> list[RegionAnchorEntry]:
    """Parse a YAML / list-of-dicts config into resolved RegionAnchorEntry.

    Schema for each entry:

    .. code-block:: yaml

        - pair_id: 30
          label: "M1 region (Beauchamp precentral)"
          mouse:
            node_ids: ["L_708", "R_808", "L_715", ...]
            # OR
            centroid_mm: [-1.5, 2.6, 1.8]
            radius_mm: 1.5
          human:
            node_ids: ["L_935", ...]
            # OR
            centroid_mm: [-35, -20, 55]
            radius_mm: 15
    """
    if isinstance(config, (str, Path)):
        if not _HAS_YAML:
            raise ImportError(
                "pyyaml required to load YAML config; install it or pass an "
                "inline list-of-dicts."
            )
        config = yaml.safe_load(Path(config).read_text())
    if not isinstance(config, list):
        raise TypeError(f"config must be list of dicts; got {type(config)}")

    out: list[RegionAnchorEntry] = []
    for entry in config:
        pid = int(entry["pair_id"])
        if pid <= 21:
            raise ValueError(
                f"region-anchor pair_ids must be >21 to avoid clashing with "
                f"the 21 Garin pair_ids; got {pid}"
            )
        label = str(entry.get("label", f"region_anchor_{pid}"))
        mouse_spec = entry.get("mouse", {})
        human_spec = entry.get("human", {})
        m_idx = _resolve_indices(
            var_m,
            node_ids=mouse_spec.get("node_ids"),
            centroid_mm=mouse_spec.get("centroid_mm"),
            radius_mm=mouse_spec.get("radius_mm"),
        )
        h_idx = _resolve_indices(
            var_h,
            node_ids=human_spec.get("node_ids"),
            centroid_mm=human_spec.get("centroid_mm"),
            radius_mm=human_spec.get("radius_mm"),
        )
        if not m_idx or not h_idx:
            raise ValueError(
                f"region-anchor pid={pid} ({label}) resolved to empty set "
                f"(|mouse|={len(m_idx)}, |human|={len(h_idx)}); check the spec."
            )
        out.append(RegionAnchorEntry(
            pair_id=pid, label=label,
            mouse_indices=m_idx, human_indices=h_idx,
        ))
    return out


def apply_region_supervision(
    M: np.ndarray,
    entries: Sequence[RegionAnchorEntry],
    *,
    lam: float = 1.0,
    lam_outside: float = 0.15,
    beta_in: float = 0.0,
    protect: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply region-anchor supervision to a cross-species cost matrix M.

    A single global compatibility mask is built across all entries before
    anything is written, so the result is order-independent. A mouse parcel
    appearing in several entries maps to the union of those entries' human
    sets, and symmetrically for human parcels. Entries must not be applied
    sequentially, which would let a later overlapping entry overwrite an
    earlier one's allowed (0-cost) cells.

    Modifies a copy and returns it.

    Parameters
    ----------
    M : (n_m, n_h) ndarray
        Cross-species cost matrix. Modified copy is returned.
    entries : sequence of RegionAnchorEntry
        Region anchors to apply.
    lam : float, default 1.0
        Scale used only when ``lam_outside`` is ``None``, giving the hard
        0/1 wall behaviour.
    lam_outside : float, default 0.15
        Cost assigned to a supervised-but-incompatible cell. The default gives
        better-calibrated π distributions (see docs/03_results.md).
    beta_in : float, default 0.0
        Cost assigned to a compatible (allowed) mouse-human pair.
    protect : (n_m, n_h) bool ndarray, optional
        Cells that must never be raised to ``lam_outside``, typically the
        point-anchor "allowed" cells, so region anchors layered on top of
        point-anchor supervision do not clobber it.

    Notes
    -----
    A cell is *region-supervised* if its mouse row or its human column is
    named by any entry. Supervised cells become ``beta_in`` if the
    mouse/human pair is compatible (co-occurs in some entry) and
    ``lam_outside`` otherwise. Unsupervised cells and protected cells are
    left untouched.
    """
    if lam_outside is None:
        lam_outside = lam   # hard 0/1 wall behaviour
    M_out = np.array(M, copy=True)
    entries = [e for e in entries if e.mouse_indices and e.human_indices]
    if not entries:
        return M_out

    n_m, n_h = M_out.shape
    # one global compatibility mask, built before anything is written
    allowed = np.zeros((n_m, n_h), dtype=bool)
    sup_m = np.zeros(n_m, dtype=bool)
    sup_h = np.zeros(n_h, dtype=bool)
    for e in entries:
        mi = np.asarray(e.mouse_indices, dtype=int)
        hi = np.asarray(e.human_indices, dtype=int)
        allowed[mi[:, None], hi[None, :]] = True
        sup_m[mi] = True
        sup_h[hi] = True

    supervised = sup_m[:, None] | sup_h[None, :]
    M_out[supervised & ~allowed] = lam_outside   # supervised but incompatible
    M_out[allowed] = beta_in                     # every compatible pair is free

    if protect is not None:
        protect = np.asarray(protect, dtype=bool)
        M_out[protect] = np.asarray(M)[protect]  # never clobber point anchors
    return M_out


def summarize_region_anchors(
    entries: Sequence[RegionAnchorEntry],
    var_m: pd.DataFrame,
    var_h: pd.DataFrame,
) -> str:
    """Human-readable summary for logs / docs."""
    lines = []
    for e in entries:
        m_xyz = var_m[["x", "y", "z"]].iloc[e.mouse_indices].to_numpy()
        h_xyz = var_h[["x", "y", "z"]].iloc[e.human_indices].to_numpy()
        m_c = m_xyz.mean(axis=0); h_c = h_xyz.mean(axis=0)
        lines.append(
            f"pid={e.pair_id} {e.label}\n"
            f"  mouse: {len(e.mouse_indices)} parcels, centroid "
            f"({m_c[0]:+5.2f}, {m_c[1]:+5.2f}, {m_c[2]:+5.2f})\n"
            f"  human: {len(e.human_indices)} parcels, centroid "
            f"({h_c[0]:+6.1f}, {h_c[1]:+6.1f}, {h_c[2]:+6.1f})"
        )
    return "\n\n".join(lines)
