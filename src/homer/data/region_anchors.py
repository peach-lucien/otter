"""Region-anchor supervision (multi-parcel anchors).

A region anchor declares: *each mouse parcel in mouse_indices is allowed to
map to any human parcel in human_indices, and forbidden everywhere else*.
This is a strict generalisation of point-anchor supervision (in which each
mouse parcel is forced to map to a single human parcel).

Why region anchors:
  - Garin's original anchors are *named anatomical regions*, not single
    points. Beauchamp 2022 also gives us region-level pairs natively.
  - Aggregating each region into a single parcel (as the colleague's
    preprocessing does) loses the within-region structure: every
    sub-parcel of motor cortex is forced to map to a single human anchor
    that's the centroid of motor + premotor + FEF + ... ~14mm anterior of
    canonical M1 (see `docs/diagnostics.md`).
  - With a region anchor instead, each mouse motor sub-parcel is free to
    map to any human precentral parcel, letting the FC/SC structure pick
    the right one within the supervised set.

Cost-matrix encoding (in cross-species M):
    For each region anchor with mouse set Mset and human set Hset:
      M[mp, :]      = lam   for mp in Mset      (forbid by default)
      M[mp, hp]     = 0     for mp in Mset, hp in Hset    (allow within region)
      M[:, hp]      = lam   for hp in Hset      (forbid by default)
      M[mp, hp]     = 0     for mp in Mset, hp in Hset    (re-allow)

Usage::

    from homer.data.region_anchors import (
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
) -> np.ndarray:
    """Apply region-anchor supervision to a cross-species cost matrix M.

    Modifies a copy and returns it. For each entry, sets:
      - M[mp, :] = lam        for mp in mouse_indices  (forbid all human)
      - M[mp, hp] = 0         for hp in human_indices  (allow within region)
      - M[:, hp] = lam        for hp in human_indices  (forbid all mouse on column)
      - M[mp, hp] = 0         re-allow within region after column forbid

    The ordering of the row-then-column writes matters: we forbid the
    column first, then re-allow the in-region cells. Identical semantics
    to ``_apply_anchor_supervision`` for point anchors when the entries
    are size-1 sets.
    """
    M_out = np.array(M, copy=True)
    for e in entries:
        mset = list(e.mouse_indices)
        hset = list(e.human_indices)
        if not mset or not hset:
            continue
        # Forbid all human cells for these mouse parcels
        M_out[mset, :] = lam
        # Allow only the in-region human cells for these mouse parcels
        m_idx_arr = np.asarray(mset)[:, None]
        h_idx_arr = np.asarray(hset)[None, :]
        M_out[m_idx_arr, h_idx_arr] = 0.0
        # Forbid all mouse cells for these human parcels (other rows)
        for hp in hset:
            mask = np.ones(M_out.shape[0], dtype=bool)
            mask[mset] = False    # don't overwrite the in-region rows
            M_out[mask, hp] = lam
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
