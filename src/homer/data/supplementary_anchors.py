"""Supplementary anchor mechanism — promote existing parcels to anchors.

Original Garin atlas defines 21 pair_ids × 2 hemispheres = 42 anchors. We can
add more anchors *without modifying the underlying FC matrix or parcellation*
by **promoting existing non-anchor parcels** to be anchors with new pair_ids
(22, 23, ...).

Why promote rather than add new rows: our 1864 mouse / 2094 human parcels are
fixed by the FC processing pipeline. We can't add a new parcel — the FC
matrix would lack data for it. We can only *select* an existing parcel and
designate it as anchor.

Workflow:

1. Author a YAML config (`config/supplementary_anchors.yaml`) listing new
   pair_ids with mouse + human node identifiers.
2. Call ``apply_supplementary_anchors(M, H, config)`` which returns
   *modified copies* of M.var and H.var with the new anchors flagged.
3. Pass the modified AnnDatas to the solver. Existing
   ``get_anchor_index(M.var)`` picks up the new anchors automatically since
   it reads ``garin_anchor`` + ``anchor_pair_id``.

YAML schema:

.. code-block:: yaml

    # config/supplementary_anchors.yaml
    - pair_id: 22
      label: "M1 narrow (precentral gyrus)"
      hemisphere_L:
        mouse_node_id: "L_708"
        human_node_id: "L_NNN"
      hemisphere_R:
        mouse_node_id: "R_708"
        human_node_id: "R_NNN"
    - pair_id: 23
      label: "CA1"
      ...

Or by xyz centroid (auto-resolves to nearest existing parcel):

.. code-block:: yaml

    - pair_id: 22
      label: "M1 narrow"
      mouse_centroid_mm: [-1.5, 2.6, 1.8]
      human_centroid_mm: [-35, -20, 55]
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


@dataclass
class SuppAnchorEntry:
    pair_id: int
    label: str
    L_mouse_idx: int
    L_human_idx: int
    R_mouse_idx: int
    R_human_idx: int


def _resolve_node(var, node_spec, centroid_mm) -> int:
    """Resolve a node spec (either node_id string or xyz centroid) to a
    positional index into var."""
    if node_spec is not None:
        if node_spec not in var.index and str(node_spec) not in var.index:
            raise KeyError(f"node_id {node_spec!r} not in var.index "
                           f"(first few: {list(var.index[:3])})")
        idx_label = node_spec if node_spec in var.index else str(node_spec)
        return int(var.index.get_loc(idx_label))
    if centroid_mm is None:
        raise ValueError("either node_id or centroid_mm required")
    xyz = var[["x", "y", "z"]].to_numpy()
    d = np.linalg.norm(xyz - np.asarray(centroid_mm)[None, :], axis=1)
    return int(d.argmin())


def parse_supplementary_anchors_config(
    config: list[dict] | str | Path,
    var_m: pd.DataFrame,
    var_h: pd.DataFrame,
) -> list[SuppAnchorEntry]:
    """Parse a list-of-dicts (loaded from YAML or passed inline) into resolved
    node indices.
    """
    if isinstance(config, (str, Path)):
        if not _HAS_YAML:
            raise ImportError("pyyaml required to load YAML config; install it or "
                              "pass an inline list-of-dicts.")
        config = yaml.safe_load(Path(config).read_text())
    if not isinstance(config, list):
        raise TypeError(f"config must be list of dicts; got {type(config)}")

    out: list[SuppAnchorEntry] = []
    for entry in config:
        pid = int(entry["pair_id"])
        if pid <= 21:
            raise ValueError(f"supplementary pair_ids must be >21 to avoid clashing "
                             f"with the 21 Garin pairs; got {pid}")
        label = str(entry.get("label", f"supp_anchor_{pid}"))
        L_mouse = _resolve_node(
            var_m,
            entry.get("hemisphere_L", {}).get("mouse_node_id"),
            entry.get("mouse_centroid_mm"),
        )
        L_human = _resolve_node(
            var_h,
            entry.get("hemisphere_L", {}).get("human_node_id"),
            entry.get("human_centroid_mm"),
        )
        # If the user gave only L-side centroids, mirror them to R by negating x
        R_mouse_node = entry.get("hemisphere_R", {}).get("mouse_node_id")
        R_human_node = entry.get("hemisphere_R", {}).get("human_node_id")
        m_centroid = entry.get("mouse_centroid_mm")
        h_centroid = entry.get("human_centroid_mm")
        R_mouse = _resolve_node(
            var_m,
            R_mouse_node,
            None if m_centroid is None else [-m_centroid[0], m_centroid[1], m_centroid[2]],
        )
        R_human = _resolve_node(
            var_h,
            R_human_node,
            None if h_centroid is None else [-h_centroid[0], h_centroid[1], h_centroid[2]],
        )
        out.append(SuppAnchorEntry(
            pair_id=pid, label=label,
            L_mouse_idx=L_mouse, L_human_idx=L_human,
            R_mouse_idx=R_mouse, R_human_idx=R_human,
        ))
    return out


def apply_supplementary_anchors(
    var_m: pd.DataFrame,
    var_h: pd.DataFrame,
    entries: list[SuppAnchorEntry],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return modified copies of var_m, var_h with supplementary anchors
    flagged via garin_anchor=True and anchor_pair_id=<pid>.

    Idempotent: if a node is already an anchor (existing Garin), this raises.
    """
    var_m_out = var_m.copy()
    var_h_out = var_h.copy()

    # Cast `subregion` from Categorical to plain object so we can set new
    # tagged values without "Cannot setitem on a Categorical" errors.
    for v in (var_m_out, var_h_out):
        if isinstance(v["subregion"].dtype, pd.CategoricalDtype):
            v["subregion"] = v["subregion"].astype("object")

    for e in entries:
        for species, idx, label_prefix, var_out in [
            ("mouse", e.L_mouse_idx, "L", var_m_out),
            ("mouse", e.R_mouse_idx, "R", var_m_out),
            ("human", e.L_human_idx, "L", var_h_out),
            ("human", e.R_human_idx, "R", var_h_out),
        ]:
            row = var_out.iloc[idx]
            if bool(row["garin_anchor"]):
                raise ValueError(
                    f"node at positional index {idx} (region={row['region']}) is "
                    f"already a Garin anchor (pair_id={int(row['anchor_pair_id'])}); "
                    f"choose a different node for supplementary pair_id={e.pair_id}."
                )
            var_out.iat[idx, var_out.columns.get_loc("garin_anchor")] = True
            var_out.iat[idx, var_out.columns.get_loc("anchor_pair_id")] = e.pair_id
            # `subregion` is purely informational; tag it for traceability
            sub_col = var_out.columns.get_loc("subregion")
            var_out.iat[idx, sub_col] = f"[supp:{e.label}] {row['subregion']}"

    return var_m_out, var_h_out


def summarize_supplementary_anchors(
    entries: list[SuppAnchorEntry],
    var_m: pd.DataFrame,
    var_h: pd.DataFrame,
) -> str:
    """Human-readable summary of an entry list — useful for docs / logs."""
    lines = []
    for e in entries:
        m_L = var_m.iloc[e.L_mouse_idx]
        m_R = var_m.iloc[e.R_mouse_idx]
        h_L = var_h.iloc[e.L_human_idx]
        h_R = var_h.iloc[e.R_human_idx]
        lines.append(
            f"pid={e.pair_id} {e.label}\n"
            f"  L mouse: {m_L['region']:30s} ({m_L['x']:+5.2f},{m_L['y']:+5.2f},{m_L['z']:+5.2f})\n"
            f"  R mouse: {m_R['region']:30s} ({m_R['x']:+5.2f},{m_R['y']:+5.2f},{m_R['z']:+5.2f})\n"
            f"  L human: {h_L['region']:30s} ({h_L['x']:+6.1f},{h_L['y']:+6.1f},{h_L['z']:+6.1f})\n"
            f"  R human: {h_R['region']:30s} ({h_R['x']:+6.1f},{h_R['y']:+6.1f},{h_R['z']:+6.1f})"
        )
    return "\n\n".join(lines)
