"""Tests for otter.data.supplementary_anchors."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from otter.data.supplementary_anchors import (
    SuppAnchorEntry,
    parse_supplementary_anchors_config,
    apply_supplementary_anchors,
    summarize_supplementary_anchors,
)
from otter.data.anchors import get_anchor_index


def _make_fake_var(n: int, n_anchors: int = 4, anchor_xyz=None) -> pd.DataFrame:
    """A minimal var DataFrame compatible with what get_anchor_index expects."""
    rng = np.random.default_rng(0)
    xyz = rng.uniform(-50, 50, (n, 3))
    if anchor_xyz is not None:
        xyz[:n_anchors] = anchor_xyz
    df = pd.DataFrame({
        "type":       np.where(np.arange(n) < n_anchors, 1, 2).astype(np.int8),
        "numid":      np.arange(1, n + 1, dtype=np.int32),
        "pairid":     np.where(np.arange(n) < n_anchors,
                               (np.arange(n) // 2) + 1, np.arange(n) + 100),
        "region":     [f"L_{i}" if i % 2 == 0 else f"R_{i}" for i in range(n)],
        "subregion":  ["-"] * n,
        "x":          xyz[:, 0],
        "y":          xyz[:, 1],
        "z":          xyz[:, 2],
        "hemisphere": np.array(["L" if i % 2 == 0 else "R" for i in range(n)]),
        "garin_anchor": np.arange(n) < n_anchors,
        "anchor_pair_id": pd.array(
            np.where(np.arange(n) < n_anchors, (np.arange(n) // 2) + 1, np.nan),
            dtype="Int64"),
    })
    df.index = df["numid"].astype(int).astype(str)
    df.index.name = "node_id"
    return df


def test_parse_node_id_form():
    var_m = _make_fake_var(20)
    var_h = _make_fake_var(20)
    config = [{
        "pair_id": 22, "label": "test_supp",
        "hemisphere_L": {"mouse_node_id": "5", "human_node_id": "5"},
        "hemisphere_R": {"mouse_node_id": "6", "human_node_id": "6"},
    }]
    entries = parse_supplementary_anchors_config(config, var_m, var_h)
    assert len(entries) == 1
    e = entries[0]
    assert e.pair_id == 22
    assert e.L_mouse_idx == 4   # node_id "5" is at positional index 4 (1-indexed numid)
    assert e.R_mouse_idx == 5


def test_parse_centroid_form_resolves_to_nearest():
    """Use centroid form; verify nearest existing parcel is selected."""
    anchor_xyz = np.array([
        [-10, +5, +0],   # node 1, L (already Garin)
        [+10, +5, +0],   # node 2, R (already Garin)
        [-20, +0, +0],   # node 3, L (already Garin)
        [+20, +0, +0],   # node 4, R (already Garin)
    ])
    var_m = _make_fake_var(20, anchor_xyz=anchor_xyz)
    var_h = _make_fake_var(20, anchor_xyz=anchor_xyz)
    # Want a centroid near (-15, +2, 0); nearest non-anchor would be one of nodes 5+
    # We'll pick nodes whose xyz happens to be near
    var_m.iloc[10, var_m.columns.get_loc("x")] = -15.0
    var_m.iloc[10, var_m.columns.get_loc("y")] =  +2.0
    var_m.iloc[10, var_m.columns.get_loc("z")] =  +0.0
    var_m.iloc[10, var_m.columns.get_loc("hemisphere")] = "L"
    var_h.iloc[10, var_h.columns.get_loc("x")] = -15.0
    var_h.iloc[10, var_h.columns.get_loc("y")] =  +2.0
    var_h.iloc[10, var_h.columns.get_loc("z")] =  +0.0
    var_h.iloc[10, var_h.columns.get_loc("hemisphere")] = "L"
    var_m.iloc[11, var_m.columns.get_loc("x")] = +15.0
    var_m.iloc[11, var_m.columns.get_loc("y")] =  +2.0
    var_m.iloc[11, var_m.columns.get_loc("z")] =  +0.0
    var_m.iloc[11, var_m.columns.get_loc("hemisphere")] = "R"
    var_h.iloc[11, var_h.columns.get_loc("x")] = +15.0
    var_h.iloc[11, var_h.columns.get_loc("y")] =  +2.0
    var_h.iloc[11, var_h.columns.get_loc("z")] =  +0.0
    var_h.iloc[11, var_h.columns.get_loc("hemisphere")] = "R"

    config = [{
        "pair_id": 22, "label": "centroid test",
        "mouse_centroid_mm": [-15, +2, 0],   # mirrored to (+15, +2, 0) for R
        "human_centroid_mm": [-15, +2, 0],
    }]
    entries = parse_supplementary_anchors_config(config, var_m, var_h)
    assert entries[0].L_mouse_idx == 10
    assert entries[0].R_mouse_idx == 11


def test_apply_marks_supplementary_as_anchor():
    var_m = _make_fake_var(20)
    var_h = _make_fake_var(20)
    config = [{
        "pair_id": 22, "label": "M1 narrow",
        "hemisphere_L": {"mouse_node_id": "5", "human_node_id": "5"},
        "hemisphere_R": {"mouse_node_id": "6", "human_node_id": "6"},
    }]
    entries = parse_supplementary_anchors_config(config, var_m, var_h)
    var_m_aug, var_h_aug = apply_supplementary_anchors(var_m, var_h, entries)

    # Originals untouched
    assert int(var_m["garin_anchor"].sum()) == 4
    # Augmented: 4 + 2 anchors
    assert int(var_m_aug["garin_anchor"].sum()) == 6
    assert int(var_h_aug["garin_anchor"].sum()) == 6
    # The 2 new anchors have pair_id 22
    new_pids = set(var_m_aug.loc[var_m_aug["garin_anchor"] & ~var_m["garin_anchor"],
                                  "anchor_pair_id"].astype(int))
    assert new_pids == {22}


def test_apply_picks_up_via_get_anchor_index():
    """After apply, the 6 anchors (4 Garin + 2 supplementary) come back."""
    var_m = _make_fake_var(20)
    var_h = _make_fake_var(20)
    config = [{
        "pair_id": 22, "label": "M1",
        "hemisphere_L": {"mouse_node_id": "5", "human_node_id": "5"},
        "hemisphere_R": {"mouse_node_id": "6", "human_node_id": "6"},
    }]
    entries = parse_supplementary_anchors_config(config, var_m, var_h)
    var_m_aug, var_h_aug = apply_supplementary_anchors(var_m, var_h, entries)
    idx_m = get_anchor_index(var_m_aug)
    idx_h = get_anchor_index(var_h_aug)
    assert len(idx_m) == 6
    assert len(idx_h) == 6
    # New anchors share the (22, L/R) keys across species
    keys_m = set(idx_m.keys); keys_h = set(idx_h.keys)
    assert (22, "L") in keys_m and (22, "L") in keys_h
    assert (22, "R") in keys_m and (22, "R") in keys_h
    # And the same total ordering
    assert idx_m.keys == idx_h.keys


def test_apply_rejects_already_anchor():
    var_m = _make_fake_var(20)
    var_h = _make_fake_var(20)
    # Try to promote node "1" which is already a Garin anchor
    config = [{
        "pair_id": 22, "label": "duplicate",
        "hemisphere_L": {"mouse_node_id": "1", "human_node_id": "5"},
        "hemisphere_R": {"mouse_node_id": "6", "human_node_id": "6"},
    }]
    entries = parse_supplementary_anchors_config(config, var_m, var_h)
    with pytest.raises(ValueError, match="already a Garin anchor"):
        apply_supplementary_anchors(var_m, var_h, entries)


def test_pair_id_le_21_rejected():
    var_m = _make_fake_var(20)
    var_h = _make_fake_var(20)
    config = [{
        "pair_id": 5,   # would clash with Garin
        "label": "bad",
        "hemisphere_L": {"mouse_node_id": "5", "human_node_id": "5"},
        "hemisphere_R": {"mouse_node_id": "6", "human_node_id": "6"},
    }]
    with pytest.raises(ValueError, match=">21"):
        parse_supplementary_anchors_config(config, var_m, var_h)
