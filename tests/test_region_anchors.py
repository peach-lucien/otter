"""Tests for homer.data.region_anchors."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from homer.data.region_anchors import (
    RegionAnchorEntry,
    parse_region_anchors_config,
    apply_region_supervision,
    summarize_region_anchors,
)


def _make_fake_var(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    xyz = rng.uniform(-50, 50, (n, 3))
    df = pd.DataFrame({
        "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
        "region": [f"node_{i}" for i in range(n)],
    })
    df.index = [str(i + 1) for i in range(n)]
    df.index.name = "node_id"
    return df


def test_parse_node_ids_form():
    var_m = _make_fake_var(20); var_h = _make_fake_var(20)
    config = [{
        "pair_id": 30, "label": "test region",
        "mouse": {"node_ids": ["1", "2", "3"]},
        "human": {"node_ids": ["5", "6"]},
    }]
    entries = parse_region_anchors_config(config, var_m, var_h)
    assert len(entries) == 1
    e = entries[0]
    assert e.pair_id == 30
    assert e.mouse_indices == [0, 1, 2]
    assert e.human_indices == [4, 5]


def test_parse_centroid_radius_form():
    var_m = pd.DataFrame({
        "x": [-1.0, +1.0, +20.0, -20.0],
        "y": [+1.0, -1.0, +20.0, -20.0],
        "z": [0.0, 0.0, 0.0, 0.0],
        "region": ["a", "b", "c", "d"],
    })
    var_m.index = ["1", "2", "3", "4"]; var_m.index.name = "node_id"
    var_h = var_m.copy()
    config = [{
        "pair_id": 30, "label": "centroid",
        "mouse": {"centroid_mm": [0, 0, 0], "radius_mm": 5.0},
        "human": {"centroid_mm": [0, 0, 0], "radius_mm": 5.0},
    }]
    entries = parse_region_anchors_config(config, var_m, var_h)
    # Both indices 0 (-1,+1,0) and 1 (+1,-1,0) are within 5mm of origin
    assert sorted(entries[0].mouse_indices) == [0, 1]
    assert sorted(entries[0].human_indices) == [0, 1]


def test_apply_region_supervision_size_1_matches_point_anchor_hard():
    """With lam_outside=1.0 (hard), region anchor of size 1 == point anchor."""
    n_m, n_h = 5, 5
    M = np.full((n_m, n_h), 0.5)   # neutral baseline
    entry = RegionAnchorEntry(pair_id=22, label="point",
                               mouse_indices=[2], human_indices=[3])
    out = apply_region_supervision(M, [entry], lam_outside=1.0)
    # Expect: row 2 is all 1.0 except column 3 which is 0.0
    assert out[2, 3] == 0.0
    assert (out[2, :] == np.array([1.0, 1.0, 1.0, 0.0, 1.0])).all()
    assert (out[:, 3] == np.array([1.0, 1.0, 0.0, 1.0, 1.0])).all()
    # Other cells unchanged
    assert out[0, 0] == 0.5
    assert out[4, 4] == 0.5


def test_apply_region_supervision_set_membership_hard():
    """With lam_outside=1.0 (hard), region anchor enforces 0/1 wall."""
    n_m, n_h = 6, 6
    M = np.full((n_m, n_h), 0.5)
    entry = RegionAnchorEntry(pair_id=22, label="motor region",
                               mouse_indices=[1, 2], human_indices=[3, 4])
    out = apply_region_supervision(M, [entry], lam_outside=1.0)
    # Mouse rows 1, 2 should have 0 at h ∈ {3,4}, 1.0 elsewhere
    assert out[1, 3] == 0.0 and out[1, 4] == 0.0
    assert out[2, 3] == 0.0 and out[2, 4] == 0.0
    assert out[1, 0] == 1.0 and out[1, 5] == 1.0
    assert out[2, 0] == 1.0 and out[2, 5] == 1.0
    # Human cols 3, 4 should have 0 only at m ∈ {1,2}, 1.0 at other rows
    assert out[0, 3] == 1.0 and out[5, 3] == 1.0
    assert out[3, 4] == 1.0 and out[4, 4] == 1.0
    # Untouched cells unchanged
    assert out[0, 0] == 0.5 and out[5, 5] == 0.5


def test_apply_region_supervision_default_is_soft():
    """Default (lam_outside=0.15) produces a soft wall: in-region cells are
    cost 0, outside cells are cost 0.15 (preferred but not forbidden)."""
    n_m, n_h = 5, 5
    M = np.full((n_m, n_h), 0.5)
    entry = RegionAnchorEntry(pair_id=22, label="soft",
                               mouse_indices=[2], human_indices=[3])
    out = apply_region_supervision(M, [entry])    # default lam_outside=0.15
    assert out[2, 3] == 0.0     # in-region still free
    assert np.allclose(out[2, [0, 1, 2, 4]], 0.15)   # outside is mild
    assert np.allclose(out[[0, 1, 3, 4], 3], 0.15)
    # Untouched cells unchanged
    assert out[0, 0] == 0.5
    assert out[4, 4] == 0.5


def test_pair_id_le_21_rejected():
    var = _make_fake_var(20)
    config = [{
        "pair_id": 5,   # would clash with Garin
        "label": "bad",
        "mouse": {"node_ids": ["1"]},
        "human": {"node_ids": ["1"]},
    }]
    with pytest.raises(ValueError, match=">21"):
        parse_region_anchors_config(config, var, var)


def test_empty_set_rejected():
    var = pd.DataFrame({
        "x": [10.0], "y": [10.0], "z": [10.0], "region": ["a"],
    })
    var.index = ["1"]; var.index.name = "node_id"
    # Centroid 100 mm away with 1 mm radius → empty
    config = [{
        "pair_id": 30, "label": "empty",
        "mouse": {"centroid_mm": [100, 100, 100], "radius_mm": 1.0},
        "human": {"node_ids": ["1"]},
    }]
    with pytest.raises(ValueError, match="empty set"):
        parse_region_anchors_config(config, var, var)


def test_apply_returns_copy_not_inplace():
    M = np.full((3, 3), 0.5)
    entry = RegionAnchorEntry(pair_id=22, label="test",
                               mouse_indices=[0], human_indices=[1])
    out = apply_region_supervision(M, [entry], lam=1.0)
    # Original M unchanged
    assert (M == 0.5).all()
    # Output M modified
    assert out[0, 1] == 0.0
