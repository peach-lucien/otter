"""Smoke tests for pipeline/05f_beauchamp_validation.py, pure-Python pieces.

These tests exercise the data-side helpers without requiring the heavy
DSURQE label volume or running the full pipeline. They should run in <1s.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_FILE = ROOT / "pipeline" / "05f_beauchamp_validation.py"


@pytest.fixture(scope="module")
def mod():
    """Load 05f as a module without executing main()."""
    spec = importlib.util.spec_from_file_location("beauchamp_val", PIPELINE_FILE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _make_fake_human_var(n=2094):
    """Generate a regular MNI-grid dataframe similar to our real human atlas."""
    # Roughly the bounding box of our actual human parcels: x∈[-63,63], y∈[-99,63], z∈[-36,81]
    rng = np.random.default_rng(0)
    x = rng.uniform(-60, 60, n)
    y = rng.uniform(-95, 60, n)
    z = rng.uniform(-30, 75, n)
    return pd.DataFrame({"x": x, "y": y, "z": z})


def test_human_membership_centroid_radius(mod):
    """A point exactly at a centroid should always be inside; a point 100mm
    away from any centroid should be outside."""
    df = pd.DataFrame({
        "x": [-35, +35,    0,  100],
        "y": [-20, -20,    0,  100],
        "z": [+55, +55,    0,  100],
    })

    class FakeH:
        var = df
    name_to_mni = {
        "precentral gyrus": (-35, -20, 55, 15),
        "absent_region":     None,
    }
    out = mod.assign_human_region_membership(FakeH, name_to_mni)
    assert out["absent_region"] is None
    mask = out["precentral gyrus"]
    # First two points are at the L and R centroids, should be in
    assert mask[0]
    assert mask[1]
    # Third (origin) is ~70mm from either side, out
    assert not mask[2]
    # Fourth (100,100,100) is well out
    assert not mask[3]


def test_human_membership_radius_boundary(mod):
    """A point exactly on the boundary radius should be included."""
    df = pd.DataFrame({"x": [-35], "y": [-20], "z": [55 - 15.0]})  # 15mm below

    class FakeH:
        var = df
    out = mod.assign_human_region_membership(
        FakeH, {"precentral gyrus": (-35, -20, 55, 15.0)}
    )
    assert out["precentral gyrus"][0]


def test_evaluate_pair_perfect_match(mod):
    """If π puts all mass on the correct human region, top-1 should be 100%."""
    n_m, n_h = 4, 20
    pi = np.zeros((n_m, n_h))
    h_mask = np.zeros(n_h, dtype=bool)
    h_mask[5:8] = True  # 3 human-target parcels
    # Each mouse parcel argmaxes onto h_mask[5,6,7,5]
    for i, h in enumerate([5, 6, 7, 5]):
        pi[i, h] = 1.0
    m_mask = np.ones(n_m, dtype=bool)
    h_xyz = np.random.default_rng(0).standard_normal((n_h, 3))
    out = mod.evaluate_pair(pi, m_mask, h_mask, h_xyz, k_top=5)
    assert out["top1"] == 1.0
    assert out["top5"] == 1.0
    assert out["mean_rank_in_region"] == 1.0


def test_evaluate_pair_argmax_outside_region(mod):
    """If argmax is outside h_mask, top-1 = 0% and top-5 depends on ranking."""
    n_m, n_h = 2, 10
    pi = np.zeros((n_m, n_h))
    h_mask = np.zeros(n_h, dtype=bool)
    h_mask[5:7] = True
    # Argmax at index 0 (outside); h_mask gets second-largest mass
    pi[0, 0] = 1.0
    pi[0, 5] = 0.5
    pi[0, 6] = 0.5
    pi[1, 1] = 1.0
    pi[1, 5] = 0.5
    pi[1, 6] = 0.5
    m_mask = np.ones(n_m, dtype=bool)
    h_xyz = np.zeros((n_h, 3))
    out = mod.evaluate_pair(pi, m_mask, h_mask, h_xyz, k_top=5)
    assert out["top1"] == 0.0          # argmax NOT in h_mask
    assert out["top5"] == 1.0          # any of top-5 covers h_mask
    assert out["n_mouse_parcels"] == 2
    assert out["n_human_parcels"] == 2


def test_dsurqe_offset_constant_is_a_3vector(mod):
    """Sanity: the calibration constant has the right shape + reasonable bounds."""
    d = mod.DSURQE_OFFSET_MM
    assert d.shape == (3,)
    # Estimated from 6 anchor pairs (Visual/Motor/Auditory), the y-shift is the
    # dominant correction (~-2.3); x and z are both within ±2mm.
    assert abs(d[0]) < 2
    assert abs(d[1]) < 5
    assert abs(d[2]) < 2


def test_beauchamp_pairs_count_22_non_cerebellar(mod):
    """Sanity: we encode the 22 non-cerebellar pairs and skip 14 cerebellar."""
    assert len(mod.BEAUCHAMP_PAIRS) == 22
    mouse_names = [m for m, _ in mod.BEAUCHAMP_PAIRS]
    assert "Visual areas" in mouse_names         # cortical anchor pair
    assert "Field CA1" in mouse_names            # hippocampal novel pair
    # Cerebellar lobules excluded
    assert "Crus 1" not in mouse_names
    assert "Lingula (I)" not in mouse_names
