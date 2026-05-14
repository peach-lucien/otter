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


# ---------------------------------------------------------------------------
# BICCN motor region anchors (integration test — needs real DSURQE atlas)


def _skip_if_no_atlas():
    """Integration tests need DSURQE atlas + AnnData cache. Skip cleanly on fresh checkouts."""
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    dsurqe = repo / "data_external/MouseHumanTranscriptomicSimilarity/AMBA/data/imaging/DSURQE_CCFv3_labels_200um.mnc"
    ann = repo / "outputs/anndata"
    if not dsurqe.exists() or not (ann / "mouse.h5ad").exists():
        pytest.skip("requires DSURQE atlas + AnnData cache (not in fresh checkouts)")
    return repo, ann


def test_build_biccn_motor_region_anchors_integration():
    """Integration: BICCN motor pack builds 2 valid entries."""
    repo, ann = _skip_if_no_atlas()
    from homer.data import load_cached
    from homer.data.anchor_packs import build_biccn_motor_region_anchors

    M, _ = load_cached("mouse", cache_dir=ann)
    H, _ = load_cached("human", cache_dir=ann)
    entries = build_biccn_motor_region_anchors(M.var, H.var, atlas_root=repo)

    assert len(entries) == 2
    m1, m2 = entries
    assert m1.pair_id == 30 and m2.pair_id == 31
    assert len(m1.mouse_indices) >= 30          # ≈ 53 parcels expected
    assert len(m2.mouse_indices) >= 30          # ≈ 48 parcels expected
    assert not (set(m1.mouse_indices) & set(m2.mouse_indices))
    assert len(m1.human_indices) >= 5           # ≈ 12 parcels expected for BA4
    assert len(m2.human_indices) >= 10          # ≈ 23 parcels expected for PMd


def test_build_tectum_region_anchors_integration():
    """Integration: Tectum pack builds 2 valid entries with disjoint pids 32/33."""
    repo, ann = _skip_if_no_atlas()
    from homer.data import load_cached
    from homer.data.anchor_packs import build_tectum_region_anchors

    M, _ = load_cached("mouse", cache_dir=ann)
    H, _ = load_cached("human", cache_dir=ann)
    entries = build_tectum_region_anchors(M.var, H.var, atlas_root=repo)

    assert len(entries) == 2
    sc, ic = entries
    assert sc.pair_id == 32 and ic.pair_id == 33
    assert "Superior" in sc.label and "Inferior" in ic.label
    # Sets non-trivial
    assert len(sc.mouse_indices) >= 20          # ≈ 53 parcels expected
    assert len(ic.mouse_indices) >= 10          # ≈ 29 parcels expected
    assert len(sc.human_indices) >= 1           # tight ball, just SC parcels
    assert len(ic.human_indices) >= 2           # tight ball, just IC parcels


def test_build_olfactory_region_anchors_integration():
    """Integration: Olfactory pack builds 2 valid entries with pids 34/35."""
    repo, ann = _skip_if_no_atlas()
    from homer.data import load_cached
    from homer.data.anchor_packs import build_olfactory_region_anchors

    M, _ = load_cached("mouse", cache_dir=ann)
    H, _ = load_cached("human", cache_dir=ann)
    entries = build_olfactory_region_anchors(M.var, H.var, atlas_root=repo)

    assert len(entries) == 2
    pir, aon = entries
    assert pir.pair_id == 34 and aon.pair_id == 35
    assert "Piriform" in pir.label and ("AON" in aon.label or "olfactory" in aon.label.lower())
    assert len(pir.mouse_indices) >= 20         # ≈ 47 parcels expected
    assert len(pir.human_indices) >= 5          # ≈ 13 parcels expected
    assert len(aon.mouse_indices) >= 3          # ≈ 9 parcels expected
    assert len(aon.human_indices) >= 1          # ≈ 6 parcels expected
    # Disjoint mouse-side sets
    assert not (set(pir.mouse_indices) & set(aon.mouse_indices))


def test_build_cingulate_region_anchors_integration():
    """Integration: Cingulate pack builds 2 valid entries with pids 36/37."""
    repo, ann = _skip_if_no_atlas()
    from homer.data import load_cached
    from homer.data.anchor_packs import build_cingulate_region_anchors

    M, _ = load_cached("mouse", cache_dir=ann)
    H, _ = load_cached("human", cache_dir=ann)
    entries = build_cingulate_region_anchors(M.var, H.var, atlas_root=repo)

    assert len(entries) == 2
    acc, rsc = entries
    assert acc.pair_id == 36 and rsc.pair_id == 37
    assert "ACC" in acc.label and "Retrosplenial" in rsc.label
    assert len(acc.mouse_indices) >= 5
    assert len(rsc.mouse_indices) >= 10
    assert not (set(acc.mouse_indices) & set(rsc.mouse_indices))


def test_build_amygdala_region_anchors_integration():
    """Integration: Amygdala pack builds 1 valid entry with pid 38."""
    repo, ann = _skip_if_no_atlas()
    from homer.data import load_cached
    from homer.data.anchor_packs import build_amygdala_region_anchors

    M, _ = load_cached("mouse", cache_dir=ann)
    H, _ = load_cached("human", cache_dir=ann)
    entries = build_amygdala_region_anchors(M.var, H.var, atlas_root=repo)

    assert len(entries) == 1
    e = entries[0]
    assert e.pair_id == 38
    assert "Amygdala" in e.label or "Cortical subplate" in e.label
    assert len(e.mouse_indices) >= 30          # ≈ 54 parcels expected
    assert len(e.human_indices) >= 2           # ≈ 6 parcels expected


def test_build_hippocampal_region_anchors_integration():
    """Integration: Hippocampal pack builds 4 valid entries with pids 39-42."""
    repo, ann = _skip_if_no_atlas()
    from homer.data import load_cached
    from homer.data.anchor_packs import build_hippocampal_region_anchors

    M, _ = load_cached("mouse", cache_dir=ann)
    H, _ = load_cached("human", cache_dir=ann)
    entries = build_hippocampal_region_anchors(M.var, H.var, atlas_root=repo)

    assert len(entries) == 4
    pids = [e.pair_id for e in entries]
    assert pids == [39, 40, 41, 42]
    labels = [e.label for e in entries]
    assert "Subiculum" in labels[0]
    assert "CA1" in labels[1]
    assert "CA3" in labels[2]
    assert "Dentate" in labels[3]
    # All sets non-trivial and pairwise mouse-disjoint
    seen = set()
    for e in entries:
        assert len(e.mouse_indices) >= 5
        assert len(e.human_indices) >= 2
        assert not (set(e.mouse_indices) & seen)
        seen.update(e.mouse_indices)


def test_build_striatum_region_anchors_integration():
    """Integration: Striatum pack builds 2 entries at pids 47, 48."""
    repo, ann = _skip_if_no_atlas()
    from homer.data import load_cached
    from homer.data.anchor_packs import build_striatum_region_anchors

    M, _ = load_cached("mouse", cache_dir=ann)
    H, _ = load_cached("human", cache_dir=ann)
    entries = build_striatum_region_anchors(M.var, H.var, atlas_root=repo)
    assert len(entries) == 2
    dl, vm = entries
    assert dl.pair_id == 47 and vm.pair_id == 48
    assert "dorsolateral" in dl.label.lower() or "Putamen" in dl.label
    assert "ventromedial" in vm.label.lower() or "Caudate" in vm.label
    # Dorsolateral and ventromedial subsets are disjoint by construction
    assert not (set(dl.mouse_indices) & set(vm.mouse_indices))
    # Reasonable sizes
    assert 5 <= len(dl.mouse_indices) <= 50
    assert 10 <= len(vm.mouse_indices) <= 80


def test_build_entorhinal_region_anchors_integration():
    """Integration: Entorhinal pack builds 1 entry at pid 49."""
    repo, ann = _skip_if_no_atlas()
    from homer.data import load_cached
    from homer.data.anchor_packs import build_entorhinal_region_anchors

    M, _ = load_cached("mouse", cache_dir=ann)
    H, _ = load_cached("human", cache_dir=ann)
    entries = build_entorhinal_region_anchors(M.var, H.var, atlas_root=repo)
    assert len(entries) == 1
    e = entries[0]
    assert e.pair_id == 49
    assert "Entorhinal" in e.label
    assert len(e.mouse_indices) >= 50          # ≈ 84
    assert len(e.human_indices) >= 2           # ≈ 6


def test_build_visual_region_anchors_integration():
    """Integration: Visual pack builds 1 entry at pid 52."""
    repo, ann = _skip_if_no_atlas()
    from homer.data import load_cached
    from homer.data.anchor_packs import build_visual_region_anchors

    M, _ = load_cached("mouse", cache_dir=ann)
    H, _ = load_cached("human", cache_dir=ann)
    entries = build_visual_region_anchors(M.var, H.var, atlas_root=repo)
    assert len(entries) == 1
    e = entries[0]
    assert e.pair_id == 52
    assert "Lateral visual" in e.label or "V2" in e.label
    assert 5 <= len(e.mouse_indices) <= 25       # ≈ 9
    assert len(e.human_indices) >= 2


def test_build_pag_region_anchors_integration():
    """Integration: PAG pack builds 1 entry at pid 54."""
    repo, ann = _skip_if_no_atlas()
    from homer.data import load_cached
    from homer.data.anchor_packs import build_pag_region_anchors

    M, _ = load_cached("mouse", cache_dir=ann)
    H, _ = load_cached("human", cache_dir=ann)
    entries = build_pag_region_anchors(M.var, H.var, atlas_root=repo)
    assert len(entries) == 1
    e = entries[0]
    assert e.pair_id == 54
    assert "PAG" in e.label or "Periaqueductal" in e.label
    assert 8 <= len(e.mouse_indices) <= 30       # ≈ 16
    assert len(e.human_indices) >= 2


def test_mouse_parcels_in_mouse_sphere_smoke():
    """Spatial selection helper finds at least the expected count of nearby parcels."""
    repo, ann = _skip_if_no_atlas()
    from homer.data import load_cached
    from homer.data.anchor_packs._dsurqe import mouse_parcels_in_mouse_sphere
    M, _ = load_cached("mouse", cache_dir=ann)
    # PAG centroid in M_var ≈ (0.03, -1.81, +1.20); ball r=0.5 should capture parcels
    near_pag = mouse_parcels_in_mouse_sphere(M.var, (0.0, -1.81, 1.20), 0.5)
    assert len(near_pag) > 0       # some parcels near PAG centroid
    # Empty radius
    empty = mouse_parcels_in_mouse_sphere(M.var, (100, 100, 100), 1.0)
    assert empty == []


def test_build_perirhinal_region_anchors_integration():
    """Integration: Perirhinal pack builds 1 entry at pid 55."""
    repo, ann = _skip_if_no_atlas()
    from homer.data import load_cached
    from homer.data.anchor_packs import build_perirhinal_region_anchors
    M, _ = load_cached("mouse", cache_dir=ann)
    H, _ = load_cached("human", cache_dir=ann)
    entries = build_perirhinal_region_anchors(M.var, H.var, atlas_root=repo)
    assert len(entries) == 1
    e = entries[0]
    assert e.pair_id == 55
    assert "Perirhinal" in e.label
    assert 3 <= len(e.mouse_indices) <= 15      # ≈ 6
    assert len(e.human_indices) >= 2


def test_build_auditory_region_anchors_integration():
    """Integration: Auditory pack builds 2 entries at pids 56, 57."""
    repo, ann = _skip_if_no_atlas()
    from homer.data import load_cached
    from homer.data.anchor_packs import build_auditory_region_anchors
    M, _ = load_cached("mouse", cache_dir=ann)
    H, _ = load_cached("human", cache_dir=ann)
    entries = build_auditory_region_anchors(M.var, H.var, atlas_root=repo)
    assert len(entries) == 2
    a1, belt = entries
    assert a1.pair_id == 56 and belt.pair_id == 57
    assert "A1" in a1.label or "Primary" in a1.label
    assert "belt" in belt.label.lower() or "A2" in belt.label
    assert 5 <= len(a1.mouse_indices) <= 20      # ≈ 9
    assert 5 <= len(belt.mouse_indices) <= 20    # ≈ 11
    assert not (set(a1.mouse_indices) & set(belt.mouse_indices))   # disjoint


def test_build_somatosensory_region_anchors_integration():
    """Integration: Somatosensory pack builds 3 entries at pids 58, 59, 60."""
    repo, ann = _skip_if_no_atlas()
    from homer.data import load_cached
    from homer.data.anchor_packs import build_somatosensory_region_anchors
    M, _ = load_cached("mouse", cache_dir=ann)
    H, _ = load_cached("human", cache_dir=ann)
    entries = build_somatosensory_region_anchors(M.var, H.var, atlas_root=repo)
    assert len(entries) == 3
    face, hand, leg = entries
    assert face.pair_id == 58 and hand.pair_id == 59 and leg.pair_id == 60
    assert "face" in face.label.lower() and "hand" in hand.label.lower() and "leg" in leg.label.lower()
    # Disjoint mouse-side sets
    assert not (set(face.mouse_indices) & set(hand.mouse_indices))
    assert not (set(hand.mouse_indices) & set(leg.mouse_indices))
    assert not (set(face.mouse_indices) & set(leg.mouse_indices))
    # Reasonable sizes
    assert len(face.mouse_indices) >= 50      # face = barrel + nose ≈ 88
    assert 10 <= len(hand.mouse_indices) <= 40    # ≈ 24
    assert 5 <= len(leg.mouse_indices) <= 25      # ≈ 14


def test_build_ppc_region_anchors_integration():
    """Integration: PPC pack builds 1 entry at pid 61."""
    repo, ann = _skip_if_no_atlas()
    from homer.data import load_cached
    from homer.data.anchor_packs import build_ppc_region_anchors
    M, _ = load_cached("mouse", cache_dir=ann)
    H, _ = load_cached("human", cache_dir=ann)
    entries = build_ppc_region_anchors(M.var, H.var, atlas_root=repo)
    assert len(entries) == 1
    e = entries[0]
    assert e.pair_id == 61
    assert "PPC" in e.label or "Posterior parietal" in e.label or "BA7" in e.label
    assert 5 <= len(e.mouse_indices) <= 20      # ≈ 10
    assert len(e.human_indices) >= 5


def test_compose_packs_disjoint_pids():
    """Combining all 14 packs yields 25 entries with unique pids."""
    repo, ann = _skip_if_no_atlas()
    from homer.data import load_cached
    from homer.data.anchor_packs import (
        build_biccn_motor_region_anchors,
        build_tectum_region_anchors,
        build_olfactory_region_anchors,
        build_cingulate_region_anchors,
        build_amygdala_region_anchors,
        build_hippocampal_region_anchors,
        build_striatum_region_anchors,
        build_entorhinal_region_anchors,
        build_visual_region_anchors,
        build_pag_region_anchors,
        build_perirhinal_region_anchors,
        build_auditory_region_anchors,
        build_somatosensory_region_anchors,
        build_ppc_region_anchors,
    )
    M, _ = load_cached("mouse", cache_dir=ann)
    H, _ = load_cached("human", cache_dir=ann)
    combined = (build_biccn_motor_region_anchors(M.var, H.var, atlas_root=repo)
                + build_tectum_region_anchors(M.var, H.var, atlas_root=repo)
                + build_olfactory_region_anchors(M.var, H.var, atlas_root=repo)
                + build_cingulate_region_anchors(M.var, H.var, atlas_root=repo)
                + build_amygdala_region_anchors(M.var, H.var, atlas_root=repo)
                + build_hippocampal_region_anchors(M.var, H.var, atlas_root=repo)
                + build_striatum_region_anchors(M.var, H.var, atlas_root=repo)
                + build_entorhinal_region_anchors(M.var, H.var, atlas_root=repo)
                + build_visual_region_anchors(M.var, H.var, atlas_root=repo)
                + build_pag_region_anchors(M.var, H.var, atlas_root=repo)
                + build_perirhinal_region_anchors(M.var, H.var, atlas_root=repo)
                + build_auditory_region_anchors(M.var, H.var, atlas_root=repo)
                + build_somatosensory_region_anchors(M.var, H.var, atlas_root=repo)
                + build_ppc_region_anchors(M.var, H.var, atlas_root=repo))
    pids = [e.pair_id for e in combined]
    expected = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                47, 48, 49, 52, 54, 55, 56, 57, 58, 59, 60, 61]
    assert pids == expected
    assert len(set(pids)) == 25               # no clashes
