"""Tests for homer.data.anchors and homer.data.networks."""
import numpy as np
import pytest

from homer.data.anchors import (
    AnchorIndex, get_anchor_index, held_out_metrics_graded,
    metrics_summary, true_assignment, kfold_pair_ids,
)
from homer.data.networks import (
    NETWORKS, PAIRID_TO_NETWORK, assign_networks, network_mismatch_mask,
)


def test_get_anchor_index_returns_correct_shape(mouse_ad):
    idx = get_anchor_index(mouse_ad.var)
    assert isinstance(idx, AnchorIndex)
    assert len(idx) == 10                         # 5 pair_ids × 2 hemis
    assert idx.pos.shape == (10,)
    assert idx.pair_ids.shape == (10,)
    assert idx.hemispheres.shape == (10,)
    assert sorted(set(idx.pair_ids.tolist())) == [1, 2, 3, 4, 5]


def test_anchor_index_sorted_consistently(mouse_ad, human_ad):
    """Both species sorted by (pair_id, hemi) → same key ordering."""
    idx_m = get_anchor_index(mouse_ad.var)
    idx_h = get_anchor_index(human_ad.var)
    assert idx_m.keys == idx_h.keys
    # true_assignment should be the identity
    np.testing.assert_array_equal(true_assignment(idx_m, idx_h), np.arange(10))


def test_metrics_summary_identity_pi_is_perfect(mouse_ad, human_ad):
    """If pi is the identity over the 10×10 anchor sub-block, top1 = 100%."""
    idx_m = get_anchor_index(mouse_ad.var)
    idx_h = get_anchor_index(human_ad.var)
    pi = np.eye(10, 10) / 10.0
    m = metrics_summary(pi, idx_m, idx_h)
    assert m["top1"] == 1.0
    assert m["pair_id"] == 1.0
    assert m["hemisphere"] == 1.0


def test_metrics_summary_random_pi_low(mouse_ad, human_ad):
    rng = np.random.default_rng(0)
    idx_m = get_anchor_index(mouse_ad.var)
    idx_h = get_anchor_index(human_ad.var)
    pi = rng.uniform(0, 1, size=(10, 10))
    m = metrics_summary(pi, idx_m, idx_h)
    # Random pi should be below perfect; just check it runs and is in [0, 1]
    assert 0.0 <= m["top1"] <= 1.0


def test_held_out_metrics_graded_keys(mouse_ad, human_ad):
    """The graded helper returns all expected keys."""
    idx_m = get_anchor_index(mouse_ad.var)
    idx_h = get_anchor_index(human_ad.var)
    pi = np.eye(20, 25) / 20.0     # full π shape
    m = held_out_metrics_graded(pi, idx_m, idx_h, held_out_pair_ids=[1, 2],
                                  var_h=human_ad.var)
    expected_keys = {"top1", "top5", "pair_id", "hemisphere",
                      "mean_rank", "median_rank", "mean_xyz_dist",
                      "median_xyz_dist", "n", "max_rank_possible"}
    assert expected_keys.issubset(m.keys())
    assert m["n"] == 4    # 2 pair_ids × 2 hemis held out


def test_assign_networks_returns_int_array(mouse_ad):
    idx = get_anchor_index(mouse_ad.var)
    nets = assign_networks(mouse_ad.var, idx)
    assert nets.dtype == np.int32
    assert nets.shape == (20,)
    # All indices should be valid network ids
    assert (nets >= 0).all()
    assert (nets < len(NETWORKS)).all()


def test_network_mismatch_mask_is_binary():
    net_m = np.array([0, 0, 1, 2])
    net_h = np.array([0, 1, 1, 2, 3])
    mask = network_mismatch_mask(net_m, net_h)
    assert mask.shape == (4, 5)
    assert mask.dtype == bool
    # Element [0, 0]: same network → False
    assert mask[0, 0] == False
    # Element [0, 1]: different network → True
    assert mask[0, 1] == True


def test_kfold_pair_ids_partition():
    """k-fold returns a partition: every pair_id is held out exactly once."""
    pair_ids = list(range(1, 22))    # the real 21 anchor pair_ids
    folds = kfold_pair_ids(pair_ids, n_splits=5, seed=42)
    assert len(folds) == 5
    held_total = []
    for visible, held in folds:
        assert sorted(visible + held) == pair_ids
        held_total += held
    assert sorted(held_total) == pair_ids


def test_pairid_to_network_covers_all_anchors():
    """Every Garin pair_id 1..21 has a network label."""
    assert set(PAIRID_TO_NETWORK.keys()) == set(range(1, 22))
    assert all(v in NETWORKS for v in PAIRID_TO_NETWORK.values())
