"""Tests for homer.data.io — parse_t_table is the unit testable bit.

Streaming + raw mat73 loading is integration-tested via the real h5ad cache;
those tests would need ~100 MB external data, so are skipped here.
"""
import numpy as np
import pandas as pd

from homer.data.io import parse_t_table


def _build_t_ht():
    """Build a minimal valid (t, ht) tuple for parse_t_table."""
    ht = ["type", "numid", "pairid", "region", "subregion", "center", "indices"]
    rows = [
        [1, 1, 1, "L_test_anchor_1", "sub", np.array([0.1, 0.2, 0.3]), np.array([0, 1, 2])],
        [1, 2, 1, "R_test_anchor_1", "sub", np.array([0.4, 0.5, 0.6]), np.array([3, 4])],
        [2, 3, 5, "L_grid_1",        "",    np.array([0.7, 0.8, 0.9]), np.array([5])],
    ]
    return rows, ht


def test_parse_t_table_basic_structure():
    t, ht = _build_t_ht()
    df = parse_t_table(t, ht)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns)[:3] == ["type", "numid", "pairid"]
    # Garin anchors flagged correctly
    assert df["garin_anchor"].tolist() == [True, True, False]
    # Hemisphere derived from region prefix
    assert df["hemisphere"].tolist() == ["L", "R", "L"]
    # anchor_pair_id: 1 for the two anchors, NA for the grid node
    pids = df["anchor_pair_id"].tolist()
    assert pids[0] == 1 and pids[1] == 1
    assert pd.isna(pids[2])


def test_parse_t_table_xyz_columns():
    t, ht = _build_t_ht()
    df = parse_t_table(t, ht)
    np.testing.assert_allclose(df["x"].values, [0.1, 0.4, 0.7])
    np.testing.assert_allclose(df["y"].values, [0.2, 0.5, 0.8])
    np.testing.assert_allclose(df["z"].values, [0.3, 0.6, 0.9])


def test_parse_t_table_index_is_node_id():
    t, ht = _build_t_ht()
    df = parse_t_table(t, ht)
    assert df.index.name == "node_id"
    assert df.index.tolist() == ["1", "2", "3"]


def test_parse_t_table_rejects_wrong_header():
    t, ht = _build_t_ht()
    bad_ht = ["type", "numid", "pairid", "region", "subregion"]    # missing center+indices
    import pytest
    with pytest.raises(ValueError, match="unrecognised ht schema"):
        parse_t_table(t, bad_ht)


def test_parse_t_table_voxel_indices_object_column():
    t, ht = _build_t_ht()
    df = parse_t_table(t, ht)
    # voxel_indices is an object column with arrays
    assert "voxel_indices" in df.columns
    np.testing.assert_array_equal(df["voxel_indices"].iloc[0], [0, 1, 2])
    np.testing.assert_array_equal(df["voxel_indices"].iloc[2], [5])
