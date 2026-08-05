"""Tests for otter.viz, viewer + notebook plotters + reports."""
import json

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Viewer (HTML + payload generation)
# ---------------------------------------------------------------------------
def test_topk_per_row_shape_and_sort():
    from otter.viz.viewer import topk_per_row
    rng = np.random.default_rng(0)
    pi = rng.uniform(0, 1, size=(8, 12))
    out = topk_per_row(pi, k=4)
    assert len(out) == 8
    assert all(len(row) == 4 for row in out)
    # Each row sorted descending by value
    for row in out:
        vals = [v for (_idx, v) in row]
        assert vals == sorted(vals, reverse=True)


def test_topk_per_col_normalises_by_col_sum():
    from otter.viz.viewer import topk_per_col
    pi = np.array([[1.0, 2.0], [3.0, 4.0]])     # col sums = 4, 6
    out = topk_per_col(pi, k=1)
    # Top entry of col 0 is row 1 (val 3) → normalised 3/4 = 0.75
    assert out[0][0][0] == 1
    assert abs(out[0][0][1] - 0.75) < 1e-3


def test_row_entropy_one_hot_is_zero():
    from otter.viz.viewer import row_entropy
    pi = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    e = row_entropy(pi)
    np.testing.assert_allclose(e, [0.0, 0.0], atol=1e-9)


def test_row_entropy_uniform_is_log_n():
    from otter.viz.viewer import row_entropy
    pi = np.array([[1.0, 1.0, 1.0, 1.0]])
    e = row_entropy(pi)
    np.testing.assert_allclose(e, [np.log(4)], atol=1e-9)


def test_build_viewer_data_shape(mouse_ad, human_ad):
    from otter.viz.viewer import build_viewer_data
    rng = np.random.default_rng(0)
    pi = rng.uniform(0, 1, size=(20, 25))
    pi /= pi.sum(axis=1, keepdims=True) * 20      # normalise to mouse uniform marginal
    payload = build_viewer_data(pi, mouse_ad, human_ad, top_k=5,
                                 pi_label="test")
    assert payload["version"] == 1
    assert payload["n_mouse_nodes"] == 20
    assert payload["n_human_nodes"] == 25
    assert payload["top_k"] == 5
    assert "mouse" in payload and "human" in payload
    assert len(payload["mouse"]["x"]) == 20
    assert len(payload["human"]["x"]) == 25
    # top_partners exists with the right depth
    assert len(payload["mouse"]["top_partners"]) == 20
    assert len(payload["mouse"]["top_partners"][0]) == 5
    # entropies and col_mass present
    assert "entropy" in payload["mouse"]
    assert "entropy" in payload["human"]
    assert "col_mass" in payload["human"]
    assert len(payload["human"]["col_mass"]) == 25


def test_build_viewer_html_contains_data(mouse_ad, human_ad):
    from otter.viz.viewer import build_viewer_data, build_viewer_html
    rng = np.random.default_rng(0)
    pi = rng.uniform(0, 1, size=(20, 25))
    payload = build_viewer_data(pi, mouse_ad, human_ad, top_k=5, pi_label="test_label")
    html = build_viewer_html(payload)
    assert "<!DOCTYPE html>" in html
    assert "plot-mouse" in html and "plot-human" in html
    # Embedded payload should be present
    assert "test_label" in html
    # JS isn't double-escaped, should NOT see e.g. "{{" in JS bodies
    assert "{{ EMBEDDED_DATA }}" not in html


def test_write_viewer_creates_files(mouse_ad, human_ad, tmp_path):
    from otter.viz.viewer import write_viewer
    rng = np.random.default_rng(0)
    pi = rng.uniform(0, 1, size=(20, 25))
    json_path, html_path = write_viewer(
        pi, mouse_ad=mouse_ad, human_ad=human_ad,
        output_dir=tmp_path, top_k=5, pi_label="test",
    )
    assert json_path.exists()
    assert html_path.exists()
    # Round-trip the JSON
    payload = json.loads(json_path.read_text())
    assert payload["pi_label"] == "test"


def test_build_viewer_data_rejects_1d_pi(mouse_ad, human_ad):
    from otter.viz.viewer import build_viewer_data
    with pytest.raises(ValueError, match="must be 2-D"):
        build_viewer_data(np.zeros(20), mouse_ad, human_ad)


# ---------------------------------------------------------------------------
# Region-first GUI
# ---------------------------------------------------------------------------
def test_build_gui_payload_smoke(mouse_ad, human_ad):
    from otter.viz.gui import build_gui_payload
    rng = np.random.default_rng(0)
    pi = rng.uniform(0, 1, size=(20, 25))
    payload = build_gui_payload(
        [{"id": "toy", "label": "Toy model", "pi": pi}],
        mouse_ad,
        human_ad,
        top_k=4,
    )
    assert payload["version"] == 1
    assert payload["models"][0]["id"] == "toy"
    assert len(payload["models"][0]["top_mouse_to_human"]) == 20
    assert len(payload["models"][0]["top_mouse_to_human"][0]) == 4
    assert payload["mouse"]["ids"][0] == "1"
    assert "groups" in payload and "mouse" in payload["groups"]


def test_build_gui_html_contains_app(mouse_ad, human_ad):
    from otter.viz.gui import build_gui_html, build_gui_payload
    pi = np.eye(20, 25)
    payload = build_gui_payload(
        [{"id": "identity", "label": "Identity-ish", "pi": pi}],
        mouse_ad,
        human_ad,
        top_k=3,
    )
    html = build_gui_html(payload)
    assert "<!DOCTYPE html>" in html
    assert "OTTER Mapping Explorer" in html
    assert "plotMouse" in html and "plotHuman" in html
    assert "Identity-ish" in html


def test_write_gui_creates_files(mouse_ad, human_ad, tmp_path):
    from otter.viz.gui import build_gui_payload, write_gui
    pi = np.eye(20, 25)
    payload = build_gui_payload(
        [{"id": "identity", "label": "Identity-ish", "pi": pi}],
        mouse_ad,
        human_ad,
        top_k=3,
    )
    data_path, html_path = write_gui(payload, output_dir=tmp_path)
    assert data_path.exists()
    assert html_path.exists()
    data = json.loads(data_path.read_text())
    assert data["models"][0]["id"] == "identity"


# ---------------------------------------------------------------------------
# Notebook plotters, verify they return Plotly Figure objects
# ---------------------------------------------------------------------------
@pytest.fixture
def go():
    """Plotly graph_objects, skip viz tests if plotly isn't installed."""
    pytest.importorskip("plotly")
    import plotly.graph_objects as go
    return go


def test_plot_brain_3d_default(mouse_ad, go):
    from otter.viz.notebook import plot_brain_3d
    fig = plot_brain_3d(mouse_ad)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    # Trace should have 20 points
    assert len(fig.data[0].x) == 20


def test_plot_brain_3d_color_modes(mouse_ad, go):
    from otter.viz.notebook import plot_brain_3d
    for mode in ("network", "hemisphere", "is_anchor"):
        fig = plot_brain_3d(mouse_ad, color_by=mode)
        assert isinstance(fig, go.Figure)


def test_plot_brain_3d_highlight(human_ad, go):
    from otter.viz.notebook import plot_brain_3d
    fig = plot_brain_3d(human_ad, color_by="highlight",
                          highlight_idx=[0, 5, 10],
                          highlight_values=[1.0, 0.5, 0.1])
    assert isinstance(fig, go.Figure)


def test_plot_brain_3d_unknown_mode_raises(mouse_ad, go):
    from otter.viz.notebook import plot_brain_3d
    with pytest.raises(ValueError, match="unknown color_by"):
        plot_brain_3d(mouse_ad, color_by="bogus")


def test_plot_pi_partners_with_fitted_model(mouse_ad, human_ad, go):
    from otter.models import SupervisedFGW
    from otter.viz.notebook import plot_pi_partners
    m = SupervisedFGW(epsilon=1e-2).fit(mouse_ad, human_ad)
    fig = plot_pi_partners(m, source_idx=0, source="mouse", top_k=5)
    assert isinstance(fig, go.Figure)


def test_plot_pi_partners_unfit_raises(go):
    from otter.models import SupervisedFGW
    from otter.viz.notebook import plot_pi_partners
    with pytest.raises(RuntimeError, match="must be fitted"):
        plot_pi_partners(SupervisedFGW(), source_idx=0)


def test_plot_pi_heatmap_small(go):
    from otter.viz.notebook import plot_pi_heatmap
    pi = np.eye(10)
    fig = plot_pi_heatmap(pi)
    assert isinstance(fig, go.Figure)
    # Small enough that no downsampling, z is 10×10
    assert fig.data[0].z.shape == (10, 10)


def test_plot_pi_heatmap_large_downsamples(go):
    from otter.viz.notebook import plot_pi_heatmap
    pi = np.random.rand(900, 1000)
    fig = plot_pi_heatmap(pi, max_size=100)
    assert isinstance(fig, go.Figure)
    # Downsampled: each axis ≤ ~max_size
    assert max(fig.data[0].z.shape) <= 200


def test_plot_per_network_heatmap_shape(go):
    from otter.viz.notebook import plot_per_network_heatmap
    df = pd.DataFrame([
        {"config": "a", "label": "A", "network": "visual",   "top1": 0.5},
        {"config": "a", "label": "A", "network": "auditory", "top1": 1.0},
        {"config": "b", "label": "B", "network": "visual",   "top1": 0.25},
        {"config": "b", "label": "B", "network": "auditory", "top1": 1.0},
    ])
    fig = plot_per_network_heatmap(df)
    assert isinstance(fig, go.Figure)


def test_plot_comparison_bars_shape(go):
    from otter.viz.notebook import plot_comparison_bars
    df = pd.DataFrame([
        {"config": "a", "label": "Config A", "notes": "",
         "anchor_top1": 0.79, "anchor_top5": 1.0,
         "fc_translation_r": 0.36, "anchor_mean_xyz_dist": 0.021},
        {"config": "fc_plus_SC", "label": "Production", "notes": "production",
         "anchor_top1": 0.81, "anchor_top5": 1.0,
         "fc_translation_r": 0.36, "anchor_mean_xyz_dist": 0.020},
    ])
    fig = plot_comparison_bars(df)
    assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def test_aggregate_anchor_cv_weighted_mean():
    from otter.viz.reports import aggregate_anchor_cv
    per_net = {
        "visual":   {"n_anchors_held": 4,  "top1": 0.5,  "top5": 1.0},
        "auditory": {"n_anchors_held": 2,  "top1": 1.0,  "top5": 1.0},
    }
    out = aggregate_anchor_cv(per_net)
    # weighted top1 = (0.5*4 + 1.0*2) / 6 = 4/6 ≈ 0.667
    assert abs(out["weighted_top1"] - 4/6) < 1e-9
    assert out["weighted_top5"] == 1.0
    assert out["n_networks"] == 2
    assert out["n_anchors_total"] == 6


def test_aggregate_anchor_cv_empty():
    from otter.viz.reports import aggregate_anchor_cv
    assert aggregate_anchor_cv({}) == {}
    # Missing n_anchors_held → also empty
    assert aggregate_anchor_cv({"visual": {"top1": 0.5}}) == {}


def test_aggregate_null_zero_when_no_trials():
    from otter.viz.reports import aggregate_null
    out = aggregate_null({}, weights_per_net={})
    assert out == {}


def test_aggregate_null_basic():
    from otter.viz.reports import aggregate_null
    null_per_net = {
        "visual":   [{"top1": 0.0}, {"top1": 0.5}],
        "auditory": [{"top1": 1.0}, {"top1": 0.5}],
    }
    weights = {"visual": 4.0, "auditory": 2.0}     # vis has 2x weight
    out = aggregate_null(null_per_net, weights_per_net=weights, key="top1")
    assert out["null_top1_n_trials"] == 2
    # Trial 0 weighted = (0.0*4 + 1.0*2)/6 = 1/3
    # Trial 1 weighted = (0.5*4 + 0.5*2)/6 = 0.5
    # mean = (1/3 + 0.5)/2 = 5/12 ≈ 0.417
    assert abs(out["null_top1_mean"] - 5/12) < 1e-3
