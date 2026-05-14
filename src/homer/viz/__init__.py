"""Visualisation: HTML viewer + inline notebook plotters + comparison reports.

Three modules:

  ``homer.viz.viewer``   — the standalone HTML viewer (mouse + human 3D
                            scatters with click-to-highlight cross-species
                            partners). Produces a self-contained .html file.

  ``homer.viz.notebook`` — Plotly figures intended for inline display in
                            Jupyter notebooks (3D scatter, partner highlight,
                            per-network heatmaps, multi-metric bars).

  ``homer.viz.reports``  — comparison-table generator that pulls all the
                            ``outputs/logs/*.json`` results and produces a
                            wide CSV / long CSV / markdown summary +
                            comparison figures.

  ``homer.viz.gui``      — region-first static GUI builder for
                            ``outputs/gui/index.html``.

Public:

    from homer.viz import (
        # viewer
        build_viewer_html, build_viewer_data, write_viewer,
        # GUI
        build_gui_payload, build_gui_html, build_visual_layers, write_gui,
        # notebook plotters
        plot_brain_3d, plot_pi_partners,
        plot_per_network_heatmap, plot_comparison_bars,
        # reports
        build_comparison_table, render_summary_md,
    )
"""
from homer.viz.viewer import (
    build_viewer_data,
    build_viewer_html,
    col_entropy,
    row_entropy,
    topk_per_col,
    topk_per_row,
    write_viewer,
)
from homer.viz.gui import (
    build_gui_html,
    build_gui_payload,
    build_visual_layers,
    write_gui,
)
from homer.viz.notebook import (
    plot_brain_3d,
    plot_comparison_bars,
    plot_per_network_heatmap,
    plot_pi_partners,
    plot_pi_partners_pair,
    plot_pi_heatmap,
    plot_pi_heatmap_ordered,
    plot_region_translation,
    region_translation_table,
)
from homer.viz.reports import (
    aggregate_anchor_cv,
    aggregate_full_space,
    aggregate_null,
    build_comparison_table,
    build_per_network_long,
    render_summary_md,
)

__all__ = [
    # viewer
    "build_viewer_data", "build_viewer_html", "write_viewer",
    "topk_per_row", "topk_per_col", "row_entropy", "col_entropy",
    # GUI
    "build_gui_payload", "build_gui_html", "build_visual_layers", "write_gui",
    # notebook plotters
    "plot_brain_3d", "plot_pi_partners", "plot_pi_partners_pair",
    "plot_per_network_heatmap", "plot_comparison_bars",
    "plot_pi_heatmap", "plot_pi_heatmap_ordered",
    "plot_region_translation", "region_translation_table",
    # reports
    "aggregate_anchor_cv", "aggregate_null",
    "build_comparison_table", "build_per_network_long",
    "render_summary_md",
]
