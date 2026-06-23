"""Inline Plotly helpers for Jupyter notebooks.

Each function returns a ``plotly.graph_objects.Figure`` so the user can
``.show()`` it inline, customise it, or save it.

Public:
    plot_brain_3d(ad, *, color_by="network", title=None) -> Figure
    plot_pi_partners(model, source_idx, *, source="mouse", top_k=20) -> Figure
    plot_pi_heatmap(pi, *, max_size=400, title=None) -> Figure
    plot_per_network_heatmap(per_network_df, *, value_col="top1") -> Figure
    plot_comparison_bars(table_df, *, metrics=("anchor_top1", ...)) -> Figure
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from homer.data.anchors import get_anchor_index
from homer.data.networks import NETWORKS, assign_networks


# Network colour palette, distinguishable on light + dark backgrounds.
NET_COLORS: dict[str, str] = {
    "auditory":        "#FF6B6B", "brainstem":    "#A78BFA", "frontal_dmn":    "#38BDF8",
    "frontoparietal":  "#34D399", "limbic":       "#FBBF24", "olfactory":      "#FACC15",
    "salience":        "#F472B6", "sensorimotor": "#67E8F9", "subcortical":    "#94A3B8",
    "temporal_dmn":    "#10B981", "visual":       "#FB923C",
}


def _import_plotly():
    """Lazy-import plotly so the rest of homer doesn't require it for non-viz use."""
    try:
        import plotly.graph_objects as go
    except ImportError as e:                                                # pragma: no cover
        raise ImportError(
            "homer.viz.notebook requires plotly; install with `pip install plotly`."
        ) from e
    return go


# ---------------------------------------------------------------------------
# 3D scatter for one species' brain
# ---------------------------------------------------------------------------
TRUST_COLORS = {
    "high":    "#10B981",   # emerald, top-1 ≥15% on Beauchamp validation
    "medium":  "#F59E0B",   # amber, 3-15%
    "low":     "#EF4444",   # red, <3%
    "unknown": "#6B7280",   # grey, outside the validated regions
}


def plot_brain_3d(ad, *,
                   color_by: str = "network",
                   highlight_idx: Optional[Sequence[int]] = None,
                   highlight_values: Optional[Sequence[float]] = None,
                   trust_tier: Optional[Sequence[str]] = None,
                   trust_score: Optional[Sequence[float]] = None,
                   title: Optional[str] = None,
                   width: int = 700,
                   height: int = 600,
                   dark: bool = True):
    """Render one species' parcels as a 3D scatter.

    Parameters
    ----------
    ad : AnnData
        The species' AnnData with x/y/z + region columns in ``var``.
    color_by : str
        ``"network"``, colour by functional-network assignment (default)
        ``"hemisphere"``. L vs R
        ``"is_anchor"``, anchors red, grid grey
        ``"highlight"``, colour the ``highlight_idx`` nodes by ``highlight_values``
                          on a viridis scale (everything else dim grey).
        ``"trust_tier"``, colour by tier (high/medium/low/unknown) from a
                          ``trust_score_*.npz`` file. Pass ``trust_tier=``
                          (an array of strings).
        ``"trust_score"``, colour by continuous trust score (red→amber→green
                          gradient, NaN → grey). Pass ``trust_score=``.
    highlight_idx, highlight_values
        Used only when color_by="highlight". Same length; values are mapped to
        viridis intensities.
    trust_tier, trust_score
        Used only with the matching color_by; arrays of length n parcels.
    """
    go = _import_plotly()
    var = ad.var
    n = len(var)
    x = var["x"].values; y = var["y"].values; z = var["z"].values
    is_anchor = var["garin_anchor"].values

    if color_by == "trust_tier":
        if trust_tier is None:
            raise ValueError("color_by='trust_tier' requires trust_tier=")
        trust_tier_arr = np.asarray(trust_tier)
        colors = [TRUST_COLORS.get(str(t), "#6B7280") for t in trust_tier_arr]
        sizes = np.where(is_anchor, 9, 4)
        text = [f"{r}, tier: {trust_tier_arr[i]}"
                for i, r in enumerate(var["region"].values)]
    elif color_by == "trust_score":
        if trust_score is None:
            raise ValueError("color_by='trust_score' requires trust_score=")
        ts = np.asarray(trust_score, dtype=np.float64)
        is_nan = np.isnan(ts)
        ts_norm = np.where(is_nan, 0.0, np.clip(ts / 0.4, 0.0, 1.0))   # cap at 40% top-1
        colors = []
        for i, v in enumerate(ts_norm):
            if is_nan[i]:
                colors.append("#6B7280")
            else:
                if v < 0.5:
                    t = v / 0.5
                    r = int(0xEF + (0xF5 - 0xEF) * t)
                    g = int(0x44 + (0xA3 - 0x44) * t)
                    b = int(0x44 + (0x0B - 0x44) * t)
                else:
                    t = (v - 0.5) / 0.5
                    r = int(0xF5 + (0x10 - 0xF5) * t)
                    g = int(0xA3 + (0xB9 - 0xA3) * t)
                    b = int(0x0B + (0x81 - 0x0B) * t)
                colors.append(f"rgb({r},{g},{b})")
        sizes = np.where(is_anchor, 9, 4)
        text = [f"{r}, score: {ts[i]:.2f}" if not is_nan[i] else f"{r}, unknown"
                for i, r in enumerate(var["region"].values)]
    elif color_by == "network":
        idx_anchor = get_anchor_index(var)
        nets_int = assign_networks(var, idx_anchor)
        nets = [NETWORKS[i] for i in nets_int]
        colors = [NET_COLORS.get(n, "#999") for n in nets]
        sizes = np.where(is_anchor, 8, 4)
        text = [f"{r} ({nets[i]})" for i, r in enumerate(var["region"].values)]
    elif color_by == "hemisphere":
        colors = ["#3B82F6" if h == "L" else "#EF4444" for h in var["hemisphere"]]
        sizes = np.where(is_anchor, 8, 4)
        text = list(var["region"].values)
    elif color_by == "is_anchor":
        colors = ["#EF4444" if a else "#94A3B8" for a in is_anchor]
        sizes = np.where(is_anchor, 12, 3)
        text = list(var["region"].values)
    elif color_by == "highlight":
        if highlight_idx is None:
            raise ValueError("color_by='highlight' requires highlight_idx")
        if highlight_values is None:
            highlight_values = np.ones(len(highlight_idx))
        max_v = max(float(np.max(highlight_values)), 1e-9)
        colors = ["rgba(180,180,180,0.15)"] * n
        sizes = np.full(n, 2.0)
        for k, (idx, v) in enumerate(zip(highlight_idx, highlight_values)):
            t = float(np.sqrt(max(v, 0) / max_v))
            colors[int(idx)] = _viridis(1 - t)
            sizes[int(idx)] = 6 + 18 * t
        text = list(var["region"].values)
    else:
        raise ValueError(f"unknown color_by: {color_by}")

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(size=sizes.tolist() if hasattr(sizes, "tolist") else sizes,
                     color=colors,
                     opacity=0.85 if dark else 0.7,
                     line=dict(width=0)),
        text=text,
        hovertemplate="%{text}<extra></extra>",
    )])
    fig.update_layout(
        title=title or f"{ad.uns.get('species', 'unknown')}, {n} nodes",
        scene=_dark_scene() if dark else dict(aspectmode="data",
                                                xaxis_title="x",
                                                yaxis_title="y",
                                                zaxis_title="z"),
        width=width, height=height,
        margin=dict(l=0, r=0, t=40, b=0),
        **(_dark_layout() if dark else {}),
    )
    return fig


def _dark_scene() -> dict:
    """Plotly 3D scene config with a dark background, markers pop better."""
    return dict(
        aspectmode="data",
        bgcolor="#111",
        xaxis=dict(title="x", backgroundcolor="#1a1a1a", gridcolor="#333", color="#bbb"),
        yaxis=dict(title="y", backgroundcolor="#1a1a1a", gridcolor="#333", color="#bbb"),
        zaxis=dict(title="z", backgroundcolor="#1a1a1a", gridcolor="#333", color="#bbb"),
    )


def _dark_layout() -> dict:
    """Plotly Layout config (figure-level) for a dark theme."""
    return dict(paper_bgcolor="#111", plot_bgcolor="#111", font=dict(color="#ddd"))


# ---------------------------------------------------------------------------
# π partner highlight, given a fitted model + source node index
# ---------------------------------------------------------------------------
def plot_pi_partners(model, source_idx: int, *,
                      source: str = "mouse",
                      top_k: int = 20,
                      width: int = 700, height: int = 600):
    """Render the cross-species partners of one source node.

    Given a fitted model (with ``.pi_``, ``._mouse_ad``, ``._human_ad``):
    look up the top-K partners on the OTHER side and render them with viridis
    colouring on the corresponding 3D brain.

    Parameters
    ----------
    model : FGWModel
        A fitted model instance (must have ``.pi_`` and ``._mouse_ad`` /
        ``._human_ad`` set by ``.fit()``).
    source_idx : int
        Positional index of the source node (into mouse_ad.var or human_ad.var).
    source : "mouse" | "human", default "mouse"
        Which side the source node is on. Partners appear on the other side.
    top_k : int, default 20
    """
    if model.pi_ is None or model._mouse_ad is None or model._human_ad is None:
        raise RuntimeError("Model must be fitted with adata refs (call .fit()).")
    pi = model.pi
    if source == "mouse":
        row = pi[source_idx, :]; other_ad = model._human_ad
    elif source == "human":
        row = pi[:, source_idx];  other_ad = model._mouse_ad
    else:
        raise ValueError(f"source must be 'mouse' or 'human', got {source!r}")
    row_sum = max(float(row.sum()), 1e-12)
    row_norm = row / row_sum
    top_partner_idx = np.argpartition(-row_norm, min(top_k, len(row_norm) - 1))[:top_k]
    top_partner_idx = top_partner_idx[np.argsort(-row_norm[top_partner_idx])]
    top_partner_vals = row_norm[top_partner_idx]

    src_ad = model._mouse_ad if source == "mouse" else model._human_ad
    src_region = src_ad.var.iloc[source_idx]["region"]
    title = (f"{source.capitalize()} #{source_idx} ({src_region}) → "
             f"top-{top_k} {('human' if source == 'mouse' else 'mouse')} partners "
             f"(brightness ∝ π)")

    return plot_brain_3d(other_ad, color_by="highlight",
                          highlight_idx=top_partner_idx.tolist(),
                          highlight_values=top_partner_vals.tolist(),
                          title=title, width=width, height=height)


def plot_pi_partners_pair(model, source_idx: int, *,
                           source: str = "mouse",
                           top_k: int = 20,
                           dark: bool = True,
                           width: int = 1100, height: int = 550):
    """Side-by-side: source node on its atlas + its π partners on the other.

    Renders a 2-panel 3D figure: left scene shows the source node highlighted
    (orange) on its species' brain; right scene shows the top-K cross-species
    partners coloured by π weight on a viridis scale.

    Parameters
    ----------
    model : FGWModel
        A fitted model with ``.pi_`` + adata refs.
    source_idx : int
        Positional index of the source node.
    source : "mouse" | "human", default "mouse"
        Which side the source node lives on.
    top_k : int, default 20
        How many partners to highlight on the other side.
    dark : bool, default True
        Black background, easier to see small markers.
    """
    go = _import_plotly()
    from plotly.subplots import make_subplots

    if model.pi_ is None or model._mouse_ad is None or model._human_ad is None:
        raise RuntimeError("Model must be fitted with adata refs (call .fit()).")
    pi = model.pi
    if source == "mouse":
        row = pi[source_idx, :]
        src_ad   = model._mouse_ad
        other_ad = model._human_ad
        other_label = "human"
    elif source == "human":
        row = pi[:, source_idx]
        src_ad   = model._human_ad
        other_ad = model._mouse_ad
        other_label = "mouse"
    else:
        raise ValueError(f"source must be 'mouse' or 'human', got {source!r}")

    # Top-K partners (normalised by row/col sum)
    row_sum = max(float(row.sum()), 1e-12)
    row_norm = row / row_sum
    top_idx = np.argpartition(-row_norm, min(top_k, len(row_norm) - 1))[:top_k]
    top_idx = top_idx[np.argsort(-row_norm[top_idx])]
    top_vals = row_norm[top_idx]

    # ---- Source side: dim grey + chosen node in orange ----
    n_src = len(src_ad.var)
    src_colors = ["rgba(160,160,160,0.18)"] * n_src
    src_sizes  = [3.0] * n_src
    src_colors[source_idx] = "#FFAA00"
    src_sizes[source_idx]  = 18.0
    src_text = [f"{r} <i>({src_ad.var['hemisphere'].iloc[i]})</i>"
                 for i, r in enumerate(src_ad.var['region'].values)]
    # Make the chosen one stand out in hover too
    src_text[source_idx] = f"<b>SELECTED</b>: {src_ad.var.iloc[source_idx]['region']}"

    # ---- Other side: viridis-coloured partners + dim rest ----
    n_oth = len(other_ad.var)
    oth_colors = ["rgba(160,160,160,0.10)"] * n_oth
    oth_sizes  = [2.0] * n_oth
    max_v = max(float(top_vals[0]), 1e-9)
    oth_text = [f"{r} <i>({other_ad.var['hemisphere'].iloc[i]})</i>"
                 for i, r in enumerate(other_ad.var['region'].values)]
    for k, (idx, v) in enumerate(zip(top_idx, top_vals)):
        if v < 1e-3: continue
        t = float(np.sqrt(max(v, 0) / max_v))
        oth_colors[int(idx)] = _viridis(1 - t)
        oth_sizes[int(idx)]  = 6.0 + 22.0 * t
        oth_text[int(idx)] = (
            f"<b>{v*100:.1f}%</b> · {other_ad.var.iloc[int(idx)]['region']} "
            f"({other_ad.var['hemisphere'].iloc[int(idx)]})"
        )

    # ---- 2-panel figure ----
    src_region = src_ad.var.iloc[source_idx]["region"]
    title = (f"{source.capitalize()} #{source_idx} ({src_region}) → "
             f"top-{top_k} {other_label} partners")

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=[
            f"🐭 {source.capitalize()} (selected = orange)",
            f"🧠 {other_label.capitalize()} (top-{top_k} partners, brightness ∝ π)",
        ] if source == "mouse" else [
            f"🧠 {source.capitalize()} (selected = orange)",
            f"🐭 {other_label.capitalize()} (top-{top_k} partners, brightness ∝ π)",
        ],
        horizontal_spacing=0.05,
    )
    fig.add_trace(go.Scatter3d(
        x=src_ad.var["x"].values, y=src_ad.var["y"].values, z=src_ad.var["z"].values,
        mode="markers",
        marker=dict(size=src_sizes, color=src_colors, opacity=0.9, line=dict(width=0)),
        text=src_text, hovertemplate="%{text}<extra></extra>", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=other_ad.var["x"].values, y=other_ad.var["y"].values, z=other_ad.var["z"].values,
        mode="markers",
        marker=dict(size=oth_sizes, color=oth_colors, opacity=0.9, line=dict(width=0)),
        text=oth_text, hovertemplate="%{text}<extra></extra>", showlegend=False,
    ), row=1, col=2)

    layout_kwargs = dict(
        title=title,
        width=width, height=height,
        margin=dict(l=0, r=0, t=80, b=0),
    )
    if dark:
        layout_kwargs.update(_dark_layout())
        layout_kwargs["scene"]  = _dark_scene()
        layout_kwargs["scene2"] = _dark_scene()
    else:
        layout_kwargs["scene"]  = dict(aspectmode="data")
        layout_kwargs["scene2"] = dict(aspectmode="data")
    fig.update_layout(**layout_kwargs)
    return fig


# ---------------------------------------------------------------------------
# Full π heatmap (only practical for downsampled or anchor-only π)
# ---------------------------------------------------------------------------
def plot_pi_heatmap(pi: np.ndarray, *,
                     max_size: int = 400,
                     title: Optional[str] = None,
                     width: int = 700, height: int = 700):
    """Render π as a heatmap. For large π (>max_size), aggregate to blocks of
    ``ceil(n/max_size)`` rows/cols and show the mean. For small π (e.g. the
    42×42 anchor sub-block), shown as-is.
    """
    go = _import_plotly()
    n_m, n_h = pi.shape
    if max(n_m, n_h) > max_size:
        bm = max(1, n_m // max_size)
        bh = max(1, n_h // max_size)
        # Block-mean downsample
        nm2 = n_m // bm * bm; nh2 = n_h // bh * bh
        z = pi[:nm2, :nh2].reshape(nm2 // bm, bm, nh2 // bh, bh).mean(axis=(1, 3))
    else:
        z = pi
    fig = go.Figure(data=go.Heatmap(
        z=z, colorscale="Viridis",
        colorbar=dict(title="π weight"),
    ))
    fig.update_layout(
        title=title or f"π heatmap ({pi.shape[0]} × {pi.shape[1]})",
        xaxis_title="Human node",
        yaxis_title="Mouse node",
        width=width, height=height,
        margin=dict(l=60, r=20, t=50, b=60),
    )
    fig.update_yaxes(autorange="reversed")
    return fig


# ---------------------------------------------------------------------------
# Per-network heatmap
# ---------------------------------------------------------------------------
def plot_per_network_heatmap(long_df, *,
                              value_col: str = "top1",
                              title: Optional[str] = None,
                              width: int = 1100, height: int = 500):
    """Render the configs × networks heatmap from a long-form DataFrame.

    Expects ``long_df`` with columns ``config | label | network | top1``
    (the format produced by ``homer.viz.reports.build_per_network_long``).
    """
    go = _import_plotly()
    pivot = long_df.pivot_table(index="label", columns="network",
                                  values=value_col, aggfunc="first")
    cols = [n for n in NETWORKS if n in pivot.columns]
    pivot = pivot[cols]
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=cols,
        y=list(pivot.index),
        colorscale="RdYlGn",
        zmin=0, zmax=1,
        colorbar=dict(title=value_col),
        text=[[f"{v:.0%}" if np.isfinite(v) else ", " for v in row]
               for row in pivot.values],
        texttemplate="%{text}",
        textfont=dict(size=10),
    ))
    fig.update_layout(
        title=title or f"Configs × networks: {value_col}",
        width=width, height=height,
        margin=dict(l=200, r=20, t=50, b=80),
    )
    return fig


# ---------------------------------------------------------------------------
# Multi-metric bars
# ---------------------------------------------------------------------------
def plot_comparison_bars(table_df, *,
                          metrics: Sequence[str] = (
                              "anchor_top1", "anchor_top5",
                              "fc_translation_r", "anchor_mean_xyz_dist",
                          ),
                          configs: Optional[Sequence[str]] = None,
                          width: int = 1100, height: int = 700):
    """Multi-panel bar chart: one panel per metric, bars per config.

    Expects ``table_df`` with columns ``config, label, anchor_top1, ...``
    the wide-form table produced by ``homer.viz.reports.build_comparison_table``.
    """
    go = _import_plotly()
    from plotly.subplots import make_subplots

    if configs is not None:
        df = table_df[table_df["config"].isin(configs)].copy()
    else:
        df = table_df.copy()
    df = df.sort_values("config").reset_index(drop=True)

    n = len(metrics)
    rows = (n + 1) // 2
    fig = make_subplots(rows=rows, cols=2, subplot_titles=list(metrics))
    for k, m in enumerate(metrics):
        r, c = k // 2 + 1, k % 2 + 1
        is_pct = "top" in m or "pair" in m or "hemi" in m
        vals = df[m].values
        labels = df["label"].values if "label" in df.columns else df["config"].values
        colors = ["#dd8452" if "production" in str(n_).lower() else "#4c72b0"
                   for n_ in df.get("notes", [""] * len(df))]
        fig.add_trace(go.Bar(
            x=vals, y=labels, orientation="h",
            marker_color=colors,
            text=[f"{v*100:.0f}%" if is_pct else f"{v:.3f}"
                   if np.isfinite(v) else ", " for v in vals],
            textposition="outside",
            showlegend=False,
        ), row=r, col=c)
    fig.update_layout(
        title="Comprehensive comparison, all configs (production = orange)",
        width=width, height=height,
        margin=dict(l=200, r=40, t=80, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_VIRIDIS = ["#fde725", "#b5de2b", "#6ece58", "#35b779", "#1f9e89",
            "#26828e", "#31688e", "#3e4a89", "#482878", "#440154"]


def _viridis(t: float) -> str:
    """t ∈ [0, 1] → 0 = bright yellow, 1 = dark purple."""
    n = len(_VIRIDIS)
    i = max(0, min(n - 1, int(t * (n - 1))))
    return _VIRIDIS[i]


# ---------------------------------------------------------------------------
# Region-level translation (aggregate over all nodes matching a region name)
# ---------------------------------------------------------------------------
def _aggregate_region_pi(model, region_query=None, *, network=None,
                          source: str = "mouse"):
    """Find source-side nodes selected by region name OR network membership,
    and return their aggregated cross-species distribution.

    Selection:
        - ``region_query`` (str, optional): case-insensitive substring matched
          against source region names. Useful for anchor regions (e.g.
          ``"thalamus"``, ``"amygdala"``).
        - ``network`` (str, optional): exact match against the source node's
          functional-network assignment. Use this for non-anchor regions
          e.g. ``network="visual"`` selects all 88 mouse visual-network nodes,
          not just the 4 visual anchors.

    Pass either or both. If both, the AND of the two filters is used.

    Returns:
        src_indices: positional indices in source AnnData
        src_regions: list of matched region names (for display)
        agg_norm:    (n_other,) normalised aggregate distribution
    """
    if model.pi_ is None or model._mouse_ad is None or model._human_ad is None:
        raise RuntimeError("Model must be fitted with adata refs (call .fit()).")
    if source not in ("mouse", "human"):
        raise ValueError(f"source must be 'mouse' or 'human', got {source!r}")
    if region_query is None and network is None:
        raise ValueError("must pass at least one of region_query or network")

    src_ad = model._mouse_ad if source == "mouse" else model._human_ad

    mask = np.ones(len(src_ad.var), dtype=bool)
    if region_query is not None:
        q = region_query.lower()
        mask &= src_ad.var["region"].str.lower().str.contains(q, regex=False).values
    if network is not None:
        idx_anchor = get_anchor_index(src_ad.var)
        net_int = assign_networks(src_ad.var, idx_anchor)
        net_names = np.array([NETWORKS[i] for i in net_int])
        mask &= (net_names == network)

    if not mask.any():
        return [], [], None
    src_indices = np.where(mask)[0].tolist()
    src_regions = src_ad.var.loc[mask, "region"].tolist()
    pi = model.pi
    if source == "mouse":
        agg = pi[src_indices, :].sum(axis=0)
    else:
        agg = pi[:, src_indices].sum(axis=1)
    agg_norm = agg / max(float(agg.sum()), 1e-12)
    return src_indices, src_regions, agg_norm


def region_translation_table(model, region_query=None, *,
                              network=None,
                              source: str = "mouse",
                              top_k: int = 10):
    """Build a tidy DataFrame of the top-K cross-species partners for a region.

    Selection (pass at least one):
        - ``region_query``: case-insensitive substring on region names
          (works for the 21 named anchor regions per side).
        - ``network``: exact match on functional-network name (covers all
          nodes in that network, including non-anchor ones).

    All matching source nodes have their π contributions summed.
    """
    import pandas as pd
    src_indices, src_regions, agg = _aggregate_region_pi(
        model, region_query=region_query, network=network, source=source,
    )
    if agg is None:
        return pd.DataFrame()
    other_ad = model._human_ad if source == "mouse" else model._mouse_ad
    top_idx = np.argpartition(-agg, min(top_k, len(agg) - 1))[:top_k]
    top_idx = top_idx[np.argsort(-agg[top_idx])]

    rows = []
    for rank, j in enumerate(top_idx, 1):
        rows.append({
            "rank":              rank,
            "partner_region":    other_ad.var.iloc[int(j)]["region"],
            "partner_hemi":      other_ad.var["hemisphere"].iloc[int(j)],
            "is_anchor":         bool(other_ad.var["garin_anchor"].iloc[int(j)]),
            "pi_share":          float(agg[int(j)]),
        })
    return pd.DataFrame(rows)


def plot_region_translation(model, region_query=None, *,
                             network=None,
                             source: str = "mouse",
                             top_k: int = 25,
                             dark: bool = True,
                             width: int = 1100, height: int = 550,
                             verbose: bool = True):
    """Side-by-side: all source nodes selected by region/network (orange, left) +
    aggregated cross-species translation distribution (viridis, right).

    Selection (pass at least one):
        - ``region_query`` (str): case-insensitive substring on region names.
          Works mainly for the 21 named anchor regions per side
          (e.g. ``"thalamus"``, ``"amygdala"``, ``"motor"``).
        - ``network`` (str): exact match on functional-network assignment.
          Covers ALL nodes in that network including the ~99% of non-anchor
          grid nodes (e.g. ``network="visual"`` selects all 88 mouse visual
          nodes, not just the 4 visual anchors).

    Pass both for AND filtering. The matched nodes' π rows (or columns when
    ``source='human'``) are summed and the aggregated distribution is rendered
    on the other species.

    Parameters
    ----------
    model : FGWModel
        A fitted model with ``.pi_`` + adata refs.
    region_query, network : str, optional
        Source-side selectors. At least one required.
    source : "mouse" | "human", default "mouse"
    top_k : int, default 25
        How many partners to highlight on the other side.
    verbose : bool, default True
        Print matched source regions to stdout.
    """
    go = _import_plotly()
    from plotly.subplots import make_subplots

    src_indices, src_regions, agg = _aggregate_region_pi(
        model, region_query=region_query, network=network, source=source,
    )
    if agg is None:
        sel = (f"region_query={region_query!r}" if region_query else "") + \
              (", " if region_query and network else "") + \
              (f"network={network!r}" if network else "")
        raise ValueError(f"no nodes matched ({sel}) on the {source} side.")

    src_ad   = model._mouse_ad if source == "mouse" else model._human_ad
    other_ad = model._human_ad if source == "mouse" else model._mouse_ad
    other_label = "human" if source == "mouse" else "mouse"

    if verbose:
        sel_str = (f"region_query={region_query!r}" if region_query else "") + \
                  (", " if region_query and network else "") + \
                  (f"network={network!r}" if network else "")
        n_unique = len(set(src_regions))
        print(f"matched {len(src_indices)} {source} node(s) ({n_unique} unique region "
              f"name{'s' if n_unique != 1 else ''}) for {sel_str}")
        # Truncate the listing if huge (e.g. 88 visual nodes)
        if len(src_regions) <= 8:
            for r in src_regions:
                print(f"   · {r}")
        else:
            for r in src_regions[:5]:
                print(f"   · {r}")
            print(f"   · ... and {len(src_regions) - 5} more")

    # ---- Source side: dim grey + matching nodes in orange ----
    n_src = len(src_ad.var)
    src_colors = ["rgba(160,160,160,0.18)"] * n_src
    src_sizes  = [3.0] * n_src
    src_text   = list(src_ad.var["region"].values)
    for i in src_indices:
        src_colors[i] = "#FFAA00"
        src_sizes[i]  = 12.0
        src_text[i]   = f"<b>SELECTED</b>: {src_ad.var.iloc[i]['region']}"

    # ---- Other side: top-K viridis ----
    n_oth = len(other_ad.var)
    oth_colors = ["rgba(160,160,160,0.10)"] * n_oth
    oth_sizes  = [2.0] * n_oth
    oth_text   = list(other_ad.var["region"].values)
    top_idx = np.argpartition(-agg, min(top_k, len(agg) - 1))[:top_k]
    top_idx = top_idx[np.argsort(-agg[top_idx])]
    max_v = max(float(agg[top_idx[0]]), 1e-9)
    for j in top_idx:
        v = float(agg[int(j)])
        if v < 1e-4: continue
        t = float(np.sqrt(v / max_v))
        oth_colors[int(j)] = _viridis(1 - t)
        oth_sizes[int(j)]  = 6.0 + 22.0 * t
        oth_text[int(j)]   = (
            f"<b>{v*100:.1f}%</b> · {other_ad.var.iloc[int(j)]['region']} "
            f"({other_ad.var['hemisphere'].iloc[int(j)]})"
        )

    sel_label = []
    if region_query: sel_label.append(f"'{region_query}'")
    if network:      sel_label.append(f"network={network!r}")
    title = (f"{source.capitalize()} {' + '.join(sel_label)} "
             f"({len(src_indices)} node{'s' if len(src_indices) != 1 else ''}) "
             f"→ aggregated top-{top_k} {other_label} partners")

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=[
            f"🐭 {source.capitalize()} (matched nodes orange)" if source == "mouse"
              else f"🧠 {source.capitalize()} (matched nodes orange)",
            f"🧠 {other_label.capitalize()} (aggregate π distribution)" if other_label == "human"
              else f"🐭 {other_label.capitalize()} (aggregate π distribution)",
        ],
        horizontal_spacing=0.05,
    )
    fig.add_trace(go.Scatter3d(
        x=src_ad.var["x"].values, y=src_ad.var["y"].values, z=src_ad.var["z"].values,
        mode="markers",
        marker=dict(size=src_sizes, color=src_colors, opacity=0.9, line=dict(width=0)),
        text=src_text, hovertemplate="%{text}<extra></extra>", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=other_ad.var["x"].values, y=other_ad.var["y"].values, z=other_ad.var["z"].values,
        mode="markers",
        marker=dict(size=oth_sizes, color=oth_colors, opacity=0.9, line=dict(width=0)),
        text=oth_text, hovertemplate="%{text}<extra></extra>", showlegend=False,
    ), row=1, col=2)

    layout_kwargs = dict(
        title=title,
        width=width, height=height,
        margin=dict(l=0, r=0, t=80, b=0),
    )
    if dark:
        layout_kwargs.update(_dark_layout())
        layout_kwargs["scene"]  = _dark_scene()
        layout_kwargs["scene2"] = _dark_scene()
    else:
        layout_kwargs["scene"]  = dict(aspectmode="data")
        layout_kwargs["scene2"] = dict(aspectmode="data")
    fig.update_layout(**layout_kwargs)
    return fig


# ---------------------------------------------------------------------------
# Ordered π heatmap with network sidebar
# ---------------------------------------------------------------------------
def plot_pi_heatmap_ordered(
    pi: np.ndarray,
    mouse_ad,
    human_ad,
    *,
    sort_by: str = "network",
    downsample_to: Optional[int] = 600,
    show_network_bars: bool = True,
    show_boundaries: bool = True,
    log_scale: bool = False,
    dark: bool = True,
    width: int = 950,
    height: int = 850,
    title: Optional[str] = None,
):
    """Render the cross-species π map with rows + columns reordered by network.

    Adds coloured network sidebars (left + top) so the block structure is
    visible at a glance. The diagonal-ish band you'll see if anchor supervision
    works comes from each network's mouse anchors mapping to its human anchors.

    Parameters
    ----------
    pi : (n_m, n_h) ndarray
        The cross-species coupling.
    mouse_ad, human_ad : AnnData
    sort_by : "network" (default) | "hemisphere"
        How to order rows + columns. Within a network, ties broken by
        original index (so anchors come first, grid nodes after).
    downsample_to : int or None, default 600
        If max(n_m, n_h) > this, block-sum-downsample for rendering speed.
        Block-sum preserves total π mass per network block. Pass None for
        full-resolution (slow in browser for >800-node atlases).
    show_network_bars : bool, default True
        Add coloured network sidebar to the top + left of the main heatmap.
    show_boundaries : bool, default True
        White lines at network boundaries on the main heatmap.
    log_scale : bool, default False
        Plot log1p(π * scale) instead of raw π, useful when the production
        solve is mostly one-hot and the off-diagonal structure would otherwise
        be invisible.
    dark : bool, default True
        Black background, light text.
    width, height : int
    title : str or None
    """
    go = _import_plotly()
    from plotly.subplots import make_subplots

    if pi.ndim != 2:
        raise ValueError(f"pi must be 2-D, got shape {pi.shape}")
    n_m, n_h = pi.shape

    # ---- 1. Compute network assignment + sort order ----
    idx_m = get_anchor_index(mouse_ad.var)
    idx_h = get_anchor_index(human_ad.var)
    net_m = assign_networks(mouse_ad.var, idx_m)        # int per node
    net_h = assign_networks(human_ad.var, idx_h)

    if sort_by == "network":
        # Stable sort by network → preserves original-index order within each net
        m_order = np.argsort(net_m, kind="stable")
        h_order = np.argsort(net_h, kind="stable")
    elif sort_by == "hemisphere":
        # Sort by (hemisphere, network), useful for spotting L/R asymmetries
        hemi_m = (mouse_ad.var["hemisphere"].values == "R").astype(int)
        hemi_h = (human_ad.var["hemisphere"].values == "R").astype(int)
        m_order = np.lexsort((net_m, hemi_m))
        h_order = np.lexsort((net_h, hemi_h))
    else:
        raise ValueError(f"unknown sort_by: {sort_by!r}")

    pi_sorted    = pi[np.ix_(m_order, h_order)]
    net_m_sorted = net_m[m_order]
    net_h_sorted = net_h[h_order]

    # ---- 2. Optional block-sum downsample for rendering ----
    bm = bh = 1
    if downsample_to is not None and max(n_m, n_h) > downsample_to:
        bm = max(1, n_m // downsample_to)
        bh = max(1, n_h // downsample_to)
        nm2 = (n_m // bm) * bm
        nh2 = (n_h // bh) * bh
        z = pi_sorted[:nm2, :nh2].reshape(nm2 // bm, bm, nh2 // bh, bh).sum(axis=(1, 3))
        # Downsample the network labels by majority vote per block
        net_m_ds = net_m_sorted[:nm2].reshape(nm2 // bm, bm)
        net_h_ds = net_h_sorted[:nh2].reshape(nh2 // bh, bh)
        net_m_render = np.array([np.bincount(r).argmax() for r in net_m_ds])
        net_h_render = np.array([np.bincount(r).argmax() for r in net_h_ds])
    else:
        z = pi_sorted
        net_m_render = net_m_sorted
        net_h_render = net_h_sorted

    if log_scale:
        z = np.log1p(z * n_m * n_h)        # scale up so log1p has dynamic range

    # ---- 3. Network boundaries (in render coordinates) ----
    m_bounds = np.where(np.diff(net_m_render) != 0)[0] + 1
    h_bounds = np.where(np.diff(net_h_render) != 0)[0] + 1

    # ---- 4. Build the network discrete colourscale ----
    n_networks = len(NETWORKS)
    network_colors = [NET_COLORS.get(n, "#999") for n in NETWORKS]
    # Discrete colourscale: each network gets one stop
    network_cscale = []
    for i, c in enumerate(network_colors):
        lo = i / n_networks; hi = (i + 1) / n_networks
        network_cscale.extend([[lo, c], [hi, c]])

    # ---- 5. Layout: 2x2 with thin sidebars ----
    if show_network_bars:
        fig = make_subplots(
            rows=2, cols=2,
            row_heights=[0.04, 0.96], column_widths=[0.04, 0.96],
            shared_xaxes=False, shared_yaxes=False,
            horizontal_spacing=0.005, vertical_spacing=0.005,
        )
        # TOP, human network bar (1 row, n_h cols)
        fig.add_trace(go.Heatmap(
            z=[net_h_render.tolist()],
            colorscale=network_cscale,
            zmin=-0.5, zmax=n_networks - 0.5,
            showscale=False,
            hovertemplate="Human · %{customdata}<extra></extra>",
            customdata=[[NETWORKS[i] for i in net_h_render]],
        ), row=1, col=2)
        # LEFT, mouse network bar (n_m rows, 1 col)
        fig.add_trace(go.Heatmap(
            z=[[v] for v in net_m_render],
            colorscale=network_cscale,
            zmin=-0.5, zmax=n_networks - 0.5,
            showscale=False,
            hovertemplate="Mouse · %{customdata}<extra></extra>",
            customdata=[[NETWORKS[i]] for i in net_m_render],
        ), row=2, col=1)
        # MAIN, π heatmap
        main_row, main_col = 2, 2
        fig.add_trace(go.Heatmap(
            z=z, colorscale="Viridis",
            colorbar=dict(title=("log(π)" if log_scale else "π"),
                           thickness=12, x=1.02),
        ), row=main_row, col=main_col)
        # Hide axes on the bars
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=2)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=2, col=1)
        # Reverse y on main so row 0 is at the top
        fig.update_yaxes(autorange="reversed", row=main_row, col=main_col)
    else:
        fig = go.Figure(data=go.Heatmap(
            z=z, colorscale="Viridis",
            colorbar=dict(title=("log(π)" if log_scale else "π")),
        ))
        fig.update_yaxes(autorange="reversed")
        main_row, main_col = None, None

    # ---- 6. Boundary lines on the main heatmap ----
    if show_boundaries:
        line_color = "#fff" if dark else "#222"
        for x in h_bounds:
            fig.add_vline(x=x - 0.5, line=dict(color=line_color, width=0.5),
                           row=main_row, col=main_col, opacity=0.6)
        for y in m_bounds:
            fig.add_hline(y=y - 0.5, line=dict(color=line_color, width=0.5),
                           row=main_row, col=main_col, opacity=0.6)

    # ---- 7. Layout polish ----
    if title is None:
        title = (f"π, sorted by {sort_by} "
                  f"(rows: {n_m} mouse, cols: {n_h} human"
                  + (f"; downsampled {bm}×{bh}" if (bm > 1 or bh > 1) else "")
                  + (")" if not log_scale else ", log scale)"))

    layout_kwargs = dict(
        title=title,
        width=width, height=height,
        margin=dict(l=20, r=80, t=70, b=20),
    )
    if dark:
        layout_kwargs.update(_dark_layout())
    fig.update_layout(**layout_kwargs)
    return fig
